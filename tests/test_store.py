"""SQLite 存储：原文去重落库、总结入库与覆盖范围、索引进度、自动总结计数。"""

from __future__ import annotations

import json

from conftest import (
    IMAGE_MESSAGE,
    QUOTE_MESSAGE,
    TEXT_MESSAGE,
    VOICE_MESSAGE,
    frame,
)
from src.bot.events import GroupMessageRecord


def test_messages_insert_dedupe_and_read(store) -> None:
    recs = [
        GroupMessageRecord.from_payload(frame(TEXT_MESSAGE, "E1")),
        GroupMessageRecord.from_payload(frame(IMAGE_MESSAGE, "E2")),
        GroupMessageRecord.from_payload(frame(QUOTE_MESSAGE, "E3")),
        GroupMessageRecord.from_payload(frame(VOICE_MESSAGE, "E4")),
    ]
    assert store.insert_messages(recs) == 4, "首次写入 4 条"

    # 同一条消息被 QQ 重复推送时不应重复入库
    assert store.insert_messages(recs) == 0, "重复写入被去重"

    recent = store.recent_messages("G_demo", limit=10)
    assert len(recent) == 4 and recent[0]["author_name"] == "小明", "取回 4 条且按时间正序"

    # 落库的是「文本化」的正文，不是原始 content：引用消息的原始 content 是
    # 空白、图片消息的文字全在附件上，直接存原始值会让这些消息在 prompt 里
    # 变成一行空白。原文（原始 d）仍完整留在 raw_json。
    by_id = {row["message_id"]: row for row in recent}
    # M2 之后图片占位带 `#短id`（view_image 的引用方式），断言跟着形态走。
    assert "[图片 photo.jpg #" in by_id["ROBOT1.0_image"]["content"], "图片消息的附件占位落库"
    assert "明天有空吗" in by_id["ROBOT1.0_quote"]["content"], "引用消息的引用正文落库"
    assert "[语音转写" in by_id["ROBOT1.0_voice"]["content"], "语音消息的 ASR 转写落库"
    assert "message_scene" in by_id["ROBOT1.0_quote"]["raw_json"], "原文仍保留在 raw_json"

    rng = store.messages_in_range(
        "G_demo", "2026-07-21T09:00:00+08:00", "2026-07-21T11:00:00+08:00", limit=10
    )
    assert len(rng) == 2, "时间范围过滤"

    assert json.loads(recent[0]["raw_json"])["group_openid"] == "G_demo", "raw_json 可反序列化"
    assert store.recent_messages("G_other", limit=10) == [], "群过滤生效"


def test_summaries_insert_coverage_and_marks(store) -> None:
    # 统计与积压要同时看原文表和总结表，所以先垫 4 条消息（与合并版一致）。
    for event, payload in (
        ("E1", TEXT_MESSAGE), ("E2", IMAGE_MESSAGE), ("E3", QUOTE_MESSAGE), ("E4", VOICE_MESSAGE)
    ):
        store.insert_messages([GroupMessageRecord.from_payload(frame(payload, event))])

    sid1 = store.insert_summary(
        group_openid="G_demo",
        instruction="总结一下",
        content="会议推迟到下午三点，小红负责准备材料。",
        coverage=[("2026-07-21T08:00:00+08:00", "2026-07-21T10:00:00+08:00")],
        message_count=12,
        requested_by="M_ming",
    )
    assert isinstance(sid1, str) and len(sid1) == 32, "总结入库返回 id"

    # Same instruction, same wording, different coverage window. A
    # content-hash primary key would silently drop this second row and lose
    # its window; a plain INSERT with a uuid keeps both.
    sid2 = store.insert_summary(
        group_openid="G_demo",
        instruction="总结一下",
        content="会议推迟到下午三点，小红负责准备材料。",
        coverage=[("2026-07-21T11:00:00+08:00", "2026-07-21T12:00:00+08:00")],
        message_count=3,
    )
    assert sid2 != sid1, "同文不同覆盖范围的总结不被吞掉"
    assert len(store.summaries_for("G_demo")) == 2, "库中确实有两篇"

    assert len(store.unindexed_summaries(limit=10)) == 2, "未索引积压为 2"
    store.mark_summaries_indexed([sid1])
    assert len(store.unindexed_summaries(limit=10)) == 1, "标记后积压剩 1"
    assert (
        store.clear_summary_marks() == 2
        and len(store.unindexed_summaries(limit=10)) == 2
    ), "清空标记后积压回满"

    one = store.summaries_for("G_demo", limit=10)[-1]  # oldest first
    assert one["message_count"] == 12, "覆盖条数被记录"
    assert str(one["ts_start"]).startswith("2026-07-21T08:00"), "覆盖范围包络起点"
    assert str(one["ts_end"]).startswith("2026-07-21T10:00"), "覆盖范围包络终点"
    assert json.loads(one["coverage_json"])[0][0].startswith("2026-07-21T08:00"), (
        "coverage_json 保留原始区间"
    )
    assert one["indexed_at"] is None, "新总结默认待索引"

    stats = store.stats()
    assert (
        stats[0]["total"] == 4
        and stats[0]["summaries"] == 2
        and stats[0]["indexed"] == 0
    ), "统计 total=4 summaries=2 indexed=0（标记刚被清空）"
    assert len(store.summaries_for(None)) == 2, "不带群过滤能看到所有总结"
    assert store.summaries_for("G_other") == [], "带群过滤只看到本群"
    groups = store.groups_with_summaries()
    assert len(groups) == 1 and groups[0]["group_openid"] == "G_demo", (
        "groups_with_summaries 只列有总结的群"
    )

    # ---- who wrote the document ------------------------------------------
    assert one["trigger"] == "at", "被 @ 触发的总结记为 at"


def test_summaries_in_range_overlaps_the_covered_window(store) -> None:
    """`summaries_in_range` answers "has this stretch been written up".

    The axis is the document's **covered window** (`ts_start`/`ts_end`), never
    `created_at`, and every row here is created *now* while covering July — which
    is exactly the shape that makes the two axes differ. A "newest N by
    created_at" listing would show all three for an October question.
    """
    # 一篇「今天写的、覆盖上周三」的稿子：这就是 @ 投稿带来的新形状。
    store.insert_summary(
        group_openid="G_demo",
        instruction="总结上周三那场争论",
        content="上周三的争论。",
        coverage=[("2026-07-15T10:00:00+08:00", "2026-07-15T11:00:00+08:00")],
        message_count=9,
    )
    store.insert_summary(
        group_openid="G_demo",
        instruction="总结今天的排期",
        content="今天的排期。",
        coverage=[("2026-07-21T08:00:00+08:00", "2026-07-21T10:00:00+08:00")],
        message_count=12,
    )
    store.insert_summary(
        group_openid="G_other",
        instruction="别群的总结",
        content="别群内容。",
        coverage=[("2026-07-21T08:30:00+08:00", "2026-07-21T09:00:00+08:00")],
        message_count=4,
    )
    # 无覆盖时段（读库做的综合稿会这样）：不可能与任何区间重叠，接受其不可见。
    store.insert_summary(
        group_openid="G_demo", instruction="综合稿", content="没有时段。",
        coverage=[], message_count=0,
    )

    def q(gid, start, end, limit=20):
        return store.summaries_in_range(gid, start, end, limit)

    rows, total = q("G_demo", "2026-07-21T09:00:00+08:00", "2026-07-21T09:30:00+08:00")
    assert total == 1 and len(rows) == 1, "落在窗口内部才算命中"
    assert rows[0]["instruction"] == "总结今天的排期", "命中的是覆盖该时段的那篇"

    rows, total = q("G_demo", "2026-10-10T00:00:00+08:00", "2026-10-10T23:00:00+08:00")
    assert total == 0 and rows == [], "全部稿子都是刚才创建的，但没有一篇覆盖十月"

    rows, total = q("G_demo", "2026-07-15T10:30:00+08:00", "2026-07-15T10:40:00+08:00")
    assert total == 1 and rows[0]["instruction"] == "总结上周三那场争论", (
        "按覆盖时段查得到「今天生成、覆盖上周」的旧稿（created_at 序会把它排到最前，"
        "查最近 N 篇的方式恰好会漏掉这种）"
    )

    rows, total = q("G_demo", "2026-07-21T10:00:00+08:00", "2026-07-21T12:00:00+08:00")
    assert total == 1, "端点相接算重叠（<= / >= 含两端）"
    assert rows[0]["instruction"] == "总结今天的排期"

    _, cross = q("G_other", "2026-07-21T08:30:00+08:00", "2026-07-21T08:40:00+08:00")
    assert cross == 1, "换个群只看到那个群的稿子"
    demo_rows, demo_total = q("G_demo", "2026-07-21T08:30:00+08:00", "2026-07-21T08:40:00+08:00")
    assert demo_total == 1 and all(r["group_openid"] == "G_demo" for r in demo_rows), (
        "群过滤不可省：别群同小时的稿子不能出现在本群清单里"
    )

    # `total` is the *pre-truncation* count — the caller renders it into a footer,
    # and a short list that reads as a complete list is how duplicates get in.
    for n in range(3):
        store.insert_summary(
            group_openid="G_demo",
            instruction=f"第 {n} 次",
            content=f"内容 {n}",
            # Three documents whose windows all sit inside the queried range.
            coverage=[(f"2026-07-22T0{n}:00:00+08:00", f"2026-07-22T0{n}:30:00+08:00")],
            message_count=n,
        )
    page, total = q("G_demo", "2026-07-22T00:00:00+08:00", "2026-07-22T03:00:00+08:00", limit=2)
    assert total == 3 and len(page) == 2, "返回页数少于命中数时，total 仍是真实总数"
    assert str(page[0]["ts_end"]).startswith("2026-07-22T02"), "截断保留覆盖时段最近的"
    assert {str(r["instruction"]) for r in page} == {"第 2 次", "第 1 次"}, "最近的 N 篇，不是前 N 篇"

    # Offsets must be normalised, not compared as text (§4.2).
    z_rows, z_total = q("G_demo", "2026-07-21T01:00:00Z", "2026-07-21T01:30:00Z")
    assert z_total == 1, "带 Z 的边界能正确比对（+08:00 与 Z 混用）"
    assert z_rows[0]["instruction"] == "总结今天的排期", "09:00+08:00 == 01:00Z"


def test_find_summary_prefix_and_scope(store) -> None:
    sid = store.insert_summary(
        group_openid="G_demo",
        instruction="总结一下",
        content="正文",
        coverage=[("2026-07-21T08:00:00+08:00", "2026-07-21T10:00:00+08:00")],
        message_count=1,
    )
    assert store.find_summary(sid)["content"] == "正文", "整 id 能查到"
    assert store.find_summary(sid[:8])["summary_id"] == sid, "8 位短 id 前缀命中"
    assert store.find_summary(f"#{sid[:6]}")["summary_id"] == sid, "去 # 前缀、6 位起判"
    assert store.find_summary(sid[:4]) is None, "太短的前缀直接拒（撞库护栏）"
    assert store.find_summary("zzzzzz") is None, "非十六进制返回 None 而不是抛"
    assert store.find_summary("") is None, "空串返回 None"
    # The scope check is the *caller's* job (`view_image` does the same), so the
    # store stays group-agnostic and returns the row for it to refuse.
    assert store.find_summary(sid)["group_openid"] == "G_demo", "行里带着归属可供校验"


def test_schema_and_migrations_agree(tmp_path) -> None:
    """`schema.sql` and `_MIGRATIONS` must describe the same table.

    The comment above `_MIGRATIONS` states the rule (`CREATE TABLE IF NOT EXISTS`
    adds a *table* to an existing database but never a *column*), and breaking it
    is invisible in tests: a fresh DB gets the column from the DDL and works,
    while the real `data/qqbot.db` — created before that column existed — then
    fails on the first write. So assert the two lists agree instead of trusting
    either one.
    """
    import re
    from pathlib import Path

    from src.store.sql_store import _MIGRATIONS

    ddl = Path("src/store/sql/schema.sql").read_text(encoding="utf-8")
    for table, column, _ddl in _MIGRATIONS:
        block = re.search(
            rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);", ddl, re.S
        )
        assert block, f"schema.sql 里没有 {table} 的建表语句"
        names = {
            line.strip().split()[0]
            for line in block.group(1).splitlines()
            if line.strip() and not line.strip().startswith(("--", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN"))
        }
        assert column in names, (
            f"{table}.{column} 在 _MIGRATIONS 里，但 schema.sql 的 DDL 没有它——"
            "新库会缺这一列（老库靠 ALTER 才有）"
        )

    # The事故 this guards is asymmetric, so open a database that predates a
    # post-release column and check `_migrate()` repairs it without touching rows.
    import sqlite3

    from src.store.sql_store import SQLStore

    old_db = tmp_path / "old.db"
    conn = sqlite3.connect(old_db)
    conn.execute(
        """
        CREATE TABLE summaries (
            summary_id TEXT PRIMARY KEY, group_openid TEXT NOT NULL,
            instruction TEXT NOT NULL, content TEXT NOT NULL,
            coverage_json TEXT, ts_start TEXT, ts_end TEXT,
            message_count INTEGER NOT NULL DEFAULT 0, requested_by TEXT,
            created_at TEXT NOT NULL, indexed_at TEXT
        )
        """
    )  # 没有 `trigger`：发布第一版时的形状
    conn.execute(
        "INSERT INTO summaries VALUES ('a'*32, 'G_demo', '总结一下', '旧稿', NULL, "
        "NULL, NULL, 0, NULL, '2026-07-21 10:00:00', NULL)"
    )
    conn.commit()
    conn.close()

    migrated = SQLStore(old_db)
    try:
        cols = {r[1] for r in migrated._conn.execute("PRAGMA table_info(summaries)")}
        assert "trigger" in cols, "缺的列被 ALTER 补回来了"
        assert [str(r["content"]) for r in migrated.summaries_for("G_demo")] == ["旧稿"], (
            "迁移不改写任何既有行"
        )
        # 再开一次：`_migrate()` 靠 PRAGMA 判重，不该第二次 ALTER。
        SQLStore(old_db).close()
    finally:
        migrated.close()


def test_messages_since_last_summary(store) -> None:
    """自动总结的计数：定义在「距上次总结之后」，按群隔离。"""
    for i in range(3):
        store.insert_messages(
            [
                GroupMessageRecord.from_payload(
                    frame({**TEXT_MESSAGE, "id": f"CNT{i}", "group_openid": "G_cnt"})
                )
            ]
        )
    assert store.messages_since_last_summary("G_cnt") == 3, "没有总结时从 epoch 起算，全部计入"

    store.insert_summary(
        group_openid="G_cnt",
        instruction="自动总结",
        content="自动生成的总结正文，没有人 @ 机器人。",
        coverage=[("2026-07-21T08:00:00+08:00", "2026-07-21T09:00:00+08:00")],
        message_count=3,
        requested_by=None,
        trigger="auto",
    )
    auto = store.summaries_for("G_cnt")[0]
    assert auto["trigger"] == "auto", "自动总结单独标记 trigger"
    assert auto["requested_by"] is None, "自动总结没有触发者"
    assert store.messages_since_last_summary("G_cnt") == 0, "总结之后计数归零"
    assert store.messages_since_last_summary("G_demo") == 0, "计数按群隔离"

    # The count is defined against the *summary*, so the bound has to be
    # moved to prove the comparison works at all. Backdating beats sleeping:
    # both timestamps have second resolution, and rows written inside the
    # same second compare equal — which is exactly the flake a sleep invites.
    with store._conn:
        store._conn.execute(
            "UPDATE summaries SET created_at = ? WHERE group_openid = ?",
            ("2026-07-21 07:00:00", "G_cnt"),
        )
    assert store.messages_since_last_summary("G_cnt") == 3, "总结被回拨后，之后的消息重新计入"
    store.insert_messages(
        [
            GroupMessageRecord.from_payload(
                frame({**TEXT_MESSAGE, "id": "CNT9", "group_openid": "G_cnt"})
            )
        ]
    )
    assert store.messages_since_last_summary("G_cnt") == 4, "计数随新消息增长"


def test_groups_with_messages(store) -> None:
    """A group can be full of raw messages and have no summary at all; the
    private-chat tool still has to be able to name it."""
    for gid in ("G_demo", "G_cnt"):
        store.insert_messages(
            [
                GroupMessageRecord.from_payload(
                    frame({**TEXT_MESSAGE, "id": f"ANY_{gid}", "group_openid": gid})
                )
            ]
        )
    assert set(store.groups_with_messages()) == {"G_demo", "G_cnt"}, (
        "groups_with_messages 列出所有有消息的群"
    )
