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
    assert "[图片 photo.jpg]" in by_id["ROBOT1.0_image"]["content"], "图片消息的附件标签落库"
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
