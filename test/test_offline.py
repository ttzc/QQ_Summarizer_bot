"""Offline checks for the pieces that do not need QQ credentials.

Run from the project root:

    uv run python test/test_offline.py

Covers the most error-prone areas: `GROUP_MESSAGE_CREATE` parsing (fixtures
mirror the official event page), the SQLite store, the summary-level vector
index, the two privacy boundaries (in-group vs. cross-group), and the
message-count auto-summary trigger.

The model, the embedding endpoint and the QQ API are all faked; only Chroma and
SQLite are real.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import botpy  # noqa: E402
from botpy.errors import SequenceNumberError  # noqa: E402

from src.agent.summarizer import SummaryResult  # noqa: E402
from src.agent.tools import CoverageLog  # noqa: E402
from src.bot.events import MAX_ELEMENT_DEPTH, GroupMessageRecord  # noqa: E402
from src.bot.sender import KIND_C2C, plan_chunks, reply_chunked  # noqa: E402
from src.config import config  # noqa: E402
from src.store.sql_store import SQLStore  # noqa: E402

PASSED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {detail}")
        raise AssertionError(label)


def frame(body: dict, event_id: str = "EVENT_0") -> dict:
    """Wrap an event body the way the gateway does."""
    return {"id": event_id, "op": 0, "s": 1, "t": "GROUP_MESSAGE_CREATE", "d": body}


# --- fixtures, mirroring the three samples on the official event page --------

TEXT_MESSAGE = {
    "id": "ROBOT1.0_text",
    "author": {
        "id": "U1",
        "username": "小明",
        "bot": False,
        "member_openid": "M_ming",
        "member_role": "member",
    },
    "content": "大家早上好呀",
    "group_openid": "G_demo",
    "message_type": 0,
    "timestamp": "2026-07-21T08:00:00+08:00",
    "message_scene": {
        "source": "default",
        "ext": ["msg_idx=REFIDX_abc==", "auth_token=tok123"],
    },
}

# The bot's own @ is stripped from `content` before delivery, so this list is the
# only signal that the bot was addressed in a GROUP_MESSAGE_CREATE.
SUMMON_MESSAGE = {
    **TEXT_MESSAGE,
    "id": "ROBOT1.0_summon",
    "content": "总结一下今天群里聊了什么",
    "mentions": [
        {"id": "U9", "username": "总结机器人", "bot": True, "member_openid": "M_bot"},
        {"id": "U1", "username": "小明", "bot": False, "member_openid": "M_ming"},
    ],
}

IMAGE_MESSAGE = {
    "id": "ROBOT1.0_image",
    "author": {
        "id": "U2",
        "username": "小红",
        "bot": False,
        "member_openid": "M_hong",
        "member_role": "owner",
    },
    "content": "分享一张今天的风景照",
    "group_openid": "G_demo",
    "message_type": 0,
    "timestamp": "2026-07-21T09:30:00+08:00",
    "message_scene": {"source": "default", "ext": ["msg_idx=REFIDX_img=="]},
    "attachments": [
        {
            "content_type": "image/jpeg",
            "filename": "photo.jpg",
            "url": "https://multimedia.nt.qq.com.cn/download/xxx",
            "width": 1920,
            "height": 1080,
            "size": 256000,
        }
    ],
}

QUOTE_MESSAGE = {
    "id": "ROBOT1.0_quote",
    "author": {
        "id": "U3",
        "username": "小华",
        "bot": False,
        "member_openid": "M_hua",
        "member_role": "admin",
    },
    "content": " ",
    "group_openid": "G_demo",
    "message_type": 103,
    "timestamp": "2026-07-21T10:10:00+08:00",
    "message_scene": {
        "source": "default",
        "ext": ["msg_idx=REFIDX_q==", "ref_msg_idx=TMP_prev", "auth_token=tok456"],
    },
    "msg_elements": [
        {
            "msg_idx": "TMP_prev",
            "message_type": 0,
            "author": {"username": "小刚"},
            "content": "=== 消息 1 ===\n[消息内容] 明天有空吗",
        }
    ],
}


def test_text_message() -> None:
    print("\n[1] 普通文本消息")
    rec = GroupMessageRecord.from_payload(frame(TEXT_MESSAGE, "EVENT_A"))

    check("message_id 取自 d.id", rec.message_id == "ROBOT1.0_text", rec.message_id)
    check("event_id 取自帧顶层 id", rec.event_id == "EVENT_A", str(rec.event_id))
    check("群标识", rec.group_openid == "G_demo")
    # botpy 的 GroupMessage._User 读不到 username —— 这是自研事件对象的意义所在
    check("昵称没被丢掉", rec.author_name == "小明", str(rec.author_name))
    check("群成员角色", rec.member_role == "member", str(rec.member_role))
    check("msg_idx 从 ext 的 key=value 解析出", rec.msg_idx == "REFIDX_abc==", str(rec.msg_idx))
    check("未引用时 ref_msg_idx 为 None", rec.ref_msg_idx is None, str(rec.ref_msg_idx))
    check("时间戳解析带时区", rec.ts.utcoffset() is not None and rec.ts.hour == 8, str(rec.ts))
    check("原始 d 保留", rec.raw.get("group_openid") == "G_demo")
    check("无 mentions 时不算召唤", not rec.mentions_bot())
    check(
        "to_line 可读",
        rec.to_line() == "08:00 小明: 大家早上好呀",
        rec.to_line(),
    )


def test_mentions() -> None:
    print("\n[1b] @ 触发识别（content 里的 @ 已被平台剥掉）")
    rec = GroupMessageRecord.from_payload(frame(SUMMON_MESSAGE))

    check("mentions 被解析", len(rec.mentions) == 2, str(len(rec.mentions)))
    check("mention 的 id 存进 openid", rec.mentions[0].openid == "U9")
    check("mention 昵称", rec.mentions[0].username == "总结机器人")
    check("识别出 @ 了机器人", rec.mentions_bot())
    check("正文即用户指令", rec.body() == "总结一下今天群里聊了什么", rec.body())


def test_image_message() -> None:
    print("\n[2] 图片附件消息")
    rec = GroupMessageRecord.from_payload(frame(IMAGE_MESSAGE))

    check("附件被解析", len(rec.attachments) == 1, str(len(rec.attachments)))
    att = rec.attachments[0]
    check("附件尺寸", att.width == 1920 and att.height == 1080)
    check("附件标签为图片", att.label() == "[图片 photo.jpg]", att.label())
    check("正文含附件标签", "[图片 photo.jpg]" in rec.body(), rec.body())
    check("群主角色", rec.member_role == "owner", str(rec.member_role))


def test_quote_message() -> None:
    print("\n[3] 引用/嵌套消息（content 为空白）")
    rec = GroupMessageRecord.from_payload(frame(QUOTE_MESSAGE))

    check("message_type=103", rec.message_type == 103, str(rec.message_type))
    check("ref_msg_idx 解析成功", rec.ref_msg_idx == "TMP_prev", str(rec.ref_msg_idx))
    check("嵌套元素被解析", len(rec.elements) == 1, str(len(rec.elements)))
    check("元素作者名", rec.elements[0].author_name == "小刚", str(rec.elements[0].author_name))
    # 关键：content 是空白，若不拼 msg_elements，这条消息在摘要里会是空的
    body = rec.body()
    check("正文从 msg_elements 拼出而非为空", "明天有空吗" in body, repr(body))
    check("嵌套元素时间戳独立", rec.ts.hour == 10)


def test_depth_guard() -> None:
    print("\n[4] msg_elements 递归深度保护")
    node: dict = {"message_type": 0, "content": "leaf"}
    for _ in range(MAX_ELEMENT_DEPTH + 10):
        node = {"message_type": 102, "content": "wrap", "msg_elements": [node]}
    rec = GroupMessageRecord.from_payload(frame({**TEXT_MESSAGE, "msg_elements": [node]}))

    depth, cursor = 0, rec.elements[0]
    while cursor.children:
        cursor = cursor.children[0]
        depth += 1
    check(f"嵌套被截断到 {MAX_ELEMENT_DEPTH} 层", depth <= MAX_ELEMENT_DEPTH, str(depth))
    check("未因深嵌套崩溃", True)


def test_bad_timestamp() -> None:
    print("\n[5] 时间戳异常时兜底")
    rec = GroupMessageRecord.from_payload(
        frame({**TEXT_MESSAGE, "timestamp": "not-a-date"})
    )
    check("回退到当前时间而非抛异常", isinstance(rec.ts, datetime))


def test_store() -> None:
    print("\n[6] SQLite 存储")
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLStore(Path(tmp) / "test.db")

        recs = [
            GroupMessageRecord.from_payload(frame(TEXT_MESSAGE, "E1")),
            GroupMessageRecord.from_payload(frame(IMAGE_MESSAGE, "E2")),
            GroupMessageRecord.from_payload(frame(QUOTE_MESSAGE, "E3")),
        ]
        check("首次写入 3 条", store.insert_messages(recs) == 3)

        # 同一条消息被 QQ 重复推送时不应重复入库
        check("重复写入被去重", store.insert_messages(recs) == 0)

        recent = store.recent_messages("G_demo", limit=10)
        check("取回 3 条且按时间正序", len(recent) == 3 and recent[0]["author_name"] == "小明")

        # 落库的是「文本化」的正文，不是原始 content：引用消息的原始 content 是
        # 空白、图片消息的文字全在附件上，直接存原始值会让这些消息在 prompt 里
        # 变成一行空白。原文（原始 d）仍完整留在 raw_json。
        by_id = {row["message_id"]: row for row in recent}
        check(
            "图片消息的附件标签落库",
            "[图片 photo.jpg]" in by_id["ROBOT1.0_image"]["content"],
            by_id["ROBOT1.0_image"]["content"],
        )
        check(
            "引用消息的引用正文落库",
            "明天有空吗" in by_id["ROBOT1.0_quote"]["content"],
            by_id["ROBOT1.0_quote"]["content"],
        )
        check(
            "原文仍保留在 raw_json",
            "message_scene" in by_id["ROBOT1.0_quote"]["raw_json"],
        )

        rng = store.messages_in_range(
            "G_demo", "2026-07-21T09:00:00+08:00", "2026-07-21T11:00:00+08:00", limit=10
        )
        check("时间范围过滤", len(rng) == 2, str(len(rng)))

        # Summaries — not messages — are what the vector index is built from.
        sid1 = store.insert_summary(
            group_openid="G_demo",
            instruction="总结一下",
            content="会议推迟到下午三点，小红负责准备材料。",
            coverage=[("2026-07-21T08:00:00+08:00", "2026-07-21T10:00:00+08:00")],
            message_count=12,
            requested_by="M_ming",
        )
        check("总结入库返回 id", isinstance(sid1, str) and len(sid1) == 32, str(sid1))

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
        check("同文不同覆盖范围的总结不被吞掉", sid2 != sid1)
        check("库中确实有两篇", len(store.summaries_for("G_demo")) == 2)

        check("未索引积压为 2", len(store.unindexed_summaries(limit=10)) == 2)
        store.mark_summaries_indexed([sid1])
        check("标记后积压剩 1", len(store.unindexed_summaries(limit=10)) == 1)
        check("清空标记后积压回满", store.clear_summary_marks() == 2
              and len(store.unindexed_summaries(limit=10)) == 2)

        one = store.summaries_for("G_demo", limit=10)[-1]  # oldest first
        check("覆盖条数被记录", one["message_count"] == 12, str(one["message_count"]))
        check(
            "覆盖范围被解析成包络",
            str(one["ts_start"]).startswith("2026-07-21T08:00")
            and str(one["ts_end"]).startswith("2026-07-21T10:00"),
            f"{one['ts_start']} ~ {one['ts_end']}",
        )
        check(
            "coverage_json 保留原始区间",
            json.loads(one["coverage_json"])[0][0].startswith("2026-07-21T08:00"),
        )
        check("新总结默认待索引", one["indexed_at"] is None)

        stats = store.stats()
        check(
            "统计 total=3 summaries=2 indexed=0（标记刚被清空）",
            stats[0]["total"] == 3 and stats[0]["summaries"] == 2 and stats[0]["indexed"] == 0,
            str(dict(stats[0])),
        )
        check("不带群过滤能看到所有总结", len(store.summaries_for(None)) == 2)
        check("带群过滤只看到本群", store.summaries_for("G_other") == [])
        groups = store.groups_with_summaries()
        check("groups_with_summaries 只列有总结的群",
              len(groups) == 1 and groups[0]["group_openid"] == "G_demo",
              str([dict(g) for g in groups]))

        check("raw_json 可反序列化", json.loads(recent[0]["raw_json"])["group_openid"] == "G_demo")
        check("群过滤生效", store.recent_messages("G_other", limit=10) == [])

        # ---- who wrote the document, and how many messages are pending -----
        check("被 @ 触发的总结记为 at", one["trigger"] == "at", str(one["trigger"]))

        for i in range(3):
            store.insert_messages(
                [
                    GroupMessageRecord.from_payload(
                        frame({**TEXT_MESSAGE, "id": f"CNT{i}", "group_openid": "G_cnt"})
                    )
                ]
            )
        check("没有总结时从 epoch 起算，全部计入", store.messages_since_last_summary("G_cnt") == 3)

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
        check("自动总结单独标记 trigger", auto["trigger"] == "auto", str(dict(auto)))
        check("自动总结没有触发者", auto["requested_by"] is None)
        check("总结之后计数归零", store.messages_since_last_summary("G_cnt") == 0)
        check("计数按群隔离", store.messages_since_last_summary("G_demo") == 0)

        # The count is defined against the *summary*, so the bound has to be
        # moved to prove the comparison works at all. Backdating beats sleeping:
        # both timestamps have second resolution, and rows written inside the
        # same second compare equal — which is exactly the flake a sleep invites.
        with store._conn:
            store._conn.execute(
                "UPDATE summaries SET created_at = ? WHERE group_openid = ?",
                ("2026-07-21 07:00:00", "G_cnt"),
            )
        check("总结被回拨后，之后的消息重新计入",
              store.messages_since_last_summary("G_cnt") == 3)
        store.insert_messages(
            [
                GroupMessageRecord.from_payload(
                    frame({**TEXT_MESSAGE, "id": "CNT9", "group_openid": "G_cnt"})
                )
            ]
        )
        check("计数随新消息增长", store.messages_since_last_summary("G_cnt") == 4)

        # A group can be full of raw messages and have no summary at all; the
        # private-chat tool still has to be able to name it.
        check(
            "groups_with_messages 列出所有有消息的群",
            set(store.groups_with_messages()) == {"G_demo", "G_cnt"},
            str(store.groups_with_messages()),
        )
        store.close()


class FakeAPI:
    """Stands in for `botpy.BotAPI`, which would otherwise hit the network."""

    def __init__(self, fail_on: int | None = None, return_none_on: int | None = None):
        self.sent: list[dict] = []
        self._fail_on = fail_on
        self._return_none_on = return_none_on

    def _record(self, kind: str, kwargs: dict):
        """Record one outbound call and mimic botpy's success/failure shapes.

        Both endpoints share one counter, so `fail_on` / `return_none_on` mean
        the same thing no matter which route a test exercises. Each entry is
        tagged with `kind` so a test can assert *which* endpoint was used.
        """
        self.sent.append({"kind": kind, **kwargs})
        n = len(self.sent)
        if self._fail_on == n:
            raise SequenceNumberError("msg_id+msg_seq 重复")
        if self._return_none_on == n:
            return None  # how botpy reports a timeout / connection reset
        return {"id": f"out-{n}"}

    async def post_group_message(self, **kwargs):
        return self._record("group", kwargs)

    async def post_c2c_message(self, *, openid, **kwargs):
        # `openid`, not `group_openid` — the one real difference between the two
        # routes (`botpy/api.py:1380` vs `:1426`). Keyword-only here so that a
        # caller passing the wrong name fails loudly instead of silently.
        return self._record("c2c", {"openid": openid, **kwargs})


def test_plan_chunks() -> None:
    print("\n[7] 回复切分")
    chunks, truncated = plan_chunks("一点短内容", 100, 5)
    check("短内容单段且不截断", chunks == ["一点短内容"] and not truncated)

    text = "\n\n".join(f"段落{i}" + "字" * 20 for i in range(6))
    chunks, truncated = plan_chunks(text, 60, 5)
    check("按段落打包", len(chunks) >= 2 and all(len(c) <= 60 for c in chunks), str(len(chunks)))
    check("段落边界未被拆开", all("段落" in c for c in chunks))
    check("未超限时不算截断", not truncated or len(chunks) > 5)

    # 500 chars at limit 100 is exactly 5 chunks — it fits, so no truncation.
    chunks, truncated = plan_chunks("字" * 500, 100, 5)
    check("恰好 5 段不算截断", len(chunks) == 5 and not truncated, str(len(chunks)))

    chunks, truncated = plan_chunks("字" * 600, 100, 5)
    check("超长单段落被硬切且每段不超限", all(len(c) <= 100 for c in chunks), str(chunks))
    check("段数上限被遵守", len(chunks) == 5, str(len(chunks)))
    check("截断被标记", truncated)
    check("末段是截断提示", "省略" in chunks[-1], chunks[-1])


def test_reply_chunked() -> None:
    print("\n[8] 发送（被动回复 / 失败路径）")

    async def scenario() -> None:
        api = FakeAPI()
        result = await reply_chunked(api, "G_demo", "MSG_1", "第一段\n\n第二段", elapsed_s=1.0)
        check("发送成功两段", result.sent == 1 and result.ok, str(result))
        check("携带 msg_id 走被动回复", api.sent[0]["msg_id"] == "MSG_1")
        check("msg_seq 从 1 开始", api.sent[0]["msg_seq"] == 1)
        check("msg_type 为 0（纯文本）", api.sent[0]["msg_type"] == 0)

        api = FakeAPI()
        result = await reply_chunked(
            api, "G_demo", "MSG_1", "正文", elapsed_s=999.0
        )
        check("超时后降级为主动消息", result.active and api.sent[0]["msg_id"] is None)

        api = FakeAPI(return_none_on=1)
        result = await reply_chunked(api, "G_demo", "MSG_1", "正文")
        check("None 返回被判为失败", not result.ok and "None" in (result.error or ""), str(result))

        api = FakeAPI(fail_on=1)
        result = await reply_chunked(api, "G_demo", "MSG_1", "正文")
        check("429 被捕获且不重试", not result.ok and len(api.sent) == 1, str(result))

        api = FakeAPI()
        result = await reply_chunked(api, "G_demo", "MSG_1", "   ")
        check("空内容不发送", result.sent == 0 and not api.sent)

        limit = config.summary.max_reply_chars
        api = FakeAPI()
        result = await reply_chunked(api, "G_demo", "MSG_1", "字" * (limit * 8))
        check("超长摘要最多发 5 条", len(api.sent) == 5, str(len(api.sent)))
        check(
            "msg_seq 严格递增",
            [m["msg_seq"] for m in api.sent] == [1, 2, 3, 4, 5],
            str([m["msg_seq"] for m in api.sent]),
        )
        check("每段都不超限", all(len(m["content"]) <= limit for m in api.sent))
        check("末段为截断提示", "省略" in api.sent[-1]["content"], api.sent[-1]["content"])
        check("截断被如实上报", result.truncated and result.sent == 5)

        # C2C shares this path but not the target keyword. This is the only
        # place the two routes actually differ, so it is worth pinning.
        api = FakeAPI()
        result = await reply_chunked(
            api, "U_alice", "C2C_MSG_1", "私聊回复", kind=KIND_C2C, elapsed_s=1.0
        )
        check("私聊走 post_c2c_message", result.ok and api.sent[0]["kind"] == "c2c", str(api.sent))
        check("私聊目标参数名是 openid", api.sent[0]["openid"] == "U_alice", str(api.sent[0]))
        check("私聊不带 group_openid", "group_openid" not in api.sent[0])
        check("私聊同样带 msg_id 走被动回复", api.sent[0]["msg_id"] == "C2C_MSG_1")

        # The group-only "expired window → active message" downgrade must NOT
        # fire for C2C: that path consumes group quota, and C2C active messages
        # follow their own rules and may simply be rejected.
        api = FakeAPI()
        result = await reply_chunked(
            api, "U_alice", "C2C_MSG_1", "私聊回复", kind=KIND_C2C, elapsed_s=999.0
        )
        check("私聊超窗不降级为主动消息", not result.active and not api.sent, str(result))
        check(
            "私聊超窗如实记为失败",
            not result.ok and "expired" in (result.error or ""),
            str(result),
        )

    asyncio.run(scenario())


class FakeSummarizer:
    """Stands in for `Summarizer`, returning a document-worthy `SummaryResult`.

    The text is deliberately longer than `MIN_DOC_CHARS` and carries a non-empty
    coverage log, because both are required for the answer to be stored —
    a fake that returned "摘要：…" would make the storage assertions below vacuous.
    """

    TEXT = (
        "摘要：群里讨论了明天的会议，决定推迟到下午三点。"
        "小红负责整理会议材料，小刚确认了会议室。"
    )

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.private_calls: list[tuple[str, str]] = []

    @staticmethod
    def _result(tool: str, rows: int) -> SummaryResult:
        coverage = CoverageLog()
        coverage.add(
            tool, rows, "2026-07-21T08:00:00+08:00", "2026-07-21T09:00:00+08:00"
        )
        return SummaryResult(text=FakeSummarizer.TEXT, coverage=coverage)

    async def summarize_group(self, group_openid: str, instruction: str) -> SummaryResult:
        self.calls.append((group_openid, instruction))
        return self._result("recent_messages", rows=5)

    async def answer_private(self, user_openid: str, instruction: str) -> SummaryResult:
        self.private_calls.append((user_openid, instruction))
        return self._result("search_summaries", rows=0)

    def group_busy(self, group_openid: str) -> bool:
        """Never busy: this fake runs no lock. `Summarizer.group_busy`'s contract."""
        return False


class ExplodingSummarizer:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def summarize_group(self, group_openid: str, instruction: str) -> SummaryResult:
        self.calls.append((group_openid, instruction))
        raise RuntimeError("LLM 挂了")

    async def answer_private(self, user_openid: str, instruction: str) -> SummaryResult:
        raise RuntimeError("LLM 挂了")

    def group_busy(self, group_openid: str) -> bool:
        return False


def test_client_seam() -> None:
    print("\n[9] Client 接缝（不连 QQ）")

    async def scenario() -> None:
        from src.bot.client import SummarizerClient

        # `Client.__init__` reads `asyncio.get_event_loop()`; running inside a
        # loop keeps that on the supported path instead of the deprecated one.
        check(
            "intent 含 public_messages (1<<25)",
            int(botpy.Intents(public_messages=True).value) == 1 << 25,
        )

        # Replace only the network half of `_bot_login`, keeping the session
        # construction that the real one performs.
        original = botpy.Client._bot_login
        loop = asyncio.get_running_loop()

        async def fake_login(self, token):
            self._connection = botpy.connection.ConnectionSession(
                max_async=1,
                connect=self.bot_connect,
                dispatch=self.ws_dispatch,
                loop=loop,
                api=self.api,
            )

        botpy.Client._bot_login = fake_login
        try:
            with tempfile.TemporaryDirectory() as tmp:
                store = SQLStore(Path(tmp) / "c.db")
                summarizer = FakeSummarizer()
                # Counted rather than asserted immediately: the interesting
                # property is *when* the indexer is woken, so the list is read
                # at points where nothing, and then something, should have
                # added to it.
                wakes: list[int] = []
                client = SummarizerClient(
                    store=store,
                    summarizer=summarizer,
                    wake_indexer=lambda: wakes.append(1),
                    bot_log=True,
                    ext_handlers=False,
                )
                client.api = FakeAPI()

                # A pristine session, to show the gap we are working around:
                # botpy builds its parser table from `parse_*` methods alone.
                probe = botpy.connection.ConnectionSession(
                    max_async=1,
                    connect=client.bot_connect,
                    dispatch=client.ws_dispatch,
                    loop=loop,
                    api=client.api,
                )
                check(
                    "botpy 原生 parser 里没有该事件",
                    "group_message_create" not in probe.parser,
                )

                # `_bot_login` builds the session, so the registration lands on
                # the session the gateway will actually read from.
                await client._bot_login(None)
                check(
                    "注册后 parser 里有该事件",
                    "group_message_create" in client._connection.parser,
                )
                # connection.py:40 makes these the same dict object, which is why
                # mutating one is enough for gateway.py to find the parser.
                check(
                    "parser 与 state.parsers 是同一个 dict",
                    client._connection.parser is client._connection.state.parsers,
                )

                parser = client._connection.parser["group_message_create"]

                # The parser runs synchronously inside the ws read loop and is
                # NOT wrapped in a try by botpy (gateway.py:99), so malformed
                # payloads must be swallowed rather than kill the connection.
                for bad in ({"t": "GROUP_MESSAGE_CREATE"}, {"d": "not-a-dict"}, None):
                    try:
                        parser(bad)
                        check(f"畸形 payload 不抛异常: {bad!r}", True)
                    except Exception as exc:  # noqa: BLE001
                        check(f"畸形 payload 不抛异常: {bad!r}", False, repr(exc))

                parser(frame(SUMMON_MESSAGE))
                await asyncio.sleep(0.05)
                check("被 @ 的消息触发了摘要", len(summarizer.calls) == 1, str(summarizer.calls))
                check(
                    "指令取自被剥掉 @ 后的正文",
                    summarizer.calls[0] == ("G_demo", "总结一下今天群里聊了什么"),
                    str(summarizer.calls),
                )
                check("摘要被回发", len(client.api.sent) == 1, str(client.api.sent))
                check(
                    "回复是带 msg_id 的被动回复",
                    client.api.sent[0]["msg_id"] == "ROBOT1.0_summon",
                    str(client.api.sent[0]),
                )
                check("消息已落库", len(store.recent_messages("G_demo", 10)) == 1)

                # Same message again — QQ can push one @-message as both events.
                parser(frame(SUMMON_MESSAGE))
                await asyncio.sleep(0.05)
                check("重复推送不再触发第二次摘要", len(summarizer.calls) == 1)
                check("重复推送不再回发", len(client.api.sent) == 1)

                parser(frame(TEXT_MESSAGE))
                await asyncio.sleep(0.05)
                check("未 @ 的消息只落库不回复", len(client.api.sent) == 1)
                check("未 @ 的消息也落库了", len(store.recent_messages("G_demo", 10)) == 2)

                # The answer itself is now a document, and it is stored
                # *before* the send, so a failure to deliver cannot lose it.
                stored = store.summaries_for("G_demo")
                check("被 @ 的回答作为总结落库", len(stored) == 1, str(len(stored)))
                check("总结记录了群与取数条数", stored[0]["group_openid"] == "G_demo"
                      and stored[0]["message_count"] == 5, str(dict(stored[0])))
                check("总结记下了触发者", stored[0]["requested_by"] == "M_ming")
                check("总结保留原始指令", stored[0]["instruction"] == "总结一下今天群里聊了什么")
                check("总结尚待索引", stored[0]["indexed_at"] is None)

                # Waking the indexer moved from "every message" to "every
                # stored summary": nothing per-message is embedded any more, and
                # the insert that does need indexing is the summary itself.
                check("三条消息本身不唤醒索引器", len(wakes) == 1, str(len(wakes)))
                check("总结入库时唤醒一次索引器", len(wakes) == 1, str(len(wakes)))

                # ---- private chat -------------------------------------------
                class _Author:
                    def __init__(self, openid: str) -> None:
                        self.user_openid = openid

                class FakeC2C:
                    def __init__(self, openid: str = "U_alice") -> None:
                        self.id = "C2C_1"
                        self.content = "之前有人提过部署方案吗"
                        self.author = _Author(openid)

                private_api = FakeAPI()
                client.api = private_api
                await client.on_c2c_message_create(FakeC2C())
                check(
                    "私聊触发了跨群回答",
                    summarizer.private_calls == [("U_alice", "之前有人提过部署方案吗")],
                    str(summarizer.private_calls),
                )
                check(
                    "私聊回复走 post_c2c_message",
                    len(private_api.sent) == 1 and private_api.sent[0]["kind"] == "c2c",
                    str(private_api.sent),
                )
                check("私聊回复发送给 user_openid", private_api.sent[0]["openid"] == "U_alice")
                # Private messages carry `user_openid`, not `group_openid`; the
                # group table's column is NOT NULL and a C2C chat is not a group.
                check(
                    "私聊消息不写入 group_messages",
                    store.recent_messages("U_alice", 10) == []
                    and len(store.recent_messages("G_demo", 10)) == 2,
                )
                check("私聊回答不产生总结", len(store.summaries_for(None)) == 1)
                check("私聊不唤醒索引器", len(wakes) == 1, str(len(wakes)))

                # Empty allowlist means everyone; a non-empty one is a gate.
                config.c2c.allowlist = ["U_allowed"]
                try:
                    gated = FakeAPI()
                    client.api = gated
                    await client.on_c2c_message_create(FakeC2C("U_blocked"))
                    check("allowlist 之外的用户被拒", not gated.sent)
                    check("被拒的私聊不消耗模型调用", len(summarizer.private_calls) == 1)

                    client.api = FakeAPI()
                    await client.on_c2c_message_create(FakeC2C("U_allowed"))
                    check("allowlist 内的用户被放行", len(client.api.sent) == 1)
                    check("放行后才产生第二次调用", len(summarizer.private_calls) == 2)
                finally:
                    config.c2c.allowlist = []
                store.close()

            # A send that fails (here: botpy's silent `None`) must not cost the
            # summary — a 429 or a timeout would otherwise lose it for good.
            with tempfile.TemporaryDirectory() as tmp:
                store = SQLStore(Path(tmp) / "c4.db")
                client = SummarizerClient(
                    store=store,
                    summarizer=FakeSummarizer(),
                    bot_log=True,
                    ext_handlers=False,
                )
                client.api = FakeAPI(return_none_on=1)
                await client._bot_login(None)
                client._connection.parser["group_message_create"](frame(SUMMON_MESSAGE))
                await asyncio.sleep(0.05)
                check("发送失败时总结仍然落库", len(store.summaries_for("G_demo")) == 1)
                store.close()

            # A failing agent must still answer the group, not go silent.
            with tempfile.TemporaryDirectory() as tmp:
                store = SQLStore(Path(tmp) / "c2.db")
                client = SummarizerClient(
                    store=store,
                    summarizer=ExplodingSummarizer(),
                    bot_log=True,
                    ext_handlers=False,
                )
                client.api = FakeAPI()
                await client._bot_login(None)
                client._connection.parser["group_message_create"](frame(SUMMON_MESSAGE))
                await asyncio.sleep(0.05)
                check("agent 抛异常时仍回一条兜底消息", len(client.api.sent) == 1)
                check(
                    "兜底文案可读",
                    "出错" in client.api.sent[0]["content"],
                    client.api.sent[0]["content"],
                )
                check("失败的调用不产生总结", store.summaries_for("G_demo") == [])
                store.close()
        finally:
            botpy.Client._bot_login = original

    asyncio.run(scenario())


class FakeEmbeddings:
    """Deterministic bag-of-characters embedding.

    Not semantic, but it rewards token overlap, which is enough to verify that
    retrieval returns the *right* message first and that the group filter holds.

    Two details are load-bearing, and getting either wrong makes this suite
    flaky rather than wrong:

    * **A stable hash, not the builtin `hash()`.** `hash()` on `str` is salted
      per process, so bucket assignment — and therefore the ranking this class
      exists to produce — changed from run to run. Asserting "the same-group
      meeting message ranks first" then passed or failed by interpreter seed.
      `crc32` is identical in every process, machine and Python version.
    * **A dimension large enough that collisions do not decide the ranking.**
      With a small `DIM`, an unrelated CJK character lands in a query bucket
      often enough to outrank a genuinely overlapping message, which is the
      other half of why the old version flaked.
    """

    DIM = 1024

    def _vec(self, text: str) -> list[float]:
        import math
        import re
        import zlib

        v = [0.0] * self.DIM
        for token in re.findall(r"[\w一-鿿]+", text.lower()):
            pieces = [token] if token.isascii() else list(token)
            for piece in pieces:
                v[zlib.crc32(piece.encode("utf-8")) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def test_rag() -> None:
    print("\n[10] 向量索引与检索（真 Chroma / 假 embedding）")
    from src.api.embedding_client import MAX_EMBED_BATCH, embedding_batch_size
    from src.rag.retriever import SummaryIndex, summary_text

    check("批量上限的保守默认值是 25", MAX_EMBED_BATCH == 25)
    check("默认批量不超过网关上限", embedding_batch_size() <= MAX_EMBED_BATCH)

    # `ignore_cleanup_errors`: Chroma memory-maps its HNSW segment files, and
    # Windows refuses to unlink them until the client is evicted. That is a
    # Chroma-on-Windows artifact — the bot keeps this directory for its whole
    # life, so it says nothing about our code.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = SQLStore(Path(tmp) / "r.db")
        db = Path(tmp) / "chroma"

        # One document per *summary*, not per message.
        store.insert_summary(
            group_openid="G_demo",
            instruction="总结一下",
            content="大家讨论了明天的会议，决定推迟到下午三点，小红负责整理会议纪要。",
            coverage=[("2026-07-21T08:00:00+08:00", "2026-07-21T10:00:00+08:00")],
            message_count=12,
        )
        store.insert_summary(
            group_openid="G_other",
            instruction="总结一下",
            content="群里的结论是会议时间改到周一，另外约了周末一起去爬山。",
            coverage=[("2026-07-21T11:00:00+08:00", "2026-07-21T12:00:00+08:00")],
            message_count=4,
        )
        rows = [dict(r) for r in store.unindexed_summaries(limit=10)]
        check("取到 2 篇待索引总结", len(rows) == 2)

        text = summary_text(rows[0])
        header = text.splitlines()[0]
        check("正文首行是群标签", header.startswith("[群"), header)
        check("首行含覆盖范围", "2026-07-21 08:00" in header and "2026-07-21 10:00" in header, header)
        check("首行含条数", "12 条" in header, header)
        check("正文含总结正文", "推迟到下午三点" in text, text)
        # Retrieval only ever sees `page_content`, so group and window must live
        # in the body — a private-chat answer has no other way to cite them.
        check("空正文没有可嵌入文本", summary_text({"content": "   "}) == "")

        index = SummaryIndex(
            embedding_function=FakeEmbeddings(),
            persist_directory=db,
            collection_name="test_summaries",
        )
        check("首次索引 2 篇", index.add(rows) == 2)
        check("入库计数正确", index.count() == 2)
        check("重复索引同一批不增容", index.add(rows) == 2 and index.count() == 2)

        hits = index.search("会议推迟到几点", k=3, group_openid="G_demo")
        check("群内检索有结果", len(hits) > 0, str(len(hits)))
        check(
            "命中的是本群的会议总结",
            hits[0][0].metadata["group_openid"] == "G_demo",
            hits[0][0].page_content,
        )
        check(
            "结果全部限定在本群",
            all(d.metadata["group_openid"] == "G_demo" for d, _ in hits),
            str([d.metadata["group_openid"] for d, _ in hits]),
        )
        check("别群的总结未被召回", all("爬山" not in d.page_content for d, _ in hits))

        # `group_openid=None` is the private-chat path: deliberately unfiltered.
        everything = index.search("会议", k=5)
        seen = {d.metadata["group_openid"] for d, _ in everything}
        check("不过滤时跨群召回（私聊路径）", seen == {"G_demo", "G_other"}, str(seen))

        check(
            "元数据都是标量",
            all(
                isinstance(v, (str, int, float, bool))
                for d, _ in everything
                for v in d.metadata.values()
            ),
        )
        check("元数据用 summary_id 作为标识", all(d.metadata["summary_id"] for d, _ in everything))

        # A blank summary has nothing to embed. It must still be marked, or it
        # would clog the backlog forever.
        store.insert_summary(
            group_openid="G_demo",
            instruction="（空）",
            content="   ",
            coverage=[("2026-07-21T13:00:00+08:00", "2026-07-21T13:01:00+08:00")],
            message_count=0,
        )

        async def drain_check() -> None:
            from src.rag.indexer import SummaryIndexer

            fresh = SummaryIndex(
                embedding_function=FakeEmbeddings(),
                persist_directory=Path(tmp) / "chroma2",
                collection_name="drain_summaries",
            )
            indexer = SummaryIndexer(store, fresh, batch=10)
            check("积压初始为 3", len(store.unindexed_summaries(limit=100)) == 3)
            total = await indexer.drain()
            check("drain 只索引了有正文的 2 篇", total == 2, str(total))
            check("空总结被跳过但已标记", len(store.unindexed_summaries(limit=100)) == 0)
            check("集合内确实只有 2 篇", fresh.count() == 2, str(fresh.count()))
            check("drain 后再跑无事可做", await indexer.drain() == 0)

        asyncio.run(drain_check())
        store.close()


def test_agent_boundary() -> None:
    print("\n[11] Agent 工具与跨群隔离（假模型 + 真 agent 循环）")
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, ToolMessage

    from src.agent.summarizer import (
        MAX_THREADS,
        NO_ANSWER,
        Summarizer,
        store_summary,
    )
    from src.agent.tools import (
        C2C_TOOLS,
        DATA_TOOLS,
        GROUP_TOOLS,
        SCOPE_ALL,
        SCOPE_GROUP,
        BotContext,
        list_groups,
        messages_across_groups,
        messages_in_range,
        recent_messages,
        search_summaries,
    )
    from src.rag.retriever import SummaryIndex, group_label

    group_by_name = {t.name: t for t in GROUP_TOOLS}
    c2c_by_name = {t.name: t for t in C2C_TOOLS}
    check(
        "群内工具集是那 4 个",
        set(group_by_name)
        == {"current_time", "recent_messages", "messages_in_range", "search_summaries"},
        str(sorted(group_by_name)),
    )
    check(
        "私聊工具集是那 4 个",
        set(c2c_by_name)
        == {"current_time", "search_summaries", "list_groups", "messages_across_groups"},
        str(sorted(c2c_by_name)),
    )
    # The two sets exist precisely because these two lines must both hold.
    check("群内工具集没有跨群能力", "list_groups" not in group_by_name)
    check("群内工具集读不到别群的原文", "messages_across_groups" not in group_by_name)
    check(
        "私聊工具集没有按群读本群原文的工具",
        "recent_messages" not in c2c_by_name and "messages_in_range" not in c2c_by_name,
    )
    # `DATA_TOOLS` drives the "tools ran but coverage is empty" warning in
    # `Summarizer._run`; a data-reading tool missing from it makes that warning
    # fire on every honest answer.
    check(
        "所有取数工具都登记在 DATA_TOOLS 里",
        {"recent_messages", "messages_in_range", "search_summaries", "messages_across_groups"}
        == set(DATA_TOOLS),
        str(sorted(DATA_TOOLS)),
    )
    check("current_time 不算取数工具", "current_time" not in DATA_TOOLS)

    # `tool_call_schema` is what `bind_tools` hands the model; injected args are
    # filtered out of it but stay in `get_input_schema()`, which is where the
    # injection machinery looks. The model must never see any of them either way.
    for name, tool in {**group_by_name, **c2c_by_name}.items():
        visible = set(tool.tool_call_schema.model_fields)
        internal = set(tool.get_input_schema().model_fields)
        check(f"{name}: 模型看不到 runtime", "runtime" not in visible, str(visible))
        check(f"{name}: runtime 仍参与注入", "runtime" in internal, str(internal))
        for hidden in ("group_openid", "scope"):
            check(
                f"{name}: 模型看不到 {hidden}",
                hidden not in visible and hidden not in internal,
                str(visible | internal),
            )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = SQLStore(Path(tmp) / "a.db")
        index = SummaryIndex(
            embedding_function=FakeEmbeddings(),
            persist_directory=Path(tmp) / "chroma",
            collection_name="agent_summaries",
        )

        # Raw messages back the time-range tools; summaries back the semantic
        # one. They are separate stores with separate lifetimes.
        #
        # The third group is a realistic 32-char openid and deliberately has **no
        # summary**: that is the state a feedback group sits in until it either
        # gets @-ed or crosses the auto-summary threshold, and private chat still
        # has to be able to name it.
        REAL_OPENID = "A1B2C3D4E5F60718293A4B5C6D7E8F90"
        rows_by_group = {
            "G_demo": [("小明", "本群机密：项目代号是蓝色鲸鱼"), ("小红", "收到，我去准备会议材料")],
            "G_other": [("外人", "别群的机密：项目代号是红色狐狸")],
            REAL_OPENID: [("路人", "没有总结的群里的原话")],
        }
        n = 0
        for gid, items in rows_by_group.items():
            for who, text in items:
                rec = GroupMessageRecord.from_payload(
                    frame(
                        {
                            "id": f"A{n}",
                            "group_openid": gid,
                            "content": text,
                            "timestamp": f"2026-07-21T0{n}:00:00+08:00",
                            "author": {"username": who, "member_openid": f"U{n}"},
                        }
                    )
                )
                store.insert_messages([rec])
                n += 1
        check("两个群的消息都已入库", len(store.recent_messages("G_demo", 10)) == 2)

        store.insert_summary(
            group_openid="G_demo",
            instruction="总结一下",
            content="本群机密：项目代号是蓝色鲸鱼，小红在准备会议材料。",
            coverage=[("2026-07-21T00:00:00+08:00", "2026-07-21T01:00:00+08:00")],
            message_count=20,
        )
        store.insert_summary(
            group_openid="G_demo",
            instruction="总结一下",
            content="本群昨天的结论：部署方案定在周五上线。",
            coverage=[("2026-07-20T00:00:00+08:00", "2026-07-20T01:00:00+08:00")],
            message_count=8,
        )
        store.insert_summary(
            group_openid="G_other",
            instruction="总结一下",
            content="别群的机密：项目代号是红色狐狸，本周不讨论上线。",
            coverage=[("2026-07-21T02:00:00+08:00", "2026-07-21T03:00:00+08:00")],
            message_count=5,
        )
        summary_rows = [dict(r) for r in store.unindexed_summaries(limit=10)]
        check("三个群的总结都已入库", len(summary_rows) == 3)
        check("总结被索引", index.add(summary_rows) == 3)

        # Tool bodies, called with a stand-in runtime (bypassing the agent loop).
        class _RT:
            def __init__(self, gid, scope=SCOPE_GROUP):
                self.context = BotContext(
                    group_openid=gid, store=store, index=index, scope=scope
                )

        async def tool_checks() -> None:
            rt = _RT("G_demo")
            text = await recent_messages.coroutine(limit=10, runtime=rt)
            check("工具按注入的群取数", "蓝色鲸鱼" in text, text)
            check("工具不会带回别群内容", "红色狐狸" not in text, text)
            check(
                "取数被记入覆盖范围",
                not rt.context.coverage.is_empty()
                and rt.context.coverage.message_count() == 2,
                str(rt.context.coverage.reads),
            )

            rt = _RT("G_demo")
            ranged = await messages_in_range.coroutine(
                start_iso="2026-07-21T00:00:00+08:00",
                end_iso="2026-07-22T00:00:00+08:00",
                limit=50,
                runtime=rt,
            )
            check("时间范围工具正常", "蓝色鲸鱼" in ranged and "红色狐狸" not in ranged, ranged)
            check(
                "时间范围被记成可入库的区间",
                rt.context.coverage.intervals() != [],
                str(rt.context.coverage.reads),
            )

            # Bounds in a different offset must still be interpreted correctly.
            utc = await messages_in_range.coroutine(
                start_iso="2026-07-20T16:00:00Z",
                end_iso="2026-07-21T16:00:00Z",
                limit=50,
                runtime=_RT("G_demo"),
            )
            check("带 Z 的边界能正确比对（时区归一）", "蓝色鲸鱼" in utc, utc)

            bad = await messages_in_range.coroutine(
                start_iso="昨天", end_iso="今天", limit=5, runtime=_RT("G_demo")
            )
            check("非法时间返回可读错误而不是抛异常", "无法解析" in bad, bad)

            # Same tool, opposite scopes. This pair is the whole point of having
            # two agents: one never leaves its group, the other always spans them.
            inside = _RT("G_demo")
            found = await search_summaries.coroutine(
                query="项目代号是什么", k=5, runtime=inside
            )
            check("群内语义检索有结果", "蓝色鲸鱼" in found, found)
            check("群内语义检索看不到别群的总结", "红色狐狸" not in found, found)
            check("检索结果自带群标签", found.startswith("[群"), found[:40])
            check(
                "只检索总结时不计入消息条数",
                inside.context.coverage.message_count() == 0
                and not inside.context.coverage.is_empty(),
                str(inside.context.coverage.reads),
            )

            across_ctx = _RT(None, scope=SCOPE_ALL)
            across = await search_summaries.coroutine(
                query="项目代号是什么", k=5, runtime=across_ctx
            )
            check(
                "私聊 scope 能跨群召回",
                "蓝色鲸鱼" in across and "红色狐狸" in across,
                across,
            )

            groups = await list_groups.coroutine(runtime=_RT(None, scope=SCOPE_ALL))
            check(
                "私聊能列出有总结的群",
                group_label("G_demo") in groups and group_label("G_other") in groups,
                groups,
            )

            # ---- raw messages across groups (the private-chat-only tool) -----
            WEEK = ("2026-07-21T00:00:00+08:00", "2026-07-22T00:00:00+08:00")

            wide_rt = _RT(None, scope=SCOPE_ALL)
            wide = await messages_across_groups.coroutine(
                start_iso=WEEK[0], end_iso=WEEK[1], runtime=wide_rt
            )
            check("跨群取原文命中多个群", "蓝色鲸鱼" in wide and "红色狐狸" in wide, wide)
            check(
                "每一行都带群标签，来源说得清",
                group_label("G_demo") in wide and group_label(REAL_OPENID) in wide,
                wide,
            )
            check(
                "跨群取数被记入覆盖范围",
                wide_rt.context.coverage.message_count() == 4,
                str(wide_rt.context.coverage.reads),
            )

            named = _RT(None, scope=SCOPE_ALL)
            only_demo = await messages_across_groups.coroutine(
                start_iso=WEEK[0], end_iso=WEEK[1], group="G_demo", runtime=named
            )
            check(
                "group= 精确 openid 收敛到该群",
                "蓝色鲸鱼" in only_demo and "红色狐狸" not in only_demo,
                only_demo,
            )
            check(
                "收敛后的取数同样入账",
                named.context.coverage.message_count() == 2,
                str(named.context.coverage.reads),
            )

            # The model only ever sees labels, so both spellings have to resolve:
            # a `[groups]` alias, and the `群<尾号>` stub for an un-aliased group.
            saved_groups = config.groups
            config.groups = {"G_demo": "项目组"}
            try:
                aliased = await messages_across_groups.coroutine(
                    start_iso=WEEK[0], end_iso=WEEK[1], group="项目组",
                    runtime=_RT(None, scope=SCOPE_ALL),
                )
                check("别名能解析成群", "蓝色鲸鱼" in aliased and "红色狐狸" not in aliased, aliased)

                stub = await messages_across_groups.coroutine(
                    start_iso=WEEK[0], end_iso=WEEK[1], group=group_label(REAL_OPENID),
                    runtime=_RT(None, scope=SCOPE_ALL),
                )
                check(
                    f"群<尾号>({group_label(REAL_OPENID)}) 能解析成群",
                    "没有总结的群里的原话" in stub and "蓝色鲸鱼" not in stub,
                    stub,
                )
            finally:
                config.groups = saved_groups

            unknown = await messages_across_groups.coroutine(
                start_iso=WEEK[0], end_iso=WEEK[1], group="查无此群",
                runtime=_RT(None, scope=SCOPE_ALL),
            )
            check("群名解析不了时提示去调 list_groups", "list_groups" in unknown, unknown)

            keyworded = await messages_across_groups.coroutine(
                start_iso=WEEK[0], end_iso=WEEK[1], keyword="狐狸",
                runtime=_RT(None, scope=SCOPE_ALL),
            )
            check(
                "keyword= 只留正文含该词的消息",
                "红色狐狸" in keyworded and "蓝色鲸鱼" not in keyworded,
                keyworded,
            )

            saved_raw_limit = config.c2c.raw_limit
            config.c2c.raw_limit = 1
            try:
                clamped = await messages_across_groups.coroutine(
                    start_iso=WEEK[0], end_iso=WEEK[1], limit=999,
                    runtime=_RT(None, scope=SCOPE_ALL),
                )
                check(
                    "limit 被 c2c.raw_limit 夹住",
                    len([ln for ln in clamped.splitlines() if ln.strip()]) == 1,
                    clamped,
                )
            finally:
                config.c2c.raw_limit = saved_raw_limit

            naive = await messages_across_groups.coroutine(
                start_iso="上周", end_iso="今天", runtime=_RT(None, scope=SCOPE_ALL)
            )
            check("时间无法解析时返回提示而不是抛异常", "无法解析" in naive, naive)

            # The time range is mandatory precisely so the tool cannot be used as
            # an unbounded full-corpus dump. That is a schema property, not a
            # runtime check, so it is asserted on the schema the model sees.
            visible = messages_across_groups.tool_call_schema.model_fields
            check(
                "取原文必须给时间范围（模型不能省略）",
                all(visible[f].is_required() for f in ("start_iso", "end_iso")),
                str({f: visible[f].is_required() for f in visible}),
            )
            check("group/keyword/limit 都可选", not visible["group"].is_required())

        asyncio.run(tool_checks())

        class ToolCallingFakeModel(FakeMessagesListChatModel):
            """Replays canned messages; ignores `bind_tools`."""

            def bind_tools(self, tools, **kwargs):  # noqa: ARG002
                return self

        # The tool call deliberately smuggles a forged `runtime` *and* a forged
        # `scope`/`group_openid`. ToolNode strips LLM-supplied values for injected
        # args (tool_node.py `stripped_args`), so the group answer must stay put.
        # This is the only test that proves the cross-group filter is unforgeable.
        forged = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_summaries",
                    "args": {
                        "query": "项目代号",
                        "k": 5,
                        "runtime": "G_other",
                        "scope": SCOPE_ALL,
                        "group_openid": "G_other",
                    },
                    "id": "call_1",
                }
            ],
        )
        final = AIMessage(
            content="总结：项目代号是蓝色鲸鱼，小红在准备会议材料，部署方案定在周五上线。"
        )
        model = ToolCallingFakeModel(responses=[forged, final])

        summarizer = Summarizer(store, index, model=model)
        result = asyncio.run(summarizer.summarize_group("G_demo", "总结一下最近的消息"))
        check("返回最终回答", "蓝色鲸鱼" in result.text, result.text)
        # Coverage is carried out of the graph by mutating the context object,
        # which only works because LangGraph passes an instance by reference.
        check("coverage 被带出 graph", not result.coverage.is_empty(), str(result.coverage.reads))
        check("这次回答构成可入库文档", result.storable())

        # Re-run to capture the message trace.
        model2 = ToolCallingFakeModel(responses=[forged, final])
        summarizer2 = Summarizer(store, index, model=model2)
        ctx = BotContext(group_openid="G_demo", store=store, index=index, scope=SCOPE_GROUP)
        state = asyncio.run(
            summarizer2._group_agent.ainvoke(
                {"messages": [{"role": "user", "content": "总结一下"}]},
                config={"configurable": {"thread_id": "probe"}},
                context=ctx,
            )
        )
        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        check("工具确实被调用", len(tool_messages) == 1, str(len(tool_messages)))
        check(
            "伪造的 runtime/scope 被忽略，检索仍限定在本群",
            "蓝色鲸鱼" in tool_messages[0].content
            and "红色狐狸" not in tool_messages[0].content,
            tool_messages[0].content,
        )
        check("coverage 随 context 对象传出", not ctx.coverage.is_empty())

        # Thread ids are the only namespace separating the two graphs' state, so
        # a group and a user that share a suffix must not share a thread.
        namespaced = Summarizer(store, index, model=ToolCallingFakeModel(responses=[final]))
        asyncio.run(namespaced.summarize_group("X1", "总结一下"))
        asyncio.run(namespaced.answer_private("X1", "之前聊过部署吗"))
        check(
            "同后缀的群与私聊是不同会话",
            set(namespaced._threads) == {"G:X1", "U:X1"},
            str(list(namespaced._threads)),
        )

        # "Is this answer a document?" — all three conditions are required.
        grounded = CoverageLog()
        grounded.add("recent_messages", 5, "2026-07-21T00:00:00+08:00", "2026-07-21T01:00:00+08:00")
        check("有取数、够长的回答算文档",
              SummaryResult(text=final.content, coverage=grounded).storable())
        check("哨兵回答不算文档",
              not SummaryResult(text=NO_ANSWER, coverage=grounded).storable())
        check("过短的回答不算文档",
              not SummaryResult(text="好的", coverage=grounded).storable())
        check("零工具调用的追问不算文档",
              not SummaryResult(text=final.content, coverage=CoverageLog()).storable())

        check(
            "不合格的回答不入库",
            store_summary(
                store,
                group_openid="G_demo",
                instruction="总结一下",
                requested_by=None,
                result=SummaryResult(text="好的", coverage=grounded),
            )
            is None,
        )
        check("库里仍只有 3 篇", len(store.summaries_for(None)) == 3, str(len(store.summaries_for(None))))

        # Memory is bounded, and private chats share the budget with groups.
        capped = Summarizer(store, index, model=ToolCallingFakeModel(responses=[final]))
        for i in range(MAX_THREADS + 3):
            capped._touch_thread(f"G:{i}")
        check("会话记忆被限制在上限内", len(capped._threads) == MAX_THREADS, str(len(capped._threads)))
        check("淘汰的是最早的会话", "G:0" not in capped._threads)
        check("保留的是最近的会话", f"G:{MAX_THREADS + 2}" in capped._threads)
        store.close()

        # Two runs on one thread must not overlap. They share one checkpointer
        # thread, and interleaved supersteps corrupt its state — reachable today
        # by two fast @s or two DMs from one user, and by an auto-summary racing
        # a summon tomorrow. botpy gives every event its own task, so nothing
        # else serialises them.
        class SlowAgent:
            """Stands in for the compiled graph, and blocks until released."""

            def __init__(self):
                self.gate = asyncio.Event()
                self.live = 0
                self.peak = 0
                self.calls = 0

            async def ainvoke(self, messages, config, context):  # noqa: ARG002
                self.calls += 1
                self.live += 1
                self.peak = max(self.peak, self.live)
                try:
                    await self.gate.wait()
                finally:
                    self.live -= 1
                context.coverage.add(
                    "recent_messages", 1,
                    "2026-07-21T00:00:00+08:00", "2026-07-21T01:00:00+08:00",
                )
                return {"messages": [final]}

        async def serialised() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                lock_store = SQLStore(Path(tmp) / "lock.db")
                agent = SlowAgent()
                busy = Summarizer(
                    lock_store, index, model=ToolCallingFakeModel(responses=[final])
                )

                def ctx() -> BotContext:
                    return BotContext(
                        group_openid="G_demo", store=lock_store, index=index,
                        scope=SCOPE_GROUP,
                    )

                first = asyncio.create_task(
                    busy._run(agent=agent, thread_key="G:G_demo", instruction="一", context=ctx())
                )
                await asyncio.sleep(0)  # let the first task reach the agent
                second = asyncio.create_task(
                    busy._run(agent=agent, thread_key="G:G_demo", instruction="二", context=ctx())
                )
                await asyncio.sleep(0.01)
                check(
                    "同一 thread 的第二次调用在锁外等待，没有并发进入 agent",
                    agent.calls == 1 and agent.peak == 1,
                    f"calls={agent.calls} peak={agent.peak}",
                )
                check("在跑时 group_busy 为真", busy.group_busy("G_demo"))

                agent.gate.set()
                results = await asyncio.gather(first, second)
                check(
                    "放行后两次都完成，且始终没有并发",
                    agent.calls == 2 and agent.peak == 1 and len(results) == 2,
                    f"calls={agent.calls} peak={agent.peak}",
                )
                check("跑完后不再 busy", not busy.group_busy("G_demo"))
                check(
                    "不同 thread 各拿各的锁",
                    busy._lock_for("G:G_demo") is not busy._lock_for("U:alice"),
                )
                lock_store.close()

        asyncio.run(serialised())


def test_auto_summary() -> None:
    print("\n[12] 到量自动总结（阈值 / 冷却 / 静默 / 通知）")

    async def scenario() -> None:
        from src.bot.client import SummarizerClient

        original = botpy.Client._bot_login
        loop = asyncio.get_running_loop()

        async def fake_login(self, token):
            self._connection = botpy.connection.ConnectionSession(
                max_async=1,
                connect=self.bot_connect,
                dispatch=self.ws_dispatch,
                loop=loop,
                api=self.api,
            )

        botpy.Client._bot_login = fake_login
        auto = config.auto_summary
        saved = (
            auto.enabled, auto.min_messages, auto.cooldown_s, auto.notify, list(auto.groups)
        )
        try:
            auto.enabled, auto.min_messages, auto.cooldown_s, auto.notify, auto.groups = (
                True, 3, 1800, False, [],
            )

            def idle(n: int) -> dict:
                """A plain group message — nobody summoned the bot."""
                return {**TEXT_MESSAGE, "id": f"ROBOT1.0_idle{n}", "content": f"闲聊第 {n} 条"}

            with tempfile.TemporaryDirectory() as tmp:
                store = SQLStore(Path(tmp) / "auto.db")
                summarizer = FakeSummarizer()
                wakes: list[int] = []
                client = SummarizerClient(
                    store=store,
                    summarizer=summarizer,
                    wake_indexer=lambda: wakes.append(1),
                    bot_log=True,
                    ext_handlers=False,
                )
                client.api = FakeAPI()
                await client._bot_login(None)
                parser = client._connection.parser["group_message_create"]

                for i in range(2):
                    parser(frame(idle(i)))
                    await asyncio.sleep(0.05)
                check("不到阈值不触发", summarizer.calls == [], str(summarizer.calls))
                check("不到阈值不发消息", client.api.sent == [])
                check("不到阈值没有总结", store.summaries_for("G_demo") == [])

                parser(frame(idle(2)))
                await asyncio.sleep(0.05)
                check("达到阈值触发一次", len(summarizer.calls) == 1, str(summarizer.calls))
                check(
                    "自动总结用配置里的指令",
                    summarizer.calls[0] == ("G_demo", config.auto_summary.instruction),
                    str(summarizer.calls),
                )

                stored = store.summaries_for("G_demo")
                check("自动总结落库", len(stored) == 1, str(len(stored)))
                check("记为 trigger=auto", stored[0]["trigger"] == "auto", str(dict(stored[0])))
                check("自动总结没有触发者", stored[0]["requested_by"] is None)
                check("自动总结唤醒一次索引器", len(wakes) == 1, str(len(wakes)))
                check("静默模式不发群消息", client.api.sent == [])

                # Cooldown. Three more messages would clear the threshold again,
                # but the attempt above already spent the window — this is what
                # keeps a failing gateway from being retried on every message.
                for i in range(3, 6):
                    parser(frame(idle(i)))
                    await asyncio.sleep(0.05)
                check("冷却期内不再触发", len(summarizer.calls) == 1, str(summarizer.calls))
                check("冷却期内不重复落库", len(store.summaries_for("G_demo")) == 1)

                # Whitelist: cooldown cleared by hand (standing in for the 1800s
                # having passed), threshold satisfied, group simply not listed.
                auto.groups = ["G_other"]
                client._last_auto.clear()
                parser(frame(idle(9)))
                await asyncio.sleep(0.05)
                check("白名单之外的群不自动总结", len(summarizer.calls) == 1, str(summarizer.calls))
                check("白名单之外的群不发消息", client.api.sent == [])
                auto.groups = []

                auto.enabled = False
                parser(frame(idle(10)))
                await asyncio.sleep(0.05)
                check("开关关掉后不再自动总结", len(summarizer.calls) == 1, str(summarizer.calls))
                auto.enabled = True

                # An @ still works exactly as before, auto-summary or not.
                parser(frame(SUMMON_MESSAGE))
                await asyncio.sleep(0.05)
                check("自动总结不妨碍被 @ 时的总结", len(summarizer.calls) == 2)
                check("被 @ 的回复照常发出", len(client.api.sent) == 1)
                check(
                    "被 @ 的那篇记为 at",
                    [row["trigger"] for row in store.summaries_for("G_demo")] == ["at", "auto"],
                    str([dict(r)["trigger"] for r in store.summaries_for("G_demo")]),
                )
                store.close()

            # ---- notify=true also puts a copy in the group --------------------
            with tempfile.TemporaryDirectory() as tmp:
                store = SQLStore(Path(tmp) / "notify.db")
                client = SummarizerClient(
                    store=store,
                    summarizer=FakeSummarizer(),
                    wake_indexer=lambda: None,
                    bot_log=True,
                    ext_handlers=False,
                )
                client.api = FakeAPI()
                await client._bot_login(None)
                auto.notify = True
                auto.min_messages = 1
                client._connection.parser["group_message_create"](frame(TEXT_MESSAGE))
                await asyncio.sleep(0.05)

                check("notify=true 时把总结发到群里", len(client.api.sent) == 1, str(client.api.sent))
                check("不附 msg_id = 主动消息（吃配额）", client.api.sent[0]["msg_id"] is None)
                check(
                    "发到群里的那份带前缀",
                    client.api.sent[0]["content"].startswith("〔自动总结〕"),
                    client.api.sent[0]["content"][:20],
                )
                stored = store.summaries_for("G_demo")
                check("同一篇总结已落库", len(stored) == 1, str(len(stored)))
                check(
                    "入库的正文不带前缀",
                    stored[0]["content"] == FakeSummarizer.TEXT,
                    stored[0]["content"][:20],
                )
                store.close()

            # ---- a failed attempt still spends the cooldown -------------------
            with tempfile.TemporaryDirectory() as tmp:
                store = SQLStore(Path(tmp) / "boom.db")
                summarizer = ExplodingSummarizer()
                client = SummarizerClient(
                    store=store,
                    summarizer=summarizer,
                    wake_indexer=lambda: None,
                    bot_log=True,
                    ext_handlers=False,
                )
                client.api = FakeAPI()
                await client._bot_login(None)
                auto.notify = False
                auto.min_messages = 1
                parser = client._connection.parser["group_message_create"]

                parser(frame(TEXT_MESSAGE))
                await asyncio.sleep(0.05)
                check("尝试过一次", len(summarizer.calls) == 1, str(summarizer.calls))
                check("自动总结失败时不发消息", client.api.sent == [])
                check("失败的总结不入库", store.summaries_for("G_demo") == [])

                parser(frame({**TEXT_MESSAGE, "id": "ROBOT1.0_boom2"}))
                await asyncio.sleep(0.05)
                check(
                    "失败后的下一条消息不立刻重试（冷却已花掉）",
                    len(summarizer.calls) == 1,
                    str(summarizer.calls),
                )
                store.close()
        finally:
            botpy.Client._bot_login = original
            (
                auto.enabled, auto.min_messages, auto.cooldown_s, auto.notify, auto.groups
            ) = saved

    asyncio.run(scenario())


if __name__ == "__main__":
    test_text_message()
    test_mentions()
    test_image_message()
    test_quote_message()
    test_depth_guard()
    test_bad_timestamp()
    test_store()
    test_plan_chunks()
    test_reply_chunked()
    test_client_seam()
    test_rag()
    test_agent_boundary()
    test_auto_summary()
    print(f"\n全部通过：{PASSED} 项断言")
