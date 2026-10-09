"""发送侧：回复切段与 `reply_chunked`（被动回复 / 降级 / 失败路径 / 群与私聊差异）。"""

from __future__ import annotations

from conftest import FakeAPI
from src.bot.sender import KIND_C2C, plan_chunks, reply_chunked
from src.config import config


def test_plan_chunks() -> None:
    chunks, truncated = plan_chunks("一点短内容", 100, 5)
    assert chunks == ["一点短内容"] and not truncated, "短内容单段且不截断"

    text = "\n\n".join(f"段落{i}" + "字" * 20 for i in range(6))
    chunks, truncated = plan_chunks(text, 60, 5)
    assert len(chunks) >= 2 and all(len(c) <= 60 for c in chunks), "按段落打包"
    assert all("段落" in c for c in chunks), "段落边界未被拆开"
    assert not truncated or len(chunks) > 5, "未超限时不算截断"

    # 500 chars at limit 100 is exactly 5 chunks — it fits, so no truncation.
    chunks, truncated = plan_chunks("字" * 500, 100, 5)
    assert len(chunks) == 5 and not truncated, "恰好 5 段不算截断"

    chunks, truncated = plan_chunks("字" * 600, 100, 5)
    assert all(len(c) <= 100 for c in chunks), "超长单段落被硬切且每段不超限"
    assert len(chunks) == 5, "段数上限被遵守"
    assert truncated, "截断被标记"
    assert "省略" in chunks[-1], "末段是截断提示"


async def test_reply_chunked_passive_and_downgrade() -> None:
    api = FakeAPI()
    result = await reply_chunked(api, "G_demo", "MSG_1", "第一段\n\n第二段", elapsed_s=1.0)
    assert result.sent == 1 and result.ok, "发送成功两段"
    assert api.sent[0]["msg_id"] == "MSG_1", "携带 msg_id 走被动回复"
    assert api.sent[0]["msg_seq"] == 1, "msg_seq 从 1 开始"
    assert api.sent[0]["msg_type"] == 0, "msg_type 为 0（纯文本）"

    api = FakeAPI()
    result = await reply_chunked(api, "G_demo", "MSG_1", "正文", elapsed_s=999.0)
    assert result.active and api.sent[0]["msg_id"] is None, "超时后降级为主动消息"


async def test_reply_chunked_failure_paths() -> None:
    api = FakeAPI(return_none_on=1)
    result = await reply_chunked(api, "G_demo", "MSG_1", "正文")
    assert not result.ok and "None" in (result.error or ""), "None 返回被判为失败"

    api = FakeAPI(fail_on=1)
    result = await reply_chunked(api, "G_demo", "MSG_1", "正文")
    assert not result.ok and len(api.sent) == 1, "429 被捕获且不重试"

    api = FakeAPI()
    result = await reply_chunked(api, "G_demo", "MSG_1", "   ")
    assert result.sent == 0 and not api.sent, "空内容不发送"


async def test_reply_chunked_chunking_limits() -> None:
    limit = config.summary.max_reply_chars
    api = FakeAPI()
    result = await reply_chunked(api, "G_demo", "MSG_1", "字" * (limit * 8))
    assert len(api.sent) == 5, "超长摘要最多发 5 条"
    assert [m["msg_seq"] for m in api.sent] == [1, 2, 3, 4, 5], "msg_seq 严格递增"
    assert all(len(m["content"]) <= limit for m in api.sent), "每段都不超限"
    assert "省略" in api.sent[-1]["content"], "末段为截断提示"
    assert result.truncated and result.sent == 5, "截断被如实上报"


async def test_reply_chunked_c2c() -> None:
    # C2C shares this path but not the target keyword. This is the only
    # place the two routes actually differ, so it is worth pinning.
    api = FakeAPI()
    result = await reply_chunked(
        api, "U_alice", "C2C_MSG_1", "私聊回复", kind=KIND_C2C, elapsed_s=1.0
    )
    assert result.ok and api.sent[0]["kind"] == "c2c", "私聊走 post_c2c_message"
    assert api.sent[0]["openid"] == "U_alice", "私聊目标参数名是 openid"
    assert "group_openid" not in api.sent[0], "私聊不带 group_openid"
    assert api.sent[0]["msg_id"] == "C2C_MSG_1", "私聊同样带 msg_id 走被动回复"

    # The group-only "expired window → active message" downgrade must NOT
    # fire for C2C: that path consumes group quota, and C2C active messages
    # follow their own rules and may simply be rejected.
    api = FakeAPI()
    result = await reply_chunked(
        api, "U_alice", "C2C_MSG_1", "私聊回复", kind=KIND_C2C, elapsed_s=999.0
    )
    assert not result.active and not api.sent, "私聊超窗不降级为主动消息"
    assert not result.ok and "expired" in (result.error or ""), "私聊超窗如实记为失败"
