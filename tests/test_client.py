"""`SummarizerClient`：GROUP_MESSAGE_CREATE 接缝（补 botpy 缺失的解析器）与
按消息量触发的自动总结。全程不连 QQ —— 登录用 `fake_bot_login` 顶替，
发送用 `FakeAPI`，模型用 `FakeSummarizer`。
"""

from __future__ import annotations

import asyncio

import botpy
import pytest

from conftest import (
    FakeAPI,
    FakeSummarizer,
    ExplodingSummarizer,
    TEXT_MESSAGE,
    SUMMON_MESSAGE,
    frame,
)
from src.agent.summarizer import SummaryResult
from src.bot.client import SummarizerClient
from src.config import config


def test_intents_public_messages() -> None:
    assert int(botpy.Intents(public_messages=True).value) == 1 << 25, (
        "intent 含 public_messages (1<<25)"
    )


async def _start(store, summarizer, wakes=None):
    """A client with the fake login patched in; returns (client, parser)."""
    client = SummarizerClient(
        store=store,
        summarizer=summarizer,
        wake_indexer=(lambda: wakes.append(1)) if wakes is not None else None,
        bot_log=True,
        ext_handlers=False,
    )
    client.api = FakeAPI()
    await client._bot_login(None)
    return client, client._connection.parser["group_message_create"]


async def test_parser_registration_and_dispatch(store, fake_bot_login):
    client = SummarizerClient(
        store=store,
        summarizer=FakeSummarizer(),
        wake_indexer=lambda: None,
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
        loop=asyncio.get_running_loop(),
        api=client.api,
    )
    assert "group_message_create" not in probe.parser, "botpy 原生 parser 里没有该事件"

    await client._bot_login(None)
    parser = client._connection.parser["group_message_create"]
    assert "group_message_create" in client._connection.parser, "注册后 parser 里有该事件"
    # connection.py:40 makes these the same dict object, which is why
    # mutating one is enough for gateway.py to find the parser.
    assert client._connection.parser is client._connection.state.parsers, (
        "parser 与 state.parsers 是同一个 dict"
    )

    # The parser runs synchronously inside the ws read loop and is
    # NOT wrapped in a try by botpy (gateway.py:99), so malformed
    # payloads must be swallowed rather than kill the connection.
    # Any raise here propagates and fails the test — which is the point.
    for bad in ({"t": "GROUP_MESSAGE_CREATE"}, {"d": "not-a-dict"}, None):
        parser(bad)
    assert True, "畸形 payload 不抛异常"


async def test_at_message_end_to_end(store, fake_bot_login):
    summarizer = FakeSummarizer(store)
    wakes: list[int] = []
    client, parser = await _start(store, summarizer, wakes)

    parser(frame(SUMMON_MESSAGE))
    await asyncio.sleep(0.05)
    assert len(summarizer.calls) == 1, "被 @ 的消息触发了摘要"
    assert summarizer.calls[0] == ("G_demo", "总结一下今天群里聊了什么"), (
        "指令取自被剥掉 @ 后的正文"
    )
    assert len(client.api.sent) == 1, "摘要被回发"
    assert client.api.sent[0]["msg_id"] == "ROBOT1.0_summon", (
        "回复是带 msg_id 的被动回复"
    )
    assert len(store.recent_messages("G_demo", 10)) == 1, "消息已落库"

    # Same message again — QQ can push one @-message as both events.
    parser(frame(SUMMON_MESSAGE))
    await asyncio.sleep(0.05)
    assert len(summarizer.calls) == 1, "重复推送不再触发第二次摘要"
    assert len(client.api.sent) == 1, "重复推送不再回发"

    parser(frame(TEXT_MESSAGE))
    await asyncio.sleep(0.05)
    assert len(client.api.sent) == 1, "未 @ 的消息只落库不回复"
    assert len(store.recent_messages("G_demo", 10)) == 2, "未 @ 的消息也落库了"

    # The answer became a document because the agent published it — mid-run,
    # which is *earlier* than the old "store after the run", so a delivery
    # failure still cannot lose it. `FakeSummarizer` writes the row through the
    # same `insert_document` the tool uses; see its docstring.
    stored = store.summaries_for("G_demo")
    assert len(stored) == 1, "agent 投稿的总结落库"
    assert (
        stored[0]["group_openid"] == "G_demo" and stored[0]["message_count"] == 5
    ), "总结记录了群与取数条数"
    assert stored[0]["requested_by"] == "M_ming", "总结记下了触发者"
    assert stored[0]["instruction"] == "总结一下今天群里聊了什么", "总结保留原始指令"
    assert stored[0]["indexed_at"] is None, "总结尚待索引"

    # Waking the indexer moved from "every message" to "every
    # stored summary": nothing per-message is embedded any more, and
    # the insert that does need indexing is the summary itself.
    assert len(wakes) == 1, "三条消息本身不唤醒索引器"
    assert len(wakes) == 1, "总结入库时唤醒一次索引器"

    # Provenance travels to the publishing tool, not to the model: the row's
    # `requested_by`/`trigger` can only be right if the client handed them over.
    assert summarizer.kwargs[0]["requested_by"] == "M_ming", "触发者作为 provenance 传下去"
    assert summarizer.kwargs[0]["trigger"] == "at", "入口记为 at"


async def test_at_answer_that_declines_to_publish_is_not_stored(store, fake_bot_login):
    """The point of the whole change: a @ that reads messages but is *a question*
    must not become a document. Storage is no longer the bot's unconditional act,
    so `publish=False` (the model declined) has to leave the corpus untouched
    while the group still gets its reply."""
    summarizer = FakeSummarizer(store, publish=False)
    wakes: list[int] = []
    client, parser = await _start(store, summarizer, wakes)

    parser(frame(SUMMON_MESSAGE))
    await asyncio.sleep(0.05)

    assert len(summarizer.calls) == 1, "照样跑了一轮"
    assert len(client.api.sent) == 1, "答复仍然发回群里"
    assert store.summaries_for("G_demo") == [], "没投稿就不入库"
    assert wakes == [], "没入库就不唤醒索引器"


class _Author:
    def __init__(self, openid: str) -> None:
        self.user_openid = openid


class FakeC2C:
    def __init__(self, openid: str = "U_alice") -> None:
        self.id = "C2C_1"
        self.content = "之前有人提过部署方案吗"
        self.author = _Author(openid)


async def test_c2c_flow(store, fake_bot_login):
    summarizer = FakeSummarizer(store)
    wakes: list[int] = []
    client, _ = await _start(store, summarizer, wakes)

    await client.on_c2c_message_create(FakeC2C())
    assert summarizer.private_calls == [("U_alice", "之前有人提过部署方案吗")], (
        "私聊触发了跨群回答"
    )
    assert len(client.api.sent) == 1 and client.api.sent[0]["kind"] == "c2c", (
        "私聊回复走 post_c2c_message"
    )
    assert client.api.sent[0]["openid"] == "U_alice", "私聊回复发送给 user_openid"
    # Private messages carry `user_openid`, not `group_openid`; the
    # group table's column is NOT NULL and a C2C chat is not a group.
    assert (
        store.recent_messages("U_alice", 10) == []
    ), "私聊消息不写入 group_messages"
    assert len(store.summaries_for(None)) == 0, "私聊回答不产生总结"
    assert len(wakes) == 0, "私聊不唤醒索引器"


async def test_c2c_allowlist_gate(store, fake_bot_login, monkeypatch):
    summarizer = FakeSummarizer()
    client, _ = await _start(store, summarizer)

    # Empty allowlist means everyone; a non-empty one is a gate.
    monkeypatch.setattr(config.c2c, "allowlist", ["U_allowed"])
    gated = FakeAPI()
    client.api = gated
    await client.on_c2c_message_create(FakeC2C("U_blocked"))
    assert not gated.sent, "allowlist 之外的用户被拒"
    assert len(summarizer.private_calls) == 0, "被拒的私聊不消耗模型调用"

    api = FakeAPI()
    client.api = api
    await client.on_c2c_message_create(FakeC2C("U_allowed"))
    assert len(api.sent) == 1, "allowlist 内的用户被放行"
    assert len(summarizer.private_calls) == 1, "放行后才产生调用"


async def test_send_failure_still_stores(store, fake_bot_login):
    # A send that fails (here: botpy's silent `None`) must not cost the
    # summary — a 429 or a timeout would otherwise lose it for good.
    client = SummarizerClient(
        store=store,
        summarizer=FakeSummarizer(store),
        bot_log=True,
        ext_handlers=False,
    )
    client.api = FakeAPI(return_none_on=1)
    await client._bot_login(None)
    client._connection.parser["group_message_create"](frame(SUMMON_MESSAGE))
    await asyncio.sleep(0.05)
    assert len(store.summaries_for("G_demo")) == 1, "发送失败时总结仍然落库"


async def test_agent_error_falls_back(store, fake_bot_login):
    # A failing agent must still answer the group, not go silent.
    client, parser = await _start(store, ExplodingSummarizer())
    parser(frame(SUMMON_MESSAGE))
    await asyncio.sleep(0.05)
    assert len(client.api.sent) == 1, "agent 抛异常时仍回一条兜底消息"
    assert "出错" in client.api.sent[0]["content"], "兜底文案可读"
    assert store.summaries_for("G_demo") == [], "失败的调用不产生总结"


# ---- auto summary (message-count trigger) -----------------------------------


def _idle(n: int) -> dict:
    """A plain group message — nobody summoned the bot."""
    return {**TEXT_MESSAGE, "id": f"ROBOT1.0_idle{n}", "content": f"闲聊第 {n} 条"}


@pytest.fixture()
def auto_on(monkeypatch):
    """Enable the auto-summary trigger with a small threshold."""
    auto = config.auto_summary
    for name, value in {
        "enabled": True,
        "min_messages": 3,
        "cooldown_s": 1800,
        "notify": False,
        "groups": [],
    }.items():
        monkeypatch.setattr(auto, name, value)
    return auto


async def test_auto_summary_threshold_and_cooldown(store, fake_bot_login, auto_on):
    summarizer = FakeSummarizer(store)
    wakes: list[int] = []
    client, parser = await _start(store, summarizer, wakes)

    for i in range(2):
        parser(frame(_idle(i)))
        await asyncio.sleep(0.05)
    assert summarizer.calls == [], "不到阈值不触发"
    assert client.api.sent == [], "不到阈值不发消息"
    assert store.summaries_for("G_demo") == [], "不到阈值没有总结"

    parser(frame(_idle(2)))
    await asyncio.sleep(0.05)
    assert len(summarizer.calls) == 1, "达到阈值触发一次"
    assert summarizer.calls[0] == ("G_demo", config.auto_summary.instruction), (
        "自动总结用配置里的指令"
    )

    stored = store.summaries_for("G_demo")
    assert len(stored) == 1, "自动总结落库"
    assert stored[0]["trigger"] == "auto", "记为 trigger=auto"
    assert stored[0]["requested_by"] is None, "自动总结没有触发者"
    assert len(wakes) == 1, "自动总结唤醒一次索引器"
    assert client.api.sent == [], "静默模式不发群消息"

    # Cooldown. Three more messages would clear the threshold again,
    # but the attempt above already spent the window — this is what
    # keeps a failing gateway from being retried on every message.
    for i in range(3, 6):
        parser(frame(_idle(i)))
        await asyncio.sleep(0.05)
    assert len(summarizer.calls) == 1, "冷却期内不再触发"
    assert len(store.summaries_for("G_demo")) == 1, "冷却期内不重复落库"

    # Whitelist: cooldown cleared by hand (standing in for the 1800s
    # having passed), threshold satisfied, group simply not listed.
    auto_on.groups = ["G_other"]
    client._last_auto.clear()
    parser(frame(_idle(9)))
    await asyncio.sleep(0.05)
    assert len(summarizer.calls) == 1, "白名单之外的群不自动总结"
    assert client.api.sent == [], "白名单之外的群不发消息"
    auto_on.groups = []

    auto_on.enabled = False
    parser(frame(_idle(10)))
    await asyncio.sleep(0.05)
    assert len(summarizer.calls) == 1, "开关关掉后不再自动总结"
    auto_on.enabled = True

    # An @ still works exactly as before, auto-summary or not.
    parser(frame(SUMMON_MESSAGE))
    await asyncio.sleep(0.05)
    assert len(summarizer.calls) == 2, "自动总结不妨碍被 @ 时的总结"
    assert len(client.api.sent) == 1, "被 @ 的回复照常发出"
    assert [row["trigger"] for row in store.summaries_for("G_demo")] == ["at", "auto"], (
        "被 @ 的那篇记为 at"
    )


async def test_auto_summary_falls_back_when_nothing_published(
    store, fake_bot_login, auto_on
):
    """The auto path stores **even if the model published nothing**.

    Its gate is the message count, and that count is `messages since the last
    stored row` — so a thin-but-successful summary that went unwritten would
    leave the count unreset and re-summarise the same batch after every cooldown,
    forever. This is the reason the auto fallback keeps the *low* code-side floor
    instead of the publishing tool's higher one.
    """
    auto_on.min_messages = 1
    summarizer = FakeSummarizer(store, publish=False)
    wakes: list[int] = []
    client, parser = await _start(store, summarizer, wakes)

    parser(frame(TEXT_MESSAGE))
    await asyncio.sleep(0.05)

    stored = store.summaries_for("G_demo")
    assert len(stored) == 1, "模型没投稿时，自动路径仍然代写一行"
    assert stored[0]["trigger"] == "auto", "代写的行也记 trigger=auto"
    assert stored[0]["content"] == FakeSummarizer.TEXT, "代写用的是答复正文"
    assert len(wakes) == 1, "代写同样唤醒索引器"


async def test_auto_summary_publishing_model_stores_exactly_one_row(
    store, fake_bot_login, auto_on
):
    """One auto run must never leave two rows.

    The model has every reason to publish under the auto instruction ("请总结本
    群最近的讨论"), and the code-side fallback would otherwise add a second row
    for the same coverage right beside it.
    """
    auto_on.min_messages = 1
    client, parser = await _start(store, FakeSummarizer(store, publish=True), wakes=[])

    parser(frame(TEXT_MESSAGE))
    await asyncio.sleep(0.05)
    assert len(store.summaries_for("G_demo")) == 1, "投稿与代写没有互踩"


async def test_auto_summary_notify_pushes_active_message(store, fake_bot_login, auto_on):
    auto_on.notify = True
    auto_on.min_messages = 1
    client, parser = await _start(store, FakeSummarizer(store), wakes=[])
    parser(frame(TEXT_MESSAGE))
    await asyncio.sleep(0.05)

    assert len(client.api.sent) == 1, "notify=true 时把总结发到群里"
    assert client.api.sent[0]["msg_id"] is None, "不附 msg_id = 主动消息（吃配额）"
    assert client.api.sent[0]["content"].startswith("〔自动总结〕"), (
        "发到群里的那份带前缀"
    )
    stored = store.summaries_for("G_demo")
    assert len(stored) == 1, "同一篇总结已落库"
    assert stored[0]["content"] == FakeSummarizer.TEXT, "入库的正文不带前缀"


class _PublishOnlySummarizer:
    """Answers with a short remark while the *document* went in through the tool.

    The two texts differ on purpose: this is what the chosen reply shape
    (群里只发要点，全文进库) produces, and it is the only way to see which of the
    two the notify branch pushes.
    """

    DOC = "全文：两小时的部署讨论，结论是周五上线，卡点是测试环境。\n" * 3
    REMARK = "已收录，覆盖 20:00~22:00 共 18 条。要点：周五上线。"

    def __init__(self, store):
        self.store = store

    async def summarize_group(self, group_openid, instruction, **kw):
        from src.agent.publish import insert_document
        from src.agent.tools import CoverageLog

        coverage = CoverageLog()
        coverage.add("recent_messages", 18, "2026-10-10T20:00:00+08:00",
                     "2026-10-10T22:00:00+08:00")
        sid = insert_document(self.store, group_openid=group_openid,
                              instruction=instruction, content=self.DOC,
                              coverage=coverage.intervals(),
                              message_count=coverage.message_count(),
                              requested_by=kw.get("requested_by"),
                              trigger=kw.get("trigger", "at"))
        return SummaryResult(text=self.REMARK, coverage=coverage,
                             published_ids=[sid], published_text=self.DOC)

    async def answer_private(self, user_openid, instruction):
        raise AssertionError("不该走到私聊")

    def group_busy(self, group_openid):
        return False


async def test_notify_pushes_the_document_not_the_remark(store, fake_bot_login, auto_on):
    """自动总结推的是**库里那篇全文**，不是模型的收尾一句。

    群里没人等答复，推"已收录，覆盖…"等于推了一条关于文件的通告。`@` 路径反过来
    推要点才是对的，因为那里有人在等。
    """
    auto_on.notify = True
    auto_on.min_messages = 1
    fake = _PublishOnlySummarizer(store)
    client, parser = await _start(store, fake, wakes=[])

    parser(frame(TEXT_MESSAGE))
    await asyncio.sleep(0.05)

    assert len(client.api.sent) == 1, "notify=true 时发一条主动消息"
    sent = client.api.sent[0]["content"]
    assert sent.startswith("〔自动总结〕") and "两小时的部署讨论" in sent, (
        "推送的是全文正文"
    )
    assert "已收录" not in sent, "不是模型那句要点"
    assert store.summaries_for("G_demo")[0]["content"] == fake.DOC, "入库的是同一份全文"


async def test_at_path_replies_the_remark_not_the_document(store, fake_bot_login):
    """同一条改造的另一半：被 @ 时群里收到的就是要点，正文只在库里。"""
    fake = _PublishOnlySummarizer(store)
    client, parser = await _start(store, fake, wakes=[])

    parser(frame(SUMMON_MESSAGE))
    await asyncio.sleep(0.05)

    assert client.api.sent[0]["content"] == fake.REMARK, "回群的是要点，不重抄正文"
    assert store.summaries_for("G_demo")[0]["content"] == fake.DOC, "正文进了库"


async def test_auto_summary_failure_spends_cooldown(store, fake_bot_login, auto_on):
    auto_on.min_messages = 1
    auto_on.notify = False
    summarizer = ExplodingSummarizer()
    client, parser = await _start(store, summarizer, wakes=[])

    parser(frame(TEXT_MESSAGE))
    await asyncio.sleep(0.05)
    assert len(summarizer.calls) == 1, "尝试过一次"
    assert client.api.sent == [], "自动总结失败时不发消息"
    assert store.summaries_for("G_demo") == [], "失败的总结不入库"

    parser(frame({**TEXT_MESSAGE, "id": "ROBOT1.0_boom2"}))
    await asyncio.sleep(0.05)
    assert len(summarizer.calls) == 1, "失败后的下一条消息不立刻重试（冷却已花掉）"
