"""两个 agent 的工具边界与隐私隔离（假模型 + 真 agent 循环 + 真注入路径）。

The two tool sets exist because the privacy boundary points in opposite
directions: a group must never reach across groups, private chat deliberately
spans them. Tests here pin both the static schemas (what the model can see) and
the runtime behaviour (forged args must be stripped).
"""

from __future__ import annotations

import warnings
from typing import get_origin

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from conftest import FakeEmbeddings, TEXT_MESSAGE, frame
from src.agent.summarizer import (
    MAX_THREADS,
    NO_ANSWER,
    Summarizer,
    SummaryResult,
    store_summary,
)
from src.agent.tools import (
    C2C_TOOLS,
    DATA_TOOLS,
    GROUP_TOOLS,
    SCOPE_ALL,
    SCOPE_GROUP,
    BotContext,
    CoverageLog,
    list_groups,
    messages_across_groups,
    messages_in_range,
    recent_messages,
    search_summaries,
)
from src.bot.events import GroupMessageRecord
from src.config import config
from src.rag.retriever import SummaryIndex, group_label

# The third group is a realistic 32-char openid and deliberately has **no
# summary**: that is the state a feedback group sits in until it either
# gets @-ed or crosses the auto-summary threshold, and private chat still
# has to be able to name it.
REAL_OPENID = "A1B2C3D4E5F60718293A4B5C6D7E8F90"

ROWS_BY_GROUP = {
    "G_demo": [("小明", "本群机密：项目代号是蓝色鲸鱼"), ("小红", "收到，我去准备会议材料")],
    "G_other": [("外人", "别群的机密：项目代号是红色狐狸")],
    REAL_OPENID: [("路人", "没有总结的群里的原话")],
}

SUMMARIES = [
    ("G_demo", "本群机密：项目代号是蓝色鲸鱼，小红在准备会议材料。", 20,
     "2026-07-21T00:00:00+08:00", "2026-07-21T01:00:00+08:00"),
    ("G_demo", "本群昨天的结论：部署方案定在周五上线。", 8,
     "2026-07-20T00:00:00+08:00", "2026-07-20T01:00:00+08:00"),
    ("G_other", "别群的机密：项目代号是红色狐狸，本周不讨论上线。", 5,
     "2026-07-21T02:00:00+08:00", "2026-07-21T03:00:00+08:00"),
]

WEEK = ("2026-07-21T00:00:00+08:00", "2026-07-22T00:00:00+08:00")

FINAL_TEXT = "总结：项目代号是蓝色鲸鱼，小红在准备会议材料，部署方案定在周五上线。"


def _seed_messages(store) -> None:
    n = 0
    for gid, items in ROWS_BY_GROUP.items():
        for who, text in items:
            rec = GroupMessageRecord.from_payload(
                frame(
                    {
                        **TEXT_MESSAGE,
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


def _seed_summaries(store) -> None:
    for gid, content, count, start, end in SUMMARIES:
        store.insert_summary(
            group_openid=gid,
            instruction="总结一下",
            content=content,
            coverage=[(start, end)],
            message_count=count,
        )


def _make_index(tmp_path, name: str = "agent_summaries") -> SummaryIndex:
    return SummaryIndex(
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path / "chroma",
        collection_name=name,
    )


def _seeded(store, tmp_path):
    """store + seeded messages + seeded summaries + index (summaries embedded)."""
    _seed_messages(store)
    _seed_summaries(store)
    index = _make_index(tmp_path)
    rows = [dict(r) for r in store.unindexed_summaries(limit=10)]
    assert len(rows) == 3, "三个群的总结都已入库"
    assert index.add(rows) == 3, "总结被索引"
    return index


class _RT:
    """Stand-in runtime for calling tool coroutines directly (bypasses ToolNode)."""

    def __init__(self, gid, scope=SCOPE_GROUP, *, store=None, index=None):
        self.context = BotContext(
            group_openid=gid, store=store, index=index, scope=scope
        )


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Replays canned messages; ignores `bind_tools`."""

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


def _forged_call() -> AIMessage:
    # The tool call deliberately smuggles a forged `runtime` *and* a forged
    # `scope`/`group_openid`. ToolNode strips LLM-supplied values for injected
    # args (tool_node.py `stripped_args`), so the group answer must stay put.
    # This is the only test that proves the cross-group filter is unforgeable.
    return AIMessage(
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


def _final() -> AIMessage:
    return AIMessage(content=FINAL_TEXT)


# ---- static: tool sets and schemas ------------------------------------------


def test_tool_sets_and_privacy_direction() -> None:
    group_by_name = {t.name: t for t in GROUP_TOOLS}
    c2c_by_name = {t.name: t for t in C2C_TOOLS}
    assert set(group_by_name) == {
        "current_time", "recent_messages", "messages_in_range", "search_summaries",
    }, "群内工具集是那 4 个"
    assert set(c2c_by_name) == {
        "current_time", "search_summaries", "list_groups", "messages_across_groups",
    }, "私聊工具集是那 4 个"
    # The two sets exist precisely because these two lines must both hold.
    assert "list_groups" not in group_by_name, "群内工具集没有跨群能力"
    assert "messages_across_groups" not in group_by_name, "群内工具集读不到别群的原文"
    assert (
        "recent_messages" not in c2c_by_name and "messages_in_range" not in c2c_by_name
    ), "私聊工具集没有按群读本群原文的工具"
    # `DATA_TOOLS` drives the "tools ran but coverage is empty" warning in
    # `Summarizer._run`; a data-reading tool missing from it makes that warning
    # fire on every honest answer.
    assert {
        "recent_messages", "messages_in_range", "search_summaries", "messages_across_groups"
    } == set(DATA_TOOLS), "所有取数工具都登记在 DATA_TOOLS 里"
    assert "current_time" not in DATA_TOOLS, "current_time 不算取数工具"


def test_injected_args_invisible_to_model() -> None:
    # `tool_call_schema` is what `bind_tools` hands the model; injected args are
    # filtered out of it but stay in `get_input_schema()`, which is where the
    # injection machinery looks. The model must never see any of them either way.
    by_name = {t.name: t for t in (*GROUP_TOOLS, *C2C_TOOLS)}
    for name, tool in by_name.items():
        visible = set(tool.tool_call_schema.model_fields)
        internal = set(tool.get_input_schema().model_fields)
        assert "runtime" not in visible, f"{name}: 模型看不到 runtime"
        assert "runtime" in internal, f"{name}: runtime 仍参与注入"
        for hidden in ("group_openid", "scope"):
            assert hidden not in visible and hidden not in internal, (
                f"{name}: 模型看不到 {hidden}"
            )


def test_runtime_annotation_is_parameterized() -> None:
    # `runtime` must be annotated `ToolRuntime[BotContext, dict]`, never bare.
    # `ToolRuntime` is a generic dataclass over `(ContextT, StateT)` and both type
    # vars carry defaults (`tool_node.py:105-106`: `ContextT` defaults to `None`).
    # Bare, pydantic substitutes those defaults while building *this* args schema,
    # so `runtime.context` is typed `None`; handing it a real `BotContext` then
    # makes `BaseTool._parse_input`'s `result_v2.model_dump()` emit a serializer
    # warning on every tool call. Validation is lenient enough that the tool still
    # runs — which is what makes it worth a test: the only symptom is stderr noise
    # that looks like a defect. Asserted statically so a bare annotation cannot
    # come back quietly.
    by_name = {t.name: t for t in (*GROUP_TOOLS, *C2C_TOOLS)}
    for name, tool in by_name.items():
        injected = tool.get_input_schema().model_fields["runtime"].annotation
        assert get_origin(injected) is not None, f"{name}: runtime 带类型参数"


# ---- tool bodies (direct coroutine calls with stand-in runtime) -------------


async def test_tool_bodies_respect_scopes(store, tmp_path) -> None:
    index = _seeded(store, tmp_path)
    assert len(store.recent_messages("G_demo", 10)) == 2, "两个群的消息都已入库"

    rt = _RT("G_demo", store=store, index=index)
    text = await recent_messages.coroutine(limit=10, runtime=rt)
    assert "蓝色鲸鱼" in text, "工具按注入的群取数"
    assert "红色狐狸" not in text, "工具不会带回别群内容"
    assert (
        not rt.context.coverage.is_empty() and rt.context.coverage.message_count() == 2
    ), "取数被记入覆盖范围"

    rt = _RT("G_demo", store=store, index=index)
    ranged = await messages_in_range.coroutine(
        start_iso="2026-07-21T00:00:00+08:00",
        end_iso="2026-07-22T00:00:00+08:00",
        limit=50,
        runtime=rt,
    )
    assert "蓝色鲸鱼" in ranged and "红色狐狸" not in ranged, "时间范围工具正常"
    assert rt.context.coverage.intervals() != [], "时间范围被记成可入库的区间"

    # Bounds in a different offset must still be interpreted correctly.
    utc = await messages_in_range.coroutine(
        start_iso="2026-07-20T16:00:00Z",
        end_iso="2026-07-21T16:00:00Z",
        limit=50,
        runtime=_RT("G_demo", store=store, index=index),
    )
    assert "蓝色鲸鱼" in utc, "带 Z 的边界能正确比对（时区归一）"

    bad = await messages_in_range.coroutine(
        start_iso="昨天", end_iso="今天", limit=5,
        runtime=_RT("G_demo", store=store, index=index),
    )
    assert "无法解析" in bad, "非法时间返回可读错误而不是抛异常"

    # Same tool, opposite scopes. This pair is the whole point of having
    # two agents: one never leaves its group, the other always spans them.
    inside = _RT("G_demo", store=store, index=index)
    found = await search_summaries.coroutine(query="项目代号是什么", k=5, runtime=inside)
    assert "蓝色鲸鱼" in found, "群内语义检索有结果"
    assert "红色狐狸" not in found, "群内语义检索看不到别群的总结"
    assert found.startswith("[群"), "检索结果自带群标签"
    assert (
        inside.context.coverage.message_count() == 0 and not inside.context.coverage.is_empty()
    ), "只检索总结时不计入消息条数"

    across = await search_summaries.coroutine(
        query="项目代号是什么", k=5,
        runtime=_RT(None, SCOPE_ALL, store=store, index=index),
    )
    assert "蓝色鲸鱼" in across and "红色狐狸" in across, "私聊 scope 能跨群召回"

    groups = await list_groups.coroutine(
        runtime=_RT(None, SCOPE_ALL, store=store, index=index)
    )
    assert (
        group_label("G_demo") in groups and group_label("G_other") in groups
    ), "私聊能列出有总结的群"


async def test_messages_across_groups(store, tmp_path, monkeypatch) -> None:
    index = _seeded(store, tmp_path)

    def rt_all():
        return _RT(None, SCOPE_ALL, store=store, index=index)

    wide_rt = rt_all()
    wide = await messages_across_groups.coroutine(
        start_iso=WEEK[0], end_iso=WEEK[1], runtime=wide_rt
    )
    assert "蓝色鲸鱼" in wide and "红色狐狸" in wide, "跨群取原文命中多个群"
    assert group_label("G_demo") in wide and group_label(REAL_OPENID) in wide, (
        "每一行都带群标签，来源说得清"
    )
    assert wide_rt.context.coverage.message_count() == 4, "跨群取数被记入覆盖范围"

    named = rt_all()
    only_demo = await messages_across_groups.coroutine(
        start_iso=WEEK[0], end_iso=WEEK[1], group="G_demo", runtime=named
    )
    assert "蓝色鲸鱼" in only_demo and "红色狐狸" not in only_demo, (
        "group= 精确 openid 收敛到该群"
    )
    assert named.context.coverage.message_count() == 2, "收敛后的取数同样入账"

    # The model only ever sees labels, so both spellings have to resolve:
    # a `[groups]` alias, and the `群<尾号>` stub for an un-aliased group.
    monkeypatch.setattr(config, "groups", {"G_demo": "项目组"})
    aliased = await messages_across_groups.coroutine(
        start_iso=WEEK[0], end_iso=WEEK[1], group="项目组", runtime=rt_all()
    )
    assert "蓝色鲸鱼" in aliased and "红色狐狸" not in aliased, "别名能解析成群"

    stub = await messages_across_groups.coroutine(
        start_iso=WEEK[0], end_iso=WEEK[1], group=group_label(REAL_OPENID),
        runtime=rt_all(),
    )
    assert "没有总结的群里的原话" in stub and "蓝色鲸鱼" not in stub, (
        f"群<尾号>({group_label(REAL_OPENID)}) 能解析成群"
    )
    monkeypatch.undo()

    unknown = await messages_across_groups.coroutine(
        start_iso=WEEK[0], end_iso=WEEK[1], group="查无此群", runtime=rt_all()
    )
    assert "list_groups" in unknown, "群名解析不了时提示去调 list_groups"

    keyworded = await messages_across_groups.coroutine(
        start_iso=WEEK[0], end_iso=WEEK[1], keyword="狐狸", runtime=rt_all()
    )
    assert "红色狐狸" in keyworded and "蓝色鲸鱼" not in keyworded, (
        "keyword= 只留正文含该词的消息"
    )

    monkeypatch.setattr(config.c2c, "raw_limit", 1)
    clamped = await messages_across_groups.coroutine(
        start_iso=WEEK[0], end_iso=WEEK[1], limit=999, runtime=rt_all()
    )
    assert len([ln for ln in clamped.splitlines() if ln.strip()]) == 1, (
        "limit 被 c2c.raw_limit 夹住"
    )
    monkeypatch.undo()

    naive = await messages_across_groups.coroutine(
        start_iso="上周", end_iso="今天", runtime=rt_all()
    )
    assert "无法解析" in naive, "时间无法解析时返回提示而不是抛异常"

    # The time range is mandatory precisely so the tool cannot be used as
    # an unbounded full-corpus dump. That is a schema property, not a
    # runtime check, so it is asserted on the schema the model sees.
    visible = messages_across_groups.tool_call_schema.model_fields
    assert all(
        visible[f].is_required() for f in ("start_iso", "end_iso")
    ), "取原文必须给时间范围（模型不能省略）"
    assert not visible["group"].is_required(), "group/keyword/limit 都可选"


# ---- real agent runs (ToolNode injection, end to end) ------------------------


async def test_forged_runtime_stripped_and_no_pydantic_noise(store, tmp_path) -> None:
    index = _seeded(store, tmp_path)
    model = ToolCallingFakeModel(responses=[_forged_call(), _final()])
    summarizer = Summarizer(store, index, model=model)
    # This goes through ToolNode's real injection (the `_RT` stand-in above
    # bypasses `_parse_input` entirely), so it is the only place the pydantic
    # serializer path is exercised end to end.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await summarizer.summarize_group("G_demo", "总结一下最近的消息")
    noise = [
        str(w.message).splitlines()[0] for w in caught if "Pydantic serializer" in str(w.message)
    ]
    assert not noise, "真实注入路径不再触发 pydantic 序列化告警"
    assert "蓝色鲸鱼" in result.text, "返回最终回答"
    # Coverage is carried out of the graph by mutating the context object,
    # which only works because LangGraph passes an instance by reference.
    assert not result.coverage.is_empty(), "coverage 被带出 graph"
    assert result.storable(), "这次回答构成可入库文档"


async def test_real_toolnode_keeps_search_scoped(store, tmp_path) -> None:
    index = _seeded(store, tmp_path)
    summarizer = Summarizer(
        store, index, model=ToolCallingFakeModel(responses=[_forged_call(), _final()])
    )
    ctx = BotContext(group_openid="G_demo", store=store, index=index, scope=SCOPE_GROUP)
    state = await summarizer._group_agent.ainvoke(
        {"messages": [{"role": "user", "content": "总结一下"}]},
        config={"configurable": {"thread_id": "probe"}},
        context=ctx,
    )
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1, "工具确实被调用"
    assert "蓝色鲸鱼" in tool_messages[0].content, "伪造的 runtime/scope 被忽略，检索仍限定在本群"
    assert "红色狐狸" not in tool_messages[0].content, "伪造的 runtime/scope 被忽略（别群不泄漏）"
    assert not ctx.coverage.is_empty(), "coverage 随 context 对象传出"


async def test_thread_namespace_lru_and_locks(store, tmp_path) -> None:
    index = _seeded(store, tmp_path)
    final = _final()

    # Thread ids are the only namespace separating the two graphs' state, so
    # a group and a user that share a suffix must not share a thread.
    namespaced = Summarizer(store, index, model=ToolCallingFakeModel(responses=[final]))
    await namespaced.summarize_group("X1", "总结一下")
    await namespaced.answer_private("X1", "之前聊过部署吗")
    assert set(namespaced._threads) == {"G:X1", "U:X1"}, "同后缀的群与私聊是不同会话"

    # Memory is bounded, and private chats share the budget with groups.
    capped = Summarizer(store, index, model=ToolCallingFakeModel(responses=[final]))
    for i in range(MAX_THREADS + 3):
        capped._touch_thread(f"G:{i}")
    assert len(capped._threads) == MAX_THREADS, "会话记忆被限制在上限内"
    assert "G:0" not in capped._threads, "淘汰的是最早的会话"
    assert f"G:{MAX_THREADS + 2}" in capped._threads, "保留的是最近的会话"

    assert capped._lock_for("G:G_demo") is not capped._lock_for("U:alice"), (
        "不同 thread 各拿各的锁"
    )


def test_storable_rules_and_rejects(store) -> None:
    # "Is this answer a document?" — all three conditions are required.
    grounded = CoverageLog()
    grounded.add(
        "recent_messages", 5, "2026-07-21T00:00:00+08:00", "2026-07-21T01:00:00+08:00"
    )
    text = _final().content
    assert SummaryResult(text=text, coverage=grounded).storable(), "有取数、够长的回答算文档"
    assert not SummaryResult(text=NO_ANSWER, coverage=grounded).storable(), "哨兵回答不算文档"
    assert not SummaryResult(text="好的", coverage=grounded).storable(), "过短的回答不算文档"
    assert not SummaryResult(text=text, coverage=CoverageLog()).storable(), (
        "零工具调用的追问不算文档"
    )

    _seed_summaries(store)
    rejected = SummaryResult(text="好的", coverage=grounded)
    assert (
        store_summary(
            store,
            group_openid="G_demo",
            instruction="总结一下",
            requested_by=None,
            result=rejected,
        )
        is None
    ), "不合格的回答不入库"
    assert len(store.summaries_for(None)) == 3, "库里仍只有 3 篇"


async def test_same_thread_is_serialised(store, tmp_path) -> None:
    # Two runs on one thread must not overlap. They share one checkpointer
    # thread, and interleaved supersteps corrupt its state — reachable today
    # by two fast @s or two DMs from one user, and by an auto-summary racing
    # a summon tomorrow. botpy gives every event its own task, so nothing
    # else serialises them.
    import asyncio

    index = _make_index(tmp_path, "lock_summaries")
    final = _final()

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

    agent = SlowAgent()
    busy = Summarizer(store, index, model=ToolCallingFakeModel(responses=[final]))

    def ctx() -> BotContext:
        return BotContext(
            group_openid="G_demo", store=store, index=index, scope=SCOPE_GROUP,
        )

    first = asyncio.create_task(
        busy._run(agent=agent, thread_key="G:G_demo", instruction="一", context=ctx())
    )
    await asyncio.sleep(0)  # let the first task reach the agent
    second = asyncio.create_task(
        busy._run(agent=agent, thread_key="G:G_demo", instruction="二", context=ctx())
    )
    await asyncio.sleep(0.01)
    assert agent.calls == 1 and agent.peak == 1, (
        "同一 thread 的第二次调用在锁外等待，没有并发进入 agent"
    )
    assert busy.group_busy("G_demo"), "在跑时 group_busy 为真"

    agent.gate.set()
    results = await asyncio.gather(first, second)
    assert agent.calls == 2 and agent.peak == 1 and len(results) == 2, (
        "放行后两次都完成，且始终没有并发"
    )
    assert not busy.group_busy("G_demo"), "跑完后不再 busy"
