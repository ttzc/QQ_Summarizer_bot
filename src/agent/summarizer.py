"""The summarising agents: one per scope, sharing a model and a checkpointer.

Two graphs rather than one, because the privacy boundary points in opposite
directions in the two scopes. In a group the model must never name a group; in
private chat it must read across all of them. `create_agent` binds tools at
construction, so the honest way to express that is two agents with two tool sets
— a single agent would have to hold the cross-group tool and gate it at runtime,
which is exactly the arrangement `tools.py` exists to avoid.

Thread ids are namespaced (`G:` / `U:`) because both agents share one checkpointer
and `thread_id` is the only separator between their state (both default to
`checkpoint_ns=""`). QQ does not promise that group and user openids are
disjoint, so an unprefixed id could let two different conversations collide.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.tools import (
    C2C_TOOLS,
    DATA_TOOLS,
    GROUP_TOOLS,
    SCOPE_ALL,
    SCOPE_GROUP,
    BotContext,
    CoverageLog,
)
from src.agent.publish import insert_document
from src.api.llm_client import get_chat_model
from src.config import PROJECT_ROOT
from src.logger import setup_logger
from src.rag.retriever import SummaryIndex
from src.store.sql_store import SQLStore

logger = setup_logger("qqbot.agent.summarizer")

PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Shared between both scopes, then the scope-specific half. Split so the output
# style, length limit and injection defences cannot drift apart between them.
COMMON_PROMPT_PATH = PROMPTS_DIR / "common.md"
GROUP_PROMPT_PATH = PROMPTS_DIR / "summarizer.md"
C2C_PROMPT_PATH = PROMPTS_DIR / "c2c.md"

# Conversation threads kept alive across both scopes. `InMemorySaver` never
# evicts on its own, so an unbounded bot would leak slowly; this bounds it. Both
# group and private threads draw on the same budget — a single LRU is simpler
# than two, at the cost of a chatty DM audience being able to push out group
# memory, which the generous ceiling makes unlikely in practice.
MAX_THREADS = 128

# LangGraph's own default is 25. A cap keeps a tool-calling loop from running
# away inside the 5-minute passive-reply window.
RECURSION_LIMIT = 25

NO_ANSWER = "抱歉，我没能就这些消息给出结果，换个说法或缩小范围再试一次？"

GROUP_FALLBACK_PROMPT = (
    "你是 QQ 群聊总结助手。用工具读取本群消息后，按话题分组总结，"
    "标注发言人与时间，并说明数据覆盖范围。输出纯文本，不要用表格。"
)

C2C_FALLBACK_PROMPT = (
    "你是 QQ 机器人。用户私聊你时，用 search_summaries 检索已有总结，"
    "需要核对原话时用 messages_across_groups 按时间范围读原文，"
    "并注明结论来自哪个群、覆盖什么时间。输出纯文本。"
)

# Below this, an answer is a conversational turn rather than a document.
MIN_DOC_CHARS = 30


def extract_reply(messages: Sequence[BaseMessage]) -> str:
    """Text of the last non-empty AI message."""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        text = _text_of(message.content)
        if text.strip():
            return text.strip()
    return ""


def _text_of(content: Any) -> str:
    """Flatten message content, which may be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _tools_called(messages: Sequence[BaseMessage]) -> set[str]:
    names: set[str] = set()
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else None
            if name:
                names.add(str(name))
    return names


@dataclass(slots=True)
class SummaryResult:
    """One answer, plus what it was based on and what it published."""

    text: str
    coverage: CoverageLog
    # Filled from the injected context after the run: what `save_summary` wrote
    # *during* it, if anything. The caller uses this to decide whether to wake
    # the indexer, and the auto path uses `published_text` (not `text`) when it
    # pushes a summary to the group.
    published_ids: list[str] = field(default_factory=list)
    published_text: str = ""

    @property
    def published(self) -> bool:
        return bool(self.published_ids)

    def storable(self) -> bool:
        """Is this answer fit to be stored **on code's initiative**?

        This is no longer the gate for the `@` path — the model publishes there
        through the `save_summary` tool, and its judgement is the one that
        matters. This method now only serves the two places where *code* decides,
        because nobody is asking the model to: the auto-summary fallback (the
        model ran on a fixed instruction and may not have published) and the
        offline `ask --save`.

        All three conditions are required:

        * not the `NO_ANSWER` sentinel and long enough to carry content;
        * **at least one data tool actually ran.**

        The floor here is deliberately `MIN_DOC_CHARS` (30), far below
        `config.summary.min_publish_chars` (200) that the tool enforces. Raising
        it would be the tempting tidy-up and a real bug: the auto trigger counts
        messages *since the last stored row*, so a thin-but-successful summary
        that refuses to be stored leaves that count unreset, and the next message
        after the cooldown re-summarises the very same batch — every cooldown,
        forever. Better one thin document than an infinite loop of LLM calls.
        """
        text = self.text.strip()
        if text == NO_ANSWER or len(text) < MIN_DOC_CHARS:
            return False
        return not self.coverage.is_empty()


def group_thread_key(group_openid: str) -> str:
    """Conversation thread for a group. Must match what `summarize_group` uses."""
    return f"G:{group_openid}"


def store_summary(
    store: SQLStore,
    *,
    group_openid: str,
    instruction: str,
    requested_by: str | None,
    result: SummaryResult,
    trigger: str = "at",
) -> str | None:
    """Write `result.text` as a document, if code is entitled to write it at all.

    The **fallback** writer, used only where nobody asked the model to publish:
    the auto-summary path when its run produced nothing via `save_summary`, and
    the offline `ask --save`. The `@` path no longer comes through here — the
    model publishes there itself, mid-run, through the tool.

    `result.published` short-circuits it. Without that check a model that
    published during an auto run would leave *two* rows for one summary: its own,
    plus this one carrying the same coverage. That is not hypothetical once the
    tool exists — the auto instruction literally says "请总结本群最近的讨论", so
    the model has every reason to publish.
    """
    if result.published:
        return None
    if not result.storable():
        return None
    return insert_document(
        store,
        group_openid=group_openid,
        instruction=instruction,
        content=result.text,
        coverage=result.coverage.intervals(),
        message_count=result.coverage.message_count(),
        requested_by=requested_by,
        trigger=trigger,
    )


def _load_prompt(*paths: Path) -> str:
    """Concatenate prompt fragments, skipping any that are missing."""
    parts: list[str] = []
    for path in paths:
        try:
            parts.append(path.read_text(encoding="utf-8").strip())
        except OSError:
            logger.warning("未找到 prompt 片段", extra={"path": str(path)})
    return "\n\n".join(p for p in parts if p)


class Summarizer:
    def __init__(
        self,
        store: SQLStore,
        index: SummaryIndex,
        *,
        model: BaseChatModel | None = None,
        common_prompt_path: Path | None = None,
        group_prompt_path: Path | None = None,
        c2c_prompt_path: Path | None = None,
        checkpointer: Any = None,
    ) -> None:
        self._store = store
        self._index = index

        common = common_prompt_path or COMMON_PROMPT_PATH
        group_prompt = _load_prompt(common, group_prompt_path or GROUP_PROMPT_PATH)
        c2c_prompt = _load_prompt(common, c2c_prompt_path or C2C_PROMPT_PATH)

        self._checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        chat_model = model or get_chat_model()
        self._group_agent = create_agent(
            chat_model,
            GROUP_TOOLS,
            system_prompt=group_prompt or GROUP_FALLBACK_PROMPT,
            context_schema=BotContext,
            checkpointer=self._checkpointer,
        )
        self._c2c_agent = create_agent(
            chat_model,
            C2C_TOOLS,
            system_prompt=c2c_prompt or C2C_FALLBACK_PROMPT,
            context_schema=BotContext,
            checkpointer=self._checkpointer,
        )
        # Thread keys kept most-recently-used last, so the oldest can be evicted.
        self._threads: OrderedDict[str, None] = OrderedDict()
        # One lock per thread key, so two runs never write the same checkpointer
        # thread at once. Not a theoretical worry: botpy schedules *each event*
        # as its own task, so two `@`s in the same group (or two DMs from one
        # user) already raced before anything auto-triggered, and every extra
        # trigger source multiplies the odds.
        self._locks: dict[str, asyncio.Lock] = {}

    # ---- memory -----------------------------------------------------------

    def _touch_thread(self, thread_key: str) -> None:
        # NB: log keys here must not be named `thread` — `logging` builds records
        # with a `thread` attribute and raises `KeyError: Attempt to overwrite`
        # when `extra` collides with one.
        self._threads.pop(thread_key, None)
        self._threads[thread_key] = None
        while len(self._threads) > MAX_THREADS:
            oldest, _ = self._threads.popitem(last=False)
            logger.info("淘汰最早会话记忆", extra={"thread_id": oldest})
            try:
                self._checkpointer.delete_thread(oldest)
            except Exception:  # noqa: BLE001 - eviction is best-effort
                logger.exception("清理会话记忆失败", extra={"thread_id": oldest})
            # Drop the lock too, but never one that is held or awaited: the
            # waiter would then share a lock with nobody and run concurrently
            # with the next arrival.
            lock = self._locks.get(oldest)
            if lock is not None and not lock.locked():
                self._locks.pop(oldest, None)

    def _lock_for(self, thread_key: str) -> asyncio.Lock:
        lock = self._locks.get(thread_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[thread_key] = lock
        return lock

    def group_busy(self, group_openid: str) -> bool:
        """Whether a run for this group is already in flight.

        The auto-summary trigger uses it to *skip* rather than queue: a group
        that keeps talking during a slow summary should not pile up a second one.
        """
        lock = self._locks.get(group_thread_key(group_openid))
        return lock is not None and lock.locked()

    # ---- entry points -----------------------------------------------------

    async def summarize_group(
        self,
        group_openid: str,
        instruction: str,
        *,
        requested_by: str | None = None,
        trigger: str = "at",
        allow_publish: bool = True,
    ) -> SummaryResult:
        """Answer `instruction` using only this group's messages.

        The three keyword arguments are **provenance for `save_summary`**: the
        requester, which entry point this turn came from, and whether publishing
        is allowed at all. They ride on the injected context rather than the tool
        signature for the same reason `group_openid` does — a model that could
        name a requester or a `trigger` would eventually invent one, and the
        audit column would stop meaning anything.

        `allow_publish=False` exists because the offline `ask` command runs this
        very agent: tools are bound at construction, so the only per-call way to
        keep a hand-run query from writing documents is a gate on the context.
        """
        return await self._run(
            agent=self._group_agent,
            thread_key=group_thread_key(group_openid),
            instruction=instruction,
            context=BotContext(
                group_openid=group_openid,
                store=self._store,
                index=self._index,
                scope=SCOPE_GROUP,
                instruction=instruction,
                requested_by=requested_by,
                trigger=trigger,
                allow_publish=allow_publish,
            ),
        )

    async def answer_private(self, user_openid: str, instruction: str) -> SummaryResult:
        """Answer a private-chat message from the summaries of every group.

        Publishing is off here by design: a private answer spans groups, and
        `summaries.group_openid` is NOT NULL — there is no honest owner for such a
        document. It is derived from summaries too, so keeping it would start a
        chain of summaries-of-summaries.
        """
        return await self._run(
            agent=self._c2c_agent,
            thread_key=f"U:{user_openid}",
            instruction=instruction,
            context=BotContext(
                group_openid=None,
                store=self._store,
                index=self._index,
                scope=SCOPE_ALL,
                instruction=instruction,
                allow_publish=False,
            ),
        )

    # ---- internals --------------------------------------------------------

    async def _run(
        self,
        *,
        agent: Any,
        thread_key: str,
        instruction: str,
        context: BotContext,
    ) -> SummaryResult:
        self._touch_thread(thread_key)
        # Serialised per thread: LangGraph's checkpointer keeps one history per
        # `thread_id`, and two overlapping `ainvoke`s on it interleave their
        # supersteps. Waiting is the right trade — a queued `@` still answers
        # well inside the 5-minute passive window.
        async with self._lock_for(thread_key):
            state = await agent.ainvoke(
                {"messages": [{"role": "user", "content": instruction}]},
                config={
                    "configurable": {"thread_id": thread_key},
                    "recursion_limit": RECURSION_LIMIT,
                },
                context=context,
            )
        messages = state.get("messages", [])

        # Coverage is carried out of the graph by mutating the context object,
        # which relies on LangGraph passing it by reference. If that ever stops
        # holding, every summary would silently store with an empty window — so
        # say something rather than let that pass unnoticed.
        called = _tools_called(messages)
        if called & DATA_TOOLS and context.coverage.is_empty():
            logger.warning(
                "agent 取了数但未记录覆盖范围（context 可能被复制）",
                extra={"thread_id": thread_key, "tools": sorted(called & DATA_TOOLS)},
            )

        logger.debug(
            "agent 完成",
            extra={"thread_id": thread_key, "messages": len(messages)},
        )
        return SummaryResult(
            text=extract_reply(messages) or NO_ANSWER,
            coverage=context.coverage,
            # Carried out of the graph the same way `coverage` is — by reference
            # on the injected context. Same caveat as `CoverageLog`: if LangGraph
            # ever copies the context, these come back empty and every publish
            # would look like a non-publish to the caller.
            published_ids=list(context.published_ids),
            published_text=context.published_text,
        )
