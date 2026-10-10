"""Tools the summariser agent may call, plus the context they read from.

**The group is never a tool argument.** `group_openid` comes from
`runtime.context`, which is injected server-side. Two reasons:

* a model that has to *choose* a group will eventually choose the wrong one, and
  this is a privacy boundary — group A must never be summarised using group B's
  messages;
* the injection is tamper-proof. `langgraph/prebuilt/tool_node.py` strips any
  LLM-supplied values for injected arguments and replaces them with trusted ones
  (see the comment at the `stripped_args` assignment), so a prompt-injected
  message cannot forge `group_openid` by "calling" the tool with an argument.

Detected by parameter *name* (`runtime`) and by annotation (`ToolRuntime`, which
subclasses `_DirectlyInjectedToolArg`) — `_get_all_injected_args` checks both.

**Annotate it `ToolRuntime[BotContext, dict]`, never bare.** `ToolRuntime` is a
generic *dataclass* over `(ContextT, StateT)` whose `TypeVar`s carry defaults
(`tool_node.py:105-106`: `StateT = TypeVar("StateT", default=dict)`,
`ContextT = TypeVar("ContextT", default=None)`). Bare, pydantic substitutes the
defaults while building the args schema, so `runtime.context` is typed `None`;
handing it a real `BotContext` then makes `BaseTool._parse_input`'s
`result_v2.model_dump()` (`langchain_core/tools/base.py:835`) emit
`PydanticSerializationUnexpectedValue` on every single tool call. Validation is
lenient enough that the tool still works, which is exactly what makes it a bad
warning: pure stderr noise that looks like a defect. Parameterising the two
type vars removes it — same schema detection (`_is_injected_arg_type` unwraps
`get_origin`), same injection, same stripped-args guarantee.

**Two tool sets, because the boundary points in opposite directions.** A group
answer must not reach outside its group; a private-chat answer is *supposed* to
span groups. Those cannot be the same graph, or the group graph would contain the
cross-group capability and safety would rest on a runtime `if`. So `GROUP_TOOLS`
has no cross-group path at all, and `C2C_TOOLS` has the cross-group ones.
`src/agent/summarizer.py` builds one agent per set.

The asymmetry is deliberate, and it is not symmetric the other way round:

* the group agent must never be able to *name* another group, so its raw tools
  read `ctx.group_openid` and nothing else;
* the private-chat agent may name one, because naming a group grants it nothing
  it does not already have — `messages_across_groups(group=None)` reads every
  group anyway. Requiring a group there would only make the model guess.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from langchain.tools import ToolRuntime, tool

from src.agent.publish import insert_document
from src.config import config
from src.logger import setup_logger
from src.media.sniff import sniff_image
from src.media.vision import describe_image
from src.rag.retriever import SummaryIndex, group_label, speaker_of
from src.store.sql_store import SQLStore

logger = setup_logger("qqbot.agent.tools")

# Cap on characters returned by one tool call. A busy group can trivially supply
# more text than a context window holds, and a blown context means no summary at
# all. When the cap bites we keep the *newest* messages (what "最近" usually
# means) and tell the model how many were dropped so it can narrow the range.
MAX_TOOL_CHARS = 12000

# QQ renders an @ as `<@32-hex-openid>` inside the message text itself, and the
# stored `instruction` keeps it byte-for-byte (like every other raw field here).
# Used only to *render a short label* — see `_label_of`. Nothing that goes into
# `content` / `raw_json` / a document body passes through it.
_MENTION_RE = re.compile(r"<@!?[0-9A-Fa-f]*>")

SCOPE_GROUP = "group"
SCOPE_ALL = "all"


def _stamp(raw: Any) -> str:
    try:
        return datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(raw or "")


def _field(row: sqlite3.Row | dict, key: str) -> Any:
    return row[key] if isinstance(row, sqlite3.Row) else row.get(key)


def _bounds(rows: Sequence[sqlite3.Row | dict]) -> tuple[str | None, str | None]:
    """Earliest and latest `ts` in a batch of message rows."""
    stamps = [str(_field(row, "ts") or "") for row in rows]
    stamps = [s for s in stamps if s]
    if not stamps:
        return (None, None)
    return (min(stamps), max(stamps))


@dataclass(slots=True)
class CoverageLog:
    """What the agent actually read, recorded by the tools as they run.

    Read back after `ainvoke`, which works because the `BotContext` instance is
    handed to LangGraph by reference: `_coerce_context` in
    `langgraph/pregel/main.py` returns an instance as-is and only rebuilds a plain
    dict. That is an implementation detail, not a documented contract, so
    `Summarizer` also warns when tool calls happened but this log stayed empty —
    turning a silent "stored with no coverage window" into a visible warning.
    """

    reads: list[dict] = field(default_factory=list)

    def add(
        self,
        tool_name: str,
        count: int,
        ts_start: str | None = None,
        ts_end: str | None = None,
        *,
        envelope: bool = True,
    ) -> None:
        """Record one read.

        `envelope=False` marks a read that **proves grounding but not coverage** —
        looking at the corpus tells you the answer is not invented out of thin
        air, yet it says nothing about having read the discussion a window spans.
        Anything reading *the library itself* must pass it; see
        `summaries_in_range` / `get_summary`.

        Skipping it would be subtle and fatal: the envelope feeds
        `summaries.ts_start/ts_end`, which is the axis the duplicate-guard query
        overlaps on. So a run that checked the library before writing would
        inherit the newest existing `ts_end` as its own, each new document would
        claim to cover "up to now", every later overlap query would hit itself,
        and the dedup rule would invert into "never publish anything".
        """
        self.reads.append(
            {
                "tool": tool_name,
                "count": int(count),
                "ts_start": ts_start,
                "ts_end": ts_end,
                "envelope": envelope,
            }
        )

    def add_rows(self, tool_name: str, rows: Sequence[sqlite3.Row | dict]) -> None:
        start, end = _bounds(rows)
        self.add(tool_name, len(rows), start, end)

    def intervals(self) -> list[tuple[str, str]]:
        """The coverage **envelope** — reads that actually span material.

        Deliberately narrower than `is_empty()`: this feeds the stored time
        window, so library-inspection reads are excluded (see `add`).
        """
        return [
            (r["ts_start"], r["ts_end"])
            for r in self.reads
            if r.get("ts_start") and r.get("ts_end") and r.get("envelope", True)
        ]

    def message_count(self) -> int:
        """Messages read *directly*. A summary-search-only answer contributes 0."""
        return sum(int(r.get("count") or 0) for r in self.reads)

    def is_empty(self) -> bool:
        """Nothing at all was read — not even the library.

        The "was this grounded in anything" test, and deliberately broader than
        `intervals()`: a synthesis built purely from existing summaries has no
        covered window of raw messages but is still not invented.
        """
        return not self.reads


@dataclass(slots=True)
class BotContext:
    """Per-invocation dependencies. Injected; outside the model's reach."""

    group_openid: str | None
    store: SQLStore
    index: SummaryIndex
    scope: str = SCOPE_GROUP
    coverage: CoverageLog = field(default_factory=CoverageLog)
    # Real vision calls spent by `view_image` in *this run*. Lives on the
    # per-invocation context, so the cap resets by construction every summary —
    # no registry, no TTL, nothing to leak.
    views_used: int = 0

    # ---- provenance: what this run was, for the document it publishes ----
    # The three fields the `save_summary` tool needs to write a complete row,
    # injected rather than passed as tool arguments for the same reason
    # `group_openid` is: the model must not be able to name a requester, pick a
    # `trigger`, or claim credit for a turn it was not asked to publish.
    instruction: str = ""
    requested_by: str | None = None
    trigger: str = "at"
    # Off for the offline `ask` without `--save`: the publishing tool is bound at
    # agent construction and cannot be removed per-call, so the gate lives here.
    allow_publish: bool = True

    # ---- per-run budgets and what got published ----------------------------
    # Same reasoning as `views_used`: both caps are *per answer*, and the context
    # object is created per answer, so this needs no bookkeeping to reset.
    publishes_used: int = 0
    doc_reads_used: int = 0
    # Read back by the caller after `ainvoke`, riding out of the graph on the
    # same by-reference context that `coverage` does (see `CoverageLog`).
    published_ids: list[str] = field(default_factory=list)
    published_text: str = ""


def _render(
    rows: list[sqlite3.Row], budget: int = MAX_TOOL_CHARS, show_group: bool = False
) -> str:
    """Chronological `[time] speaker: text` lines, newest kept on overflow.

    `show_group` prefixes each line with its group. A cross-group result needs
    it — a private-chat answer has to say *which* group a quote came from — while
    the in-group tools render under an implicit "here", so they leave it off.
    """
    if not rows:
        return "（没有符合条件的消息）" if show_group else "（本群没有符合条件的消息）"

    kept: list[str] = []
    used = 0
    for row in reversed(rows):  # newest first while budgeting
        who = speaker_of(row)
        if show_group:
            who = f"{group_label(row['group_openid'])}·{who}"
        line = f"[{_stamp(row['ts'])}] {who}: {(row['content'] or '').strip()}"
        if used + len(line) > budget:
            break
        used += len(line) + 1
        kept.append(line)
    kept.reverse()

    dropped = len(rows) - len(kept)
    body = "\n".join(kept)
    if dropped:
        body += f"\n……（较早的 {dropped} 条因长度上限未显示，可缩小时间范围后重试）"
    return body


def _resolve_group(ref: str, store: SQLStore) -> str | None:
    """Map what the model read back from `list_groups` to a `group_openid`.

    Private chat sees group *labels*, not ids, so the model can only echo a
    label: either a `[groups]` alias or the `群<尾号>` stub `group_label` falls
    back to. A bare openid is accepted too, in case one was handed over.
    """
    ref = ref.strip()
    if not ref:
        return None
    if ref in config.groups:
        return ref
    for openid, alias in config.groups.items():
        if ref == alias or ref in alias:
            return openid

    # Groups without an alias are only reachable through the label stub, which
    # needs the real list of groups — the aliases alone are not enough. Both
    # sources are needed: a group that has raw messages but not one summary yet
    # (still under the auto-summary threshold) is nameable just the same, and it
    # is invisible to `list_groups`.
    known = list(config.groups)
    try:
        known += [row["group_openid"] for row in store.groups_with_summaries()]
        known += store.groups_with_messages()
    except Exception:  # noqa: BLE001 - alias-only resolution is still useful
        logger.debug("读取群列表失败，仅用配置里的别名", exc_info=True)

    # `群<尾号>` is what `group_label` shows for an un-aliased group.
    tail = ref[1:] if ref.startswith("群") else None
    for openid in known:
        if ref == openid or (tail and len(tail) >= 4 and openid.endswith(tail)):
            return openid
    return None


@tool
async def current_time(*, runtime: ToolRuntime[BotContext, dict]) -> str:
    """获取当前本地时间（ISO8601，含时区）。

    用它把"今天""昨天""最近一小时"这类相对说法换算成具体时间范围，再交给
    messages_in_range。
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


@tool
async def recent_messages(
    limit: int = 200, *, runtime: ToolRuntime[BotContext, dict]
) -> str:
    """取本群最近 limit 条消息，按时间正序，每行形如"[2026-07-21 08:00] 昵称: 内容"。

    当用户说"总结一下""刚才聊了什么"而没有给出明确时间范围时用它。
    """
    ctx: BotContext = runtime.context
    if not ctx.group_openid:
        return "（本场景没有群上下文，无法读取原文）"
    limit = max(1, min(int(limit), 2000))
    rows = ctx.store.recent_messages(ctx.group_openid, limit)
    ctx.coverage.add_rows("recent_messages", rows)
    return _render(rows)


@tool
async def messages_in_range(
    start_iso: str,
    end_iso: str,
    limit: int = 500,
    *,
    runtime: ToolRuntime[BotContext, dict],
) -> str:
    """取本群在指定时间范围内（含两端）的消息，按时间正序。

    参数是 ISO8601 字符串，如 "2026-07-21T00:00:00+08:00"。用于"今天""上周"
    这类指令：先调 current_time 算出具体边界，再调本工具。
    """
    ctx: BotContext = runtime.context
    if not ctx.group_openid:
        return "（本场景没有群上下文，无法读取原文）"
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except (TypeError, ValueError):
        return (
            f"时间格式无法解析：start_iso={start_iso!r} end_iso={end_iso!r}。"
            "请使用 ISO8601，例如 2026-07-21T00:00:00+08:00。"
        )
    if start > end:
        start, end = end, start

    limit = max(1, min(int(limit), 5000))
    rows = ctx.store.messages_in_range(
        ctx.group_openid, start.isoformat(), end.isoformat(), limit
    )
    ctx.coverage.add_rows("messages_in_range", rows)
    return _render(rows)


@tool
async def search_summaries(
    query: str, k: int = 5, *, runtime: ToolRuntime[BotContext, dict]
) -> str:
    """对**已有的总结文档**做语义检索，返回最相关的 k 篇（含群、时间范围与正文）。

    适用于"之前是不是聊过 X""有人提过 Y 吗"这类按主题而非按时间的查找。

    它只覆盖**已经被 @ 总结过**的话题：没有总结过的讨论，这里查不到，需要改用
    按时间读原文的工具。群内调用只搜本群；私聊调用跨全部群。
    """
    ctx: BotContext = runtime.context
    # `None` is the cross-group path and only the private-chat scope may take it.
    group = ctx.group_openid if ctx.scope == SCOPE_GROUP else None
    k = max(1, min(int(k), 50))
    hits = ctx.index.search(query, k=k, group_openid=group)
    if not hits:
        return "（没有检索到相关总结）"

    starts, ends = [], []
    lines: list[str] = []
    used = 0
    for doc, score in hits:
        # `page_content` already leads with "[群] 时间范围 · N 条", so the group
        # and window need no repetition here.
        line = f"{doc.page_content}（距离 {score:.3f}）"
        if used + len(line) > MAX_TOOL_CHARS:
            break
        used += len(line) + 1
        lines.append(line)
        meta = doc.metadata or {}
        if meta.get("ts_start"):
            starts.append(str(meta["ts_start"]))
        if meta.get("ts_end"):
            ends.append(str(meta["ts_end"]))

    ctx.coverage.add(
        "search_summaries",
        0,  # summaries were read, not messages; `message_count` stays honest
        min(starts) if starts else None,
        max(ends) if ends else None,
    )
    return "\n".join(lines)


@tool
async def list_groups(*, runtime: ToolRuntime[BotContext, dict]) -> str:
    """列出目前**有总结**的群（群名、总结篇数、时间范围）。仅私聊场景可用。

    当用户问"你都知道哪些群""有没有关于 X 的群"时用它，好让用户知道可检索的
    范围；也可以据此判断某个群根本还没积累总结。
    """
    ctx: BotContext = runtime.context
    rows = ctx.store.groups_with_summaries(limit=config.c2c.max_groups_shown)
    if not rows:
        return "（还没有任何群的总结）"
    return "\n".join(
        f"{group_label(row['group_openid'])}：{row['summaries']} 篇，"
        f"最近 {str(row['last_at'])[:19]}"
        for row in rows
    )


@tool
async def messages_across_groups(
    start_iso: str,
    end_iso: str,
    group: str | None = None,
    keyword: str | None = None,
    limit: int | None = None,
    *,
    runtime: ToolRuntime[BotContext, dict],
) -> str:
    """按时间范围取**所有群**的原始聊天记录（每行带群名、时间、发言人）。仅私聊可用。

    用于核对原话、找细节、查某人具体说了什么——这些是总结里可能被丢掉的东西。

    时间范围**必填**：先调 current_time 把"上周""这两天"换算成 ISO8601。一次别
    取太大的范围；返回里如果提示有内容未显示，就缩小范围重来。

    group 可选：填群名可以把范围收敛到某个群（群名用 list_groups 看到的那个，或上一次
    取原文时每行前缀里的那个）；不填就是全部群。填错会提示你去调 list_groups。
    keyword 可选：只保留正文包含这个词的消息。
    """
    ctx: BotContext = runtime.context
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except (TypeError, ValueError):
        return (
            f"时间格式无法解析：start_iso={start_iso!r} end_iso={end_iso!r}。"
            "请使用 ISO8601，例如 2026-07-21T00:00:00+08:00。"
        )
    if start > end:
        start, end = end, start

    group_openid: str | None = None
    if group and group.strip():
        group_openid = _resolve_group(group, ctx.store)
        if group_openid is None:
            return (
                f"找不到名为「{group}」的群。可以先用 list_groups 看看有哪些群，"
                "或者不填 group 直接跨全部群检索。"
            )

    cap = max(1, int(config.c2c.raw_limit))
    limit = cap if limit is None else max(1, min(int(limit), cap))
    rows = ctx.store.messages_in_range(
        group_openid,
        start.isoformat(),
        end.isoformat(),
        limit,
        keyword=(keyword or "").strip() or None,
    )
    ctx.coverage.add_rows("messages_across_groups", rows)
    return _render(rows, show_group=True)


def _read_media_file(rel_path: str) -> bytes:
    """按 `[media].dir / path` 读落盘图片。dir 在调用时解析——测试覆盖配置即生效。"""
    return (config.media.dir_path / rel_path).read_bytes()


@tool
async def view_image(
    media_ref: str,
    focus: str = "",
    *,
    runtime: ToolRuntime[BotContext, dict],
) -> str:
    """查看一张已存图（图片附件）的内容。两个 scope 都可调用。

    media_ref 是消息正文占位 `[图片 xxx.jpg #4d9f2a1c]` 里 # 后面的短 id。
    **只在图对当前问题真正重要时才调**——看图花时间和调用额度。不带 focus
    得到通用描述（首次看图会现场看并缓存，之后所有人都直接读到文本）；带
    focus（例如"图里有没有提到上线时间"）得到针对性回答，不缓存。
    """
    ctx: BotContext = runtime.context
    row = ctx.store.find_media(media_ref)
    if row is None:
        return "找不到这张图——请使用正文里 [图片 … #短id] 的短 id（至少 6 位十六进制）。"
    # The short id is 8 hex characters; guessing one from another group is
    # conceivable, so the group scope refuses instead of trusting luck.
    if ctx.scope == SCOPE_GROUP and row["group_openid"] != ctx.group_openid:
        return "无权查看其他群的图片。"
    if row["status"] != "stored" or not row["path"]:
        return f"这张图现在看不了（状态：{row['status']}）。"
    if not focus.strip() and row["description"]:
        return f"[图片描述] {row['description']}"
    if ctx.views_used >= int(config.media.max_views):
        return (
            f"本次回答查看图片已达上限（{config.media.max_views} 张），"
            "请基于已有信息作答。"
        )

    try:
        data = await asyncio.to_thread(_read_media_file, row["path"])
    except OSError:
        logger.warning("media 文件缺失", extra={"media_id": row["media_id"], "path": row["path"]})
        return "这张图的文件缺失，看不了。"
    # 32 MiB 是**落盘**护栏（外链下载的官方约束）；内联 base64 是另一回事——
    # 几 MB 的 body 会被网关在传输层拒掉，每张都拒、每次扣额度、永远看不到。
    # 所以看一遍尺寸，超限直接给明确文案，不进入调用路径。
    if len(data) > int(config.media.max_inline_bytes):
        return (
            f"这张图有 {len(data) // (1 << 20)} MB，超过内联看图上限"
            f"（{int(config.media.max_inline_bytes) // (1 << 20)} MB），看不了。"
        )
    sniffed = sniff_image(data)
    if sniffed is None:
        return "这张图的文件内容不是可查看的图片格式。"
    _ext, mime = sniffed

    prompt = (
        f"{focus.strip()}\n（请结合图片回答；若图中有相关文字，请逐字引用。）"
        if focus.strip()
        else config.media.describe_prompt
    )
    # 额度只花在**真实视觉调用**上：读盘/魔数/尺寸这些没走到模型的路径不扣——
    # 否则一个坏文件就能把 max_views 白白吃空，把好图挤在门外。
    ctx.views_used += 1
    try:
        answer = await describe_image(data, mime, prompt)
    except Exception as exc:  # noqa: BLE001 - 一次工具失败不拖垮整轮（同 messages_in_range 的约定）
        logger.warning(
            "view_image 视觉调用失败",
            extra={"media_id": row["media_id"], "err": repr(exc)[:200]},
        )
        return f"这张图暂时看不了：{repr(exc)[:120]}"
    if not answer:
        return "模型对这张图没有返回内容。"

    if not focus.strip():
        # 通用描述缓存回写：只回写第一次，之后取数/检索直接看到文本、零调用。
        # 带 focus 的定向回答不缓存——它只服务于这一次提问。
        ctx.store.save_media_description(row["media_id"], answer)
        return f"[图片描述] {answer}"
    return f"[图片定向回答] {answer}"


def _label_of(instruction: Any) -> str:
    """The one-line "what is this document" for a listing.

    Built from the stored `instruction` rather than a new column: that column is
    NOT NULL, already carries the requester's own words ("总结一下今天"), and is
    strictly better than a title the model invents at publish time — those come
    out as "群聊总结" / "今日讨论", cost tokens to produce, and need a paragraph
    of prompt to be any good. Adding a column for it would also be the only
    schema change this feature needs.

    Mention markup is stripped **here only**. Real `instruction` values start with
    QQ's own `<@32-hex-openid>` token (真机 2026-10-09 库里的行就是这样), which would
    spend the whole 40-char budget on an openid the reader cannot use — the group
    is already implicit in "this listing is for my group". The column itself stays
    byte-for-byte, as everywhere else in this project.
    """
    text = _MENTION_RE.sub(" ", str(instruction or ""))
    text = " ".join(text.split())
    if not text:
        return "（无指令）"
    return text[:40] + "…" if len(text) > 40 else text


@tool
async def summaries_in_range(
    start_iso: str,
    end_iso: str,
    limit: int = 20,
    *,
    runtime: ToolRuntime[BotContext, dict],
) -> str:
    """看**库里**哪些总结的覆盖时段与给定范围相交。仅群内场景。

    投稿前的自查工具：刚用 messages_in_range 读完 `[start, end]` 的原文，就用
    **同一段时间**问这里——空返回说明这段还没被写过，有返回说明已经有人写过，
    除非你能补充新东西，否则不要再投一篇。

    时间参数是 ISO8601（如 "2026-07-21T00:00:00+08:00"），与 messages_in_range
    同一套。返回只有清单（短 id、原指令摘要、覆盖时段、条数），**不含正文**；
    需要比对内容再用 get_summary 读那一篇。
    """
    ctx: BotContext = runtime.context
    if not ctx.group_openid:
        return "（本场景没有群上下文，无法查本群的总结清单）"
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except (TypeError, ValueError):
        return (
            f"时间格式无法解析：start_iso={start_iso!r} end_iso={end_iso!r}。"
            "请使用 ISO8601，例如 2026-07-21T00:00:00+08:00。"
        )
    if start > end:
        start, end = end, start

    cap = max(1, min(int(limit), 100))
    rows, total = ctx.store.summaries_in_range(
        ctx.group_openid, start.isoformat(), end.isoformat(), cap
    )
    if total == 0:
        return "这段时间库里还没有任何总结。"
    # Listing the library is grounding, but it proves nothing about having read
    # the discussion, so it must not enter the coverage envelope — see `add`.
    ctx.coverage.add("summaries_in_range", 0, envelope=False)
    lines = [
        f"{row['summary_id'][:8]}｜{_label_of(row['instruction'])}｜"
        f"{_stamp(row['ts_start'])}~{_stamp(row['ts_end'])}｜"
        f"{int(row['message_count'] or 0)} 条｜{'自动' if row['trigger'] == 'auto' else '被@'}"
        for row in rows
    ]
    if total > len(rows):
        lines.append(
            f"……（该时段共 {total} 篇，此处只列覆盖时段最近的 {len(rows)} 篇；"
            "如需更早的，请把范围缩小到具体某天再查）"
        )
    return "\n".join(lines)


@tool
async def get_summary(
    summary_ref: str,
    *,
    runtime: ToolRuntime[BotContext, dict],
) -> str:
    """按短 id 读**一篇**已入库总结的正文。仅群内场景。

    summary_ref 是 summaries_in_range 每行开头的那个短 id（至少 6 位）。
    只有在你确实需要比对旧稿的措辞或结论时才调——正文很长，读一篇占一轮。
    """
    ctx: BotContext = runtime.context
    # Refuse outright rather than read "no group" as "every group": this tool is
    # only in the group set, so an empty scope here means a wiring mistake, and
    # the fail-open version of it would be a cross-group read.
    if not ctx.group_openid:
        return "（本场景没有群上下文，无法读本群总结）"
    row = ctx.store.find_summary(summary_ref)
    if row is None:
        return (
            f"找不到这篇总结：{summary_ref!r}。请用 summaries_in_range 返回的"
            "行首短 id（至少 6 位）。"
        )
    # Same reasoning as `view_image`: the short id is guessable, so the group
    # scope refuses rather than trusting luck.
    if row["group_openid"] != ctx.group_openid:
        return "无权查看其他群的总结。"
    if ctx.doc_reads_used >= int(config.summary.max_doc_reads):
        return (
            f"本次回答读取已有总结已达上限（{config.summary.max_doc_reads} 篇），"
            "请基于已有信息作答。"
        )
    ctx.doc_reads_used += 1
    body = str(row["content"] or "").strip()
    # Reading a document is grounding, not coverage: it must stay out of the
    # envelope the published row will carry.
    ctx.coverage.add("get_summary", 0, envelope=False)
    if len(body) > MAX_TOOL_CHARS:
        body = body[:MAX_TOOL_CHARS] + "\n……（正文过长，此处已截断）"
    return (
        f"[已入库总结｜{_label_of(row['instruction'])}｜"
        f"{_stamp(row['ts_start'])}~{_stamp(row['ts_end'])}｜"
        f"{int(row['message_count'] or 0)} 条]\n{body}"
    )


@tool
async def save_summary(
    content: str,
    *,
    runtime: ToolRuntime[BotContext, dict],
) -> str:
    """把一份**值得长期留存**的总结写入知识库（本群的文档库）。

    只在这一轮的产出确实是一篇文档时调用：用户要的是「这段时间聊了什么」这类
    可独立阅读的总结，或你对已有总结做了真正新增的综合。**只是回答一个事实问题
    （谁说过 X / 有没有提过 Y / 某个细节是什么）时不要调用**，直接回答即可。

    判据由你掌握，但写之前应当先查过 summaries_in_range——同一段时间已经有稿子
    就别再投一篇，除非你确实梳理出了新东西。`content` 是要入库的**完整正文**
    （按话题组织、标注发言人与时间、结尾说明覆盖范围）；调用成功后，回群里只需
    给出要点与覆盖范围，**不要把正文再抄一遍**。
    """
    ctx: BotContext = runtime.context
    if not ctx.group_openid:
        return "（本场景没有群上下文，无法入库）"
    if not ctx.allow_publish:
        return "这个场景不允许写入知识库，请直接回答。"

    text = (content or "").strip()
    floor = int(config.summary.min_publish_chars)
    if len(text) < floor:
        return (
            f"未入库：正文只有 {len(text)} 字，低于一篇文档的下限（{floor} 字）。"
            "如果用户只是问了个问题，请直接回答，不要投稿。"
        )
    if ctx.coverage.is_empty():
        return (
            "未入库：这一轮没有读过任何消息或已有总结，不能凭会话记忆立档。"
            "要投稿请先用 recent_messages / messages_in_range / search_summaries 取数。"
        )
    budget = int(config.summary.max_publish_per_run)
    if ctx.publishes_used >= budget:
        return (
            f"未入库：本次回答已投稿 {ctx.publishes_used} 篇，达到上限（{budget} 篇）。"
            "请把内容合并成一篇，或直接基于已有信息作答。"
        )

    try:
        summary_id = insert_document(
            ctx.store,
            group_openid=ctx.group_openid,
            instruction=ctx.instruction,
            content=text,
            coverage=ctx.coverage.intervals(),
            message_count=ctx.coverage.message_count(),
            requested_by=ctx.requested_by,
            trigger=ctx.trigger,
        )
    except Exception as exc:  # noqa: BLE001 - a failed write must not kill the turn
        logger.warning(
            "save_summary 入库失败",
            extra={"group": ctx.group_openid, "err": repr(exc)[:200]},
        )
        return f"入库失败：{repr(exc)[:120]}。请直接把你的结论回答给群里。"

    ctx.publishes_used += 1
    ctx.published_ids.append(summary_id)
    ctx.published_text = text
    logger.info(
        "总结已由 agent 投稿入库",
        extra={
            "group": ctx.group_openid,
            "trigger": ctx.trigger,
            "summary_id": summary_id,
            "chars": len(text),
            "messages": ctx.coverage.message_count(),
        },
    )
    return f"已入库（{summary_id[:8]}），覆盖 {ctx.coverage.message_count()} 条消息。"


GROUP_TOOLS = [
    current_time,
    recent_messages,
    messages_in_range,
    search_summaries,
    view_image,
    summaries_in_range,
    get_summary,
    save_summary,
]
C2C_TOOLS = [
    current_time,
    search_summaries,
    list_groups,
    messages_across_groups,
    view_image,
]

# Tools that put material into the coverage log — as opposed to `current_time`
# (which only says what time it is) and `view_image` (which reads a picture, not
# messages). Used by `Summarizer` to warn when a run called a data tool yet
# recorded no window.
#
# `summaries_in_range` / `get_summary` **are** listed: they genuinely read stored
# data, and a run that consulted the library must not trip that warning. Their
# entries carry `envelope=False`, so they widen "was this grounded" without
# touching the time window — which is the whole point of the split in `add`.
# `save_summary` is a writer, not a reader, so it is not here.
DATA_TOOLS = frozenset(
    {
        "recent_messages",
        "messages_in_range",
        "search_summaries",
        "messages_across_groups",
        "summaries_in_range",
        "get_summary",
    }
)
