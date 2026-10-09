"""`botpy.Client` subclass filling the two gaps the SDK leaves open.

Gap 1 — botpy 1.2.1 does not know `GROUP_MESSAGE_CREATE` at all. Its event table
(`ConnectionState.parsers`, built by scanning `parse_*` methods) has no entry, so
`gateway.on_message` looks the event name up, misses, and logs
`_parser unknown event ...` — the event is simply dropped. Since the whole point
of this bot is to see *all* group traffic, we register the missing parser.

Gap 2 — `botpy.message.GroupMessage` reads only `member_openid` out of `author`
and ignores `message_type` / `message_scene` / `msg_elements` / `mentions`
entirely, so a summariser built on it loses speaker names and quoted content.
`src/bot/events.py` parses the raw payload instead.

Registration point: `_bot_login` builds the `ConnectionSession`, so overriding it
and adding a key afterwards is enough — `Client._bot_login` runs before any
websocket is opened, and `connection.py:40` makes `ConnectionSession.parser` the
very dict `gateway.py` reads.

**Private chat needs neither patch.** botpy already implements
`parse_c2c_message_create` (`connection.py:210-211`), and the `public_messages`
intent covers C2C as well as groups, so `on_c2c_message_create` is delivered
normally. Private messages are never written to `group_messages` — that table's
`group_openid` is `NOT NULL` and a C2C author carries `user_openid` instead, so
the group parsing path simply does not apply.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol

import botpy
from botpy.flags import Intents

from src.agent.summarizer import SummaryResult, store_summary
from src.bot.events import GroupMessageRecord
from src.bot.sender import KIND_C2C, reply_chunked
from src.config import config
from src.logger import setup_logger
from src.store.sql_store import SQLStore

logger = setup_logger("qqbot.bot.client")

FALLBACK_REPLY = "抱歉，处理这条指令时出错了，请稍后再试。"


class SummarizerLike(Protocol):
    """What the client needs from the agent layer (`src/agent/summarizer.py`)."""

    async def summarize_group(
        self, group_openid: str, instruction: str
    ) -> SummaryResult: ...

    async def answer_private(
        self, user_openid: str, instruction: str
    ) -> SummaryResult: ...

    def group_busy(self, group_openid: str) -> bool: ...


class SummarizerClient(botpy.Client):
    def __init__(
        self,
        *,
        store: SQLStore,
        summarizer: SummarizerLike,
        wake_indexer: Callable[[], None] | None = None,
        wake_media: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        # public_messages (1<<25) is the intent that carries group/C2C events,
        # GROUP_MESSAGE_CREATE included. There is no separate flag for it.
        super().__init__(
            Intents(public_messages=True),
            is_sandbox=config.qq.is_sandbox,
            **kwargs,
        )
        self._store = store
        self._summarizer = summarizer
        self._wake_indexer = wake_indexer
        # 图片落盘队列的拨铃（`MediaWorker.wake`）。同步段里它只是 set 一个
        # Event——下载与视觉调用全部活在后台，回调里一分钱网络花销都没有。
        self._wake_media = wake_media
        # `group_openid` → `time.monotonic()` of the last auto-summary *attempt*.
        # In memory only: a restart that finds the backlog still over the
        # threshold simply summarises once more, which costs one document.
        self._last_auto: dict[str, float] = {}

    # ---- botpy seam -------------------------------------------------------

    async def _bot_login(self, token) -> None:
        await super()._bot_login(token)
        self._connection.parser["group_message_create"] = self._parse_group_message_create
        logger.info(
            "已注册 group_message_create 解析器（botpy 原生不支持此事件）"
        )

    def _parse_group_message_create(self, payload: dict) -> None:
        """Adapt the raw frame into a `GroupMessageRecord`, then dispatch.

        Called *synchronously* from the websocket read loop and **not** wrapped
        in a try by botpy (`gateway.py:99`), so anything raised here escapes all
        the way to `bot_connect`, triggering `on_error` and a full reconnect.
        A single malformed payload must therefore never raise.
        """
        try:
            record = GroupMessageRecord.from_payload(payload)
        except Exception:  # noqa: BLE001 - see docstring: nothing may escape
            logger.exception("解析 GROUP_MESSAGE_CREATE 失败，已丢弃该条")
            return
        self.ws_dispatch("group_message_create", record)

    # ---- lifecycle --------------------------------------------------------

    async def on_ready(self) -> None:
        logger.info("机器人已上线，开始接收群消息", extra={"appid": config.qq.appid})
        if config.c2c.enabled and not config.c2c.allowlist:
            # Warn, not info: this is the setting with the largest blast radius
            # in the project — every group's *raw messages*, not just its
            # summaries, readable by anyone who can DM the bot.
            logger.warning(
                "私聊检索面向所有人开放（c2c.allowlist 为空），任何能私聊机器人的"
                "用户都可检索所有群的总结与聊天原文"
            )

    async def on_error(self, event_method: str, *args: Any, **kwargs: Any) -> None:
        """botpy defaults to `traceback.print_exc()`, which bypasses our logger.

        Always reached from inside an `except` block in the SDK, so
        `logger.exception` picks up the live traceback.
        """
        try:
            logger.exception("事件处理出错", extra={"event": event_method})
        except Exception:  # noqa: BLE001 - must never raise
            pass

    # ---- inbound ----------------------------------------------------------

    async def on_group_message_create(self, record: GroupMessageRecord) -> None:
        """Every message in every group the bot is in (privileged event).

        This is the only path that sees the whole group, so it is also the only
        one that runs the message-count auto-summary. Awaiting it here does not
        stall anything: botpy schedules every event as its own task
        (`client.py:250` in the SDK), so the websocket read loop keeps running.
        """
        if not await self._ingest(record):
            return  # duplicate event — the insert count is the trigger guard
        if record.mentions_bot():
            await self._respond(record)
            return
        await self._maybe_auto_summarize(record.group_openid)

    async def on_group_at_message_create(self, message: Any) -> None:
        """Fallback for deployments without the "receive all messages" privilege.

        Only @-messages arrive in that mode, and botpy cannot supply a nickname
        (see `GroupMessageRecord.from_at_message`).
        """
        record = GroupMessageRecord.from_at_message(message)
        if not await self._ingest(record):
            return
        await self._respond(record)

    async def on_c2c_message_create(self, message: Any) -> None:
        """Private messages, answered from summaries across all groups."""
        user_openid = _user_openid_of(message)
        if not user_openid:
            logger.warning("私聊消息缺少 user_openid，已丢弃")
            return
        if not self._c2c_allowed(user_openid):
            # Checked before the agent runs so a disallowed chat costs nothing.
            logger.warning("私聊被拒绝（allowlist 或开关）", extra={"user": user_openid})
            return

        content = (getattr(message, "content", "") or "").strip()
        if not content:
            return
        await self._respond_private(user_openid, content, getattr(message, "id", None))

    async def on_group_add_robot(self, event: Any) -> None:
        logger.info("机器人被加入群聊", extra={"group": event.group_openid})

    async def on_group_del_robot(self, event: Any) -> None:
        logger.info("机器人被移出群聊", extra={"group": event.group_openid})

    # ---- helpers ----------------------------------------------------------

    def _c2c_allowed(self, user_openid: str) -> bool:
        """An empty allowlist means everyone. See the warning in `on_ready`."""
        if not config.c2c.enabled:
            return False
        allowlist = config.c2c.allowlist
        return not allowlist or user_openid in allowlist

    async def _ingest(self, record: GroupMessageRecord) -> bool:
        """Persist a message. Returns whether it was new.

        The insert count doubles as the trigger guard: when the all-messages
        privilege is on, the same @-message can arrive as *both* events, and the
        `message_id` primary key makes the second one a no-op — so the bot
        replies exactly once.
        """
        if not record.message_id or not record.group_openid:
            logger.warning(
                "消息缺少 id 或 group_openid，已丢弃",
                extra={"message_id": record.message_id, "group": record.group_openid},
            )
            return False

        try:
            inserted = self._store.insert_messages([record])
        except Exception:  # noqa: BLE001 - storage failure must not kill the task
            logger.exception(
                "消息入库失败", extra={"message_id": record.message_id}
            )
            return False

        if not inserted:
            logger.debug(
                "重复消息，跳过", extra={"message_id": record.message_id}
            )
            return False

        # No `wake_indexer()` here, deliberately. Raw messages are no longer
        # embedded — summaries are — so waking on every message would have the
        # indexer spin on traffic while finding nothing, and the one insert that
        # does need indexing (a new summary) would go unwoken. See `_store_summary`.
        #
        # The *media* wake is the mirror image: raw messages ARE its backlog, so
        # a message carrying an image must ring it — but only image messages do,
        # so ordinary traffic never touches the queue.
        if self._wake_media is not None and any(
            att.is_image() for att in record.attachments
        ):
            self._wake_media()
        return True

    def _store_summary(
        self,
        *,
        group_openid: str,
        instruction: str,
        requested_by: str | None,
        result: SummaryResult,
        trigger: str = "at",
    ) -> str | None:
        """Persist a summary and wake the indexer. Never raises.

        Returns the new `summary_id`, or `None` when the answer was not a
        document — which the auto-summary path also reads as "nothing worth
        pushing to the group".
        """
        try:
            summary_id = store_summary(
                self._store,
                group_openid=group_openid,
                instruction=instruction,
                requested_by=requested_by,
                result=result,
                trigger=trigger,
            )
        except Exception:  # noqa: BLE001 - a storage failure must not lose the reply
            logger.exception("总结入库失败", extra={"group": group_openid})
            return None

        if summary_id is None:
            logger.debug(
                "本次回答不构成文档，不入库",
                extra={"group": group_openid, "chars": len(result.text)},
            )
            return None

        if self._wake_indexer is not None:
            self._wake_indexer()
        logger.info(
            "总结已入库",
            extra={
                "group": group_openid,
                "trigger": trigger,
                "chars": len(result.text),
                "messages": result.coverage.message_count(),
            },
        )
        return summary_id

    # ---- auto-summary -----------------------------------------------------

    async def _maybe_auto_summarize(self, group_openid: str) -> None:
        """Summarise unprompted once a group has talked enough.

        Everything before the final `await` is synchronous and free — this runs
        on *every* inbound message of every group, so it must not reach for the
        network or the database until it has decided to.
        """
        cfg = config.auto_summary
        if not cfg.enabled:
            return
        if cfg.groups and group_openid not in cfg.groups:
            return

        now = time.monotonic()
        last = self._last_auto.get(group_openid)
        if last is not None and now - last < cfg.cooldown_s:
            return

        if self._summarizer.group_busy(group_openid):
            # Skip rather than queue: a group that keeps talking through a slow
            # summary should not stack a second one up behind it.
            logger.debug("本群已有总结在跑，跳过自动总结", extra={"group": group_openid})
            return

        needed = max(1, int(cfg.min_messages))
        try:
            pending = self._store.messages_since_last_summary(group_openid)
        except Exception:  # noqa: BLE001 - counting must not kill the task
            logger.exception("统计未总结消息数失败", extra={"group": group_openid})
            return
        if pending < needed:
            return

        # Written *before* the attempt, so a failed summary cools down too.
        # Otherwise a gateway outage would retry on the very next message, and
        # keep retrying, forever.
        self._last_auto[group_openid] = now
        logger.info(
            "触发自动总结",
            extra={"group": group_openid, "pending": pending, "threshold": needed},
        )
        await self._auto_summarize(group_openid)

    async def _auto_summarize(self, group_openid: str) -> None:
        """Run one auto-summary. Never raises; the cooldown is already recorded."""
        cfg = config.auto_summary
        try:
            result = await self._summarizer.summarize_group(
                group_openid, cfg.instruction
            )
        except Exception:  # noqa: BLE001 - must not kill the event task
            logger.exception("自动总结失败", extra={"group": group_openid})
            return

        # Stored first, and regardless of `notify`: whether the group gets a copy
        # does not change whether the answer is a document, and a failed send
        # would otherwise lose it for good.
        summary_id = self._store_summary(
            group_openid=group_openid,
            instruction=cfg.instruction,
            requested_by=None,
            result=result,
            trigger="auto",
        )
        if summary_id is None or not cfg.notify:
            # Either it was not a document (nothing worth pushing) or the group
            # was not asked for a copy. Both end here.
            return

        # `msg_id=None` makes this an *active* message: it consumes quota (20/min
        # per group) and needs the owner to have enabled bot-initiated pushes.
        # The prefix marks it as unprompted in the chat — the stored document is
        # the clean copy and stays unprefixed.
        await reply_chunked(
            self.api,
            group_openid,
            None,
            f"〔自动总结〕\n{result.text}",
        )

    async def _respond(self, record: GroupMessageRecord) -> None:
        received_at = time.monotonic()
        instruction = record.body()
        try:
            result = await self._summarizer.summarize_group(
                record.group_openid, instruction
            )
            text = result.text
        except Exception:  # noqa: BLE001 - answer the group even when the agent dies
            logger.exception("生成摘要失败", extra={"group": record.group_openid})
            text = FALLBACK_REPLY
            result = None

        if result is not None:
            # Stored before the send, and independently of it: whether the group
            # received the reply does not change whether the answer is a document,
            # and a 429 or a `None` return would otherwise lose it for good.
            self._store_summary(
                group_openid=record.group_openid,
                instruction=instruction,
                requested_by=record.author_openid,
                result=result,
                trigger="at",  # someone summoned the bot; see `_auto_summarize`
            )

        # A summary can take a while; `reply_chunked` downgrades to an active
        # message if the 5-minute passive window has run out.
        await reply_chunked(
            self.api,
            record.group_openid,
            record.message_id,
            text,
            elapsed_s=time.monotonic() - received_at,
        )

    async def _respond_private(
        self, user_openid: str, instruction: str, reply_to_msg_id: str | None
    ) -> None:
        received_at = time.monotonic()
        try:
            result = await self._summarizer.answer_private(user_openid, instruction)
            text = result.text
        except Exception:  # noqa: BLE001 - always answer the user
            logger.exception("私聊回答失败", extra={"user": user_openid})
            text = FALLBACK_REPLY

        # Private answers are not stored. They are derived from summaries, so
        # keeping them would build summaries of summaries, and they have no
        # single owning group to file them under.
        await reply_chunked(
            self.api,
            user_openid,
            reply_to_msg_id,
            text,
            kind=KIND_C2C,
            elapsed_s=time.monotonic() - received_at,
        )


def _user_openid_of(message: Any) -> str | None:
    """`C2CMessage.author.user_openid`, defensively — botpy gives no accessor."""
    author = getattr(message, "author", None)
    openid = getattr(author, "user_openid", None) if author is not None else None
    return str(openid) if openid else None
