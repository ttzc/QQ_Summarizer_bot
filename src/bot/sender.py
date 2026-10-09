"""Outbound replies to a group or a private chat, with QQ's rules baked in.

Three platform rules shape this module (all from the official send-message docs):

1. A passive reply must carry the inbound message's `msg_id` (or `event_id`) and
   is only valid for **5 minutes** after that message arrived.
2. **At most 5 replies** per inbound message, and `msg_seq` must differ each time
   — resending the same `msg_id` + `msg_seq` *fails*.
3. Sending at all requires the websocket gateway to be online, so this only ever
   runs inside the long-lived bot process.

A summary can easily exceed one message, hence the chunking.

Group and C2C share this path because the two endpoints are otherwise identical:
same route shape, same `payload = locals()` body, same passive window. The one
difference that actually bites is the keyword name of the target — groups take
`group_openid`, users take `openid` (`botpy/api.py:1380` vs `:1426`), which is why
there is a single dispatch point rather than two copies of this function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from botpy.errors import SequenceNumberError

from src.config import config
from src.logger import setup_logger

logger = setup_logger("qqbot.bot.sender")

KIND_GROUP = "group"
KIND_C2C = "c2c"

# Official hard limit on replies per inbound message. A constant, not config:
# exceeding it is not a tuning choice, it is an API error. Assumed to apply to
# C2C as well — only the 5-minute window is documented on `post_c2c_message`'s
# docstring; an over-send fails as a `SequenceNumberError` and is handled below,
# so the assumption is not load-bearing.
MAX_PASSIVE_REPLIES = 5

TRUNCATION_NOTICE = "……（内容过长，已省略后续部分，可缩小范围后重试）"


@dataclass(slots=True)
class SendResult:
    sent: int = 0
    truncated: bool = False
    active: bool = False  # sent as an active message, i.e. quota was consumed
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.sent > 0


def plan_chunks(text: str, limit: int, max_chunks: int) -> tuple[list[str], bool]:
    """Split `text` into at most `max_chunks` pieces of at most `limit` chars.

    Returns `(chunks, truncated)`. Paragraph breaks are respected where possible;
    a single paragraph longer than the limit is hard-split, since an oversized
    payload would simply be rejected by the API.
    """
    if max_chunks < 1:
        raise ValueError("max_chunks must be >= 1")

    atoms: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        while len(block) > limit:
            atoms.append(block[:limit])
            block = block[limit:]
        if block:
            atoms.append(block)

    chunks: list[str] = []
    current = ""
    for atom in atoms:
        candidate = f"{current}\n{atom}" if current else atom
        if len(candidate) <= limit:
            current = candidate
        else:
            chunks.append(current)
            current = atom
    if current:
        chunks.append(current)

    if len(chunks) <= max_chunks:
        return chunks, False

    # Drop what does not fit and say so, rather than silently losing it.
    return chunks[: max_chunks - 1] + [TRUNCATION_NOTICE], True


async def _post(
    api,
    kind: str,
    target_id: str,
    content: str,
    *,
    msg_id: str | None,
    msg_seq: int,
):
    """Send one chunk to one target. The `kind` branch is the whole difference.

    `msg_seq` is annotated `int` on the group signature and `str` on the C2C one,
    which is a typo upstream — botpy never validates it, and an int is sent
    either way.
    """
    if kind == KIND_C2C:
        return await api.post_c2c_message(
            openid=target_id,
            msg_type=0,  # 0 = plain text
            content=content,
            msg_id=msg_id,
            msg_seq=msg_seq,
        )
    return await api.post_group_message(
        group_openid=target_id,
        msg_type=0,  # 0 = plain text; markdown templates are deprecated
        content=content,
        msg_id=msg_id,
        msg_seq=msg_seq,
    )


async def reply_chunked(
    api,
    target_id: str,
    reply_to_msg_id: str | None,
    text: str,
    *,
    kind: str = KIND_GROUP,
    elapsed_s: float = 0.0,
) -> SendResult:
    """Send `text` to a group or a private chat, passively when still in window.

    `elapsed_s` is how long ago the inbound message arrived. For a group, past
    `summary.passive_reply_deadline_s` the reply is sent as an *active* message
    instead: it consumes quota, but a summary that took too long to produce
    should still be delivered rather than rejected for an expired window.

    That downgrade is deliberately **not** applied to C2C. It was designed for
    groups, and C2C active messages follow their own rules — an attempt may just
    be rejected, which the user experiences as silence anyway. Failing loudly and
    logging is the more honest outcome.
    """
    result = SendResult()
    text = (text or "").strip()
    if not text:
        result.error = "empty text"
        logger.warning("发送被跳过：内容为空", extra={"kind": kind, "target": target_id})
        return result

    if kind not in (KIND_GROUP, KIND_C2C):
        result.error = f"unknown kind {kind!r}"
        logger.error("发送被跳过：未知目标类型", extra={"kind": kind})
        return result

    limit = config.summary.max_reply_chars
    chunks, result.truncated = plan_chunks(text, limit, config.summary.max_replies)

    passive = reply_to_msg_id is not None
    if passive and elapsed_s >= config.summary.passive_reply_deadline_s:
        if kind == KIND_C2C:
            result.error = "passive window expired before the reply was ready"
            logger.error(
                "私聊被动回复窗口已过期，放弃发送",
                extra={
                    "target": target_id,
                    "elapsed_s": round(elapsed_s, 1),
                    "deadline_s": config.summary.passive_reply_deadline_s,
                },
            )
            return result
        passive = False
        result.active = True
        logger.warning(
            "被动回复窗口即将/已经过期，改用主动消息（消耗配额）",
            extra={
                "target": target_id,
                "elapsed_s": round(elapsed_s, 1),
                "deadline_s": config.summary.passive_reply_deadline_s,
            },
        )

    for index, chunk in enumerate(chunks, start=1):
        try:
            sent = await _post(
                api,
                kind,
                target_id,
                chunk,
                msg_id=reply_to_msg_id if passive else None,
                msg_seq=index,
            )
        except SequenceNumberError as exc:
            # 429. Either the target is rate-limited or this msg_id+msg_seq pair
            # was already used; retrying would replay the same pair and fail
            # again, so stop here.
            result.error = f"sequence/rate-limit rejected chunk {index}: {exc}"
            logger.error(
                "发送被拒（429：频率限制或 msg_id+msg_seq 重复）",
                extra={"kind": kind, "target": target_id, "chunk": index, "seq": index},
            )
            break
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill the task
            result.error = f"{type(exc).__name__} on chunk {index}: {exc}"
            logger.exception(
                "发送失败",
                extra={"kind": kind, "target": target_id, "chunk": index},
            )
            break

        # botpy's http layer swallows timeouts and connection resets and returns
        # None instead of raising (http.py:191-195), so None must be treated as
        # a failure here rather than mistaken for a successful send.
        if sent is None:
            result.error = f"chunk {index} returned None (timeout or connection reset)"
            logger.error(
                "发送结果为空（botpy 在超时/连接重置时静默返回 None）",
                extra={"kind": kind, "target": target_id, "chunk": index},
            )
            break

        result.sent += 1

    logger.info(
        "回复已发送",
        extra={
            "kind": kind,
            "target": target_id,
            "chunks": result.sent,
            "planned": len(chunks),
            "truncated": result.truncated,
            "active": result.active,
            "error": result.error,
        },
    )
    return result
