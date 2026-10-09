"""Typed wrapper around the raw `GROUP_MESSAGE_CREATE` event body.

Why not use `botpy.message.GroupMessage`? Because its nested `_User` reads only
`member_openid`, silently dropping `username` and `member_role`; and it never
reads `message_type`, `message_scene`, `msg_elements` or `ark_data` at all. For
a summariser that needs to say *who* said *what*, that is most of the signal.

This module parses the raw `d` dict instead, so nothing is lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# `msg_elements` nests recursively. Merged-forward chats could in principle nest
# arbitrarily deep; cap it so a hostile payload cannot blow the stack.
MAX_ELEMENT_DEPTH = 5

# Voice is not listed: it is decided by `Attachment.is_voice()` before this
# table is consulted, because "no transcription" must render differently from
# "some other attachment".
_KIND_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("image/", "图片"),
    ("video/", "视频"),
    ("file", "文件"),
)


def _parse_ts(value: Any) -> datetime:
    """RFC3339 -> datetime, falling back to 'now' rather than raising.

    Losing a timestamp must never cost us the message itself.
    """
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now()


def _parse_ext(scene: dict) -> dict[str, str]:
    """`message_scene.ext` is a list of ``"key=value"`` strings, not a mapping."""
    ext = scene.get("ext")
    if isinstance(ext, dict):  # tolerate a shape change rather than crashing
        return {str(k): str(v) for k, v in ext.items()}
    out: dict[str, str] = {}
    for item in ext or []:
        if isinstance(item, str) and "=" in item:
            key, _, value = item.partition("=")
            out[key] = value
    return out


@dataclass(slots=True)
class Attachment:
    url: str | None = None
    filename: str | None = None
    content_type: str | None = None
    size: int | None = None
    width: int | None = None
    height: int | None = None
    # Official docs (GROUP_MESSAGE_CREATE, MessageAttachment): `asr_refer_text`
    # 是"语音消息 ASR **参考**结果" — the wording itself promises nothing, so
    # absence is a normal state, hence the explicit no-transcript placeholder in
    # `label()`. The docs also deliver `voice_wav_url` (SILK→WAV conversion by
    # QQ, same `rkey`-signed URL shape as images); deliberately *not* parsed —
    # audio stays out of scope (ROADMAP M1), and the whole payload persists in
    # `raw_json` regardless, so the field is never lost, just unread.
    asr_refer_text: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Attachment":
        return cls(
            url=data.get("url"),
            filename=data.get("filename"),
            content_type=data.get("content_type"),
            size=data.get("size"),
            width=data.get("width"),
            height=data.get("height"),
            asr_refer_text=data.get("asr_refer_text"),
        )

    @classmethod
    def from_object(cls, obj: Any) -> "Attachment":
        """Adapt botpy's `_Attachments` (used by the @-message fallback path)."""
        return cls(
            url=getattr(obj, "url", None),
            filename=getattr(obj, "filename", None),
            content_type=getattr(obj, "content_type", None),
            size=getattr(obj, "size", None),
            width=getattr(obj, "width", None),
            height=getattr(obj, "height", None),
            # botpy 1.2.1's `_Attachments` does not parse `asr_refer_text` at
            # all (verified: the string appears nowhere in the package), so on
            # the @-fallback path a voice message always shows "no
            # transcription". Read it anyway: the moment an SDK release adds
            # the field, transcripts start flowing without touching this file.
            asr_refer_text=getattr(obj, "asr_refer_text", None),
        )

    def is_voice(self) -> bool:
        """Whether this attachment is a voice message.

        Detection is `content_type`-driven, never `message_type`-driven: the
        official docs enumerate `content_type` as `voice` / `image/jpeg` /
        `image/png` / `image/gif` / `video/mp4` / `file` (bare `voice`, despite
        the column being called "MIME 类型"), and their own example — like the
        real corpus — tags an image message `message_type: 0`. Bare `voice` is
        the documented spelling; `audio/…` is accepted defensively because the
        same docs *do* deliver MIME spellings for images on the wire.
        """
        ct = (self.content_type or "").lower()
        return ct.startswith("voice") or ct.startswith("audio")

    def label(self) -> str:
        """Compact stand-in for the attachment, e.g. ``[图片 photo.jpg]``."""
        if self.asr_refer_text:
            return f"[语音转写 {self.asr_refer_text}]"
        if self.is_voice():
            # QQ gave no transcript. The filename is a hex ID carrying no
            # signal, so say plainly "voice, nothing recovered" — a summariser
            # reading `[语音（无转写）]` at least knows someone spoke there,
            # instead of the turn quietly vanishing from the digest.
            return "[语音（无转写）]"
        kind = "附件"
        for prefix, name in _KIND_BY_PREFIX:
            if (self.content_type or "").startswith(prefix):
                kind = name
                break
        return f"[{kind} {self.filename}]" if self.filename else f"[{kind}]"


@dataclass(slots=True)
class Mention:
    """One entry of `mentions`.

    Load-bearing: `GROUP_MESSAGE_CREATE` delivers `content` with the bot's own @
    prefix already stripped ("已去除@机器人的前缀"), so this list is the *only* way
    to tell that the bot was summoned when receiving all messages.
    """

    openid: str | None = None  # the payload calls this field `id`
    username: str | None = None
    is_bot: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "Mention":
        return cls(
            openid=data.get("id") or data.get("member_openid"),
            username=data.get("username"),
            is_bot=bool(data.get("bot")),
        )


@dataclass(slots=True)
class MsgElement:
    """One entry of `msg_elements` (quote / merged-forward / parallel message)."""

    message_type: int = 0
    content: str | None = None
    msg_idx: str | None = None
    author_name: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    children: list["MsgElement"] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict, depth: int = 0) -> "MsgElement":
        children: list[MsgElement] = []
        if depth < MAX_ELEMENT_DEPTH:
            children = [
                cls.from_dict(child, depth + 1)
                for child in data.get("msg_elements") or []
                if isinstance(child, dict)
            ]
        author = data.get("author") or {}
        return cls(
            message_type=data.get("message_type") or 0,
            content=data.get("content"),
            msg_idx=data.get("msg_idx"),
            author_name=author.get("username") if isinstance(author, dict) else None,
            attachments=[
                Attachment.from_dict(a)
                for a in data.get("attachments") or []
                if isinstance(a, dict)
            ],
            children=children,
        )

    def render(self) -> str:
        parts: list[str] = []
        if self.content and self.content.strip():
            parts.append(self.content.strip())
        parts.extend(att.label() for att in self.attachments)
        parts.extend(child.render() for child in self.children)
        return " ".join(p for p in parts if p)


@dataclass(slots=True)
class GroupMessageRecord:
    message_id: str
    group_openid: str
    content: str
    ts: datetime
    message_type: int = 0
    event_id: str | None = None
    author_openid: str | None = None
    author_name: str | None = None
    member_role: str | None = None
    msg_idx: str | None = None
    ref_msg_idx: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    elements: list[MsgElement] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_payload(cls, frame: dict) -> "GroupMessageRecord":
        """Build from a whole WebSocket frame (the event body lives under `d`)."""
        body = frame.get("d") or {}
        author = body.get("author") or {}
        ext = _parse_ext(body.get("message_scene") or {})
        msg_idx = ext.get("msg_idx")
        ref_msg_idx = ext.get("ref_msg_idx")
        return cls(
            message_id=str(body.get("id") or ""),
            event_id=frame.get("id"),
            group_openid=body.get("group_openid") or "",
            author_openid=author.get("member_openid"),
            author_name=author.get("username"),
            member_role=author.get("member_role"),
            content=body.get("content") or "",
            message_type=body.get("message_type") or 0,
            ts=_parse_ts(body.get("timestamp")),
            msg_idx=msg_idx,
            ref_msg_idx=ref_msg_idx,
            attachments=[
                Attachment.from_dict(a)
                for a in body.get("attachments") or []
                if isinstance(a, dict)
            ],
            elements=[
                MsgElement.from_dict(el)
                for el in body.get("msg_elements") or []
                if isinstance(el, dict)
            ],
            mentions=[
                Mention.from_dict(m)
                for m in body.get("mentions") or []
                if isinstance(m, dict)
            ],
            raw=body,
        )

    @classmethod
    def from_at_message(cls, message: Any) -> "GroupMessageRecord":
        """Fallback for bots without the "receive all messages" permission.

        Only `GROUP_AT_MESSAGE_CREATE` arrives in that mode, and botpy's
        `GroupMessage` cannot give us a nickname — hence `author_name=None`.
        `mentions` is left empty because botpy's `_User` drops the `bot` flag;
        the event itself firing already proves the bot was addressed.
        """
        return cls(
            message_id=str(getattr(message, "id", "") or ""),
            event_id=getattr(message, "event_id", None),
            group_openid=getattr(message, "group_openid", "") or "",
            author_openid=getattr(getattr(message, "author", None), "member_openid", None),
            author_name=None,
            member_role=None,
            content=getattr(message, "content", "") or "",
            message_type=0,
            ts=_parse_ts(getattr(message, "timestamp", None)),
            attachments=[
                Attachment.from_object(a)
                for a in getattr(message, "attachments", None) or []
            ],
            raw={},
        )

    def mentions_bot(self) -> bool:
        """Whether this message summoned a robot — i.e. the summarise trigger.

        Checks the `bot` flag rather than comparing ids: the mention `id` is an
        OpenID, not the appid, so there is nothing to compare against. With more
        than one bot in the group this can also fire on *another* bot's @; the
        worst case is one wasted (still cached) summarisation.
        """
        return any(m.is_bot for m in self.mentions)

    # ---- rendering --------------------------------------------------------

    def display_name(self) -> str:
        return self.author_name or (
            f"成员{self.author_openid[-4:]}" if self.author_openid else "未知成员"
        )

    def body(self) -> str:
        """Full text of the message, including media and quoted/forwarded parts.

        For `message_type` 102 (merged forward) and 103 (quote) the top-level
        `content` is usually blank and the real text sits in `msg_elements` —
        without this, those messages would reach the LLM as empty lines.
        """
        parts: list[str] = []
        text = (self.content or "").strip()
        if text:
            parts.append(text)
        parts.extend(att.label() for att in self.attachments)
        for element in self.elements:
            rendered = element.render()
            # Avoid echoing the same text twice when an element mirrors the body.
            if rendered and rendered not in parts:
                parts.append(rendered)
        return " ".join(p for p in parts if p)

    def to_line(self) -> str:
        """One prompt/storage line: ``HH:MM 昵称: 正文``."""
        return f"{self.ts.strftime('%H:%M')} {self.display_name()}: {self.body()}"
