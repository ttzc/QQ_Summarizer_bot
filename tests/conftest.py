"""Shared fixtures and fakes for the offline suite.

Isolation policy is the opposite of heavier projects': every test builds its
own SQLite/Chroma under pytest's `tmp_path`, so there is no global state to
reset — the only shared surface is `config`, mutated via `monkeypatch.setattr`
so restoration is automatic. Models, embeddings and the QQ API are all faked;
only Chroma and SQLite are real.

Files mirror the major features: events parsing, the SQLite store, the sender,
the RAG index, the two agents' tool boundary, and the client (event seam +
auto-summary trigger).
"""

from __future__ import annotations

import asyncio

import pytest
from botpy.errors import SequenceNumberError

from src.agent.summarizer import SummaryResult
from src.agent.tools import CoverageLog
from src.store.sql_store import SQLStore


def frame(body: dict, event_id: str = "EVENT_0") -> dict:
    """Wrap an event body the way the gateway does."""
    return {"id": event_id, "op": 0, "s": 1, "t": "GROUP_MESSAGE_CREATE", "d": body}


# --- fixtures, mirroring the official event page (updated 2026-09-16) --------

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

# Shape taken from the official GROUP_MESSAGE_CREATE docs (MessageAttachment
# table, updated 2026-09-16): voice attachments carry the *bare*
# `content_type: "voice"` plus `asr_refer_text` ("ASR 参考结果" — non-committal
# by name) and `voice_wav_url` (QQ 已做 SILK→WAV 转换). There is no voice-specific
# `message_type` — the docs' own image example is `message_type: 0`, matching
# the real corpus — so detection must read `content_type`. `voice_wav_url` is
# deliberately not parsed (audio out of scope, ROADMAP M1); it still survives
# in `raw_json`, and the assertion below pins that ignoring it changes nothing.
VOICE_MESSAGE = {
    "id": "ROBOT1.0_voice",
    "author": {
        "id": "U1",
        "username": "小明",
        "bot": False,
        "member_openid": "M_ming",
        "member_role": "member",
    },
    "content": " ",
    "group_openid": "G_demo",
    "message_type": 0,
    "timestamp": "2026-07-21T08:30:00+08:00",
    "message_scene": {"source": "default", "ext": ["msg_idx=REFIDX_voice=="]},
    "attachments": [
        {
            "content_type": "voice",
            "filename": "6A3051F3A1B2C3D4.silk",
            "url": "https://multimedia.nt.qq.com.cn/download?appid=xxx&fileid=xxx&rkey=xxx&spec=0",
            "voice_wav_url": "https://multimedia.nt.qq.com.cn/download?appid=xxx&fileid=wav&rkey=xxx&spec=0",
            "size": 20480,
            "asr_refer_text": "明天下午三点记得交周报",
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


# ---- shared fakes -----------------------------------------------------------


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


class FakeSummarizer:
    """Stands in for `Summarizer`, and for what the agent's publishing now does.

    Two separate jobs, because the storage decision moved: an `@` answer is no
    longer stored by the bot, it is published *during the run* by the
    `save_summary` tool. So this fake both returns an answer **and** writes the
    row that tool would have written, through the same `insert_document` the tool
    calls. Tests then still assert against real rows — a fake that merely set
    `published_ids` would make every storage assertion vacuous.

    `publish=False` stands in for the model declining to publish (a question, not
    a document), which is now the interesting second case.

    The text is deliberately longer than the code-side floors and carries a
    non-empty coverage log, so the auto-summary *fallback* also has something
    worth storing when the model does not publish.
    """

    TEXT = (
        "摘要：群里讨论了明天的会议，决定推迟到下午三点。"
        "小红负责整理会议材料，小刚确认了会议室。"
    )

    def __init__(self, store=None, publish: bool = True):
        self.store = store
        self.publish = publish and store is not None
        self.calls: list[tuple[str, str]] = []
        self.private_calls: list[tuple[str, str]] = []
        self.kwargs: list[dict] = []

    @staticmethod
    def _result(tool: str, rows: int) -> SummaryResult:
        coverage = CoverageLog()
        coverage.add(
            tool, rows, "2026-07-21T08:00:00+08:00", "2026-07-21T09:00:00+08:00"
        )
        return SummaryResult(text=FakeSummarizer.TEXT, coverage=coverage)

    async def summarize_group(
        self, group_openid: str, instruction: str, **kw
    ) -> SummaryResult:
        self.calls.append((group_openid, instruction))
        # Keep what the bot passed as provenance: a test asserting that
        # `requested_by` / `trigger` reach the row would otherwise have nothing
        # to read back, since the fake is what writes the row.
        self.kwargs.append(kw)
        result = self._result("recent_messages", rows=5)
        if self.publish:
            from src.agent.publish import insert_document

            summary_id = insert_document(
                self.store,
                group_openid=group_openid,
                instruction=instruction,
                content=self.TEXT,
                coverage=result.coverage.intervals(),
                message_count=result.coverage.message_count(),
                requested_by=kw.get("requested_by"),
                trigger=kw.get("trigger", "at"),
            )
            result.published_ids = [summary_id]
            result.published_text = self.TEXT
        return result

    async def answer_private(self, user_openid: str, instruction: str) -> SummaryResult:
        self.private_calls.append((user_openid, instruction))
        # Private answers publish nothing, ever — no tool for it, no owning group.
        return self._result("search_summaries", rows=0)

    def group_busy(self, group_openid: str) -> bool:
        """Never busy: this fake runs no lock. `Summarizer.group_busy`'s contract."""
        return False


class ExplodingSummarizer:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def summarize_group(
        self, group_openid: str, instruction: str, **kw
    ) -> SummaryResult:
        self.calls.append((group_openid, instruction))
        raise RuntimeError("LLM 挂了")

    async def answer_private(self, user_openid: str, instruction: str) -> SummaryResult:
        raise RuntimeError("LLM 挂了")

    def group_busy(self, group_openid: str) -> bool:
        return False


class FakeEmbeddings:
    """Deterministic bag-of-characters embedding.

    Not semantic, but it rewards token overlap, which is enough to verify that
    retrieval returns the *right* message first and that the group filter holds.

    Two details are load-bearing, and getting either wrong makes the suite
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


# ---- shared fixtures --------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    """A fresh SQLite store per test; closed on teardown."""
    s = SQLStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture()
def fake_bot_login(monkeypatch):
    """Replace the network half of `botpy.Client._bot_login` with local session
    construction — the same stand-in the old suite patched by hand.

    `Client.__init__` reads `asyncio.get_event_loop()`, so construct sessions
    from inside the running loop (every consumer here is an async test).
    monkeypatch restores the original automatically.
    """
    import botpy

    async def _fake(self, token):  # noqa: ARG001 - signature must match Client._bot_login
        self._connection = botpy.connection.ConnectionSession(
            max_async=1,
            connect=self.bot_connect,
            dispatch=self.ws_dispatch,
            loop=asyncio.get_running_loop(),
            api=self.api,
        )

    monkeypatch.setattr(botpy.Client, "_bot_login", _fake)
