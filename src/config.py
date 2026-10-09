"""Application configuration.

`config.toml` holds everything that is not a secret — model names, gateway URLs,
thresholds — and refers to `.env` only for credentials, via `${VAR}`.

Load order matters: `load_dotenv()` runs at import time, *before* the config is
parsed, so those placeholders resolve.

Two requirements shape this:

1. `config.toml` is resolved relative to the project root, not the CWD, so the
   bot keeps working no matter which directory it is launched from.
2. A field that is effectively unset — env var missing, or the value left empty
   on purpose — is exposed as `None` via the `resolved_*` properties instead of
   leaking the literal ``"${VAR}"`` (or an empty string) into API clients, where
   it would fail much later as a 401 or an unknown-model error.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = PROJECT_ROOT / "config.toml"

_PLACEHOLDER_RE = re.compile(r"\$\{(\w+)\}")


def _expand(value: str) -> str:
    def _replacer(m: re.Match[str]) -> str:
        # Keep the placeholder intact when the env var is missing, so the gap
        # surfaces via `resolved_*` rather than crashing at import time.
        return os.getenv(m.group(1), m.group(0))

    return _PLACEHOLDER_RE.sub(_replacer, value)


def _expand_dict(raw: dict) -> dict:
    out: dict = {}
    for key, value in raw.items():
        if isinstance(value, str):
            out[key] = _expand(value)
        elif isinstance(value, dict):
            out[key] = _expand_dict(value)
        else:
            out[key] = value
    return out


def _resolve(value: str) -> str | None:
    """A configured string, or `None` when it is effectively unset.

    Three ways a field can be blank, all collapsed here: `${VAR}` was never
    substituted (env var missing), the value was left empty in `config.toml`, or
    it is whitespace. Callers read `None` as "fall back to the client's default",
    which is why an empty `base_url` means the provider's own endpoint rather
    than the empty string being sent as a URL.
    """
    text = value.strip()
    return None if not text or text.startswith("${") else text


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class QQConfig(BaseModel):
    appid: str = "${QQ_APPID}"
    secret: str = "${QQ_SECRET}"
    is_sandbox: bool = False

    @property
    def resolved_appid(self) -> str | None:
        return _resolve(self.appid)

    @property
    def resolved_secret(self) -> str | None:
        return _resolve(self.secret)


class LLMConfig(BaseModel):
    """Any OpenAI-compatible endpoint — no provider is hard-coded.

    `model` and `base_url` are written in `config.toml`; only `api_key` comes
    from `.env`. An empty `base_url` falls back to the official OpenAI endpoint.
    """

    model: str = ""
    base_url: str = ""
    api_key: str = "${LLM_API_KEY}"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: int = 60

    @property
    def resolved_model(self) -> str | None:
        return _resolve(self.model)

    @property
    def resolved_base_url(self) -> str | None:
        return _resolve(self.base_url)

    @property
    def resolved_api_key(self) -> str | None:
        return _resolve(self.api_key)


class EmbeddingConfig(BaseModel):
    """Kept separate from `llm` so the embedding gateway can differ."""

    model: str = ""
    base_url: str = ""
    api_key: str = "${EMBED_API_KEY}"
    # Conservative default; `embedding_client.embedding_batch_size()` clamps it
    # to `MAX_EMBED_BATCH`. Raise both if the gateway accepts larger batches.
    batch_size: int = 25

    @property
    def resolved_model(self) -> str | None:
        return _resolve(self.model)

    @property
    def resolved_base_url(self) -> str | None:
        return _resolve(self.base_url)

    @property
    def resolved_api_key(self) -> str | None:
        return _resolve(self.api_key)


class StoreConfig(BaseModel):
    sqlite_path: str = "data/qqbot.db"
    chroma_path: str = "data/chroma_db"
    # The collection holds summary documents, not messages. Renaming it was also
    # the migration: an existing `group_messages` collection holds raw-message
    # vectors that the unfiltered private-chat query would happily return.
    chroma_collection: str = "summaries"

    @property
    def sqlite_file(self) -> Path:
        return _resolve_path(self.sqlite_path)

    @property
    def chroma_dir(self) -> Path:
        return _resolve_path(self.chroma_path)


class SummaryConfig(BaseModel):
    default_recent_n: int = 200
    max_reply_chars: int = 1500
    # Hard API limit: one inbound message may be replied to at most 5 times.
    max_replies: int = 5
    # Past this many seconds the 5-minute passive-reply window is too close to
    # call, so the summary is sent as an active message instead.
    passive_reply_deadline_s: int = 270


class AutoSummaryConfig(BaseModel):
    """Summarise a group on its own once enough messages pile up.

    The point is group corpora that nobody thinks to summon the bot in: without
    this, a group that never sees an `@` never contributes to the knowledge base
    at all. It can still be switched off with `enabled = false`.
    """

    enabled: bool = True
    # Messages accumulated since this group's last summary before one is made.
    # Kept at least as large as `summary.default_recent_n`, which is roughly how
    # far back the agent reads: below it, consecutive summaries overlap.
    min_messages: int = 200
    # Minimum spacing between attempts, *including failed ones* — otherwise a
    # gateway outage would re-trigger on the very next message, forever.
    cooldown_s: int = 1800
    # Send the result to the group as well. Off by default: an active message
    # consumes quota (20/min per group) and needs the owner to have enabled
    # bot-initiated pushes. The document is stored either way.
    notify: bool = False
    # Empty = every group. Listing `group_openid`s restricts the trigger to them.
    groups: list[str] = Field(default_factory=list)
    instruction: str = "自动总结：请总结本群最近的讨论，按话题归类，标注发言人与时间。"


class C2CConfig(BaseModel):
    """Private-chat retrieval.

    Private chat can read every group's summaries **and, via
    `messages_across_groups`, their raw messages**. That is the intended
    behaviour, not a leak — but it is also the setting with the largest blast
    radius here, so `allowlist` exists as the tightening knob. Empty means
    everyone; listing `user_openid`s restricts it to them.
    """

    enabled: bool = True
    allowlist: list[str] = Field(default_factory=list)
    max_groups_shown: int = 20
    # Hard ceiling on rows one private-chat raw-message lookup may return.
    raw_limit: int = 200


class MediaConfig(BaseModel):
    """图片落盘与按需查看（M2，docs/MEDIA.md）。

    后台 `MediaWorker` 只负责把图片字节尽快抓到本地（这是对 `rkey` 限时签名的
    唯一防御）；"看图"本身发生在 `view_image` 工具里，一次调用、结果缓存。
    `enabled = false` 时 Worker 不启动，但 media 行照常入库——队列在库里，
    以后打开开关即补跑。
    """

    enabled: bool = True
    dir: str = "data/media"          # 根目录；内部按消息发送日期分桶 YYYY-MM-DD/
    batch: int = 10                  # Worker 每批处理的 pending 行数
    max_bytes: int = 33_554_432      # 32 MiB；先按 event_size 预判，下载后按真实字节复核
    # view_image 内联 base64 的尺寸上限。与 max_bytes 是两条线：32 MiB 是"值得
    # 存"的下限护栏，这个是"发得出去"的上限护栏——base64 膨胀 ~33%，几 MB 的
    # body 会被网关直接拒，且每张被拒的图都白扣一次 max_views。
    max_inline_bytes: int = 4_194_304
    attempts_max: int = 3            # 网络/5xx 重试上限，超过转 failed
    download_timeout_s: int = 30
    interval_s: int = 15             # 无唤醒时的兜底轮询间隔
    backoff_s: int = 60              # 一批失败后的退避
    # 每次 agent 运行（一个 BotContext）允许 view_image 真实调用视觉模型的上限。
    # 防的是模型逐图上瘾把 5 分钟被动回复窗口吃光；计数在 ctx 上，天然按轮重置。
    max_views: int = 6
    detail: str = "low"              # 官方档位：low 缩到 512×512，摘要够用且省 token
    describe_prompt: str = (
        "描述这张图片：先说主体内容，再逐字转写图中出现的文字。2-4 句中文。"
    )

    @property
    def dir_path(self) -> Path:
        return _resolve_path(self.dir)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: str = "logs"

    @property
    def dir_path(self) -> Path:
        return _resolve_path(self.dir)


class AppConfig(BaseModel):
    qq: QQConfig = QQConfig()
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    store: StoreConfig = StoreConfig()
    summary: SummaryConfig = SummaryConfig()
    c2c: C2CConfig = C2CConfig()
    auto_summary: AutoSummaryConfig = AutoSummaryConfig()
    media: MediaConfig = MediaConfig()
    logging: LoggingConfig = LoggingConfig()
    # Optional `[groups]` section: `group_openid` → a human name, used in
    # retrieved document text and answers. QQ exposes no way to resolve the name.
    groups: dict[str, str] = Field(default_factory=dict)


def _load() -> AppConfig:
    if not _CONFIG_PATH.exists():
        print(f"⚠️  {_CONFIG_PATH} not found, using defaults")
        return AppConfig()
    try:
        with _CONFIG_PATH.open("rb") as fh:
            raw = tomllib.load(fh)
    except Exception as exc:
        print(f"❌  failed to parse {_CONFIG_PATH}: {exc}")
        raise
    return AppConfig(**_expand_dict(raw))


config: AppConfig = _load()
