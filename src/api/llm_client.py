"""Chat model factory.

Deliberately provider-agnostic: the gateway is whatever `[llm].base_url` points
at, including nothing at all (then the OpenAI default endpoint applies).

Unset settings are *omitted* rather than passed as the literal ``"${VAR}"``, so a
gateway that needs no API key (a local deployment, say) still works, and a
genuinely missing key fails with langchain's own clear error instead of a 401
from a faraway endpoint.
"""

from __future__ import annotations

from langchain.chat_models import BaseChatModel, init_chat_model

from src.config import config
from src.logger import setup_logger

logger = setup_logger("qqbot.api.llm")

_model: BaseChatModel | None = None


def get_chat_model() -> BaseChatModel:
    """Return the process-wide chat model, building it on first use."""
    global _model
    if _model is not None:
        return _model

    cfg = config.llm
    overrides: dict = {
        "model": cfg.model,
        "model_provider": "openai",
        # None means "env var unset" — leave the key out so langchain applies
        # its own default (or the provider needs none at all).
        "openai_api_key": cfg.resolved_api_key,
        "base_url": cfg.resolved_base_url,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "timeout": cfg.timeout,
    }
    kwargs = {k: v for k, v in overrides.items() if v is not None}

    logger.info(
        "初始化 chat 模型",
        extra={"model": cfg.model, "base_url": cfg.resolved_base_url or "(默认)"},
    )
    _model = init_chat_model(**kwargs)
    return _model
