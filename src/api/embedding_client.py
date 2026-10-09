"""Embedding model factory, plus the batch-size limit that goes with it.

An OpenAI-compatible gateway is not necessarily a *faithful* OpenAI endpoint, so
the client is kept on the conservative path. Three settings, each guarding
against a concrete incompatibility:

* ``tiktoken_enabled=False`` — otherwise langchain tokenises locally and sends
  token *ids* instead of text, which many gateways do not accept;
* ``check_embedding_ctx_length=False`` — skips the local context-length probe,
  which also runs through tiktoken;
* ``chunk_size<=25`` — the batching knob. 25 is a deliberately conservative
  default, not a measured provider ceiling; raise ``MAX_EMBED_BATCH`` if the
  gateway in use accepts larger batches.

`embedding_batch_size()` is the single source of truth for the limit, so the
indexer and `reindex` cannot drift apart.
"""

from __future__ import annotations

from langchain.embeddings import Embeddings, init_embeddings

from src.config import config
from src.logger import setup_logger

logger = setup_logger("qqbot.api.embedding")

# Conservative ceiling on inputs per request. Not a measured provider limit —
# it keeps a mis-set `batch_size` from producing oversized requests. Raise it
# if the gateway in use accepts larger batches.
MAX_EMBED_BATCH = 25

_embeddings: Embeddings | None = None


def embedding_batch_size() -> int:
    """Configured batch size, clamped to the gateway's hard limit."""
    configured = config.embedding.batch_size
    if configured > MAX_EMBED_BATCH:
        logger.warning(
            "embedding.batch_size 超过网关上限，已钳制",
            extra={"configured": configured, "using": MAX_EMBED_BATCH},
        )
        return MAX_EMBED_BATCH
    return max(1, configured)


def get_embeddings() -> Embeddings:
    """Return the process-wide embedding model, building it on first use."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    cfg = config.embedding
    overrides: dict = {
        "model": cfg.model,
        "provider": "openai",
        "openai_api_key": cfg.resolved_api_key,
        "base_url": cfg.resolved_base_url,
        "tiktoken_enabled": False,
        "check_embedding_ctx_length": False,
        "chunk_size": embedding_batch_size(),
    }
    kwargs = {k: v for k, v in overrides.items() if v is not None}

    logger.info(
        "初始化 embedding 模型",
        extra={
            "model": cfg.model,
            "base_url": cfg.resolved_base_url or "(默认)",
            "batch_size": embedding_batch_size(),
        },
    )
    _embeddings = init_embeddings(**kwargs)
    return _embeddings
