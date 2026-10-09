"""Background indexer: moves stored summaries into the vector store.

Why this exists rather than embedding inside the reply path: a summary is only
written once the agent has finished, and the bot must stay free to answer other
groups meanwhile. Instead the handler writes to SQLite and pokes `wake()`; this
loop does the embedding.

The backlog lives in SQLite (`summaries WHERE indexed_at IS NULL`), not in
memory, so a crash or restart resumes exactly where it left off and never
re-embeds what was already done.
"""

from __future__ import annotations

import asyncio

from src.api.embedding_client import embedding_batch_size
from src.logger import setup_logger
from src.rag.retriever import SummaryIndex
from src.store.sql_store import SQLStore

logger = setup_logger("qqbot.rag.indexer")

# Summaries fetched per pass. The embedding client chunks these into batches of
# `embedding_batch_size()` internally, so this can safely exceed the per-request
# batch ceiling. Summaries arrive far more slowly than messages did, so a small
# batch is plenty.
INDEX_BATCH = 50


class SummaryIndexer:
    def __init__(
        self,
        store: SQLStore,
        index: SummaryIndex,
        *,
        interval_s: float = 10.0,
        backoff_s: float = 30.0,
        batch: int = INDEX_BATCH,
    ) -> None:
        self._store = store
        self._index = index
        self._interval = interval_s
        self._backoff = backoff_s
        self._batch = batch
        self._wake = asyncio.Event()

    def wake(self) -> None:
        """Signal that a new summary is waiting. Safe to call from a handler."""
        self._wake.set()

    async def index_once(self) -> int:
        """Index up to one batch. Returns how many summaries were indexed.

        Embedding runs in a worker thread so the event loop keeps serving
        messages; the rows are copied to plain dicts first, because reading
        `sqlite3.Row` objects off the loop thread while the connection is in use
        elsewhere is exactly the cross-thread hazard `sql_store.py` warns about.
        """
        rows = self._store.unindexed_summaries(limit=self._batch)
        if not rows:
            return 0

        payload = [dict(row) for row in rows]
        indexed = await asyncio.to_thread(self._index.add, payload)
        if indexed < len(payload):
            logger.warning(
                "部分总结无可嵌入文本，已跳过",
                extra={"fetched": len(payload), "indexed": indexed},
            )

        # Mark the whole batch, including the skipped ones: a blank summary has
        # nothing to embed now and would otherwise clog the backlog forever.
        self._store.mark_summaries_indexed([row["summary_id"] for row in payload])
        logger.debug("索引批次完成", extra={"indexed": indexed})
        return indexed

    async def drain(self) -> int:
        """Index until the backlog is empty. Used by `reindex`."""
        total = 0
        while True:
            done = await self.index_once()
            if not done:
                return total
            total += done

    async def run_forever(self) -> None:
        logger.info("索引任务启动", extra={"interval_s": self._interval})
        while True:
            # Cleared *before* working, so a wake() arriving mid-batch is not
            # lost and the wait below returns immediately.
            self._wake.clear()
            try:
                indexed = await self.index_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a bad batch must not end the loop
                logger.exception("索引批次失败，退避后重试")
                await asyncio.sleep(self._backoff)
                continue

            if indexed:
                continue  # backlog may remain; keep draining without waiting
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
