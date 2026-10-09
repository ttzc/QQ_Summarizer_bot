"""Chroma-backed semantic index over **summaries**, not raw messages.

The read side of RAG, plus the index handle the writer (`indexer.py`) uses.

Why the document unit is a summary: a busy group is mostly "哈哈哈" and "收到".
Embedding each of those gives you a vector store full of fragments, and a query
like "之前有人提过部署方案吗" matches whichever fragment happens to share a word
rather than the discussion that reached a conclusion. One document per
summarisation is both cheaper and a better retrieval unit. Raw messages stay in
SQLite and are still reachable by time range — just not semantically.

Two constraints shape this:

* **Metadata is stored as scalars** (`str | int | float | bool`). chromadb 1.5.9
  actually tolerates `list` and `None` too, but only *nested dicts* are rejected
  outright. Scalars are a deliberate choice, not a hard limit: they are what the
  equality filters below operate on.
* **The document text carries the group and coverage window**, since retrieval
  only ever sees `page_content`. Without that header a private-chat answer could
  not say *which* group or *when* — and private chat is precisely the surface
  that reads across groups.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Sequence

from langchain_chroma import Chroma

from src.api.embedding_client import get_embeddings
from src.config import config
from src.logger import setup_logger

logger = setup_logger("qqbot.rag.retriever")


def speaker_of(row: sqlite3.Row | dict) -> str:
    """Display name for a stored message row, mirroring `display_name` on the event."""
    name = row["author_name"] if isinstance(row, sqlite3.Row) else row.get("author_name")
    openid = (
        row["author_openid"] if isinstance(row, sqlite3.Row) else row.get("author_openid")
    )
    if name:
        return name
    return f"成员{openid[-4:]}" if openid else "未知成员"


def group_label(group_openid: str | None) -> str:
    """Human-readable group name: the configured alias if any, else a short stub.

    QQ exposes no way to resolve a `group_openid` to a group's real name, so
    without an alias the best available handle is a suffix. Configure `[groups]`
    in `config.toml` to get readable names in answers.
    """
    if not group_openid:
        return "未知群"
    alias = config.groups.get(group_openid)
    return alias or f"群{group_openid[-6:]}"


def _stamp(raw: Any) -> str:
    try:
        return datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(raw or "")


def summary_text(row: sqlite3.Row | dict) -> str:
    """`page_content` for a summary row, or '' when there is nothing to embed."""
    get = (lambda k: row[k]) if isinstance(row, sqlite3.Row) else row.get
    content = (get("content") or "").strip()
    if not content:
        return ""

    start = _stamp(get("ts_start"))
    end = _stamp(get("ts_end"))
    count = get("message_count") or 0
    window = f"{start} ~ {end}" if start and end else (start or end or "时间范围未知")
    header = f"[{group_label(get('group_openid'))}] {window} · {count} 条"
    return f"{header}\n{content}"


class SummaryIndex:
    """Wraps one Chroma collection. Filtering by group is optional, not implied."""

    def __init__(
        self,
        *,
        embedding_function: Any = None,
        persist_directory: Any = None,
        collection_name: str | None = None,
    ) -> None:
        self._collection_name = collection_name or config.store.chroma_collection
        self._persist_directory = str(persist_directory or config.store.chroma_dir)
        self._embedding_function = embedding_function
        self._store = self._open()
        logger.info(
            "向量库就绪",
            extra={
                "collection": self._collection_name,
                "path": self._persist_directory,
            },
        )

    def _open(self) -> Chroma:
        return Chroma(
            collection_name=self._collection_name,
            persist_directory=self._persist_directory,
            embedding_function=self._embedding_function or get_embeddings(),
        )

    def reset(self) -> None:
        """Drop every vector and reopen an empty collection.

        Needed by `reindex --reset`. Pair it with
        `SQLStore.clear_summary_marks()` — clearing one without the other leaves
        the two stores disagreeing about what has been indexed.
        """
        try:
            self._store.delete_collection()
        except Exception:  # noqa: BLE001 - a missing collection is already reset
            logger.debug("删除向量集合失败或集合不存在，直接重建", exc_info=True)
        # The old handle points at a deleted collection, so reopen.
        self._store = self._open()
        logger.info("向量集合已重置", extra={"collection": self._collection_name})

    def add(self, rows: Sequence[sqlite3.Row | dict]) -> int:
        """Embed and upsert summaries. Returns how many were actually indexed.

        Upsert, not insert: Chroma keys on `summary_id`, so re-running `reindex`
        overwrites rather than duplicating.
        """
        texts: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []
        for row in rows:
            text = summary_text(row)
            if not text:  # a blank summary has nothing to embed
                continue
            get = (lambda k: row[k]) if isinstance(row, sqlite3.Row) else row.get
            texts.append(text)
            ids.append(str(get("summary_id")))
            metadatas.append(
                {
                    "summary_id": str(get("summary_id")),
                    "group_openid": str(get("group_openid") or ""),
                    "created_at": str(get("created_at") or ""),
                    "ts_start": str(get("ts_start") or ""),
                    "ts_end": str(get("ts_end") or ""),
                    "message_count": int(get("message_count") or 0),
                }
            )
        if not texts:
            return 0
        self._store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return len(texts)

    def search(
        self, query: str, k: int = 20, group_openid: str | None = None
    ) -> list[tuple[Any, float]]:
        """Relevance search, most relevant first.

        `group_openid=None` means **no filter** and is not an oversight — it is
        the private-chat path, which is allowed to read every group. Pass a group
        to confine the search to it, which is what the in-group path always does:
        without the filter a summary in group A could quote group B.
        """
        if group_openid is None:
            return self._store.similarity_search_with_score(query, k=k)
        return self._store.similarity_search_with_score(
            query, k=k, filter={"group_openid": group_openid}
        )

    def count(self) -> int:
        return self._store._collection.count()


_index: SummaryIndex | None = None


def get_summary_index() -> SummaryIndex:
    """Process-wide index, built on first use."""
    global _index
    if _index is None:
        _index = SummaryIndex()
    return _index
