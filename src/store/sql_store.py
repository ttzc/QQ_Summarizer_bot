"""SQLite persistence for inbound group messages and the summaries made of them.

Two tables with very different lifetimes:

  * `group_messages` — every message the bot received, append-only, never
    embedded. Read back by time range and by count.
  * `summaries` — one row per successful group summarisation. **These** are what
    the vector index is built from; `indexed_at IS NULL` is the indexer backlog.

Concurrency note — this is a long-running asyncio service that writes
continuously, which rules out the naive single-connection setup that would be
fine for a short-lived CLI process.

Here:
  * one connection, `check_same_thread=False`, touched **only from the event
    loop thread** (never handed to `asyncio.to_thread` — sqlite3 is synchronous,
    and sharing the connection across threads is the actual hazard);
  * WAL *plus* `busy_timeout`, so a stray second process cannot deadlock us.

**Why no lock.** Every method here is fully synchronous — there is no `await`
between the first and last statement of a transaction. Under asyncio that makes
each call atomic with respect to the event loop, so concurrent handler tasks
cannot interleave two writes on the shared connection. That absence of `await`
is the actual invariant; if a write path ever grows one, it needs an
`asyncio.Lock` around it. Long or blocking work (embedding calls, network I/O)
must not move in here at all — it belongs in the background indexer.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from src.config import config
from src.logger import setup_logger

if TYPE_CHECKING:
    from src.bot.events import GroupMessageRecord

logger = setup_logger("qqbot.store.sql")

_SCHEMA_SQL = Path(__file__).parent / "sql" / "schema.sql"

# Columns added after the first release; applied lazily by `_migrate()`.
# `(table, column, ddl)` — `CREATE TABLE IF NOT EXISTS` in `schema.sql` adds a
# *table* to an existing database but never a column, so every later column has
# to be listed here as well as in the DDL.
_MIGRATIONS: list[tuple[str, str, str]] = [
    ("summaries", "trigger", "TEXT NOT NULL DEFAULT 'at'"),
]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sort_key(raw: str) -> str:
    """UTC ISO string, so lexicographic order matches chronological order.

    Timestamps here carry a `+08:00`-style offset; comparing those as text is
    wrong as soon as two rows disagree about the offset, and comparing parsed
    `datetime`s directly blows up when one is naive. Normalising to a UTC string
    sidesteps both while keeping a plain `min`/`max` usable.
    """
    try:
        return datetime.fromisoformat(raw).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return str(raw or "")


def _hull(intervals: Sequence[tuple[str, str]]) -> tuple[str | None, str | None]:
    """Widest (earliest start, latest end) across coverage intervals.

    An *envelope*, not a guarantee of continuity: the agent may read "昨天" and
    "前天" in two separate calls, and the span between them was never read. The
    exact intervals are kept in `coverage_json`; this is for display.
    """
    starts = [s for s, _ in intervals if s]
    ends = [e for _, e in intervals if e]
    if not starts or not ends:
        return (None, None)
    return (min(starts, key=_sort_key), max(ends, key=_sort_key))


class SQLStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else config.store.sqlite_file
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA_SQL.read_text(encoding="utf-8"))
        self._conn.commit()
        self._migrate()
        logger.info(f"SQLite store opened: {self._db_path}")

    def _migrate(self) -> None:
        """Add columns that post-date the table. Add-only; nothing is rewritten."""
        known: dict[str, set[str]] = {}
        for table, column, ddl in _MIGRATIONS:
            if table not in known:
                cursor = self._conn.execute(f"PRAGMA table_info({table})")
                known[table] = {row[1] for row in cursor.fetchall()}
            if column not in known[table]:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                self._conn.commit()
                known[table].add(column)
                logger.info(f"migration: added column {table}.{column}")

    # ---- writes -----------------------------------------------------------

    def insert_messages(self, records: Sequence["GroupMessageRecord"]) -> int:
        """Insert messages, skipping duplicates. Returns the number actually stored."""
        if not records:
            return 0
        rows = [
            (
                rec.message_id,
                rec.event_id,
                rec.group_openid,
                rec.author_openid,
                rec.author_name,
                rec.member_role,
                rec.content,
                rec.message_type,
                rec.ts.isoformat(),
                rec.msg_idx,
                rec.ref_msg_idx,
                int(bool(rec.attachments)),
                json.dumps(rec.raw, ensure_ascii=False, default=str),
                _now(),
            )
            for rec in records
        ]
        with self._conn:  # one transaction for the batch
            before = self._conn.total_changes
            self._conn.executemany(
                """
                INSERT OR IGNORE INTO group_messages (
                    message_id, event_id, group_openid, author_openid, author_name,
                    member_role, content, message_type, ts, msg_idx, ref_msg_idx,
                    has_media, raw_json, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            inserted = self._conn.total_changes - before
        if inserted != len(rows):
            logger.debug(
                "duplicate messages skipped",
                extra={"submitted": len(rows), "inserted": inserted},
            )
        return inserted

    def insert_summary(
        self,
        *,
        group_openid: str,
        instruction: str,
        content: str,
        coverage: Sequence[tuple[str, str]],
        message_count: int,
        requested_by: str | None = None,
        trigger: str = "at",
    ) -> str:
        """Store one summary. Returns its new `summary_id`.

        A plain INSERT, deliberately: see the note on `summaries` in `schema.sql`
        for why this is not a content-hash upsert.
        """
        summary_id = uuid.uuid4().hex
        ts_start, ts_end = _hull(coverage)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO summaries (
                    summary_id, group_openid, instruction, content,
                    coverage_json, ts_start, ts_end, message_count,
                    requested_by, trigger, created_at, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    summary_id,
                    group_openid,
                    instruction,
                    content,
                    json.dumps([list(iv) for iv in coverage], ensure_ascii=False)
                    if coverage
                    else None,
                    ts_start,
                    ts_end,
                    int(message_count),
                    requested_by,
                    trigger,
                    _now(),
                ),
            )
        return summary_id

    def mark_summaries_indexed(self, summary_ids: Iterable[str]) -> None:
        stamp = _now()
        rows = [(stamp, sid) for sid in summary_ids]
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                "UPDATE summaries SET indexed_at = ? WHERE summary_id = ?", rows
            )

    def clear_summary_marks(self) -> int:
        """Forget which summaries were indexed. Returns how many marks were dropped.

        Pairs with dropping the vector collection: the marks and the vectors must
        be reset together, or `reindex` would skip summaries whose vectors are
        gone.
        """
        with self._conn:
            before = self._conn.total_changes
            self._conn.execute("UPDATE summaries SET indexed_at = NULL")
            return self._conn.total_changes - before

    # ---- reads ------------------------------------------------------------

    def recent_messages(self, group_openid: str, limit: int) -> list[sqlite3.Row]:
        """Newest `limit` messages, returned oldest-first (ready for a prompt)."""
        cursor = self._conn.execute(
            """
            SELECT * FROM (
                SELECT * FROM group_messages
                WHERE group_openid = ?
                ORDER BY datetime(ts) DESC
                LIMIT ?
            ) ORDER BY datetime(ts) ASC
            """,
            (group_openid, limit),
        )
        return cursor.fetchall()

    def messages_in_range(
        self,
        group_openid: str | None,
        start_iso: str,
        end_iso: str,
        limit: int,
        keyword: str | None = None,
    ) -> list[sqlite3.Row]:
        """Messages in an inclusive time window, oldest first.

        `group_openid=None` means **every group** and is not an oversight: only
        the private-chat path may pass it (the in-group tools always supply their
        injected group). Same shape as `summaries_for`.

        Both bounds go through SQLite's `datetime()` rather than being compared
        as text. ISO strings only sort correctly when every value carries the
        same UTC offset, and these bounds come from the *model* — it may emit
        `2026-07-21T00:00:00Z`, a naked date, or local time with no offset.
        `datetime()` normalises all of those to UTC.
        """
        clauses = ["datetime(ts) >= datetime(?)", "datetime(ts) <= datetime(?)"]
        params: list[Any] = [start_iso, end_iso]
        if group_openid is not None:
            clauses.append("group_openid = ?")
            params.append(group_openid)
        if keyword:
            # Plain substring match; there is no FTS index on this column and the
            # windows are bounded, so a scan of one window is acceptable.
            clauses.append("content LIKE '%' || ? || '%'")
            params.append(keyword)
        params.append(limit)

        cursor = self._conn.execute(
            f"""
            SELECT * FROM group_messages
            WHERE {' AND '.join(clauses)}
            ORDER BY datetime(ts) ASC
            LIMIT ?
            """,
            params,
        )
        return cursor.fetchall()

    def messages_since_last_summary(self, group_openid: str) -> int:
        """How many messages landed after this group's most recent summary.

        The auto-summary trigger reads this on every inbound message, so it uses
        the `(group_openid, ingested_at)` index. `ingested_at` (when *we* stored
        it) rather than `ts` (when the speaker claims to have sent it): the
        question is "how much has arrived since", which the platform timestamp
        cannot answer.

        With no summary at all the bound falls back to the epoch, so a group that
        already had a backlog summarises it on the first trigger.
        """
        cursor = self._conn.execute(
            """
            SELECT COUNT(*) FROM group_messages
            WHERE group_openid = ?
              AND datetime(ingested_at) > datetime(COALESCE(
                    (SELECT MAX(created_at) FROM summaries WHERE group_openid = ?),
                    '1970-01-01T00:00:00'))
            """,
            (group_openid, group_openid),
        )
        return int(cursor.fetchone()[0])

    def unindexed_summaries(self, limit: int) -> list[sqlite3.Row]:
        """Backlog for the vector indexer, oldest first."""
        cursor = self._conn.execute(
            """
            SELECT * FROM summaries
            WHERE indexed_at IS NULL
            ORDER BY datetime(created_at) ASC, rowid ASC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()

    def summaries_for(
        self, group_openid: str | None = None, limit: int = 100
    ) -> list[sqlite3.Row]:
        """Stored summaries, newest first. `None` spans every group.

        The unfiltered form is what private chat searches over; the filtered one
        is what a group scopes itself to.
        """
        if group_openid is None:
            cursor = self._conn.execute(
                """
                SELECT * FROM summaries
                ORDER BY datetime(created_at) DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            )
        else:
            cursor = self._conn.execute(
                """
                SELECT * FROM summaries
                WHERE group_openid = ?
                ORDER BY datetime(created_at) DESC, rowid DESC
                LIMIT ?
                """,
                (group_openid, limit),
            )
        return cursor.fetchall()

    def groups_with_messages(self) -> list[str]:
        """Every group the bot has received messages from.

        A different set from `groups_with_summaries`: a group can be full of raw
        messages and still have no summary (nobody @-ed the bot and the count has
        not reached the auto-summary threshold yet). Private chat must be able to
        *name* such a group — `messages_across_groups` takes a label — and
        `list_groups` does not show it, so this list is what that name resolves
        against.
        """
        cursor = self._conn.execute("SELECT DISTINCT group_openid FROM group_messages")
        return [row[0] for row in cursor.fetchall()]

    def groups_with_summaries(self, limit: int = 50) -> list[sqlite3.Row]:
        """Groups that have at least one summary, busiest first."""
        cursor = self._conn.execute(
            """
            SELECT group_openid,
                   COUNT(*)          AS summaries,
                   MIN(created_at)   AS first_at,
                   MAX(created_at)   AS last_at
            FROM summaries
            GROUP BY group_openid
            ORDER BY summaries DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()

    def stats(self) -> list[sqlite3.Row]:
        """Per group: message volume and summary counts.

        Two independent aggregates joined at the group level — joining
        `group_messages` to `summaries` directly would multiply both counts.
        """
        cursor = self._conn.execute(
            """
            SELECT g.group_openid,
                   g.total,
                   g.first_ts,
                   g.last_ts,
                   COALESCE(s.summaries, 0) AS summaries,
                   COALESCE(s.indexed, 0)   AS indexed
            FROM (
                SELECT group_openid,
                       COUNT(*)  AS total,
                       MIN(ts)   AS first_ts,
                       MAX(ts)   AS last_ts
                FROM group_messages
                GROUP BY group_openid
            ) AS g
            LEFT JOIN (
                SELECT group_openid,
                       COUNT(*)            AS summaries,
                       COUNT(indexed_at)   AS indexed
                FROM summaries
                GROUP BY group_openid
            ) AS s ON s.group_openid = g.group_openid
            ORDER BY g.total DESC
            """
        )
        return cursor.fetchall()

    def close(self) -> None:
        self._conn.close()


_store: SQLStore | None = None


def get_sql_store() -> SQLStore:
    global _store
    if _store is None:
        _store = SQLStore()
    return _store
