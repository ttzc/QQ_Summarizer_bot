"""The single place that turns text into a knowledge document.

Living in its own module is not ceremony: the two kinds of writer both need it,
and they sit on opposite sides of an import edge. The `save_summary` **tool**
(`src/agent/tools.py`) writes while the agent is running — the model has decided
this answer is worth keeping — while the **code-side fallbacks**
(`src/agent/summarizer.store_summary`, used by the auto-summary path and the
offline `ask --save`) write after the run. Since `summarizer` imports `tools`, a
helper placed in either of those would be a cycle.

What this module deliberately does *not* hold is **policy**. Whether something
deserves to become a document is exactly the judgement this project moved out of
the code (see `docs/ARCHITECTURE.md` §4.3 and the `save_summary` tool), so the
writer takes the decision as given and only normalises the shape of a row: no
length floors, no "is it grounded" checks, no silence-on-failure. Callers run
their own gates and say so, because a refusal the model can read and act on is
worth more than one it cannot see.
"""

from __future__ import annotations

from typing import Sequence

from src.store.sql_store import SQLStore


def insert_document(
    store: SQLStore,
    *,
    group_openid: str,
    instruction: str,
    content: str,
    coverage: Sequence[tuple[str, str]],
    message_count: int,
    requested_by: str | None = None,
    trigger: str = "at",
) -> str:
    """Write one document row and return its `summary_id`.

    `trigger` records the **entry point**, not who decided: `'at'` for a turn
    that began with someone summoning the bot (whether the model or the fallback
    wrote the row), `'auto'` for the message-count trigger. Keeping it two-valued
    is deliberate — the audit question this column answers is "how much of the
    corpus did the bot write on its own initiative", and a third value for
    "agent chose to publish" would split that signal in two.

    Raises whatever `SQLStore` raises. The tool catches and tells the model; the
    fallbacks catch and log. Neither lets a write failure kill the turn.
    """
    return store.insert_summary(
        group_openid=group_openid,
        instruction=instruction,
        content=content,
        coverage=coverage,
        message_count=message_count,
        requested_by=requested_by,
        trigger=trigger,
    )
