"""JSON Lines logging.

Call `setup_logger()` once, as the first thing a process does, before any
module that logs. Safe to call from every module: only the first call configures
anything.

Three requirements shape the configuration:

1. **Handlers live on the true root logger**, not on the `qqbot` logger. On the
   `qqbot` logger, third-party records (botpy's API errors and their `trace_id`,
   httpx, …) never reach the log file at all — exactly the records you want when
   the bot misbehaves.
2. **The root logger is pinned to DEBUG and the console handler carries the
   configured level.** Putting the configured level on the root logger while the
   file handler asks for DEBUG filters DEBUG out before it ever reaches the file,
   making the file handler's level meaningless.
3. **Stray root stream handlers are removed.** botpy calls
   `logging.basicConfig()` at import time, planting a level-NOTSET stderr handler
   on the root logger; combined with requirement 2 that would mirror every DEBUG
   record to stderr. Removing them makes the outcome independent of import order.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys

from src.config import config

_APP_LOGGER = "qqbot"

# Third-party loggers that are pure noise at DEBUG and would otherwise flood the
# log file. botpy sits here deliberately: it logs every raw websocket frame,
# which we already persist ourselves.
_NOISY = (
    "aiohttp",
    "botpy",
    "chromadb",
    "httpcore",
    "httpx",
    "urllib3",
    "websockets",
)

# Everything logging puts on a record by default; anything *else* was passed by
# the caller via `extra=` and belongs in the JSON payload.
_RESERVED_KEYS = frozenset(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime", "taskName"}

_configured = False


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per line. `extra={...}` fields are merged into the entry."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_KEYS:
                entry[key] = value
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        # ensure_ascii=False keeps Chinese readable instead of \uXXXX soup.
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logger(name: str = _APP_LOGGER, *, console: bool = False) -> logging.Logger:
    """Configure logging once and return a logger for `name`.

    Re-entrant: only the first call attaches handlers, so `console=True` has an
    effect only on that first call.
    """
    global _configured
    if _configured:
        return logging.getLogger(name)

    root = logging.getLogger()
    log_dir = config.logging.dir_path
    log_dir.mkdir(parents=True, exist_ok=True)

    # See fix 3 in the module docstring.
    for handler in list(root.handlers):
        if type(handler) is logging.StreamHandler:
            root.removeHandler(handler)

    root.setLevel(logging.DEBUG)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonLinesFormatter())
    root.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(
            getattr(logging, config.logging.level.upper(), logging.INFO)
        )
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
            )
        )
        root.addHandler(stream_handler)

    for noisy in _NOISY:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    return logging.getLogger(name)
