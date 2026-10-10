"""`qqbot` command line entry point.

Subcommands:

    qqbot run                 start the bot (long-lived; needs real QQ credentials)
    qqbot stats               message and summary counts per group
    qqbot summaries           list the stored summaries
    qqbot reindex             build/repair the summary index [--reset]
    qqbot ask "..."           run one answer offline (no QQ connection)

Heavy imports happen inside each handler so that `qqbot --help` and `stats` do
not pay for loading Chroma or the LLM stack.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys

from src.config import config
from src.logger import setup_logger

# Deliberately NOT `setup_logger(...)` here: configuration is one-shot, so doing
# it at import time would make `--verbose` a silent no-op further down. `main()`
# configures logging once it knows whether the flag was passed.
logger = logging.getLogger("qqbot.cli")


def _ensure_utf8_stdout() -> None:
    """Without this, Chinese output raises UnicodeEncodeError on Windows (cp936)."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")


def _missing_models() -> list[str]:
    """Unset model names, by config section.

    A model name is not a secret, so it lives in `config.toml` — which also means
    "forgot to fill it in" is an ordinary configuration error and deserves to be
    reported before the bot connects to QQ rather than on the first summary.
    """
    checks = (("[llm].model", config.llm), ("[embedding].model", config.embedding))
    return [name for name, section in checks if not section.resolved_model]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qqbot",
        description="QQ 群消息总结机器人：被 @ 时按指令总结群聊，并把每次总结作为 RAG 知识库。",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="同时输出到控制台")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="启动机器人（长驻，需真实 QQ 凭据）")
    p_run.set_defaults(handler=_cmd_run)

    p_stats = sub.add_parser("stats", help="查看各群消息数与总结数")
    p_stats.set_defaults(handler=_cmd_stats)

    p_summaries = sub.add_parser("summaries", help="列出已入库的总结")
    p_summaries.add_argument("--group", help="只看某个群；省略则列出全部群")
    p_summaries.add_argument("--limit", type=int, default=20, help="最多列出多少篇")
    p_summaries.set_defaults(handler=_cmd_summaries)

    p_reindex = sub.add_parser("reindex", help="把 SQLite 里的总结灌进向量库")
    p_reindex.add_argument(
        "--reset",
        action="store_true",
        help="先清空向量集合与索引标记，再全量重建（用于修复索引）",
    )
    p_reindex.set_defaults(handler=_cmd_reindex)

    p_media = sub.add_parser("media", help="列出图片附件的处理状态（M2，只读排查）")
    p_media.add_argument(
        "--status",
        default="pending",
        help="pending / stored / expired / skipped / failed / all（默认 pending）",
    )
    p_media.add_argument("--limit", type=int, default=20)
    p_media.set_defaults(handler=_cmd_media)

    p_ask = sub.add_parser("ask", help="脱机跑一次问答（不连 QQ，需要真实 LLM 凭据）")
    p_ask.add_argument("instruction", help="要问的话，例如“总结最近 50 条”")
    p_ask.add_argument("--group", help="群 openid；库里只有一个群时可省略")
    p_ask.add_argument(
        "--all",
        action="store_true",
        help="走私聊 scope：检索所有群的总结与聊天原文（忽略 --group）",
    )
    p_ask.add_argument(
        "--save", action="store_true", help="把这次回答作为总结入库（仅群内 scope）"
    )
    p_ask.set_defaults(handler=_cmd_ask)

    return parser


# ---- commands --------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from src.agent.summarizer import Summarizer
    from src.bot.client import SummarizerClient
    from src.media.worker import MediaWorker
    from src.rag.indexer import SummaryIndexer
    from src.rag.retriever import get_summary_index
    from src.store.sql_store import get_sql_store

    appid, secret = config.qq.resolved_appid, config.qq.resolved_secret
    if not appid or not secret:
        print(
            "❌ 缺少 QQ 凭据：请复制 .env.example 为 .env，填好 QQ_APPID / QQ_SECRET。\n"
            "   （config.toml 的 [qq] 段通过 ${QQ_APPID} / ${QQ_SECRET} 读取它们）",
            file=sys.stderr,
        )
        return 2
    if missing := _missing_models():
        print(
            f"❌ config.toml 里还没有模型名：{'、'.join(missing)}。\n"
            "   模型名与 base_url 是配置项，写在 config.toml；.env 只放 API key。",
            file=sys.stderr,
        )
        return 2

    store = get_sql_store()
    index = get_summary_index()
    indexer = SummaryIndexer(store, index)
    media_worker = MediaWorker(store)
    summarizer = Summarizer(store, index)
    client = SummarizerClient(
        store=store,
        summarizer=summarizer,
        wake_indexer=indexer.wake,
        wake_media=media_worker.wake,
        # bot_log=True keeps botpy logging on the root logger (so its warnings and
        # trace_ids land in our JSON log); ext_handlers=False stops it from also
        # writing a `botpy.log` file into the CWD.
        bot_log=True,
        ext_handlers=False,
    )

    async def amain() -> None:
        # The client is built *inside* the running loop on purpose: `Client.__init__`
        # calls `asyncio.get_event_loop()`, which warns in 3.12+ and raises in 3.14
        # when there is no running loop. Here it finds one, so the deprecated path
        # is never taken — no need for the pre-emptive `set_event_loop` dance.
        async with client:
            # 两条后台循环：总结→向量（indexer），消息→图片落盘（media）。
            # 都是"SQLite 里躺着积压、wake 只是拨铃"的形态，崩了重启自然续跑。
            background = [
                asyncio.create_task(indexer.run_forever(), name="indexer"),
                asyncio.create_task(media_worker.run_forever(), name="media"),
            ]
            try:
                await client.start(appid, secret)
            finally:
                for task in background:
                    task.cancel()
                for task in background:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    logger.info("机器人启动", extra={"appid": appid, "sandbox": config.qq.is_sandbox})
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        store.close()
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from src.store.sql_store import get_sql_store

    store = get_sql_store()
    rows = store.stats()
    if not rows:
        print("还没有任何消息。机器人上线后会把收到的群消息存下来。")
        return 0

    print(
        f"{'群 openid':<28} {'消息数':>8} {'总结':>6} {'已索引':>7} {'待总结':>7} "
        f"{'图片(待/存/述)':>14} {'首条':<20} {'末条':<20}"
    )
    for row in rows:
        # "Messages since this group's last summary" — the number the auto
        # trigger compares against its threshold, so it is also the number to
        # look at when wondering whether `min_messages` is set right.
        pending = store.messages_since_last_summary(row["group_openid"])
        # 图片三格：待抓取 / 已落盘 / 已有描述缓存（M2，docs/MEDIA.md）。
        media_cell = (
            f"{row['media_pending']}/{row['media_stored']}/{row['media_described']}"
        )
        print(
            f"{row['group_openid']:<28} {row['total']:>8} {row['summaries']:>6} "
            f"{row['indexed']:>7} {pending:>7} {media_cell:>14} "
            f"{str(row['first_ts'])[:19]:<20} {str(row['last_ts'])[:19]:<20}"
        )
    if store.unindexed_summaries(limit=1):
        print("\n有未索引的总结，运行 `qqbot reindex` 补齐。")
    if config.auto_summary.enabled:
        groups = "、".join(config.auto_summary.groups) or "全部群"
        print(
            f"\n待总结 = 距本群上次总结新增的消息数；到 "
            f"{config.auto_summary.min_messages} 条会自动总结一次（适用范围：{groups}）。"
        )
    store.close()
    return 0


def _cmd_summaries(args: argparse.Namespace) -> int:
    """Show what is actually in the knowledge base — the thing RAG searches."""
    from src.rag.retriever import group_label
    from src.store.sql_store import get_sql_store

    store = get_sql_store()
    rows = store.summaries_for(args.group, limit=max(1, args.limit))
    if not rows:
        print(
            "还没有总结。被 @ 总结成功后、或消息攒够自动总结时都会写入；也可以用 "
            "`qqbot ask \"...\" --save` 造一篇。"
        )
        return 0

    for row in rows:
        start = str(row["ts_start"] or "")[:16]
        end = str(row["ts_end"] or "")[:16]
        window = f"{start} ~ {end}" if start and end else "时间范围未知"
        indexed = "已索引" if row["indexed_at"] else "待索引"
        origin = "自动" if row["trigger"] == "auto" else "被@"
        print(
            f"[{group_label(row['group_openid'])}] {window} · "
            f"{row['message_count']} 条 · {indexed} · {origin} · {row['created_at']}"
        )
        preview = " ".join((row["content"] or "").split())
        print(f"  {preview[:120]}{'…' if len(preview) > 120 else ''}")
    print(f"\n共 {len(rows)} 篇。")
    store.close()
    return 0


def _cmd_reindex(args: argparse.Namespace) -> int:
    from src.rag.indexer import SummaryIndexer
    from src.rag.retriever import get_summary_index
    from src.store.sql_store import get_sql_store

    store = get_sql_store()
    index = get_summary_index()

    if args.reset:
        # Both sides must be cleared together, or `drain` would skip exactly the
        # summaries whose vectors we just deleted.
        dropped_marks = store.clear_summary_marks()
        index.reset()
        print(f"已重置：清空 {dropped_marks} 条索引标记与全部向量。")

    indexer = SummaryIndexer(store, index)

    async def drain() -> int:
        return await indexer.drain()

    total = asyncio.run(drain())
    print(f"索引完成：本次写入 {total} 篇，集合内共 {index.count()} 篇。")
    store.close()
    return 0


def _cmd_media(args: argparse.Namespace) -> int:
    """图片附件处理状态一览（只读）。删文件=删语料，所以这里没有删除动词。"""
    from src.store.sql_store import get_sql_store

    store = get_sql_store()
    rows = store.media_rows(args.status, limit=max(1, args.limit))
    if not rows:
        print(f"没有状态为 {args.status!r} 的图片附件。")
        store.close()
        return 0
    for row in rows:
        short = row["media_id"][:8]
        name = row["filename"] or "(无名)"
        tail = row["path"] or row["last_error"] or ""
        mark = "已看图" if row["description"] else ""
        print(
            f"[{short}] {row['status']:<8} {name} · {row['group_openid'][:12]}… "
            f"· 第 {row['attempts']} 次 {mark} {tail}"
        )
    print(f"\n共 {len(rows)} 行（status={args.status}）。")
    store.close()
    return 0


def _pick_group(store) -> str | None:
    """Resolve the group for `ask` when `--group` was not given."""
    known = [row["group_openid"] for row in store.stats()]
    if not known:
        print(
            "❌ 库里还没有消息，先运行机器人收集一些，或用 --group 指定。",
            file=sys.stderr,
        )
        return None
    if len(known) == 1:
        print(f"（库中只有一个群，使用 {known[0]}）")
        return known[0]
    print("❌ 库中有多个群，请用 --group 指定其一，或用 --all 跨群检索：", file=sys.stderr)
    for gid in known:
        print(f"   {gid}", file=sys.stderr)
    return None


def _cmd_ask(args: argparse.Namespace) -> int:
    from src.agent.summarizer import Summarizer, store_summary
    from src.rag.retriever import get_summary_index
    from src.store.sql_store import get_sql_store

    store = get_sql_store()
    summarizer = Summarizer(store, get_summary_index())

    if args.all:
        if args.save:
            print(
                "⚠️  --all 走的是私聊 scope，其回答不作为文档入库，--save 已忽略。",
                file=sys.stderr,
            )
        result = asyncio.run(summarizer.answer_private("cli", args.instruction))
        print(result.text)
        store.close()
        return 0

    group = args.group or _pick_group(store)
    if not group:
        store.close()
        return 2

    # `--save` is what opens the publishing gate for this run: the agent here is
    # the *same* object the bot uses, and its tools are bound at construction, so
    # the only per-call way to stop a hand-run query from writing documents is
    # `allow_publish=False`. Without it the model may still *ask* to publish and
    # be told no.
    result = asyncio.run(
        summarizer.summarize_group(
            group, args.instruction, allow_publish=bool(args.save)
        )
    )
    print(result.text)

    if result.published:
        # The model published on its own (only possible with --save, per above).
        print(
            f"\n（agent 投稿入库 {len(result.published_ids)} 篇："
            f"{'、'.join(i[:8] for i in result.published_ids)}；"
            "运行 `qqbot reindex` 建索引。）"
        )
    elif args.save:
        summary_id = store_summary(
            store,
            group_openid=group,
            instruction=args.instruction,
            requested_by=None,
            result=result,
        )
        if summary_id:
            print(f"\n（已入库 {summary_id}，运行 `qqbot reindex` 建索引。）")
        else:
            print(
                "\n（这次回答不构成可入库的总结：没有取数，或内容过短；"
                "模型自己也没有调用 save_summary。）"
            )
    store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    # First and only configuration point — see the note on `logger` above.
    setup_logger(console=args.verbose)
    try:
        return args.handler(args)
    except Exception as exc:  # noqa: BLE001 - a CLI must exit with a message, not a trace
        logger.exception("命令执行失败")
        # The message matters here: the likeliest failures are configuration ones
        # (missing model, bad key), and sending the user to a log file for those
        # is worse than echoing the one line that says what to fix.
        print(f"❌ {args.command} 执行失败：{exc}\n   详见 logs/app.log", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
