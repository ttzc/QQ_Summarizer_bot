"""后台图片抓取器——只落盘，零 LLM（MEDIA.md D5）。

形态刻意对齐 `SummaryIndexer`：backlog 在 SQLite（`media.status='pending'`），
崩溃/重启后从库里继续，永不重复抓已存好的图；`wake()` 是同步段的 O(1) 拨铃。
职责边界与全项目一致：**事件回调不做网络，store 里不 await**——下载是这里的
asyncio 活，写盘进 `to_thread`，落完只留一条原子 UPDATE 给 store。

为什么"只落盘"就能解掉 URL 时效：`rkey` 是限时签名链接（实测 17 分钟内可下载），
队列正常秒级~分钟级排空，抓下来的字节此后与外链无关；"看图"发生在
`view_image` 工具里，按需、一次性、结果缓存。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from src.config import config
from src.logger import setup_logger
from src.media.sniff import sniff_image
from src.store.sql_store import SQLStore

logger = setup_logger("qqbot.media.worker")

# (url, timeout_s) -> (HTTP 状态码, 字节)。可替换的注入点——脱机测试用假下载器。
HttpGet = Callable[[str, int], Awaitable[tuple[int, bytes]]]


def _date_bucket(ts: Any) -> str:
    """日期桶 = **消息发送日期**（D1）：跨天重试也落回原桶。解析失败退到今天——
    宁可在桶命名上退让，也不让一条消息因为时间戳格式而丢失图片。"""
    try:
        return datetime.fromisoformat(str(ts)).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return datetime.now().strftime("%Y-%m-%d")


class MediaWorker:
    def __init__(
        self,
        store: SQLStore,
        *,
        http_get: HttpGet | None = None,
        root: Path | None = None,
        batch: int | None = None,
        max_bytes: int | None = None,
        attempts_max: int | None = None,
        download_timeout_s: int | None = None,
        interval_s: float | None = None,
        backoff_s: float | None = None,
    ) -> None:
        cfg = config.media
        self._store = store
        # 注入点优先；默认实现走**共享** AsyncClient（见 _default_get）。
        self._http_get = http_get if http_get is not None else self._default_get
        self._client: Any = None  # httpx.AsyncClient，惰性建、aclose() 收
        # root 是 `[media].dir` 的落点；构造时读配置，测试传 tmp_path。
        self._root = Path(root) if root is not None else cfg.dir_path
        self._batch = batch if batch is not None else cfg.batch
        self._max_bytes = max_bytes if max_bytes is not None else cfg.max_bytes
        self._attempts_max = (
            attempts_max if attempts_max is not None else cfg.attempts_max
        )
        self._download_timeout = (
            download_timeout_s
            if download_timeout_s is not None
            else cfg.download_timeout_s
        )
        self._interval = interval_s if interval_s is not None else cfg.interval_s
        self._backoff = backoff_s if backoff_s is not None else cfg.backoff_s
        self._wake = asyncio.Event()

    def wake(self) -> None:
        """告知有新图待抓。处理器回调里调用——只是 set 一个 Event，同步、免费。"""
        self._wake.set()

    async def _default_get(self, url: str, timeout: int) -> tuple[int, bytes]:
        """默认下载器：**一个进程一个连接**。

        队列排空速度是 M2 全套设计唯一与 URL 时效赛跑的东西；每张图一次冷 TLS
        握手等于亲手把尾部图片推向过期。httpx 惰性导入：假件场景和 `--help`
        都不该为它付导入成本（与 embedding 侧同一规矩）。
        """
        import httpx

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._download_timeout, follow_redirects=True
            )
        response = await self._client.get(url)
        return response.status_code, response.content

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def process_once(self) -> int:
        """处理至多一批。返回本次成功落盘的行数。"""
        rows = self._store.pending_media(self._batch)
        stored = 0
        for row in rows:
            data = dict(row)
            try:
                if await self._process_one(data):
                    stored += 1
            except Exception as exc:  # noqa: BLE001 - 一行失败不停整批
                logger.exception("media 行处理异常", extra={"media_id": data["media_id"]})
                await self._retryable(data, f"internal: {exc!r}"[:200])
        if stored:
            logger.debug("media 批次完成", extra={"fetched": len(rows), "stored": stored})
        return stored

    async def _process_one(self, row: dict) -> bool:
        media_id = row["media_id"]
        name = row["filename"] or "图片"

        if not row["url"]:
            self._store.note_media_failure(
                media_id, "failed", f"[图片 {name}：无下载链接]", error="no url"
            )
            return False
        # 先看事件声称的大小，能不下就不下；下载后仍按真实字节复核。
        if int(row["event_size"] or 0) > self._max_bytes:
            self._store.note_media_failure(
                media_id,
                "skipped",
                f"[图片 {name}：超过大小上限]",
                error=f"event_size {row['event_size']} > {self._max_bytes}",
            )
            return False

        try:
            status, data = await self._http_get(row["url"], self._download_timeout)
        except Exception as exc:  # noqa: BLE001 - 网络异常是可重试失败
            await self._retryable(row, f"download error: {exc!r}"[:200])
            return False

        if 400 <= status < 500:
            # 签名失效/资源不存在：重试没有意义，也不该再往下耗 LLM。
            self._store.note_media_failure(
                media_id, "expired", f"[图片 {name}：链接已过期]", error=f"HTTP {status}"
            )
            return False
        if status >= 500 or not data:
            await self._retryable(row, f"HTTP {status}")
            return False

        if len(data) > self._max_bytes:
            self._store.note_media_failure(
                media_id,
                "skipped",
                f"[图片 {name}：超过大小上限]",
                error=f"bytes {len(data)} > {self._max_bytes}",
            )
            return False
        sniffed = sniff_image(data)
        if sniffed is None:
            # **不当作 skipped**：附件的 content_type 声明了是图片，字节却不是
            # 图片——最常见的真相不是"QQ 发了 HEIC"，而是 CDN 对过期签名回了
            # 200 + 错误页。skipped 是终态，会把它永久判死；判成可重试，让
            # attempts 去收敛（真不支持的格式最终也是 failed，殊途同归）。
            await self._retryable(row, "200 but body is not a whitelisted image")
            return False

        ext, _mime = sniffed
        sha = hashlib.sha256(data).hexdigest()
        existing = self._store.find_media_path_by_sha(sha)
        if existing is not None:
            path = existing  # 全局去重（D1）：同一字节永远只有一份文件
        else:
            path = await asyncio.to_thread(
                self._write_file, data, _date_bucket(row.get("message_ts")), sha, ext
            )

        self._store.mark_media_stored(media_id, sha, path)
        logger.debug("图片已落盘", extra={"media_id": media_id, "path": path})
        return True

    async def _retryable(self, row: dict, error: str) -> None:
        """可重试失败：记一次 attempts；用尽才转 failed 终态并留可见说明。

        转 failed 时**不传** error——保留上一次 `bump_media_attempt` 写进去的
        技术原因（HTTP 码/异常），`qqbot media --status failed` 才有东西可查。
        """
        self._store.bump_media_attempt(row["media_id"], error)
        if int(row["attempts"] or 0) + 1 >= self._attempts_max:
            name = row["filename"] or "图片"
            self._store.note_media_failure(
                row["media_id"], "failed", f"[图片 {name}：抓取失败]", error=None
            )

    def _write_file(self, data: bytes, bucket: str, sha: str, ext: str) -> str:
        """tmp+rename 原子写盘。返回相对 `[media].dir` 的路径（`<日期>/<sha>.<ext>`）。

        已存在同名文件（并发重启后的幂等路径）则直接复用。
        """
        directory = self._root / bucket
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / f"{sha}.{ext}"
        if not final.exists():
            tmp = directory / f".{sha}.{ext}.tmp"
            tmp.write_bytes(data)
            os.replace(tmp, final)  # 同目录 rename：读者永远只会看到完整文件
        return f"{bucket}/{sha}.{ext}"

    def _sweep_tmp(self) -> int:
        """清掉孤儿 `.{sha}.ext.tmp`——它们只可能来自上一次进程在 write 与
        rename 之间被杀（部署/OOM/Ctrl-C）。本进程此刻还没开始写，目录里躺着
        的每一个 tmp 都是死的。best-effort：清不掉不阻塞启动。"""
        removed = 0
        try:
            for tmp in self._root.rglob(".*.tmp"):
                try:
                    tmp.unlink()
                    removed += 1
                except OSError:
                    pass
        except OSError:
            pass  # 目录还不存在
        return removed

    async def run_forever(self) -> None:
        if not config.media.enabled:
            logger.info("media Worker 未启用（[media].enabled=false），行仍入库待补跑")
            return
        swept = await asyncio.to_thread(self._sweep_tmp)
        if swept:
            logger.info("清理了上次中断留下的临时文件", extra={"count": swept})
        logger.info("media 抓取任务启动", extra={"interval_s": self._interval})
        try:
            while True:
                # 先清铃再干活：处理中新来的 wake 不会丢，下面等待会立刻返回。
                self._wake.clear()
                try:
                    stored = await self.process_once()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - 坏批次不能结束循环
                    logger.exception("media 批次失败，退避后重试")
                    await asyncio.sleep(self._backoff)
                    continue

                if stored:
                    continue  # 积压可能还有，不等间隔继续排空
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self._interval)
                except asyncio.TimeoutError:
                    pass
        finally:
            await self.aclose()
