"""M2 图片管线：media 行随消息同事务落库、MediaWorker 只落盘、view_image 按需看图。

假件只有两个注入点——下载器（`http_get`）与一次性视觉调用（monkeypatch
`src.agent.tools.describe_image`）；SQLite、文件写盘、SQL 替换锚点全部真实。
方案文档 docs/MEDIA.md，表语义 docs/DATA_MODEL.md §2.7。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conftest import (
    IMAGE_MESSAGE,
    TEXT_MESSAGE,
    VOICE_MESSAGE,
    FakeAPI,
    FakeSummarizer,
    frame,
)
from src.agent.tools import SCOPE_ALL, SCOPE_GROUP, BotContext, DATA_TOOLS, view_image
from src.bot.client import SummarizerClient
from src.bot.events import GroupMessageRecord
from src.config import config
from src.media.sniff import sniff_image
from src.media.worker import MediaWorker

JPEG_BYTES = b"\xff\xd8\xff" + b"fake-jpeg-body" * 8
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"body-body"
GIF_BYTES = b"GIF89a" + b"x" * 16
WEBP_BYTES = b"RIFF\x10\x00\x00\x00WEBPVP8L"
NOT_IMAGE = b"%PDF-1.4 not an image"


def make_http(status: int = 200, data: bytes = JPEG_BYTES, error: Exception | None = None):
    """A fake downloader that counts calls. `error` raises instead of returning."""
    calls = {"count": 0}

    async def http_get(url: str, timeout: int) -> tuple[int, bytes]:
        calls["count"] += 1
        if error is not None:
            raise error
        return status, data

    http_get.calls = calls
    return http_get


def make_describe(text: str = "一张界面设计稿，左侧导航含消息与总结两栏", error: Exception | None = None):
    calls = {"count": 0, "prompts": []}

    async def describe(data: bytes, mime: str, prompt: str) -> str:
        calls["count"] += 1
        calls["prompts"].append(prompt)
        if error is not None:
            raise error
        return text

    describe.calls = calls
    return describe


class _RT:
    """Stand-in runtime (same trick as test_agent: direct coroutine calls)."""

    def __init__(self, gid, scope=SCOPE_GROUP, *, store):
        self.context = BotContext(
            group_openid=gid, store=store, index=None, scope=scope
        )


@pytest.fixture()
def media_dir(tmp_path, monkeypatch) -> Path:
    """Point `[media].dir` at the test sandbox — worker and tool both resolve it live."""
    root = tmp_path / "media"
    monkeypatch.setattr(config.media, "dir", str(root))
    return root


async def _store_one_image(store, http_get, *, rec=None):
    rec = rec or GroupMessageRecord.from_payload(frame(IMAGE_MESSAGE, "E1"))
    assert store.insert_messages([rec]) == 1
    worker = MediaWorker(store, http_get=http_get)
    assert await worker.process_once() == 1
    return rec


# ---- 落库：media 行与消息同事务 ------------------------------------------------


def test_insert_creates_media_rows_with_short_id(store) -> None:
    rec = GroupMessageRecord.from_payload(frame(IMAGE_MESSAGE, "E1"))
    assert store.insert_messages([rec]) == 1, "消息插入成功"

    rows = store.pending_media(10)
    assert len(rows) == 1, "带图消息同事务插一行 media（pending）"
    row = rows[0]
    short = row["media_id"][:8]
    assert row["placeholder"] == f"[图片 photo.jpg #{short}]", "占位符嵌了 8 位短 id"

    content = store.recent_messages("G_demo", 10)[0]["content"]
    assert f"[图片 photo.jpg #{short}]" in content, "正文占位与 placeholder 逐字一致（回写锚点）"
    assert row["message_id"] == rec.message_id and row["group_openid"] == "G_demo"

    # 重复事件：消息插不进去，media 分支根本不执行——去重免费。
    assert store.insert_messages([rec]) == 0, "重复推送 0 条"
    assert len(store.pending_media(10)) == 1, "重复推送不重复排队"

    # 语音与纯文本不产生 media 行（M2 只管图片形态）。
    for payload in (VOICE_MESSAGE, TEXT_MESSAGE):
        store.insert_messages([GroupMessageRecord.from_payload(frame(payload, "X"))])
    assert len(store.pending_media(10)) == 1, "语音/文本附件不入图片队列"


# ---- worker：只落盘，零 LLM ----------------------------------------------------


async def test_worker_stores_file(store, media_dir) -> None:
    rec = await _store_one_image(store, make_http())
    row = store.media_rows("stored")[0]

    sha = hashlib.sha256(JPEG_BYTES).hexdigest()
    assert row["sha256"] == sha and row["status"] == "stored"
    # IMAGE_MESSAGE 的 ts 是 2026-07-21 → 日期桶取消息发送日期（D1）。
    assert row["path"] == f"2026-07-21/{sha}.jpg", "path 相对 [media].dir，含日期桶"
    assert (media_dir / row["path"]).read_bytes() == JPEG_BYTES, "字节完整落盘"
    assert row["description"] is None, "worker 不做任何视觉调用、不回写描述"

    pending = store.pending_media(10)
    assert pending == [], "stored 之后不再回到队列"
    content = store.recent_messages("G_demo", 10)[0]["content"]
    # 占位仍带短 id 原样在正文里——描述要等 view_image 按需产生（D2）。
    assert f"[图片 photo.jpg #{row['media_id'][:8]}]" in content


async def test_worker_reuses_path_across_days(store, media_dir) -> None:
    await _store_one_image(store, make_http())
    later = {
        **IMAGE_MESSAGE,
        "id": "ROBOT1.0_image_later",
        "timestamp": "2026-07-22T09:30:00+08:00",
    }
    await _store_one_image(store, make_http(), rec=GroupMessageRecord.from_payload(frame(later, "E2")))

    rows = store.media_rows("stored")
    assert len(rows) == 2, "同图第二次出现是新的一行（一次出现一行）"
    assert rows[0]["path"] == rows[1]["path"], "全局去重：跨日期桶也复用旧文件"
    assert rows[1]["path"].startswith("2026-07-21/"), "桶由首次入库日期决定"
    files = list(media_dir.rglob("*.jpg"))
    assert len(files) == 1, "磁盘上只有一份"


async def test_worker_expired_and_retry_policy(store, media_dir) -> None:
    rec = GroupMessageRecord.from_payload(frame(IMAGE_MESSAGE, "E1"))
    store.insert_messages([rec])
    worker = MediaWorker(store, http_get=make_http(status=404, data=b""))
    assert await worker.process_once() == 0
    row = store.media_rows("expired")[0]
    assert row["path"] is None, "expired 没有落盘"
    assert await worker.process_once() == 0 and store.pending_media(10) == []

    content = store.recent_messages("G_demo", 10)[0]["content"]
    assert "[图片 photo.jpg：链接已过期]" in content, "取数里写明看不了，模型不会去调必败工具"

    # 网络/5xx：attempts 累加，用尽才 failed。
    rec2 = GroupMessageRecord.from_payload(frame({**IMAGE_MESSAGE, "id": "R2"}, "E2"))
    store.insert_messages([rec2])
    flaky = MediaWorker(store, http_get=make_http(status=503, data=b""), attempts_max=2)
    await flaky.process_once()
    assert store.pending_media(10), "第一次 5xx 仍在队列里等重试"
    assert store.pending_media(10)[0]["attempts"] == 1
    await flaky.process_once()
    assert store.media_rows("failed"), "attempts 用尽转 failed 终态"


async def test_worker_skips_format_and_size(store, media_dir) -> None:
    store.insert_messages([GroupMessageRecord.from_payload(frame(IMAGE_MESSAGE, "E1"))])
    worker = MediaWorker(store, http_get=make_http(data=NOT_IMAGE))
    await worker.process_once()
    assert store.media_rows("skipped"), "魔数不在白名单 → skipped"
    assert store.pending_media(10) == [], "skipped 不重试，也不耗任何 LLM"

    # event_size 预判：超限直接跳过，连下载都不发生。
    big = {
        **IMAGE_MESSAGE,
        "id": "BIG1",
        "attachments": [{**IMAGE_MESSAGE["attachments"][0], "size": 1 << 40}],
    }
    store.insert_messages([GroupMessageRecord.from_payload(frame(big, "E2"))])
    counted = make_http()
    worker2 = MediaWorker(store, http_get=counted, max_bytes=1 << 20)
    await worker2.process_once()
    assert counted.calls["count"] == 0, "event_size 超限时不下载"
    assert store.media_rows("skipped"), "event_size 预判 → skipped"

    # 真实字节复核：事件没报大小，也要在下载后按实际字节拦下。
    sneaky = {
        **IMAGE_MESSAGE,
        "id": "BIG2",
        "attachments": [{k: v for k, v in IMAGE_MESSAGE["attachments"][0].items() if k != "size"}],
    }
    store.insert_messages([GroupMessageRecord.from_payload(frame(sneaky, "E3"))])
    worker3 = MediaWorker(store, http_get=make_http(data=JPEG_BYTES * 10), max_bytes=64)
    await worker3.process_once()
    assert store.pending_media(10) == []


def test_sniff_whitelist() -> None:
    assert sniff_image(JPEG_BYTES) == ("jpg", "image/jpeg")
    assert sniff_image(PNG_BYTES) == ("png", "image/png")
    assert sniff_image(GIF_BYTES) == ("gif", "image/gif")
    assert sniff_image(WEBP_BYTES) == ("webp", "image/webp")
    assert sniff_image(NOT_IMAGE) is None


# ---- view_image：按需看图 + 缓存回写 --------------------------------------------


async def test_view_image_describe_and_cache(store, media_dir, monkeypatch) -> None:
    await _store_one_image(store, make_http())
    fake = make_describe()
    monkeypatch.setattr("src.agent.tools.describe_image", fake)
    row = store.media_rows("stored")[0]
    short = row["media_id"][:8]

    out = await view_image.coroutine(media_ref=short, runtime=_RT("G_demo", store=store))
    assert out.startswith("[图片描述]") and "界面设计稿" in out, "首次看图：现场调视觉"
    assert fake.calls["count"] == 1
    assert fake.calls["prompts"][0] == config.media.describe_prompt, "无 focus 用配置的通用描述 prompt"

    again = await view_image.coroutine(media_ref=short, runtime=_RT("G_demo", store=store))
    assert fake.calls["count"] == 1, "第二次命中缓存，不再调视觉"
    assert "界面设计稿" in again

    saved = store.media_rows("stored")[0]
    assert saved["description"], "通用描述已缓存进 media.description"
    content = store.recent_messages("G_demo", 10)[0]["content"]
    assert f"[图片 photo.jpg #{short}：一张界面设计稿" in content, "占位被描述替换且保留短 id"

    # 幂等：第二次保存不会再次替换正文（守卫 = description IS NULL）。
    assert store.save_media_description(row["media_id"], "另一段描述") is False
    content_after = store.recent_messages("G_demo", 10)[0]["content"]
    assert "另一段描述" not in content_after


async def test_view_image_focus_not_cached(store, media_dir, monkeypatch) -> None:
    await _store_one_image(store, make_http())
    fake = make_describe(text="图里没有提到上线时间")
    monkeypatch.setattr("src.agent.tools.describe_image", fake)
    short = store.media_rows("stored")[0]["media_id"][:8]

    out = await view_image.coroutine(
        media_ref=short, focus="有没有提到上线时间", runtime=_RT("G_demo", store=store)
    )
    assert out.startswith("[图片定向回答]") and "上线时间" in out
    assert "有没有提到上线时间" in fake.calls["prompts"][0], "focus 进 prompt"
    row = store.media_rows("stored")[0]
    assert row["description"] is None, "定向回答不落缓存"


async def test_view_image_group_guard_and_states(store, media_dir, monkeypatch) -> None:
    await _store_one_image(store, make_http())
    monkeypatch.setattr("src.agent.tools.describe_image", make_describe())
    short = store.media_rows("stored")[0]["media_id"][:8]

    denied = await view_image.coroutine(media_ref=short, runtime=_RT("G_other", store=store))
    assert "无权" in denied, "群内 scope 拿别群短 id → 拒绝"
    allowed = await view_image.coroutine(
        media_ref=short, runtime=_RT(None, SCOPE_ALL, store=store)
    )
    assert allowed.startswith("[图片描述]"), "私聊 scope 本就跨群，放行"

    assert "找不到" in await view_image.coroutine(media_ref="abcdef", runtime=_RT("G_demo", store=store))
    assert "找不到" in await view_image.coroutine(media_ref="abc", runtime=_RT("G_demo", store=store)), "短 id 至少 6 位"

    # 未落盘的行：pending 态可读但看不来。
    store.insert_messages(
        [GroupMessageRecord.from_payload(frame({**IMAGE_MESSAGE, "id": "P1"}, "E2"))]
    )
    pending_row = store.pending_media(10)[0]
    out = await view_image.coroutine(
        media_ref=pending_row["media_id"][:8], runtime=_RT("G_demo", store=store)
    )
    assert "看不了" in out and "pending" in out


async def test_view_image_limit_and_failure(store, media_dir, monkeypatch) -> None:
    for n in ("L1", "L2"):
        store.insert_messages(
            [GroupMessageRecord.from_payload(frame({**IMAGE_MESSAGE, "id": n}, f"E{n}"))]
        )
    worker = MediaWorker(store, http_get=make_http())
    await worker.process_once()
    shorts = [row["media_id"][:8] for row in store.media_rows("stored")]

    monkeypatch.setattr("src.agent.tools.describe_image", make_describe())
    monkeypatch.setattr(config.media, "max_views", 1)
    ctx = _RT("G_demo", store=store)
    first = await view_image.coroutine(media_ref=shorts[0], runtime=ctx)
    assert first.startswith("[图片描述]")
    capped = await view_image.coroutine(media_ref=shorts[1], runtime=ctx)
    assert "上限" in capped, "同一次运行里第 2 张被 max_views 拦下（限的是真实调用）"

    # 视觉调用抛异常 → 可读错误文本，不炸整轮。
    boom = _RT("G_demo", store=store)
    monkeypatch.setattr(
        "src.agent.tools.describe_image", make_describe(error=RuntimeError("gateway down"))
    )
    err = await view_image.coroutine(media_ref=shorts[1], runtime=boom)
    assert "看不了" in err and "gateway down" in err


def test_view_image_not_a_data_tool() -> None:
    assert "view_image" not in DATA_TOOLS, "看图不算取数：不影响 coverage/storable"


# ---- client 接线：带图消息拨铃，普通消息不打扰 ---------------------------------


async def test_message_ingest_wakes_media_only_for_images(store, fake_bot_login) -> None:
    wakes: list[int] = []
    client = SummarizerClient(
        store=store,
        summarizer=FakeSummarizer(),
        wake_media=lambda: wakes.append(1),
        bot_log=True,
        ext_handlers=False,
    )
    client.api = FakeAPI()
    await client._bot_login(None)
    parser = client._connection.parser["group_message_create"]

    parser(frame(IMAGE_MESSAGE, "W1"))
    await client._ingest(GroupMessageRecord.from_payload(frame(IMAGE_MESSAGE, "W1")))
    assert len(wakes) >= 1, "带图消息唤醒抓取器"
    before = len(wakes)
    await client._ingest(GroupMessageRecord.from_payload(frame(TEXT_MESSAGE, "W2")))
    assert len(wakes) == before, "普通消息不碰图片队列"
