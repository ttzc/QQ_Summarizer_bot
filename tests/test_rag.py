"""总结级向量索引与检索（真 Chroma / 假 embedding），及后台索引器。"""

from __future__ import annotations

from pathlib import Path

from conftest import FakeEmbeddings
from src.rag.retriever import SummaryIndex, summary_text


def test_embed_batch_defaults() -> None:
    from src.api.embedding_client import MAX_EMBED_BATCH, embedding_batch_size

    assert MAX_EMBED_BATCH == 25, "批量上限的保守默认值是 25"
    assert embedding_batch_size() <= MAX_EMBED_BATCH, "默认批量不超过网关上限"


def _make_index(tmp_path: Path, name: str = "summaries") -> SummaryIndex:
    return SummaryIndex(
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path / "chroma",
        collection_name=name,
    )


def test_summary_text_and_search_scoping(tmp_path, store) -> None:
    # One document per *summary*, not per message.
    store.insert_summary(
        group_openid="G_demo",
        instruction="总结一下",
        content="大家讨论了明天的会议，决定推迟到下午三点，小红负责整理会议纪要。",
        coverage=[("2026-07-21T08:00:00+08:00", "2026-07-21T10:00:00+08:00")],
        message_count=12,
    )
    store.insert_summary(
        group_openid="G_other",
        instruction="总结一下",
        content="群里的结论是会议时间改到周一，另外约了周末一起去爬山。",
        coverage=[("2026-07-21T11:00:00+08:00", "2026-07-21T12:00:00+08:00")],
        message_count=4,
    )
    rows = [dict(r) for r in store.unindexed_summaries(limit=10)]
    assert len(rows) == 2, "取到 2 篇待索引总结"

    text = summary_text(rows[0])
    header = text.splitlines()[0]
    assert header.startswith("[群"), "正文首行是群标签"
    assert "2026-07-21 08:00" in header and "2026-07-21 10:00" in header, "首行含覆盖范围"
    assert "12 条" in header, "首行含条数"
    assert "推迟到下午三点" in text, "正文含总结正文"
    # Retrieval only ever sees `page_content`, so group and window must live
    # in the body — a private-chat answer has no other way to cite them.
    assert summary_text({"content": "   "}) == "", "空正文没有可嵌入文本"

    index = _make_index(tmp_path, "test_summaries")
    assert index.add(rows) == 2, "首次索引 2 篇"
    assert index.count() == 2, "入库计数正确"
    assert index.add(rows) == 2 and index.count() == 2, "重复索引同一批不增容"

    hits = index.search("会议推迟到几点", k=3, group_openid="G_demo")
    assert len(hits) > 0, "群内检索有结果"
    assert hits[0][0].metadata["group_openid"] == "G_demo", "命中的是本群的会议总结"
    assert all(d.metadata["group_openid"] == "G_demo" for d, _ in hits), (
        "结果全部限定在本群"
    )
    assert all("爬山" not in d.page_content for d, _ in hits), "别群的总结未被召回"

    # `group_openid=None` is the private-chat path: deliberately unfiltered.
    everything = index.search("会议", k=5)
    seen = {d.metadata["group_openid"] for d, _ in everything}
    assert seen == {"G_demo", "G_other"}, "不过滤时跨群召回（私聊路径）"

    assert all(
        isinstance(v, (str, int, float, bool))
        for d, _ in everything
        for v in d.metadata.values()
    ), "元数据都是标量"
    assert all(d.metadata["summary_id"] for d, _ in everything), "元数据用 summary_id 作为标识"


async def test_indexer_drains_backlog(tmp_path, store) -> None:
    from src.rag.indexer import SummaryIndexer

    store.insert_summary(
        group_openid="G_demo",
        instruction="总结一下",
        content="大家讨论了明天的会议，决定推迟到下午三点，小红负责整理会议纪要。",
        coverage=[("2026-07-21T08:00:00+08:00", "2026-07-21T10:00:00+08:00")],
        message_count=12,
    )
    store.insert_summary(
        group_openid="G_demo",
        instruction="总结一下",
        content="群里的结论是会议时间改到周一，另外约了周末一起去爬山。",
        coverage=[("2026-07-21T11:00:00+08:00", "2026-07-21T12:00:00+08:00")],
        message_count=4,
    )
    # A blank summary has nothing to embed. It must still be marked, or it
    # would clog the backlog forever.
    store.insert_summary(
        group_openid="G_demo",
        instruction="（空）",
        content="   ",
        coverage=[("2026-07-21T13:00:00+08:00", "2026-07-21T13:01:00+08:00")],
        message_count=0,
    )

    fresh = SummaryIndex(
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path / "chroma2",
        collection_name="drain_summaries",
    )
    indexer = SummaryIndexer(store, fresh, batch=10)
    assert len(store.unindexed_summaries(limit=100)) == 3, "积压初始为 3"
    total = await indexer.drain()
    assert total == 2, "drain 只索引了有正文的 2 篇"
    assert len(store.unindexed_summaries(limit=100)) == 0, "空总结被跳过但已标记"
    assert fresh.count() == 2, "集合内确实只有 2 篇"
    assert await indexer.drain() == 0, "drain 后再跑无事可做"
