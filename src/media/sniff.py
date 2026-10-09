"""图片格式的魔数判定（MEDIA.md D6）。

官方文档明示"格式按真实文件内容判定，不看文件名与 MIME"——对 QQ 正好：真机的
`filename` 是十六进制串，`content_type` 存在裸词与 MIME 两种写法。唯一的可信来源
是字节本身。白名单外的返回 `None`，worker 直接标 `skipped`——不浪费任何 LLM 调用。
"""

from __future__ import annotations

# (匹配函数, 存储扩展名, 送模型用的 MIME)。webp 是 RIFF 容器，需要看第 8..12 字节。
_MAGIC: tuple[tuple[bytes, str, str], ...] = (
    (b"\xff\xd8\xff", "jpg", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"GIF87a", "gif", "image/gif"),
    (b"GIF89a", "gif", "image/gif"),
)


def sniff_image(data: bytes) -> tuple[str, str] | None:
    """`bytes -> (ext, mime)`，白名单外为 `None`。

    GIF 动图按首帧理解即可——摘要模型要的是内容不是动画；`[图片 …]` 占位里
    `content_type` 仍原样可查（真机见过 `image/gif`）。
    """
    for magic, ext, mime in _MAGIC:
        if data.startswith(magic):
            return (ext, mime)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ("webp", "image/webp")
    return None
