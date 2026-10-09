"""一次性"看图"调用，供 `view_image` 工具使用（MEDIA.md D2/D3）。

刻意独立于 agent 循环：一条一次性的 HumanMessage，`ainvoke` 即完——不进
checkpointer、不进任何会话历史，因此 base64 永远不会在后续轮次里被重复计费。
官方 vision 硬约束：**图只能放 user message**（放 system/assistant/tool 直接
400），送图方式选 base64 内联（字节已在本地，不再赌 `rkey` 外链的存活）。
"""

from __future__ import annotations

import base64
from typing import Any

from langchain_core.messages import HumanMessage

from src.api.llm_client import get_chat_model
from src.config import config


async def describe_image(data: bytes, mime: str, prompt: str) -> str:
    """让 `[llm]` 的模型看一张图并回答 `prompt`。返回纯文本。

    无状态、无会话——这是"工具内部的一次调用"，不是"agent 看到了图"。
    任何网关错误原样抛出，由调用方（view_image）翻译成给模型的错误文本。
    """
    url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                # detail 由配置决定（默认 low：缩到 512×512，摘要够用且省 token）。
                "image_url": {"url": url, "detail": config.media.detail},
            },
        ]
    )
    result = await get_chat_model().ainvoke([message])
    return _text_of(getattr(result, "content", ""))


def _text_of(content: Any) -> str:
    """message content 可能是 str 或 block 列表，统一压成文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return "" if content is None else str(content).strip()
