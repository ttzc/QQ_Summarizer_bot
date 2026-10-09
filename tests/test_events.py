"""事件解析：`GroupMessageRecord` 直接吃原始 WS 帧（fixtures 对齐官方事件页）。

The point of this module is what botpy's `GroupMessage` throws away —
`author.username`, `message_type`, `msg_elements`, `mentions[].bot` — so the
tests here are mostly "the field survived".
"""

from __future__ import annotations

from datetime import datetime

from src.bot.events import MAX_ELEMENT_DEPTH, Attachment, GroupMessageRecord

from conftest import (
    IMAGE_MESSAGE,
    QUOTE_MESSAGE,
    SUMMON_MESSAGE,
    TEXT_MESSAGE,
    VOICE_MESSAGE,
    frame,
)


def test_text_message() -> None:
    rec = GroupMessageRecord.from_payload(frame(TEXT_MESSAGE, "EVENT_A"))

    assert rec.message_id == "ROBOT1.0_text", "message_id 取自 d.id"
    assert rec.event_id == "EVENT_A", "event_id 取自帧顶层 id"
    assert rec.group_openid == "G_demo", "群标识"
    # botpy 的 GroupMessage._User 读不到 username —— 这是自研事件对象的意义所在
    assert rec.author_name == "小明", "昵称没被丢掉"
    assert rec.member_role == "member", "群成员角色"
    assert rec.msg_idx == "REFIDX_abc==", "msg_idx 从 ext 的 key=value 解析出"
    assert rec.ref_msg_idx is None, "未引用时 ref_msg_idx 为 None"
    assert rec.ts.utcoffset() is not None and rec.ts.hour == 8, "时间戳解析带时区"
    assert rec.raw.get("group_openid") == "G_demo", "原始 d 保留"
    assert not rec.mentions_bot(), "无 mentions 时不算召唤"
    assert rec.to_line() == "08:00 小明: 大家早上好呀", "to_line 可读"


def test_mentions() -> None:
    """@ 触发识别（content 里的 @ 已被平台剥掉）。"""
    rec = GroupMessageRecord.from_payload(frame(SUMMON_MESSAGE))

    assert len(rec.mentions) == 2, "mentions 被解析"
    assert rec.mentions[0].openid == "U9", "mention 的 id 存进 openid"
    assert rec.mentions[0].username == "总结机器人", "mention 昵称"
    assert rec.mentions_bot(), "识别出 @ 了机器人"
    assert rec.body() == "总结一下今天群里聊了什么", "正文即用户指令"


def test_image_message() -> None:
    """图片附件消息。"""
    rec = GroupMessageRecord.from_payload(frame(IMAGE_MESSAGE))

    assert len(rec.attachments) == 1, "附件被解析"
    att = rec.attachments[0]
    assert att.width == 1920 and att.height == 1080, "附件尺寸"
    assert att.label() == "[图片 photo.jpg]", "附件标签为图片"
    assert "[图片 photo.jpg]" in rec.body(), "正文含附件标签"
    assert rec.member_role == "owner", "群主角色"


def test_voice_message() -> None:
    """语音消息（ASR 转写 / 无转写占位）。"""
    rec = GroupMessageRecord.from_payload(frame(VOICE_MESSAGE))
    att = rec.attachments[0]
    assert att.is_voice(), "官方写法 content_type=voice 被识别为语音"
    assert att.label() == "[语音转写 明天下午三点记得交周报]", "转写进 label"
    # 顶部 content 是空白 —— 转写就是这条消息的正文全部（同引用消息的处境）。
    assert rec.body() == "[语音转写 明天下午三点记得交周报]", "正文即转写"
    assert "明天下午三点记得交周报" in rec.to_line(), "to_line 保留转写"
    # 官方还有个 voice_wav_url（QQ 已把 SILK 转成 WAV）。音频不在范围内（M1 取舍），
    # 所以不解析 —— 但逐字 raw 必须原样保留，日后改主意时字段还在。
    assert "wav" not in rec.body(), "voice_wav_url 不进渲染"
    assert rec.raw["attachments"][0]["voice_wav_url"].startswith("https://"), (
        "voice_wav_url 留在 raw 里没丢"
    )

    # 真机落库的附件是 MIME 形态（图片以 image/jpeg 下发），语音可能同样以
    # audio/… 出现，两种写法都要认。
    mime = {
        **VOICE_MESSAGE,
        "attachments": [{**VOICE_MESSAGE["attachments"][0], "content_type": "audio/amr"}],
    }
    assert GroupMessageRecord.from_payload(frame(mime)).attachments[0].is_voice(), (
        "audio/ 形态也识别为语音"
    )

    # 无转写：用明确占位告诉摘要"这里说过话、但没听清"，而不是让这一轮发言
    # 从摘要里悄悄消失。文件名是十六进制串，没有信息量，不进 label。
    silent_att = {k: v for k, v in VOICE_MESSAGE["attachments"][0].items() if k != "asr_refer_text"}
    silent = {**VOICE_MESSAGE, "id": "ROBOT1.0_voice_noasr", "attachments": [silent_att]}
    rec2 = GroupMessageRecord.from_payload(frame(silent))
    assert rec2.body() == "[语音（无转写）]", "无转写时占位且不带文件名"

    # @ 退路：botpy 1.2.1 的 `_Attachments` 根本不解析 asr_refer_text
    # （site-packages 全文零命中），所以那条路上 getattr 恒为 None、只会落进
    # 无转写占位。from_object 仍要把它读上——SDK 哪天补了这个字段就自动生效。
    class _BotpyAtt:
        content_type = "voice"
        filename = "6A3051F3.silk"
        url = "https://u"
        size = 10
        width = None
        height = None
        asr_refer_text = "未来 SDK 会透传的转写"

    assert Attachment.from_object(_BotpyAtt()).label() == "[语音转写 未来 SDK 会透传的转写]", (
        "from_object 在 SDK 提供 asr_refer_text 时透传"
    )


def test_quote_message() -> None:
    """引用/嵌套消息（content 为空白）。"""
    rec = GroupMessageRecord.from_payload(frame(QUOTE_MESSAGE))

    assert rec.message_type == 103, "message_type=103"
    assert rec.ref_msg_idx == "TMP_prev", "ref_msg_idx 解析成功"
    assert len(rec.elements) == 1, "嵌套元素被解析"
    assert rec.elements[0].author_name == "小刚", "元素作者名"
    # 关键：content 是空白，若不拼 msg_elements，这条消息在摘要里会是空的
    assert "明天有空吗" in rec.body(), "正文从 msg_elements 拼出而非为空"
    assert rec.ts.hour == 10, "嵌套元素时间戳独立"


def test_depth_guard() -> None:
    """msg_elements 递归深度保护。"""
    node: dict = {"message_type": 0, "content": "leaf"}
    for _ in range(MAX_ELEMENT_DEPTH + 10):
        node = {"message_type": 102, "content": "wrap", "msg_elements": [node]}
    rec = GroupMessageRecord.from_payload(frame({**TEXT_MESSAGE, "msg_elements": [node]}))

    depth, cursor = 0, rec.elements[0]
    while cursor.children:
        cursor = cursor.children[0]
        depth += 1
    assert depth <= MAX_ELEMENT_DEPTH, f"嵌套被截断到 {MAX_ELEMENT_DEPTH} 层"
    # 旧套件里"未因深嵌套崩溃"是条恒真的垫底断言。换成真行为，但注意：
    # 截断本来就丢弃 MAX 层以下的叶子（护栏的语义），所以断言只能是
    # "渲染整棵被截断的树不抛异常且有产出"，而不是"触到叶子"。
    assert "wrap" in rec.body(), "深嵌套渲染不崩溃且有产出（走的是截断后的树）"


def test_bad_timestamp() -> None:
    """时间戳异常时兜底。"""
    rec = GroupMessageRecord.from_payload(frame({**TEXT_MESSAGE, "timestamp": "not-a-date"}))

    assert isinstance(rec.ts, datetime), "回退到当前时间而非抛异常"
