"""图片的存储与按需查看（M2）。

方案文档 `docs/MEDIA.md`；表结构 `docs/DATA_MODEL.md` §2.7。三个部件：

* `sniff.py` — 按真实字节判定图片格式（官方规则：格式不看文件名与 MIME）；
* `worker.py` — 后台 `MediaWorker`，只下载落盘，零 LLM；
* `vision.py` — `view_image` 工具内部使用的一次性看图调用。
"""
