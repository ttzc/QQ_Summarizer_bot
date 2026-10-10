# ROADMAP

记录计划中的功能与其取舍。现状与约束的权威描述见 [`CLAUDE.md`](CLAUDE.md)（§5.1 多模态选型、§4.1.3 自动总结）与 [`docs/`](docs/)；本文只写**要做什么、按什么顺序、为什么这个顺序**。

## 版本总览

| 里程碑 | 内容 | 依赖 |
|:---|:---|:---|
| **M1** | 语音 ASR 文本入库（不存音频文件） | 无 — 真机验证为主 |
| **M2** | 图片的描述入库与读取 | M1；图片 URL 时效验证 |
| **M3** | PDF 转图，逐页描述入库与读取 | M2；PyMuPDF |
| **M4** | docx / ppt 等先转 PDF，复用 M3 管线 | M3；LibreOffice headless |
| **S1** | 私聊会话：按用户分离（已就位，补测试） | 无 |
| **S2** | 私聊 / 群会话持久化（SqliteSaver） | `uv add langgraph-checkpoint-sqlite` |

多模态线与会话线相互独立，可并行推进。编号即实现顺序。

---

## 多模态线

### 既定的取舍（先钉死，避免每个里程碑重新讨论）

1. **不存音频文件本身。** 能直接读音频的全模态模型太贵；QQ 官方在语音消息上已经附带 `asr_refer_text`（服务端转写文本），我们只存这份文本。没有转写的语音，就没有可用信息——接受这个损失。
2. **文档类附件一律降级为图片。** PDF/docx/ppt 都走"转成页面图片 → 视觉模型描述 → 描述文本入库"，与图片同一条语义管线。系统里**永远只有文本进向量库**，Chroma、检索、总结、prompt 预算全部不用改。
3. **转换发生在后台批处理，绝不进事件回调。** 事件回调里连 embedding 都不许调（CLAUDE.md §4.1），取图/转图/调视觉模型比 embedding 更慢更贵，同理。落库只记"这个附件待转写"，后台消化。
4. **视觉能力不需要新模型段。** `[llm]` 已是 `deepseek-flash`（V4.1 Flash，原生多模态，`input_modalities = ["text","image"]`），CLAUDE.md §5.1.1 已定案，不加 `[vision]` 段。

### M1 · 语音 ASR 文本入库

现状：`Attachment`（`src/bot/events.py`）已解析 `asr_refer_text`，`label()` 已渲染成 `[语音转写 …]` 并拼进 `body()` ——**有转写的语音其实已经在落库路径上**。缺的是确认与补齐：

- [x] 无转写时的占位改为 `[语音（无转写）]`（`Attachment.label()`，2026-10-09）。语音判定认 `content_type` 的 `voice` 与 `audio/*` 两种形态——官方事件页（2026-09-16 版）的枚举就是裸词 `voice`（同页却称该列是"MIME 类型"，且图片确实以 `image/jpeg` 到线），两边都收。
- [x] 官方文档核实 MessageAttachment：**没有语音专属的 `message_type`**（文档自己的图片示例就是 `message_type: 0`，与真机库一致），检测只能走 `content_type`；`asr_refer_text` 官方名"语音消息 ASR **参考**结果"（string）——名字自己就不承诺必有；另发现 **`voice_wav_url`**（QQ 已做 SILK→WAV 转换，URL 与图片同款 `rkey` 签名结构），按"音频不入库"的取舍不解析、但随 `raw_json` 原样保留。
- [x] 脱机测试补 `[2b] 语音消息` 组（9 条断言，套件现 **275** 条）：转写进 `label()`/`body()`/落库 `content`；无转写占位不带十六进制文件名；`voice_wav_url` 不进渲染但留在 `raw_json`；`from_object` 在 SDK 透传时生效。
- [x] 已查明：botpy 1.2.1 的 `_Attachments` 不解析 `asr_refer_text`（包内全文零命中），所以 @ 退路恒显示无转写；`from_object` 已按 getattr 读取，SDK 补上字段即自动生效。
- [ ] 真机观察：`asr_refer_text` 在真实语音上的**覆盖率**、长度、原文还是摘要风格。字段的存在与类型已由文档钉死，剩下的只能等第一条真语音落库（**当前语料里语音消息为 0 条**，2026-10-09 清点 `data/qqbot.db`，5 条带附件的全是图片）。
- 不做：音频下载、音频存储、自建 ASR。

### M2 · 图片的存储与读取 —— 方案已定稿：[`docs/MEDIA.md`](docs/MEDIA.md)

现状：图片只渲染成 `[图片 <十六进制串>.jpg]`，信息量≈0（CLAUDE.md §5.1）。库里已经有 `url`，只是没人读。

方案骨架：**落盘解决时效，按需看图解决语义**（定稿细节全部在 MEDIA.md）——**2026-10-10 脱机实现完成**，套件 51 项全绿（本次投稿改造后为 72 项）。

- [x] `data/media/YYYY-MM-DD/<sha256>.<ext>` 本地存图（`src/media/worker.py`）：按**消息发送日期**分桶，内容哈希全局去重（已登记即复用旧 path；tmp+rename 原子写盘）。URL 时效就此降级：队列分钟级排空 ≪ 实测 17 分钟有效期。
- [x] `media` 表（DATA_MODEL §2.7；一行 = 一次附件出现，uuid 主键 + 哈希列文件去重；pending → stored 终态，失败分态 expired / skipped / failed）+ `MediaWorker` 只落盘、零 LLM；`insert_messages` 同事务插行去重免费；占位嵌 8 位短 id：`[图片 xx.jpg #a1b2c3d4]`。
- [x] **`view_image(media_ref, focus?)`** 进群与私聊两个工具集（改造后群 8 个 / 私聊 5 个）：一次性视觉调用（不进主循环/checkpointer），通用描述缓存回写 `description`+`content`（`save_media_description` 的 `description IS NULL` 守卫 = 幂等锁），focus 不落缓存；群归属行级校验 + `max_views` 限流 + 失败返回错误文本。`src/media/vision.py`，不进 `DATA_TOOLS`。
- [x] 图片**不预生成描述**（安静的讨论零成本）；魔数白名单 `src/media/sniff.py`；`qqbot media` 只读排查 + `stats` 图片三格；带图消息才拨 `wake_media`。
- [x] 脱机验收 = `tests/test_media.py` 12 项（假下载器 + 假视觉、真 SQLite + 真写盘），`test_agent.py` 工具集断言已同步 4→5。
- [ ] 真机：第一条真实图片 pending→stored；一次 @ 总结中模型主动 `view_image` 的全链路；隔天重放验证离线积压场景（决定 expired 要不要配人工重放）。
- 不做（本期）：图片进向量库、图搜图；把图块塞进 agent 循环反复放大看像素 → **M2.5** 保留选项（机制代价清单在 MEDIA.md 末节，触发条件出现再立项；与 `view_image` 缓存语义兼容，只加不减）。

### M2.x · 首版边界（2026-10-10 code-review 挂账）

- [ ] **嵌套图片**：引用/合并转发（`msg_elements`）里的图片附件目前不入队列——标签渲进正文但没有 `#短id`，不可查看。修法是让 `MsgElement.render` 的锚点也走 id 注入链。
- ~~全局关 thinking~~ —— **已决定不做（2026-10-10）**：thinking 保持开启，接受 reasoning 占预算的代价。真机若出现截断，调大 `[llm].max_tokens` 而不是关思考（决定记录：CLAUDE.md §5.1.1.2）。
- [ ] `expired` 的人工重放命令：等 §五 真机观察（离线积压一晚的 pending 还能不能下载）出结论再定。

### M3 · PDF 转图

PDF 是 `Attachment(content_type="application/pdf")`，同样先进 `media` 表（M2 的表扩一个文件形态分支即可，见 MEDIA.md），只是处理管线多一步：

- [ ] 依赖 `pymupdf`：打开 PDF → 逐页渲染 PNG（限 DPI，够模型认字即可）→ 逐页走 M2 的"视觉模型描述"→ 拼接成 `[PDF <name>，共 N 页] 第1页: … 第2页: …` 回写。
- [ ] 护栏：页数上限（如 30）、单文件上限（32 MiB，官方外链下载须 60 秒内完成）、超限记 `[PDF 过大，仅转写前 N 页]`。
- [ ] 扫描版 vs 文本版：先试 `page.get_text()`——能抠出文字就直接用文本（免费、准确、快），抠不出才转图走视觉模型。多数"别人发的 PDF"走这条免费路。
- [ ] 脱机测试：生成一个一页文字 PDF + 一页图片 PDF 的 fixture，验证两条分支。

### M4 · docx / ppt → PDF

- [ ] 首选 **LibreOffice headless**（`soffice --headless --convert-to pdf`）：保真度最高，docx/pptx/xlsx/老格式通吃。代价是外部进程依赖，README 要写明安装；子进程超时（如 120s）与并发数限制（同时只转一个）都要做。
- [ ] 纯 Python 兜底（仅当无法部署 LibreOffice 时再考虑）：`python-docx` 抽 docx 文本、`python-pptx` 抽 pptx 文本+备注。**明确这是降级**：排版、图示、贴图全丢，只适合"文字型文档"。
- [ ] 转换产物是临时文件：转完 → 进 M3 管线 → 删除，不落库原件。
- [ ] 未知扩展名 / 转换失败：记 `[文件 <name>，暂不支持]`，入日志，不重试成环。

---

## session / memory 线

### S1 · 私聊会话按用户分离（已就位）

`Summarizer` 的 thread key 已经是 `U:<user_openid>`（群为 `G:<group_openid>`），同一用户跨重启前共享一条私聊会话、不同用户互不可见；`test_agent_boundary` 已断言同后缀的群与私聊不串线程。这条基本是确认项：

- [ ] `qqbot` 加一个只读的会话清单能力（或并入 `stats`）：当前活跃 thread 数、每用户私聊最近活跃时间——排查"这个人是不是被 LRU 淘汰过"用。

### S2 · 持久化（核心工作）

现状：checkpointer 是 `InMemorySaver`，**重启即丢所有会话记忆**（`Summarizer.__init__`，`src/agent/summarizer.py`）；LRU `MAX_THREADS=128` 群与私聊共享预算。

- [ ] `uv add langgraph-checkpoint-sqlite`，`Summarizer` 构造 checkpointer 时换成 `SqliteSaver`（库放 `data/checkpoints.db`，与 `qqbot.db` 分开：checkpoint 是可丢弃的运行态，不配进主库一起备份/迁移）。
- [ ] **重新设计 LRU 的语义**。现在 `_touch_thread` 淘汰时调 `delete_thread`——在内存版里是"丢记忆"，在持久版里是**删数据**。改成两层：内存 LRU 只管锁与热缓存，淘汰不再删库；库侧按 `TTL`（如 30 天未活跃才清）在启动或低频任务里清扫。`MAX_THREADS` 因此可以放大。
- [ ] 群与私聊分开预算（各一半），避免私聊受众把群记忆挤掉——持久化后两边都值得留。
- [ ] `reindex --reset` 之外新增 `qqbot sessions`（或 `reset-session [--user <openid>] [--group <id>]`）：手动清某人的会话，隐私兜底手段。
- [ ] 脱机测试：进程内新建第二个 `Summarizer` 指向同一个 checkpoint 库，断言上一轮对话仍可见（模拟重启）；TTL 清扫按伪造时间戳验证。
- 不做：跨机分布式存储；把私聊**消息内容**另建表存（现在私聊只留会话记忆，不留原文，这是有意的——`group_messages` 的 `NOT NULL group_openid` 也容不下它。若以后想按时间回查"我上周问过什么"，再立一条里程碑，别混进 S2）。

### S3 · （可选，暂不排期）会话压缩

长命 thread 的 `messages` 是 append-only，群会话靠自动总结续命，私聊没人管会无限长。等 S2 落地、看到真实的 thread 长度分布后，再决定要不要用 `trim_messages` 或"旧对话折叠成一段 system 摘要"。现在做是瞎猜预算。

---

## 顺序的理由（备查）

- **ASR 最先**：不是"最容易"，而是**已经做了一半**——`asr_refer_text` 已解析已入库，剩下的是验证与补洞，几乎零风险地先把一类消息的语义覆盖拉满。
- **图片第二**：唯一能直接喂给现有 `deepseek-flash` 的模态，且 URL 已在库；难点只有时效，所以时效验证排在 M2 的第一项。
- **PDF 第三、docx/ppt 第四**：两者都复用前者的产出——PDF 复用图片的视觉描述管线，docx/ppt 复用 PDF 管线。倒过来做等于把同一个坑踩三遍。
