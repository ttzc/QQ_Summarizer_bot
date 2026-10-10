# Changelog

本文件记录 QQ Summarizer Bot 对用户可见的行为变化。

- **正式发布前不打 tag、不用版本号，一律按日期分节**。一节 = 一次改动（同一天可以有多节，**新的在上面**），末尾给 commit 短哈希可回溯完整说明。`pyproject.toml` 里的 `version = "0.1.0"` 从初始提交起没动过，也不代表发布状态。将来真要发布时，把小节标题换成版本号、并补上 tag。
- ⚠️ **这份文件会被机器人自己读到**（`changelog` 工具，群内与私聊都能问"最近新增了什么功能"），所以写在这里的陈述**会被说出口**：改了能力就把对应那节补上，别让"工具集有几个"之类的数字烂在原地。
- 记**行为、能力、配置、存储**四类的变化，另加改变"怎么验证这个项目"的基础设施（测试框架、CI）。纯内部重构与排版不单列。
- ⚠️ **测试计数在 2026-10-10 换过单位**：之前是单个 `test/test_offline.py` 里的**断言条数**（255→258→265→275），迁移到 pytest 之后是**测试项数**（39→51→54→72→78）。两个数字不可直接比较，275 → 39 不是回退。
- 计划中的路线见 [ROADMAP.md](ROADMAP.md)，架构决策的理由见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 2026-10-10 — 机器人能自述改动：`changelog` 工具

用户可以直接问"最近新增了什么功能""那个图片的 bug 什么时候修的""哪次改动动了表结构"，不必翻仓库。

### Added

- **`changelog(sections = 3)`** 工具，**同时进群与私聊两个工具集**（这是第一个两边都有的"非取数"工具——它读的是本项目的开发记录，不属于任何群，也就不存在跨群泄漏面）。返回文件头 + 最近的 N 个日期节（`sections` 夹在 1–10），没列全时**报出总节数**。
- 三种退化路径都不静默交白卷（都会被读成"没有更新记录"）：文件读不到 → 一句实话；分不出 `## ` 节 → 回退原文开头；**单个日期节自己就比预算长** → 硬截断那一节的开头并标注"已截断"。第三种不是假想情形——changelog 只会一节节变长。

### 安全边界（🔴 不可加参数）

- **路径是模块常量，不是工具参数**。代码旁边的 `.env` 里躺着 QQ `clientSecret` 和两个 api key，而模型读的是群里任何人写的文本——只要这个工具能吃文件名，"帮我看下 `.env` 第三行"就是一次凭据外泄。所以唯一的旋钮是"要几节"，它是**尺寸**不是**定位符**；有一条测试直接钉住可见参数集只能是 `{sections}`。
- 读更新记录**不进 coverage**，也不在 `DATA_TOOLS` 里：只查过 changelog 的一轮**不允许投稿**，否则会出现一篇"归属某个群、依据却是我们自己的文档"的稿子。

提交：本节引入时**不写自己的哈希**——记录这条改动的 commit 不可能知道自己的哈希（`--amend` 会换掉它）。哈希在下一次提交时回填；在那之前可用 `git log --oneline -- src/agent/tools.py` 查到。

---

## 2026-10-10 — 入库决定权交给 agent

被 @ 不再必然产生一篇文档。此前每次 @ 都会落库一行，只用三个启发式过滤（非哨兵 / ≥30 字 / 落过地），于是"有人提过 X 吗"这类三行字的答复也进了知识库。现在由 agent 判断这一轮是否值得长期留存并**显式投稿**。

### Added

- **`save_summary(content)`** 工具——知识库的**唯一写入者**。模型自己写好正文，工具直接落库；下限（`min_publish_chars`）与每轮预算（`max_publish_per_run`）由代码守，**不过时把原因作为文本返回给模型**，它能改正再投，而不是被静默丢弃。
- **`summaries_in_range(start_iso, end_iso, limit)`**——按**覆盖时段重叠**查本群已有哪些总结。空返回即答案（"这段还没被写过"）。`limit` 只是截断护栏，命中超出时返回里**必须报出真实总数**。
- **`get_summary(summary_ref)`**——按短 id 读某一篇正文，每轮限 `max_doc_reads` 篇。
- 群内工具集 **5 → 8 个**；私聊工具集仍是 5 个（投稿能力不给私聊）。
  （**上面**那节 `changelog` 工具把群内 / 私聊变成 9 / 6——本节记录的是当时的状态，不改。）
- `[summary]` 新增三个配置项：`min_publish_chars = 200` / `max_publish_per_run = 2` / `max_doc_reads = 3`。
- 可观测性：日志新增 `agent 投稿入库` 与 `答复判定为问答，未入库`——知识库涨得慢时后者是唯一观察抓手（该调的是 prompt 判据，不是门槛数字）。

### Changed

- **群内被 @ 的落库时机提前到 agent 运行中途**（原先是"跑完再存"）。回群失败、甚至 agent 随后崩掉，都丢不了已写下的文档。
- **发到群里的回复变成要点**，全文只进库；正文因此只生成一次。
- 文档的三个来源变为：被 @ 且模型投稿（`trigger='at'`）、到量自动（模型投稿或代码代写，`trigger='auto'`）、离线 `ask --save`。`trigger` 仍只有两个值——它记**入口**，不记决定者。
- **自动总结路径不受投稿管**：它的门禁本就是"距上次入库新增 ≥ `min_messages` 条"，且它是知识库主粮。`store_summary` 在模型已投稿时短路，一次运行只落一行。

### Fixed

- 清单标题位曾把 QQ 的 `<@32位hex>` @ 标记当作正文渲染，40 字预算几乎全喂给 openid；现在**只在渲染标签时**剔掉标记，`instruction` 列本身仍逐字不动。

### 兼容性

- **零结构变更**：不加列、不改 DDL、不动 `_MIGRATIONS`，老库直接可用；也没动 `summary_text()`，**向量库无需重建**。
- 私聊答复仍不入库（`group_openid` 是 NOT NULL，跨群结论没有归属地），投稿工具只进群的工具集。

提交：`1bfce4b`

---

## 2026-10-10 — 图片管线（M2）与其后的 code-review 修复

### Added

- **图片落盘**：后台 `MediaWorker` 只下载落盘、零 LLM——`data/media/<消息日期>/<sha256>.<ext>`，魔数白名单判格式（不看文件名与 MIME），tmp + rename 原子写，同一字节全局去重。新增 `media` 表（一次附件出现一行，`pending → stored/expired/skipped/failed`）。
- **`view_image(media_ref, focus?)`** 工具进群与私聊两个工具集：agent 觉得重要时才看图，一次性视觉调用返回文字；通用描述缓存回写、第二次看零成本，focus 定向回答不落缓存；群归属按行校验，每轮 `max_views` 限流。
- 消息正文里的图片占位带上短 id：`[图片 xx.jpg #4d9f2a1c]`，这就是 `view_image` 的引用方式。
- CLI：`qqbot media`（查看队列状态）、`qqbot stats` 增加图片三格；`[media]` 配置段。
- 群与私聊工具集各 4 → 5 个。

### Fixed

- `find_media` 的归一化顺序（先 `strip` 再去 `#`）：从占位符里截取的 `" #短id"` 此前会被误判为找不到。
- `max_views` 只扣在**真实视觉调用**上：文件缺失、超过内联上限这些没走到模型的路径不再白吃额度。
- HTTP 200 但字节不是图片时不再当场判 `skipped`（CDN 对过期签名常回 200 + 错误页，而 `skipped` 是终态 = 永久丢图），改为按可重试收敛、用尽转 `failed`。
- `MediaWorker` 改持共享 `httpx.AsyncClient`（冷 TLS ×N 正是与签名 URL 时效赛跑的敌人），退出时 `aclose`；启动清扫遗留的 `.{sha}.tmp`。
- 新增 `max_inline_bytes = 4 MiB`：32 MiB 是**落盘**护栏，内联 base64 是另一条线，超限给明确文案且不扣额度。
- `note_media_failure` 不再用展示文案覆盖 `last_error`（保留 HTTP 码 / 异常供排查）。

### 已知边界

- 引用消息 / 合并转发里**嵌套**的图片不入队（M2.x）。
- 决定：LLM 的 **thinking 保持开启**，不注入 `disabled`——用 reasoning token 占预算换推理质量；真机看到截断就调大 `[llm].max_tokens`，而不是关思考。

提交：`07778b8`、`f8a3d43`（方案） · `54b058c`（修复与决定）

---

## 2026-10-10 — 测试与 CI

- 单个 1700 行的 `test/test_offline.py` 迁移为 **pytest 套件**：`tests/` 按主要功能分文件（events / store / sender / rag / agent / client + 后续 media），共享 fixtures 与假件收进 `conftest.py`，248 处旧断言逐条迁移。
- 新增 **GitHub Actions CI**：push / PR 到 main 跑 `uv sync && uv run pytest`，全套脱机、不需要任何凭据。
- dev 依赖走 `[dependency-groups]`，`asyncio_mode = "auto"`。
- 顺手替换了一条恒真断言（"未因深嵌套崩溃"→ 截断树渲染确有产出）。

提交：`37e014c`

---

## 2026-10-09 — 语音消息（M1）与路线图

- **语音 ASR**：取官方 `asr_refer_text` 渲染成 `[语音转写 …]`；**没有转写时渲染 `[语音（无转写）]`**，让摘要至少知道这里说过一次话。语音判定认 `content_type`（`voice` 与 `audio/*` 双写法）——`message_type` 不可用（官方没有语音专属值，官方图片示例与真机库都是 `0`）。
- 官方另有 `voice_wav_url`（QQ 已做 SILK→WAV 转换）：按"音频不入库"的取舍**不解析**，但随逐字 `raw_json` 保留，日后反悔成本为零。
- 新增 [ROADMAP.md](ROADMAP.md)：多模态 M1→M4、会话/记忆 S1→S2 的实现顺序与既定取舍。

提交：`d028c0a`、`6e8a524`

---

## 2026-10-09 — 模型与网关

- `[llm]` 切到 **DeepSeek 官方 `deepseek-flash`**（V4.1 Flash，原生多模态：一个模型同时管文本与视觉，所以**不需要**新增 `[vision]` 段）。`.env` 变量名不变，只换值。
- `[embedding]` 留在原网关——DeepSeek 官方没有 embedding 接口。两个供应商并存正是 `[llm]`/`[embedding]` 分段的设计用意。
- 用项目自己的 Key 实测钉死两件事并写进 CLAUDE.md：模型 ID 以账号 `/v1/models` 为准（`input_modalities` 含 `image`）；**thinking 默认是开的且 reasoning token 计入 `max_tokens`**（压到 64 会 HTTP 200 返回空正文）。
- 排障记录：`code=20041` = 模型不是 VLM，`code=20012` = 模型在本站不存在；都是 HTTP 400 但含义完全不同。

提交：`ecf06ba`、`d576ec5`、`776d5e4`

---

## 2026-10-09 — 配置分层与首次真机联调

### Changed

- **配置按敏感度分层**：`.env` 收缩到 4 个凭据变量（`QQ_APPID` / `QQ_SECRET` / `LLM_API_KEY` / `EMBED_API_KEY`），模型名与 `base_url` 写进 `config.toml`——换网关不必动 `.env`。
- 修掉一个真隐患：`model` 未设置时会把字面量 `"${LLM_MODEL}"` 当模型名发出去；现在未展开的 `${VAR}`、空串统一收敛为 `None`，API 客户端在缺模型名时带明确报错停下，CLI 启动前预检并把原因打到 stderr。
- **`content` 列存的是文本化正文**（`body()`）而非原始 content：引用 / 合并转发 / 语音转写 / 附件标签这些"有文字但不在 content 字段里"的消息，不再以空行进入 prompt；逐字原文仍在 `raw_json`。

### Fixed

- 工具注入触发的 **pydantic 序列化告警**：`runtime` 注解必须是 `ToolRuntime[BotContext, dict]`，裸 `ToolRuntime` 会让 pydantic 用 `ContextT` 的默认值 `None` 解析 `runtime.context`，于是每次工具调用都刷一条看起来像缺陷的 stderr 噪音（功能不受影响——正因如此它值得单独测一次）。

### 文档（真机结论）

- 首次真机跑通并记账：全量消息权限、真实 embedding 网关、真实 LLM 全链路（@ → 落库 → 建索引 → 回复）、私聊链路、自动总结的 `group_busy` 闸门。
- **更正一处错误**：「获取群内全部消息」的开关在**手机 QQ 的群设置**里（群 → 群设置 → 群机器人 → 机器人设置 → 机器人可获取的群聊消息范围），**不在开放平台、不需要审核**；此前文档写的"需在开放平台申请并通过审核"是错的。并给出判别事件走了全量路径还是 @ 退路的方法（看 `event_id` / `author_name` / `raw_json` 三个字段）。
- 补 MIT 许可证（`LICENSE`）。

提交：`eceecd2`、`76e4d04`、`972fb4c`、`8a30fad`

---

## 2026-10-09 — 初始提交

第一版可用实现，确立了本项目的所有承重结构：

- **补 botpy 缺失的事件解析器**：SDK 1.2.1 不认识 `GROUP_MESSAGE_CREATE`（会打一行日志后丢弃），改为继承 `Client` 在 `_bot_login` 里注册解析器；配套自研 `GroupMessageRecord` 直接吃原始 payload，保住被 `GroupMessage` 丢掉的**发言人昵称** / `message_type` / `msg_elements` / `mentions`。
- **三层存储**：SQLite 存原文全量与总结表（唯一真相源，`INSERT OR IGNORE` 按 `message_id` 去重），Chroma **一篇总结一个向量文档**，`InMemorySaver` 存多轮会话（LRU 上限 128 条线程，群与私聊共享）。
- **两个 agent，权限方向刻意相反**：群内检索恒带群过滤且**不含任何跨群能力**；私聊可跨群检索总结并按时间范围读原文。能力差异由工具集表达，不由运行时 `if` 表达。群标识由 `ToolRuntime` 服务端注入，模型碰不到也伪造不了。
- **按消息量触发的自动总结**：判定全在同步段（不碰网络），冷却**先写后试**、失败也消耗冷却，避免网关故障时逐条轰炸。
- **被动回复合规**：分段 ≤5 条、`msg_seq` 递增、接近 5 分钟窗口降级为主动消息（私聊不降级）。
- 事件回调里绝不落 embedding：落库与建索引拆成"同步快写 + 后台批处理"，进度用 `summaries.indexed_at` 传递。

脱机测试 255 条断言全绿。

提交：`eb75d5a`
