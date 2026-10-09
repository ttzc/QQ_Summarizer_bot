# 数据模型

本项目把数据放在三处，每一处解决一个不同的问题。本文说明它们各自存什么、怎么保证一致，以及若干**必须遵守的约束**。

架构与数据流见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 一、三层总览

| 层 | 位置 | 存什么 | 谁写 | 谁读 | 生命周期 |
|:---|:---|:---|:---|:---|:---|
| **SQLite** | `data/qqbot.db` | 消息全量 + 总结全量 + 索引进度标记 + `media` 附件队列/状态（M2，§2.7） | 事件回调（同步、快，消息与 media 行同事务）、总结入库、后台 MediaWorker | 工具（`view_image`）、MediaWorker、CLI | 永久，唯一真相源 |
| **Chroma** | `data/chroma_db/` | 每篇**总结**一个向量文档 | 后台 indexer（批量、慢） | `search_summaries` 工具 | 可从 SQLite 完全重建 |
| **内存** | 进程内 | agent 多轮会话状态 | `Summarizer` | agent 自身 | 进程退出即丢 |

**唯一真相源是 SQLite。** Chroma 是派生的：`qqbot reindex` 能在向量库丢失或损坏后从 SQLite 全量重建。会话记忆则是纯缓存，丢了只影响追问的连贯性。

注意 **SQLite 里两类数据的角色完全不同**：`group_messages` 是原文，只按时间/条数读，不 embed；`summaries` 是知识文档，是向量的唯一来源。这个划分是"一次总结一篇文档"的直接结果（见 §3.0）。

---

## 二、SQLite

位置：`config.toml` 的 `[store].sqlite_path`（默认 `data/qqbot.db`）。
DDL：`src/store/sql/schema.sql`。

### 2.1 连接与环境设置

```python
sqlite3.connect(path, check_same_thread=False)
PRAGMA journal_mode=WAL      # 允许读写并发，也便于外部工具只读查看
PRAGMA busy_timeout=5000     # 万一有第二个进程，不立即 SQLITE_BUSY
```

- **单连接**，只在事件循环线程上使用；`check_same_thread=False` 是为了满足声明，不是在鼓励跨线程。
- **不加 `asyncio.Lock`**：每个方法内部没有 `await`，在 asyncio 下调用天然原子、任务无法交错。这是不变式——**如果哪个写路径将来引入了 `await`，必须补锁**。详见 `sql_store.py` 的模块 docstring。
- `busy_timeout` 是必需的：短命 CLI 进程可以不管它，但本项目的长驻写服务不能因为偶发的第二个进程就立刻 `SQLITE_BUSY` 失败。

### 2.2 `group_messages` — 消息主表

一条群消息一行。

| 列 | 类型 | 说明 |
|:---|:---|:---|
| `message_id` | TEXT **PK** | QQ 的消息 ID。**主键即去重机制** |
| `event_id` | TEXT | WS 帧顶层的 `id`，与消息 ID 不同；被动回复时二者可择一 |
| `group_openid` | TEXT NOT NULL | 群标识（匿名化，非群号） |
| `author_openid` | TEXT | 发言者标识（匿名化，非 QQ 号） |
| `author_name` | TEXT | 发言人**昵称**——botpy 的 `GroupMessage` 会丢掉，本项目从原始 `d` 里取回 |
| `member_role` | TEXT | `member` / `admin` / `owner` |
| `content` | TEXT | 正文的**文本形态**：原始文本 + 附件占位（图片为 `[图片 photo.jpg #短id]`，`#` 后是 media 短 id，M2）+ 语音转写（或 `[语音（无转写）]`）+ 引用 / 合并转发的元素正文。看过/失败的图片占位会被原位替换成 `[图片 …：描述]` / `[图片 …：已过期]`（§2.7）。**不是逐字原文**——逐字的在 `raw_json`（见 §2.6） |
| `message_type` | INTEGER | 0 文本 / 3 卡片 / 101 并行 / 102 聊天记录 / 103 引用 |
| `ts` | TEXT NOT NULL | ISO8601 |
| `msg_idx` | TEXT | 来自 `message_scene.ext` |
| `ref_msg_idx` | TEXT | 被引用消息的序号，来自 `message_scene.ext` |
| `has_media` | INTEGER | 0/1，是否含图片等附件 |
| `raw_json` | TEXT | 原始 `d` 全量，向前兼容用（目前**只写不读**） |
| `ingested_at` | TEXT NOT NULL | 入库时刻（与 `ts` 区分：`ts` 是发言人发消息的时间） |

索引：

```sql
CREATE INDEX idx_gm_group_ts        ON group_messages(group_openid, ts);
CREATE INDEX idx_gm_group_ingested  ON group_messages(group_openid, ingested_at);
```

第一个覆盖按群取最近 N 条、按群取时间区间这两类查询；第二个专门服务自动总结的计数（§2.3 的 `messages_since_last_summary`）——那条查询在**每条群消息**上都会跑一次，不能退化成全表统计。

**去重**：靠主键 + `INSERT OR IGNORE`。QQ 的下行 payload **没有 `msg_seq`**（那是频道事件才有的字段），所以不能靠序号去重，只能靠消息 `id`。这也顺带解决了"同一条 @ 消息同时以两个事件到达"的问题——两处回调都写库，第二次插入 0 行，`client._ingest()` 据此返回 `False`，机器人只回一次。

### 2.3 `summaries` — 知识文档表

**一次成功的总结一行**（被 @ 触发，或消息到量后自动触发），同时也是向量库里一篇文档的来源。原文表只按时间查，这张表才做语义检索。

| 列 | 类型 | 说明 |
|:---|:---|:---|
| `summary_id` | TEXT **PK** | `uuid4().hex`。**不是内容哈希**，理由见下 |
| `group_openid` | TEXT NOT NULL | 这次总结属于哪个群 |
| `instruction` | TEXT NOT NULL | 触发它的指令原文（自动总结用的是 `[auto_summary].instruction`） |
| `content` | TEXT NOT NULL | 总结正文，即向量文档正文。**自动总结入库的是不带 `〔自动总结〕` 前缀的干净正文**，前缀只加在发进群的那一份上 |
| `coverage_json` | TEXT | 本次实际读过的区间列表 `[[start,end], ...]` |
| `ts_start` / `ts_end` | TEXT | `coverage` 的**包络**（min/max），供展示与向量 metadata |
| `message_count` | INTEGER | 本次**直接读取**的原文条数（只检索总结时为 0） |
| `requested_by` | TEXT | 触发者的 `member_openid`。**自动总结为 `NULL`**——没人触发 |
| `trigger` | TEXT NOT NULL | `'at'` = 有人 @ 了机器人；`'auto'` = 到量自动触发。默认 `'at'`，用于审计知识库里有多少篇是机器人自己攒的 |
| `created_at` | TEXT NOT NULL | 入库时刻 |
| `indexed_at` | TEXT | **`NULL` = 待索引**。这就是新的 backlog |

索引与查询：

```sql
CREATE INDEX idx_sum_group_created ON summaries(group_openid, created_at);
```

⚠️ `created_at` 不只是展示字段：**自动总结的阈值就以它为界**（"本群最后一篇总结是什么时候"），因为任何一次总结——被 @、自动、`qqbot ask --save`——都应该让计数归零。

```sql
-- backlog（配 ORDER BY datetime(created_at), rowid 保证同一秒内也稳定）
SELECT * FROM summaries WHERE indexed_at IS NULL ORDER BY datetime(created_at) ASC, rowid ASC LIMIT ?;
```

⚠️ **mark 的是整批**，包括那些因为没有可嵌入文本而被 `SummaryIndex.add()` 跳过的行（例如正文为空白）。否则它们会永远留在 backlog 里，每次扫描都被重新捞出来。

```sql
-- 自动总结的阈值计数
SELECT COUNT(*) FROM group_messages
 WHERE group_openid = ?
   AND datetime(ingested_at) > datetime(COALESCE(
        (SELECT MAX(created_at) FROM summaries WHERE group_openid = ?),
        '1970-01-01T00:00:00'))
```

三个要点：

- 用 `ingested_at`（**入库时刻**）而不是 `ts`：问的是"此后我们又收到了多少条"，与发言人自报的时间戳无关。两者格式同源（都是本地时间字符串），所以能用 `datetime()` 直接比。
- **没有总结时退到 epoch**：一个已有积压的群会立刻触发第一次自动总结（然后冷却），这是期望行为，不是 bug。
- 走 `idx_gm_group_ingested`。它在每条群消息上跑一次，不是低频查询。

⚠️ **`summary_id` 用 uuid 而不是内容哈希**。内容哈希 + `INSERT OR IGNORE` 有个静默丢数据的失败模式：同一个群连问两次"总结一下"，模型可能给出同样的套话，第二次那行连同它**不同的覆盖范围**会被无声吃掉。幂等其实在别处已经有了——触发层用消息 `message_id` 去重（同一条 @ 只会总结一次），重索引的幂等由 Chroma 的 `upsert` 按 id 保证。所以这里就是一条普通 `INSERT`。

### 2.4 时间戳一律走 `datetime()` 比较

```sql
WHERE group_openid = ?
  AND datetime(ts) >= datetime(?)
  AND datetime(ts) <= datetime(?)
ORDER BY datetime(ts) ASC
```

**不能按字符串直接比。** ISO 字符串只有在所有值携带同一 UTC 偏移时排序才正确，而这些边界来自**模型**——它可能输出 `2026-07-21T00:00:00Z`、裸日期 `2026-07-21`、或带本地偏移的时间。`datetime()` 把它们统一归一到 UTC 再比较。

同理 `ORDER BY` 也用 `datetime(ts)`：按时序取数不能被偏移差异打乱。

### 2.5 迁移

`SQLStore._migrate()` 在构造时跑一次，把后加的列补到已有库上。`_MIGRATIONS` 是**表感知的三元组**：

```python
_MIGRATIONS: list[tuple[str, str, str]] = [
    ("summaries", "trigger", "TEXT NOT NULL DEFAULT 'at'"),
]
```

对每张出现过的表各查一次 `PRAGMA table_info(table)`，缺列就 `ALTER TABLE ... ADD COLUMN`，然后 `commit`。**为什么需要它**：`schema.sql` 全部是 `CREATE TABLE IF NOT EXISTS`，而它**只会给已存在的数据库补一张新表，永远不会给已存在的表补一列**。所以 `trigger` 这种后加的列必须**同时**写在 `schema.sql`（给新库）和这里（给旧库），少一处就有一半的库缺列。

机制本身保持克制：只在构造时跑、只加列、不改写任何数据。`ALTER TABLE ADD COLUMN` 带 `NOT NULL DEFAULT` 在 SQLite 上是 O(1) 的元数据操作，已有行直接取默认值。

### 2.6 已知的存储取舍

- **`content` 存的是渲染文本，不是逐字原文**。写库时取的是 `GroupMessageRecord.body()`：QQ 在引用（103）与合并转发（102）消息上把顶部 `content` 留空，正文只在 `msg_elements` 里；语音消息的文字部分是附件上的 `asr_refer_text`。若存原始 `content`，这些消息在取数时就是**一行空白**。
- **老数据不会自动重写**。逐字原文（整个 `d`）仍在 `raw_json` 里，所以这一步可回填，但**升级前入库的行仍是老语义下的 `d.content`**——引用 / 纯图片那类在取数时看起来就是一行空白。需要时从 `raw_json` 重建即可。
- **`msg_elements` 不单独建表**。引用 / 合并转发的嵌套内容在事件解析阶段被摊平成文本写进 `content`，结构本身不留存；要支持"展开某条合并转发"得改这里。
- **`raw_json` 目前无消费者**。它保证即使解析逻辑有遗漏，原始数据也没丢，未来可以据此回填（例如按上面的取舍回填旧行）。

### 2.7 `media` — 图片附件表（M2，已实现）

一条**附件出现**一行（不是一张去重后的图）：同图被多人转发时共享同一个磁盘文件（按 `sha256` 复用 `path`），但每次出现都有自己的处理状态、短 id 与回写锚点。方案的动机与管线见 `MEDIA.md`；本节只管表结构。

| 列 | 类型 | 说明 |
|:---|:---|:---|
| `media_id` | TEXT **PK** | `uuid4().hex`。与 `summary_id` 同款教训：**不用内容哈希做主键**（哈希 + `INSERT OR IGNORE` 会吃掉"第二次出现"）。消息正文里的 `#短id` = 它的前 8 位 |
| `message_id` | TEXT NOT NULL，FK → `group_messages` | `ON DELETE CASCADE` |
| `group_openid` | TEXT NOT NULL | 群内 `view_image` 的归属校验靠它（短 id 只有 8 hex，可被猜，必须显式拒绝跨群） |
| `placeholder` | TEXT NOT NULL | **写库当时**渲染出的占位文本（含 `#短id`），描述回写时对 `content` 做定向 `replace` 的锚点。事后重算 `label()` 一旦格式演进过就替换不中 |
| `url` / `filename` / `content_type` / `event_size` / `width` / `height` | — | 事件原始信息。`event_size` 在下载前先做超限预判 |
| `status` | TEXT NOT NULL | `pending`（默认）→ `stored`（终态，等工具按需取用）；失败分态 `expired`（下载 4xx，签名失效，不耗 LLM）/ `skipped`（魔数不在白名单 / 超限）/ `failed`（网络/5xx 重试耗尽） |
| `sha256` | TEXT | 下载字节的哈希 = 存储文件名。已登记过即复用旧 `path`（跨天转发、跨日期桶都不复制文件） |
| `path` | TEXT | 相对 `[media].dir` 的落盘路径：`YYYY-MM-DD/<sha256>.<ext>`（绝对路径 = `dir / path`，工具在调用时解析配置，便于测试沙箱）。日期 = **消息发送日期**（跨天重试也落回原桶），桶由首次入库决定 |
| `description` | TEXT | 通用描述的**缓存**（`view_image` 无 focus 调用产出）。带 focus 的定向回答不落这里 |
| `attempts` / `last_error` | INTEGER / TEXT | 重试簿记 |
| `created_at` / `updated_at` | TEXT | 入库 / 状态变更时刻 |

```sql
CREATE INDEX idx_media_pending ON media(status, created_at) WHERE status = 'pending';  -- Worker 取队
```

**写路径**：`SQLStore.insert_messages()` 在插入消息的**同一事务**里，为图片形态的附件（`Attachment.is_image()` 的 `content_type` 判定）插 media 行，并把该消息 `content` 里的占位换成带短 id 的版本。去重免费——重复事件时消息本体 `INSERT OR IGNORE` 插不进，media 分支根本不执行。`§2.2` 的 `has_media` 从此与"media 表有行"**近似**等价（保留它只为不查表就能过滤）：说"近似"是因为目前只有**顶层**附件会建行，`msg_elements` 里嵌套的图片暂不入列（MEDIA.md 已知边界，M2.x）。

**一条存了图的消息，三跳找到自己的图**：

```
group_messages.content   "15:03 小红: 看这个 [图片 6A3051F3.jpg #4d9f2a1c]"
        │ 取数工具把整行喂给模型；模型想看图 →
        ▼
media (media_id LIKE '4d9f2a1c%')   status=stored, path=2026-10-10/e3b0c4….jpg
        │ view_image 读文件 → 一次性视觉调用 →
        ▼
磁盘 data/media/2026-10-10/e3b0c4….jpg          ← 字节，与 rkey 寿命无关
```

看过的图，`content` 里的占位会变成 `[图片 6A3051F3.jpg #4d9f2a1c：<通用描述>]`——回写与 `media.description` 在**同一事务**，不存在"状态说看过、正文还是占位"的中间态。

⚠️ **`path` 非空 ≠ 图能看**：磁盘文件是这条记录存在的全部理由，但清理本期不做（见 §六）；`expired`/`skipped` 的行 `path` 恒为 `NULL`，占位分别被改注 `[图片 已过期]` / `[图片 格式不支持]`，让取数可见、模型不会去调一个必然失败的工具。

---

## 三、Chroma（向量库）

位置：`[store].chroma_path`（默认 `data/chroma_db/`），collection 名 `[store].chroma_collection`（默认 `summaries`）。

### 3.0 一篇总结 = 一个文档

这是本项目最重要的一条数据决策。曾经的做法是**每条消息**一个向量文档，问题有两层：

- **成本与噪音**：活跃群里大量消息是"哈哈哈""+1""收到"，长度短、信息量低，却各占一个向量槽位，把语义空间稀释掉。
- **检索粒度错**：真正的知识单元是**一次总结的结论**。问"之前有人提过部署方案吗"，命中的应该是一段讨论的结论，而不是碰巧共用一个词的碎片。

所以现在只有 `summaries` 里的行会被 embed。产出一篇文档有两条路：

| 来源 | 何时产生 | `trigger` |
|:---|:---|:---|
| 群内被 @ | 有人召唤机器人并得到一份有效总结 | `'at'` |
| 自动总结 | 本群"距上次总结新增的消息"达到 `[auto_summary].min_messages` | `'auto'` |
| 手工 | `qqbot ask "..." --save`（开发期造文档用） | `'at'`（默认值） |

**没有定时任务、没有日报**：自动总结由**消息量**驱动，不按钟点。这样语料量天然与讨论热度挂钩，安静期不会攒出一堆复述相同内容的空文档。

原文仍在 SQLite，按时间范围照常可查（群内与私聊都能），只是不参与语义检索——所以"总结文档"覆盖的始终只是话题的一个子集。

### 3.1 文档与元数据

```python
id       = summary_id                                   # 用 summary_id 作向量 ID → upsert 而非重复插入
content  = "[群_demo] 2026-07-21 08:00 ~ 10:00 · 12 条\n<总结正文>"   # 见下
metadata = {                                            # 全部是标量
    "summary_id":    str,
    "group_openid":  str,
    "created_at":    str,
    "ts_start":      str,
    "ts_end":        str,
    "message_count": int,
}
```

**为什么把群与覆盖范围写进 `page_content`**：检索时只有 `page_content` 会进入模型上下文。私聊是跨群检索的那一侧，如果正文里没有群标签和时间范围，模型根本说不出"这是哪个群、什么时候的结论"。`metadata` 里的同名字段服务的是**过滤**（群内检索）而不是渲染。

**为什么用 `summary_id` 当向量 ID**：Chroma 按 ID upsert。重跑 `reindex` 时是覆盖而不是插入重复项。

### 3.2 ⚠️ metadata 用标量

chromadb 1.5.9 的 `validate_metadata` 实际接受 `str | int | float | bool | list | None`，**只有嵌套 dict 会抛异常**。这里仍只用标量：一是它们正是下面那个等值过滤器操作的类型，二是语义清楚，不用去想"这个 list 会不会被当成一个值比较"。

### 3.3 检索的群过滤是**可选**的——这是唯一一处例外

```python
# 群内：必须限定本群
similarity_search_with_score(query, k=k, filter={"group_openid": group_openid})

# 私聊：不过滤，允许读到所有群
similarity_search_with_score(query, k=k)
```

`SummaryIndex.search(..., group_openid=None)` 里的 `None` **不是疏忽**，而是私聊那条路径的全部意义所在。两条相反的规则由**两个不同的 agent**表达（群 agent 的工具集里根本没有跨群能力），而不是同一个 agent 里的运行时 `if`——见 `ARCHITECTURE.md`。

检索返回 `(Document, score)`，score 是距离（越小越近）。`search_summaries` 工具会把它附在每行末尾（`（距离 0.123）`）。

### 3.4 空正文不入库

`SummaryIndex.add()` 对 `summary_text()` 返回空串的行直接跳过。返回的是**实际写入条数**，indexer 拿它与取出的行数比较，不一致就记 warning。跳过的行仍会被 `mark_summaries_indexed` 标记，理由见 §2.3。

---

## 四、会话记忆（内存）

`Summarizer` 用 `langgraph.checkpoint.memory.InMemorySaver`。**群与私聊是两个 agent，但共用同一个 checkpointer**，而两者都默认 `checkpoint_ns=""`，所以 `thread_id` 是它们之间**唯一**的命名空间分隔符：

| 场景 | `thread_id` |
|:---|:---|
| 群 | `G:<group_openid>` |
| 私聊 | `U:<user_openid>` |

前缀是必须的：QQ 并不保证群 `openid` 与用户 `openid` 不相交，不加前缀就可能让两个不同的会话撞进同一条线程。同一串 key 必须同时用在 `summarize_group` / `answer_private`、`_touch_thread`、`delete_thread` 三处——否则 LRU 会按另一个 key 淘汰，`delete_thread` 落空，checkpoint 静默泄漏。

| 项 | 值 |
|:---|:---|
| 持久化 | ❌ 进程重启即丢 |
| 淘汰策略 | LRU，`MAX_THREADS = 128`（**群与私聊共享这一个预算**） |
| 淘汰动作 | `checkpointer.delete_thread(oldest)` |

`InMemorySaver` 自身不会淘汰任何东西，线程数只增不减，所以 LRU 把上界钉在 128。群和私聊共用一个 LRU 比维护两个更简单，代价是话痨的私聊用户可能挤掉群记忆——上限给得足够宽，实践中不常发生。

**重启丢失的后果**：只影响追问的连贯性（"那再往前一点"这类依赖上一轮上下文的指令）。消息本身在 SQLite，重新提问即可。

要跨重启保留，需 `uv add langgraph-checkpoint-sqlite` 后换成 `SqliteSaver`。

---

## 五、一致性

### 5.1 两个索引状态必须同步重置

`summaries.indexed_at`（SQLite）与 collection（Chroma）表达的是同一件事的两半：

- 只清 SQLite 标记 → 总结被当作未索引，重新 embed 一遍（浪费，但无害）。
- 只清 Chroma → **标记还在，那篇总结永远不会被重新索引，向量永久缺失**。

所以 `reindex --reset` 必须两件事一起做：

```python
dropped_marks = store.clear_summary_marks()   # 先清标记
index.reset()                                 # 再清向量
```

顺序无所谓，但**缺一不可**。

### 5.2 索引是最终一致

**总结入库**与**进入向量库**之间存在延迟（默认最多 10 秒一轮；入库时会 `wake()` 唤醒索引器，所以通常远快于此）。因此：

- **`search_summaries` 可能漏掉刚生成的那篇总结**——这是设计上接受的，不是 bug。
- **`messages_in_range` / `recent_messages` 不受影响**，它们直接读 SQLite。
- 系统提示词里明确告知模型这一边界，让它对"最近"的提问优先走按时间取数。

原始消息不参与索引，所以"刚刚那几句语义检索不到"是必然的：**语义检索只覆盖已经生成过总结的话题**——被 @ 触发的、到量自动触发的、或 `--save` 造的那几篇。一个还在 `min_messages` 阈值以下、又没人 @ 过的讨论，语义上就是查不到的，与消息新旧无关。

但这**不等于那段讨论丢了**：原文逐条在 `group_messages` 里，群内用按时间范围取数的工具、私聊用 `messages_across_groups`，都能拿到。所以 agent 面对"最近聊了什么"这类问题时应该走按时间取数，而不是语义检索——系统提示词里就是这么要求的。

### 5.3 崩溃恢复

一次总结里，落库与（可能有的）发送的顺序是**先落库**。两条触发路径都遵守这一条：被 @ 时发的是回给群里的分段回复，自动总结时发的是 `notify = true` 的那条主动消息，而**默认静默的自动总结压根不发**。

| 崩溃时点 | 恢复行为 |
|:---|:---|
| agent 跑完前 | 什么都没发生。被 @ 的那条没有回复，只能重问；自动总结则因为**冷却是在发起前就写下的**，要等满 `cooldown_s` 才会再试一次（这正是想要的：网关故障时不会每条消息都重试） |
| 总结入库后、向量写入前 | 下一轮被重新捡起并 embed（幂等：upsert） |
| 向量写入后、标记前 | 同上，重复 embed 一批，幂等 |
| 标记后 | 无操作 |

**为什么先落库再发送**：回答是不是一篇文档，与群里收没收到那条回复无关。若把落库挂在 `reply_chunked(...).ok` 上，一次 429 或 botpy 的静默 `None`（见 `CLAUDE.md`：HTTP 层吞掉超时后返回 `None` 而不是抛异常）就会**永久丢掉**这篇总结且无从恢复。自动总结的 `notify` 分支同理——发不出去只是少一条群消息，知识库不该跟着受影响。

顺带一提，自动总结入库的是**不带 `〔自动总结〕` 前缀的干净正文**（见 §2.3），前缀只贴在发出去的那一份上：文档的正文不该因为分享渠道而变形。

因为 `mark_summaries_indexed` 与向量写入不是同一个事务，最坏情况是**重复 embed 一批**。用 `summary_id` 当向量 ID 使这种重复是幂等的。

---

## 六、数据生命周期

| 数据 | 有无清理机制 |
|:---|:---|
| `group_messages` | ❌ 只增不减。目前没有归档或过期策略 |
| `summaries` | ❌ 只增不减；`indexed_at` 是唯一的可变列 |
| `media` 行 | ❌ 只增不减（M2，§2.7）；`status`/`sha256`/`path`/`description` 随后台处理与工具调用更新 |
| `data/media/` 磁盘文件 | ❌ 无 TTL；按消息日期分桶，增长 ∝ 图片量（估算 ≈20MB/天/5 活跃群）。不设便捷删除命令——删文件=删语料（M2，§2.7） |
| Chroma collection | 仅在 `reindex --reset` 时整体重建 |
| agent 会话记忆 | ✅ LRU 淘汰 + 进程退出 |
| 自动总结冷却（内存） | ✅ 进程退出即丢。若重启时积压仍超阈值，会立刻再自动总结一次——多一篇文档，无害 |
| `logs/app.log` | ✅ 轮转，10 MB × 5 份 |

**自动总结的冷却刻意不落库**（`client._last_auto`，`group_openid → time.monotonic()`）。它是一次性的限流闸门，不是数据：持久化它的代价（多一张表、多一处写路径、多一个要迁移的 schema）远大于收益，而丢失它的后果仅仅是重启后多发一次请求。记录的是**上一次尝试**的时刻而非成功时刻，理由见 §5.3 第一行。

`raw_json` 保留原始 payload 会让单行体积偏大（尤其含卡片消息时）。目前没有容量上限，长期运行的群需要自行关注 `data/` 目录大小。

**旧 collection 不会被自动删除**：collection 名从 `group_messages` 改成 `summaries` 之后，旧向量仍留在 `data/chroma_db/` 里但永不被查询（新代码的私聊路径是**无过滤**查询，若沿用旧名会把原始消息向量捞出来，既是隐私泄漏也是污染——所以改名是必需的迁移，而不是可选的美化）。确认不需要后手动删掉该目录即可回收空间；代码不代劳。

---

## 七、字段来源速查

排查"某个字段怎么是空的"时对照此表：

| 字段 | 来源 | 可能为空的原因 |
|:---|:---|:---|
| `author_name` | `d.author.username` | 走 `GROUP_AT_MESSAGE_CREATE` 降级路径时 botpy 拿不到昵称，恒为 `NULL`；展示层会退化成 `成员<openid 后 4 位>` |
| `member_role` | `d.author.member_role` | 同上 |
| `content` | `GroupMessageRecord.body()`（= `d.content` + 附件标签 + `asr_refer_text` + `msg_elements` 摊平） | 真的没有任何文本时（纯图片无正文）只剩 `[图片 photo.jpg]` 这类占位；**升级前入库的旧行**按老语义存的是 `d.content`，引用 / 合并转发那类会看起来是空的 |
| `msg_idx` / `ref_msg_idx` | `d.message_scene.ext` | 非引用消息通常没有 |
| `ts` | `d.timestamp` | 解析失败时回退为入库时刻（`datetime.now()`），不会为 `NULL` |
| `event_id` | WS 帧顶层 `id` | 走 botpy `GroupMessage` 时取 `message.event_id` |
| `message_type` | `d.message_type` | 降级路径恒为 0（botpy 不读该字段） |

`summaries` 侧的对应表格：

| 字段 | 来源 | 可能为空的原因 |
|:---|:---|:---|
| `coverage_json` | 工具调用时记录 | 没有任何取数工具成功执行（只调了 `current_time`，或纯追问零工具调用）——但这种回答按 §3.0 的判据根本不会入库 |
| `ts_start` / `ts_end` | `coverage` 的包络 | 同上；`coverage` 里的区间缺任一端时该次读取不参与包络 |
| `message_count` | 各取数工具返回的行数之和 | **只做语义检索时为 0**——`search_summaries` 读的是总结不是原文，计数不虚增 |
| `requested_by` | `record.author_openid` | **自动总结为 `NULL`**——没人触发；用 `qqbot ask --save` 手工造文档时也是 `NULL`（CLI 没有 QQ 身份） |
| `trigger` | 触发路径（`_respond` 传 `'at'`，`_auto_summarize` 传 `'auto'`） | 恒不为 `NULL`（`NOT NULL DEFAULT 'at'`）。老库里的行经 §2.5 的迁移后也全是 `'at'`——迁移之前确实只有被 @ 这一条路 |
| `indexed_at` | 后台 indexer | `NULL` 表示还没进向量库，属正常中间态 |
