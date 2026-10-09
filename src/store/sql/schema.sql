-- Inbound QQ group messages, one row per message.
--
-- `message_id` is the primary key on purpose: QQ may deliver the same message
-- more than once (the payload has no `msg_seq`, unlike the channel events), so
-- dedup happens for free via INSERT OR IGNORE.
CREATE TABLE IF NOT EXISTS group_messages (
    message_id    TEXT PRIMARY KEY,
    event_id      TEXT,                      -- WS frame top-level id
    group_openid  TEXT NOT NULL,
    author_openid TEXT,
    author_name   TEXT,                      -- dropped by botpy's GroupMessage
    member_role   TEXT,                      -- member / admin / owner
    -- The message as *text*: raw `content` plus placeholders for attachments
    -- (`[图片 photo.jpg]`), voice ASR transcripts, and the bodies of quoted /
    -- merged-forward elements. Not the verbatim payload — for quote (103) and
    -- forward (102) messages the raw text is empty and this is the only copy of
    -- the words; the original `d` is in `raw_json`. Rows written before this
    -- column changed meaning hold the raw text alone.
    content       TEXT,
    message_type  INTEGER,                   -- 0 text, 3 card, 101/102/103 ...
    ts            TEXT NOT NULL,             -- ISO8601
    msg_idx       TEXT,                      -- from message_scene.ext
    ref_msg_idx   TEXT,                      -- quoted message, from message_scene.ext
    has_media     INTEGER NOT NULL DEFAULT 0,
    raw_json      TEXT,                      -- original `d`, kept for forward compat
    ingested_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gm_group_ts ON group_messages(group_openid, ts);

-- Serves `messages_since_last_summary()`, which the auto-summary trigger runs
-- once per inbound message: counting `ingested_at` past the group's last
-- summary timestamp must not degrade into a scan.
CREATE INDEX IF NOT EXISTS idx_gm_group_ingested ON group_messages(group_openid, ingested_at);

-- One row per successful group summarisation. This is the unit of the vector
-- store: the summary text is embedded, the raw messages are not.
--
-- `indexed_at` lives here rather than in a side table because the backlog query
-- is then just `WHERE indexed_at IS NULL`, and `reindex --reset` is one UPDATE.
--
-- `summary_id` is a uuid rather than a hash of the content: a content hash
-- combined with INSERT OR IGNORE would silently collapse two summaries that
-- happen to render identically (two "总结一下" over a quiet group), losing the
-- second one's coverage window. Idempotency is already handled elsewhere — the
-- trigger layer dedups on `message_id`, and Chroma upserts on `summary_id`.
CREATE TABLE IF NOT EXISTS summaries (
    summary_id    TEXT PRIMARY KEY,
    group_openid  TEXT NOT NULL,
    instruction   TEXT NOT NULL,             -- the @-instruction that produced it
    content       TEXT NOT NULL,             -- the summary body, i.e. the document
    coverage_json TEXT,                      -- [[start,end], ...] actually read
    ts_start      TEXT,                      -- hull of coverage, for display
    ts_end        TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    requested_by  TEXT,                      -- member_openid of whoever asked
    trigger       TEXT NOT NULL DEFAULT 'at', -- 'at' = someone summoned the bot
                                             -- 'auto' = message-count trigger
    created_at    TEXT NOT NULL,
    indexed_at    TEXT                       -- NULL = waiting to be embedded
);

CREATE INDEX IF NOT EXISTS idx_sum_group_created ON summaries(group_openid, created_at);
