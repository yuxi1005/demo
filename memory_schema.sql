CREATE EXTENSION IF NOT EXISTS ltree;
CREATE EXTENSION IF NOT EXISTS vector;

-- 记忆主表：与你的 MemoryUnit 字段一一对应
CREATE TABLE memory (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  content          text        NOT NULL,
  importance       real        NOT NULL DEFAULT 0.5,
  embedding        vector(1024),              -- bge-m3 维度=1024
  timestamp        timestamptz  NOT NULL DEFAULT now(),
  retrieval_count  integer      NOT NULL DEFAULT 0,
  last_accessed_ts timestamptz
);

--ALTER TABLE memory ADD COLUMN embedding  vector(1024);
-- 常用索引（向量检索 + 时间/排序
CREATE INDEX idx_memory_embedding_ivf ON memory
  USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
CREATE INDEX idx_memory_timestamp ON memory(timestamp);

-- （可选：多级标签，用 ltree，后面再做也行）
CREATE TABLE tag (
  id    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  path  ltree NOT NULL UNIQUE      -- 例如: tech.frontend.react
);
CREATE TABLE memory_tag (
  memory_id uuid REFERENCES memory(id) ON DELETE CASCADE,
  tag_id    uuid REFERENCES tag(id)    ON DELETE CASCADE,
  PRIMARY KEY(memory_id, tag_id)
);
CREATE INDEX idx_tag_path_gist ON tag USING GIST(path);

SELECT * FROM memory;


ALTER TABLE memory
  ADD COLUMN IF NOT EXISTS importance        real    NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS embedding         vector(1024),
  ADD COLUMN IF NOT EXISTS timestamp         timestamptz NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS retrieval_count   integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS last_accessed_ts  timestamptz,
  ADD COLUMN IF NOT EXISTS tags              jsonb;


-- 1) 事件表（最小字段集）
CREATE TABLE IF NOT EXISTS event (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title        text,                      -- 可空，后续可自动生成/手动命名
  type         text,                      -- meeting|study|bugfix|shopping|misc
  status       text DEFAULT 'active',     -- active|dormant|closed
  created_at   timestamptz DEFAULT now(),
  updated_at   timestamptz DEFAULT now(),
  start_ts     timestamptz DEFAULT now(),
  end_ts       timestamptz,
  centroid     vector(1024),              -- 主题向量（用Python维护）
  member_count int DEFAULT 0,
  tags         ltree[]                    -- 可选
);

-- 2) 给 memory 加事件外键（最小改动）
ALTER TABLE memory
  ADD COLUMN IF NOT EXISTS event_id uuid REFERENCES event(id);

-- 3) 常用索引
CREATE INDEX IF NOT EXISTS idx_memory_event ON memory(event_id);
CREATE INDEX IF NOT EXISTS idx_event_updated_at ON event(updated_at);
CREATE INDEX IF NOT EXISTS idx_event_centroid_ivf
  ON event USING ivfflat (centroid vector_cosine_ops) WITH (lists=50);

SELECT * FROM event;

SELECT 
    m.content,
    m.importance,
    m.timestamp,
    e.id AS event_id,
    e.title AS event_title,
    e.type AS event_type,
    e.status AS event_status
FROM memory m
JOIN event e ON m.event_id = e.id
ORDER BY m.timestamp DESC;

BEGIN;

TRUNCATE TABLE
  event,
  memory,
  memory_tag,
  tag
RESTART IDENTITY CASCADE;

COMMIT;
