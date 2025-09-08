import uuid, requests
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import os, re, json, time
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from utils import (
    _l2_normalize,
    _cos,
    _as_float_list,
    suggest_event_title_simple,
    normalize_tag_path,
    is_valid_tag_path,
    match_tag_prefix,
    embed_text,
    load_bgem3,
)
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import datetime, math

load_dotenv()  # 默认会读取当前目录下的 .env


@dataclass
class MemoryUnit:
    """记忆单元"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    importance: float = 0.0
    retrieval_count: int = 0
    last_accessed_ts: Optional[datetime.datetime] = None
    tags: List[str] = field(default_factory=list)  # optional hierarchical tags
    event_id: Optional[str] = None

    @staticmethod
    def new(
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ) -> "MemoryUnit":
        tags = tags or []
        cleaned = []
        for t in tags[:3]:
            t = normalize_tag_path(t)
            if is_valid_tag_path(t) and t not in cleaned:
                cleaned.append(t)
        return MemoryUnit(
            content=content,
            importance=max(0.0, min(1.0, float(importance))),
            tags=cleaned,
            embedding=embedding,
        )


# ---------------------- Storage (PostgreSQL) ----------------------
class MemoryStore:
    """
    记忆存储：PostgreSQL + pgvector
    """

    def __init__(self, *, ensure_schema: bool = True):
        dsn = os.getenv("PG_DSN")
        if not dsn:
            raise RuntimeError("PG_DSN is not set in environment (.env).")
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        if ensure_schema:
            self._init_schema()

    # ----- schema -----
    def _init_schema(self):
        # Dimension for pgvector; default 1024 (bge-m3 text)
        dim = int(os.getenv("PGVECTOR_DIM", "1024"))
        with self.conn.cursor() as cur:
            # pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # main table
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory (
                    id UUID PRIMARY KEY,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0,
                    embedding VECTOR({dim}),
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
                    retrieval_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_ts TIMESTAMPTZ NULL,
                    tags JSONB NULL
                );
                """
            )
            # index for cosine distance (requires ANALYZE after large inserts)
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS memory_embedding_idx
                ON memory USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
                """
            )
            # --- event table (minimal) + memory.event_id column
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS event (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title        TEXT,
            type         TEXT,
            status       TEXT DEFAULT 'active',
            created_at   TIMESTAMPTZ DEFAULT now(),
            updated_at   TIMESTAMPTZ DEFAULT now(),
            start_ts     TIMESTAMPTZ DEFAULT now(),
            end_ts       TIMESTAMPTZ,
            centroid     VECTOR(%s),
            member_count INT DEFAULT 0,
            tags         JSONB
            );""",
                (dim,),
            )

            cur.execute(
                "ALTER TABLE memory ADD COLUMN IF NOT EXISTS event_id UUID REFERENCES event(id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_event ON memory(event_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_updated_at ON event(updated_at);"
            )
            cur.execute(
                """
            CREATE INDEX IF NOT EXISTS event_centroid_ivf
            ON event USING ivfflat (centroid vector_cosine_ops) WITH (lists=50);
            """
            )

    def exec(self, sql: str, params: tuple = ()):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)

    def query(self, sql: str, params: tuple = ()) -> List[dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def query_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, params)
            r = cur.fetchone()
            return dict(r) if r else None

    # ----- basic CRUD -----
    def add(self, m: MemoryUnit):
        """增加或更新（如果已存在）记忆单元"""
        with self.conn.cursor() as cur:
            vector_literal = None
            if m.embedding is not None:
                vector_literal = "[" + ",".join(f"{x:.6f}" for x in m.embedding) + "]"
            cur.execute(
                """
                INSERT INTO memory (
                    id, content, importance, embedding, timestamp,
                    retrieval_count, last_accessed_ts, tags
                )
                VALUES (
                    %s, %s, %s,
                    CASE WHEN %s IS NULL THEN NULL ELSE %s::vector END,
                    %s, %s, %s, %s
                )
                ON CONFLICT (id) DO UPDATE SET
                  content=EXCLUDED.content,
                  importance=EXCLUDED.importance,
                  embedding=COALESCE(EXCLUDED.embedding, memory.embedding),
                  timestamp=EXCLUDED.timestamp,
                  tags=COALESCE(EXCLUDED.tags, memory.tags)
                """,
                (
                    m.id,
                    m.content,
                    float(m.importance),
                    vector_literal,
                    vector_literal,
                    m.timestamp,
                    int(m.retrieval_count),
                    m.last_accessed_ts,
                    json.dumps(m.tags) if m.tags else None,
                ),
            )

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM memory WHERE id=%s", (memory_id,))
            row = cur.fetchone()
            return self._row_to_mu(row) if row else None

    def get_all(self) -> List[MemoryUnit]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM memory ORDER BY timestamp DESC")
            return [self._row_to_mu(r) for r in cur.fetchall()]

    def delete(self, memory_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM memory WHERE id=%s", (memory_id,))
            return cur.rowcount > 0

    def clear(self):
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE memory")

    def update_retrieval_stats(self, memory_id: str, inc: int = 1):
        """更新某记忆的检索统计"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory
                   SET retrieval_count = retrieval_count + %s,
                       last_accessed_ts = now()
                 WHERE id=%s
                """,
                (inc, memory_id),
            )

    def search_by_embedding(
        self, query_embedding: List[float], top_k: int = 3
    ) -> List["MemoryUnit"]:
        """
        Preferred: use pgvector cosine distance.
        Fallback: load all rows and compute cosine in Python if embedding is NULL or pgvector not available.
        """
        vector_literal = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT *, (1.0 - (embedding <=> %s::vector)) AS cos_sim
                      FROM memory
                     WHERE embedding IS NOT NULL
                     ORDER BY embedding <=> %s::vector ASC
                     LIMIT %s
                    """,
                    (vector_literal, vector_literal, int(top_k)),
                )
                rows = cur.fetchall()
                return [self._row_to_mu(r) for r in rows]
        except Exception:
            # Fallback: compute in Python
            all_rows = self.get_all()

            def _cos(a, b):
                if not a or not b:
                    return 0.0
                dot = sum(x * y for x, y in zip(a, b))
                na = sum(x * x for x in a) ** 0.5
                nb = sum(y * y for y in b) ** 0.5
                return dot / (na * nb) if na and nb else 0.0

            scored = [
                (m, _cos(query_embedding, m.embedding)) for m in all_rows if m.embedding
            ]
            scored.sort(key=lambda t: t[1], reverse=True)
            return [m for m, _ in scored[:top_k]]

    # memory.py — MemoryStore 内新增
    def search_candidates_by_embedding(
        self, query_embedding: List[float], limit: int = 64, probes: int = 10
    ) -> List["MemoryUnit"]:
        """
        只负责“近似召回”，不做最终排序。
        - limit: 候选集大小（建议 >= top_k*6）
        - probes: ivfflat.probes 提高召回率（越大越准，略慢）
        """
        vector_literal = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            try:
                cur.execute("SET ivfflat.probes = %s", (int(probes),))
            except Exception:
                pass
            cur.execute(
                """
                SELECT id, content, importance, embedding, timestamp, retrieval_count, last_accessed_ts, tags
                FROM memory
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
                """,
                (vector_literal, int(limit)),
            )
            rows = cur.fetchall()
        return [self._row_to_mu(r) for r in rows]

    # ------ 事件 新增 ---------
    # 1) 最近的事件（active/dormant，近3天）
    def get_recent_events(self, days=3) -> List[dict]:
        sql = """
        SELECT id, centroid, updated_at, member_count, title, status
        FROM event
        WHERE updated_at >= now() - interval '%s days' AND status IN ('active','dormant')
        """
        return self.query(sql, (days,))

    # 2) 创建事件
    def create_event(
        self, title: str, centroid: List[float], etype="misc"
    ) -> Optional[Dict[str, Any]]:
        sql = """INSERT INTO event(title, type, centroid)
                VALUES (%s, %s, %s) RETURNING id, centroid, member_count"""
        return self.query_one(sql, (title, etype, centroid))


    def touch_event_with_embedding(
        self, event_id: str, new_emb: List[float], alpha=0.7
    ):
        """
        用新的 embedding 更新事件的 centroid。"""
        ev = self.query_one(
            "SELECT centroid, member_count FROM event WHERE id=%s", (event_id,)
        )
        old_vec = (
            _as_float_list(ev["centroid"]) if ev and ev["centroid"] is not None else []
        )
        new_vec = _as_float_list(new_emb)

        if old_vec:
            L = min(len(old_vec), len(new_vec))
            mixed = [alpha * old_vec[i] + (1 - alpha) * new_vec[i] for i in range(L)]
            vec = _l2_normalize(mixed)
            sql = """UPDATE event SET centroid=%s, member_count=member_count+1, updated_at=now()
                    WHERE id=%s"""
            self.exec(sql, (vec, event_id))
        else:
            vec = _l2_normalize(new_vec)
            sql = """UPDATE event SET centroid=%s, member_count=1, updated_at=now()
                    WHERE id=%s"""
            self.exec(sql, (vec, event_id))

    
    def bind_memory_event(self, memory_id: str, event_id: str):
        """
        将记忆单元绑定到事件
        """
        self.exec(
            """UPDATE memory SET event_id=%s WHERE id=%s""", (event_id, memory_id)
        )

    @staticmethod
    def _row_to_mu(r) -> MemoryUnit:
        tags = []
        try:
            if r["tags"] is not None:
                if isinstance(r["tags"], list):
                    tags = r["tags"]
                else:
                    tags = json.loads(r["tags"])
        except Exception:
            tags = []
        emb = None
        if r["embedding"] is not None:
            # psycopg2 returns vector as a Python list via pgvector extension;
            # if not, it may be a memoryview/str; handle common cases
            if isinstance(r["embedding"], (list, tuple)):
                emb = list(r["embedding"])
            else:
                # try to parse "[...]" into list
                s = str(r["embedding"]).strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        emb = [float(x) for x in s[1:-1].split(",")]
                    except Exception:
                        emb = None
        event_id = str(r["event_id"]) if "event_id" in r and r["event_id"] else None
        return MemoryUnit(
            id=str(r["id"]),
            content=r["content"],
            embedding=emb,
            timestamp=r["timestamp"],
            importance=float(r["importance"]),
            retrieval_count=int(r["retrieval_count"]),
            last_accessed_ts=r["last_accessed_ts"],
            tags=tags,
            event_id=event_id,  # ⬅️ 新增
        )

    def get_event_context(self, memory_id: str, k_siblings: int = 5):
        row = self.query_one("SELECT event_id FROM memory WHERE id=%s", (memory_id,))
        if not row or not row.get("event_id"):
            return None, []
        ev_id = row["event_id"]
        header = self.query_one(
            """
            SELECT id, title, status, start_ts, updated_at FROM event WHERE id=%s
        """,
            (ev_id,),
        )
        sib_rows = self.query(
            """
            SELECT id, content, importance, timestamp
            FROM memory
            WHERE event_id=%s AND id<>%s
            ORDER BY (EXTRACT(EPOCH FROM (now()-timestamp))) * -0.6 + importance * 0.4 DESC
            LIMIT %s
        """,
            (ev_id, memory_id, k_siblings),
        )
        siblings = [self.get(r["id"]) for r in sib_rows]  # 复用 _row_to_mu
        return header, [s for s in siblings if s]

    def search_events_by_embedding(
        self,
        embedding: List[float],
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        在 event 表里按质心向量做近似最近邻检索（pgvector / ivfflat），
        回退：当数据库侧不可用时，Python 侧计算余弦相似度。
        返回字段：id, centroid(list[float]), member_count, title, status, updated_at
        """
        if not embedding:
            return []

        vector_literal = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

        # 1) 首选：数据库侧 ANN（<=> 为 pgvector 的距离；1 - 距离 = 余弦相似度）
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # 尝试提高召回率；失败也不影响主流程
                try:
                    cur.execute("SET ivfflat.probes = %s", (10,))
                except Exception:
                    pass

                cur.execute(
                    """
                    SELECT
                        id,
                        centroid,
                        member_count,
                        title,
                        status,
                        updated_at
                    FROM event
                    WHERE centroid IS NOT NULL
                    ORDER BY centroid <=> %s::vector ASC
                    LIMIT %s
                    """,
                    (vector_literal, int(limit)),
                )
                rows = cur.fetchall() or []

            def _as_vec(v):
                if v is None:
                    return None
                if isinstance(v, (list, tuple)):
                    return list(v)
                s = str(v).strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        return [float(x) for x in s[1:-1].split(",")]
                    except Exception:
                        return None
                return None

            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "id": str(r["id"]),
                        "centroid": _as_vec(r["centroid"]),
                        "member_count": int(r["member_count"]) if r["member_count"] is not None else 0,
                        "title": r.get("title"),
                        "status": r.get("status"),
                        "updated_at": r.get("updated_at"),
                    }
                )
            return out

        except Exception:
            # 2) 回退：Python 侧计算相似度并排序
            ev_rows = self.query(
                """
                SELECT id, centroid, member_count, title, status, updated_at
                FROM event
                WHERE centroid IS NOT NULL
                """
            )

            cand: List[Tuple[float, Dict[str, Any]]] = []
            q = _as_float_list(embedding)
            for ev in ev_rows:
                cen = _as_float_list(ev.get("centroid"))
                if not cen:
                    continue
                sim = _cos(q, cen)
                ev_out = {
                    "id": str(ev["id"]),
                    "centroid": cen,
                    "member_count": int(ev["member_count"]) if ev["member_count"] is not None else 0,
                    "title": ev.get("title"),
                    "status": ev.get("status"),
                    "updated_at": ev.get("updated_at"),
                }
                cand.append((sim, ev_out))

            cand.sort(key=lambda x: x[0], reverse=True)
            return [ev for _, ev in cand[: max(1, int(limit))]]



# ---------------------- Toolboxes ----------------------


class ForgettingManager:
    def by_importance(self, store: "MemoryStore", threshold: float = 0.2) -> List[str]:
        ids = [m.id for m in store.get_all() if m.importance < threshold]
        return ids

    def by_time_decay(self, store: "MemoryStore", days: int = 90) -> List[str]:
        now = datetime.datetime.now()
        ids = []
        for m in store.get_all():
            last = m.last_accessed_ts or m.timestamp
            if (now - last).days >= days:
                ids.append(m.id)
        return ids

class ReflectionManager:
    def generate_insight_v1(self, store: "MemoryStore") -> Optional[MemoryUnit]:
        return None




def assign_event_for_units(
    store: MemoryStore, turn_text: str, turn_emb: List[float], units: List[MemoryUnit]
):
    # 1) 近 3 天事件
    recents = store.get_recent_events(days=3) or []
    best = None
    best_sim = -1.0
    for ev in recents:
        cen = _as_float_list(ev.get("centroid"))
        if cen is None:
            continue
        sim = _cos(_as_float_list(turn_emb), cen)
        if sim > best_sim:
            best, best_sim = ev, sim

    # 2) 复用或新建
    if best is not None and best_sim >= 0.72:
        ev_id = best["id"]
    else:
        # 新建事件
        mu_texts = [mu.content for mu in units if getattr(mu, "content", None)]
        title = suggest_event_title_simple(turn_text, mu_texts, etype="misc")
        ev = store.create_event(
            title=title, centroid=_as_float_list(turn_emb), etype="misc"
        )
        if ev is None:
            raise RuntimeError("create_event returned None")
        ev_id = ev["id"]

    # 3) 绑定 & 更新
    for mu in units:
        store.bind_memory_event(mu.id, ev_id)
        seed_emb = mu.embedding or turn_emb
        store.touch_event_with_embedding(ev_id, _as_float_list(seed_emb), alpha=0.7)


_bgem3 = load_bgem3("BAAI/bge-m3", device="cpu")
SYSTEM_PROMPT_EXTRACTION = """
You are a memory extraction assistant.
The user provides a conversation transcript.
Return a JSON array of objects, each object only containing:
- "content": concise atomic fact (< 100 chars)
- "importance": float in [0,1]
Rules:
- Importance ∈ [0,1] based on how critical the fact is for future recall.
- Keep content concise.
- Output only valid JSON, no extra commentary.
- Always return a JSON array even if you extract only one memory.

"""
def build_mu_from_raw(rawdata: str, *, api_key_path: str = "../API-KEY.txt",
                      model: str = "deepseek-chat", timeout: int = 60) -> List["MemoryUnit"]:
    """
    从用户原文 rawdata 经 LLM 抽取 JSON 数组，再构造 MemoryUnit 列表返回。
    彻底单函数：无类、无中间状态、无额外辅助函数。
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION},
            {"role": "user", "content": rawdata},
        ],
        "temperature": 0,
        "stream": False,
    }

    # 读取 API Key
    with open(api_key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 简单重试
    attempt, max_attempts = 0, 2
    while True:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
                attempt += 1
                continue
            raise RuntimeError(f"调用 LLM 失败：{e}") from e

        if r.status_code in (429, 500, 502, 503, 504):
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
                attempt += 1
                continue
            raise RuntimeError(f"LLM API 错误 {r.status_code}：{r.text[:500]}")
        if r.status_code != 200:
            raise RuntimeError(f"LLM API 错误 {r.status_code}：{r.text[:500]}")
        try:
            data = r.json()
        except Exception:
            raise RuntimeError("LLM 返回内容非 JSON，无法解析。")
        break

    # ---- 提取模型文本，完全内联 ----
    content = ""
    if isinstance(data, dict):
        if "choices" in data:
            try:
                choice0 = data["choices"][0]
                if isinstance(choice0, dict):
                    msg = choice0.get("message")
                    if isinstance(msg, dict) and "content" in msg:
                        content = msg["content"]
                    if not content and "text" in choice0:
                        content = choice0["text"]
            except Exception:
                pass
        if not content:
            msg = data.get("message")
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
        if not content:
            content = data.get("response", "") or ""

    # 去掉可能的 ```json 围栏
    content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", (content or "").strip())
    if not content:
        raise RuntimeError("LLM 返回为空或结构异常。")

    # ---- 解析 JSON 数组 ----
    items = None
    try:
        arr = json.loads(content)
        if isinstance(arr, list):
            items = arr
    except Exception:
        pass
    if items is None:
        m = re.search(r"\[\s*{[\s\S]*?}\s*\]", content)
        if not m:
            raise RuntimeError("无法从 LLM 文本中提取 JSON 数组。")
        arr = json.loads(m.group(0))
        if not isinstance(arr, list):
            raise RuntimeError("提取片段不是 JSON 数组。")
        items = arr

    # ---- 构造 MemoryUnit ----
    memory_units: List[MemoryUnit] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        # 处理 content
        c = str(it.get("content", "") or "").strip()
        if not c:
            continue
        # importance
        try:
            p = float(it.get("importance", 0.0))
        except Exception:
            p = 0.0
        p = max(0.0, min(1.0, p))
        # embedding
        c_embedding = embed_text(_bgem3, f"passage:{c}").tolist()
        memory_units.append(MemoryUnit(content=c, importance=p, embedding=c_embedding))

    return memory_units
