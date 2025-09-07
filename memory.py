import uuid
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

    # 3) 事件心向量与心跳（Python 里做 EMA / 平均数更简单）
    def touch_event_with_embedding(
        self, event_id: str, new_emb: List[float], alpha=0.7
    ):
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

    # 4) 给记忆绑定事件
    def bind_memory_event(self, memory_id: str, event_id: str):
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


@dataclass
class EventDraft:
    """仅驻留内存的事件草稿：聚拢相近回合，不落库。"""

    title_hint: Optional[str] = None
    centroid: Optional[List[float]] = None  # 单位球面上的心向量
    members: List[MemoryUnit] = field(
        default_factory=list
    )  # 记忆单元（已写入memory表，但未绑定event_id）
    start_ts: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_ts: datetime.datetime = field(default_factory=datetime.datetime.now)

    def _norm(self, v: List[float]) -> List[float]:
        return _l2_normalize(v or [])

    def _cos(self, a: List[float], b: List[float]) -> float:
        return _cos(a, b)

    def _weighted_update(
        self, base: Optional[List[float]], vecs: List[Tuple[List[float], float]]
    ) -> List[float]:
        """对单位向量做加权平均后再归一化；vecs: [(vec, weight), ...]"""
        if not vecs:
            return base or []
        L = len(vecs[0][0])
        acc = [0.0] * L  # accumulator（累加器）
        tot = 0.0
        for v, w in vecs:
            if not v or w <= 0:
                continue
            nv = self._norm(v)
            for i in range(min(L, len(nv))):
                acc[i] += nv[i] * w
            tot += w
        if base and tot > 0:
            # 轻度“惯性”，避免心向量抖动：旧心向量等效权重 = sum(w) * 0.2
            base_n = self._norm(base)
            bw = tot * 0.2
            for i in range(min(L, len(base_n))):
                acc[i] += base_n[i] * bw
            tot += bw
        if tot <= 0:
            return base or []
        return self._norm([x / tot for x in acc])

    def add_turn_mem(
        self,
        turn_text: str,
        turn_emb: List[float],
        units: List[MemoryUnit],
        tau_minutes: int = 240,
    ):
        """
        将本轮抽取到的记忆并入草稿，同时用“重要度×时间衰减”做加权更新心向量。
        tau_minutes: 时间常数，默认 4 小时。
        """
        now = datetime.datetime.now()
        self.last_ts = now
        if self.title_hint is None:
            # 第一轮先给个标题线索（最终建库时会用 utils.suggest_event_title_simple 再生成）
            self.title_hint = (turn_text or "").strip()[:40]

        # 构造用于更新心向量的加权样本
        vecs: List[Tuple[List[float], float]] = []
        for mu in units:
            imp = max(0.0, min(1.0, float(getattr(mu, "importance", 0.0))))
            w_imp = 0.2 + 0.8 * imp # 重要度最低给 0.2 基线，避免完全忽略；最高 1.0
            # 时间衰减：越新权重越高
            dt_min = max(0.0, (now - (mu.timestamp or now)).total_seconds() / 60.0) #经过的分钟数
            w_time = math.exp(-dt_min / max(1e-6, float(tau_minutes))) # 分钟数做指数衰减
            w = w_imp * (0.5 + 0.5 * w_time)  # 稍抬近期样本权重，时间重要性范围是 [0.5,1.0]
            if mu.embedding:
                vecs.append((mu.embedding, w))
            self.members.append(mu)

        # turn 级别嵌入也并入一点，稳住主题
        if turn_emb:
            vecs.append((turn_emb, 0.6))

        self.centroid = self._weighted_update(self.centroid, vecs)
        if not self.members:
            self.start_ts = now  # 第一次真正纳入成员时再刷新起始时间


class EventDraftManager:
    """
    管理“当前事件草稿”。只在明确结束时 finalize → 真·事件表 & 绑定关系落库。
    """

    def __init__(
        self,
        store: MemoryStore,
        sim_threshold: float = 0.72,
        max_idle_minutes: int = 180,
    ):
        self.store = store
        self.sim_threshold = float(sim_threshold)
        self.max_idle_minutes = int(max_idle_minutes)
        self.current: Optional[EventDraft] = None # 当前草稿

    def _similar_to_current(self, emb: List[float]) -> float:
        """计算与当前草稿心向量的相似度；无草稿或无心向量时返回 -1.0"""
        if not (self.current and self.current.centroid and emb):
            return -1.0
        return _cos(_as_float_list(emb), _as_float_list(self.current.centroid))

    def ingest(self, turn_text: str, turn_emb: List[float], units: List[MemoryUnit]):
        """
        将本轮”相似的记忆“聚拢进当前草稿；如未有草稿则新建。
        不触库（不创建event，不写event_id）。
        turn_text/turn_emb: 本轮对话文本及其嵌入
        units: 本轮抽取到的记忆单元列表
        """
        now = datetime.datetime.now()
        # 空集合直接返回（没有新增记忆就不扰动状态）
        if not units and not turn_emb:
            return

        if self.current is None:
            self.current = EventDraft()
            self.current.add_turn_mem(turn_text, turn_emb, units)
            return

        # 若与当前心向量相似度过低，或草稿闲置时间过长 ——> 开启新草稿（但不自动落库旧草稿）
        sim = self._similar_to_current(
            turn_emb or (units[0].embedding if units and units[0].embedding else [])
        )
        idle_min = (now - self.current.last_ts).total_seconds() / 60.0
        if (sim >= 0 and sim < self.sim_threshold) or (
            idle_min > self.max_idle_minutes
        ):
            # 开启新的主题草稿；旧草稿仍驻内存，等待你手动 finalize()/discard()
            self.current = EventDraft()

        self.current.add_turn_mem(turn_text, turn_emb, units)

    def finalize_current(self, etype: str = "misc") -> Optional[str]:
        """
        明确结束当前事件：创建 event 记录并绑定成员 memory.event_id。
        返回新建 event_id；若无草稿或成员为空，返回 None。
        """
        if not self.current or not self.current.members:
            self.current = None
            return None

        draft = self.current
        self.current = None  # 先清空，避免中途异常导致重复 finalize

        # 生成标题（复用你现有 utils 的简易标题器）
        mu_texts = [m.content for m in draft.members if getattr(m, "content", None)]
        title = suggest_event_title_simple(
            draft.title_hint or "", mu_texts, etype=etype
        )

        centroid = _as_float_list(draft.centroid or [])
        if not centroid and draft.members:
            # 兜底：用成员的平均向量
            vecs = [m.embedding for m in draft.members if m.embedding]
            if vecs:
                L = len(vecs[0])
                avg = [0.0] * L
                for v in vecs:
                    for i in range(min(L, len(v))):
                        avg[i] += v[i]
                centroid = _l2_normalize([x / max(1, len(vecs)) for x in avg])

        if not centroid:
            # 实在没有向量，就不建事件，避免脏数据
            return None

        ev = self.store.create_event(title=title, centroid=centroid, etype=etype)
        if not ev:
            return None
        ev_id = ev["id"]

        # 绑定成员并刷新事件心向量（轻量 EMA）
        for m in draft.members:
            try:
                self.store.bind_memory_event(m.id, ev_id)
                seed = m.embedding or centroid
                self.store.touch_event_with_embedding(
                    ev_id, _as_float_list(seed), alpha=0.7
                )
            except Exception:
                # 单条绑定失败不影响其他
                continue

        # 更新事件时间段（可选：以草稿起止时间覆写）
        try:
            self.store.exec(
                "UPDATE event SET start_ts=%s, end_ts=%s, status='active', updated_at=now() WHERE id=%s",
                (draft.start_ts, draft.last_ts, ev_id),
            )
        except Exception:
            pass

        return ev_id

    def discard_current(self):
        """放弃当前草稿（不建事件、不绑定）。"""
        self.current = None

    # —— 可选工具：把当前草稿快速切段，但不立即落库
    def new_segment(self):
        """结束当前段并开一个新的空草稿（旧草稿仍驻内存，等待你 finalize/ discard）。"""
        self.current = EventDraft()


# ---------------------- Toolboxes ----------------------
class RetrievalManager:
    """检索工具箱：包含所有检索记忆的方法。"""

    def retrieve_by_embedding_python(
        self, store: MemoryStore, query_embedding: List[float], top_k: int = 3
    ) -> List[MemoryUnit]:
        """基于嵌入向量的检索方法"""
        matched_memories = []

        for memory in store.get_all():
            if memory.embedding is not None:
                # 计算余弦相似度
                similarity = self._cosine_similarity(query_embedding, memory.embedding)
                matched_memories.append((memory, similarity))

        # 按照相似度排序并返回前 top_k 个记忆
        matched_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in matched_memories[:top_k]]

    # 兼容旧接口：融合检索（本地计算）
    def retrieve_by_fusion(
        self,
        store: "MemoryStore",
        query_embedding: List[float],
        tag_prefix: Optional[str] = None,
        top_k: int = 3,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        half_life_days: float = 30.0,
    ) -> List[MemoryUnit]:
        now = datetime.datetime.now()

        def time_decay(ts: datetime.datetime) -> float:
            dt_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            return pow(0.5, dt_days / max(1e-6, half_life_days))

        candidates = []
        for mem in store.get_all():
            if mem.embedding is None:
                continue
            if tag_prefix and not any(
                match_tag_prefix(t, tag_prefix) for t in mem.tags
            ):
                continue
            sim = self._cosine_similarity(query_embedding, mem.embedding)
            td = time_decay(mem.timestamp)
            score = alpha * sim + beta * float(mem.importance) + gamma * td
            candidates.append((score, mem))

        candidates.sort(key=lambda x: x[0], reverse=True)
        out = [m for _, m in candidates[:top_k]]
        for m in out:
            store.update_retrieval_stats(m.id, 1)
        return out

    # memory.py — RetrievalManager 内新增
    def retrieve_by_embedding_DB_python(
        self, store: "MemoryStore", query_embedding: List[float], top_k: int = 3
    ) -> List[MemoryUnit]:
        """
        Hybrid：DB 近似召回 + Python 精确重排
        - 既保留 DB 的速度，又尽量贴近你旧版的“准”的感觉
        """
        import math

        def cos(a, b):
            if not a or not b:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0

        # 1) 先用 DB 召回更大的候选集
        cand_k = max(top_k * 6, 48)
        try:
            cands = store.search_candidates_by_embedding(
                query_embedding, limit=cand_k, probes=12
            )
        except Exception:
            cands = [m for m in store.get_all() if m.embedding]

        # 2) Python 精确余弦重排（复刻你之前的排序方式）
        scored = [(m, cos(query_embedding, m.embedding)) for m in cands if m.embedding]
        scored.sort(key=lambda t: t[1], reverse=True)
        out = [m for m, _ in scored[:top_k]]

        # 3) 更新检索统计
        for m in out:
            store.update_retrieval_stats(m.id, 1)

        return out

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


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


# ---------------------- Update Manager (LLM extraction) ----------------------
from utils import load_bgem3, embed_text
import requests

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

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", (s or "").strip())


def build_importance(v) -> float:
    try:
        x = float(v)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def build_text(v) -> str:
    return (v if isinstance(v, str) else str(v or "")).strip()


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


# shared embedder (CPU ok)
_bgem3 = load_bgem3("BAAI/bge-m3", device="cpu")


class UpdateManager:
    """
    build_memories_from_raw
    保持原有接口不变，但把关键中间结果挂到实例属性上，便于在 UI 中展示：
      - self.last_http_json:   HTTP 层原始 JSON（str）
      - self.last_model_text:  模型文本（去围栏后）（str）
      - self.last_parsed_json: 被 json.loads 的数组字符串（str）
    """

    def __init__(self):
        self.last_http_json: str = ""
        self.last_model_text: str = ""
        self.last_parsed_json: str = ""

    def build_memories_from_raw(self, rawdata: str) -> List[MemoryUnit]:
        url = "https://api.deepseek.com/v1/chat/completions"
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION},
                {"role": "user", "content": rawdata},
            ],
            "temperature": 0,
            "stream": False,
        }
        with open("../API-KEY.txt", "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        self.last_http_json = ""
        self.last_model_text = ""
        self.last_parsed_json = ""

        # simple retry
        attempt, max_attempts = 0, 2
        while True:
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=60)
            except requests.exceptions.RequestException as e:
                if attempt < max_attempts:
                    time.sleep(2**attempt)
                    attempt += 1
                    continue
                raise RuntimeError(f"【网络错误】调用 LLM 失败：{e}") from e
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt < max_attempts:
                    time.sleep(2**attempt)
                    attempt += 1
                    continue
                raise RuntimeError(
                    f"【API错误】{r.status_code}，重试 {max_attempts} 次后仍失败：{r.text[:500]}"
                )
            if r.status_code != 200:
                raise RuntimeError(f"【API错误】{r.status_code}：{r.text[:500]}")
            data = r.json()
            break

        # 保存 HTTP 层原始 JSON（字符串化，避免非序列化类型）
        try:
            self.last_http_json = json.dumps(data, ensure_ascii=False, indent=2)[:20000]
        except Exception:
            self.last_http_json = str(data)[:20000]

        # 提取模型文本（choices[0].message.content）
        content = ""
        if isinstance(data, dict) and "choices" in data:
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
        if not content and isinstance(data, dict):
            msg = data.get("message")
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
            if not content:
                content = data.get("response", "")

        content = _strip_code_fences(content)
        self.last_model_text = content  # 记录去围栏后的模型原文

        if not content:
            raise RuntimeError(f"【解析错误】LLM 返回为空或未知结构：{str(data)[:500]}")

        # 解析为 JSON 数组
        try:
            # 直接尝试
            items = json.loads(content)
            if not isinstance(items, list):
                raise ValueError("非数组")
            self.last_parsed_json = content
        except Exception:
            # 回退：宽松正则抽取首个 JSON 数组
            m = re.search(r"\[\s*{[\s\S]*?}\s*\]", content)
            if not m:
                # 把原始文本和 HTTP JSON 都留给 UI 查看
                raise RuntimeError(
                    "【解析错误】无法从模型文本中提取 JSON 数组，请在侧边栏调试区查看 last_model_text / last_http_json。"
                )
            arr_text = m.group(0)
            self.last_parsed_json = arr_text
            items = json.loads(arr_text)
            if not isinstance(items, list):
                raise RuntimeError("【解析错误】提取到的片段不是 JSON 数组。")

        # 构造 MemoryUnit
        memory_units: List[MemoryUnit] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            c = build_text(it.get("content", ""))
            p = build_importance(it.get("importance", 0.0))
            c_embedding = embed_text(_bgem3, f"passage:{c}").tolist() if c else None
            if c:
                memory_units.append(
                    MemoryUnit(content=c, importance=p, embedding=c_embedding)
                )
        return memory_units


# ---------------------- System facade ----------------------
class MemorySystem:
    def __init__(
        self,
        store: MemoryStore,
        retriever: RetrievalManager,
        forgetter: "ForgettingManager",
        reflecter: "ReflectionManager",
    ):
        self.store = store
        self.retriever = retriever
        self.forgetter = forgetter
        self.reflecter = reflecter

    def add_memory(self, memory: MemoryUnit):
        self.store.add(memory)

    def perform_forgetting(self, strategy: str, **kwargs):
        ids_to_forget = []
        if strategy == "importance":
            ids_to_forget = self.forgetter.by_importance(self.store, **kwargs)
        elif strategy == "time_decay":
            ids_to_forget = self.forgetter.by_time_decay(self.store, **kwargs)
        for mem_id in ids_to_forget:
            self.store.delete(mem_id)

    def print_status(self):
        memories = self.store.get_all()
        print(f"总记忆数: {len(memories)}")
