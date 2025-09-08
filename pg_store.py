# pg_store.py
import os, datetime
from typing import List, Optional
import psycopg2
import psycopg2.extras
from dataclasses import asdict
from memory import MemoryUnit  # 直接用你现有 dataclass
from dotenv import load_dotenv

load_dotenv()
PG_DSN = os.getenv("PG_DSN")

class PostgresStore:
    def __init__(self):
        self.conn = psycopg2.connect(PG_DSN)
        self.conn.autocommit = True

    def add(self, m: MemoryUnit):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO memory (id, content, importance, embedding, timestamp, retrieval_count, last_accessed_ts)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO UPDATE SET
                  content=EXCLUDED.content,
                  importance=EXCLUDED.importance,
                  embedding=EXCLUDED.embedding,
                  timestamp=EXCLUDED.timestamp
            """, (
                m.id, m.content, float(m.importance),
                m.embedding,  # psycopg2 会把 list[float] 映射为 vector
                m.timestamp, int(m.retrieval_count), m.last_accessed_ts
            ))

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM memory WHERE id=%s", (memory_id,))
            row = cur.fetchone()
            return self._row_to_mu(row) if row else None

    def get_all(self) -> List[MemoryUnit]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM memory")
            return [self._row_to_mu(r) for r in cur.fetchall()]

    def delete(self, memory_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM memory WHERE id=%s", (memory_id,))
            return cur.rowcount > 0

    def clear(self):
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE memory")

    def update_retrieval_stats(self, memory_id: str, inc: int = 1):
        with self.conn.cursor() as cur:
            cur.execute("""
              UPDATE memory
                 SET retrieval_count = retrieval_count + %s,
                     last_accessed_ts = now()
               WHERE id=%s
            """, (inc, memory_id))

    @staticmethod
    def _row_to_mu(r) -> MemoryUnit:
        return MemoryUnit(
            id=str(r["id"]),
            content=r["content"],
            embedding=list(r["embedding"]) if r["embedding"] is not None else None,
            timestamp=r["timestamp"],
            importance=float(r["importance"]),
            retrieval_count=int(r["retrieval_count"]),
            last_accessed_ts=r["last_accessed_ts"]
        )
