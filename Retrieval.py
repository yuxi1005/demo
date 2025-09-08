import datetime
import math
from typing import List, Optional
from utils import match_tag_prefix
from memory import MemoryUnit, MemoryStore

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