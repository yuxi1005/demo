import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from utils import (
    _l2_normalize,
    _cos,
    _as_float_list,
    suggest_event_title_simple
)
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import datetime, math
from memory import MemoryUnit, MemoryStore

load_dotenv()  # 默认会读取当前目录下的 .env

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


# ------------------- Enhanced EventDraftManager -------------------

class EventDraftManager:
    """
    管理“活跃事件草稿池”。按“三步路由”处理新到的 Event 类记忆：
    1) 先尝试并入活跃草稿（宽松阈值 THRESHOLD_JOIN_DRAFT）
    2) 再尝试链接历史事件（严格阈值 THRESHOLD_JOIN_EVENT）
    3) 否则新建一个全新的 EventDraft
    """

    def __init__(
        self,
        store: MemoryStore,
        THRESHOLD_JOIN_DRAFT: float = 0.72,   # 步骤1：宽松
        THRESHOLD_JOIN_EVENT: float = 0.86,   # 步骤2：严格
        max_idle_minutes: int = 180,
        max_active_drafts: int = 8,           # 软上限：避免草稿池无限增长
    ):
        self.store = store
        self.THRESHOLD_JOIN_DRAFT = float(THRESHOLD_JOIN_DRAFT)
        self.THRESHOLD_JOIN_EVENT = float(THRESHOLD_JOIN_EVENT)
        self.max_idle_minutes = int(max_idle_minutes)
        self.max_active_drafts = int(max_active_drafts)

        # 改：由“单一 current”升级为“活跃草稿池”
        self._active: List[EventDraft] = []

    # ========== 工具方法 ==========

    def _best_matching_draft(self, emb: List[float]) -> Optional[tuple]:
        """
        在活跃草稿池中寻找与给定向量最相似的草稿。
        返回 (idx, sim)；若无有效候选返回 None。
        """
        if not self._active or not emb:
            return None
        best_idx, best_sim = -1, -1.0
        e = _as_float_list(emb)
        for i, d in enumerate(self._active):
            if not (d and d.centroid):
                continue
            sim = _cos(_as_float_list(d.centroid), e)
            if sim > best_sim:
                best_idx, best_sim = i, sim
        return (best_idx, best_sim) if best_idx >= 0 else None

    def _prune_idle_drafts(self):
        """基于空闲时间简单清理活跃草稿池。"""
        now = datetime.datetime.now()
        kept = []
        for d in self._active:
            idle_min = (now - d.last_ts).total_seconds() / 60.0
            if idle_min <= self.max_idle_minutes:
                kept.append(d)
        self._active = kept[: self.max_active_drafts]

    def _bind_to_event(self, ev_id: str, units: List[MemoryUnit], centroid_hint: Optional[List[float]] = None):
        """
        把本轮记忆直接绑定到历史事件，并用 EMA 刷新事件质心。
        """
        for m in units:
            try:
                self.store.bind_memory_event(m.id, ev_id)
                seed = m.embedding or centroid_hint
                if seed:
                    self.store.touch_event_with_embedding(ev_id, _as_float_list(seed), alpha=0.7)
            except Exception:
                continue

    # ========== 对外主流程：三步路由 ==========

    def route_event_memory(
        self,
        turn_text: str,
        turn_emb: List[float],
        units: List[MemoryUnit],
        *,
        search_event_limit: int = 8,
    ) -> Dict[str, Any]:
        """
        三步路由入口（你在“当被分类为 Event 的新记忆产生时”调用它）

        返回形如：
        {
          "action": "merge_draft" | "link_event" | "new_draft" | "noop",
          "detail": {...}
        }
        """
        now = datetime.datetime.now()
        self._prune_idle_drafts()

        # 空输入直接略过
        if not units and not turn_emb:
            return {"action": "noop", "detail": {"reason": "empty_input"}}

        # 代表本轮主题向量（优先 turn_emb，其次第一条记忆的 embedding）
        anchor = turn_emb or ((units[0].embedding) if units and units[0].embedding else [])

        # ========== 第一步：匹配活跃草稿（宽松阈值） ==========
        if anchor:
            best = self._best_matching_draft(anchor)
        else:
            best = None

        if best is not None:
            idx, sim = best
            if sim >= self.THRESHOLD_JOIN_DRAFT:
                # 并入最相似草稿
                self._active[idx].add_turn_mem(turn_text, turn_emb, units)
                return {
                    "action": "merge_draft",
                    "detail": {"draft_index": idx, "similarity": sim, "draft_title_hint": self._active[idx].title_hint},
                }

        # ========== 第二步：链接历史事件（严格阈值） ==========
        # 需要 MemoryStore 提供 search_events_by_embedding(embedding, limit) → List[{"id","centroid",...}]
        # 若你暂时没有该接口，会自动跳过此步。
        if anchor:
            try:
                ev_cands = self.store.search_events_by_embedding(anchor, limit=search_event_limit) or []
            except Exception:
                ev_cands = []

            # 计算最相似历史事件
            best_ev_id, best_ev_sim, best_ev_centroid = None, -1.0, None
            for ev in ev_cands:
                c = _as_float_list(ev.get("centroid"))
                if not c:
                    continue
                sim = _cos(c, _as_float_list(anchor))
                if sim > best_ev_sim:
                    best_ev_id, best_ev_sim, best_ev_centroid = ev.get("id"), sim, c

            if best_ev_id and best_ev_sim >= self.THRESHOLD_JOIN_EVENT:
                # 直接把本轮记忆绑定到历史事件上，并刷新事件质心
                self._bind_to_event(best_ev_id, units, centroid_hint=best_ev_centroid)
                return {
                    "action": "link_event",
                    "detail": {"event_id": best_ev_id, "similarity": best_ev_sim},
                }

        # ========== 第三步：新建全新 EventDraft ==========
        new_draft = EventDraft()
        new_draft.add_turn_mem(turn_text, turn_emb, units)
        self._active.append(new_draft)
        # 轻度约束草稿池大小
        if len(self._active) > self.max_active_drafts:
            self._prune_idle_drafts()

        return {
            "action": "new_draft",
            "detail": {"draft_index": len(self._active) - 1, "title_hint": new_draft.title_hint},
        }

    # ====== 兼容 & 管理接口 ======

    def finalize_draft(self, draft_index: int, etype: str = "misc") -> Optional[str]:
        """
        将指定草稿落库为真正事件，并绑定成员。
        返回新建 event_id；若 index 不合法或成员为空，返回 None。
        """
        if draft_index < 0 or draft_index >= len(self._active):
            return None

        draft = self._active[draft_index]
        if not draft or not draft.members:
            # 移除空草稿
            self._active.pop(draft_index)
            return None

        # 生成标题
        mu_texts = [m.content for m in draft.members if getattr(m, "content", None)]
        title = suggest_event_title_simple(draft.title_hint or "", mu_texts, etype=etype)

        # 计算质心
        centroid = _as_float_list(draft.centroid or [])
        if not centroid and draft.members:
            vecs = [m.embedding for m in draft.members if m.embedding]
            if vecs:
                L = len(vecs[0])
                avg = [0.0] * L
                for v in vecs:
                    for i in range(min(L, len(v))):
                        avg[i] += v[i]
                centroid = _l2_normalize([x / max(1, len(vecs)) for x in avg])

        if not centroid:
            # 没有有效向量，不建事件
            self._active.pop(draft_index)
            return None

        ev = self.store.create_event(title=title, centroid=centroid, etype=etype)
        if not ev:
            # 创建失败，保留草稿以便重试
            return None
        ev_id = ev["id"]

        # 绑定成员 & 轻量 EMA 刷新
        for m in draft.members:
            try:
                self.store.bind_memory_event(m.id, ev_id)
                seed = m.embedding or centroid
                self.store.touch_event_with_embedding(ev_id, _as_float_list(seed), alpha=0.7)
            except Exception:
                continue

        # 更新事件时间段
        try:
            self.store.exec(
                "UPDATE event SET start_ts=%s, end_ts=%s, status='active', updated_at=now() WHERE id=%s",
                (draft.start_ts, draft.last_ts, ev_id),
            )
        except Exception:
            pass

        # 移除已落库的草稿
        self._active.pop(draft_index)
        return ev_id

    # 便捷方法：落库所有有成员的草稿
    def finalize_all(self, etype: str = "misc") -> List[str]:
        res = []
        # 注意：索引变化，倒序处理更安全
        for idx in reversed(range(len(self._active))):
            ev_id = self.finalize_draft(idx, etype=etype)
            if ev_id:
                res.append(ev_id)
        return res

    # 向后兼容：保留旧 API，但内部走新流程
    def ingest(self, turn_text: str, turn_emb: List[float], units: List[MemoryUnit]):
        """
        为兼容旧调用保留。等价于 route_event_memory，不过不返回动作详情。
        """
        _ = self.route_event_memory(turn_text, turn_emb, units)

    def discard_all(self):
        """丢弃全部活跃草稿（不落库、不绑定）。"""
        self._active = []

    @property
    def active_drafts(self) -> List[EventDraft]:
        """只读访问活跃草稿池。"""
        return list(self._active)
