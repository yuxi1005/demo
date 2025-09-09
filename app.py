# ------------- 标准库 -------------
import uuid
import re
import json
import time
import traceback
import datetime as dt
from datetime import datetime
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import (
    List, Dict, Any, Optional, Tuple,
    Iterable, Literal, overload, cast
)

# ------------- 第三方库 -------------
import streamlit as st

# ------------- 项目内部 -------------
from llm_clients import chat_with_memories
from memory import (
    MemoryUnit,
    MemoryStore,
    assign_event_for_units,
    # build_mu_from_raw,
)
from utils import (
    load_bgem3, embed_text, clean_stream, _cos,
    call_llm_segment_and_extract, run_llm, _ensure_float_ts
)
from Retrieval import RetrievalManager
@st.cache_resource
def get_bgem3():
    return load_bgem3("BAAI/bge-m3")


bgem3 = get_bgem3()

# ================== Streamlit UI 配置 ==================
st.set_page_config(page_title="🍝yuxi's LLM长时记忆实验", layout="wide")
st.title("🍝yuxi's LLM长时记忆实验")
stream_mode = True


# ================== 初始化核心对象（持久化到 session） ==================
def ensure_core_in_session():
    if "memory_store" not in st.session_state:
        st.session_state.memory_store = MemoryStore()
    if "retriever" not in st.session_state:
        st.session_state.retriever = RetrievalManager()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = 3
    if "mem_view_mode" not in st.session_state:
        st.session_state.mem_view_mode = "recent"  # "recent" | "paged"
    if "mem_page" not in st.session_state:
        st.session_state.mem_page = 1
    if "mem_page_size" not in st.session_state:
        st.session_state.mem_page_size = 20
    if "provider" not in st.session_state:
        st.session_state.provider = "ollama"  # or "ollama" "deepseek"
    if "model_name" not in st.session_state:
        st.session_state.model_name = (
            "qwen2.5:14b"  # ollama: "qwen2.5:14b","qwen2.5:14b",deepseek-chat
        )
    if "recent_k" not in st.session_state:
        st.session_state.recent_k = 3

ensure_core_in_session()

memory_store: MemoryStore = st.session_state.memory_store
# forgetter: ForgettingManager = st.session_state.forgetter
# reflecter: ReflectionManager = st.session_state.reflecter
# memory_system: MemorySystem = st.session_state.memory_system
# updater: UpdateManager = st.session_state.updater
retrieval_manager: RetrievalManager = st.session_state.retriever

# ================== Sidebar 配置（写回 retriever） ==================
with st.sidebar:
    st.header("🛠️ 记忆系统配置")

    # 设置检索方法
    retrieval_method = st.selectbox(
        "记忆检索方法",
        ["Hybrid (DB recall + Python rerank)", "fusion", "Python (legacy cosine)"],
        help="Hybrid 推荐：DB 先取候选，再用 Python 精确余弦重排。",
    )

    if retrieval_method.startswith("fusion"):
        retriever = retrieval_manager.retrieve_by_fusion  # 现有 DB 版
    elif retrieval_method.startswith("Python"):
        retriever = retrieval_manager.retrieve_by_embedding_python  # 旧版 Python 余弦
    else:
        retriever = retrieval_manager.retrieve_by_embedding_DB_python  # 新增 Hybrid

    # 设置检索配置
    st.divider()
    st.header("📚 记忆库视图")
    mem_view_mode = st.radio(
        "显示模式",
        ["最近 20 条", "分页"],
        index=0 if st.session_state.mem_view_mode == "recent" else 1,
    )
    st.session_state.mem_view_mode = (
        "recent" if mem_view_mode == "最近 20 条" else "paged"
    )

    if st.session_state.mem_view_mode == "paged":
        page_size = st.number_input(
            "每页条数",
            min_value=5,
            max_value=100,
            value=st.session_state.mem_page_size,
            step=5,
        )
        st.session_state.mem_page_size = int(page_size)

    st.divider()
    st.header("🧹 工具")

    def _ts_iso(ts):
        return ts.isoformat() if isinstance(ts, dt.datetime) else None

    # 导出记忆
    def export_memories() -> bytes:
        store = st.session_state.memory_store
        rows = []
        for m in store.get_all():  # 你的遍历方式按实际API调整
            rows.append({
                "id": getattr(m, "id", None),
                "content": getattr(m, "content", None),
                "timestamp": _ts_iso(getattr(m, "timestamp", None)),
                "importance": getattr(m, "importance", 0.0),
                "retrieval_count": getattr(m, "retrieval_count", 0),
                "last_accessed_ts": _ts_iso(getattr(m, "last_accessed_ts", None)),
                "embedding_dim": len(getattr(m, "embedding", []) or []),
            })
        return json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")

    st.download_button(
    "📤 导出记忆（JSON）",
    data=export_memories(),
    file_name=f"memories_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
    use_container_width=True,
)

    # 导入记忆
    uploaded = st.file_uploader("📥 导入记忆（JSON）", type=["json"])
    if uploaded is not None:
        try:
            payload = json.load(uploaded)
            count = 0
            for item in payload.get("memories", []):
                content = item.get("content")
                if not content:
                    continue
                m = MemoryUnit(content=content)
                # 可选恢复统计字段（你的类若支持可加）
                if hasattr(m, "id") and item.get("id"):
                    m.id = item["id"]
                if hasattr(m, "timestamp") and item.get("timestamp"):
                    try:
                        m.timestamp = dt.datetime.fromisoformat(item["timestamp"])
                    except Exception:
                        pass
                if (
                    hasattr(m, "retrieval_count")
                    and item.get("retrieval_count") is not None
                ):
                    m.retrieval_count = int(item["retrieval_count"])
                if hasattr(m, "last_accessed_ts") and item.get("last_accessed_ts"):
                    try:
                        m.last_accessed_ts = dt.datetime.fromisoformat(
                            item["last_accessed_ts"]
                        )
                    except Exception:
                        pass
                memory_store.add(m)
                count += 1
            st.success(f"导入完成：{count} 条记忆。")
        except Exception as e:
            st.error(f"导入失败：{e}")

    # 一键清空记忆
    if st.button("🗑️ 一键清空记忆", type="secondary", use_container_width=True):
        try:
            if hasattr(memory_store, "clear"):
                memory_store.clear()
            else:
                # 没有 clear 方法就逐条删除
                for m in list(memory_store.get_all()):
                    if hasattr(memory_store, "delete"):
                        memory_store.delete(m.id)
            st.success("记忆已清空。")
        except Exception as e:
            st.error(f"清空失败：{e}")

    st.header("📚 记忆库实时视图")
    all_memories_view = st.container()

# ================== 历史消息渲染 ==================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ================== 触发批量处理的条件函数 ==================
def should_trigger_by_rounds(
    round_cap: int = 10,
    *,
    roles_for_rounds: Iterable[str] = ("user",),  # 计“轮”的角色，默认只算用户
) -> Tuple[bool, int]:
    """
    仅按“轮数”判断是否触发。
    返回: (should_by_rounds, rounds_since_last)
      - should_by_rounds: 自上次批处理以来，满足 roles_for_rounds 的消息数 >= round_cap
      - rounds_since_last: 自 last_batch_msg_idx 之后的“轮数”（按角色过滤计数）
    说明：
      - 一“轮” = 一条满足 roles_for_rounds 的消息；通常就是 user 发言次数。
      - 不处理 idle、时间等其他条件。
    """
    msgs = getattr(st.session_state, "messages", [])
    if not msgs:
        return False, 0

    last_idx_prev = int(st.session_state.get("last_batch_msg_idx", -1))
    start_idx = max(-1, last_idx_prev) + 1
    role_set = set(roles_for_rounds)

    rounds_since_last = 0
    if start_idx < len(msgs):
        for m in msgs[start_idx:]:
            if (m or {}).get("role") in role_set:
                rounds_since_last += 1

    return (rounds_since_last >= round_cap), rounds_since_last

def _keep_last_two_sentences(text: str) -> str:
    """
    保留文本的最后两句话。
    用中英文常见句号/问号/叹号作为分隔。
    """
    # 用正则切分句子，保留分隔符
    parts = re.split(r'([。！？!?\.])', text)
    # 把句子和标点拼回去
    sentences = ["".join(parts[i:i+2]).strip() for i in range(0, len(parts), 2) if parts[i].strip()]
    if len(sentences) <= 2:
        return text.strip()
    return "".join(sentences[-2:]).strip()

def _parse_ts(ts: Any) -> Optional[float]:
    """统一转 epoch 秒；失败返回 None。支持 datetime/秒/毫秒/字符串数字。"""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.timestamp()
    try:
        x = float(ts)
    except Exception:
        return None
    return x / 1000.0 if x > 1e12 else x  # 13位毫秒 → 秒

def _sanitize_msg(x: Any) -> Dict[str, Any]:
    """
    把任意对象标准化为 {role, content, ts} 的 dict。
    - role/内容缺失时给空字符串；
    - ts 尝试解析为秒，否则 None（不丢字段，便于后续使用）。
    """
    if isinstance(x, dict):
        role = str(x.get("role", "") or "")
        content = str(x.get("content", "") or "")
        ts = _parse_ts(x.get("ts"))
        return {"role": role, "content": content, "ts": ts}
    # 兜底：把奇怪类型收敛成一条“未知角色”的消息
    return {"role": "", "content": str(x), "ts": None}

def get_latest_messages(
    k: int = 1,
    *,
    roles_for_idle: Iterable[str] = ("user",),   # 计算空闲时长时关注的角色
    return_last_ts: bool = False,                # True: 返回 (messages, last_ts)
) -> List[Dict[str, str]] | Tuple[List[Dict[str, str]], Optional[float]]:
    """
    从 Streamlit 会话状态中获取最近的对话消息。
    - 如果 k=1，只返回最后一条用户消息。
    - 如果 k>1，返回最近 k 轮完整的用户-助手对话。
    对助手消息只保    留最后两句话（通过 _keep_last_two_sentences）。
    额外：当 return_last_ts=True 时，返回 (消息列表, roles_for_idle 的最后一条消息时间戳[秒])。
    """

    raw = getattr(st.session_state, "messages", None)
    if not raw:
        return ([], None) if return_last_ts else []

    # 先把全量消息“标准化”，避免后面因类型混入而炸
    messages: List[Dict[str, Any]] = [_sanitize_msg(m) for m in raw]

    latest_messages: List[Dict[str, Any]] = []
    conversation_count = 0

    # === 1) 反向采集，得到最近 k 轮（按 user 计轮次）===
    for i in range(len(messages) - 1, -1, -1):
        m = dict(messages[i])  # 复制，避免改源
        if m.get("role") == "assistant":
            m["content"] = _keep_last_two_sentences(m.get("content", "") or "")

        if m.get("role") == "user":
            conversation_count += 1

        latest_messages.append(m)
        if conversation_count >= k:
            break

    latest_messages.reverse()

    # === 2) k==1 且仅需最后一条用户消息时的优化返回 ===
    if k == 1 and latest_messages and not return_last_ts:
        for msg in reversed(latest_messages):
            if msg.get("role") == "user":
                return [msg]

    # === 3) 需要返回最后一条“关注角色”的时间戳 ===
    last_ts: Optional[float] = None
    if return_last_ts:
        role_set = set(roles_for_idle)
        for m in reversed(messages):  # 用全量 messages 计算 last_ts（不受 k 截断）
            if m.get("role") in role_set:
                t = m.get("ts")
                if t is None:
                    t = _parse_ts(m.get("ts"))  # 再兜一层，虽按理已在 _sanitize_msg 做过
                if t is not None:
                    last_ts = t
                    break

    return (latest_messages, last_ts) if return_last_ts else latest_messages

def compute_adaptive_k(
    max_k: int = 20,
    min_k: int = 1,
    idle_threshold_sec: int = 5 * 60,  # 满足：最近一次停滞 ≥ 此阈值
    turn_threshold: int = 4,           # 满足：最近用户轮数 ≥ 此阈值
) -> int:
    """
    自适应返回 k：
    - 若“停滞时间”或“最近轮数”任一触发，则返回对应的 k（并裁剪到 [min_k, max_k]）
    - 否则返回 1（仅处理本轮）
    不做任何 embedding/相似度计算，轻量稳健。
    """
    msgs = getattr(st.session_state, "messages", [])
    if not msgs:
        return 1

    n = len(msgs)
    if n == 0:
        return 1

    # ---------- 条件 A：长时间停滞 ----------
    # 找到“最近一次大停顿”后第一条消息，把该点之后的用户轮数作为 k 候选
    idle_k: Optional[int] = None
    # 仅当消息都有 ts 时才启用停滞判断；否则忽略此条件
    if all(m.get("ts") is not None for m in msgs[-min(n, 2*max_k):]):
        now_ts = time.time()
        last_ts = msgs[-1].get("ts") or now_ts
        prev_ts = last_ts
        user_turns_after_idle = 0
        for i in range(n - 1, -1, -1):
            ts_i = msgs[i].get("ts")
            gap = prev_ts - ts_i
            prev_ts = ts_i
            if msgs[i].get("role") == "user":
                user_turns_after_idle += 1
            if gap >= idle_threshold_sec:
                idle_k = max(1, user_turns_after_idle)
                break
        # 若一直没有达到停滞阈值，idle_k 维持 None

    # ---------- 条件 B：最近轮数较多 ----------
    # 从结尾向前数最近的用户消息条数，作为 k 候选
    recent_user_turns = 0
    for i in range(n - 1, -1, -1):
        if msgs[i].get("role") == "user":
            recent_user_turns += 1
        # 不必扫太多，够用就停
        if recent_user_turns >= max_k:
            break
    turns_k: Optional[int] = recent_user_turns if recent_user_turns >= turn_threshold else None

    # ---------- 合并策略 ----------
    # 优先使用“停滞触发”的 k，否则看“轮数触发”，否则返回 1
    if isinstance(idle_k, int):
        k = max(min_k, min(idle_k, max_k))
        return k

    if isinstance(turns_k, int):
        k = max(min_k, min(turns_k, max_k))
        return k

    return 1

# 水位管理
def _get_watermark() -> int:
    return int(st.session_state.get("processed_until_msg_idx", -1))

def _set_watermark(idx: int):
    st.session_state["processed_until_msg_idx"] = int(idx)

def persist_segments_except_last_with_wm(
    store: "MemoryStore",
    result: dict,
    *,
    verbose: bool = True,
    log=None,
    allow_persist_when_single_segment: bool = False,
) -> list:
    """
    处理 LLM 切分+抽取结果，持久化“非最后段”的记忆单元。记录水位。
    """
    import traceback

    def _L(*args):
        if verbose:
            (log or print)(*args)

    segments = result.get("segments") or []
    if not isinstance(segments, list):
        _L("⚠️ result['segments'] 不是 list：", type(segments))
        segments = []

    _L("—— persist_segments_except_last_with_wm ——")
    _L("segments.len =", len(segments))

    if not segments:
        _L("❎ 空 segments，直接返回。")
        return []

    wm = _get_watermark()
    _L("当前水位(wm) =", wm)

    # 打印每个 segment 的概览
    for idx, seg in enumerate(segments):
        b = seg.get("begin_turn", None)
        e_raw = seg.get("end_turn", None)
        try:
            e = int(e_raw) if e_raw is not None else None
        except Exception:
            e = None
        mems = seg.get("memories", []) or []
        _L(f"  seg[{idx}]: begin={b}, end={e}, memories={len(mems)}", "(最后段)" if idx == len(segments)-1 else "")

    # 选可用段：默认丢最后段；如果只有1段且允许兜底，则不丢
    if len(segments) == 1 and allow_persist_when_single_segment:
        candidate = segments  # 兜底：单段也入库
        _L("⚠️ 只有 1 段，且允许单段兜底：不过滤最后段。")
    else:
        candidate = segments[:-1]
        if len(segments) == 1:
            _L("❗ 只有 1 段，且禁止单段兜底：按规则丢弃最后段 => 0 入库。")

    # end_turn 过滤 + 记录过滤原因
    usable = []
    for idx, seg in enumerate(candidate):
        e_raw = seg.get("end_turn", None)
        try:
            e = int(e_raw)
        except Exception:
            e = None
        if e is None:
            _L(f"  ✂️ 过滤 seg(end_turn 无法解析)：end_turn={e_raw}")
            continue
        if e <= wm:
            _L(f"  ✂️ 过滤 seg(end_turn≤wm)：end_turn={e} ≤ wm={wm}")
            continue
        usable.append(seg)

    _L("通过过滤的段数 usable.len =", len(usable))

    out_units = []
    max_end = wm

    for seg in usable:
        try:
            end_turn = int(seg.get("end_turn", -1))
        except Exception:
            end_turn = -1
        max_end = max(max_end, end_turn)

        seg_mems = seg.get("memories", []) or []
        _L(f"→ 处理 seg(end_turn={end_turn})，memories={len(seg_mems)}")

        for i, m in enumerate(seg_mems):
            c = (m.get("content") or "").strip()
            if not c:
                _L(f"    ✂️ mem[{i}] content 为空，跳过")
                continue
            # importance
            try:
                p = float(m.get("importance", 0.0))
            except Exception:
                p = 0.0
            p = max(0.0, min(1.0, p))

            _L(f"    · mem[{i}] content='{c[:40]}{'...' if len(c)>40 else ''}', importance={p}")

            # embedding
            try:
                raw_emb = embed_text(bgem3, f"passage:{c}")
                if hasattr(raw_emb, "tolist"):   # numpy array 或 torch tensor
                    emb = raw_emb.tolist()
                else:
                    emb = list(raw_emb) if raw_emb is not None else None
                
                if not emb:
                    _L("      ⚠️ embedding 为空或 None")
            except Exception as ee:
                _L("      ❌ 计算 embedding 失败：", repr(ee))
                _L(traceback.format_exc())
                continue

            # DB 插入
            mu = MemoryUnit(content=c, importance=p, embedding=emb)
            try:
                store.add(mu)  # 确认这里的方法名/签名与 MemoryStore 一致
                out_units.append(mu)
                _L("      ✅ 入库成功。")
            except Exception as de:
                _L("      ❌ 入库失败：", repr(de))
                _L(traceback.format_exc())
                # 不中断整个流程，继续其他 mem

    # 推进水位
    if out_units:
        try:
            _L(f"推进水位：{wm} → {max_end}")
            _set_watermark(max_end)
        except Exception as we:
            _L("⚠️ 设置水位失败：", repr(we))
    else:
        _L("⚠️ 本次没有任何记忆入库，水位不变：", wm)

    _L("—— 结束：新增记忆条数 =", len(out_units))
    return out_units


# ================== 主聊天逻辑 ==================
if prompt := st.chat_input("请输入"):
    msgs, last_ts = get_latest_messages(
    k=compute_adaptive_k(),
    roles_for_idle=("user",),
    return_last_ts=True,
    )
    last_ts = _ensure_float_ts(last_ts)
    idle_seconds: float = 0.0 if last_ts is None else (time.time() - last_ts)

    st.session_state.messages.append({"role": "user", "content": prompt, "ts": time.time()})
    st.session_state.to_generate = True
    st.session_state.turn_id = str(uuid.uuid4())
    with st.chat_message("user", avatar="👧🏻"):
        st.markdown(prompt)

    if st.session_state.get("to_generate") and (
        st.session_state.get("turn_id") != st.session_state.get("handled_turn_id")
    ):
        prompt_embedding = embed_text(bgem3, f"query:{prompt}").tolist()

        with st.chat_message("assistant", avatar="🤡"):
            # 1) 检索记忆
            with st.spinner("正在检索相关记忆..."):
                all_memories = memory_store.get_all()
                retrieved_memories = (
                    retriever(
                        memory_store, prompt_embedding, top_k=st.session_state.top_k
                    )
                    or []
                )

                expanded_mems = []
                for m in retrieved_memories:
                    expanded_mems.append(m)
                    header, sibs = memory_store.get_event_context(m.id, k_siblings=5)
                    expanded_mems.extend(sibs)

                with st.expander("🔍 本轮检索到的记忆"):
                    if expanded_mems:
                        for mem in expanded_mems:
                            st.info(
                                f"**内容:** {mem.content}\n\n**检索次数:** {mem.retrieval_count}"
                            )
                            # ⬇️ 新增：显示所属事件与兄弟记忆
                            header, siblings = memory_store.get_event_context(
                                mem.id, k_siblings=3
                            )
                            if header:
                                st.caption(
                                    f"事件：《{header.get('title') or '未命名事件'}》"
                                    f"｜状态：{header.get('status')}｜时间窗：{header.get('start_ts')} → {header.get('updated_at')}"
                                )
                                for s in siblings:
                                    st.write(
                                        f"· 兄弟：{s.content}（imp={getattr(s,'importance',0):.2f}）"
                                    )
                    else:
                        st.warning("未检索到相关记忆。")

            # 2) 生成回复
            st.caption(
                f"provider={st.session_state.provider}, model={st.session_state.model_name}"
            )
            with st.spinner("正在生成回复..."):
                k = 5 if st.session_state.recent_k >= 5 else st.session_state.recent_k
                recent_dialog = cast(List[Dict[str, str]], get_latest_messages(k=k, return_last_ts=False) if k else [])
                
                assistant_response_stream = chat_with_memories(
                    provider=st.session_state.provider,
                    model=st.session_state.model_name,
                    history=recent_dialog,
                    retrieved_memories=expanded_mems,
                    current_query=prompt,
                    stream=stream_mode,
                )

                cleaned_stream = clean_stream(assistant_response_stream)
                full_text = st.write_stream(cleaned_stream)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_text, "ts": time.time()}
                )


            # 3) 提取记忆
            MIN_IMPORTANCE = 0.30
            store = st.session_state.memory_store

            try:
                # 触发条件
                if idle_seconds >= 5 * 60:
                    should, reason, new_count = True, "idle", 0
                    st.info(f"✅ 触发批量抽取记忆: 超过5分钟未活跃, new_count={new_count}, idle_seconds={idle_seconds:.1f}s")
                else:
                    should, new_count = should_trigger_by_rounds()
                    reason = "rounds" if should else "none"
                    if should:
                        st.info(f"✅ 触发批量抽取记忆: 新增消息数={new_count}, idle_seconds={idle_seconds:.1f}s")
                    else:
                        st.caption(f"未触发批量抽取记忆, 新增消息数={new_count}, idle_seconds={idle_seconds:.1f}s")
                if not should:
                    st.caption("未触发批量（新增消息数未达标或停滞时间不足）。")
                    st.stop()

                # 拼接对话块
                msgs = st.session_state.messages
                dialogue_block = "\n".join(
                    f"{i}. ({m.get('role','')}) {m.get('content','')}" for i, m in enumerate(msgs)
                )

                # 切分+抽取
                with st.spinner("批量处理：切分并提取（跳过最后话题）…"):
                    result = call_llm_segment_and_extract(dialogue_block=dialogue_block, run_llm=run_llm)

                # 调试概览
                try:
                    segs = (result or {}).get("segments", []) or []
                    st.caption(f"[调试] LLM segments={len(segs)}（将忽略最后一段）")
                    for i, s in enumerate(segs[:5]):
                        ms = (s or {}).get("memories", []) or []
                        st.caption(f"  seg[{i}]: end_turn={s.get('end_turn')}, memories={len(ms)}")
                except Exception as e:
                    st.warning(f"[调试] 打印 result 概览失败：{e}")

                # 仅收集“非最后段”的记忆；不上游入库，这里统一入库
                raw_items = []
                for s in segs[:-1]:
                    mems = (s or {}).get("memories", []) or []
                    raw_items.extend(mems)

                # 统一转 MemoryUnit（就地写，不建小函数）
                new_units = []
                for x in raw_items:
                    if isinstance(x, MemoryUnit):
                        mu = x
                    else:
                        mu = MemoryUnit(
                            content=(x.get("content", "") or "").strip(),
                            importance=float(x.get("importance", 0.0) or 0.0),
                        )
                    new_units.append(mu)

                # 读取库内内容用于去重（按 content）
                try:
                    existing_contents = {getattr(m, "content", None) for m in store.get_all()}
                    existing_contents.discard(None)
                except Exception:
                    existing_contents = set()

                # 重要性过滤 + 去重（库内 + 批内）
                seen_in_batch, accepted = set(), []
                for mu in new_units:
                    try:
                        c = (getattr(mu, "content", "") or "").strip()
                        if not c:
                            continue
                        imp = float(getattr(mu, "importance", 0.0) or 0.0)
                        if imp < MIN_IMPORTANCE:
                            continue
                        if c in existing_contents or c in seen_in_batch:
                            continue
                        accepted.append(mu)
                        seen_in_batch.add(c)
                    except Exception:
                        st.error("处理单条记忆时出错")
                        st.code(traceback.format_exc())

                st.caption(f"[调试] 过滤后待入库条数={len(accepted)}（阈值≥{MIN_IMPORTANCE:.2f}）")
                if accepted[:3]:
                    st.caption(f"[调试] 预览前3条：{[mu.content for mu in accepted[:3]]}")



                # ===（建议）仅当 accepted 非空时再算 embedding ===
                if accepted:
                    try:
                        contents = [mu.content for mu in accepted]
                        # 1) 批量计算
                        embeddings = embed_text(bgem3, [f"passage:{c}" for c in contents])
                        # 2) 断言条数一致（极端情况下保护）
                        if len(embeddings) != len(accepted):
                            st.warning(f"embedding 返回条数异常：expected={len(accepted)}, got={len(embeddings)}；将尝试逐条回退")
                            # 逐条回退
                            new_embs = []
                            for mu in accepted:
                                try:
                                    e = embed_text(bgem3, f"passage:{mu.content}")
                                    new_embs.append(e)
                                except Exception:
                                    new_embs.append(None)
                            embeddings = new_embs

                        # 3) 写回每条 mu.embedding
                        for mu, emb in zip(accepted, embeddings):
                            if emb is not None and hasattr(emb, "tolist"):
                                mu.embedding = emb.tolist()
                            else:
                                mu.embedding = emb  # 可能是 None 或已是 list

                    except Exception as e:
                        st.warning(f"批量计算 embedding 失败：{e}。将逐条回退。")
                        # fallback：逐条计算，尽量不丢
                        for mu in accepted:
                            try:
                                emb = embed_text(bgem3, f"passage:{mu.content}")
                                mu.embedding = emb.tolist() if hasattr(emb, "tolist") else emb
                            except Exception:
                                mu.embedding = None
                                st.caption(f"⚠️ 单条 embedding 失败，content='{mu.content[:30]}…' 将以 None 入库")

                # —— 入库（逐条 add）——
                added_count = 0
                for mu in accepted:
                    try:
                        r = store.add(mu)
                        if getattr(mu, "id", None) is None:
                            mu.id = r if isinstance(r, (str, int)) else getattr(r, "id", None)
                        existing_contents.add(getattr(mu, "content", ""))
                        added_count += 1
                    except Exception:
                        st.error(f"写入数据库失败：{getattr(mu, 'content', '')}")
                        st.code(traceback.format_exc())

                st.info(f"✅ 批量触发：reason={reason}, new_count={new_count}, 新增记忆={added_count}")

                # —— 事件归属（仅对本轮 accepted）——
                if accepted:
                    try:
                        assign_event_for_units(store, prompt, prompt_embedding, accepted)
                    except Exception as e:
                        st.warning(f"事件归属失败：{e}")


                st.info(f"✅ 批量触发：reason={reason}, new_count={new_count}, 新增记忆={added_count}")



                # 可视化（仅展示本轮 accepted）
                if accepted:
                    with st.expander(f"🧠 本轮新增 {len(accepted)} 条记忆（≥ {MIN_IMPORTANCE:.2f}）", expanded=False):
                        for mu in accepted:
                            mu_id = getattr(mu, "id", None)
                            mu_fresh = None
                            if mu_id:
                                try:
                                    mu_fresh = store.get(mu_id)
                                except Exception:
                                    mu_fresh = None
                            mu_fresh = mu_fresh or mu

                            st.info(f"- {getattr(mu_fresh, 'content', '')}\n（importance={getattr(mu_fresh, 'importance', 0.0):.2f}）")

                            ev_id = getattr(mu_fresh, "event_id", None)
                            if ev_id:
                                try:
                                    header, _ = store.get_event_context(mu_fresh.id, k_siblings=0)
                                except Exception:
                                    header = None
                                title = (header or {}).get("title") or "未命名事件"
                                status = (header or {}).get("status")
                                start_ts = (header or {}).get("start_ts")
                                updated_at = (header or {}).get("updated_at")
                                st.caption(f"事件ID：`{ev_id}`｜事件《{title}》｜状态：{status}｜时间窗：{start_ts} → {updated_at}")
                            else:
                                st.caption("（未绑定事件）")

                # 推进“水位”
                st.session_state["last_batch_msg_idx"] = len(msgs) - 1

            except Exception as e:
                st.warning(f"记忆批量处理失败：{e}")
                st.code(traceback.format_exc())



    st.session_state.handled_turn_id = st.session_state.turn_id
    st.session_state.to_generate = False

    # === 自动滚动到底部 ===
    st.markdown(
        "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
        unsafe_allow_html=True,
    )

# ================== Sidebar 实时视图（最近 20 / 分页） ==================
with all_memories_view:
    st.markdown("---")
    all_mems = memory_store.get_all()
    total = len(all_mems)
    st.write(f"**当前记忆总数: {total}**")

    # 按时间排序（若有 timestamp）
    def mem_ts(m):
        ts = getattr(m, "timestamp", None)
        return ts if isinstance(ts, datetime) else datetime.min

    all_mems_sorted = sorted(all_mems, key=mem_ts, reverse=True)

    if st.session_state.mem_view_mode == "recent":
        show_list = all_mems_sorted[:20]
        st.caption("显示最近 20 条：")
    else:
        page_size = st.session_state.mem_page_size
        total_pages = max((total + page_size - 1) // page_size, 1)
        # 限制页码
        st.session_state.mem_page = max(1, min(st.session_state.mem_page, total_pages))
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("上一页", disabled=st.session_state.mem_page <= 1):
                st.session_state.mem_page = max(1, st.session_state.mem_page - 1)
        with col2:
            st.markdown(
                f"<div style='text-align:center'>第 {st.session_state.mem_page} / {total_pages} 页</div>",
                unsafe_allow_html=True,
            )
        with col3:
            if st.button("下一页", disabled=st.session_state.mem_page >= total_pages):
                st.session_state.mem_page = min(
                    total_pages, st.session_state.mem_page + 1
                )

        start = (st.session_state.mem_page - 1) * page_size
        end = start + page_size
        show_list = all_mems_sorted[start:end]

    for mem in show_list:
        title = mem.content[:40] + ("..." if len(mem.content) > 40 else "")
        with st.expander(title):
            st.markdown(f"**ID:** `{getattr(mem, 'id', 'N/A')}`")
            st.markdown(f"**内容:** {mem.content}")
            ts = getattr(mem, "timestamp", None)
            if isinstance(ts, datetime):
                st.markdown(f"**创建时间:** {ts.strftime('%Y-%m-%d %H:%M:%S')}")
            count = getattr(mem, "retrieval_count", 0)
            st.markdown(f"**检索次数:** {count}")
            last_ts = getattr(mem, "last_retrieved_ts", None)
            if isinstance(last_ts, datetime):
                st.markdown(f"**上次检索:** {last_ts.strftime('%Y-%m-%d %H:%M:%S')}")
