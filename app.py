import streamlit as st
import uuid
import datetime
import json
import time
from dataclasses import dataclass, field
from llm_clients import chat_with_memories
from typing import List, Dict, Any, Optional
from memory import (
    MemoryUnit,
    MemoryStore,
    RetrievalManager,
    ForgettingManager,
    MemorySystem,
    ReflectionManager,
    UpdateManager,
)
from utils import load_bgem3, embed_text, clean_stream


@st.cache_resource
def get_bgem3():
    return load_bgem3("BAAI/bge-m3")


bgem3 = get_bgem3()

# ================== Streamlit UI 配置 ==================
st.set_page_config(page_title="🍝yuxi's LLM长时记忆实验", layout="wide")
st.title("🍝yuxi's LLM长时记忆实验")
st.caption("先做个小垃圾")
stream_mode = True


# ================== 初始化核心对象（持久化到 session） ==================
def ensure_core_in_session():
    if "memory_store" not in st.session_state:
        st.session_state.memory_store = MemoryStore()
    if "retriever" not in st.session_state:
        st.session_state.retriever = RetrievalManager()
    if "forgetter" not in st.session_state:
        st.session_state.forgetter = ForgettingManager()
    if "reflecter" not in st.session_state:
        st.session_state.reflecter = ReflectionManager()
    if "updater" not in st.session_state:
        st.session_state.updater = UpdateManager()
    if "memory_system" not in st.session_state:
        st.session_state.memory_system = MemorySystem(
            st.session_state.memory_store,
            st.session_state.retriever,
            st.session_state.forgetter,
            st.session_state.reflecter,
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retrieval_strategy" not in st.session_state:
        st.session_state.retrieval_strategy = "bm25"
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
forgetter: ForgettingManager = st.session_state.forgetter
reflecter: ReflectionManager = st.session_state.reflecter
memory_system: MemorySystem = st.session_state.memory_system
updater: UpdateManager = st.session_state.updater

# ================== Sidebar 配置（写回 retriever） ==================
with st.sidebar:
    st.header("🛠️ 记忆系统配置")

    # 设置检索方法
    retrieval_method = st.selectbox(
        "记忆检索方法",
        ["Vector Search", "BM25 (Keyword)", "Hybrid Search"],
        help="选择从记忆库中检索信息时使用的方法。",
    )

    # 设置 Top-K 数量
    top_k = st.slider("Top-K 记忆", 1, 10, 3, help="单次检索返回最相关的记忆数量。")

    # 设置 provider 和 model 配置
    provider = st.selectbox(
        "Provider",
        ["deepseek", "ollama"],
        index=0 if st.session_state.get("provider", "deepseek") == "deepseek" else 1,
    )
    st.session_state.provider = provider

    model_name = st.text_input(
        "Model",
        value=(
            st.session_state.get("model_name", "deepseek-chat")
            if provider == "deepseek"
            else "qwen2.5:14b"  # qwen2.5:14b
        ),
        help="deepseek 如 deepseek-chat；ollama 如 llama3 / qwen2.5 等本地模型名",
    )
    st.session_state.model_name = model_name

    # 设置历史对话轮数
    recent_k = st.slider(
        "历史对话轮数 k", 0, 10, 3, help="取最近 k 轮（user↔assistant）作为历史上下文"
    )
    st.session_state.recent_k = recent_k

    # 选择 MemoryManager
    retrieval_manager = RetrievalManager()

    # 根据选择的检索方法，调用对应的检索函数
    if retrieval_method == "Vector Search":
        retriever = (
            retrieval_manager.retrieve_by_embedding
        )  # 这里可以扩展实际的 vector 搜索实现
    else:
        pass  # "Hybrid Search" 可以组合两种方法，简单示例

    # 设置检索配置
    st.session_state.top_k = top_k
    st.session_state.retriever = retriever
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
        return ts.isoformat() if isinstance(ts, datetime.datetime) else None

    # 导出记忆
    def export_memories() -> bytes:
        data = []
        for m in memory_store.get_all():
            data.append(
                {
                    "id": getattr(m, "id", None),
                    "content": m.content,
                    "timestamp": _ts_iso(getattr(m, "timestamp", None)),
                    "importance": getattr(m, "importance", 0.0),
                    "retrieval_count": getattr(m, "retrieval_count", 0),
                    "last_retrieved_ts": _ts_iso(getattr(m, "last_retrieved_ts", None)),
                }
            )
        return json.dumps({"memories": data}, ensure_ascii=False, indent=2).encode(
            "utf-8"
        )

    st.download_button(
        "📤 导出记忆（JSON）",
        data=export_memories(),
        file_name=f"memories_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
                        m.timestamp = datetime.datetime.fromisoformat(item["timestamp"])
                    except Exception:
                        pass
                if (
                    hasattr(m, "retrieval_count")
                    and item.get("retrieval_count") is not None
                ):
                    m.retrieval_count = int(item["retrieval_count"])
                if hasattr(m, "last_accessed_ts") and item.get("last_accessed_ts"):
                    try:
                        m.last_accessed_ts = datetime.datetime.fromisoformat(
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


def get_latest_messages(k: int = 1) -> List[Dict[str, str]]:
    """
    从 Streamlit 会话状态中获取最近的对话消息。

    Args:
        k (int):
            - 如果 k=1，只返回最后一条用户消息。
            - 如果 k>1，返回最近 k 轮完整的用户-助手对话。

    Returns:
        List[Dict[str, str]]: 一个包含消息字典的列表。
    """
    if not hasattr(st.session_state, "messages") or not st.session_state.messages:
        return []

    messages = st.session_state.messages
    latest_messages = []

    # 始终从最新的消息开始向前遍历
    # 使用 set 记录已找到的对话轮次，以处理 k > 1 的情况
    conversation_count = 0

    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]

        # 找到一条用户消息，即找到一轮对话
        if message["role"] == "user":
            conversation_count += 1

        # 始终添加当前消息
        latest_messages.append(message)

        # 如果达到了指定的对话轮次 k，停止遍历
        if conversation_count >= k:
            break

    # 因为是从后向前遍历，所以需要反转列表以保持时间顺序
    latest_messages.reverse()

    # 特殊处理 k=1 的情况，只返回最后一条用户消息
    if k == 1 and latest_messages:
        # 找到最后一条消息中 role 为 "user" 的那条
        for msg in reversed(latest_messages):
            if msg["role"] == "user":
                return [msg]

    return latest_messages

# ================== 主聊天逻辑 ==================
if prompt := st.chat_input("请输入"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.to_generate = True
    st.session_state.turn_id = str(uuid.uuid4())
    with st.chat_message("user", avatar="👧🏻"):
        st.markdown(prompt)

    if st.session_state.get("to_generate") and (
    st.session_state.get("turn_id") != st.session_state.get("handled_turn_id")
):
        prompt_embedding = embed_text(bgem3, prompt).tolist()

        with st.chat_message("assistant", avatar="🤡"):
            # 1) 检索记忆
            with st.spinner("正在检索相关记忆..."):
                all_memories = memory_store.get_all()
                retrieved_memories = (
                    retriever(memory_store, prompt_embedding, top_k=st.session_state.top_k)
                    or []
                )

                for mem in retrieved_memories:
                    memory_store.update_retrieval_stats(mem.id, 1)

                with st.expander("🔍 本轮检索到的记忆"):
                    if retrieved_memories:
                        for mem in retrieved_memories:
                            st.info(
                                f"**内容:** {mem.content}\n\n"
                                f"**检索次数:** {mem.retrieval_count}"
                            )
                    else:
                        st.warning("未检索到相关记忆。")

            # 2) 生成回复
            st.caption(
                f"provider={st.session_state.provider}, model={st.session_state.model_name}"
            )
            with st.spinner("正在生成回复..."):
                k = int(st.session_state.get("recent_k", 0))
                k = max(k, 0)

                history = [
                    m
                    for m in st.session_state.messages
                    if m.get("role") in ("user", "assistant")
                ]

                history_wo_current = history[:-1] if history and history[-1].get("role") == "user" else history
                recent_dialog = history_wo_current[-2 * k :] if k else []

                assistant_response_stream = chat_with_memories(
                    provider=st.session_state.provider,
                    model=st.session_state.model_name,
                    recent_dialog=recent_dialog,
                    retrieved_memories=retrieved_memories,
                    current_query=prompt,
                    stream=stream_mode,  
                )

                cleaned_stream = clean_stream(assistant_response_stream)
                full_text = st.write_stream(cleaned_stream)
                print(f"yuxi see full_text:\n{full_text}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_text}
                )

            # 3) 更新记忆
            # 3.1 组织原始对话文本（只拿本轮：用户输入 + 助手回答）
            print(f"yuxi see:\n full text:{full_text}")
            raw_for_memory = f"User: {prompt},\nAssistant: {full_text}"

            # 3.2 调用提取器（DeepSeek），返回 MemoryUnit 列表（已自带 importance 与 embedding）
            updater: UpdateManager = st.session_state.updater
            try:
                with st.spinner("正在从本轮对话中提取记忆…"):
                    new_units = updater.build_memories_from_raw(raw_for_memory)  # -> List[MemoryUnit]
            except Exception as e:
                st.warning(f"记忆提取失败：{e}")
                new_units = []

            # 3.3 重要性阈值过滤（可调），写入记忆库
            MIN_IMPORTANCE = 0.30
            added = 0
            for mu in new_units:
                try:
                    # 去重（简单基于 content）；也可以做 embedding 近邻判重
                    if any(m.content == mu.content for m in memory_store.get_all()):
                        continue
                    if getattr(mu, "importance", 0.0) >= MIN_IMPORTANCE:
                        st.session_state.memory_system.add_memory(mu)
                        added += 1
                except Exception:
                    pass

            # 3.4 可选：给出本轮新增记忆的可视化
            if added:
                with st.expander(f"🧠 本轮新增 {added} 条记忆（≥ {MIN_IMPORTANCE:.2f}）", expanded=False):
                    for mu in new_units:
                        if getattr(mu, "importance", 0.0) >= MIN_IMPORTANCE:
                            st.info(f"- {mu.content}  \n（importance={mu.importance:.2f}）")
            else:
                st.caption("本轮未新增记忆或重要性较低。")
                
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
        return ts if isinstance(ts, datetime.datetime) else datetime.datetime.min

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
            if isinstance(ts, datetime.datetime):
                st.markdown(f"**创建时间:** {ts.strftime('%Y-%m-%d %H:%M:%S')}")
            count = getattr(mem, "retrieval_count", 0)
            st.markdown(f"**检索次数:** {count}")
            last_ts = getattr(mem, "last_retrieved_ts", None)
            if isinstance(last_ts, datetime.datetime):
                st.markdown(f"**上次检索:** {last_ts.strftime('%Y-%m-%d %H:%M:%S')}")
