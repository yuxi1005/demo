import streamlit as st
import uuid
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- 1. 记忆单元定义 (Memory Unit Definition) ---
# 这是我们记忆系统的原子单位
@dataclass
class MemoryUnit:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    embedding: List[float] = field(default_factory=list) # 模拟向量
    retrieval_count: int = 0
    last_retrieved_ts: Optional[datetime.datetime] = None
    
    def __str__(self):
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] {self.content}"

# --- 2. 核心模块的模拟实现 (Mock Implementations of Core Modules) ---

# 2.1 记忆存储模块 (Memory Store)
# 在这个原型中，我们使用 Streamlit 的 session_state 来模拟一个内存数据库。
# 每次刷新页面，记忆会清空。
class MemoryStore:
    def __init__(self):
        if 'memory_store' not in st.session_state:
            st.session_state.memory_store = {}

    def add_memory(self, memory_unit: MemoryUnit):
        """向记忆库中添加一条新记忆"""
        st.session_state.memory_store[memory_unit.id] = memory_unit

    def get_memory(self, memory_id: str) -> MemoryUnit:
        """根据ID获取一条记忆"""
        return st.session_state.memory_store.get(memory_id)

    def get_all_memories(self) -> List[MemoryUnit]:
        """获取所有记忆，并按时间倒序排列"""
        memories = list(st.session_state.memory_store.values())
        return sorted(memories, key=lambda m: m.timestamp, reverse=True)

    def update_memory_retrieval_stats(self, memory_id: str):
        """更新记忆的检索次数和时间"""
        if memory_id in st.session_state.memory_store:
            st.session_state.memory_store[memory_id].retrieval_count += 1
            st.session_state.memory_store[memory_id].last_retrieved_ts = datetime.datetime.now()

# 2.2 记忆检索模块 (Memory Retrieval)
# 模拟不同的检索策略
class MemoryRetriever:
    def retrieve(self, query: str, memories: List[MemoryUnit], strategy: str, top_k: int = 3) -> List[MemoryUnit]:
        """根据查询和策略检索记忆"""
        if not memories:
            return []

        if strategy == "BM25 (Keyword)":
            # 简单的关键词匹配模拟BM25
            query_words = set(query.lower().split())
            scored_memories = []
            for mem in memories:
                mem_words = set(mem.content.lower().split())
                score = len(query_words.intersection(mem_words))
                if score > 0:
                    scored_memories.append((mem, score))
            # 按分数排序
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            return [mem for mem, score in scored_memories[:top_k]]

        elif strategy == "Vector Search":
            # 模拟向量搜索：随机返回几个，假装它们是语义最相关的
            import random
            random.shuffle(memories)
            return memories[:top_k]
            
        elif strategy == "Hybrid Search":
            # 模拟混合搜索：结合两种策略的结果
            bm25_results = self.retrieve(query, memories, "BM25 (Keyword)", top_k)
            vector_results = self.retrieve(query, memories, "Vector Search", top_k)
            # 合并并去重
            combined = {mem.id: mem for mem in bm25_results + vector_results}.values()
            return list(combined)[:top_k]

        return []

# 2.3 记忆更新模块 (Memory Update)
class MemoryUpdater:
    def generate_memories_from_conversation(self, user_input: str, llm_response: str) -> List[MemoryUnit]:
        """
        模拟LLM从对话中提取关键信息作为记忆。
        在真实应用中，这里会调用一个LLM来做这件事。
        """
        # 简化处理：将用户的关键陈述句作为记忆
        # 例如，如果用户说 "我的猫叫Tom"，我们就记下 "用户的猫叫Tom"
        memories = []
        if "我叫" in user_input or "我是" in user_input:
            name = user_input.split("我叫")[-1].split("我是")[-1].strip("。， ")
            memories.append(MemoryUnit(content=f"用户的名字是{name}"))
        
        if "我的猫叫" in user_input:
            cat_name = user_input.split("我的猫叫")[-1].strip("。， ")
            memories.append(MemoryUnit(content=f"用户的猫叫{cat_name}"))

        # 也可以把对话本身作为一个摘要记忆
        summary = f"用户提到了'{user_input[:30]}...', 模型的回复是'{llm_response[:30]}...'"
        memories.append(MemoryUnit(content=summary))
        
        return memories

# 2.4 LLM核心模块 (LLM Core)
class LLMCore:
    def get_response(self, prompt: str, context_memories: List[MemoryUnit]) -> str:
        """
        模拟LLM生成回复。
        它会明确指出自己是基于哪些记忆来回答的，以便我们观察效果。
        """
        if not context_memories:
            return f"我没有任何关于您提到的内容的记忆。基于通用知识来回答：对于'{prompt}'，我建议..."

        memory_str = "\n".join([f"- {mem.content}" for mem in context_memories])
        response_prompt = f"""
        [System Instruction]
        You are a helpful assistant with a long-term memory.
        Based on the following retrieved memories, answer the user's question.
        
        [Retrieved Memories]
        {memory_str}
        
        [User's Question]
        {prompt}
        
        [Your Answer]
        """
        # 模拟的回复
        return f"根据我记得的信息：\n{memory_str}\n\n对于您的问题'{prompt}'，我的回答是：...（这里是基于记忆生成的回答）..."


# --- 3. Streamlit UI 构建 ---

st.set_page_config(page_title="🤡LLM长时记忆实验平台", layout="wide")

st.title("🤡LLM 长时记忆实验平台")
st.caption("一个用于实验不同记忆机制的模块化对话系统原型")

# 初始化核心组件
memory_store = MemoryStore()
retriever = MemoryRetriever()
updater = MemoryUpdater()
llm = LLMCore()

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ 记忆系统配置")
    
    retrieval_strategy = st.selectbox(
        "记忆检索策略",
        ["BM25 (Keyword)", "Vector Search", "Hybrid Search"],
        help="选择从记忆库中检索信息时使用的方法。"
    )
    
    top_k = st.slider("Top-K 记忆", 1, 10, 3, help="单次检索返回最相关的记忆数量。")

    st.header("📚 记忆库实时视图")
    all_memories_view = st.container() # 创建一个容器以便后续更新


# --- 主聊天界面 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入"):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 核心工作流 ---
    with st.chat_message("assistant"):
        # 1. 检索记忆
        with st.spinner("正在检索相关记忆..."):
            all_memories = memory_store.get_all_memories()
            retrieved_memories = retriever.retrieve(prompt, all_memories, retrieval_strategy, top_k)
            
            # 更新被检索到的记忆的统计数据
            for mem in retrieved_memories:
                memory_store.update_memory_retrieval_stats(mem.id)

            # 在UI上展示检索到的记忆，方便调试
            with st.expander("🔍 本轮检索到的记忆"):
                if retrieved_memories:
                    for mem in retrieved_memories:
                        st.info(f"**内容:** {mem.content}\n\n**检索次数:** {mem.retrieval_count}")
                else:
                    st.warning("未检索到相关记忆。")

        # 2. 生成回复
        with st.spinner("正在生成回复..."):
            assistant_response = llm.get_response(prompt, retrieved_memories)
            st.markdown(assistant_response)

        # 3. 更新记忆
        with st.spinner("正在更新记忆库..."):
            new_memories = updater.generate_memories_from_conversation(prompt, assistant_response)
            for mem in new_memories:
                memory_store.add_memory(mem)
            
            if new_memories:
                with st.expander("💡 本轮新增的记忆"):
                    for mem in new_memories:
                        st.success(f"**内容:** {mem.content}")


    # 将助手的回复也加入消息历史
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# --- 实时更新Sidebar的记忆库视图 ---
with all_memories_view:
    st.markdown("---")
    memories_in_store = memory_store.get_all_memories()
    st.write(f"**当前记忆总数: {len(memories_in_store)}**")
    
    for mem in memories_in_store:
        with st.expander(f"{mem.content[:40]}..."):
            st.markdown(f"**ID:** `{mem.id}`")
            st.markdown(f"**内容:** {mem.content}")
            st.markdown(f"**创建时间:** {mem.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**检索次数:** {mem.retrieval_count}")
            if mem.last_retrieved_ts:
                st.markdown(f"**上次检索:** {mem.last_retrieved_ts.strftime('%Y-%m-%d %H:%M:%S')}")

