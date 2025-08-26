import uuid
import datetime, time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import requests, json
import json, re, time, requests
from typing import List, Dict
from utils import load_bgem3, embed_text

bgem3 = load_bgem3("BAAI/bge-m3", device="cpu")

# --- 1. 核心数据结构 ---


@dataclass
class MemoryUnit:
    """记忆单元，系统的原子数据。后续可以增补修改"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None  
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    importance: float = 0.0
    retrieval_count: int = 0
    last_accessed_ts: Optional[datetime.datetime] = None


class MemoryStore:
    """
    一个简单的内存记忆仓库，作为所有管理器的共享数据源。
    """

    def __init__(self):
        self._memories: Dict[str, MemoryUnit] = {}

    def add(self, memory: MemoryUnit):
        self._memories[memory.id] = memory

    def update(self, memory_id: str, memory: MemoryUnit):
        self._memories[memory_id] = memory

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        return self._memories.get(memory_id)

    def get_all(self) -> List[MemoryUnit]:
        return list(self._memories.values())

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

    def clear(self):
        """清空所有记忆。"""
        self._memories.clear()

    def update_retrieval_stats(self, memory_id: str, retrieval_count: int):
        if memory_id in self._memories:
            memory = self._memories[memory_id]
            memory.retrieval_count += retrieval_count
            memory.last_accessed_ts = datetime.datetime.now()


class RetrievalManager:
    """检索工具箱：包含所有检索记忆的方法。"""

    def retrieve_by_embedding(self, store: MemoryStore, query_embedding: List[float], top_k: int = 3) -> List[MemoryUnit]:
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

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


class ForgettingManager:
    """遗忘工具箱：包含所有遗忘记忆的方法。"""

    def by_importance(self, store: MemoryStore, threshold: float = 0.2) -> List[str]:
        """方法1：遗忘重要性低于阈值的记忆。"""
        return []

    def by_time_decay(self, store: MemoryStore, days: int = 7) -> List[str]:
        """方法2：遗忘长期未被访问的记忆。"""
        return []


class ReflectionManager:
    """反思工具箱：包含所有从记忆中生成新见解的方法。"""

    def generate_insight_v1(self, store: MemoryStore) -> Optional[MemoryUnit]:
        pass


SYSTEM_PROMPT_EXTRACTION = """
You are a memory extraction assistant.
The user provides a conversation transcript.
Return a JSON array of objects, each object only containing:
- "content": concise atomic fact (< 100 chars)
- "importance": float in [0,1]

Example:
[
  {"content": "Alice's birthday is 1990-05-12", "importance": 0.85},
  {"content": "Project X deadline is 2025-09-30", "importance": 0.92}
]

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

class UpdateManager:
    """
    从原始对话中提取记忆unit：raw str2memory unit
    先用最简单的：llm + 规则提取，后续需要专门设计各个属性值的获取和计算方法
    """
    def build_memories_from_raw(self, rawdata: str) -> List[MemoryUnit]:
        """调用 deepseek-chat，返回完整的 MemoryUnit 列表"""
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

        # --- 调用 + 重试（含可重试状态码） ---
        attempt, max_attempts = 0, 2
        resp = None
        while True:
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=60)
            except requests.exceptions.RequestException as e:
                if attempt < max_attempts:
                    time.sleep(2 ** attempt); attempt += 1; continue
                raise RuntimeError(f"【网络错误】调用 LLM 失败：{e}") from e

            if r.status_code in (429, 500, 502, 503, 504):
                if attempt < max_attempts:
                    time.sleep(2 ** attempt); attempt += 1; continue
                raise RuntimeError(f"【API错误】{r.status_code}，重试 {max_attempts} 次后仍失败：{r.text[:500]}")
            if r.status_code != 200:
                raise RuntimeError(f"【API错误】{r.status_code}：{r.text[:500]}")
            resp = r
            break

        # --- 解析响应，兼容 DeepSeek(OpenAI风格) / Ollama ---
        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"【解析错误】返回非 JSON：{e}｜原始：{resp.text[:500]}") from e

        content = ""
        # 1) OpenAI/DeepSeek: choices[0].message.content
        if isinstance(data, dict) and "choices" in data:
            try:
                choice0 = data["choices"][0]
                if isinstance(choice0, dict):
                    msg = choice0.get("message")
                    if isinstance(msg, dict) and "content" in msg:
                        content = msg["content"]
                    if not content and "text" in choice0:  # 极少见代理
                        content = choice0["text"]
            except Exception:
                pass

        # 2) Ollama: {"message":{"content":...}} 或 {"response": "..."}
        if not content and isinstance(data, dict):
            msg = data.get("message")
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
            if not content:
                content = data.get("response", "")

        content = _strip_code_fences(content)
        if not content:
            raise RuntimeError(f"【解析错误】LLM 返回为空或未知结构：{str(data)[:500]}")

        # --- 解析为 JSON 数组（容错：从文本中抽取顶层数组） ---
        try:
            items = json.loads(content)
            if not isinstance(items, list):
                raise ValueError("非数组")
        except Exception:
            m = re.search(r"\[\s*{[\s\S]*?}\s*\]", content)
            if not m:
                raise RuntimeError(f"【解析错误】无法从内容中提取 JSON 数组：{content[:500]}")
            items = json.loads(m.group(0))
            if not isinstance(items, list):
                raise RuntimeError(f"【解析错误】提取到的内容不是 JSON 数组：{m.group(0)[:500]}")

        # --- 校验/构造 MemoryUnit ---
        memory_units: List[MemoryUnit] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            c = build_text(it.get("content", ""))
            p = build_importance(it.get("importance", 0.0))
            # bgem3对c做嵌入
            c_embedding = embed_text(bgem3, c).tolist() if c else None
            if c:
                memory_units.append(MemoryUnit(content=c, importance=p, embedding=c_embedding))
        return memory_units


class MemorySystem:
    def __init__(
        self,
        store: MemoryStore,
        retriever: RetrievalManager,
        forgeter: ForgettingManager,
        reflecter: ReflectionManager,
    ):
        # 持有共享的数据仓库
        self.store = store

        # 持有所有功能的“工具箱”实例
        self.retriever = retriever
        self.forgetter = forgeter
        self.reflecter = reflecter
        print("工具箱式记忆系统已初始化。")

    # 提供一些便捷的顶层接口
    def add_memory(self, memory: MemoryUnit):
        """便捷方法：直接添加记忆。"""
        self.store.add(memory)

    def perform_forgetting(self, strategy: str, **kwargs):
        """
        顶层遗忘接口，根据策略名调用对应工具箱里的方法。
        """
        ids_to_forget = []
        if strategy == "importance":
            ids_to_forget = self.forgetter.by_importance(self.store, **kwargs)
        elif strategy == "time_decay":
            ids_to_forget = self.forgetter.by_time_decay(self.store, **kwargs)
        else:
            print(f"警告: 未知的遗忘策略 '{strategy}'")

        print(f"准备遗忘 {len(ids_to_forget)} 条记忆...")
        for mem_id in ids_to_forget:
            self.store.delete(mem_id)

    def print_status(self):
        """打印当前记忆库状态。"""
        print("\n--- 记忆库状态 ---")
        memories = self.store.get_all()
        print(f"总记忆数: {len(memories)}")
        for mem in sorted(memories, key=lambda m: m.timestamp):
            print(f"  - [重要性: {mem.importance}] {mem.content[:60]}")
        print("---------------------\n")
