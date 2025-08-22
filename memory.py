import uuid
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- 1. 核心数据结构 ---

@dataclass
class MemoryUnit:
    """记忆单元，系统的原子数据。"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    importance: float = 0.0
    access_count: int = 0
    last_accessed_ts: Optional[datetime.datetime] = None

class MemoryStore:
    """
    一个简单的内存记忆仓库，作为所有管理器的共享数据源。
    """
    def __init__(self):
        self._memories: Dict[str, MemoryUnit] = {}

    def add(self, memory: MemoryUnit):
        self._memories[memory.id] = memory

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        return self._memories.get(memory_id)

    def get_all(self) -> List[MemoryUnit]:
        return list(self._memories.values())

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

# --- 2. 功能管理器 (Functional Managers as Toolboxes) ---

class SearchManager:
    """检索工具箱：包含所有检索记忆的方法。"""
    def search_by_keyword(self, store: MemoryStore, query: str, top_k: int = 3) -> List[MemoryUnit]:
        """方法1：基于关键词检索。"""
        print(f"Executing search_by_keyword for '{query}'...")
        found = [mem for mem in store.get_all() if query.lower() in mem.content.lower()]
        found.sort(key=lambda m: m.importance, reverse=True)
        return found[:top_k]
    
    def search_by_recency(self, store: MemoryStore, top_k: int = 3) -> List[MemoryUnit]:
        """方法2：基于新近度检索。"""
        print(f"Executing search_by_recency...")
        all_mems = store.get_all()
        all_mems.sort(key=lambda m: m.timestamp, reverse=True)
        return all_mems[:top_k]

class ForgettingManager:
    """遗忘工具箱：包含所有遗忘记忆的方法。"""
    def by_importance(self, store: MemoryStore, threshold: float = 0.2) -> List[str]:
        """方法1：遗忘重要性低于阈值的记忆。"""
        print(f"Executing forget.by_importance (threshold: {threshold})...")
        return [mem.id for mem in store.get_all() if mem.importance < threshold]

    def by_time_decay(self, store: MemoryStore, days: int = 7) -> List[str]:
        """方法2：遗忘长期未被访问的记忆。"""
        print(f"Executing forget.by_time_decay (days: {days})...")
        now = datetime.datetime.now()
        return [mem.id for mem in store.get_all() if mem.access_count == 0 and (now - mem.timestamp).days > days]

class ReflectionManager:
    """反思工具箱：包含所有从记忆中生成新见解的方法。"""
    def generate_insight_v1(self, store: MemoryStore) -> Optional[MemoryUnit]:
        """方法1：通过分析关键词生成洞察（模拟）。"""
        print("Executing reflect.generate_insight_v1...")
        if not store.get_all(): return None
        
        insight = MemoryUnit(
            content="反思洞察V1: 用户的核心关注点似乎是项目进展和个人安排。",
            importance=0.95
        )
        return insight

# --- 3. 顶层协调器 (The Top-Level Orchestrator) ---

class MemorySystem:
    def __init__(self):
        # 持有共享的数据仓库
        self.store = MemoryStore()
        
        # 持有所有功能的“工具箱”实例
        self.search = SearchManager()
        self.forget = ForgettingManager()
        self.reflect = ReflectionManager()
        
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
            ids_to_forget = self.forget.by_importance(self.store, **kwargs)
        elif strategy == "time_decay":
            ids_to_forget = self.forget.by_time_decay(self.store, **kwargs)
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


# --- 4. 示例用法 ---
if __name__ == "__main__":
    # 1. 初始化系统
    memory_system = MemorySystem()
    
    # 2. 添加记忆
    memory_system.add_memory(MemoryUnit(content="项目Phoenix下周一到期。", importance=0.9))
    memory_system.add_memory(MemoryUnit(content="今天天气不错。", importance=0.1))
    memory_system.add_memory(MemoryUnit(content="记得买牛奶。", importance=0.25))
    
    memory_system.print_status()

    # 3. 进行实验：调用特定工具箱里的特定方法
    
    # --- 实验检索 ---
    # 你想修改关键词检索算法？只需要去修改 SearchManager.search_by_keyword 方法。
    print("--- 测试检索 ---")
    results = memory_system.search.search_by_keyword(memory_system.store, query="项目")
    print(f"关键词检索结果: {[r.content for r in results]}")
    
    recent_results = memory_system.search.search_by_recency(memory_system.store, top_k=2)
    print(f"新近度检索结果: {[r.content for r in recent_results]}")


    # --- 实验遗忘 ---
    # 你想对比不同的遗忘策略？直接调用顶层接口并切换策略名即可。
    print("\n--- 测试遗忘 ---")
    memory_system.perform_forgetting(strategy="importance", threshold=0.2)
    print("执行'按重要性遗忘'后：")
    memory_system.print_status()
    
    # 恢复一条记忆用于下一个测试
    memory_system.add_memory(MemoryUnit(content="今天天气不错。", importance=0.1))
    
    # 调用另一种遗忘策略
    memory_system.perform_forgetting(strategy="time_decay", days=0)
    print("执行'按时间遗忘'后：")
    memory_system.print_status()


    # --- 实验反思 ---
    print("\n--- 测试反思 ---")
    new_insight = memory_system.reflect.generate_insight_v1(memory_system.store)
    if new_insight:
        memory_system.add_memory(new_insight)
    memory_system.print_status()
