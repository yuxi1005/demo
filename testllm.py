# test_llm_client.py
from llm_clients import chat_with_memories
from memory import MemoryUnit


def main():
    # 模拟近期两轮对话
    recent_dialog = [
        {"role": "user", "content": "你好，我想搭建一个带长期记忆的对话系统。"},
        {
            "role": "assistant",
            "content": "好的，我们可以设计记忆存储、检索和遗忘模块。",
        },
    ]

    # 模拟检索到的记忆
    retrieved_memories = [
        MemoryUnit(content="用户习惯在 Windows 环境下开发项目。"),
        MemoryUnit(content="用户正在尝试不同的记忆检索方式，比如 BM25 与向量检索。"),
        "项目目标是验证不同记忆系统对对话质量的影响。",
    ]

    # 当前 query
    current_query = "那我应该如何开始接入记忆压缩模块？"

    # 测试调用 deepseek
    try:
        reply = chat_with_memories(
            provider="deepseek",
            model="deepseek-chat",
            recent_dialog=recent_dialog,
            retrieved_memories=retrieved_memories,
            current_query=current_query,
        )
        print("=== DeepSeek 回复 ===")
        print(reply)
    except Exception as e:
        print("调用 DeepSeek 出错：", e)

    # 测试调用 ollama（需要本地有模型）
    try:
        reply = chat_with_memories(
            provider="ollama",
            model="qwen2.5:14b",  # qwen2.5:14b
            recent_dialog=recent_dialog,
            retrieved_memories=retrieved_memories,
            current_query=current_query,
        )
        print("=== Ollama 回复 ===")
        print(reply)
    except Exception as e:
        print("调用 Ollama 出错：", e)


if __name__ == "__main__":
    main()
