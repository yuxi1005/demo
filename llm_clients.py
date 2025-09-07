import os, json, time, requests, re
from typing import List, Dict, Optional, Any, Sequence, Tuple, Union
from memory import MemoryUnit

# ========== 类型别名 ==========
RoleMsg = Dict[str, str]
DialogLike = Union[
    Sequence[RoleMsg],  # [{"role":"user","content":"..."}, ...]
    Sequence[Tuple[str, str]],  # [("user","你好"), ("assistant","在的"), ...]
]


# ========== 工具函数 ==========
def _normalize_dialog(dialog: DialogLike) -> List[RoleMsg]:
    """将多种'近期对话'表示统一转换为 OpenAI/DeepSeek/Ollama 的消息格式。"""
    if not dialog:
        return []
    first = dialog[0]
    if isinstance(first, dict) and "role" in first and "content" in first:
        return [{"role": m["role"], "content": str(m["content"])} for m in dialog]  # type: ignore
    elif isinstance(first, tuple) and len(first) == 2:
        return [{"role": r, "content": str(c)} for (r, c) in dialog]  # type: ignore
    else:
        raise ValueError(
            "Unsupported dialog format: expected list[dict] or list[tuple(role, content)]."
        )


def _safe_text(resp) -> str:
    try:
        return str(resp.json())
    except Exception:
        return resp.text


# ========== 公共入口（对外唯一函数） ==========
def chat_with_memories(
    provider: str,
    model: Optional[str],
    recent_dialog: DialogLike,
    retrieved_memories: Optional[Sequence[Union[str, MemoryUnit]]],
    current_query: str,
    *,
    timeout: int = 60,
    max_retries: int = 4,
    base_url: Optional[str] = None,
    stream: bool = False,
):
    """
    一键对话入口：
    - provider: "deepseek" | "ollama"
    - model: deepseek 如 "deepseek-chat"；ollama 为本地模型名，如 "qwen2.5:14b"
    - recent_dialog: 近期 k 轮对话
    - retrieved_memories: 检索到的记忆（字符串或 MemoryUnit；自动读取 .content）
    - current_query: 当前用户 query（必要时会补到最后一条 user）
    - stream: True 则返回一个生成器；False 返回完整字符串
    """
    history = _normalize_dialog(recent_dialog)

    client = LLMClient(
        provider=provider,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        base_url=base_url,
    )
    return client.chat(
        current_query=current_query,
        messages=history,
        memories=retrieved_memories,
        stream=stream,
    )


# ========== 统一客户端 ==========
class LLMClient:
    """
    统一对话客户端：
    - chat(messages, memories=None, stream=False)
    - 如传入 memories，会将“检索记忆”注入到最前的 system prompt
    - 带指数退避重试与更友好的错误信息
    """

    def __init__(
        self,
        provider: str = "deepseek",
        model: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 4,
        base_url: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.timeout = timeout
        self.max_retries = max_retries

        if self.provider == "deepseek":
            self.model = model or "deepseek-chat"
            self.base_url = (
                base_url or "https://api.deepseek.com/v1"
            )  # ✅ 正确的 DeepSeek 基址
            with open("../API-KEY.txt", "r") as file:
                self.api_key = file.read().strip()
            if not self.api_key:
                raise RuntimeError("未找到 DeepSeek API Key：请创建 ./API-KEY.txt")
        elif self.provider == "ollama":
            self.model = model or "qwen2.5:14b"
            self.base_url = base_url or "http://localhost:11434"
            self.api_key = None
        else:
            raise ValueError("Unsupported provider. Use 'deepseek' or 'ollama'.")

    # ---------- 对外方法 ----------
    def chat(
        self,
        current_query: str,
        messages: List[Dict[str, str]],
        memories: Optional[Sequence[Any]] = None,
        stream: bool = False,
    ):
        prepared = self._prepare_messages_with_memories(messages, memories, current_query)
        if self.provider == "deepseek":
            return self._chat_deepseek(prepared)
        elif self.provider == "ollama":
            if stream:
                return self._chat_ollama_stream(prepared)
            else:
                return self._chat_ollama(prepared)
        else:
            raise ValueError("Unsupported provider")

    # ---------- 私有：提示词拼装 ----------
    def _prepare_messages_with_memories(self, messages, memories, current_query):
        # --- 记忆文本 ---
        mem_texts = []
        if memories:
            for m in memories:
                if not m:
                    continue
                if isinstance(m, str):
                    mem_texts.append(m.strip())
                else:  # MemoryUnit
                    mem_texts.append((getattr(m, "content", "") or "").strip())
        memory_block = "\n".join(f"- {t}" for t in mem_texts if t) or "（无）"

        # --- 历史对话（保留 role 结构 + 也拼到 system 里便于小模型理解）---
        lines = []
        for msg in messages or []:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{role.capitalize()}: {content}")
        history_block = "\n".join(lines) or "（无）"

        # --- 真正格式化 system 提示 ---
        system_prompt = (
            f"You are a helpful assistant with long-term memory.\n"
            f"Conversation (last k turns):\n{history_block}\n\n"
            f"Relevant Memories:\n{memory_block}\n\n"
            "Policy:\n"
            "- Treat Conversation as primary; use memories only when helpful.\n"
            "- No chain-of-thought; do NOT output <think>.\n"
            "- Match user's language; be concise.\n"
            "Now answer the latest user query."
        )

        # --- 输出的消息序列：system +（可选）历史 + 本轮 user ---
        out = [{"role": "system", "content": system_prompt}]
        # 如果希望把历史也当作真正消息发给模型，可保留：
        out.extend(messages or [])
        # 确保有当前这轮用户问题
        if current_query:
            out.append({"role": "user", "content": str(current_query)})
        return out

    # ---------- 私有：DeepSeek ----------
    def _chat_deepseek(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages}

        attempt = 0
        while True:
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    attempt += 1
                    continue
                raise RuntimeError(f"【网络错误】调用 DeepSeek 失败：{e}") from e

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < self.max_retries:
                    ra = resp.headers.get("Retry-After")
                    (
                        time.sleep(int(ra))
                        if ra and str(ra).isdigit()
                        else time.sleep(2**attempt)
                    )
                    attempt += 1
                    continue
                raise RuntimeError(
                    f"【API错误】{resp.status_code}，重试 {self.max_retries} 次后仍失败：{_safe_text(resp)}"
                )

            if resp.status_code != 200:
                raise RuntimeError(f"【API错误】{resp.status_code}：{_safe_text(resp)}")

            try:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                raise RuntimeError(
                    f"【解析错误】DeepSeek 返回无法解析：{e}｜原始：{resp.text}"
                ) from e

    # ---------- 私有：通用小工具 ----------
    def _strip_code_fences(self, text: str) -> str:
        """去掉 ``` / ```json 围栏，保留内部纯文本。"""
        if not isinstance(text, str):
            return str(text)
        s = text.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if len(lines) >= 2:
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                s = "\n".join(lines).strip()
        return s

    # ---------- 私有：Ollama（非流式） ----------
    def _chat_ollama(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/api/chat"  # ✅ 使用 self.base_url
        headers = {"Content-Type": "application/json"}
        options = {"gpu_layers": 999, "num_predict": 256, "num_ctx": 2048}
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        attempt = 0
        while True:
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    attempt += 1
                    continue
                raise RuntimeError(f"【网络错误】调用 Ollama 失败：{e}") from e

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    attempt += 1
                    continue
                raise RuntimeError(
                    f"【API错误】{resp.status_code}，重试 {self.max_retries} 次后仍失败：{_safe_text(resp)}"
                )

            if resp.status_code != 200:
                raise RuntimeError(f"【API错误】{resp.status_code}：{_safe_text(resp)}")

            try:
                data = resp.json()
            except Exception as e:
                raise RuntimeError(
                    f"【解析错误】返回非 JSON：{e}｜原始：{resp.text[:500]}"
                ) from e

            content = ""
            if isinstance(data, dict):
                msg = data.get("message") or {}
                if isinstance(msg, dict):
                    content = msg.get("content") or ""
                if not content:
                    content = data.get("response") or ""
            if not isinstance(content, str):
                content = str(content or "")
            return self._strip_code_fences(content).strip()

    # ---------- 私有：Ollama（流式） ----------
    def _chat_ollama_stream(self, messages):
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}

        options = {"num_predict": 256, "num_ctx": 2048}
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": options,
        }

        with requests.post(url, headers=headers, json=payload, stream=True) as r:
            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue

                data = json.loads(line.decode("utf-8"))
                if data.get("done") is True:
                    return

                chunk = data.get("response") or data.get("message", {}).get(
                    "content", ""
                )
                if chunk:
                    yield chunk
