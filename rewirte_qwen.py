# -*- coding: utf-8 -*-
import re
from typing import Optional, Dict, Literal, TypedDict
import ollama

MODEL_NAME = "qwen3:4b"

# ---------------- Heuristics：触发 & 黑名单 ----------------
RE_SHORT = re.compile(r'^\s*[\u4e00-\u9fff\w]{0,4}\s*$')
RE_YESNO  = re.compile(r'^(好的|行|可以|嗯|是的|没错|不对|不是|不行|不要)[。…!！~\?？]*$')
RE_PRON   = re.compile(r'(这(个|些)?|那(个|些)?|它|他|她|此|如此|这事|那事)')
RE_CONT   = re.compile(r'(然后呢|接着呢|下一步|后来呢|还有吗|再说|继续说)')
RE_REF    = re.compile(r'(按你(刚才|之前)说的|照你说的|关于(这|那)一点|你提到的)')
RE_ELLQ   = re.compile(r'(怎么(实现|做|办)|优缺点|风险|代价).{0,4}\?$')
RE_PUNC   = re.compile(r'^[\.\!！\?？、，,…～~\s]+$')

def should_rewrite(user_text: str) -> bool:
    t = (user_text or "").strip()
    return any([
        RE_SHORT.match(t) is not None,
        RE_YESNO.match(t) is not None,
        bool(RE_PRON.search(t)),
        bool(RE_CONT.search(t)),
        bool(RE_REF.search(t)),
        bool(RE_ELLQ.search(t)),
        RE_PUNC.match(t) is not None,
    ])

RE_CODE = re.compile(r"```|`[^`]+`|class\s+\w+|def\s+\w+|import\s+\w+|\{.*\}|<[^>]+>")
RE_URL  = re.compile(r"https?://\S+")
RE_CAPS = re.compile(r"\b[A-Z0-9_]{3,}\b")
RE_NUMS = re.compile(r"\b\d{6,}\b")

def in_blacklist(text: str) -> bool:
    return any([
        bool(RE_CODE.search(text)),
        bool(RE_URL.search(text)),
        bool(RE_CAPS.search(text)),
        bool(RE_NUMS.search(text)),
    ])

# ---------------- Prompt（system + few-shot） ----------------
SYSTEM_PROMPT = """你是一个对话改写助手。
任务：把用户当前输入改写成一条可以单独理解的完整句子。
严格要求：
- 只补全缺失信息，不改变语义，不新增事实或细节。
- 保持原始语言：中文就中文，英文就英文，混合就混合；不要翻译。
- 如果当前输入本身已经可独立理解，则原样返回，不要改写。
- 输出只包含改写后的那一句，不要解释、不要前后缀。
"""

FEWSHOT = [
    # 中文
    ("这个方法可以减少内存占用。", "为什么？", "为什么这个方法可以减少内存占用？"),
    ("我们刚刚讨论了A方案的优点。", "那缺点呢？", "A方案的缺点是什么？"),
    # 英文
    ("We discussed option A in detail.", "what about B?", "What about option B?"),
    ("This trick speeds up the data loader significantly.", "why?", "Why does this trick speed up the data loader significantly?"),
]

def build_messages(context: str, utterance: str):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    # few-shot as user/assistant pairs
    for c,u,y in FEWSHOT:
        msgs.append({"role": "user", "content": f"Context:\n{c}\nUtterance:\n{u}\nRewrite:"})
        msgs.append({"role": "assistant", "content": y})
    # current sample
    msgs.append({"role": "user", "content": f"Context:\n{(context or '').strip()}\nUtterance:\n{(utterance or '').strip()}\nRewrite:"})
    return msgs

# ---------------- 实用工具 ----------------
STOPWORDS = set(list("的了呢吗呀哦啊嗯呃啊吧啦") + ["、","：","；"])
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())

def _last_clause(text: str, max_len: int = 36) -> str:
    """取上下文最后一句/从最后一个句号/问号/叹号处截断，兜底场景用"""
    t = (text or "").strip()
    m = re.split(r"[。！？?!]\s*", t)
    tail = m[-2] if t.endswith(tuple("。！？?!")) and len(m) >= 2 else m[-1]
    tail = tail.strip()
    return tail[-max_len:] if len(tail) > max_len else tail

def _has_new_entities(rewritten: str, context: str, utterance: str) -> bool:
    """非常简单的'新实体'检测：若出现大量在上下文+原句未出现的非停用词/长词，认为可能幻觉"""
    base = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]+", (context or "") + " " + (utterance or "")))
    toks = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]+", rewritten or "")
    novel = [t for t in toks if t not in base and (len(t) >= 2) and (t not in STOPWORDS)]
    # 阈值很保守：允许1-2个新词，超过则判可疑
    return len(novel) >= 3

# ---------------- Ollama 推理 ----------------
def _rewrite_ollama(context: str, utterance: str) -> str:
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=build_messages(context, utterance),
        options={
            "temperature": 0.0,        # 更保守
            "top_p": 0.9,
            "num_predict": 96,
            "repeat_penalty": 1.05,
            "stop": ["\n\n", "\nContext:", "\nUtterance:", "\nRewrite:"],  # 早停，防多说
        }
    )
    text = (resp.get("message", {}) or {}).get("content", "").strip()
    text = re.sub(r'^(["“])|(["”])$', "", text).strip()
    return text

# ---------------- 类型定义 ----------------
class RewriteResult(TypedDict):
    used: bool
    rewritten: Optional[str]
    reason: Literal["blacklist", "not_needed", "no_change", "rewritten", "fallback"]

# ---------------- 回退规则（硬兜底） ----------------
def _fallback_rewrite(context: str, utterance: str) -> str:
    tail = _last_clause(context)
    u = (utterance or "").strip()
    # 常见模板兜底
    if re.fullmatch(r'(为什么|为何)(？|\?)?', u):
        return f"为什么{tail}？"
    if re.fullmatch(r'(那|那么)?(缺点|优点|风险)(呢|是什么)?[？\?]?', u):
        # 简单猜测主题词
        return f"{tail}的{re.sub(r'^(那|那么)', '', u).replace('呢','').replace('？','').replace('?','')}是什么？"
    if re.fullmatch(r'(怎么|如何)做(呢)?[？\?]?', u):
        return f"如何{tail}？"
    # 通用拼接（尽量不改义）
    return f"{tail}{u if u.endswith(('？','?','。','.')) else u+'。'}"

# ---------------- 主函数 ----------------
def rewrite_if_needed(prev_assistant: str, user_text: str) -> RewriteResult:
    if in_blacklist(user_text):
        return {"used": False, "rewritten": None, "reason": "blacklist"}
    if not should_rewrite(user_text):
        return {"used": False, "rewritten": None, "reason": "not_needed"}

    rw = _rewrite_ollama(prev_assistant, user_text)
    if not rw or _norm(rw) == _norm(user_text) or _has_new_entities(rw, prev_assistant, user_text):
        # 回退：用规则模板拼
        return {"used": True, "rewritten": _fallback_rewrite(prev_assistant, user_text), "reason": "fallback"}

    return {"used": True, "rewritten": rw, "reason": "rewritten"}

# ---------------- 小测 ----------------
if __name__ == "__main__":
    tests = [
        {"ctx": "这个方法可以减少内存占用。", "utt": "为什么？"},
        {"ctx": "我们刚刚讨论了A方案的优点。", "utt": "那缺点呢？"},
        {"ctx": "We discussed option A in detail.", "utt": "what about B?"},
        {"ctx": "The API rate limit is 60 rpm.", "utt": "那怎么绕开？"},
        {"ctx": "我要做图神经网络节点分类。", "utt": "可以。"},              # 应触发改写成完整确认句
        {"ctx": "这是完全独立的问题。", "utt": "我想了解图神经网络的基本概念"},  # 不触发改写
        {"ctx": "Return a JSON object with fields id and name.", "utt": "给我一个 class User 吧"}, # 黑名单
    ]
    for ex in tests:
        r = rewrite_if_needed(ex["ctx"], ex["utt"])
        print(f"\nCTX: {ex['ctx']}\nUTT: {ex['utt']}\nUSED: {r['used']}\nREASON: {r['reason']}\nREWRITTEN: {r['rewritten']}")
