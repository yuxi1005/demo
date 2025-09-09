from sentence_transformers import SentenceTransformer
import torch
from typing import Union, List, Dict, Any, Optional, Tuple, cast
import math, re, datetime
import os
import requests
import jieba
import jieba.analyse
import nltk
from nltk.corpus import stopwords
import json
import re
import datetime as dt
from typing import List
import requests
 

_TAG_RE = re.compile(r"^[^/]{1,12}(?:/[^/]{1,12}){0,2}$")  # up to 3 levels
def normalize_tag_path(path: str) -> str:
    return path.strip().strip("/").replace("\\", "/")

def is_valid_tag_path(path: str) -> bool:
    return bool(_TAG_RE.match(path))

def match_tag_prefix(tag: str, query_path: str) -> bool:
    tag = normalize_tag_path(tag)
    qp = normalize_tag_path(query_path)
    return tag == qp or tag.startswith(qp + "/")


class StopwordTokenizer:
    CN_STOPWORDS_URL = "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt"
    CN_STOPWORDS_PATH = "stopwords_cn.txt"

    def __init__(self):
        self._download_if_not_exists(self.CN_STOPWORDS_URL, self.CN_STOPWORDS_PATH)
        self.cn_stopwords = self._load_stopwords(self.CN_STOPWORDS_PATH)       
        try:
            self.en_stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            self.en_stopwords = set(stopwords.words("english"))

        jieba.analyse.set_stop_words(self.CN_STOPWORDS_PATH)

    def _download_if_not_exists(self, url: str, path: str, timeout: int = 15):
        if not os.path.exists(path):
            print(f"正在下载 {url} -> {path}")
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            with open(path, "w", encoding="utf-8") as f:
                f.write(resp.text)

    def _load_stopwords(self, path: str) -> set:
        with open(path, "r", encoding="utf-8") as f:
            return set(w.strip() for w in f if w.strip())

    def is_stopword(self, token: str) -> bool:
        if not token:
            return True
        t = token.strip()
        if not t:
            return True
        if t.lower() in self.en_stopwords:
            return True
        if t in self.cn_stopwords:
            return True
        return False

    def tokenize(self, text: str) -> list[str]:
        words = jieba.cut(text or "")
        return [
            w for w in words
            if re.search(r"[A-Za-z0-9\u4e00-\u9fa5]", w)
            and not self.is_stopword(w)
        ]


_TOKENIZER = StopwordTokenizer()


def _cos(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def load_bgem3(
    name: str = "BAAI/bge-m3", device: str | None = None
) -> SentenceTransformer:
    """
    一次性加载到 GPU，常驻内存/显存。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(name, device=device, trust_remote_code=True)  # -> GPU/CPU
    return model

bgem3 = load_bgem3("BAAI/bge-m3")

def embed_text(model: SentenceTransformer, text: Union[str, List[str]], ):
    """
    GPU 上编码，默认做 L2 归一化，返回 numpy。
    """
    return model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 检索更稳
        batch_size=32,              # 需要批量时可调
        show_progress_bar=False,
    )




def clean_stream(gen):
    for chunk in gen:
        # 统一成 str
        if isinstance(chunk, (bytes, bytearray)):
            chunk = chunk.decode("utf-8", "ignore")
        elif not isinstance(chunk, str):
            chunk = str(chunk)
        # 清洗：去掉罕见控制字符，但保留正常内容
        chunk = chunk.replace("\r", "")
        yield chunk

def _as_float_list(x) -> List[float]:
    """把各种输入转成 float 列表"""
    if x is None:
        return []
    # numpy 数组
    if hasattr(x, "tolist"):
        try:
            x = x.tolist()
        except Exception:
            pass
    if isinstance(x, str):
        # 处理 "[0.1, 0.2, 0.3]" 这种情况
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        return [float(v) for v in s.split(",") if v.strip()]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]

def _l2_normalize(vec: List[float]) -> List[float]:
    """把向量归一化到 L2=1，避免 EMA 后模长变化"""
    vec = _as_float_list(vec)
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]

def _short(s: str, max_len: int = 12) -> str:
    s = re.sub(r"[\s,:;，。！？!?.]+", "", s).strip()
    return s[:max_len] if len(s) > max_len else s

def _keywords_from_text(text: str, top_k: int = 6) -> List[str]:
    """
    使用已定义的 StopwordTokenizer 去停用词后，再做唯一化与裁剪。
    """
    tokens = _TOKENIZER.tokenize(text or "") # 分词 + 去停用词
    tokens = [t for t in tokens if re.search(r"[A-Za-z0-9\u4e00-\u9fa5]", t)]
    # 去重，保序
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:top_k]

def suggest_event_title_simple(
    turn_text: str,
    memory_units: List[str] | None = None,
    etype: str | None = None
) -> str:
    """
    根据对话内容和记忆单元，生成事件标题。
    逻辑：基于去停用词后的 token 做一个简易的排序打分。
    """
    kws = _keywords_from_text(turn_text)
    for mu in (memory_units or []):
        kws += _keywords_from_text(mu or "")
    # 打分：越靠前越高，长度 2~12 更好
    scored = []
    for i, k in enumerate(kws):
        score = max(0, 8 - i) + (2 if 2 <= len(k) <= 12 else 0)
        scored.append((score, k))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    head = scored[0][1] if scored else None

    prefix = ""
    if etype in ("study", "paper"):
        prefix = "论文-"
    elif etype in ("bugfix", "engineering"):
        prefix = "修复-"
    elif etype in ("shopping", "purchase"):
        prefix = "采购-"
    elif etype in ("meeting", "sync"):
        prefix = "会议-"

    if head:
        return _short(prefix + head, 12)
    # 兜底
    ts = datetime.datetime.now().strftime("%m-%d %H:%M")
    return f"未命名事件（{ts}）"


def call_llm_segment_and_extract(dialogue_block: str, run_llm) -> Dict:
    """
    调用 LLM 做对话切分与记忆抽取，返回解析后的 JSON 结构。
    dialogue_block: 多轮对话文本块，格式：{i}. ({role}) {content}\n
    run_llm: 调用 LLM 的函数，参数为 prompt，返回字符串结果
    """
    with open("../extract_prompt.txt", "r", encoding="utf-8") as f:
        extract_prompt = f.read()
    prompt = f"""
            {extract_prompt}
            Dialogue begins below:
            {dialogue_block}
            """.strip()

    raw = run_llm(prompt).strip()
    raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw)
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r'\{\s*"segments"\s*:\s*\[.*?\]\s*\}', raw, flags=re.S)
        data = json.loads(m.group(0)) if m else {"segments": []}

    # —— 解析与清洗（确保只有 content/importance，长度与取值合法）——
    if not isinstance(data, dict):
        data = {"segments": []}
    segs = data.get("segments", [])
    if not isinstance(segs, list):
        segs = []
    data["segments"] = segs

    all_imps = []
    for i, seg in enumerate(segs):
        if not isinstance(seg, dict):
            segs[i] = {"start_turn": 0, "end_turn": 0, "memories": []}
            continue
        mems = seg.get("memories", [])
        if not isinstance(mems, list):
            mems = []
        cleaned_mems = []
        for m in mems:
            if not isinstance(m, dict):
                continue
            c = str(m.get("content", "") or "").strip()
            if not c:
                continue
            # 限长：≤100 字符
            if len(c) > 100:
                c = c[:100]
            # importance 归一化到 [0,1]
            try:
                p = float(m.get("importance", 0.0))
            except Exception:
                p = 0.0
            if p < 0.0: p = 0.0
            if p > 1.0: p = 1.0
            cleaned_mems.append({"content": c, "importance": p})
            all_imps.append(p)
        seg["memories"] = cleaned_mems

    # —— 批内重标定（仅当“全一样”或“全都很高(>=0.95)”时触发）——
    if all_imps:
        uniq = set(round(p, 3) for p in all_imps)
        if len(uniq) == 1 or all(p >= 0.95 for p in all_imps):
            lo, hi = 0.2, 0.9
            total = sum(len(seg.get("memories", [])) for seg in segs)
            if total == 1:
                for seg in segs:
                    for m in seg.get("memories", []):
                        m["importance"] = 0.5
            elif total > 1:
                idx = 0
                for seg in segs:
                    for m in seg.get("memories", []):
                        new_p = lo + (hi - lo) * (idx / (total - 1))
                        m["importance"] = float(round(new_p, 3))
                        idx += 1

    return data


import requests
import json
import re
import time


def run_llm(prompt_text: str, model: str = "deepseek-chat", api_key_path: str = "../ds_key.txt") -> str:
    """
    调用 deepseek/moonshot 接口，返回 LLM 生成的字符串。
    prompt_text: 要求 LLM 严格返回 JSON
    """
    # 读取 API Key
    with open(api_key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.0,  # 为了更确定的 JSON
    }

    # 简单重试机制
    attempt, max_attempts = 0, 2
    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
                attempt += 1
                continue
            raise RuntimeError(f"调用 LLM 失败：{e}") from e

        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
                attempt += 1
                continue
            raise RuntimeError(f"LLM API 错误 {resp.status_code}：{resp.text[:500]}")
        resp.raise_for_status()

        data = resp.json()
        break

    # ---- 提取模型文本 ----
    content = ""
    if "choices" in data and isinstance(data["choices"], list):
        choice0 = data["choices"][0]
        if isinstance(choice0, dict):
            msg = choice0.get("message")
            if isinstance(msg, dict):
                content = msg.get("content", "")
            if not content and "text" in choice0:
                content = choice0["text"]

    if not content:
        # fallback
        content = data.get("response", "") or ""

    # 去掉可能的 ```json 围栏
    content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", (content or "").strip())
    return content


def _ensure_float_ts(ts: Any) -> Optional[float]:
    """
    把各种可能的时间戳形态转成 epoch 秒（float）：
    - float/int（支持毫秒） -> float 秒
    - dict（含 'ts'）       -> 递归取其 ts
    - datetime              -> timestamp()
    - 数字字符串            -> float（支持毫秒）
    - 其他 / 失败           -> None
    """
    if ts is None:
        return None
    # dict: 可能拿错了整条消息
    if isinstance(ts, dict):
        return _ensure_float_ts(ts.get("ts"))
    # datetime
    if isinstance(ts, dt.datetime):
        return ts.timestamp()
    # 数字/字符串
    try:
        x = float(ts)
        return x / 1000.0 if x > 1e12 else x
    except Exception:
        # 再尝试把字符串转成 datetime
        try:
            return dt.datetime.fromisoformat(str(ts)).timestamp()
        except Exception:
            return None