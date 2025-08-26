# utils.py
from sentence_transformers import SentenceTransformer
import torch
from typing import Union, List

def load_bgem3(name: str = "BAAI/bge-m3", device: str | None = None) -> SentenceTransformer:
    """
    一次性加载到 GPU，常驻内存/显存。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(name, device=device, trust_remote_code=True)  # -> GPU
    return model

def embed_text(model: SentenceTransformer, text: Union[str, List[str]]):
    """
    GPU 上编码，默认做 L2 归一化，返回 numpy。
    """
    return model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,   # 检索更稳
        batch_size=32,               # 需要批量时可调
        show_progress_bar=False,
    )


import re

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