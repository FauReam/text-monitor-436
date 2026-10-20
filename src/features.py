"""特征层：TF-IDF（本地简单特征）+ 冻结的 sentence-transformers（外部预训练特征）。

合规要点：embedding 模型只前向、不训练；TF-IDF 只在训练子集上 fit（防泄漏）。
所有 embedding 结果按 (model, text) 缓存到 data/cache，DGX 上只需编码一次。
"""
import hashlib
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import CACHE

FEATURES = [
    "tfidf-word",
    "tfidf-char",
    "st-all-MiniLM-L6-v2",
    "st-bge-base-en-v1.5",
    "st-bge-large-en-v1.5",
]

ST_MODELS = {
    "st-all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "st-bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "st-bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
}


class TfidfFeaturizer:
    def __init__(self, kind: str):
        if kind == "tfidf-word":
            self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        elif kind == "tfidf-char":
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                       min_df=2, sublinear_tf=True)
        else:
            raise ValueError(kind)

    def fit_transform(self, texts):
        return self.vec.fit_transform(texts)

    def transform(self, texts):
        return self.vec.transform(texts)


_st_cache = {}


def _cache_paths(key: str):
    """新文件写入两级分片路径（CACHE/ab/abcdef....npy）；旧的扁平路径
    （CACHE/abcdef....npy，9M+ 文件把 ext4 目录索引写满，新建报 ENOSPC）
    只读兼容。"""
    return (os.path.join(CACHE, key[:2], key + ".npy"),
            os.path.join(CACHE, key + ".npy"))


def _encode_st(model_name: str, texts, batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    if model_name not in _st_cache:
        m = SentenceTransformer(ST_MODELS[model_name])
        if os.environ.get("ST_FP16") == "1":
            # GB10 fp32 CUDA 核很弱，fp16 快 ~4x；与 fp32 的 embedding 余弦
            # 相似度 >0.9997（实测 2000 条真实文本），对 AUROC 影响可忽略。
            m = m.half()
        _st_cache[model_name] = m
    model = _st_cache[model_name]
    out = np.full((len(texts), model.get_sentence_embedding_dimension()),
                  np.nan, dtype=np.float32)
    todo = []
    for i, t in enumerate(texts):
        key = hashlib.md5((model_name + t).encode("utf-8")).hexdigest()
        sharded, legacy = _cache_paths(key)
        if os.path.exists(sharded):
            out[i] = np.load(sharded)
        elif os.path.exists(legacy):
            out[i] = np.load(legacy)
        else:
            todo.append((i, t, sharded))
    for lo in range(0, len(todo), batch_size):
        chunk = todo[lo:lo + batch_size]
        emb = model.encode([c[1] for c in chunk], show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
        for (i, _, path), e in zip(chunk, emb):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, e.astype(np.float32))
            out[i] = e
    return out


class Featurizer:
    """统一接口：featurize(name)(train_texts) 后 transform(test_texts)。"""

    def __init__(self, name: str):
        self.name = name
        self.kind = "tfidf" if name.startswith("tfidf") else "st"
        if self.kind == "tfidf":
            self.tfidf = TfidfFeaturizer(name)

    def fit_transform(self, texts):
        if self.kind == "tfidf":
            return self.tfidf.fit_transform(texts)
        return _encode_st(self.name, list(texts))

    def transform(self, texts):
        if self.kind == "tfidf":
            return self.tfidf.transform(texts)
        return _encode_st(self.name, list(texts))
