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


def _encode_st(model_name: str, texts, batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    if model_name not in _st_cache:
        _st_cache[model_name] = SentenceTransformer(ST_MODELS[model_name])
    model = _st_cache[model_name]
    out = np.full((len(texts), model.get_sentence_embedding_dimension()),
                  np.nan, dtype=np.float32)
    todo = []
    for i, t in enumerate(texts):
        key = hashlib.md5((model_name + t).encode("utf-8")).hexdigest()
        path = os.path.join(CACHE, key + ".npy")
        if os.path.exists(path):
            out[i] = np.load(path)
        else:
            todo.append((i, t, path))
    for lo in range(0, len(todo), batch_size):
        chunk = todo[lo:lo + batch_size]
        emb = model.encode([c[1] for c in chunk], show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
        for (i, _, path), e in zip(chunk, emb):
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
