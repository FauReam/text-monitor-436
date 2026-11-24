"""统一训练协议：分组 k-fold CV 选 C → 训练集重训 → test 只评估一次。

C 的选择只在训练数据内部进行（train 子集上的 GroupKFold），不碰 val/test。
"""
import hashlib
import os

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from metrics import compute_metrics, scores_of
from utils import CACHE

C_GRID = [0.1, 1.0, 10.0]
PROTOCOL_VERSION = "foldwise-tfidf-v3"


def make_classifier(name: str, C: float, seed: int):
    if name == "lr":
        return LogisticRegression(C=C, max_iter=5000, random_state=seed)
    if name == "svm":
        return LinearSVC(C=C, random_state=seed, max_iter=20000)
    raise ValueError(name)


def _full_train_key(feat_name: str, clf_name: str, seed: int,
                    input_mode: str, train_df) -> str:
    import pandas as pd

    columns = ["problem_id", "problem_text", "step_text", "label"]
    content = pd.util.hash_pandas_object(train_df[columns], index=False)
    data_hash = hashlib.sha256(content.to_numpy().tobytes()).hexdigest()
    fp = (f"{PROTOCOL_VERSION}|{feat_name}|{clf_name}|{seed}|{input_mode}|"
          f"{data_hash}|{C_GRID}")
    return hashlib.md5(fp.encode()).hexdigest()


def fit_full_cached(feat_name: str, clf_name: str, train_df, seed: int = 0,
                    input_mode: str = "problem+step"):
    """全量训练集上 (特征, 分类器) 拟合结果的磁盘缓存。

    E1 的 n=full 点与 E2/E3/E4/E5 训练的是同一个模型（同数据、同种子、
    同分组 CV），重复训练代价大（TF-IDF 全量 + 3折×3C 的 CV）。
    TF-IDF 在每个内部训练折独立 fit；冻结 embedding 可预先编码。
    缓存键包含协议版本及有序数据内容，旧协议缓存不会复用。
    """
    import joblib

    from data import input_texts
    from features import Featurizer

    key = _full_train_key(feat_name, clf_name, seed, input_mode, train_df)
    path = os.path.join(CACHE, f"fitfull_{key}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    fz = Featurizer(feat_name)
    texts = input_texts(train_df, input_mode)
    ytr = train_df["label"].to_numpy()
    groups = train_df["problem_id"].to_numpy()
    if fz.kind == "tfidf":
        C = select_C_tfidf(clf_name, fz, texts, ytr, groups, seed)
    Xtr = fz.fit_transform(input_texts(train_df, input_mode))
    if fz.kind != "tfidf":
        C = select_C_cv(clf_name, Xtr, ytr, groups, seed)
    clf = make_classifier(clf_name, C, seed)
    clf.fit(Xtr, ytr)
    info = {"C": C, "protocol_version": PROTOCOL_VERSION,
            "train": compute_metrics(ytr, scores_of(clf, Xtr))}
    joblib.dump((fz, clf, info), path)
    return fz, clf, info


def select_C_cv(name: str, X, y, groups, seed: int = 0, n_splits: int = 3) -> float:
    """选 C；仅适用于不从当前数据学习参数的预编码冻结特征。"""
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:  # 样本太少做不了分组 CV 时用默认 C
        return 1.0
    best_c, best_score = 1.0, -np.inf
    gkf = GroupKFold(n_splits=n_splits)
    for C in C_GRID:
        fold_scores = []
        for tr, va in gkf.split(X, y, groups):
            if len(np.unique(np.asarray(y)[tr])) < 2 or len(np.unique(np.asarray(y)[va])) < 2:
                raise ValueError("Grouped CV fold has one class; revise the training split.")
            clf = make_classifier(name, C, seed)
            clf.fit(X[tr], y[tr])
            m = compute_metrics(np.asarray(y)[va], scores_of(clf, X[va]))
            fold_scores.append(m["auroc"])
        s = float(np.mean(fold_scores))
        if s > best_score:
            best_score, best_c = s, C
    return best_c


def select_C_tfidf(name, featurizer, texts, y, groups, seed=0, n_splits=3):
    """折内拟合词表、min_df 和 IDF，同一折特征用于所有 C。"""
    y, groups = np.asarray(y), np.asarray(groups)
    if len(np.unique(groups)) < n_splits:
        return 1.0
    scores = {C: [] for C in C_GRID}
    for tr, va in GroupKFold(n_splits=n_splits).split(texts, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            raise ValueError("Grouped CV fold has one class; revise the training split.")
        vectorizer = clone(featurizer.tfidf.vec)
        Xtr = vectorizer.fit_transform([texts[i] for i in tr])
        Xva = vectorizer.transform([texts[i] for i in va])
        for C in C_GRID:
            clf = make_classifier(name, C, seed)
            clf.fit(Xtr, y[tr])
            scores[C].append(compute_metrics(y[va], scores_of(clf, Xva))["auroc"])
    return max(C_GRID, key=lambda C: float(np.mean(scores[C])))


def fit_and_eval(name: str, featurizer, train_df, test_df, seed: int = 0,
                 input_mode: str = "problem+step") -> dict:
    """一次完整协议。返回 test/train 指标 + 选出的 C。"""
    from data import input_texts

    texts = input_texts(train_df, input_mode)
    ytr = train_df["label"].to_numpy()
    groups = train_df["problem_id"].to_numpy()
    if featurizer.kind == "tfidf":
        C = select_C_tfidf(name, featurizer, texts, ytr, groups, seed)
    Xtr = featurizer.fit_transform(texts)
    if featurizer.kind != "tfidf":
        C = select_C_cv(name, Xtr, ytr, groups, seed)

    clf = make_classifier(name, C, seed)
    clf.fit(Xtr, ytr)

    Xte = featurizer.transform(input_texts(test_df, input_mode))
    yte = test_df["label"].to_numpy()

    return {
        "C": C,
        "protocol_version": PROTOCOL_VERSION,
        "test": compute_metrics(yte, scores_of(clf, Xte)),
        "train": compute_metrics(ytr, scores_of(clf, Xtr)),
        "y_test": yte.tolist(),
        "s_test": scores_of(clf, Xte).tolist(),
    }
