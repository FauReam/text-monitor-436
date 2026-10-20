"""统一训练协议：分组 k-fold CV 选 C → 训练集重训 → test 只评估一次。

C 的选择只在训练数据内部进行（train 子集上的 GroupKFold），不碰 val/test。
"""
import hashlib
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from metrics import compute_metrics, scores_of
from utils import CACHE

C_GRID = [0.1, 1.0, 10.0]


def make_classifier(name: str, C: float, seed: int):
    if name == "lr":
        return LogisticRegression(C=C, max_iter=5000, random_state=seed)
    if name == "svm":
        return LinearSVC(C=C, random_state=seed, max_iter=20000)
    raise ValueError(name)


def _full_train_key(feat_name: str, clf_name: str, seed: int,
                    input_mode: str, train_df) -> str:
    fp = (f"{feat_name}|{clf_name}|{seed}|{input_mode}|{len(train_df)}|"
          f"{train_df['problem_id'].nunique()}|{int(train_df['label'].sum())}")
    return hashlib.md5(fp.encode()).hexdigest()


def fit_full_cached(feat_name: str, clf_name: str, train_df, seed: int = 0,
                    input_mode: str = "problem+step"):
    """全量训练集上 (特征, 分类器) 拟合结果的磁盘缓存。

    E1 的 n=full 点与 E2/E3/E4/E5 训练的是同一个模型（同数据、同种子、
    同分组 CV），重复训练代价大（TF-IDF 全量 + 3折×3C 的 CV）。
    首次调用与 fit_and_eval 的训练半段完全一致：fit_transform →
    GroupKFold 选 C → 全量重训；结果 (featurizer, clf, info) 存到
    data/cache/fitfull_*.joblib，后续调用直接读盘，数值不变。
    缓存键含 (特征, 分类器, 种子, 输入模式, 行数, 题目数, 标签和) 指纹。
    """
    import joblib

    from data import input_texts
    from features import Featurizer

    key = _full_train_key(feat_name, clf_name, seed, input_mode, train_df)
    path = os.path.join(CACHE, f"fitfull_{key}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    fz = Featurizer(feat_name)
    Xtr = fz.fit_transform(input_texts(train_df, input_mode))
    ytr = train_df["label"].to_numpy()
    groups = train_df["problem_id"].to_numpy()
    C = select_C_cv(clf_name, Xtr, ytr, groups, seed)
    clf = make_classifier(clf_name, C, seed)
    clf.fit(Xtr, ytr)
    info = {"C": C, "train": compute_metrics(ytr, scores_of(clf, Xtr))}
    joblib.dump((fz, clf, info), path)
    return fz, clf, info


def select_C_cv(name: str, X, y, groups, seed: int = 0, n_splits: int = 3) -> float:
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:  # 样本太少做不了分组 CV 时用默认 C
        return 1.0
    best_c, best_score = 1.0, -np.inf
    gkf = GroupKFold(n_splits=n_splits)
    for C in C_GRID:
        fold_scores = []
        for tr, va in gkf.split(X, y, groups):
            clf = make_classifier(name, C, seed)
            clf.fit(X[tr], y[tr])
            m = compute_metrics(np.asarray(y)[va], scores_of(clf, X[va]))
            fold_scores.append(np.nan_to_num(m["auroc"]))
        s = float(np.mean(fold_scores))
        if s > best_score:
            best_score, best_c = s, C
    return best_c


def fit_and_eval(name: str, featurizer, train_df, test_df, seed: int = 0,
                 input_mode: str = "problem+step") -> dict:
    """一次完整协议。返回 test/train 指标 + 选出的 C。"""
    from data import input_texts

    Xtr = featurizer.fit_transform(input_texts(train_df, input_mode))
    ytr = train_df["label"].to_numpy()
    groups = train_df["problem_id"].to_numpy()
    C = select_C_cv(name, Xtr, ytr, groups, seed)

    clf = make_classifier(name, C, seed)
    clf.fit(Xtr, ytr)

    Xte = featurizer.transform(input_texts(test_df, input_mode))
    yte = test_df["label"].to_numpy()

    return {
        "C": C,
        "test": compute_metrics(yte, scores_of(clf, Xte)),
        "train": compute_metrics(ytr, scores_of(clf, Xtr)),
        "y_test": yte.tolist(),
        "s_test": scores_of(clf, Xte).tolist(),
    }
