"""统一训练协议：分组 k-fold CV 选 C → 训练集重训 → test 只评估一次。

C 的选择只在训练数据内部进行（train 子集上的 GroupKFold），不碰 val/test。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from metrics import compute_metrics, scores_of

C_GRID = [0.1, 1.0, 10.0]


def make_classifier(name: str, C: float, seed: int):
    if name == "lr":
        return LogisticRegression(C=C, max_iter=5000, random_state=seed)
    if name == "svm":
        return LinearSVC(C=C, random_state=seed, max_iter=20000)
    raise ValueError(name)


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
