"""指标：balanced accuracy / AUROC / F1（不报裸 accuracy）+ bootstrap CI。"""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)


def scores_of(clf, X):
    """连续决策分数：LR 与 LinearSVC 统一用 decision_function。

    这样 sign(score) 与 clf.predict 一致。不能用 predict_proba：
    概率恒在 (0,1)，np.sign / s>0 会把所有样本判成正类，
    在标签不平衡时产生 balanced_acc=0.5、risk≈正类比例 等退化假象。
    """
    return clf.decision_function(X)


def compute_metrics(y_true, y_score) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.sign(np.asarray(y_score, dtype=float))
    y_pred = np.where(y_pred == 0, 1, y_pred)
    out = {
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auroc"] = float("nan")
    return out


def bootstrap_ci(y_true, y_score, n_boot: int = 1000, seed: int = 0,
                 metric: str = "auroc") -> list:
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], y_score[idx])
        vals.append(m[metric])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
