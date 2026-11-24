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


def paired_problem_bootstrap(y_true, scores, groups, baselines,
                             n_boot=1000, seed=0):
    """整题有放回重采样，所有方法共享同一组题目权重。

    估计的是步骤加权 AUROC 差的不确定性，条件于当前固定训练模型；
    不包含训练抽样随机性，也不将探索性测试变成独立确认。
    使用组出现次数作为每行 sample_weight，等价于复制该题所有步骤。
    """
    y = np.asarray(y_true)
    groups = np.asarray(groups)
    scores = {name: np.asarray(s, dtype=float) for name, s in scores.items()}
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups):
        raise ValueError("Labels and groups must be aligned one-dimensional arrays.")
    if not scores or not baselines or n_boot < 2:
        raise ValueError("Provide scores, baselines and at least two bootstrap draws.")
    if any(s.shape != y.shape or not np.isfinite(s).all() for s in scores.values()):
        raise ValueError("Every score vector must be finite and aligned with labels.")
    if len(np.unique(y)) != 2:
        raise ValueError("Binary labels with both classes are required.")
    if any(b not in scores for b in baselines):
        raise ValueError("Every baseline must have a score vector.")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two independent problem groups are required.")
    pairs = [(method, baseline) for baseline in baselines
             for method in scores if method not in baselines]
    if not pairs:
        raise ValueError("Provide at least one non-baseline method.")
    point = {name: float(roc_auc_score(y, s)) for name, s in scores.items()}
    draws = {pair: [] for pair in pairs}
    rng = np.random.RandomState(seed)
    valid = 0
    for _ in range(n_boot * 20):
        sampled = rng.randint(len(unique), size=len(unique))
        counts = np.bincount(sampled, minlength=len(unique))
        weights = counts[inverse]
        if len(np.unique(y[weights > 0])) < 2:
            continue
        aucs = {name: roc_auc_score(y, s, sample_weight=weights)
                for name, s in scores.items()}
        for pair in pairs:
            draws[pair].append(aucs[pair[0]] - aucs[pair[1]])
        valid += 1
        if valid == n_boot:
            break
    if valid != n_boot:
        raise ValueError("Not enough two-class bootstrap draws; inspect group/label structure.")
    return {
        "unit": "problem_id", "n_problems": len(unique), "n_steps": len(y),
        "n_boot": valid, "seed": seed, "auroc": point,
        "uncertainty": "conditional_on_fixed_training; step-weighted AUROC",
        "comparisons": {f"{m} - {b}": {
            "delta": point[m] - point[b],
            "ci95": np.percentile(draws[(m, b)], [2.5, 97.5]).tolist(),
        } for m, b in pairs},
    }
