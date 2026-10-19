"""E5（可选）：拒答机制（Chow 规则）与风险-覆盖率曲线。

监视器置信度低于阈值时输出"悬置"。画 risk（在已判样本上的错误率）
vs coverage（已判样本占比）曲线：曲线越靠左下，监视器越可用。
注意：本实验监视器味最重，不进期末核心图；特征家族/维度不同，
沿"TF-IDF → MiniLM → bge-base → bge-large"的顺序只作描述性比较。
"""
import argparse

import numpy as np

from data import group_split, input_texts, load_steps
from features import FEATURES, Featurizer
from protocol import fit_and_eval, make_classifier
from metrics import scores_of
from utils import save_json, set_seed


def risk_coverage(y_true, s, n_points: int = 50):
    """按 |置信度| 从高到低扫描：返回 (coverage, risk) 序列。"""
    y_true = np.asarray(y_true)
    conf = np.abs(np.asarray(s))
    order = np.argsort(-conf)
    coverages = np.linspace(0.02, 1.0, n_points)
    curve = []
    for c in coverages:
        k = max(1, int(len(order) * c))
        idx = order[:k]
        y_hat = np.where(np.asarray(s)[idx] > 0, 1, -1)
        risk = float(np.mean(y_hat != y_true[idx]))
        curve.append({"coverage": float(c), "risk": risk})
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="prm800k")
    ap.add_argument("--features", nargs="*", default=FEATURES)
    ap.add_argument("--classifier", default="lr",
                    help="LR 的 predict_proba 作置信度，最贴合 Chow 规则")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)

    df = load_steps(a.dataset)
    train_df, _, test_df = group_split(df, seed=a.seed)
    yte = test_df["label"].to_numpy()

    results = {}
    for feat in a.features:
        fz = Featurizer(feat)
        r = fit_and_eval(a.classifier, fz, train_df, test_df, seed=a.seed)
        clf = make_classifier(a.classifier, r["C"], a.seed)
        clf.fit(fz.fit_transform(input_texts(train_df)),
                train_df["label"].to_numpy())
        s = scores_of(clf, fz.transform(input_texts(test_df)))
        results[feat] = {
            "C": r["C"], "risk_coverage": risk_coverage(yte, s),
            "risk_at_full": risk_coverage(yte, s, n_points=1)[0]["risk"],
        }
        print(f"{feat}: risk@coverage=1.0 -> {results[feat]['risk_at_full']:.4f}")

    save_json({"dataset": a.dataset, "classifier": a.classifier, "seed": a.seed,
               "note": "拒答=Chow 规则；特征家族/维度不同，仅描述性比较",
               "results": results}, "e5_abstention.json")
    print("saved results/e5_abstention.json")


if __name__ == "__main__":
    main()
