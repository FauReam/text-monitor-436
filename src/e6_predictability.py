"""E6（可选，仅作 discussion）：能否不测就预测？

对每种特征，在训练集上算一个廉价统计量（类间/类内散度比，H-score 的思路；
TF-IDF 高维稀疏，先 SVD 到 128 维），与 E1 全量点实测的 Δ 算 Spearman 相关。
只有 4 个点，明确标注为初步观察 / future work，不声称构造了估计器。
"""
import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD

from data import group_split, input_texts, load_steps
from features import FEATURES, Featurizer
from utils import load_json, save_json, set_seed


def separation_statistic(X, y) -> float:
    """tr(S_b) / tr(S_w)：类间散度与类内散度之比。"""
    if hasattr(X, "tocsr"):  # 稀疏 TF-IDF 先降维
        X = TruncatedSVD(n_components=128, random_state=0).fit_transform(X)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    mu = X.mean(axis=0)
    sb, sw = 0.0, 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        muc = Xc.mean(axis=0)
        sb += len(Xc) * float(((muc - mu) ** 2).sum())
        sw += float(((Xc - muc) ** 2).sum())
    return float(sb / max(sw, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="prm800k")
    ap.add_argument("--features", nargs="*", default=FEATURES)
    ap.add_argument("--classifier", default="lr")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)

    df = load_steps(a.dataset)
    train_df, _, _ = group_split(df, seed=a.seed)
    ytr = train_df["label"].to_numpy()

    e1 = load_json(f"e1_learning_curves_{a.dataset}.json")["results"]

    stats, deltas = {}, {}
    for feat in a.features:
        fz = Featurizer(feat)
        X = fz.fit_transform(input_texts(train_df))
        stats[feat] = separation_statistic(X, ytr)
        # 实测 Δ：相对 tfidf-word 在 full 点的 AUROC 差（取各种子均值）
        base = e1["tfidf-word"][a.classifier]["full"]
        cur = e1[feat][a.classifier]["full"]
        deltas[feat] = float(np.mean([r["test"]["auroc"] for r in cur])
                             - np.mean([r["test"]["auroc"] for r in base]))
        print(f"{feat}: S={stats[feat]:.4f}  delta={deltas[feat]:+.4f}")

    xs = [stats[f] for f in a.features]
    ys = [deltas[f] for f in a.features]
    rho, p = spearmanr(xs, ys)
    save_json({"dataset": a.dataset, "classifier": a.classifier,
               "note": "4 个点的初步观察，仅作 discussion / future work",
               "separation_stat": stats, "delta_vs_tfidf_word": deltas,
               "spearman_rho": float(rho), "spearman_p": float(p)},
              "e6_predictability.json")
    print(f"spearman rho={rho:.3f} (p={p:.3f}); "
          "saved results/e6_predictability.json")


if __name__ == "__main__":
    main()
