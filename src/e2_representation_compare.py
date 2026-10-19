"""E2：表征对比（必选 1.3 + novelty 1.6）。

在全量训练数据上横向对比四种特征 × 两种分类器，带 bootstrap CI。
结论以 Δ = AUROC(embedding) − AUROC(TF-IDF) 表述。
"""
import argparse

import numpy as np

from data import group_split, load_steps
from features import FEATURES, Featurizer
from metrics import bootstrap_ci, scores_of
from protocol import fit_and_eval
from utils import save_json, set_seed

CLASSIFIERS = ["lr", "svm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="prm800k")
    ap.add_argument("--features", nargs="*", default=FEATURES)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)

    df = load_steps(a.dataset)
    train_df, _, test_df = group_split(df, seed=a.seed)

    results = {}
    for feat in a.features:
        results[feat] = {}
        for clf_name in CLASSIFIERS:
            r = fit_and_eval(clf_name, Featurizer(feat), train_df, test_df,
                             seed=a.seed)
            ci = bootstrap_ci(np.array(r["y_test"]), np.array(r["s_test"]))
            results[feat][clf_name] = {
                "C": r["C"], "test": r["test"], "train": r["train"],
                "auroc_ci95": ci,
            }
            print(f"{feat} {clf_name}: test AUROC={r['test']['auroc']:.4f} "
                  f"CI={ci} (C={r['C']})")

    # Δ 汇总：每个 embedding 相对 tfidf-word / tfidf-char 的 AUROC 差
    deltas = {}
    for base in ("tfidf-word", "tfidf-char"):
        if base not in results:
            continue
        for feat in a.features:
            if feat.startswith("st-"):
                deltas[f"{feat} - {base}"] = {
                    c: round(results[feat][c]["test"]["auroc"]
                             - results[base][c]["test"]["auroc"], 4)
                    for c in CLASSIFIERS
                }
    save_json({"dataset": a.dataset, "seed": a.seed,
               "results": results, "delta_auroc": deltas},
              "e2_representation_compare.json")
    print("saved results/e2_representation_compare.json")


if __name__ == "__main__":
    main()
