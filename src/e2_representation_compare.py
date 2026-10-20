"""E2：表征对比（必选 1.3 + novelty 1.6）。

在全量训练数据上横向对比四种特征 × 两种分类器，带 bootstrap CI。
结论以 Δ = AUROC(embedding) − AUROC(TF-IDF) 表述。
全量拟合经 fit_full_cached 与 E1/E3/E4/E5 共享（数值与重复训练一致）。
"""
import argparse

import numpy as np

from data import group_split, input_texts, load_steps
from features import FEATURES
from metrics import bootstrap_ci, compute_metrics, scores_of
from protocol import fit_full_cached
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
    yte = test_df["label"].to_numpy()

    results = {}
    for feat in a.features:
        results[feat] = {}
        for clf_name in CLASSIFIERS:
            fz, clf, info = fit_full_cached(feat, clf_name, train_df,
                                            seed=a.seed)
            s_test = scores_of(clf, fz.transform(input_texts(test_df)))
            test_m = compute_metrics(yte, s_test)
            ci = bootstrap_ci(yte, np.asarray(s_test))
            results[feat][clf_name] = {
                "C": info["C"], "test": test_m, "train": info["train"],
                "auroc_ci95": ci,
            }
            print(f"{feat} {clf_name}: test AUROC={test_m['auroc']:.4f} "
                  f"CI={ci} (C={info['C']})", flush=True)

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
