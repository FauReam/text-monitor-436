"""E1（最高优先级，唯一硬缺口）：学习曲线 Δ(n)。

n ∈ {20, 50, 100, 500, 2000, 全量} × 特征 × 分类器 × 10 个种子（分层抽样）。
对应：必选 1.4（小样本）、1.5 sample efficiency；train/test 同时记录供 E3 画差距图。
预登记预测 1：gap(n) = AUROC(embedding) − AUROC(TF-IDF) 峰值出现在 n ≤ 100。
"""
import argparse

import numpy as np

from data import group_split, load_sst2, load_steps, subsample_stratified
from features import FEATURES, Featurizer
from protocol import fit_and_eval
from utils import save_json, set_seed

N_LIST = [20, 50, 100, 500, 2000, None]  # None = 全量
N_SEEDS = 10
CLASSIFIERS = ["lr", "svm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="prm800k", choices=["prm800k", "sst2"])
    ap.add_argument("--features", nargs="*", default=FEATURES)
    ap.add_argument("--classifiers", nargs="*", default=CLASSIFIERS)
    ap.add_argument("--n-list", nargs="*", type=int,
                    default=[n for n in N_LIST if n])
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="冒烟测试：只跑 tfidf-word + 一个 embedding，n 只到 100，2 种子")
    a = ap.parse_args()
    set_seed(a.seed)

    if a.quick:
        a.features = ["tfidf-word", "st-all-MiniLM-L6-v2"]
        a.n_list = [20, 100]
        a.n_seeds = 2

    df = load_sst2() if a.dataset == "sst2" else load_steps(a.dataset)
    train_df, val_df, test_df = group_split(df, seed=a.seed)
    trainval = train_df  # val 合并进 train，由分组 CV 内部选 C
    print(f"train={len(trainval)} test={len(test_df)} "
          f"problems(train)={trainval['problem_id'].nunique()}")

    results = {}
    for feat in a.features:
        results[feat] = {}
        for clf_name in a.classifiers:
            results[feat][clf_name] = {}
            for n in a.n_list + [None]:
                runs = []
                for s in range(a.n_seeds):
                    sub = (trainval if n is None else
                           subsample_stratified(trainval, n, seed=a.seed * 1000 + s))
                    r = fit_and_eval(clf_name, Featurizer(feat), sub, test_df,
                                     seed=a.seed * 1000 + s)
                    runs.append({
                        "seed": a.seed * 1000 + s, "n": len(sub), "C": r["C"],
                        "test": r["test"], "train": r["train"],
                    })
                    print(f"{feat} {clf_name} n={len(sub)} seed={s} "
                          f"AUROC={r['test']['auroc']:.4f}")
                key = "full" if n is None else str(n)
                results[feat][clf_name][key] = runs

    save_json({"dataset": a.dataset, "seed": a.seed, "n_seeds": a.n_seeds,
               "results": results}, f"e1_learning_curves_{a.dataset}.json")
    print("saved results/e1_learning_curves_%s.json" % a.dataset)


if __name__ == "__main__":
    main()
