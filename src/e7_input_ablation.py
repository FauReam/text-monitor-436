"""E7 输入消融：step-only / step-first vs 主设定 problem+step。

动机（评委意见）：输入 = problem + "\\n" + step，step 在尾部；
MiniLM 上限 256 / bge 512 token，长题目会截掉 step，可能人为压低 embedding。
本消融在 E1 全量点（n=full, seed=0）上比较三种输入构造：
    problem+step  主设定（结果命中 fit_full_cached，零开销）
    step+problem  step 前置，截断只损失题目
    step          仅当前步（预登记中既定的消融）
输出 results/input_ablation.json。
用法：python src/e7_input_ablation.py --features st-all-MiniLM-L6-v2 ...
"""
import argparse
import json
import os

import numpy as np

from data import group_split, input_texts, load_steps
from features import FEATURES
from metrics import compute_metrics, scores_of
from protocol import fit_full_cached

MODES = ["problem+step", "step+problem", "step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="prm800k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", nargs="*", default=FEATURES)
    ap.add_argument("--classifiers", nargs="*", default=["lr", "svm"])
    ap.add_argument("--out", default="results/input_ablation.json")
    a = ap.parse_args()

    df = load_steps(a.dataset)
    train_df, _, test_df = group_split(df, seed=a.seed)
    yte = test_df["label"].to_numpy()

    out = {}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    out.setdefault("dataset", a.dataset)
    out.setdefault("seed", a.seed)
    out["note"] = ("输入构造消融（E1 全量点）：problem+step=主设定；"
                   "step+problem=step 前置防截断；step=仅当前步")
    res = out.setdefault("results", {})

    for feat in a.features:
        for mode in MODES:
            for clf_name in a.classifiers:
                if res.get(feat, {}).get(mode, {}).get(clf_name):
                    continue  # 断点续跑
                fz, clf, info = fit_full_cached(feat, clf_name, train_df,
                                                seed=a.seed, input_mode=mode)
                Xte = fz.transform(input_texts(test_df, mode))
                m = compute_metrics(yte, scores_of(clf, Xte))
                res.setdefault(feat, {}).setdefault(mode, {})[clf_name] = {
                    "C": info["C"], "test": m,
                    "train_auroc": info["train"]["auroc"],
                }
                print(feat, mode, clf_name, "test auroc:",
                      round(m["auroc"], 4), flush=True)
                with open(a.out, "w") as f:
                    json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
