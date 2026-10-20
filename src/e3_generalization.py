"""E3：泛化 Δ(偏移)（必选 1.5 generalization）。

训练：PRM800K（MATH，人工标签）。测试：PRM800K test（同分布）+ Math-Shepherd（GSM8K，自动标签）。
注意：这是复合分布偏移（题型、来源、标签方式同时变化），报告时不作单因素归因。
另输出 train vs test 指标对比（L06–L07 的 generalization 分析）。
全量拟合经 fit_full_cached 与 E1/E2/E4/E5 共享（数值与重复训练一致）。
"""
import argparse

from data import group_split, input_texts, load_steps
from features import FEATURES
from metrics import compute_metrics, scores_of
from protocol import fit_full_cached
from utils import save_json, set_seed

CLASSIFIERS = ["lr", "svm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dataset", default="prm800k")
    ap.add_argument("--transfer-dataset", default="mathshepherd")
    ap.add_argument("--features", nargs="*", default=FEATURES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input-mode", default="problem+step")
    ap.add_argument("--out", default="e3_generalization.json")
    a = ap.parse_args()
    set_seed(a.seed)

    df = load_steps(a.train_dataset)
    train_df, _, test_in = group_split(df, seed=a.seed)
    test_transfer = load_steps(a.transfer_dataset)
    print(f"train={len(train_df)} in-domain test={len(test_in)} "
          f"transfer test={len(test_transfer)}")

    results = {}
    for feat in a.features:
        results[feat] = {}
        for clf_name in CLASSIFIERS:
            fz, clf, info = fit_full_cached(feat, clf_name, train_df,
                                            seed=a.seed,
                                            input_mode=a.input_mode)
            # 同分布 test
            m_in = compute_metrics(test_in["label"].to_numpy(),
                                   scores_of(clf, fz.transform(
                                       input_texts(test_in, a.input_mode))))
            # 迁移 test：同一 featurizer（embedding 无状态；TF-IDF 已在训练集 fit）
            m_tt = compute_metrics(test_transfer["label"].to_numpy(),
                                   scores_of(clf, fz.transform(
                                       input_texts(test_transfer,
                                                   a.input_mode))))
            results[feat][clf_name] = {
                "C": info["C"],
                "train_metrics": info["train"],
                "test_in_domain": m_in,
                "test_transfer": m_tt,
                "delta_transfer_minus_in": round(
                    m_tt["auroc"] - m_in["auroc"], 4),
                "train_test_gap": round(
                    info["train"]["auroc"] - m_in["auroc"], 4),
            }
            print(f"{feat} {clf_name}: in={m_in['auroc']:.4f} "
                  f"transfer={m_tt['auroc']:.4f} "
                  f"train-test gap={results[feat][clf_name]['train_test_gap']}",
                  flush=True)

    save_json({"train_dataset": a.train_dataset,
               "transfer_dataset": a.transfer_dataset, "seed": a.seed,
               "input_mode": a.input_mode,
               "note": "复合分布偏移（题型+来源+标签方式同时变化），不作单因素归因",
               "results": results}, a.out)
    print("saved results/e3_generalization.json")


if __name__ == "__main__":
    main()
