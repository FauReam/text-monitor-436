"""E3：泛化 Δ(偏移)（必选 1.5 generalization）。

训练：PRM800K（MATH，人工标签）。测试：PRM800K test（同分布）+ Math-Shepherd（GSM8K，自动标签）。
注意：这是复合分布偏移（题型、来源、标签方式同时变化），报告时不作单因素归因。
另输出 train vs test 指标对比（L06–L07 的 generalization 分析）。
"""
import argparse

from data import group_split, load_steps
from features import FEATURES, Featurizer
from protocol import fit_and_eval
from utils import save_json, set_seed

CLASSIFIERS = ["lr", "svm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dataset", default="prm800k")
    ap.add_argument("--transfer-dataset", default="mathshepherd")
    ap.add_argument("--features", nargs="*", default=FEATURES)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)

    df = load_steps(a.train_dataset)
    train_df, _, test_in = group_split(df, seed=a.seed)
    test_transfer = load_steps(a.transfer_dataset)
    print(f"train={len(train_df)} in-domain test={len(test_in)} "
          f"transfer test={len(test_transfer)}")

    results = {}
    for feat in a.features:
        fz = Featurizer(feat)
        results[feat] = {}
        for clf_name in CLASSIFIERS:
            r_in = fit_and_eval(clf_name, fz, train_df, test_in, seed=a.seed)
            # 迁移：用同一 featurizer（embedding 无状态；TF-IDF 已在训练集 fit）
            from data import input_texts
            from metrics import compute_metrics, scores_of
            from protocol import make_classifier
            clf = make_classifier(clf_name, r_in["C"], a.seed)
            from features import Featurizer as Fz  # 复用已 fit 的实例
            Xtr = fz.fit_transform(input_texts(train_df))
            clf.fit(Xtr, train_df["label"].to_numpy())
            Xtt = fz.transform(input_texts(test_transfer))
            m_tt = compute_metrics(test_transfer["label"].to_numpy(),
                                   scores_of(clf, Xtt))
            results[feat][clf_name] = {
                "C": r_in["C"],
                "train_metrics": r_in["train"],
                "test_in_domain": r_in["test"],
                "test_transfer": m_tt,
                "delta_transfer_minus_in": round(
                    m_tt["auroc"] - r_in["test"]["auroc"], 4),
                "train_test_gap": round(
                    r_in["train"]["auroc"] - r_in["test"]["auroc"], 4),
            }
            print(f"{feat} {clf_name}: in={r_in['test']['auroc']:.4f} "
                  f"transfer={m_tt['auroc']:.4f} "
                  f"train-test gap={results[feat][clf_name]['train_test_gap']}")

    save_json({"train_dataset": a.train_dataset,
               "transfer_dataset": a.transfer_dataset, "seed": a.seed,
               "note": "复合分布偏移（题型+来源+标签方式同时变化），不作单因素归因",
               "results": results}, "e3_generalization.json")
    print("saved results/e3_generalization.json")


if __name__ == "__main__":
    main()
