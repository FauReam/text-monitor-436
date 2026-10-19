"""E4：鲁棒 Δ(扰动)（必选 1.5 robustness）。

在干净测试集上训练的模型，分别评估 none / typo / number / latex / combo
五种扰动下的性能衰减；比较各特征的衰减幅度（Δ(扰动) 以差值语言报告）。
预登记预测 3：embedding 的 AUROC 衰减幅度不大于 TF-IDF。
"""
import argparse

from data import group_split, input_texts, load_steps
from features import FEATURES, Featurizer
from metrics import compute_metrics, scores_of
from perturb import PERTURBATIONS
from protocol import fit_and_eval, make_classifier
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
        fz = Featurizer(feat)
        results[feat] = {}
        for clf_name in CLASSIFIERS:
            r = fit_and_eval(clf_name, fz, train_df, test_df, seed=a.seed)
            clf = make_classifier(clf_name, r["C"], a.seed)
            Xtr = fz.fit_transform(input_texts(train_df))
            clf.fit(Xtr, train_df["label"].to_numpy())
            yte = test_df["label"].to_numpy()
            per_pert = {}
            for pname, pfun in PERTURBATIONS.items():
                texts = [pfun(t, seed=a.seed) for t in input_texts(test_df)]
                m = compute_metrics(yte, scores_of(clf, fz.transform(texts)))
                per_pert[pname] = m
            clean = per_pert["none"]["auroc"]
            results[feat][clf_name] = {
                "C": r["C"], "per_perturbation": per_pert,
                "auroc_drop": {p: round(clean - m["auroc"], 4)
                               for p, m in per_pert.items() if p != "none"},
            }
            print(f"{feat} {clf_name}: clean={clean:.4f} drops="
                  f"{results[feat][clf_name]['auroc_drop']}")

    save_json({"dataset": a.dataset, "seed": a.seed,
               "perturbations": list(PERTURBATIONS), "results": results},
              "e4_robustness.json")
    print("saved results/e4_robustness.json")


if __name__ == "__main__":
    main()
