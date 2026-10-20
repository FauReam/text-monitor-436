"""E7 补充：step-only 输入下同分布（PRM800K test）Δ 的 paired bootstrap CI。

与 results/e3_step_transfer_delta_ci.json（迁移集）平行，评估集换成
PRM800K 分组 test（step-only 输入）。paired：同一次重采样下算各特征 AUROC
再取差。embedding 特征若已预编码（pre_encode_ablation.py），分钟级跑完。
用法：python src/e7_indomain_step_ci.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from data import input_texts, load_steps, group_split
from protocol import fit_full_cached
from utils import save_json, set_seed

FEATS = ["tfidf-word", "st-all-MiniLM-L6-v2",
         "st-bge-base-en-v1.5", "st-bge-large-en-v1.5"]
N_BOOT = 1000


def main():
    set_seed(0)
    train_df, _, test_df = group_split(load_steps("prm800k"), seed=0)
    y = test_df["label"].to_numpy()
    texts = input_texts(test_df, "step")

    scores = {}
    for feat in FEATS:
        fz, clf, _ = fit_full_cached(feat, "svm", train_df, seed=0,
                                     input_mode="step")
        from metrics import scores_of
        scores[feat] = scores_of(clf, fz.transform(texts))
        print(feat, "scored", flush=True)

    rng = np.random.RandomState(0)
    n = len(y)
    base = {f: roc_auc_score(y, s) for f, s in scores.items()}
    diffs = {f: [] for f in FEATS if f != "tfidf-word"}
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        a_t = roc_auc_score(y[idx], scores["tfidf-word"][idx])
        for f in diffs:
            diffs[f].append(roc_auc_score(y[idx], scores[f][idx]) - a_t)

    out = {"note": "step-only 输入，同分布 test 集 embedding−TF-IDF(word) 的"
                   " paired bootstrap Δ（svm，1000 次）",
           "input_mode": "step", "classifier": "svm", "split": "test",
           "base_auroc": {f: round(v, 4) for f, v in base.items()},
           "results": {}}
    for f, dl in diffs.items():
        dl = np.array(dl)
        out["results"][f"{f} - tfidf-word"] = {
            "delta_mean": float(np.mean(dl)),
            "ci95": [float(np.percentile(dl, 2.5)),
                     float(np.percentile(dl, 97.5))],
        }
    save_json(out, "e7_indomain_step_delta_ci.json")
    print("saved results/e7_indomain_step_delta_ci.json")
    for k, v in out["results"].items():
        print(k, round(v["delta_mean"], 4),
              [round(x, 4) for x in v["ci95"]])


if __name__ == "__main__":
    main()
