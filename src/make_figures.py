"""results/*.json → figures/*.png（+pdf）。轴必带标签；图服务于 Δ(θ) 语言。

缺哪个 results 文件就跳过哪张图——先跑哪个实验就先出哪张。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import FIGURES, RESULTS

FEAT_STYLE = {
    "tfidf-word": {"label": "TF-IDF (word)", "color": "#888888", "ls": "--"},
    "tfidf-char": {"label": "TF-IDF (char)", "color": "#bbbbbb", "ls": ":"},
    "st-all-MiniLM-L6-v2": {"label": "MiniLM-L6", "color": "#1f77b4", "ls": "-"},
    "st-bge-base-en-v1.5": {"label": "bge-base", "color": "#2ca02c", "ls": "-"},
    "st-bge-large-en-v1.5": {"label": "bge-large", "color": "#d62728", "ls": "-"},
}
EMB = ["st-all-MiniLM-L6-v2", "st-bge-base-en-v1.5", "st-bge-large-en-v1.5"]
ORDER = ["tfidf-word", "tfidf-char"] + EMB


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {name}")


def _load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        print(f"skip {name} (not found)")
        return None
    with open(path) as f:
        return json.load(f)


def fig_e1(dataset="prm800k", clf="lr", suffix=""):
    d = _load(f"e1_learning_curves_{dataset}.json")
    if not d:
        return
    res = d["results"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for feat in ORDER:
        if feat not in res or clf not in res[feat]:
            continue
        pts = res[feat][clf]
        ns = sorted((k for k in pts), key=lambda k: (k == "full", int(k) if k != "full" else 0))
        xs, mu, lo, hi = [], [], [], []
        for k in ns:
            v = [r["test"]["auroc"] for r in pts[k]]
            x = np.mean([r["n"] for r in pts[k]])
            xs.append(x)
            mu.append(np.mean(v))
            lo.append(np.mean(v) - 1.96 * np.std(v))
            hi.append(np.mean(v) + 1.96 * np.std(v))
        st = FEAT_STYLE[feat]
        ax.plot(xs, mu, label=st["label"], color=st["color"], ls=st["ls"],
                marker="o", ms=4)
        ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("training samples n (log scale)")
    ax.set_ylabel("test AUROC")
    ax.legend()
    _save(fig, f"e1_learning_curve{suffix}")

    # Δ(n)：相对 tfidf-word
    if "tfidf-word" in res:
        fig, ax = plt.subplots(figsize=(7, 4))
        base = res["tfidf-word"][clf]
        for feat in EMB:
            if feat not in res:
                continue
            pts, bpts = res[feat][clf], base
            xs, ys = [], []
            for k in pts:
                if k not in bpts:
                    continue
                xs.append(np.mean([r["n"] for r in pts[k]]))
                ys.append(np.mean([r["test"]["auroc"] for r in pts[k]])
                          - np.mean([r["test"]["auroc"] for r in bpts[k]]))
            st = FEAT_STYLE[feat]
            ax.plot(xs, ys, label=f"{st['label']} − TF-IDF", color=st["color"],
                    marker="o", ms=4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("training samples n (log scale)")
        ax.set_ylabel("Δ AUROC (embedding − TF-IDF)")
        ax.legend()
        _save(fig, f"e1_gap{suffix}")


def fig_e3(clf="lr", suffix=""):
    d = _load("e3_generalization.json")
    if not d:
        return
    res = d["results"]
    feats = [f for f in ORDER if f in res]
    x = np.arange(len(feats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    in_d = [res[f][clf]["test_in_domain"]["auroc"] for f in feats]
    tr = [res[f][clf]["test_transfer"]["auroc"] for f in feats]
    ax.bar(x - w / 2, in_d, w, label="in-domain (PRM800K/MATH)")
    ax.bar(x + w / 2, tr, w, label="transfer (Math-Shepherd/GSM8K)")
    ax.set_xticks(x, [FEAT_STYLE[f]["label"] for f in feats])
    ax.set_ylabel("test AUROC")
    ax.legend()
    _save(fig, f"e3_generalization{suffix}")

    fig, ax = plt.subplots(figsize=(8, 4))
    tr_m = [res[f][clf]["train_metrics"]["auroc"] for f in feats]
    te_m = [res[f][clf]["test_in_domain"]["auroc"] for f in feats]
    ax.bar(x - w / 2, tr_m, w, label="train AUROC")
    ax.bar(x + w / 2, te_m, w, label="test AUROC")
    ax.set_xticks(x, [FEAT_STYLE[f]["label"] for f in feats])
    ax.set_ylabel("AUROC")
    ax.legend()
    _save(fig, f"e3_train_test{suffix}")


def fig_e4(clf="lr", suffix=""):
    d = _load("e4_robustness.json")
    if not d:
        return
    res = d["results"]
    perts = [p for p in res[next(iter(res))][clf]["auroc_drop"]]
    feats = [f for f in ORDER if f in res]
    x = np.arange(len(perts))
    w = 0.8 / len(feats)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, f in enumerate(feats):
        drops = [res[f][clf]["auroc_drop"][p] for p in perts]
        ax.bar(x + i * w, drops, w, label=FEAT_STYLE[f]["label"],
               color=FEAT_STYLE[f]["color"])
    ax.set_xticks(x + 0.4 - w / 2, perts)
    ax.set_ylabel("AUROC drop vs clean")
    ax.legend()
    _save(fig, f"e4_robustness{suffix}")


def fig_e5():
    d = _load("e5_abstention.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for feat in ORDER:
        if feat not in d["results"]:
            continue
        curve = d["results"][feat]["risk_coverage"]
        st = FEAT_STYLE[feat]
        ax.plot([c["coverage"] for c in curve], [c["risk"] for c in curve],
                label=st["label"], color=st["color"], ls=st["ls"])
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("risk (error rate on answered)")
    ax.legend()
    _save(fig, "e5_risk_coverage")


def fig_e6():
    d = _load("e6_predictability.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for feat, s in d["separation_stat"].items():
        ax.scatter(s, d["delta_vs_tfidf_word"][feat],
                   color=FEAT_STYLE.get(feat, {}).get("color", "k"))
        ax.annotate(FEAT_STYLE.get(feat, {}).get("label", feat),
                    (s, d["delta_vs_tfidf_word"][feat]), fontsize=8)
    ax.set_xlabel("separation statistic  tr(S_b)/tr(S_w)  (train set)")
    ax.set_ylabel("Δ AUROC vs TF-IDF (test)")
    ax.set_title(f"Spearman ρ={d['spearman_rho']:.2f} (n=4, preliminary)")
    _save(fig, "e6_scatter")


if __name__ == "__main__":
    for clf in ("lr", "svm"):
        suffix = "" if clf == "lr" else "_svm"
        fig_e1(clf=clf, suffix=suffix)
        fig_e3(clf=clf, suffix=suffix)
        fig_e4(clf=clf, suffix=suffix)
    fig_e5()
    fig_e6()
