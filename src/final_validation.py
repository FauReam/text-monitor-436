"""Corrected full-data evaluation. Historical result JSON files are untouched.

Run from the project root:
  python3 src/final_validation.py --out results/final_v3_step.json
Evaluations remain exploratory because these test sets were previously inspected.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data import group_split, input_texts, load_steps
from features import FEATURES
from metrics import paired_problem_bootstrap, scores_of
from protocol import PROTOCOL_VERSION, fit_full_cached
from utils import DATA


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", nargs="+", default=FEATURES)
    ap.add_argument("--input-mode", choices=["problem+step", "step+problem", "step"],
                    default="step")
    ap.add_argument("--classifier", choices=["lr", "svm"], default="svm")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/final_v3_step.json")
    a = ap.parse_args()
    baselines = ["tfidf-word", "tfidf-char"]
    if any(b not in a.features for b in baselines):
        ap.error("Include both word and character TF-IDF baselines.")
    if not any(f not in baselines for f in a.features):
        ap.error("Include at least one pretrained feature.")
    target = Path(a.out)
    if target.exists():
        ap.error(f"Output already exists: {target}. Choose a new output path.")
    for split in ("in_domain", "transfer"):
        score_path = target.with_name(target.stem + f"_{split}_scores.npz")
        if score_path.exists():
            ap.error(f"Score output already exists: {score_path}. Choose a new output path.")
    # Validate local data before any encoder can download weights.
    required = [Path(DATA) / f"{name}_steps.csv" for name in ("prm800k", "mathshepherd")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        ap.error("Missing local data (no models have been loaded): " + ", ".join(missing))
    train, _, test = group_split(load_steps("prm800k"), seed=a.seed)
    transfer = load_steps("mathshepherd")
    overlap = set(train.problem_id) & set(transfer.problem_id)
    if overlap:
        raise ValueError(f"Transfer contains {len(overlap)} training problem IDs; audit overlap.")
    splits = {"in_domain": test, "transfer": transfer}
    texts = {k: input_texts(df, a.input_mode) for k, df in splits.items()}
    all_scores = {k: {} for k in splits}
    chosen = {}
    for feature in a.features:
        fz, clf, info = fit_full_cached(feature, a.classifier, train,
                                        seed=a.seed, input_mode=a.input_mode)
        chosen[feature] = info["C"]
        for name in splits:
            all_scores[name][feature] = scores_of(clf, fz.transform(texts[name]))
        print(f"Scored {feature}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "protocol_version": PROTOCOL_VERSION, "input_mode": a.input_mode,
        "classifier": a.classifier, "seed": a.seed, "selected_C": chosen,
        "status": "corrected_protocol_same_exploratory_holdouts",
        "note": "Problem bootstrap conditions on fixed fitted models; no claim of prospective confirmation.",
        "splits": {},
    }
    for name, df in splits.items():
        score_path = target.with_name(target.stem + f"_{name}_scores.npz")
        if score_path.exists():
            raise FileExistsError(score_path)
        np.savez_compressed(score_path, labels=df.label.to_numpy(),
                            problem_id=df.problem_id.astype(str).to_numpy(dtype=str),
                            **all_scores[name])
        output["splits"][name] = paired_problem_bootstrap(
            df.label.to_numpy(), all_scores[name], df.problem_id.to_numpy(),
            baselines, n_boot=a.n_boot, seed=a.seed)
        output["splits"][name]["scores_file"] = str(score_path)
    with target.open("x") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
