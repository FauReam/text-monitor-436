"""原始 PRM800K jsonl → data/prm800k_steps.csv

原始格式（openai/prm800k GitHub 发布版，phase1/phase2 jsonl）每行：
{
  "question": {"problem": ..., "subject": ..., "level": ..., ...},
  "label": {"steps": [{"completions": [{"text": ..., "rating": +1/0/-1, ...}],
                        "chosen_completion": int | null, ...}], ...}
}
取每个 step 的 chosen completion；rating +1→correct, -1→incorrect, 0→neutral。
若你的 HF 镜像 schema 不同，改成输出同 schema 的 csv 即可，下游无需改动。
"""
import argparse
import json

import pandas as pd

from data import LABEL_MAP, problem_id_of

RATING = {1: "correct", -1: "incorrect", 0: "neutral"}


def iter_rows(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert(paths, out):
    rows = []
    skipped = 0
    for path in paths:
        for ex in iter_rows(path):
            problem = ex["question"]["problem"]
            pid = problem_id_of(problem)
            for st in ex["label"]["steps"]:
                ch = st.get("chosen_completion")
                if ch is None:
                    skipped += 1
                    continue
                comp = st["completions"][ch]
                rating = comp.get("rating")
                if rating not in RATING:
                    skipped += 1
                    continue
                rows.append({
                    "problem_id": pid,
                    "problem_text": problem,
                    "step_text": comp["text"],
                    "label": LABEL_MAP[RATING[rating]],
                })
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} steps ({df['problem_id'].nunique()} problems) "
          f"-> {out}; skipped {skipped}; label counts:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--out", default="data/prm800k_steps.csv")
    a = ap.parse_args()
    convert(a.raw, a.out)
