"""原始 PRM800K jsonl → data/prm800k_steps.csv

原始格式（openai/prm800k GitHub 发布版，phase1/phase2 jsonl）每行：
{
  "question": {"problem": ..., "subject": ..., "level": ..., ...},
  "label": {"steps": [{"completions": [{"text": ..., "rating": +1/0/-1, ...}],
                        "chosen_completion": int | null, ...}], ...}
}
取每个 step 的全部带 rating 的 completions（不能只取 chosen completion——
chosen 几乎全是正类，会导致正类占比 ~99.8% 的标签退化）。
rating +1→correct, -1→incorrect, 0→neutral；无 rating 的跳过。
--max-per-step > 0 时截断每步 completions 数（保留全部负/中性例，正例按序截断）。
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


def convert(paths, out, max_per_step: int = 0):
    rows = []
    skipped = 0
    for path in paths:
        for ex in iter_rows(path):
            problem = ex["question"]["problem"]
            pid = problem_id_of(problem)
            for st in ex["label"]["steps"]:
                raw_comps = st.get("completions") or []  # human_completion 步骤为 None
                comps = [c for c in raw_comps if c.get("rating") in RATING]
                skipped += len(raw_comps) - len(comps)
                if max_per_step > 0 and len(comps) > max_per_step:
                    neg = [c for c in comps if c["rating"] <= 0]
                    pos = [c for c in comps if c["rating"] > 0]
                    comps = neg + pos[:max(0, max_per_step - len(neg))]
                for comp in comps:
                    text = (comp.get("text") or "").strip()
                    if not text:
                        skipped += 1
                        continue
                    rows.append({
                        "problem_id": pid,
                        "problem_text": problem,
                        "step_text": text,
                        "label": LABEL_MAP[RATING[comp["rating"]]],
                    })
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} steps ({df['problem_id'].nunique()} problems) "
          f"-> {out}; skipped {skipped}; label counts:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--out", default="data/prm800k_steps.csv")
    ap.add_argument("--max-per-step", type=int, default=0,
                    help=">0 时每步最多保留 N 条 completions（负/中性例全保留）")
    a = ap.parse_args()
    convert(a.raw, a.out, a.max_per_step)
