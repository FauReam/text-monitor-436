"""Math-Shepherd jsonl → data/mathshepherd_steps.csv

原始格式（peiyi9979/Math-Shepherd，GSM8K.jsonl / MATH.jsonl）每行：
{"input": 题目, "label": "step1ки+step2ки- ...  "}
每个步骤以 "ки" 结尾，后接 +（好步骤）或 -（坏步骤）。
映射：+→correct，-→incorrect。无 neutral。
若你的镜像 schema 不同，改成输出同 schema 的 csv 即可。
"""
import argparse
import json
import re

import pandas as pd

from data import LABEL_MAP, problem_id_of


def parse_trajectory(problem: str, traj: str):
    parts = traj.split("ки")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.*)([+-])\s*$", part, flags=re.S)
        if not m:
            continue
        text, sign = m.group(1).strip(), m.group(2)
        if text:
            yield {
                "problem_id": problem_id_of(problem),
                "problem_text": problem,
                "step_text": text,
                "label": LABEL_MAP["correct" if sign == "+" else "incorrect"],
            }


def convert(paths, out):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                rows.extend(parse_trajectory(ex["input"], ex["label"]))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} steps ({df['problem_id'].nunique()} problems) -> {out}; "
          f"label counts:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--out", default="data/mathshepherd_steps.csv")
    a = ap.parse_args()
    convert(a.raw, a.out)
