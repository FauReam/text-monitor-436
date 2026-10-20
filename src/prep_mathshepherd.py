"""Math-Shepherd jsonl → data/mathshepherd_steps.csv

支持两种原始格式（逐行自动识别）：
1. 原版 GitHub 发布（GSM8K.jsonl / MATH.jsonl）每行：
   {"input": 题目, "label": "step1ки+step2ки- ...  "}
   每个步骤以 "ки" 结尾，后接 +（好步骤）或 -（坏步骤）。
2. HF 镜像（peiyi9979/Math-Shepherd, math-shepherd.jsonl）每行：
   {"input": 题目 + " Step 1: ...ки Step 2: ...ки",
    "label": 题目 + " Step 1: ... +\nStep 2: ... -", "task": "GSM8K"|"MATH"}
   步骤分隔符 "\\nStep k:"，每步以 +/- 结尾；"Step k:" 前缀不入 step_text。

映射：+→correct，-→incorrect。无 neutral。
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


_STEP_RE = re.compile(r"(?=Step \d+:)")


def parse_hf_record(ex):
    """peiyi9979/Math-Shepherd 镜像格式：label = 题目 + "Step 1: ... +\nStep 2: ... -"。"""
    label = ex["label"]
    m = re.search(r"\bStep 1:", label)
    if not m:
        return
    problem = label[:m.start()].strip()
    chunks = _STEP_RE.split(label[m.start():])
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        m2 = re.match(r"^Step \d+:\s*(.*)\s*([+-])\s*$", ch, flags=re.S)
        if not m2:
            continue
        text, sign = m2.group(1).strip(), m2.group(2)
        if text:
            yield {
                "problem_id": problem_id_of(problem),
                "problem_text": problem,
                "step_text": text,
                "label": LABEL_MAP["correct" if sign == "+" else "incorrect"],
            }


def convert(paths, out, task=None):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if task and ex.get("task") not in (None, task):
                    continue
                if "task" in ex:  # HF 镜像格式
                    rows.extend(parse_hf_record(ex))
                else:
                    rows.extend(parse_trajectory(ex["input"], ex["label"]))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} steps ({df['problem_id'].nunique()} problems) -> {out}; "
          f"label counts:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--out", default="data/mathshepherd_steps.csv")
    ap.add_argument("--task", default="GSM8K",
                    help="只保留该 task（HF 镜像格式有效；默认 GSM8K）")
    a = ap.parse_args()
    convert(a.raw, a.out, task=a.task)
