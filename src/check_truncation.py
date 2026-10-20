"""截断排查：输入 = problem + "\n" + step，step 在最后。

各模型的 SentenceTransformer max_seq_length 上限（读自 sentence_bert_config.json）：
    all-MiniLM-L6-v2 = 256，bge-base/large-en-v1.5 = 512。
若 problem+step 超长，尾部（即被判断的 step）被截掉 → 人为压低 embedding 表现。

本脚本只分词、不编码，输出 results/truncation_check.json：
    每 (数据集 × 模型)：总长分位数、超限比例、step token 丢失比例的分位数。
用法：HF_HUB_OFFLINE=1 python src/check_truncation.py
"""
import json
import os

import numpy as np
from transformers import AutoTokenizer

from data import load_steps
from features import ST_MODELS

MAX_LEN = {
    "st-all-MiniLM-L6-v2": 256,
    "st-bge-base-en-v1.5": 512,
    "st-bge-large-en-v1.5": 512,
}

DATASETS = ["prm800k", "mathshepherd"]


def batched_lengths(tok, texts, batch=4096):
    """整批分词，返回每条文本的 token 数（含 special tokens）。"""
    out = np.empty(len(texts), dtype=np.int32)
    for lo in range(0, len(texts), batch):
        enc = tok(texts[lo:lo + batch], add_special_tokens=True,
                  return_length=True, truncation=False)
        out[lo:lo + batch] = enc["length"]
    return out


def pct(x, q):
    return round(float(np.percentile(x, q)), 4)


def main():
    report = {}
    for ds in DATASETS:
        df = load_steps(ds)
        prob = df["problem_text"].fillna("").tolist()
        step = df["step_text"].fillna("").tolist()
        joint = [p + "\n" + s for p, s in zip(prob, step)]
        report[ds] = {"n_rows": len(df)}
        for name, hf_id in ST_MODELS.items():
            tok = AutoTokenizer.from_pretrained(hf_id)
            max_len = MAX_LEN[name]
            l_joint = batched_lengths(tok, joint)
            l_step = batched_lengths(tok, step)
            over = l_joint > max_len
            # step 在最后，所有被截 token 都来自 step
            lost = np.maximum(0, l_joint - max_len)
            frac_lost = lost / np.maximum(l_step, 1)
            report[ds][name] = {
                "max_seq_length": max_len,
                "joint_len": {q: pct(l_joint, q) for q in (50, 90, 95, 99, 100)},
                "step_only_len": {q: pct(l_step, q) for q in (50, 90, 95, 99, 100)},
                "pct_rows_truncated": round(float(over.mean() * 100), 4),
                "pct_rows_lose_step_token": round(float((lost > 0).mean() * 100), 4),
                "pct_rows_lose_gt10pct_step": round(float((frac_lost > 0.1).mean() * 100), 4),
                "pct_rows_lose_gt50pct_step": round(float((frac_lost > 0.5).mean() * 100), 4),
                "pct_rows_lose_entire_step": round(float((frac_lost >= 1.0).mean() * 100), 4),
                "step_frac_lost_median_among_truncated": (
                    pct(frac_lost[over], 50) if over.any() else 0.0),
            }
            print(ds, name, "truncated %:", report[ds][name]["pct_rows_truncated"],
                  flush=True)
    out = os.path.join("results", "truncation_check.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("saved", out)


if __name__ == "__main__":
    main()
