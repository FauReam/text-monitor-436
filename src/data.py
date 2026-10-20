"""标准化数据层。

统一 schema（步骤级二分类）：
    problem_id   题目分组 id（同一道题的所有步骤共享，划分时绝不跨集合）
    problem_text 题目文本
    step_text    当前步骤文字
    label        1 = correct, -1 = incorrect, 0 = neutral（主实验丢弃 0）

输入定义（主设定）：problem_text + step_text；"仅当前步"作消融。
"""
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from utils import DATA

LABEL_MAP = {"correct": 1, "incorrect": -1, "neutral": 0}


def problem_id_of(problem_text: str) -> str:
    return hashlib.md5(problem_text.strip().encode("utf-8")).hexdigest()[:12]


def load_steps(name: str, drop_neutral: bool = True) -> pd.DataFrame:
    df = pd.read_csv(f"{name}" if name.endswith(".csv") else
                     __import__("os").path.join(DATA, f"{name}_steps.csv"))
    if drop_neutral:
        df = df[df["label"] != 0].reset_index(drop=True)
    return df


def input_texts(df: pd.DataFrame, mode: str = "problem+step") -> list:
    if mode == "problem+step":
        return (df["problem_text"].fillna("") + "\n" + df["step_text"]).tolist()
    if mode == "step":
        return df["step_text"].tolist()
    raise ValueError(mode)


def group_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1,
                seed: int = 0):
    """按题目分组划分 train/val/test；同一 problem_id 不跨集合。"""
    gss = GroupShuffleSplit(1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, groups=df["problem_id"]))
    trainval, test = df.iloc[tr_idx], df.iloc[te_idx]
    rel_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(1, test_size=rel_val, random_state=seed)
    tr2, va2 = next(gss2.split(trainval, groups=trainval["problem_id"]))
    return (trainval.iloc[tr2].reset_index(drop=True),
            trainval.iloc[va2].reset_index(drop=True),
            test.reset_index(drop=True))


def subsample_stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """分层（按 label）抽 n 条；n 小于类别数时保证每类至少 1 条。"""
    rng = np.random.RandomState(seed)
    classes = df["label"].unique()
    per = max(1, n // len(classes))
    parts = []
    for c in classes:
        pool = df[df["label"] == c]
        k = min(per, len(pool))
        parts.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
    out = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return out.reset_index(drop=True)


def load_sst2() -> pd.DataFrame:
    """保底管线：SST-2 转成本 schema（problem_id 逐行唯一，无分组泄漏问题）。"""
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    rows = []
    for split in ("train", "validation"):
        for i, r in enumerate(ds[split]):
            rows.append({
                "problem_id": f"sst2-{split}-{i}",
                "problem_text": "",
                "step_text": r["sentence"],
                "label": 1 if r["label"] == 1 else -1,
            })
    return pd.DataFrame(rows)
