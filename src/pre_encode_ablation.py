"""E7 消融的预编码：step+problem 与 step-only 两种输入（PRM800K train+test）。

同一 GPU 多进程编码会严重互相拖慢：逐模型串行执行。
    for m in st-all-MiniLM-L6-v2 st-bge-base-en-v1.5 st-bge-large-en-v1.5; do
        ST_FP16=1 python src/pre_encode_ablation.py --model $m
    done
"""
import argparse

from data import group_split, input_texts, load_steps
from features import _encode_st
from utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    a = ap.parse_args()
    set_seed(0)

    df = load_steps("prm800k")
    train_df, _, test_df = group_split(df, seed=0)
    texts = set()
    for mode in ("step+problem", "step"):
        texts.update(input_texts(train_df, mode))
        texts.update(input_texts(test_df, mode))
    texts = sorted(texts)
    print(f"{a.model}: {len(texts)} unique texts", flush=True)
    _encode_st(a.model, texts, batch_size=a.batch_size)
    print(f"{a.model} done", flush=True)


if __name__ == "__main__":
    main()
