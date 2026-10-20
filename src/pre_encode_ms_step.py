"""预编码：Math-Shepherd 迁移集的 step-only 文本（E3 输入消融用）。

单 GPU 串行：逐模型执行。
    for m in st-all-MiniLM-L6-v2 st-bge-base-en-v1.5 st-bge-large-en-v1.5; do
        HF_HUB_OFFLINE=1 ST_FP16=1 python src/pre_encode_ms_step.py --model $m
    done
"""
import argparse

from data import input_texts, load_steps
from features import _encode_st
from utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    a = ap.parse_args()
    set_seed(0)

    texts = sorted(set(input_texts(load_steps("mathshepherd"), "step")))
    print(f"{a.model}: {len(texts)} unique texts", flush=True)
    _encode_st(a.model, texts, batch_size=a.batch_size)
    print(f"{a.model} done", flush=True)


if __name__ == "__main__":
    main()
