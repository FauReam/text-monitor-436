"""预编码工具：把全部实验所需的文本一次性写入 embedding 缓存（data/cache/）。

覆盖：E1/E2/E5/E6 的 train+test、E3 迁移测试集、E4 的四种扰动测试集。

用法（逐模型串行；同一 GPU 上多进程编码会严重互相拖慢，勿并行）：
    for m in st-all-MiniLM-L6-v2 st-bge-base-en-v1.5 st-bge-large-en-v1.5; do
        python src/pre_encode.py --model $m
    done

编码完成后，E1–E6 的 transform 全部命中缓存。
（保留 --shard/--n-shards 仅用于多机/多 GPU 场景；单机单卡必须 n-shards=1。）
"""
import argparse
import hashlib

from data import group_split, input_texts, load_steps
from features import _encode_st
from perturb import PERTURBATIONS
from utils import set_seed


def all_texts(seed: int = 0):
    """E1–E6 全部待编码文本（与各脚本里的构造方式严格一致）。"""
    df = load_steps("prm800k")
    train_df, _, test_df = group_split(df, seed=seed)
    texts = list(input_texts(train_df)) + list(input_texts(test_df))
    texts += list(input_texts(load_steps("mathshepherd")))
    for pname, pfun in PERTURBATIONS.items():
        if pname == "none":
            continue
        texts.extend(pfun(t, seed=seed) for t in input_texts(test_df))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512)
    a = ap.parse_args()
    set_seed(0)

    texts = all_texts()
    mine = [t for t in set(texts)
            if int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % a.n_shards
            == a.shard]
    print(f"{a.model} shard {a.shard}/{a.n_shards}: {len(mine)} unique texts",
          flush=True)
    _encode_st(a.model, mine, batch_size=a.batch_size)
    print(f"{a.model} shard {a.shard} done", flush=True)


if __name__ == "__main__":
    main()
