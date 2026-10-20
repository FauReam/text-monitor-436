"""预编码工具：把 E3（迁移测试集）/ E4（扰动测试集）需要的全部文本
按内容哈希分片，多进程并行写入 embedding 缓存（data/cache/*.npy）。

用法（例：2 个 bge 模型 × 3 分片 = 6 个并行进程）：
    python src/pre_encode.py --model st-bge-large-en-v1.5 --shard 0 --n-shards 3

所有分片跑完后，E3/E4 的 transform 全部命中缓存，分钟级完成。
分片按 md5(text) 取模：同一文本只进一个分片，进程间无写冲突。
"""
import argparse
import hashlib

from data import group_split, input_texts, load_steps
from features import _encode_st
from perturb import PERTURBATIONS
from utils import set_seed


def all_aux_texts(seed: int = 0):
    """E3/E4 全部待编码文本（与两脚本里的构造方式严格一致）。"""
    df = load_steps("prm800k")
    _, _, test_df = group_split(df, seed=seed)
    transfer = load_steps("mathshepherd")
    texts = list(input_texts(transfer))
    for pname, pfun in PERTURBATIONS.items():
        if pname == "none":
            continue
        texts.extend(pfun(t, seed=seed) for t in input_texts(test_df))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    a = ap.parse_args()
    set_seed(0)

    texts = all_aux_texts()
    mine = [t for t in set(texts)
            if int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % a.n_shards
            == a.shard]
    print(f"shard {a.shard}/{a.n_shards}: {len(mine)} unique texts", flush=True)
    _encode_st(a.model, mine, batch_size=a.batch_size)
    print(f"shard {a.shard} done", flush=True)


if __name__ == "__main__":
    main()
