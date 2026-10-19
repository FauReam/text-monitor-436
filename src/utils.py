"""共用工具：种子、IO、路径。"""
import json
import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
CACHE = os.path.join(ROOT, "data", "cache")

for _d in (DATA, RESULTS, FIGURES, CACHE):
    os.makedirs(_d, exist_ok=True)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_json(obj, name: str) -> str:
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def load_json(name: str):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)
