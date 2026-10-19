#!/bin/bash
# 一键复现（在 DGX 上执行）。数据准备见 README 第 2 步。
set -e
cd "$(dirname "$0")"

echo "== 冒烟测试（SST-2）=="
python src/e1_learning_curves.py --dataset sst2 --quick

echo "== E1 学习曲线 Δ(n)（最优先）=="
python src/e1_learning_curves.py --dataset prm800k

echo "== E3 泛化 Δ(偏移) =="
python src/e3_generalization.py

echo "== E4 鲁棒 Δ(扰动) =="
python src/e4_robustness.py

echo "== E2 表征对比（novelty）=="
python src/e2_representation_compare.py

echo "== E5/E6 可选 =="
python src/e5_abstention.py || true
python src/e6_predictability.py || true

echo "== 出图 + 更新 PPT =="
python src/make_figures.py
python slides/make_slides.py
