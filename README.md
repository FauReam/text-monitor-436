# 文字监视器 × Project P：Δ(θ) 测量报告

> CSI 436/536 Machine Learning（Fall 2026）· Project Code "P"
> 研究问题（课程固定）：预训练表征是否提升下游 ML 任务，尤其在有限训练数据下。
> 应用载体：便宜的文本监视器——判断数学推理步骤的对错（PRM800K / Math-Shepherd）。

## 核心形式化

固定课上算法 g 与训练数据，唯一变量是特征来源 φ：

**Δ(θ) = L_test(g∘φ_本地简单特征； θ) − L_test(g∘φ_外部预训练； θ)**，θ =（n, 分布偏移, 扰动）

交付物 = Δ(θ) 的测量报告：摘要句（是否 + 效应量 + 区间）+ 三条边界曲线。
防漂移规则：图可画两条线，结论句一律用差值语言。

## 目录

```
├── README.md              本文件
├── PREREGISTRATION.md     预登记：跑实验前锁定的预测与判定标准
├── one-pager.md           中期 one-pager（10/20 交）
├── requirements.txt
├── run_all.sh             一键复现（DGX 上执行）
├── data/                  标准化后的步骤级数据（由 prep 脚本生成）
├── src/
│   ├── utils.py           种子、IO
│   ├── data.py            标准化 schema、按题目分组划分、分层子抽样
│   ├── prep_prm800k.py    原始 PRM800K jsonl → data/prm800k_steps.csv
│   ├── prep_mathshepherd.py  Math-Shepherd jsonl → data/mathshepherd_steps.csv
│   ├── features.py        TF-IDF(word/char) + 冻结 sentence-transformers（带缓存）
│   ├── perturb.py         规则式扰动：typo / 数字格式 / LaTeX 写法
│   ├── metrics.py         balanced acc / AUROC / F1 + bootstrap CI
│   ├── protocol.py        分组 CV 选 C + 训练 + 单次 test 评估的统一协议
│   ├── e1_learning_curves.py   Δ(n)：学习曲线（10 种子分层抽样）
│   ├── e2_representation_compare.py  表征对比（novelty）
│   ├── e3_generalization.py    Δ(偏移)：MATH→GSM8K + train/test 差距
│   ├── e4_robustness.py        Δ(扰动)：规则扰动下的衰减
│   ├── e5_abstention.py        （可选）拒答机制：风险-覆盖率曲线
│   ├── e6_predictability.py    （可选，discussion）廉价统计量 vs 实测 Δ 的相关
│   └── make_figures.py    results/*.json → figures/*.png/pdf
└── slides/
    └── make_slides.py     生成 midterm.pptx / final.pptx（figures/ 有图则嵌入，否则占位框）
```

## 在 DGX 上运行

```bash
# 1. 同步代码后，建环境
pip install -r requirements.txt

# 2. 准备数据（原始 jsonl 路径按实际位置改）
python src/prep_prm800k.py --raw /path/to/phase1_train.jsonl --raw /path/to/phase2_train.jsonl --out data/prm800k_steps.csv
python src/prep_mathshepherd.py --raw /path/to/GSM8K.jsonl --out data/mathshepherd_steps.csv

# 3. 管线保底冒烟测试（SST-2，全自动下载，几分钟）
python src/e1_learning_curves.py --dataset sst2 --quick

# 4. 全量实验（顺序即优先级；E5/E6 可选）
python src/e1_learning_curves.py --dataset prm800k
python src/e3_generalization.py
python src/e4_robustness.py
python src/e2_representation_compare.py
python src/e5_abstention.py        # 可选
python src/e6_predictability.py    # 可选
python src/make_figures.py
python slides/make_slides.py       # 嵌入最新 figures 重新生成 pptx
```

所有脚本固定种子（`--seed`，默认 0）；所有结果写入 `results/`，图写入 `figures/`。

## 合规

- 纯 Python；embedding 模型全程冻结（只前向，无梯度回传）；下游仅逻辑回归 / 线性 SVM。
- 代码为原创；第三方库见 requirements.txt；AI 使用披露见期末报告末节。
