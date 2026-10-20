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

### 本次实际复现记录（DGX Spark / GB10, aarch64, torch 2.9.1+cu130）

- 数据来自 hf-mirror：`tasksource/PRM800K`（与官方 LFS 字节数一致）与
  `peiyi9979/Math-Shepherd`（`--task GSM8K`，schema 与 GitHub 原版不同，
  由 prep 脚本自动识别）。模型下载设 `HF_ENDPOINT=https://hf-mirror.com`。
- E1 按特征 5 进程并行（`--out-suffix`），全量点 `--full-seeds 1`
  （全量无抽样随机性，各种子训练集完全相同），最后合并 5 个分片 JSON 为
  `results/e1_learning_curves_prm800k.json`。
- 全量训练结果经 `fit_full_cached` 在 E1/E2/E3/E4/E5 间共享（数值不变）。
- E3/E4 的 embedding 编码先用 `src/pre_encode.py` 按 md5 分片 6 进程并行灌缓存，
  再跑正式脚本时全部命中缓存。

## 结果摘要（2026-10-20 跑完，种子 0）

**任务**：PRM800K 步骤二分类（对=541,039 / 错=1,223，0.2% 正类的极端不平衡）。
主指标 AUROC；全量训练 352k 步，test 102k 步（按题目分组）。

**E1 · Δ(n) 学习曲线（预测 1：部分成立）**
- 峰值位置 ✓：三种 embedding 对 TF-IDF(word) 的 AUROC 差均在 n=50 附近达峰
  （MiniLM +0.029 / bge-base +0.026，10 种子均值；LR/SVM 一致）。
- "gap 保持非负" ✗：n≥2000 后 bge 系转负；全量端 SVM 下
  tfidf-char 0.704 ≥ bge-large 0.688 ≈ tfidf-word 0.688，MiniLM 0.654。
- 工件注记：bge+LR 全量点 AUROC<0.5（0.2% 正类下 log-loss 的排序退化），
  故正文以 SVM 为主对照、LR 并列给出。

**E2 · 全量表征对比**：embedding − TF-IDF 的 Δ 在全量端为 -0.05~+0.01
（区间跨 0，bootstrap CI 见 results/e2_representation_compare.json）。

**E3 · Δ(偏移)（预测 2）/ E4 · Δ(扰动)（预测 3）/ E5 / E6**：见下节与 results/。

## 合规

- 纯 Python；embedding 模型全程冻结（只前向，无梯度回传）；下游仅逻辑回归 / 线性 SVM。
- 代码为原创；第三方库见 requirements.txt；AI 使用披露见期末报告末节。
