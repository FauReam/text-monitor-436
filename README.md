# 文字监视器 × Project P：Δ(θ) 测量报告

> CSI 436/536 Machine Learning（Fall 2026）· Project Code "P"
> 研究问题（课程固定）：预训练表征是否提升下游 ML 任务，尤其在有限训练数据下。
> 应用载体：便宜的文本监视器——判断数学推理步骤的对错（PRM800K / Math-Shepherd）。

## 核心形式化

固定课上算法 g 与训练数据，唯一变量是特征来源 φ：

**Δ(θ) = L_test(g∘φ_本地简单特征； θ) − L_test(g∘φ_外部预训练； θ)**，θ =（n, 分布偏移, 扰动）

符号约定：正文与图表报告的差值均为 AUROC 差（embedding − TF-IDF），相当于上式取
L = 1 − AUROC——**Δ > 0 表示预训练表征更好，Δ < 0 表示 TF-IDF 更好**。

交付物 = Δ(θ) 的测量报告：摘要句（是否 + 效应量 + 区间）+ 三条边界曲线。
防漂移规则：图可画两条线，结论句一律用差值语言。

## 目录

```
├── README.md              本文件
├── PREREGISTRATION.md     预登记：跑实验前锁定的预测与判定标准
├── one-pager.md           中期 one-pager（10/20 交；正式版为 one-pager.tex/pdf）
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
│   ├── check_truncation.py     截断排查：输入 token 长度分布与被截比例
│   ├── e7_input_ablation.py    输入消融：problem+step vs step-first vs step-only
│   ├── pre_encode.py / pre_encode_ablation.py  embedding 缓存预灌工具
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
- bge-large 的 E3/E4 尾部文本以 fp16 编码（`ST_FP16=1`，与 fp32 的逐向量余弦 > 0.9997，
  不影响 AUROC 结论；见 `src/features.py`）。同 GPU 多进程并发编码吞吐骤降，
  正式编码一律单进程串行。
- 中期 one-pager 的 LaTeX 源与 PDF 为 `one-pager.tex` / `one-pager.pdf`。

## 结果摘要（v2 标签修复版，2026-10-20 跑完）

**任务**：PRM800K 步骤二分类——1,015,027 步 / 10,825 题，二分类 911,867 行（错例占 27.9%）。
按题目分组划分（seed=0）：train 632,422 / val 88,872 / test 190,573 步
（val 由 group_split 预留，本轮实验未使用——C 选择在 train 内部 GroupKFold 完成）。
主指标 AUROC。
迁移测试集：Math-Shepherd（GSM8K，613,666 步，47%/53% 近平衡）。

> 首版（v1）结果基于错误的标签构造（只取 chosen completion，正类占 99.8% 退化），
> 已全部作废；本节全部为修复后（取全部带人工 rating 的 completions）的重跑结果。

**E1 · Δ(n) 学习曲线（预登记预测 1：不成立）**
- n=20 时所有特征（含 TF-IDF）均 ≈ 0.51 AUROC，近随机、区间重叠——预训练没有买到小样本优势。
- n ≥ 50 起 TF-IDF 反超并保持到全量；embedding − TF-IDF(word) 的 Δ（SVM，10 种子均值）
  从 −0.003（n=50）单调走负到 −0.045（bge-large，full）；MiniLM 全量端 −0.064。
- 全量 AUROC（SVM）：tfidf-word 0.696 / tfidf-char 0.687 / bge-large 0.650 / bge-base 0.650 / MiniLM 0.631。

**E2 · 全量表征对比（带 bootstrap CI）**：embedding 全量端显著为负，
Δ = −0.037 ~ −0.064（bge-large − tfidf-word，SVM = −0.045，CI 不含 0；
逐特征 CI 见 results/e2_representation_compare.json）。

**E3 · Δ(偏移)（预登记预测 2：部分成立）**
- 同分布端 TF-IDF 更高（0.696 vs 0.650），迁移端（MATH→GSM8K）反转：
  bge-large 0.571 > tfidf-word 0.551。
- paired bootstrap（1000 次，SVM，results/e3_transfer_delta_ci.json）：
  bge-large Δ = +0.020，CI[0.018, 0.021]（显著正）；
  bge-base +0.006，CI[0.004, 0.008]（显著正）；
  MiniLM −0.001，CI[−0.002, 0.001]（跨 0，不显著）。
- 退化幅度：tfidf-word −0.145 vs bge-large −0.080——embedding 更耐偏移；
  train−test gap 同样悬殊：TF-IDF 0.153–0.168 vs embedding 0.040–0.065。
- 注意：MATH→GSM8K 是复合偏移（题型 + 来源 + 标签方式同时变），不作单因素归因。
- **step-only 输入下（E7 的更优构造，results/e3_generalization_step.json，SVM）**：
  bge-large 迁移端 0.595 vs tfidf-word 0.527，Δ 从 +0.020 放大到 **+0.069**；
  paired bootstrap（1000 次，results/e3_step_transfer_delta_ci.json）：
  bge-large +0.069 CI[0.067, 0.071]、bge-base +0.049 CI[0.048, 0.051]、
  MiniLM +0.030 CI[0.029, 0.032]——**三个 embedding 全部显著为正**（MiniLM 也转正）。
  退化幅度 tfidf-word −0.181 vs bge-large −0.116。
  TF-IDF 迁移端反而更差（0.551 → 0.527）——题目文本恰含可迁移的词汇线索，
  去掉后 TF-IDF 更依赖 MATH 特有表面特征，embedding 则受益于更聚焦的语义表征。
  "偏移下 embedding 更好"在更干净的输入构造下更强，而非更弱。

**E4 · Δ(扰动)（预登记预测 3：成立，效应量小）**
- 规则扰动（typo / 数字格式 / LaTeX / 组合）下所有特征衰减 ≤ 0.007 AUROC。
- latex / combo 下 embedding 衰减（≤ 0.002，个别为负即扰动后反而略升）
  ≤ TF-IDF（tfidf-char SVM 0.005 / 0.007）。

**E5 · 拒答机制（可选）**：risk@coverage=1.0 下 TF-IDF 0.271 略优于 embedding
（0.276–0.278）；低覆盖端各特征均把 risk 压到 ≈ 0.05（Chow 规则有效），TF-IDF 优势保持。

**E6 · 可预测性（可选，discussion）**：训练集散度统计量与实测 Δ 同号，
Spearman ρ = 0.900（p = 0.037，n = 4 点）——仅作初步观察 / future work。

**稳健性补查**（回应"训练集 28/72 不平衡"的质疑）：50/50 重采样平衡训练集与
`class_weight='balanced'` 两种处理下，各特征 AUROC 变化均 < 0.01
（results/robustness_balance_check.json）——上述结论不由类别比例驱动。

**E7 · 输入构造消融（评委意见驱动，重要修正）**

挑战（评委意见）：主设定输入 = problem + "\n" + step，step 在尾部——
(a) 长题目会把 step 截断（MiniLM 上限 256 / bge 512 token）；
(b) 题目文本可能稀释 step 的表征。"同分布 Δ < 0"到底是特征本身还是输入构造？

- 截断量化（results/truncation_check.json，src/check_truncation.py）：
  输入中位 89 token（p99 = 291）；MiniLM 截断 1.64% 的行（0.56% 丢失整个 step），
  bge 仅 0.04%，Math-Shepherd ≈ 0。**截断真实存在但量级太小，单独不足以解释差距。**
- 消融（results/input_ablation.json，src/e7_input_ablation.py，E1 全量点，
  test AUROC，SVM）：

  | 特征 | problem+step | step-first | step-only |
  |---|---|---|---|
  | tfidf-word | 0.6958 | 0.6944 | 0.7077 |
  | tfidf-char | 0.6873 | 0.6872 | 0.7037 |
  | MiniLM | 0.6314 | 0.6467 | 0.6810 |
  | bge-base | 0.6498 | 0.6680 | 0.7037 |
  | bge-large | 0.6505 | 0.6769 | **0.7114** |

- step 前置（只修截断/位置）对 embedding +0.015 ~ +0.026，对 TF-IDF ≈ 0；
  去掉题目（step-only）对 embedding 再 +0.03 以上，对 TF-IDF 仅 +0.012 ~ +0.016。
  即主因是 (b) 稀释效应——题目文本主导了 embedding 的均值池化/CLS，截断贡献很小。
- **结论修正**：同分布端的 Δ < 0 大半是输入构造的产物而非表征本身——
  step-only 下 bge-large（0.7114）反超 TF-IDF（0.7077），Δ 由 −0.045 转为 +0.004。
  "预训练表征是否提升"的答案因此对输入构造高度敏感，这是本次测量最重要的边界之一。

**当前主要挑战**（剩余未解）：E7 已回答"同分布 Δ<0 的主因"（输入构造的稀释效应，
非截断、非表征本身）；遗留问题——(a) PRM800K 的 step 可能自带上文关键信息，
step-only 为何反而更优值得深挖；(b) 28/72 类别比例（已有平衡补查排除）；
(c) 复合偏移的单因素分解仍未做。

三条预登记预测最终判定：**预测 1 不成立 · 预测 2 部分成立 · 预测 3 成立（效应量小）**。
一句话结论：主设定（problem+step 输入）下同分布 Δ < 0，但 E7 消融表明这大半是
输入构造的产物——step-only 下 bge-large 反超 TF-IDF；**在分布偏移下预训练表征
出现小幅但显著的收益**（主设定 bge 系 +0.006 ~ +0.020；step-only 下三个 embedding
全部显著为正，+0.030 ~ +0.069）。
预训练特征的价值主要体现在"分布会漂"的制度，而非"数据少"的制度；
且"是否提升"的答案对输入构造高度敏感。

## 合规

- 纯 Python；embedding 模型全程冻结（只前向，无梯度回传）；下游仅逻辑回归 / 线性 SVM。
- 代码为原创；第三方库见 requirements.txt。
- **AI 使用披露（§4.8）**：实现代码由 AI 助手在我组指导下编写（另用于文字润色）；研究框架、实验设计、数据检查与消融实验设计均由本组完成。我们已通读全部代码，能够逐行解释。期末报告末节附同样声明。
