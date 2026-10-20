# 中期展示讲稿（5:00）— 逐页演讲词 + 技术解释

> 配套文件：`slides/midterm_talk_5min.html`（按 N 键，本讲稿就在每页注释面板里）。
> 总词量约 685 词，按 145 词/分钟 ≈ 4:43，留 15 秒余量。演讲词照念即可；技术解释是理解和 Q&A 用的，不用念。

| 页 | 内容 | 时段 | 累计 |
|---|---|---|---|
| 1 | 标题：开口即 headline | 0:15 | 0:15 |
| 2 | 内容冲刺：交付要求全覆盖 | 0:30 | 0:45 |
| 3 | 得分点清单（5 项 ✓） | 0:30 | 1:15 |
| 4 | Δ(n) 丑曲线（主图） | 1:15 | 2:30 |
| 5 | 解剖 + 我们怎么处理 | 0:50 | 3:20 |
| 6 | 消融反转（sign flips） | 1:00 | 4:20 |
| 7 | 计划 + 收尾 | 0:25 | 4:45 |
| 8 | backup（不计时，仅 Q&A） | — | — |

---

## 第 1 页 · 标题（0:00–0:15）

**演讲词（约 40 词）**

"The value of pretrained features is regime-dependent: in-distribution, they do not help — when the dataset shifts, they do. We measured this on a cheap text monitor that judges math reasoning steps, as three curves: training size, distribution shift, and perturbation."

**技术解释**

regime（制度）= 我们主动改变的实验条件轴：n（训练样本数）、分布偏移（MATH→GSM8K）、扰动（typo/格式）。headline 就是整篇结论：同分布 Δ 为负（−0.045~−0.064）、偏移下 Δ 为正（+0.020）、扰动下持平（≤0.007）。第一句必须直接给结论（Clarity 分），不铺垫、不念标题、不介绍组员；念完停一拍，用 n / shift / noise 三个词给路线图再翻页。

## 第 2 页 · 内容冲刺（0:15–0:45）

**演讲词（约 80 词）**

"Sixty seconds of everything required. Background: verifying LLM reasoning step-by-step is expensive, so we built a cheap text-only monitor — is this step correct? Setup: we measure delta — pretrained features minus TF-IDF — across three regimes; the only thing we vary is where the features come from. Data: PRM800K, nine hundred eleven thousand human-labeled steps, twenty-eight percent wrong, split grouped by problem. Methods: frozen embeddings, plus logistic regression and linear SVM from class, against TF-IDF. Metrics: balanced, ten seeds, test set touched once."

**技术解释**

便宜监视器：输入=题目+当前步文字，输出=对/错，替代昂贵的 GPT 级逐步查错或人工标注。"only variable"是 Soundness 核心——分类器/数据/协议全固定，唯一变的是特征来源。grouped by problem：同一道题的多条步骤文字高度相似，按步骤随机分=泄漏、指标虚高；我们按题分（train 632,422 / test 190,573）。balanced 指标：错例仅 27.9%，裸 accuracy 误导。每个 n 做 10 种子分层抽样报均值±区间；C 只在训练集内 grouped CV 选；test 只评一次（防多重比较）。数字要念准：911,867 / 27.9% / grouped by problem。

## 第 3 页 · 得分点清单（0:45–1:15）

**演讲词（约 70 词）**

"And the scorecard. Soundness: a leakage-free grouped split, balanced metrics, ten-seed intervals, hyperparameters chosen inside train only. Organization: checklist first, one story after. Clarity: every conclusion is a gap with an interval, never a bare number. Progress: four experiments complete on clean data. Challenges: we shipped one bug, caught it, and re-ran everything. Checklist done — now the number behind the headline."

**技术解释**

这页是元信息：五个评分项各一句证据，评委直接对号入座，每条 4–5 秒匀速过。takeaway（"Checklist done…"）放慢加重，停一拍再翻页——全场唯一一次"推销"。若被问为什么这样组织：先证明覆盖度，再把 3 分钟投给信息量最大的部分（负结果+消融），而不是平均分配。

## 第 4 页 · Δ(n) 丑曲线（1:15–2:30，全场最慢）

**演讲词（约 165 词）**

"Here is the honest curve. The textbook expectation: pretrained features should help most when training data is scarce — the sample-efficiency story. What we measured is the opposite. At n equals twenty, everything sits at zero point five one — near random — for every feature; there is no signal to harvest. From fifty on, the gap — embedding minus TF-IDF — turns negative and stays negative, monotone, ten seeds out of ten. At full data, TF-IDF beats every embedding by five AUROC points or more, and the ninety-five percent interval excludes zero. Read the gap, not the levels. This is not a failed experiment — it localizes where pretrained features do not pay: in-distribution, with abundant data, on a narrow task."

**技术解释**

图读法：左图 x=n(log)、y=ΔAUROC(embedding−TF-IDF)，0 以下为负，三条线=三个 embedding；右图是 n=20–500 的配对差值放大（LR/SVM 两面板）。n=20 全≈0.51：20 个样本学不出任何可泛化决策面，"小样本优势"无从谈起——不是 embedding 赢不了，是谁都没学出来。配对差值=同一种子内两组特征各训一个分类器再相减，消除抽样方差；n=50/100/500 单点 CI 跨 0，但 10/10 种子同号且 4 点单调——方向一致、单点功效不足，照实说。全量端 −0.045~−0.064 是 SVM 口径；LR 口径 −0.037 起——被追问给口径。手要指图。

## 第 5 页 · 解剖 + 处理（2:30–3:20）

**演讲词（约 125 词）**

"Why does the baseline win? Three reasons. This is one narrow domain — the signal lives in tokens like 'therefore', equals signs, LaTeX fragments — and similarity-trained embeddings blur exactly those tokens. Second: with six hundred thirty-two thousand training steps, sparse features are not starving; small data is not this task's bottleneck. Third: embeddings were pretrained for meaning, not for correctness — an objective mismatch. And why do we trust the result? Because the two wrong-looking numbers on the way here were handled. The first was our fault: our label construction kept only chosen completions — ninety-nine point eight percent positive. The metrics degenerated, we caught it, and re-ran everything. The second — the negative gap — survived every check we could throw at it."

**技术解释**

必问"in-domain 为什么 TF-IDF 反而赢"：步级对错判断是窄任务，char n-gram 基线本来强；embedding 优势在词面重叠失效时（跨数据集）才体现——下一页就是证据。泄漏自查：首版按步骤随机分，同题步骤出现在两侧→指标虚高；改按题分组后回落。标签 bug：v1 只取 chosen completion→正类 99.8%、指标退化（接近常数分类器）被我们发现；改用全部带人工 rating 的 completions 重跑=v2，今天所有数字都是 v2。不平衡补查：50/50 重采样与 class_weight=balanced 两路，AUROC 变化均 <0.01——结论不由类别比例驱动。

## 第 6 页 · 消融反转（3:20–4:20，全场高潮）

**演讲词（约 145 词）**

"Now the turn. We change the regime — train on MATH, test on GSM8K — and the sign flips. bge-large beats TF-IDF by plus zero point zero two zero, with a tight interval. bge-base is plus zero point zero zero six. MiniLM is nothing. A clean size gradient — not a fluke. The mechanism is in the overfitting numbers: TF-IDF's train-test gap is three times larger — it wins in-domain by memorizing surface tokens, and memorization does not travel. Perturbations: both degrade less than zero point zero zero seven — parity, we claim nothing there. Selective prediction: TF-IDF is still marginally better — reported as-is. So the refined thesis: pretrained features are not most valuable when data is scarce — they are most valuable when the distribution drifts."

**技术解释**

paired bootstrap（测试集内 1000 次重抽样）：bge-large Δ=+0.020 CI[0.018,0.021]、bge-base +0.006 CI[0.004,0.008]、MiniLM −0.001 CI 跨 0——增益随 embedding 尺寸单调=规模梯度。机制：train−test gap TF-IDF 0.153–0.168 vs embedding 0.040–0.065（≈3×），同分布赢靠记忆词面、记忆不可迁移。复合偏移披露：MATH→GSM8K 题型+来源+标签方式同时变（人工→自动），只做复合归因，不声称单因素——被追问主动说。E4 规则扰动（typo/数字格式/LaTeX）：衰减均 ≤0.007、embedding≤TF-IDF，诚实 null。E5 拒答（Chow 规则：置信度低于阈值拒答）：risk@满覆盖 TF-IDF 0.271 略优，低覆盖端都压到 ≈0.05——一句话带过不遮掩。

## 第 7 页 · 计划 + 收尾（4:20–4:45）

**演讲词（约 65 词）**

"For the final: we have started E6 — predicting the downstream gap from properties of the representation itself; correlation zero point nine on four points, preliminary. We widen the shift grid to map exactly where the sign flips. The deliverable is three delta curves, written as a measurement report. We asked whether pretrained representations help. The answer is not yes or no — it is a map. We drew the first three coordinates. Thank you."

**技术解释**

E6：用廉价可算的表征统计量（训练集散度等）预测实测 Δ——Spearman ρ=0.900、p=0.037 但 n=4 点，必须带 preliminary；被问就说点数少、期末扩"表征×制度"网格。期末交付：Δ(n)、Δ(偏移)、Δ(扰动) 三条曲线+摘要效应量的测量报告。若被问 AUROC 绝对值为何不高：步级对错本身难，我们报的是特征来源间的 Δ 而非绝对水平。理想节奏 4:45 讲完，留 15 秒余量。

## 第 8 页 · backup（不计时）

无演讲词，被问到才翻。屏上已有：全数字表 + 四个预演问题（in-domain 为何 TF-IDF 赢 / 为何不随机分 / AUROC 为何不高 / 复合偏移）。AI 使用被问时照实答：实现代码由 AI 在我组指导下编写；研究框架、实验设计、数据检查与消融设计由本组完成；我们已通读全部代码，能逐行解释。

---

## 排练要点

- 掐表两遍：第 7 页讲完应在 4:45 ± 0:10
- 三处停顿：第 1 页 headline 后、第 3 页 "Checklist done" 后、第 6 页 refined thesis 逐字念
- 数字必念准：911,867 · 27.9% · −0.045~−0.064 · +0.020 [0.018, 0.021] · 0.153 vs 0.040
- 禁用词："失败"——一律说 "negative result / boundary localization"
