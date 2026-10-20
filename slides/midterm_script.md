# 中期展示讲稿（5:00）— 逐页演讲词 + 技术解释（E7 主线版）

> 配套文件：`slides/midterm_talk_5min.html`（按 N 键，本讲稿就在每页注释面板里）。
> 总词量约 690 词 ≈ 4:40，留 20 秒余量。演讲词全部是可照念的完整口语句；技术解释供理解和 Q&A，不用念。
> **主线（评委意见后改版）**：第一答案"同分布没用"→ 评委问"是特征还是输入"→ E7 消融证明主因是输入稀释 → 修输入后同分布追平、迁移收益翻三倍。

| 页 | 内容 | 时段 | 累计 |
|---|---|---|---|
| 1 | 标题：headline（第一答案被推翻） | 0:15 | 0:15 |
| 2 | 内容冲刺：交付要求全覆盖 | 0:30 | 0:45 |
| 3 | 得分点清单（5 项 ✓） | 0:30 | 1:15 |
| 4 | 第一答案：Δ(n) 为负（旧输入构造） | 1:15 | 2:30 |
| 5 | 解剖：特征还是输入？（E7 消融） | 0:50 | 3:20 |
| 6 | 反转放大：step-only 迁移 Δ 翻三倍 | 1:00 | 4:20 |
| 7 | 计划 + 收尾 | 0:25 | 4:45 |
| 8 | backup（不计时，仅 Q&A） | — | — |

---

## 第 1 页 · 标题（0:00–0:15）

**演讲词（约 55 词，语速中快）**

"Do pretrained features help a cheap text monitor that judges math reasoning steps? Our first answer was no — not in-distribution. That answer was mostly an artifact of how we fed the input, not of the representations. Fixing the input closed the in-domain gap — and tripled the payoff when the distribution shifts."

**技术解释**

必背三组数：−0.045~−0.064（旧构造同分布，显著）、+0.004（step-only 同分布追平，**点估计不报显著**）、+0.069 [0.067, 0.071]（step-only 迁移，显著）。第一句直接给结论，念完停一拍再翻页。

## 第 2 页 · 内容冲刺（0:15–0:45）

**演讲词（约 95 词，语速稍快）**

"First, sixty seconds of everything required. Verifying a language model's reasoning step by step is expensive, so we built a cheap monitor that reads one math step and judges it right or wrong. We measure the gap between pretrained features and TF-IDF across three regimes, plus an input ablation — and the only thing we vary is where the features come from. The data is PRM800K, nine hundred eleven thousand human-labeled steps, twenty-eight percent wrong, split grouped by problem. The learners are frozen embeddings plus logistic regression and linear SVM, against TF-IDF. Every metric is balanced, over ten seeds, test set touched once."

**技术解释**

便宜监视器：输入=题目+当前步文字（主设定），输出=对/错。"only variable"是 Soundness 核心——分类器/数据/协议全固定，唯一变的是特征来源。grouped by problem：同一道题的多条步骤文字高度相似，按步骤随机分=泄漏、指标虚高；按题分（train 632,422 / test 190,573）。balanced 指标：错例仅 27.9%，裸 accuracy 误导。10 种子分层抽样；C 只在训练集内 grouped CV 选；test 只评一次。数字要念准：911,867 / 27.9% / grouped by problem。

## 第 3 页 · 得分点清单（0:45–1:15）

**演讲词（约 70 词）**

"And our scorecard. For soundness, a leakage-free grouped split, balanced metrics, ten-seed intervals, hyperparameters chosen inside train only. For clarity, every conclusion is a gap with an interval, never a bare number. Four experiments are complete on clean data — we shipped one bug, caught it, and re-ran everything — and when reviewers challenged our headline, we ran the ablation that changed the answer. Checklist done. Now the story."

**技术解释**

五个评分项评委直接对号入座，口播只点 soundness 和 clarity 两个最硬的。takeaway 放慢加重、停一拍再翻页。这版叙事更强：负结果不只是如实报告，还被评委问题推动做成了消融——"回应评审"本身就是 Challenges 分的直接证据。

## 第 4 页 · 第一答案：Δ(n) 为负（1:15–2:30，全场最慢）

**演讲词（约 150 词）**

"Here is the first answer, and it did not look good. The textbook says pretrained features should help most when training data is scarce — the sample-efficiency story. We measured the opposite. At n equals twenty, everything sits at zero point five one, near random, for every feature — there is simply no signal to harvest. From fifty on, the gap between embedding and TF-IDF turns negative and stays negative, monotone, ten seeds out of ten. At full data, TF-IDF beats every embedding by five AUROC points or more, with a ninety-five percent interval that excludes zero. We reported this as an honest boundary — pretrained features do not pay in-domain. Then came the question that changed the talk — is this the features, or the way we feed them?"

**技术解释**

图读法：左图 x=n(log)、y=ΔAUROC(embedding−TF-IDF)，三条线=三个 embedding；右图 n=20–500 配对差值放大。**图注是 input = problem+step（旧构造），别念成全部结论**。n=20 全≈0.51：谁都学不出来。配对差值消抽样方差；单点 CI 跨 0 但 10/10 种子同号、4 点单调——方向一致、单点功效不足，照实说。−0.045~−0.064 是 SVM 口径（LR −0.037 起）。结尾的 reviewer 问句是钩子，语气放慢。

## 第 5 页 · 解剖：特征还是输入？（2:30–3:20，E7）

**演讲词（约 140 词，语速稍快）**

"So — features, or input? Three suspects. First, truncation: our input puts the step at the tail of a long prompt, and MiniLM caps at two hundred fifty-six tokens. We measured it — the median input is eighty-nine tokens; MiniLM truncates one point six percent of rows, bge almost none. Real, but far too small. Second, position — moving the step to the front helps embeddings by up to two points, and TF-IDF not at all. Third, dilution — dropping the problem text entirely gives embeddings another three-plus points, and TF-IDF barely one. The problem was dominating the pooling. And here is the verdict — with step-only input, bge-large draws level with TF-IDF, zero point seven one one against zero point seven oh eight. The in-domain 'no' was mostly the input, not the features."

**技术解释**

三嫌犯逐一排除：截断（MiniLM 256 / bge 512 token 上限；中位 89、p99=291；MiniLM 1.6% 行被截、0.6% 整步丢失，bge 0.04%）量级不足；位置（step 前置 +0.015~0.026）；稀释（去题目再 +0.03 以上，主因——题目文本主导均值池化/CLS）。**+0.004 是点估计、无区间——只说"draws level / 追平"，不说"显著反超"**。若被追问旧解释（窄任务/char n-gram 强）：那解释的是 TF-IDF 为何是强基线；E7 解释的是 embedding 为何被压低——两者并存，主因是稀释。

## 第 6 页 · 反转放大（3:20–4:20，全场高潮）

**演讲词（约 155 词）**

"Now the payoff. We keep the fixed input and move the distribution — we train on MATH and test on GSM8K. With the original input, bge-large beat TF-IDF by plus zero point zero two zero. With the step-only input, the gap triples — plus zero point zero six nine — and now all three embeddings are significantly positive, the size gradient intact. And notice what happens to TF-IDF — removing the problem text hurts its transfer, zero point five five one down to zero point five two seven. The problem carried transferable lexical cues; without them, TF-IDF is stuck memorizing MATH-specific surface tokens. Perturbations stay at parity, and selective prediction still slightly favors TF-IDF — we report both as-is. So the refined thesis is that once the input lets them speak, pretrained features pay everywhere — and most when the distribution drifts."

**技术解释**

step-only 迁移端（SVM，paired bootstrap 1000 次）：tfidf-word 0.5265 / MiniLM 0.5569 / bge-base 0.5757 / bge-large 0.5954；Δ = +0.030 / +0.049 / +0.069，CI [0.029,0.032] / [0.048,0.051] / [0.067,0.071]——全部显著为正且随尺寸单调。迁移退化 bge-large −0.116 vs tfidf-word −0.181。复合偏移披露不变：题型+来源+标签方式同时变，只做复合归因。图是旧构造（图注已标明），新数字在右侧 bullet——若评委问图为何不是 step-only：E7 是评审后新增，数字有 JSON 背书。

## 第 7 页 · 计划 + 收尾（4:20–4:45）

**演讲词（约 72 词）**

"For the final, we will dig into why step-only helps at all, decompose the compound shift into single factors, and grow E6, our predictor of the downstream gap from representation properties — the correlation is zero point nine on four points, still preliminary. We asked whether pretrained representations help. The answer survived a hostile review — yes, if the input lets them speak, and loudest when the distribution drifts. Thank you."

**技术解释**

E6：Spearman ρ=0.900、p=0.037、n=4 点——必须带 preliminary。期末交付：三条 Δ 曲线 × 两种输入构造，测量报告式。若被问 AUROC 绝对值：任务本身难，我们报 Δ 不报绝对水平。讲完停在 takeaway，4:45 为理想点。

## 第 8 页 · backup（不计时）

无演讲词，被问到才翻。数字表已含 E7/step-only 全部口径 + 五个预演问题。AI 使用被问时照实答：实现代码由 AI 在我组指导下编写；研究框架、实验设计、数据检查与消融设计由本组完成；已通读全部代码，能逐行解释。

---

## 排练要点

- 掐表两遍：第 7 页讲完应在 4:45 ± 0:10
- 三处停顿：第 1 页 headline 后、第 3 页 "Checklist done" 后、第 6 页 refined thesis 逐字念
- 数字必念准：911,867 · 27.9% · −0.045~−0.064 · 0.711 vs 0.708 · +0.069 [0.067, 0.071] · 0.551→0.527
- 口径纪律：step-only 同分布只说"追平"，不说"反超"；禁用词"失败"，一律 "negative result / artifact / boundary"
