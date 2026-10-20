# 中期交付 × 得分点 执行手册（目标：7/7）— E7 主线版

> 交付物：one-pager（2 分）+ 中期展示（5 分，五项各 1 分）。
> 文件：`one-pager.pdf`（提交）、`slides/midterm_talk_5min.html`（主讲，白底 8 页）+ `midterm_talk_5min.pdf`（兜底）+ `midterm_script.md`（逐页讲稿）。
> **主线（评委意见后改版）**：第一答案"同分布没用"→ 评委问"是特征还是输入"→ E7 消融：主因是输入稀释（非截断、非表征本身）→ 修输入后同分布追平（0.711 vs 0.708）、迁移 Δ 从 +0.020 翻三倍到 +0.069（三者全显著）。

---

## A. 格式自查（先过格式，再谈内容）

| 要求 | 出处 | 我们的落实 | 状态 |
|---|---|---|---|
| one-pager 一页、全组一份 | syllabus §4.1 | `one-pager.pdf` 实测 1 页（pdflatex 确认） | ✅ |
| 展示严格 5 分钟、到点打断 | L11 + syllabus | 7 页计时 + 1 页 backup（不计时，仅 Q&A）；讲稿 690 词 ≈ 4:40 | ✅ |
| Figure > tables > bullet points > words | L11 评分偏好 | 第 4/6 页核心证据为图；第 5 页一张消融小表；其余全 bullet | ✅ |
| 图必须有轴标签；表必须有行列定义 | L11 | 学习曲线带 x/y 标签、图例、95% 区间带；E7 表行列均标注 | ✅ |

## B. 五项得分点 × 项目内容（每项 1 分）

### 1. Soundness（第 2 页冲刺 + 第 5 页消融）

- 数据：911,867 步、错例 27.9%；train 632,422 / val 88,872（预留未用）/ test 190,573，**按题目分组**
- 协议：冻结 embedding + 逻辑回归/线性 SVM；唯一变量 = 特征来源；balanced accuracy / AUROC；10 种子分层抽样；分组 CV 选 C；test 只评一次
- E7 消融本身是 Soundness 加分项：截断量化（MiniLM 1.6% 行 / bge 0.04%）→ 位置（+0.015~0.026）→ 稀释（+0.03 以上），三嫌犯逐一排除
- 口播句：*"The only thing we vary is where the features come from."*

### 2. Organization（全 deck）

- **开口第一句 = headline**：*"Do pretrained features help? Our first answer was no — and it was mostly an artifact of the input construction."*
- 叙事弧：第一答案（第 4 页）→ 评委之问 → 解剖（第 5 页）→ 反转放大（第 6 页）→ 计划（第 7 页）
- 预演必问 "同分布追平是否显著"：+0.004 是点估计，只说 draws level，不说反超

### 3. Clarity（图的规范 + 差值语言）

- 每页一图/一表一主张；结论句一律差值语言（"Δ = −0.045 ~ −0.064"、"+0.069 [0.067, 0.071]"）
- 旧图图注已标 input = problem+step（旧构造），新数字在 bullet——被问图为何不是 step-only：E7 是评审后新增，数字有 JSON 背书
- 口播句（指图）：*"Read the gap, not the levels."*

### 4. Current progress（第 4–6 页）

- E1–E4 全部完成（v2 干净数据）+ E7 输入消融（评委意见驱动）
- 关键数字：旧构造全量端 Δ = −0.045 ~ −0.064（SVM，CI 不含 0）；step-only 追平 0.711 vs 0.708；step-only 迁移 bge-large +0.069 [0.067, 0.071]、bge-base +0.049、MiniLM +0.030（全显著）
- 口播句：*"The in-domain 'no' was mostly the input, not the features."*

### 5. Challenges（第 3 页一句 + 第 5 页全页）

1. **标签构造 bug**（v1 只取 chosen → 99.8% 正例）：已修复全量重跑，今天数字全为 v2
2. **同分布负 Δ 的归因**（评委意见）：E7 三嫌犯排除法——截断量级不足、位置小贡献、稀释是主因
3. **复合分布偏移**：MATH→GSM8K 题型+来源+标签方式同时变，不作单因素归因，分解在计划里
4. **类别不平衡**：50/50 重采样 + class_weight 两路补查，AUROC 变化 <0.01

## C. 5 分钟逐页时间表（留 15–20 秒余量）

| 页 | 内容 | 时长 | 累计 |
|---|---|---|---|
| 1 | 标题 + headline（第一答案被推翻） | 0:15 | 0:15 |
| 2 | 内容冲刺：背景/设定/数据/方法/指标 | 0:30 | 0:45 |
| 3 | 得分点清单 5 项 ✓ | 0:30 | 1:15 |
| 4 | 第一答案：Δ(n) 为负（旧构造，主图） | 1:15 | 2:30 |
| 5 | 解剖：特征还是输入？（E7 表） | 0:50 | 3:20 |
| 6 | 反转放大：step-only 迁移翻三倍 | 1:00 | 4:20 |
| 7 | 计划 + 收尾 | 0:25 | 4:45 |
| — | 缓冲 | 0:15 | 5:00 |

## D. one-pager 与新主线的对照

one-pager.tex 已同步：Δ(θ) 的 θ 增加 input construction 轴；Current progress 段含 E7 消融与 step-only 迁移新数字；headline 改为 *"conditional on input construction — and largest under distribution shift"*；AI 披露（§4.8）与致谢在末节。

## E. 上台前 checklist

- [ ] `one-pager.pdf` 为最新编译版（含 E7 + 致谢 + AI 披露，1 页）
- [ ] HTML 稿双击能开（图已 base64 内嵌）；PDF 兜底同目录
- [ ] 排练两遍掐表：4:45 ± 0:10 讲完第 7 页
- [ ] 分工：谁讲哪几页、谁答 Q&A，提前定
- [ ] Q&A 预演四个必问：①追平是否显著（点估计，只说 draws level）②为何不随机分（同题泄漏）③AUROC 为何不高（任务难，报 Δ）④复合偏移（复合归因，分解在计划）
- [ ] AI 使用被问时照实答：代码由 AI 在我组指导下编写；思路/数据检查/消融设计由人完成；已通读全部代码能逐行解释

## F. 若被追问"接下来做什么"

- 深挖 step-only 为何反而更优（PRM800K 步骤可能自带上下文）
- 复合偏移的单因素分解（题型/来源/标签方式逐一固定）
- E6 扩网格（ρ=0.90，n=4，preliminary）；期末交付三条 Δ 曲线 × 两种输入构造的测量报告
