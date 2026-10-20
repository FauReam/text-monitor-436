# 中期交付 × 得分点 执行手册（目标：7/7）

> 交付物：one-pager（2 分）+ 中期展示（5 分，五项各 1 分）。
> 文件：`one-pager.pdf`（提交）、`slides/midterm.pptx`（展示，8 页）。

---

## A. 格式自查（先过格式，再谈内容）

| 要求 | 出处 | 我们的落实 | 状态 |
|---|---|---|---|
| one-pager 一页、全组一份 | syllabus §4.1 | `one-pager.pdf` 实测 1 页 | ✅ |
| 展示严格 5 分钟、到点打断 | L11 + syllabus | 7 页正片 + 1 页 backup（backup 不计时，仅 Q&A） | ✅ |
| 内容顺序 Background → Problem setup → Challenges → Current progress → Plans | L11 建议 | deck 页序已按此排（Challenges 在 Current progress 之前） | ✅ |
| Figure > tables > bullet points > words | L11 评分偏好 | 核心证据 1 张图；无表格；其余全是 bullet | ✅ |
| 图必须有轴标签；表必须有行列定义 | L11 | 学习曲线带 x/y 标签、图例、95% 区间带 | ✅ |

## B. 五项得分点 × 项目内容（每项 1 分）

### 1. Soundness（第 4 页 Data & protocol）

评委要看：方法是否站得住。我们的弹药全在这一页：

- 数据：911,867 步、错例 27.9%（v2 修复版）；train 632,422 / test 190,573，**按题目分组**，同一题的步骤不跨集合
- 协议：冻结 embedding + 逻辑回归/线性 SVM（课上算法）；唯一变量 = 特征来源；balanced accuracy / AUROC（不平衡不报裸 accuracy）；10 种子分层抽样；分组 CV 选 C；test 只评一次；种子固定
- 口播句：*"The only thing we vary is where the features come from; everything else — data, classifier, protocol — is held fixed."*

预判提问：
- "为什么不按步骤随机分？" → 同一道题的多个步骤文字高度相似，随机分会把答案泄漏进训练集，所有指标虚高——我们第一轮查的就是这类问题。
- "标签不平衡？" → 27.9% 错例 + 分层抽样 + balanced 指标；另有 50/50 重采样补查，AUROC 变化 <0.01。

### 2. Organization（全 deck）

- 页序 = 官方建议顺序；每页标题本身就是该页信息句
- 时间分配见 C 节；backup 页（要求清单）只在 Q&A 时用
- **开口第一句 = headline**（Clarity 同源）：*"The value of pretrained features is regime-dependent: in-distribution they do not help; when the dataset shifts, they do."* 随后声明三张图对应 n / shift / noise 三条曲线；Δ(θ) 记号不作开场，放在 setup 页
- 口播句：开场 15 秒内给出路线图：*"Background, setup, the challenge we hit, what we have so far, and the plan."*
- 预演必问 "in-domain 为什么 TF-IDF 反而赢"：步级对错判断是窄任务，字符 n-gram 基线本来就很强；词向量的优势在词面重叠失效时（跨数据集）才体现（E3 +0.020 即证据）

### 3. Clarity（图的规范 + 差值语言）

- 唯一的结果图：x 轴 = 训练样本数（log），y 轴 = test AUROC，5 条线带图例，阴影 = 10 种子 95% 区间
- 每页 bullet ≤ 6 行；结论句一律差值语言（"Δ = −0.045 ~ −0.064"），不报孤立绝对数
- 口播句（指图）：*"Read the gap, not the levels: below n=20 nothing learns; from n=50 on, TF-IDF pulls ahead."*

### 4. Current progress（第 6 页，图页）

已有的是完整管线 + 干净数据上的全量学习曲线：

- v2 数据（91 万步，错例 27.9%）；学习曲线 E1 完成（n=20…全量 × 5 特征 × 2 分类器 × 10 种子）
- 结论句：**n=20 时所有特征 ≈ 0.51（近随机）；n ≥ 50 起 Δ 转负并单调走负，full 端 −0.045 ~ −0.064**——同分布端预训练未提升，这是如实报告的边界
- 补一句防误读：*"The in-domain gap is negative — but on the transfer axis the sign flips: bge-large beats TF-IDF by +0.020 with CI [0.018, 0.021]."*
- 口播句：*"This is not a failed result — it localizes where the value of pretrained features is NOT."*

### 5. Challenges（第 5 页）

四个真实挑战，每个都带处理方案：

1. **标签构造 bug**：首轮只取 chosen completion → 正类 99.8%、指标退化；已修复为全部人工标注 completions 并重跑（本页数字均为 v2）
2. **类别不平衡**：分层抽样 + balanced 指标 + 50/50 稳健性补查
3. **复合分布偏移**：泛化测试题型+来源+标签方式同时变 → 不作单因素归因
4. **同分布负结果的表述**：Δ 为负按"制度依赖"讲成边界定位——"价值不在同分布制度、在漂移制度"，口头禁用"失败"二字

口播句：*"Our biggest challenge was self-inflicted: our first label construction threw away almost all negatives. We caught it because the metrics degenerated — and re-ran everything."*

## C. 5 分钟逐页时间表（留 15 秒余量）

| 页 | 内容 | 时长 | 累计 |
|---|---|---|---|
| 1 | 标题 + 一句话主题 | 0:15 | 0:15 |
| 2 | Background：查错太贵，便宜监视器 | 0:40 | 0:55 |
| 3 | Problem setup：固定问题 + Δ(θ) | 0:45 | 1:40 |
| 4 | Data & protocol（Soundness） | 0:40 | 2:20 |
| 5 | Challenges（bug 故事在这页） | 0:50 | 3:10 |
| 6 | Current progress（图 + 差值结论） | 1:10 | 4:20 |
| 7 | Plans | 0:25 | 4:45 |
| — | 缓冲 | 0:15 | 5:00 |

## D. one-pager 与内容顺序的对照

| 官方建议顺序 | one-pager 对应段 |
|---|---|
| Background | Task and data 段（标注成本动机） |
| Problem setup | Research question + Δ(θ) 公式段 |
| Challenges | Challenges and plans 段（含 bug 修复） |
| Current progress | Current progress 段（v2 + preliminary 说明） |
| Plans | 末段（E1→E3→E4→E2，期末三图） |

## E. 上台前 checklist

- [ ] `one-pager.pdf` 里 `[members]` 已替换为组员姓名并重新编译
- [ ] deck 用今天最新版（含 v2 数字的第 6 页）
- [ ] 排练两遍掐表：4:45 ± 0:10 讲完第 7 页
- [ ] 分工：谁讲哪几页、谁答 Q&A，提前定
- [ ] Q&A 预演三个必问题：负结果≠失败（Δ(θ) 框架 + 偏移轴正结果）；AUROC 为何不高（任务本身难，我们报 Δ 不报绝对值）；泄漏怎么防（按题目分组）
- [ ] AI 使用披露：口头被问时照实答（写作/调试辅助，实质性内容均已验证）

## F. 若被追问"接下来做什么"

E3（泛化）、E4（鲁棒）已在手：偏移下 bge-large Δ=+0.020 显著正、扰动下 embedding 衰减 ≤ TF-IDF。期末交付三条 Δ 曲线 + 测量报告式结论。**中期只亮学习曲线一张图是刻意的——一张图讲透，胜过五张图讲散。**
