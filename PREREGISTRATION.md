# 预登记（Preregistration）

> 在任何实验运行之前写下，事后对照。证伪的预测照样报告——Δ ≈ 0 是边界结论，不是失败。

## 固定定义

- **任务**：数学推理步骤二分类（correct vs incorrect；neutral 不进主实验）。
- **Δ(θ) = L_test(g∘φ_TFIDF； θ) − L_test(g∘φ_embedding； θ)**，主指标 AUROC（辅：balanced accuracy、F1）。
- **g**：逻辑回归 / 线性 SVM；超参 C 由分组 k-fold CV 在训练子集内选出；test 只报一次。
- **划分**：按题目分组（GroupShuffleSplit/GroupKFold），同一题目的任何步骤不跨集合。
- **小样本**：每个 n 用 10 个种子分层重复抽样，报均值 ± 95% 区间。

## 预测（2026-10-19 锁定）

- **预测 1（Δ(n) 的形状）**：gap(n) = AUROC(embedding) − AUROC(TF-IDF) 的峰值出现在小样本区（n ≤ 100）；随 n 增大，gap 收窄但保持非负。
- **预测 2（Δ(偏移)）**：PRM800K(MATH) → Math-Shepherd(GSM8K) 的迁移下，两种特征都退化，但 Δ(偏移) 仍为正。
- **预测 3（Δ(扰动)）**：规则式扰动（typo/数字格式/LaTeX）下，embedding 的 AUROC 衰减幅度不大于 TF-IDF。

## 判定规则

- 预测成立：均值方向一致且 95% 区间不含 0（预测 1 以峰值位置为准）。
- 部分成立：方向一致但区间跨 0 → 报告为"部分成立"并分析。
- 不成立：方向相反 → **同样写入报告与期末展示**，作为边界结论。
