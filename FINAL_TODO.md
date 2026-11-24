# 436 期末项目：远程执行待办

更新：2026-11-23。执行地点：已有数据和模型/特征缓存的远程机器。

## 当前状态

- 已有历史实验、期中展示和 `slides/final.pptx` 草稿。
- 本次已修正折内 TF-IDF、训练缓存指纹，并新增按题目配对 bootstrap 的评估入口。
- 本地合成数据检查 4 项通过，**尚未远程重跑，尚未生成新的实验成绩或期末定稿**。
- 旧结果为探索性分析。修正代码后重新评估同一批数据，仍不能称为新的独立确认。

## P0：确认环境并验证代码

- [ ] 拉取本次提交，记录 commit SHA、Python/依赖版本、硬件、运行命令和随机种子。
- [ ] 确认 `data/prm800k_steps.csv`、`data/mathshepherd_steps.csv` 和原有特征缓存可用；记录数据来源和指纹。
- [ ] 检查按题目分组的 train/val/test 隔离，以及迁移数据与训练题目的重合；有重合先记录并解决，不静默继续。
- [ ] 运行以下检查。缺数据或单类别交叉验证折时应停止并报告原因，不跳过失败后继续汇总。

```bash
python3 tests/test_final_protocol.py
python3 -m py_compile src/protocol.py src/metrics.py src/final_validation.py src/e1_learning_curves.py
```

旧拟合模型缓存不会被新协议复用；冻结 embedding 的逐文本缓存仍可复用。
不要把 `run_all.sh` 的旧结果文件直接当成本轮新结果，也不要覆盖历史 JSON。

## P1：补齐决定结论的实验

### 1. 两种输入的全量评估与按题目置信区间

- [ ] 两种输入都使用修正后的同一协议；同时保留 word/char TF-IDF 基线。
- [ ] 在同分布和迁移数据上，报告各方法 AUROC，以及相对两个基线的配对差值和题目级 95% 区间。

```bash
python3 src/final_validation.py --input-mode step --out results/final_v3_step.json
python3 src/final_validation.py --input-mode problem+step --out results/final_v3_joint.json
```

入口会保存逐样本分数和题目 ID，拒绝覆盖已有输出。若上次中途失败留下分数文件，先保留失败记录，换新的输出名称重跑，或实现明确校验协议/数据身份的恢复流程；不要直接删除旧文件冒充首次运行。

验收：新 JSON 的 `protocol_version` 为 `foldwise-tfidf-v3`；`unit` 为 `problem_id`；包含相对 word 和 char 的比较。区间条件于固定训练模型，不包含训练抽样随机性；两者不能混淆。

### 2. 补 step-only 小样本曲线，并重跑对应原输入对照

- [ ] 优先跑 word、char、BGE-large，SVM；每个 n=20/50/100/500/2000 用相同 10 个种子；全量各一次拟合。

```bash
python3 src/e1_learning_curves.py --input-mode step --features tfidf-word tfidf-char st-bge-large-en-v1.5 --classifiers svm --n-seeds 10 --full-seeds 1 --out-suffix _final_v3_step
python3 src/e1_learning_curves.py --input-mode problem+step --features tfidf-word tfidf-char st-bge-large-en-v1.5 --classifiers svm --n-seeds 10 --full-seeds 1 --out-suffix _final_v3_joint
```

验收：相同 n/seed 的方法使用同一批训练步骤，记录实际 n 与独立题目数。曲线明确注明误差线是训练抽样 SD，还是采用另外定义的不确定性估计，不能把 SD 写成测试集 95% CI。不得凭全量结果推断 step-only 的小样本优势。

- [ ] 在共同覆盖的性能范围内检查达到同一 AUROC 是否需要更少标签；若没有稳定收益，明确报告负结果，不外推或承诺标签节省率。

## P2：有余力再做的机制检验

- [ ] 先在未截断样本上重新比较输入方式，检查变化是否仍存在。
- [ ] 分别编码题目和步骤再拼接，与联合编码、仅步骤比较；控制划分、分类器和选 C 规则。
- [ ] 增加打乱题目与步骤对应关系的辅助对照，并说明拼接导致特征维度变化；不要把一次消融直接当作“池化稀释已证实”。
- [ ] 只有找到经核实从未查看、按题目独立的新数据时，才安排前瞻确认；重新切分已经探索过的数据不算独立确认。

P2 不阻塞 P1。先交付可靠的基础比较，再决定是否值得扩展模型和机制实验。

## P3：更新期末展示

推荐主线：**输入方式会不会改变预训练特征的收益，以及这种收益能否迁移、能否节省标签？**

历史结果仅作重跑前的参考：

| SVM / 同分布 AUROC | 题目＋步骤 | 仅步骤 |
|---|---:|---:|
| BGE-large | 0.6505 | 0.7114 |
| word TF-IDF | 0.6958 | 0.7077 |
| char TF-IDF | 0.6873 | 0.7037 |

历史 step-only 迁移 AUROC：BGE-large 0.5954、word 0.5265、char 0.5585。MiniLM 0.5569，未超过 char；不能概括为所有 embedding 都更好。

- [ ] 用 P1 新结果更新主表和图，所有输入和基线写清楚；不混用旧协议基线与新协议模型。
- [ ] 主讲保留：问题与公平比较 → 原输入的负结果 → 输入消融 → 迁移 → step-only 学习曲线 → 结论与限制。
- [ ] `final.pptx` 第 1/3/4 页：纠正“所有条件 10 种子”“test 只评一次”和不清楚的误差线说明。
- [ ] 第 1/7 页：用新题目级区间替换旧步骤级显著性说法；保留两个 TF-IDF 基线。
- [ ] 第 8/12 页：较小 train-test gap 不证明学到语义；“输入稀释是主因/问题已解决”改成有实验支持的观察与待验证机制。
- [ ] 同分布 +0.0038 的历史点估计不能直接称为优越或等效；迁移约 0.595 的绝对能力必须说明。
- [ ] 评分要求核对移至附录；参考文献、复现说明和实际 AI 使用披露保留。
- [ ] 按老师实际时间要求排练；没有获知要求前不假定展示时长。

## 远程交付与完成标准

- [ ] 提交新指标 JSON、图表、执行日志和可复现命令，记录相对旧结果的变化及失败任务。
- [ ] 分数 NPZ、数据和模型放远程存储，记录位置与校验值；不把大型缓存提交到 Git。
- [ ] 更新英文期末 PPT/讲稿和 README，以实际完成的实验为准；不把计划写成结果。
- [ ] 在本文件勾选已完成事项，说明未完成原因，再 commit/push 代码、指标与展示材料。

可接受的结论包括“未发现小样本优势”。完成标准是验证可靠、结论受证据约束、展示可复现，不是必须让预训练模型获胜。
