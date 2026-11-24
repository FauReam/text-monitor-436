# 运行日志（final_v3 远程重跑）

## 2026-11-24 · P0 环境验证 + 全量重跑启动

**触发**：特征缓存（data/cache，约 930 万文件）在机器维护中丢失；代码/数据/结果完好。
恰逢 FINAL_TODO（2026-11-23）发布新协议 `foldwise-tfidf-v3`，本轮按新计划全量重跑。

### P0 记录

- commit：`a5693075a6b9d493326c07860f10ad164f952b8d`（rebase 自 origin/main `0c89773`
  + 本地 pre_encode 覆盖 E7/step-only 文本的扩展）
- 环境：Python 3.12.3；torch 2.9.1+cu130（CUDA 可用）；sklearn 1.9.1；
  sentence-transformers 6.1.0；numpy 2.5.4；pandas 3.0.6
- 硬件：NVIDIA GB10（aarch64，20 核 CPU，121GB 内存）
- 种子：所有脚本默认 `--seed 0`；E1 学习曲线 10 种子（seed×1000+s）
- 数据指纹（md5）：
  - `data/prm800k_steps.csv` = `d32fc90fd5814b9641fa2ac428f9a28a`（1,761,405 行含表头）
  - `data/mathshepherd_steps.csv` = `3f8b7c25f373783587fd44ae935a2d05`（613,674 行含表头）
  - 来源：hf-mirror 的 tasksource/PRM800K 与 peiyi9979/Math-Shepherd（prep 脚本生成，见 README）
- 隔离检查（实跑 `src/data.py:group_split`）：
  - train/val/test 题数 7576/1083/2165，两两交集 = 0
  - 迁移集 7508 题，与 train/test 的 problem_id 交集 = 0，**题目文本交集 = 0**
- 单元测试：`tests/test_final_protocol.py` 4 项全过；`py_compile` 通过

### 执行中的任务（对应 FINAL_TODO P1）

| 任务 | 命令 | 状态 |
|---|---|---|
| 全量预编码（GPU 串行） | `pre_encode.py --model {MiniLM,bge-base,bge-large(fp16)}`，3,089,020 unique 文本/模型 | MiniLM 完成；bge-base 进行中 |
| E1 TF-IDF 双模式（CPU） | `e1_learning_curves.py --input-mode {step,problem+step} --features tfidf-word tfidf-char --classifiers svm --n-seeds 10 --full-seeds 1 --out-suffix _final_v3_{step,joint}_tfidf` | step 模式 tfidf-word 已至 n=2000 |
| E1 bge-large 双模式 | 同上但 `--features st-bge-large-en-v1.5 --out-suffix _final_v3_{step,joint}_bge`，待 bge 编码完成后启动，之后合并 JSON | 排队 |
| final_validation 双模式 | `final_validation.py --input-mode {step,problem+step} --out results/final_v3_{step,joint}.json`（含题目级配对 bootstrap、NPZ 分数） | 排队（需全部编码） |
| P2 concat 机制检验 | 编码已顺带覆盖；有余力再做 | 未启动 |

### 注意事项

- 旧协议结果文件（results/ 已提交的历史 JSON）不覆盖：新结果一律写 `_final_v3_*` 新文件名。
- 逐文本 embedding 缓存按 FINAL_TODO 允许复用（冻结模型、内容寻址 md5(model+text)）；
  拟合缓存键含协议版本与数据哈希，旧协议缓存不会复用。
- 编码缓存改两级分片存储（data/cache/ab/hash.npy）：旧扁平目录曾写满 ext4 目录索引
  导致 ENOSPC，详见 commit 9347b9a。

## 2026-11-25 · P1 阶段产出（部分完成）

### 已完成

- **全量预编码**：3 模型 × 3,089,020 unique 文本全部入缓存（11h，GPU 串行）
- **E1 学习曲线（新协议，SVM，10 种子 + 全量点）**
  - `results/e1_learning_curves_prm800k_final_v3_step.json`（已合并 tfidf/bge 分片：
    tfidf-word、tfidf-char、bge-large 三特征齐全）
  - step 模式全量点：tfidf-word 0.7077 / tfidf-char 0.7037 / bge-large 0.7114
  - 关键读数：bge-large(step) 在 n=20…2000 每个点都高于 tfidf-word(step)
    （0.53/0.56/0.56/0.63/0.66 vs 0.51/0.53/0.55/0.60/0.63）
  - 分片文件：`*_final_v3_step_tfidf.json`、`*_final_v3_step_bge.json`、
    `*_final_v3_joint_bge.json`（joint tfidf 的 char 全量拟合仍在跑）
- **final_validation（step 模式）**：`results/final_v3_step.json`
  （protocol_version=foldwise-tfidf-v3，unit=problem_id，配对 bootstrap 1000 次）
  - 同分布：bge-large − tfidf-word = +0.0038，CI[−0.0071, +0.0147] **跨 0（追平）**；
    − tfidf-char = +0.0077，CI[−0.0002, +0.0158] 临界
  - 迁移：bge-large − tfidf-word = **+0.0690，CI[+0.063, +0.074] 显著**；
    − tfidf-char = +0.0369 显著；MiniLM vs char = −0.0017 跨 0（不能概括为全部更好）
  - 分数 NPZ（按 FINAL_TODO 只记录位置与校验值，不入 git）：
    - `results/final_v3_step_in_domain_scores.npz`（5.9MB）md5 `88f7516f1e1bdfaf32d8269e84ee3af0`
    - `results/final_v3_step_transfer_scores.npz`（25MB）md5 `04535fc5a62560807db6b8c4629e491c`

### 进行中

- TF-IDF joint 的 char 全量拟合（最后一个训练任务）→ 完成后跑
  `final_validation --input-mode problem+step --out results/final_v3_joint.json`，
  并合并 joint 曲线 JSON
