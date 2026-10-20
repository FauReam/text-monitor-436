# 中期展示讲稿：7 页主讲 + 2 页备份

> 英文演讲词为朗读部分，中文解释和 Q&A 不朗读。HTML 内按 N 显示同一份逐页讲稿。
> 英文朗读约 498 词。目标用时 4:45，预留 15 秒；这不是实测时长，请按个人语速掐表排练。
> 顺序按 L11 建议：Background → Problem setup → Challenges → Current progress → Plans。展示页以图为主，不再口播评分项。

| 页 | 内容 | 时长 | 累计 |
|---|---|---|---|
| 1 | Pretrained features for math-step monitoring | 0:25 | 0:25 |
| 2 | A controlled comparison of feature representations | 0:45 | 1:10 |
| 3 | The challenge: what text should the monitor read? | 0:35 | 1:45 |
| 4 | No observed sample-efficiency gain with the original input | 0:45 | 2:30 |
| 5 | Step-only input narrows the in-domain gap | 0:50 | 3:20 |
| 6 | BGE transfer gains depend on the baseline | 0:50 | 4:10 |
| 7 | Validation and the next experiments | 0:35 | 4:45 |
| 8–9 | Backup，仅 Q&A | 不朗读 | — |

## 第 1 页：Pretrained features for math-step monitoring（0:00–0:25）

**英文演讲词（49 词）**

Our project asks whether pretrained text features help a lightweight classifier judge math reasoning steps. This connects Project P to a concrete binary task: is the current step correct? Our preliminary result depends strongly on what text we encode. The clearest benefit appears in transfer to a different dataset.

**技术解释，不朗读**

这里的 lightweight 指冻结特征上的线性分类器。没有测量编码耗时或部署成本，不声称运行成本优于其他系统。目标是比较表征，不是证明系统能可靠验证数学。

## 第 2 页：A controlled comparison of feature representations（0:25–1:10）

**英文演讲词（78 词）**

We compare word and character TF-IDF with three frozen encoders: MiniLM, BGE base, and BGE large. We train logistic regression and linear SVM, without fine-tuning the encoders. Our prepared PRM800K dataset has about nine hundred twelve thousand labeled steps. We split by problem, so steps from one problem cannot cross train and test. AUROC is our main metric, with balanced accuracy as a secondary check. The figures today use linear SVM and identify the input in every comparison.

**技术解释，不朗读**

编码器：sentence-transformers/all-MiniLM-L6-v2、BAAI/bge-base-en-v1.5、BAAI/bge-large-en-v1.5。TF-IDF word=(1,2)，char_wb=(3,5)，均 min_df=2。C 网格0.1/1/10。训练/预留验证/测试步骤632422/88872/190573。分组划分防止同题步骤跨集合，但不能据此声称整个协议完全无泄漏。现有 TF-IDF 在内部 CV 前 fit，需在各折内部重拟合后重跑。AUROC 不是 balanced accuracy。

## 第 3 页：The challenge: what text should the monitor read?（1:10–1:45）

**英文演讲词（65 词）**

A key challenge is deciding what the monitor should read. Our original input joins the problem and current step. We compare that with putting the step first, and with using the step alone. Each input uses the same data split. Removing context might help a sentence encoder focus on the step, but that is a hypothesis. It could also remove information needed to judge correctness.

**技术解释，不朗读**

这三种输入使用同一数据划分，但各自按当前协议选 C，因此是输入条件的端到端比较，并非固定超参数下纯池化因果实验。原始标签构造 bug 已修复，本次使用修复后结果。MiniLM 截断1.6388%，BGE0.0395%，但少量截断本身不能定量排除所有影响。没有真实外部评审经历的证据，不使用 reviewer / hostile review 叙述。

## 第 4 页：No observed sample-efficiency gain with the original input（1:45–2:30）

**英文演讲词（70 词）**

First, the limited-data experiment uses the original problem-plus-step input. The horizontal axis is the number of labeled training steps. The vertical axis is test AUROC. Points average ten training subsamples, and error bars show one standard deviation. At twenty labels, mean performance is near chance. From fifty through two thousand, all three embedding means remain below word TF-IDF. We have not yet tested whether step-only input changes this learning curve.

**技术解释，不朗读**

n=20/50/100/500/2000，每点10个按标签分层抽取的训练子样本，固定题目分组测试集。误差线为样本标准差（ddof=1），不是95% CI。full点只有1次，不放入重复抽样曲线。小样本各种子差值可变号，不能说10/10同号或严格单调。此图仅支持原始输入下的观察。

## 第 5 页：Step-only input narrows the in-domain gap（2:30–3:20）

**英文演讲词（81 词）**

Next, the full-data input ablation. Each row is one representation, and moving right means higher AUROC. The blue diamonds show step-only input. Removing the problem improves embeddings by about five to six AUROC points, more than it improves TF-IDF. BGE large reaches zero point seven one one, compared with word TF-IDF at zero point seven zero eight. These are point estimates from one fit per condition. They neither establish superiority nor statistical equivalence, and they do not isolate a pooling mechanism.

**技术解释，不朗读**

E7为全量训练 seed=0，每条件1次。MiniLM 0.6314→0.6810，base 0.6498→0.7037，large 0.6505→0.7114；word 0.6958→0.7077，char 0.6873→0.7037。large相对word精确差值+0.0037714。只能说点估计接近，不能证明相等、等效、显著胜出。位序、长度、上下文信息和正则化均可能参与。

## 第 6 页：BGE transfer gains depend on the baseline（3:20–4:10）

**英文演讲词（81 词）**

For transfer, we train on PRM800K and evaluate on Math-Shepherd. Here, step-only BGE large reaches zero point five nine five. Its gain is about zero point zero six nine over word TF-IDF, but only zero point zero three seven over the stronger character baseline. BGE base also exceeds both. MiniLM slightly trails character TF-IDF. All transfer scores remain modest. This is a compound shift in problem type, data source, and labeling, so we cannot attribute the gain to just one factor.

**技术解释，不朗读**

step-only SVM：word0.526453，char0.558510，MiniLM0.556855，base0.575722，large0.595379。large-word=+0.068926；large-char=+0.036869；base-char=+0.017212；MiniLM-char=-0.001655。旧输入 large-word=+0.019587，large-char=+0.008635。现有CI只有对word的按步骤paired bootstrap，未处理同题相关性，不能套用到char或声称所有embedding显著优于两个基线。

## 第 7 页：Validation and the next experiments（4:10–4:45）

**英文演讲词（74 词）**

For the final, our priority is stronger validation. We will fit TF-IDF inside each cross-validation fold and replace step-level bootstrap intervals with intervals that resample whole problems. We will also run step-only learning curves and compare separate encodings of the problem and step. Because the input ablation reused evaluation sets, we treat it as exploratory. Our current conclusion is conditional: input construction and baseline choice change the apparent value of pretrained features. Thank you.

**技术解释，不朗读**

优先顺序：修TF-IDF折内预处理并重跑→按题目paired bootstrap→step-only小样本曲线→问题和步骤分别编码拼接。输入消融是探索性结果，不能再称test touched once；只有尚未查看且能保证独立的数据才能作最终确认集。两个数据集可能存在题目重叠，需要另查。复合偏移不等于单一domain因果效应。AI披露需要包含本次分析核对与改稿，组员提交前需逐项确认理解。

## Q&A 口径

- **是否证明 embedding 比 TF-IDF 好？** 当前只支持条件性的比较。step-only BGE 两个模型的迁移点估计超过两个基线，MiniLM没有超过char基线。同分布BGE-large与word差+0.0038，还不能断言优越或等效。
- **为什么不用全部10种子CI？** 只有原始输入的小样本曲线每点10次训练抽样，全量和E7每条件一次。曲线误差线是1 SD，不是95% CI。固定测试集上的重复训练子样本不等于10个独立测试集。
- **原CI说明什么？** BGE-large相对word的step-only迁移按步骤配对bootstrap区间为[0.067,0.071]。它忽略同题相关性，且不包含训练随机性，不作为当前显著性结论；对char基线没有现成区间。
- **是否已解释输入机制？** 没有。低截断率和输入消融支持继续检查上下文/池化/位置，不能单独证明稀释为主因。TF-IDF和embedding也都按每个输入条件另选C。
- **AUROC不高怎么办？** 直接承认绝对水平有限。数学步骤正确性可能需要完整上下文或实际计算。当前研究的是表征比较，不是部署级验证器。不能用“只看Δ”回避能力限制。
- **测试集是否只用一次？** 不能这样说。多个实验和探索性输入消融复用了评估集。只在确认从未查看且独立时，才能把后续数据称作未触碰确认集。
- **代码与数据是否已重新核验？** 本轮按现有JSON重写与重画，没有重新运行原实验。本地项目没有原始数据/模型缓存，无法在本轮补做折内TF-IDF、按题目CI或step-only学习曲线。已有代码限制在报告和展示中列出。
- **AI使用？** AI在组员指导下生成实现代码，也协助本次分析核对、图表及文字改写。组员必须理解并验证提交内容。不要在未实际完成代码通读前声称“已经理解每一行”。

## 排练与操作

- 英文讲稿先按自然语速排练两遍，目标第7页在4:45内结束。不要念中文解释、来源脚注和备份页。
- 指图时说明横轴、纵轴、输入模式和基线。重点数字：0.711 vs 0.708；迁移0.595；相对word +0.069、相对char +0.037。
- 第4页只谈原始输入的样本效率，第5页是全量训练输入消融，不能混为同一个结论。
- HTML离线可用：方向键翻页，N注释，T计时/暂停，R重置，F全屏，Home首页，End主讲末页。主讲7页后才进入备份。
- PDF备用版本包含7页主讲和2页Q&A，讲到第7页停。最终提交前，全组确认结论、AI披露与实际工作相符。
