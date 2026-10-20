"""生成 midterm.pptx 与 final.pptx。

- 每页 = 标题 + bullet points；figure 页嵌入 figures/*.png（不存在则灰色占位框）。
- 重新运行本脚本即可把最新实验图嵌进 PPT。
"""
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
OUT = os.path.dirname(os.path.abspath(__file__))

W, H = Inches(13.333), Inches(7.5)
GRAY = RGBColor(0xDD, 0xDD, 0xDD)
DARK = RGBColor(0x33, 0x33, 0x33)


def new_deck():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def add_slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.25), W - Inches(1), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = DARK
    return s


def add_bullets(slide, items, top=1.3, left=0.7, width=None, size=18):
    width = width or (W - Inches(1.4))
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), width,
                                  H - Inches(top) - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (level, text) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = level
        p.font.size = Pt(size - 2 * level)
        p.space_after = Pt(6)
    return tb


def add_figure_or_placeholder(slide, png_name, caption, top=1.3, height=4.6):
    path = os.path.join(FIG, png_name)
    if os.path.exists(path):
        pic = slide.shapes.add_picture(path, Inches(0), Inches(top),
                                       height=Inches(height))
        pic.left = int((W - pic.width) / 2)
        cap_top = Inches(top + height + 0.1)
    else:
        box = slide.shapes.add_shape(
            1, Inches(3.2), Inches(top + 0.6), Inches(6.9), Inches(3.2))
        box.fill.solid()
        box.fill.fore_color.rgb = GRAY
        box.line.color.rgb = RGBColor(0x99, 0x99, 0x99)
        box.text_frame.text = f"[占位] {png_name}\n跑完实验后重跑 make_slides.py 自动嵌入"
        for p in box.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            p.font.size = Pt(16)
        cap_top = Inches(top + 4.0)
    cap = slide.shapes.add_textbox(Inches(0.7), cap_top, W - Inches(1.4),
                                   Inches(1.4))
    tf = cap.text_frame
    tf.word_wrap = True
    for i, line in enumerate(caption):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(15)


# ---------------- 中期展示（10/20，严格 5 分钟，五项各 1 分） ----------------

def build_midterm():
    prs = new_deck()

    s = add_slide(prs, "Can a cheap text monitor tell right from wrong reasoning?")
    add_bullets(s, [
        (0, "CSI 436/536 Machine Learning · Project P · Midterm"),
        (0, "Group members: [组员姓名]"),
        (0, "一句话主题：便宜的文本分类器，能不能读懂数学推理步骤的对错？"),
        (0, "且按课程规定回答：预训练表征在其中到底多买了什么（Δ(θ) 测量报告）"),
    ], top=2.2, size=20)

    s = add_slide(prs, "Background: why monitoring reasoning matters")
    add_bullets(s, [
        (0, "大模型的多步数学推理里常有错误步骤"),
        (1, "逐步人工检查太贵：PRM800K 一个数据集就花了 80 万条人工步骤标注"),
        (0, "如果便宜的文本分类器能自动判对错，标注成本可以被大幅压缩"),
        (0, "但“判对错”需要的知识从哪来？——这正是课程问题的用武之地"),
    ], size=20)

    s = add_slide(prs, "Problem setup: the fixed course question, formalized")
    add_bullets(s, [
        (0, "课程固定问题：预训练表征是否提升下游任务（尤其有限数据下）？"),
        (0, "我们的形式化：Δ(θ) = L_test(简单特征) − L_test(预训练特征)"),
        (1, "θ =（训练样本数 n, 分布偏移, 扰动）三个评测制度"),
        (1, "同一套课上算法（逻辑回归 / 线性 SVM），唯一变量是特征来源"),
        (0, "合规设定：预训练模型冻结只作特征提取器；对照组 = TF-IDF"),
        (0, "交付物 = 测量报告：是否（带效应量）+ Δ 在三根轴上的边界"),
    ], size=20)

    s = add_slide(prs, "Data & protocol (Soundness)")
    add_bullets(s, [
        (0, "PRM800K（OpenAI, MIT）：MATH 推理步骤，人工标注 对/错"),
        (0, "Math-Shepherd（GSM8K，自动标签）：仅作迁移测试集"),
        (0, "输入 = 题目 + 当前步；按题目分组划分，同一题步骤绝不跨 train/test"),
        (0, "标签不平衡 → 主指标 balanced accuracy / AUROC，不报裸 accuracy"),
        (0, "每个 n 用 10 个种子分层重复抽样，报均值 ± 区间；test 只评一次"),
        (0, "预登记已锁定 3 条预测，证伪也照样报告"),
    ], size=19)

    s = add_slide(prs, "Current progress")
    add_figure_or_placeholder(s, "e1_learning_curve.png", [
        "学习曲线（LR，10 种子均值 ± 95% 区间，标签修复后的 v2 数据：91 万步、错例 27.9%、按题目分组）",
        "结论句（差值语言）：n=20 时所有特征 ≈ 0.51 近随机；n ≥ 50 起 Δ 转负并单调走负，full 端 −0.048 ~ −0.064",
        "——同分布端预训练未提升，这是如实报告的边界；泛化/扰动轴见 E3/E4（偏移下 bge-large Δ=+0.020 显著正）",
    ])

    s = add_slide(prs, "Challenges")
    add_bullets(s, [
        (0, "首轮数据构造只取 chosen completion → 正类占 99.8%，指标退化"),
        (1, "已修复为全部人工标注 completions 并重跑——本页数字均为 preliminary"),
        (0, "类别不平衡：对/错步骤比例悬殊 → 分层抽样 + balanced 指标"),
        (0, "泛化测试是复合分布偏移（题型 + 来源 + 标签方式同时变）→ 不作单因素归因"),
        (0, "预登记若被证伪（如 gap 峰值不在小样本端）→ 如实报告为边界结论"),
        (0, "拒答阈值（监视器何时说“不知道”）怎么定 → 风险-覆盖率曲线（可选 E5）"),
    ], size=19)

    s = add_slide(prs, "Plans")
    add_bullets(s, [
        (0, "E1 学习曲线 Δ(n)：n = 20 … 全量 × 4 种特征 × 2 种分类器（最优先）"),
        (0, "E3 泛化 Δ(偏移)：PRM800K(MATH) → Math-Shepherd(GSM8K) + train/test 差距"),
        (0, "E4 鲁棒 Δ(扰动)：typo / 数字格式 / LaTeX 规则扰动下的衰减对比"),
        (0, "E2 三档预训练表征对比（novelty）；E5/E6 可选，不挤占 E1–E4"),
        (0, "期末交付三张核心图：Δ(n)、Δ(偏移)、Δ(扰动)"),
    ], size=20)

    s = add_slide(prs, "Backup: requirement checklist")
    add_bullets(s, [
        (0, "1.2 冻结特征提取器 + 课上模型：sklearn 管道，无梯度回传 ✓"),
        (0, "1.3 简单特征对照：TF-IDF(word/char) 贯穿全部实验 ✓"),
        (0, "1.4 小样本：学习曲线 n 从 20 起，10 种子分层 ✓"),
        (0, "1.5 三轴：sample efficiency ← Δ(n)；generalization ← Δ(偏移)；robustness ← Δ(扰动) ✓"),
        (0, "1.6 novelty：三个预训练表征 + 两种分类器 ✓"),
    ], size=19)

    out = os.path.join(OUT, "midterm.pptx")
    prs.save(out)
    print("saved", out)


# ---------------- 期末展示（12/1，10%） ----------------

def build_final():
    prs = new_deck()

    s = add_slide(prs, "预训练表征值多少：一个 Δ(θ) 测量报告")
    add_bullets(s, [
        (0, "CSI 436/536 · Project P · Final · Group: [组员姓名]"),
        (0, "一句话结论（PRM800K，10 种子，SVM）：预训练表征的价值高度依赖评测制度"),
        (1, "同分布全量数据：Δ = −0.045 ~ −0.064 AUROC——预训练不提升，TF-IDF 反而更好（CI 不含 0）"),
        (1, "小样本端 n=20：所有特征 ≈ 0.51（近随机），预训练没有买到 sample efficiency"),
        (1, "分布偏移（MATH→GSM8K）：bge-large Δ = +0.020，CI[0.018, 0.021]，显著为正"),
        (1, "规则扰动：embedding 衰减 ≤ TF-IDF，但所有衰减 ≤ 0.007 AUROC"),
    ], top=1.8, size=19)

    s = add_slide(prs, "Recap: question and formalization")
    add_bullets(s, [
        (0, "课程固定问题：预训练表征是否提升下游任务？"),
        (0, "Δ(θ) = L_test(TF-IDF) − L_test(embedding)；θ = (n, 偏移, 扰动)"),
        (0, "唯一变量 = 特征来源；g = 逻辑回归 / 线性 SVM；全部冻结、无 fine-tune"),
        (0, "任务载体：数学推理步骤判对错（PRM800K 人工标签，错例占 27.9%）"),
    ], size=20)

    s = add_slide(prs, "Setup (Soundness)")
    add_bullets(s, [
        (0, "按题目分组划分（同一题步骤不跨集合）；test 只评一次"),
        (0, "balanced accuracy / AUROC；每个 n 10 种子分层抽样，均值 ± 区间"),
        (0, "预登记 3 条预测，本节末逐条对照成立与否"),
    ], size=20)

    s = add_slide(prs, "Result 1: Δ(n) — sample efficiency")
    add_figure_or_placeholder(s, "e1_learning_curve_svm.png", [
        "学习曲线（SVM，均值 ± 95% 区间）：n=20 时各特征均 ≈ 0.51（近随机，区间重叠）；n ≥ 50 起 TF-IDF 持续领先",
        "结论句（差值语言）：在本任务上预训练表征未能降低所需标注量——任何 n 处 Δ ≤ 0",
    ])

    s = add_slide(prs, "Result 1b: the gap itself")
    add_figure_or_placeholder(s, "e1_gap_svm.png", [
        "Δ(n) = AUROC(embedding) − AUROC(TF-IDF)：gap 全程非正且随 n 单调走负，full 端 −0.045 ~ −0.064",
        "预登记预测 1（峰值在 n ≤ 100）：不成立——峰值不在小样本端，小样本端所有特征都还在随机水平",
    ])

    s = add_slide(prs, "Result 2: Δ(偏移) — generalization")
    add_figure_or_placeholder(s, "e3_generalization_svm.png", [
        "同分布端 TF-IDF 更高（0.696 vs 0.650）；迁移端反转：bge-large 0.571 vs TF-IDF 0.551",
        "paired bootstrap (1000 次)：bge-large Δ=+0.020 CI[0.018,0.021]，bge-base +0.006 CI[0.004,0.008]，MiniLM −0.001 跨 0",
        "预登记预测 2（偏移下 Δ 仍为正）：部分成立——bge 系显著为正，MiniLM 不显著；复合偏移不作单因素归因",
    ])

    s = add_slide(prs, "Result 2b: train vs test")
    add_figure_or_placeholder(s, "e3_train_test_svm.png", [
        "train−test gap：TF-IDF 0.153–0.168，embedding 仅 0.040–0.065",
        "TF-IDF 拟合了大量只在 MATH 内有效的表面特征；embedding 表示更“抗过拟合”，迁移端衰减更小（−0.08 vs −0.14）",
    ])

    s = add_slide(prs, "Result 3: Δ(扰动) — robustness")
    add_figure_or_placeholder(s, "e4_robustness_svm.png", [
        "规则式扰动（typo / 数字格式 / LaTeX）下所有特征衰减都很小（≤ 0.007 AUROC）",
        "latex / combo 下 embedding 衰减（≤ 0.002）≤ TF-IDF（最大 0.005–0.007）",
        "预登记预测 3（embedding 衰减 ≤ TF-IDF）：成立——但效应量小，方向性结论",
    ])

    s = add_slide(prs, "When do pretrained representations help most?")
    add_bullets(s, [
        (0, "量化答案 = 三条 gap 曲线的实测形态"),
        (1, "n 轴：Δ 最接近 0 处在 n=20（≈ 0，但大家都在随机水平）；随 n 增大走负，full 端 −0.045 ~ −0.064"),
        (1, "偏移轴：bge-large Δ = +0.020（显著正）；扰动轴：衰减差 ≤ 0.005（embedding 略优）"),
        (0, "一句话修正版：预训练特征不是在“数据少”时最值钱，而是在“分布会漂”时最值钱"),
        (0, "本任务同分布端收益为负 = 如实报告的边界结论"),
    ], size=20)

    s = add_slide(prs, "Novelty (1.6) & optional analyses")
    add_bullets(s, [
        (0, "三个预训练表征（MiniLM / bge-base / bge-large）× 两种分类器"),
        (1, "注：家族与维度不同，只表述为模型间对比，不声称纯规模效应"),
        (0, "E5 拒答机制：risk@coverage=1.0 TF-IDF 0.271 略优于 embedding（0.276–0.278），优势不大"),
        (0, "E6 初步观察：训练集散度统计量与实测 Δ 同号，Spearman ρ=0.90（p=0.037，n=4），仅作 future work"),
    ], size=19)

    s = add_slide(prs, "Limitations & honest boundaries")
    add_bullets(s, [
        (0, "只覆盖静态错误识别；不研究对抗性躲避（需 RL，超出课程范围）"),
        (0, "复合分布偏移不做单因素归因；小样本结论均带区间"),
        (0, "Δ < 0 的同分布结论如实报告为边界；训练集天然 28/72 不平衡，补做 50/50 重采样稳健性检验（AUROC 变化 < 0.01）"),
    ], size=20)

    s = add_slide(prs, "Requirement checklist")
    add_bullets(s, [
        (0, "1.1 是否：摘要句带效应量与区间 ✓   1.2 冻结提取器 + 课上模型 ✓"),
        (0, "1.3 TF-IDF 对照贯穿 ✓   1.4 小样本学习曲线（10 种子）✓"),
        (0, "1.5 sample efficiency / generalization / robustness = Δ(n)/Δ(偏移)/Δ(扰动) ✓"),
        (0, "1.6 novelty：多表征 × 多算法 ✓   代码：种子固定、README 一键复现 ✓"),
    ], size=19)

    out = os.path.join(OUT, "final.pptx")
    prs.save(out)
    print("saved", out)


if __name__ == "__main__":
    build_midterm()
    build_final()
