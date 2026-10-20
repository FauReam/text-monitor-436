# 中期材料使用说明

已根据原始 course.html syllabus 与 L11_SVM.pdf 第21页的展示要求重写。

- one-pager.pdf：一页提交版。one-pager.tex 和 figures/ 是可编辑来源。
- slides/midterm_talk_5min.html：7页主讲 + 2页Q&A，图表内嵌，离线打开可用。
- slides/midterm_script.md：逐页英文朗读稿、中文技术解释和Q&A。必讲英文409词（约3:20–3:30），总时长目标4:45以内、5:00硬上限。时间是排练目标，尚未实测。
- slides/midterm_talk_5min.pdf：同内容备用版，第7页为主讲结束。

HTML：方向键翻页，N显示讲稿，T启动/暂停计时，R重置，F全屏，End跳到主讲第7页。

## 内容边界

对小样本曲线只报告原始 problem+step 输入；误差线是10次训练子样本的1 SD。
全量与输入消融每条件为seed=0的一次拟合，没有伪装成10次实验。
迁移图同时保留word和char基线：step-only BGE-large比word高0.069，比char高0.037；MiniLM没有超过char。
输入稀释仍为假设，0.711 vs 0.708不能证明统计等效或优越。

本轮没有重跑模型。折内TF-IDF预处理、按题目重采样CI和step-only小样本曲线仍待补做，不能把改稿当成修复了实验协议。原实验结果文件和模型代码未改动。

提交前需要全组掐表排练，并核实AI披露及对代码、结果的理解。中期展示的五项rubric不是完整21分保证。

制图结果位于figures/；JSON原始结果位于results/。完整本轮交付包另存于Codex输出目录。
