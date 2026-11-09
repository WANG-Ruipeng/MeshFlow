# Bench 三机制固定预算实验报告

**状态：DRAFT / PENDING EVALUATION。当前文件是收口骨架，不包含预判结果。**

## 1. 问题与假说

在同一 EXPOSURE 起点上，分别验证共享两面 PATCH、经过训练的两轮 LOOP、局部质量 RL，
是否改善 Bench 的碎面与几何质量，同时保住 C 采用、主体与支撑。三者是独立快筛，不同时组合。

## 2. 实际方法与对照

- PLAIN200 对照 PATCH200 与 LOOP200；PLAIN267 提供约匹配 Transformer block 计算量的 LOOP 对照。
- LOOP 重复最后四层、双轮 FM 深监督，保持 C 输入；这不是冻结层重复，也不是完整 LiFT 复现。
- PATCH 固定 C-only 两三角共享拓扑；训练/推理共享噪声与速度投影，D0 后恢复同一共享解码合同。
- RL16 与 RL_ZERO 使用同一局部随机过程、mask 和预存噪声；冻结主干，仅更新 rank4 adapter。
- 同30 NATURAL TRAIN父；同12请求、同 C/N/原始 z/查询/相机/截面。不得挑 checkpoint 或挑例。

## 3. 执行与资源

PENDING：填写真实更新、成功/失败/额外调用、时间、显存和并行范围。PyTorch allocated/reserved
不等于进程显存；设备总显存包括用户程序。工程修复与计数见 engineering_events、audits。

## 4. 训练曲线

PENDING：引用 control/finalize.py 生成的 PNG/CSV。PATCH 独立坐标分母、LOOP 双轮目标不同，
不能仅靠跨臂绝对 loss 大小宣布优劣。RL奖励也不能替代真实最终成品评价。

## 5. RAW / BASE_POST / CLEAN_FINAL_POST 完整指标

PENDING：全部177字段与逐请求配对保留，报告关键代价。I 的支持域 NA 必须保留；
共边/共点改善不自动表示非法相交减少。PATCH后端共享恢复是候选流程的一部分。

## 6. 实际五轴图审与固定截面

PENDING_ACTUAL_VIEW：252 cells，AI_PROXY、非盲。合理形状∩合理使用C∩支撑保留
不是无碎面、无重层、无交叉认证。必须另外报告裂缝/悬片与重层标签及不确定性。

## 7. 三个决定

PATCH：PENDING。LOOP：PENDING。RL：PENDING。
先回答各自相对匹配对照是否有稳定改善，再讨论能否保留；不自动合并机制或延长训练。

## 8. 边界与交付

小面板、短预算、已有开发请求；不是普遍泛化认证。不更改默认模型、不自动清理、commit或push。
交付实际权重/状态、固定输入、RAW和POST、完整177字段、配对表、图集、实际图审、账本与恢复入口。
