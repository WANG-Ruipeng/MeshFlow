本轮方法与复现

本轮响应新的用户反馈，允许训练既有A方向网络。旧交接文件中的“冻结A”继续适用于原始资产和旧实验记录；新训练只写独立layout_student模块和新run。这不改动官方EquiDiT、OT、训练脚本，未启动新AE/FM或完整生成训练。

预先固定：
- 数据：上一轮16个已见开发父对象，主场2050节点/F4096，输出512面。
- 教师：只用Fixed→优化的已审核终点；两次旧面积采样均改善≥2%则移动，否则Fixed零位移。全16保留，不只取大改善。
- 划分：SHA256('a-taskalign-split-v1:'+UID)排序前12训练、后4内部检查；曾经用于开发，不是新确认集。
- 学生：原DomainVelocity 29,907参数，原代理点256/seed4501；两训练种子10、11都从原A10初始化。
- 训练：batch4，均衡父对象循环，AdamW lr0.0003/wd0.0001，3000更新，最终权重固定。每对象曝光1000次。
- 目标：32步归一线性球面桥上离散切向速度。若当前u与下一桥点q，则v=32*(q/(u·q)-u)。内部float64算标签，原网络float32、禁TF32。该目标为监督回归，不是新增latent FM。
- 推理：原CPU float32网络/float64角域保护，32步，最终实际接受方向重新查询半径。新权重不会自动替换原exporter。
- 新质量回退：预算/有效性/实际三维角度不合格时输出Fixed；不根据RMS挑输出，不把回退当学习收益，raw网格不删。
- 评测：两组新的8192点双向面积采样至真实三角面；seed192401/292401+排序UID索引。分母按原对象固定，训练种子与采样种子不当作新对象。
- 指标名称：min_triangle_angle_deg；triangle_min_angle_p05_deg取每个三角形最小角后线性5%分位；triangle_shape_quality_p05为无量纲4sqrt(3)area/sum(edge²)的5%分位，绝不混用。
- 热计时：每对象3次重复，保存各次原始计时和网格哈希一致性。raw生成＋自身精确审计不包含Fixed门槛准备；quality_fallback包含Fixed准备/审计＋raw生成/审计＋选择。模型载入、评测面积RMS和AE/FM不混入布局时延。

在固定MeshFlow worktree根目录运行，下面RUN使用尚不存在的新目录名：
```bash
PY=/home/warpwang/miniforge3/envs/meshflow/bin/python
RUN=output/a_taskaligned_reproduction_new
$PY -m pytest -q -p no:cacheprovider experiments/a_radial/layout_student
$PY -m experiments.a_radial.layout_student.prepare --source-run output/layout_value_20260924_01 --run "$RUN"
$PY -m experiments.a_radial.layout_student.train --run "$RUN" --seed 10
$PY -m experiments.a_radial.layout_student.train --run "$RUN" --seed 11
$PY -m experiments.a_radial.layout_student.oracle_diagnostic --run "$RUN"
$PY -m experiments.a_radial.layout_student.evaluate --run "$RUN"
$PY -m experiments.a_radial.layout_student.fit_diagnostic --run "$RUN"
$PY -m experiments.a_radial.layout_student.integrity --run "$RUN"
$PY -m experiments.a_radial.layout_student.summarize --run "$RUN"
$PY -m experiments.a_radial.layout_student.visualize --run "$RUN"
```

训练中间step_01000/02000只用于恢复。恢复须保持原协议、源码、训练预算及hash，使用--resume PATH --output-dir training/resumed_seed_10_new，防止覆盖旧输出；不能按内部检查结果更换模型。检查点含optimizer/scheduler、Python/NumPy/Torch CPU/CUDA RNG、均衡sampler、配置和轨迹。相同环境的CPU、RTX5080 CUDA连续/恢复已实测逐位一致；不承诺跨硬件逐位相同。

离线oracle读取教师端点，专门核验数值链能否实现目标，禁止并入部署能力；连续与离散两种oracle均通过全部16端点精确审计和严格质量门槛。因此本轮若学生不合格，不能直接归咎32步积分或guard无法实现这些端点。

教师贴近质量边界，近似标签回归不保证实际面质量。方向回归loss、原始输出质量、回退率和同主场RMS必须分别报告；不能通过放宽门槛、换检查对象、挑checkpoint或再抽代理点来改变本轮结论。这里未做容量、输入信息或新损失的消融，负结果不能直接归因某一个因素。

两个后续任务分别登记：
1. A模块：本轮有限训练验证完成后，依据实际部署结果再决定需要何种质量监督或可行化机制；不启动优化器/架构大搜索。
2. 完整生成对照：独立补完整官方EquiDiT GPU前反向、mask/CFG与恢复测试，再按既定配置做AE/FM和官方训练、固定最终权重统一采样及五行评价。此任务不以A模块先胜出为前置条件。历史短训尚不能证明基线充分拟合，Public Post也没有取得Official Full资格。

本轮全量结果与原始数据证据均保留。数据和测试成功不等于正式SIGGRAPH论文结论；这里没有新的类别泛化证据。

封存更正：保存索引dtype与旧账本逻辑dtype的区别、双哈希语义及实际评测源码见[更正记录](code_corrections/README_zh.md)。原始结果未覆盖；图表数值未变化。
