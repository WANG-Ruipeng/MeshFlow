import hashlib,json,statistics
from pathlib import Path

run=Path('/mnt/e/MeshFlow/.worktrees/ponq-gate1-20260926/output/ponq_metric0_20260926_01')
out=run/'post_analysis/report'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
results=read(out/'RESULT_NUMBERS.json')
audit=read(run/'post_analysis/independent_audit_v2/METRIC_AUDIT.json')
assert audit['passed'] and audit['check_count']==7163
assert results['decision']['status']=='PAUSE_CURRENT_JOINT_RECIPE'
assert not (out/'VERDICT_zh.md').exists()
traces={arm:[json.loads(s) for s in (run/rel).read_text().splitlines() if s.strip()][-100:] for arm,rel in [('B0','control/TRAIN_TRACE.jsonl'),('M0','training/M0/TRAIN_TRACE.jsonl')]}
comparison={}
for field in ['old22_loss','x_mse','n_mse','F_mse','gradient_norm_before_clip']:
    vals={arm:statistics.mean(float(row['loss' if arm=='B0' and field=='old22_loss' else field]) for row in rows) for arm,rows in traces.items()}
    comparison[field]={**vals,'M0_relative_change_pct':100*(vals['M0']/vals['B0']-1)}
comparison['clip_active_count_last100']={arm:sum(row['gradient_norm_before_clip']>1 for row in rows) for arm,rows in traces.items()}
comparison['scope']='same last100 continuation batches; field means and original22-channel mean are comparable; newly weighted M0 total is not compared against old B0 total'
(out/'COMMON_UNIT_COMPARISON.json').write_text(json.dumps(comparison,indent=2,ensure_ascii=False)+chr(10))
report='''# SINGLE_STATE_METRIC0 最终裁定

**裁定：暂停当前“高斯噪声 → 联合生成位置、raw法向与F共22通道 → PoNQ提取”的具体配方。本轮保留为损失度量带来局部改善的结果，不晋级为可用生成器，不再自动追加训练、扫权重或更换网络。**

这是一项研究资源分配决定，范围限于本轮状态表示、网络、耦合和预算。它没有证明所有Flow Matching不可行，也没有证明更大模型永远学不会；但当前配方在单个薄壁目标上尚未建立可用性，不足以继续承担SIGGRAPH生成论文的核心方法。

实验已执行完成，全部10个阶段退出0；没有后续训练任务。本轮新增6000次优化更新，训练部分实际254.808秒。官方MeshFlow主工作树保持干净；实验在独立worktree内，官方长训仍未恢复。

本轮实际只改变一个变量：三个字段内部平均速度MSE的固定正权重。M0与归档B0均为原宽128 DeepSets，88,214参数，从共同3000步完整状态分支，到累计9000步；6000个后续批次逐批哈希一致。优化器、随机状态、数据、归一化、位置Hungarian匹配、raw法向、F表示和64步Euler均保持原协议，采样速度没有乘loss权重。

确认数据为同一固定薄U形快照的32份新噪声，两臂共享，合计64次完整自由生成和审核。它不是32个新对象，也不是训练种子重复。参考网格为原508面目标；提取输出保留实际面数，没有强制改成508面，没有与旧64份噪声合并统计。

|指标：本轮新32噪声|原损失B0|固定标定损失M0|
|---|---:|---:|
|成功提取|32/32|32/32|
|完整几何有效|6/32（18.75%）|5/32（15.625%）|
|有效且表面RMS≤0.20%|0/32|0/32|
|全部输出表面RMS中位数|2.17309%|1.88425%|
|全部输出表面RMS均值|2.23744%|1.90044%|
|全部输出最小表面RMS|1.60087%|1.44924%|
|预测→目标RMS中位数|2.58977%|2.28481%|
|目标→预测RMS中位数|1.62562%|1.38955%|
|绑定位置RMS中位数|5.26436%|5.22359%|
|法向角中位数|43.0954°|37.8776°|
|raw法向长度RMS中位数|0.22801|0.18186|
|C相对误差中位数|0.46861|0.43672|
|实际面数：最少/中位/最多|278/335/408|296/351/406|

表面RMS用目标包围盒对角线归一化，采用原8192面积样本/方向的双向表面评估。完整几何有效要求封闭、定向、二维流形、顶点link为简单环、Betti=[1,0,1]，以及原精确相交、退化、重复面和图容量检查。成功提取不等于有效；有效也不等于精度达标。

**误差改善是真实观测，仍远未达到交付目标。** 相同噪声配对下，M0有27/32份RMS下降、5/32上升；配对差中位为−0.356894个百分点。两臂全样本中位数从2.17309%降至1.88425%，降低约13.3%；均值降低15.06%。两个表面距离方向均改善，不能把本轮概括成“完全没用”。

但M0最好的一份1.44924%仍是0.20%门槛的7.25倍，而且这一份已经几何有效。即使仅挑有效输出，也没有接近精度线。联合通过仍是0/32，预登记推进条件是至少24/32；不是只差一两个临界样本。几何有效计数6→5也没有给出有效率提高的证据，不能把单种子的一份差值推广为稳定退化。

M0仍有21/32份顶点link非简单环、14/32份非流形边、10/32份不连通，这些失败集合重叠。两臂本轮均未出现精确自交、退化或域外位置；这不足以替代其余合同。形状精度与组合拓扑都仍是问题，不能仅归因为某个拓扑修复步骤。

损失干预也确实发生了。在共同3000步模型、固定24个条件路径探针、排除输出层的85,376个共享参数上，标定得到：

|字段|位置x|raw法向n|F|
|---|---:|---:|---:|
|原字段平均MSE系数|3/22|3/22|16/22|
|本轮冻结系数|0.7593693413|0.2218289670|0.4429175909|
|起点原加权共享参数梯度RMS|0.168106|0.575464|1.537136|
|起点新加权共享参数梯度RMS|0.936134|0.936134|0.936134|

共同尺度用完整梯度Gram交叉项计算，旧/新总共享梯度RMS为1.6819983816/1.6819982895，相对差5.47e−8；没有异常梯度、没有夹权重、没有搜索。权重只标定一次并冻结，后续5000/7000/9000检查点只是观测，不用于挑模型。

相同末100批次、相同单位的速度误差变化为：

|字段|B0平均MSE|M0平均MSE|相对变化|
|---|---:|---:|---:|
|位置|0.0508845|0.0383107|−24.71%|
|法向|0.3502434|0.3071150|−12.31%|
|F|0.2470892|0.2744939|+11.09%|
|按原22通道平均重新计算的总MSE|0.2344005|0.2467354|+5.26%|

不能把M0的新加权总loss直接与B0旧loss作数值比较。这里使用共同单位，说明权重改变实际影响了拟合取舍，且有部分效果传到自由生成曲面。固定路径上F速度误差上升，也不能直接叫作部署Q退化：本轮自由终点F/C误差多数反而改善。路径速度回归与自由终点是不同测量。

初态共享梯度匹配不等于全程平衡，也不等于匹配AdamW更新。9000步M0三个字段加权梯度RMS已分别约1.2011/1.6756/1.0890；两臂末100步的全局梯度裁剪均生效。此次控制的是固定loss度量，不能声称隔离了所有优化因素。

Hungarian实际成本经源码与探针核验，只是物理位置三维平方距离之和，法向和F对选中匹配的成本贡献均为0；整个22通道目标按同一位置置换绑定。因此，“F主导了匹配”不是本轮支持的解释。本轮没有改变匹配。

**研究上的终局决定是：当前联合生成配方停止推进，损失分配这一诊断结果归档。** 更长训练、逐点容量增强、单层注意力和本轮固定度量调整，均未提供足以让该具体配方晋级的证据。本轮不再开启另一组权重、另一层注意力或更长Euler测试，也不降低质量线。

此前“有效初态＋已知对应运动 → Q更新 → 重新提取”的递推正结果继续独立保留。它支持形变模块的数值可行性，不能替代无条件生成证据。若以后重新提出FM生成方案，需要新的状态组织或约束机制及清楚的可检验假设；本次任务不自动启动这些新工作。

独立审核最终7163/7163项通过，包括原清单、完整恢复、6000批流、独立批次重放、CPU重算24个共享梯度Gram、64份输出和四个保存检查点。CPU/GPU Gram相对误差约7.32e−6，在事先审核容差2e−4内。检查项数量不是独立统计样本量。

首轮审核有两项失败来自审核器把旧NPZ的float32时间标签与预登记Python常量做精确元组比较；审核器已改为按预登记时间表核验，首轮失败记录保留，修正后通过。没有重训、重抽样、更改门槛或修改原实验清单。

历史B0控制已验证源码、数据、3000/9000状态和6000训练序列；在当前环境重放一份归档噪声的终点字节完全一致，最大误差0。历史driver/CUDA/scipy/cuDNN等完整运行环境没有全部记录，不能宣称每个历史环境字段都完全一致；历史训练时间仅作背景。

本轮当场测量：B0的32份积分6.919秒、后验诊断0.153秒、提取及审核13.364秒；M0相应为6.223、0.152、12.770秒。这是各一次顺序运行，没有重复计时，不能据此声称M0更快。核心裁定依据是同预算下的几何交付，且两臂联合达标均为零。

[64行自由输出账本](FREE_LEDGER_64.csv) · [完整数值](RESULT_NUMBERS.json) · [末100批同单位对照](COMMON_UNIT_COMPARISON.json) · [原冻结协议](../../RUN_PROTOCOL.json) · [独立审核](../independent_audit_v2/METRIC_AUDIT.json)

[全32配对误差图](../visuals/fresh_paired_surface_rms.png) · [固定ID 0–3网格图](../visuals/fresh_samples_000_to_003.png) · [共享梯度图](../visuals/weighted_parameter_gradient_rms.png)

所有补充报告和图表由独立清单绑定；原实验697项文件清单保持不变。本文件为该实验的研究判断，不将单目标结果升级为多对象泛化或论文胜负结论。
'''
(out/'VERDICT_zh.md').write_text(report,encoding='utf-8')
refs=[run/'MANIFEST.json',run/'RUN_PROTOCOL.json',out/'MANIFEST.json',run/'post_analysis/independent_audit_v2/METRIC_AUDIT.json',run/'post_analysis/visuals/SUPPLEMENTAL_MANIFEST.json']
extra=[out/'VERDICT_zh.md',out/'COMMON_UNIT_COMPARISON.json',Path(__file__)]
bound={'scope':'interpretation and common-unit statistics; supplements immutable main experiment and earlier report; no model or samples modified','original_manifest_sha256':sha(run/'MANIFEST.json'),'references':[{'path':str(p.relative_to(run)),'sha256':sha(p)} for p in refs],'files':[{'path':str(p.relative_to(run)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in extra]}
(out/'INTERPRETATION_MANIFEST.json').write_text(json.dumps(bound,indent=2,ensure_ascii=False)+chr(10))
print(json.dumps({'status':'WRITTEN','decision':results['decision']['status'],'report':str(out/'VERDICT_zh.md'),'checks':audit['check_count'],'last100_common':comparison},ensure_ascii=False,indent=2))