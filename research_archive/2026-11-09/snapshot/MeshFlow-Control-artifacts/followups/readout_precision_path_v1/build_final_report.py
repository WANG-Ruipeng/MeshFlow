from pathlib import Path
import csv,json,statistics,hashlib
r=Path('/mnt/e/MeshFlow-Control-artifacts/followups/readout_precision_path_v1')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def mean(rr,k):return statistics.mean(float(x[k]) for x in rr)
plan=read(r/'plan.json');budget=read(r/'budget.json');cost=read(r/'cost_summary.json')['by_path'];m=rows(r/'rollout_metrics.csv');p=rows(r/'analysis/prediction_precision_interactions.csv');l=rows(r/'analysis/layer_precision_metrics.csv')
index={(x['model_label'],x['request_id'],x['precision'],x['readout']):x for x in m}
order=[('LEGACY_BF16','ON'),('LEGACY_BF16','OFF'),('FP32_REFERENCE','ON'),('FP32_REFERENCE','OFF')]
def values(label,req,key,scale=1,fmt='.3f'):
    return ' / '.join(format(float(index[label,req,a,b][key])*scale,fmt) for a,b in order)
fp=[x for x in l if x['precision']=='FP32_REFERENCE']
s=['# READOUT_PRECISION_PATH_AND_ROLLOUT_V1','',
'**执行完成；结论：MIXED_TRADEOFF，未建立 readout 的稳定质量收益。** 本轮不建议把精度匹配训练列为优先下一步。数值加回更忠实，但固定状态中的分支预测增量反而全部减小；完整生成显示局部款式和指标取舍，没有持续改善稀疏 C 的使用。FP32 继续作为显式诊断参照，默认模型/采样精度不变。','',
'## 实际执行与完整性','',
'| 项目 | 实际结果 |','|---|---|',
'| 新模型 forward | **1260 = 12 接线预检 + 48 FP32 固定状态 + 1200 完整采样** |',
'| 新完整生成尝试 / 完成 | **24 / 24**，均为独立 Euler50；失败 0、未决 0 |',
'| 可信缓存 | 8 条旧 BF16_ON 完整轨迹/RAW/评分；48 份旧 BF16 单步结果，均核对身份 |',
'| 训练 / backward / optimizer update / 新 Gaussian / 新条件 | **全部 0** |',
'| 评价 / 实际看图 | 32/32 评分（24 新 + 8 缓存）；32/32 AI 审阅，8 无参考页后 4 P1 参考页，均四视图+wire |',
'| 保护合同 | 权重文件和参数状态、C、原始 Gaussian、mask、原报告/数组/评分不变；6 次模型加载后的执行阶段 RNG hash 前后一致 |',
'| 硬保持和递推 | 24 条新轨迹的 1200 步逐一复核原 FP32 Euler 表达式；全部51态 C、N/K、finite、槽位、NPZ/PLY 字节核验通过 |',
'| 缺失 / 科学调用错误 | 0 / 0；人评 NOT_RUN，P2 GT/S/E 为 NA，P1 E 原检测仍为 PARTIAL |',
'| GPU 累计持锁 | 214.741 秒（约 3.58 分钟），上限2700秒；单GPU worker |','',
'这些是四个已观察条件，每条件一个原 Gaussian，在两个不同权重上测量；不是八个独立条件、不是新泛化测试。完整性 PASS 不会把几何检测的 PARTIAL/UNRESOLVED 变成确定结论。','',
'## A. 数值路径与分支增量','',
'LEGACY_BF16 保持原混合精度：FP32参数/积分，原主干 BF16、readout 内部 FP32。FP32_REFERENCE 显式关闭相关 autocast，最高 FP32 matmul、TF32关闭、相同math SDPA，不把最终BF16输出转float冒充高精度。结构、预测参数化、known/free合同及四个readout层顺序不变。OFF只跳新增readout；Geo、role、known输入和联合attention照常执行。CLEAN H0每次前向重新计算，不跨步缓存。','',
'接线证据：4次旧 BF16_ON 回放逐位一致；8次“真实计算readout但实际加零”与同精度OFF逐位一致。4个FP32_OFF参照占用并复用正式48次预算，未额外调用。12次完整FP32 dtype探针覆盖ON/OFF/ZERO及关键前向模块；96条正式FP32 ON层均实际FP32。保存完整数组的16代表层，实际 h+c 与 h_after 在CPU上逐位一致。其余80层保存即时完整统计和操作数hash，未声称保存全数组。详见 precision_contract.json / dtype_audit.json。','',
f"**有效加回更忠实。** 全96个FP32层记录中 ‖e‖/‖r‖ 范围 {min(float(x['effective_over_residual_l2']) for x in fp):.8f}–{max(float(x['effective_over_residual_l2']) for x in fp):.8f}，cos(e,r)最小 {min(float(x['effective_residual_cosine']) for x in fp):.9f}，‖e−r‖/‖r‖ 最大 {max(float(x['effective_error_relative_l2']) for x in fp):.7f}。非零r但e=0的比例最高仍有 {max(float(x['residual_nonzero_swallowed_fraction']) for x in fp):.5%}，没有要求其绝对为零。",
'',
'旧 BF16 第12层的对应方向余弦均值 B/C 为0.222/0.177，范数比为0.295/0.238；此前标量舍入现象真实存在。96个旧同h/r的CPU局部FP32加法对照也支持这一点。但两种完整前向的h/r本身可以不同，局部加法参照不代表下游完整模型效果，标量不变比例不等于信息丢失率。','',
'**最终分支预测增量没有放大：24/24 个固定状态中 ‖D_FP‖ < ‖D_BF‖。** D表示同精度ON−OFF的free速度。下表是各模型12个状态的描述均值，不能当作12个独立条件。','',
'| 模型 | BF分支差 / 自身ON速度 | FP分支差 / 自身ON速度 | ‖D_FP‖/‖D_BF‖ 范围 | 两种增量方向余弦范围 |','|---|---:|---:|---:|---:|']
for label in plan['labels']:
    rr=[x for x in p if x['label']==label]
    ran=lambda k:f"{min(float(x[k]) for x in rr):.3f}–{max(float(x[k]) for x in rr):.3f}"
    s.append(f"| {label} | {mean(rr,'D_BF_relative_to_own_ON'):.4%} | {mean(rr,'D_FP_relative_to_own_ON'):.4%} | {ran('D_FP_over_D_BF_l2')} | {ran('D_FP_D_BF_cosine')} |")
s+=['',
'按数据P1/P2、模型和t的分层结果完整保留在 analysis/prediction_precision_group_summary.csv，不用总体均值覆盖小K差异。FP32去掉或改变了原低精度路径中较大的开关差异，是与证据相容的解释；不能把它断言为纯噪声，也不能用“增量更小”直接判失败。','',
'关闭分支的精度差 P=v_FP_OFF−v_BF_OFF，以及交互 J=D_FP−D_BF 均保留可无损复算向量。24组都精确满足 v_FP_ON−v_BF_ON=P+J。P的范数范围 B=0.13236–0.76423、C=0.13236–0.81438；J并非某一次残差加法的唯一因果贡献，因为整个前向精度都改变了。四格来自各模型自己的旧ON状态；完整rollout则各自自由推进，二者没有混用。','',
'## B/C. 完整生成：条件作用与主干精度效应','',
'以下每格按 **BF_ON / BF_OFF / FP_ON / FP_OFF** 顺序列出，单位为百分比。I使用原C面积查询的crossing/near-duplicate并集；FF和小角以free面数为分母。低I不自动表示C被语义正确使用，低FF也不能替代完整形状审阅。','',
'| 条件 | 模型 | I 四格 % | FF 四格 % | 小角<5° 四格 % |','|---|---|---|---|---|']
for q in plan['requests']:
    short=q['request_id'].replace('P1__','').split('_C1')[0] if q['panel']=='P1' else q['request_id'].replace('__n0','')
    if len(short)>16:short=short[:8]+'…'
    for label in plan['labels']:
        rid=q['request_id'];s.append(f"| {short} | {label[0]} | {values(label,rid,'ConflictCov_C',100)} | {values(label,rid,'FF_illegal_affected_free_fraction',100)} | {values(label,rid,'free_minimum_angle_lt5_fraction',100)} |")
s+=['',
'**33ec（普通K）：没有清楚的条件使用改善。** B的四格整体形状几乎一致，S始终0；FP_ON的I不变、Dsym相对BF_ON几乎不变。C在FP32 ON/OFF都出现支撑位置/重叠变化；FP_ON参考Dsym下降约4.18%，同时I从0.0977%升至0.2930%、FF从90.2299%升至91.3793%。图中仍是合理可辨的椅形，但不能称更好的C整合。生成实体弧顶靠背与来源开放矩形靠背的差别按款式处理，没有当作重建失败。','',
'**f596（普通K）：B的显著参考变化主要也存在于OFF。** B FP_ON相对BF_ON：参考S增加8.6716个百分点，Dsym下降46.05%；FP_OFF相对BF_OFF的Dsym也下降约48.88%。两格FP32都从带扶手样式转为无扶手，这个变化不依赖readout开启。','',
'这里还存在反向代价：B FP_ON的GT→F MSE增加25.33%，F→GT MSE下降74.41%，512查询coverage从50.9766%降至40.0391%。因此更小的双向均方距离没有代表更完整的参考覆盖；I从0增到0.3906%，小角从5.0761%增到6.5990%。同为FP32时，B ON的Dsym/S/I/FF/coverage均没有优于OFF，且ON图中座下三角尖片更显眼。来源距离不是开放生成唯一标准，这组结果不能包装为readout控制成功。','',
'C/f596四格大体都是无扶手高背椅。FP32_ON相对BF16_ON的参考S略高、FF持平，但ON的Dsym从0.000586518升到0.000631304，coverage下降。分支Dsym效应从BF16的ON优于OFF翻成FP32的ON劣于OFF，S却相反；没有统一方向。','',
'**P2两面座板：变化小，主问题仍在。** 全部输出仍为低宽凳/沙发式主体，橙色面板覆盖或包围输入C；背板斜缝、基座开口和边缘尖片仍可见。此款式本身没有被判错，也没有要求必须四条腿。B FP_ON的I略升；C FP_ON的I下降0.7324个百分点、FF下降1.5789个百分点，但小角增加1.5789个百分点。C的OFF精度变化也降低I约0.8301个百分点。没有看到FP32开启readout后才出现的明确C使用改善；部分重层可能是厚度，保留语义不确定性。','',
'**P2座板+背板：最明显的是款式跳变，原C用途问题没有解开。** 所有四格仍在输入蓝背板的相对座边生成另一块橙色靠背，侧视相隔接近座深，不能简单解释成贴近的薄壁。普通椅子用途下不协调，特殊用途未做人评确认。B FP_OFF与C BF_OFF变成密条/实体弯背和更细支撑，其他多为梯形横档背；切换精度/分支会改变款式，但并未消除双端竖板。','',
'低I在这里也伴随代价：B FP_OFF的I为3.4668%，比FP_ON的8.0078%低，但小角面30.8511%，远高于FP_ON的4.2553%；C BF_OFF的I为3.3203%，小角28.7234%，相对BF_ON为7.9102%和4.2553%。这些是实测tradeoff，不能用一个低冲突值认定更好成品。','',
'所有逐输出指标及原分项/并集、上下界、p95/coverage、面积、退化和连接诊断见 rollout_metrics.csv 与evaluation逐例JSON；1120行 paired_rollout_comparisons.csv 保存branch_BF、branch_FP、precision_ON、precision_OFF、interaction。差分符号按原指标解释，不统一称负值改善。P1的S/E/来源距离标为REFERENCE_COMPLETION_DIAGNOSTICS；P2这几项保持NA，不创建目标。原固定C/GT查询及预测free采样随机流保持，新Q_F在各新表面重新映射。','',
'## D. 使用价值、成本与下一步决策','',
'**本轮没有建立“更忠实数值路径让现有readout更有用”的稳定证据。** 有真实数值改善、有完整轨迹差别，也有少数指标收益；但同精度ON未持续优于OFF，一些明显精度效应在OFF也出现，稀疏C的主要可见问题仍然存在。准确状态为MIXED_TRADEOFF；其含义是小面板上的生成质量方向混合，不是统计显著结论。','',
'因此，**暂不将精度匹配训练作为优先投入，也不因这次结果放大残差或继续增加readout容量。** 当前readout尚未展示值得单凭数值舍入问题就扩大训练的条件作用。保留FP32_REFERENCE用于显式数值参照，维持当前START默认。这个决定只基于现有混合精度训练权重的推理证据：本轮没有做FP32训练，不能证明BF16导致过去训练失败，也不能排除别的条件任务组织将来有效。','',
'成本记录如下，都是本轮实际生成，未额外测速。每条含50步和逐步日志/数组保存；没有把探针或并发CPU开销当部署性能。','',
'| 路径 | 新轨迹数 | 每条总耗时均值 s | 50次前向计时之和均值 s |','|---|---:|---:|---:|']
for c in cost:s.append(f"| {c['precision']} {c['readout']} | {c['n']} | {c['wall_mean']:.3f} | {c['forward_mean']:.3f} |")
s+=['',
'该小N面板下未出现明显FP32成本障碍，但这些单次带日志测量不能证明高精度部署更快。最大rollout PyTorch allocated=607,490,048 bytes、reserved=660,602,880 bytes；二者分列且不能相加，非进程VRAM。设备总显存/利用率/CPU/RAM逐阶段记录，进程显存不可得记NA。load峰不在每条重置后的PyTorch窗口内；阶段末rollout integrity峰只是最后重置窗口，不能当整阶段峰。评分为2进程×2线程、29.544秒，渲染7.668秒，AI实际审阅与其分开记录。','',
'## 可恢复性、源码及交付','',
'- 正式数学路径只对native/readout增加显式可选开关，新precision模块和三个工具入口、8项小CPU测试；原backbone/评分源码/默认sample没有改动。源码86文件完整hash与7项新增/修改diff保留。',
'- 原独立repo没有HEAD，已有未提交源码全部保留。原E:/MeshFlow只读；没有reset/clean、commit/push、依赖升级、训练、模型晋级、后处理或第三种精度。',
'- 纯CPU准备时修复一次父目录缺失；审计时修复旧dtype日志字段解析。另12个preflight记录的辅助cell_id后缀写成off，真实call_id、mode、数组配对均正确，保留原件并披露。以上不影响科学调用，零模型重试。',
'- 总入口八阶段见 README_RESUME.md。独立核验复查了1260调用闭合、24新轨迹的实际递推、32输出及所有受保护身份。完成后再次调用rollout入口只验证并跳过，前向/持锁账本hash不变。',
'- 原369MiB BF16层捕获引用复用，没有复制。新16层完整捕获、其余完整统计/来源hash、24条51态/50速度/RAW/PLY、逐例分数、完整配对和实际看图记录均保留。','',
'主要入口：','',
'- [打开完整四格图库](OPEN_GALLERY.html) · [32条实际视觉审阅](visual_review.csv)（AI_VISUAL_REVIEW；human usability NOT_RUN）。',
'- [192条逐层表](layer_precision_metrics.csv) · [24个固定状态四格](fixed_state_pairs.csv) · [预测精度交互](prediction_precision_interactions.csv)。',
'- [32份完整生成评分](rollout_metrics.csv) · [所有逐指标四格比较](paired_rollout_comparisons.csv) · [评分分解说明](evaluation/FOUR_CELL_FINDINGS.md)。',
'- [冻结计划](plan.json) · [权重/输入身份](checkpoint_and_input_manifest.json) · [精度合同](precision_contract.json) · [实际dtype审计](dtype_audit.json)。',
'- [预算](budget.json) · [独立完成核验](independent_verification.json) · [恢复测试](completed_resume_check.json) · [代码diff](code.diff) · [源码hash](source_hashes.csv)。',
'- [总入口](E:/MeshFlow-Control/tools/run_readout_precision.py) · [实际数值分析代码](E:/MeshFlow-Control/tools/readout_precision_analysis.py) · [原评分器调用及渲染适配](E:/MeshFlow-Control/tools/readout_precision_evaluation.py)。','',
'固定任务完成并停止。未自动启动精度匹配训练或任何新实验。','']
(r/'READOUT_PRECISION_PATH_AND_ROLLOUT_REPORT.md').write_text('\n'.join(s),encoding='utf-8')
print(json.dumps(dict(report='READOUT_PRECISION_PATH_AND_ROLLOUT_REPORT.md',lines=len(s),bytes=(r/'READOUT_PRECISION_PATH_AND_ROLLOUT_REPORT.md').stat().st_size)))