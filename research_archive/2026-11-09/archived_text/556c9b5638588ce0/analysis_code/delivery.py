from core import *
import shutil,difflib,zipfile,platform

def fmt(x,d=6):
 if x is None:return 'NA'
 if isinstance(x,bool):return str(x)
 if isinstance(x,(int,np.integer)):return str(x)
 if isinstance(x,(float,np.floating)):return f'{x:.{d}g}'
 return str(x)
def table(headers,rows):
 return '\n|'+'|'.join(headers)+'|\n|'+'|'.join('---' for _ in headers)+'|\n'+''.join('|'+'|'.join(fmt(x).replace('|','/') for x in r)+'|\n' for r in rows)+'\n'

def supplemental_checks():
 records=recs();w=readcsv(OUT/'width_per_output.csv');crossledger=js(SRC/'analysis_ledger.json');checks=[];exported=[]
 # Bank query identities and correspondence: actual transformed frozen points, not new draws.
 for key,rs in groups(w,['arm','phase_step','uid','noise_index']).items():
  ix={r['alpha']:r for r in rs};r1=records[ix[1.]['case_id']];b1=arrays(r1['input_path']);q1=arrays(r1['control_points_path'])['points'];cid=rs[0]['curve_id']
  op=crossledger['operations'][cid+'_cross'];oldidentity=op['identity']
  for a in ALPHAS:
   r=ix[a];record=records[r['case_id']];cp=resolve(record['control_points_path']);bank=arrays(record['input_path']);query=arrays(cp)['points'];expected=q1.copy();expected[:,0]*=np.float32(a)
   assert np.array_equal(query,expected) and digest(cp)==record['control_points_sha256']
   assert oldidentity[str(a)]==r['raw_sha256']
   checks.append(dict(curve_id=cid,alpha=a,point_hash=ah(query),Q1_hash=ah(q1),exact_transported_query=True,query_file_hash_match=True,old_cross_ledger_raw_identity_match=True,C_ids_same=ah(bank['source_face_ids'])==ah(b1['source_face_ids']),free_slots_same=ah(bank['free_source_ids'])==ah(b1['free_source_ids'])))
   if key[1]==3000:
    raw=arrays(r['raw_path']);K=int(bank['K']);path=OUT/'evidence_arrays'/f'{cid}_a{int(round(a*100))}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,free=raw['raw'][K:].reshape(-1,3,3),C=bank['C'],GT_free=bank['gt_free'].reshape(-1,3,3),Q=query,L0=bank['L0'],L_alpha=bank['L'],source_face_ids=bank['source_face_ids'],free_source_ids=bank['free_source_ids'],noise_sha256=np.array(ah(raw['noise'])),alpha=np.array(a))
    exported.append(dict(file=str(path.relative_to(OUT)),sha256=digest(path),bytes=path.stat().st_size,source_raw=r['raw_path'],source_raw_sha256=r['raw_sha256'],scope='copied existing endpoint arrays, no generated candidate'))
 # Reconcile exact original alpha-wise summary rather than treating brief as source truth.
 os=readcsv(SRC/'B_width_summary.csv');old=readcsv(SRC/'B_width_response.csv');summarychecks=[]
 metricmap={'slope':'b_self','slope_abs_error':None,'X_span_abs_error_L0':'actual_GT_abs_error_L0','response_abs_residual':None}
 for r in os:
  sub=[x for x in w if all(x[k]==r[k] for k in ('arm','phase_step','split','alpha'))]
  for m,newkey in metricmap.items():
   if newkey:val=mean(x[newkey] for x in sub)
   elif m=='slope_abs_error':val=mean(abs(x['b_self']-1) for x in sub)
   else:val=mean(abs(x['r_alpha']) for x in sub)
   assert close(val,r[m]);summarychecks.append(dict(arm=r['arm'],phase_step=r['phase_step'],split=r['split'],alpha=r['alpha'],metric=m,original=r[m],raw_derived=val,difference=val-r[m],match=True))
 savej(OUT/'evidence_arrays/manifest.json',exported)
 arraysreadme='这些是原B终点144个输出的free/C/GT/Q小型副本；L0、源面ID及噪声字节hash一并保留。没有保存新的Gaussian，没有模型调用。完整原raw含状态轨迹仍只读留在源目录。manifest记录原raw及导出副本SHA256。\n'
 (OUT/'evidence_arrays/README.md').write_text(arraysreadme,encoding='utf8')
 # Missing means missing: expected applicability gaps and unmeasured phases have explicit NA.
 missing=readcsv(OUT/'A_missing_NA.csv')+readcsv(OUT/'width_missing.csv')
 for arm in ARMS:
  for phase in (1000,2500):
   missing.append(dict(module='A_train_generation',arm=arm,phase_step=phase,status='NA',reason='Original protocol did not measure train intermediate generation; no interpolation'))
   missing.append(dict(module='B_intermediate_generation',arm=arm,phase_step=phase,status='NA',reason='No original intermediate B generation; no replacement outputs'))
 missing.append(dict(module='auxiliary_independent_audit',status='NA_OPTIONAL',reason='Exact CHAIR_BRIEF_INDEPENDENT_AUDIT_20261003.md not found; local followup report exists and is only secondary corroboration'))
 for c in readcsv(OUT/'width_per_curve.csv'):
  if c['phase_step']==3000 and c['noise_index']==1:missing.append(dict(module='B_phase0_noise1',arm=c['arm'],uid=c['uid'],status='NA',reason='Only first-noise phase0 was registered; no paired phase0 noise1'))
 outs=[writecsv('query_and_curve_identity_audit.csv',checks),writecsv('original_B_summary_reconciliation.csv',summarychecks),writecsv('missing_and_NA.csv',missing),'evidence_arrays/manifest.json','evidence_arrays/README.md']
 return dict(query_identity_checks=len(checks),original_B_summary_checks=len(summarychecks),endpoint_small_arrays=len(exported),blocked=sum(r['status']=='BLOCKED' for r in missing)),outs

def write_report():
 get=lambda n:readcsv(OUT/n)
 widths=get('width_summary.csv');curves=get('width_per_curve.csv');noise=get('noise_paired.csv');seed=get('width_by_noise.csv');ph=get('width_phase_paired.csv');cross=get('cross_summary.csv');ca=get('cross_all_targets_summary.csv');cp=get('cross_model_paired.csv');coverage=get('coverage_summary.csv');covpar=get('coverage_per_parent.csv');covpair=get('coverage_paired.csv');part=get('coverage_partition_summary.csv');pull=get('pullback_summary.csv');pc=get('pullback_per_case.csv');tea=get('teacher_recomputed_summary.csv');dec=get('error_decomposition_macro.csv');ana=get('analytic_curve_comparison.csv');dispatch=js(OUT/'CPU_dispatch.json')
 ep=[r for r in widths if r['phase_step']==3000];geo_train=[r for r in noise if r['arm']=='Geo' and r['split']=='train']
 checks=js(OUT/'stages/02_width.json')['result'];endv=[r for r in curves if r['phase_step']==3000 and r['split']=='validation']
 report=['# Chair 条件控制逐例归因报告\n',
 '**本轮已实际完成固定离线任务。** 216 个 B 原输出、72 条真实响应曲线、648 个原矩阵单元、270 条 A 固定查询点距离向量和288行 teacher 原表全部可读。新增几何计算只有96个终点逆仿射候选（每个两类距离）及一个小型点到三角面 fixture；模型加载、前向、反向、训练更新、OT、新Gaussian、新模型生成、CUDA初始化均为0。核心分析没有 BLOCKED；18个A40几何不适用请求及未测中间/第二噪声起点保留NA。\n',
 '**能确定的结论：** 旧3×3单元分数无误，但旧“correct-column”解释用了固定输出行；应更新按目标的条件优势结论。Geo训练均值确实混合了明显不同的两噪声响应。验证组基准自由宽度偏窄的证据由raw直接支持；撤销X变换后，完整自由几何到GT的距离仍高。Geo覆盖收益有明确适用范围；Joint在本面板没有同时保住覆盖、条件优势与相交质量，局部接口改善仍成立。\n',
 '所有汇总先在曲线/case计算，再父内平均噪声或alpha，最后父等权。Native=T1+FM；Native+Geo=T1+Geo+FM；Native+Geo+Joint=T1+Geo+JEdge5。三臂各自从MVP48累计1000分叉，本阶段新增3000、终点4000；不是旧双对象JEdge5_3000、MVP48_5000或fit2。旧账本确认9000次历史更新、582条历史生成、288次teacher；本轮只读这些记录。\n',
 '## Q1：3×3到底测了什么，哪些旧结论需要更新？\n',
 '`E[alpha_in,beta_target]` 的行是生成条件，列是目标。原 `width_analytics.py` 在 `matching = [r for r in cross if r["alpha"]==alpha]` 固定输出行，再取非对角目标均值。因此旧 `wrong_mean_minus_diagonal_RMS_L0`、两个diagonal-win字段均为行口径。648格CSV与原评分JSON完全一致，原代码与冻结code.diff一致，原几何评分器hash与protocol一致：**单元分数及行统计计算核对通过；将其称为固定目标列的条件利用是解释/汇总轴错误。** 原文件不变，本目录新增列统计。\n',
 '`A_column(b)=mean_{a!=b}E[a,b]-E[b,b]` 回答同一目标下正确C是否有益。`A_row(a)=mean_{b!=a}E[a,b]-E[a,a]` 只回答同一输出偏好哪个目标。216项旧值均逐项吻合行公式；改用列公式后，117项优势符号、117项严格胜负改变。单元分数、宽度和teacher结论不因此失效；旧按alpha的“正确条件赢/输”及由此推出普遍向心收缩的论证需撤回。\n',
 '终点列优势（%L0，正数为正确输入较好）及严格同时胜过两个错误输入的比例如下。每格仅4父×2噪声。\n']
 report.append(table(['模型','划分','目标beta','列优势 %L0','严格列胜率','旧行优势 %L0'],[[NAMES[r['arm']],r['split'],r['alpha_beta'],100*r['A_column'],r['column_strict_win'],100*r['A_row']] for r in cross if r['phase_step']==3000]))
 report += ['训练组 beta=1.1 的三臂列优势均为正，原行优势却均为负，不能把旧现象直接解释为扩张条件普遍无效。验证组 beta=1.1 的平均列优势仍为负，但每臂有4/8条曲线严格列胜；存在逐父/噪声异质性。三目标的平均行优势与平均列优势相等是代数恒等式，不是新增独立证据。\n',
 'No-response正控复用每组真实F1的三个原距离，重复成三行，并核对原No-response解析候选raw及对应评分：72组每列优势全部为0、无严格列胜。相同行[1,2,3] fixture的行优势为[1.5,0,-1.5]，列优势为[0,0,0]；全等、零对角、行列标签置换等8类fixture均通过。正控是解析解释，不能算模型性能。原点集是Q1运输后的固定点，属于单向诊断，并非三个形变表面各自均匀的双向RMS。\n',
 '证据：[完整矩阵](cross_matrix.csv)、[逐项方向审计](cross_direction_audit.csv)、[真实No-response审计](no_response_direction_audit.csv)、[源码行定位](metric_source_locations.csv)、[8类CPU测试](cpu_fixture_results.json)。\n',
 '## Q2：真实父/噪声曲线是否支持此前均值推导？\n',
 '支持，但只支持这组登记数据：Geo训练 b_self 的noise0父均值为0.431636，noise1为1.244131，两者均值0.837884。3/4训练父从欠响应跨到过响应；第一个父两噪声都欠响应。不是所有父统一变好。下表来自真实曲线，不从宏平均反推。\n']
 report.append(table(['Geo训练父UID','p/g n0','p/g n1','b_self n0','b_self n1','b_GT n0','b_GT n1','b1−b0'],[[r['uid'],r['p_over_g_noise0'],r['p_over_g_noise1'],r['b_self_noise0'],r['b_self_noise1'],r['b_GT_noise0'],r['b_GT_noise1'],r['b_noise1_minus_noise0']] for r in geo_train]))
 report.append(table(['模型','划分','p/g','b_self均值','b_GT均值','abs(mean(b)−1)','mean(abs(b−1))'],[[NAMES[r['arm']],r['split'],r['p_over_g'],r['b_self'],r['b_GT'],r['abs_mean_b_minus_one'],r['b_self_abs_error']] for r in ep]))
 report += ['Geo训练平均斜率比Native更接近1，但逐曲线平均|b−1|约0.56677，大于Native的0.54334，不能据平均斜率宣布控制更稳。验证三臂各有2/4父的斜率在两噪声间变号。两份噪声不是两次独立训练复现，不能推断总体噪声分布或seed因果效应。\n',
 '216行raw的宽度/中位数与斜率共2808项核对无差异；72条曲线均检查N/K、源面集合、free槽位、实际Gaussian字节hash。所有单侧斜率、逆向、平局、单调及绝对变化保存在[逐曲线表](width_per_curve.csv)；[逐父噪声配对](noise_paired.csv)和[按噪声汇总](width_by_noise.csv)完整保留两组结果。相同斜率在三个alpha行只计一次。\n',
 '## Q3：基准偏差与相对响应怎样共同决定绝对误差？\n',
 '`e=(p_a-a*g)/L0 = a*(p-g)/L0 + (p_a-a*p)/L0 = base+response`；平方保留 `2*base*response`，不把两个平方项称为不重叠因果贡献。量纲为L0归一化；原绝对误差用实际存档GT宽度g_a，另外列 `GT_affine_roundoff=(a*g-g_a)/L0`，因此实际带符号误差为e+roundoff。原FP32目标变换造成的最大修正为'+fmt(checks['max_target_FP32_affine_residual_L0'])+' L0，预登记容差内；分解恒等式最大残差'+fmt(checks['max_exact_residual'])+'。未放宽对账容差。\n',
 '宽度严格沿用有效free面角等角权q95−q05，raw先转FP64，再NumPy linear quantile；不改成bbox，不按面积加权。L0来自未变形完整父bbox对角线；原重建分数仍用L_alpha，新诊断统一L0。alpha=1的相对残差恒为0只是定义。\n',
 f'验证终点三臂共{len(endv)}条曲线中，{sum(r["p_over_g"]<.9 for r in endv)}条均有p/g<0.9。这是已观察基准偏窄，不是从矩阵方向推断。Geo验证均值p/g=0.650821；基准项绝对值均值0.169173 L0，相对响应项绝对值均值0.018562 L0，但两者可抵消，不能相加解释最终绝对误差。\n',
 'Geo真正共有第一噪声的起终点比较，每划分4父，跨三个alpha平均：\n']
 pr=[]
 for split in ('train','validation'):
  rs=[r for r in ph if r['arm']=='Geo' and r['split']==split]
  for label,m in [('基准|p−g|/L0','base_abs_L0'),('相对残差平均绝对值','mean_abs_r'),('响应项平均绝对值/L0','mean_abs_response'),('最终原GT绝对宽度误差/L0','mean_actual_GT_abs_error_L0'),('抵消量 |base|+|response|−|e|','mean_cancellation'),('平方展开交叉项均值','mean_cross_term')]:
   pr.append([split,label,mean(r[m+'_phase0'] for r in rs),mean(r[m+'_phase3000'] for r in rs),mean(r[m+'_delta'] for r in rs)])
 report.append(table(['划分','量','phase0','phase3000','变化'],pr))
 report += ['Geo验证响应残差减小，同时基准宽度变差、项间抵消减少，最终绝对宽度误差从0.139460升到0.156422 L0。验证父341bcae1…和534b0e6e…、训练父12c2b708…呈现“响应改善、基准与最终误差恶化”；训练父2ce61518…两项绝对量都减小但抵消也减少，最终误差可略升。这里是代数和配对描述，不是训练因果链。\n',
 '新增b_GT满足b_GT=b_self*(p/g)，所以基准偏窄时self斜率看似合理也不能保证实际GT尺度响应。X/Y/Z位置及Y/Z跨度误差均在[216行逐输出表](width_per_output.csv)；[平方/符号分解逐父表](error_decomposition_summary.csv)保留交叉项；[24组起终点配对](width_phase_paired.csv)只用同一Gaussian字节。\n',
 '## Q4：No-response与Post-scale平均误差相同，是否意味代码错？\n',
 '不意味着。对p<0.9g、对称alpha取值0.9/1/1.1，No-response的平均|p−a*g|和Post-scale的平均|a*p−a*g|都等于g−p，而两者响应斜率分别为0和1。验证终点24条曲线都属于这一偏窄情形，存档解析参照均值近似相同；实际Post-scale系数来自C1与C_alpha的点积拟合，不由GT free估计。\n']
 va=[r for r in ana if r['phase_step']==3000 and r['split']=='validation']
 report += ['这24条实际曲线中两解析方法平均绝对误差差值的最大绝对值为 '+fmt(max(abs(r['post_minus_no_mean_error_L0']) for r in va))+' L0。432个原解析候选的坐标使用原存档hat逐一核对；重新CPU点积的末位舍入另列，不替换原hat。解析方法使用编辑前C、编辑前输出与当前C，信息合同比只读当前C的生成器更丰富，不能作为GT驱动部署校正。\n',
 '证据：[解析坐标与hat审计](analytic_source_audit.csv)、[每曲线解析比较](analytic_curve_comparison.csv)。\n',
 '## 新几何诊断：撤销已知仿射后，剩下多少偏差？\n',
 '终点48曲线的两侧alpha共96候选。按真实存档变换系数float32(alpha)的倒数，仅离线撤销X缩放。每候选两类评价：与F1的对称表面RMS（两方向各512个固定协议面积样本）；统一Q1到pullback的单向固定点RMS/覆盖。原点到三角面实现、chunk=32；没有ICP、刚体对齐或GT槽位匹配。同预测槽坐标RMS仅作辅助。每候选的两种评价、两个方向及分母均入表。\n']
 report.append(table(['模型','划分','pullback↔F1 %L0','Q1→F1 %L0','Q1→pullback %L0','pullback覆盖','覆盖−F1 pp'],[[NAMES[r['arm']],r['split'],100*r['symmetric_RMS_L0'],100*r['F1_fixed_GT_RMS_L0'],100*r['fixed_GT_RMS_L0'],r['fixed_GT_coverage_fraction'],100*r['fixed_GT_coverage_minus_F1']] for r in pull]))
 report += ['验证三臂pullback到GT约7.62–7.67%L0，F1本身约7.42–7.56%L0；pullback与F1约1.01–1.16%L0。距离定义不同，不能相减做因果分摊，但这些数值与“共同基准形状偏差仍大”相容。它们并非精确仿射复现：表面差非零；不能称撤销后已正确。\n',
 '训练组Geo的pullback↔F1约1.92%L0，为三臂中较大；pullback固定点RMS比F1增加约0.120%L0、覆盖下降4.785个百分点。Joint训练pullback到GT的平均RMS比F1反而低约0.302%L0，同时覆盖下降6.030个百分点。这一局部收益应保留，也再次说明RMS与coverage不可互换。未经阈值定义不把“近/远”转成成功率。\n',
 '证据：[96个逐候选结果](pullback_per_case.csv)、[逐父结果](pullback_per_parent.csv)、`pullback_cache/`中的真实距离/采样点、`figures/geometry_*`固定相机图。图的对象按每划分UID字典序第一父、noise0预登记，非按结果挑图。\n',
 '## Q5：Geo覆盖增加具体来自哪些查询点，距离尾部怎样变化？\n',
 'A的270条原评分JSON已保存每个固定查询点到free的距离，本轮全部复用且核对查询点hash、GT补集与原分数；新增A距离计算为0。coverage用独立seed=9973的512点，原R12→free RMS用seed=9911的另一组点，二者hash不同。本节明确命名fixed_query_MSE，不能冒充原R12 RMS。\n',
 '同GT、同Q、同输入噪声的每对分为both_in/enter/leave/both_out，阈值沿用d≤0.01L。每组MSE贡献为sum((d_new/L)^2−(d_old/L)^2)/512。656组分区贡献与164对总差的最大闭合残差为1.735e−18；每对enter−leave也精确等于覆盖差。下表为Geo−Native父等权增量，MSE列乘10,000方便阅读。\n']
 gc=[r for r in coverage if r['comparison']=='Geo-Native']
 report.append(table(['划分','K比例','父数','覆盖 pp','fixed MSEΔ×1e4','enter贡献×1e4','both_out贡献×1e4','p95距离Δ %L','覆盖↑且MSE差父数'],[[r['split'],r['ratio'],r['n_parents'],100*r['coverage_fraction_delta'],1e4*r['fixed_query_MSE_delta'],1e4*r['enter_MSE_contribution'],1e4*r['both_out_MSE_contribution'],100*r['d_p95_L_delta'],r['coverage_up_MSE_worse_parents']] for r in gc]))
 report += ['训练40%是明确的局部正结果：进入阈值的点和仍在阈值外的点都降低MSE，覆盖+6.138个百分点、p95尾部下降1.229%L。其余三组里，新进入点带来负MSE贡献，但both_out的正贡献更大或伴随leave代价，覆盖上升与平均MSE恶化可以同时发生。\n',
 '这不只是父对象平均造成的表象：Geo−Native在训练20%、验证20%、验证40%分别有2/8、4/8、3/6父同时覆盖上升而fixed MSE变差；对应原R12指标为3/8、4/8、3/6。两个计数的差别正是点集不同，不能混用。验证20%的父33ec1e64…覆盖+0.977个百分点，却有MSEΔ=0.00109417，其中both_out贡献0.00108435；341bcae1…的MSEΔ=0.000757824，both_out贡献0.000765573。完整逐case/逐父和坐标向量保留，可定位哪些固定点付出代价，而非仅按宏平均推测。没有剔除这些父。\n',
 '证据：[逐点分区表](coverage_partition.csv)、[逐case联表](coverage_paired.csv)、[逐父联表](coverage_per_parent.csv)、`coverage_vectors/`与`coverage_pair_vectors/`（含Q坐标、距离、分区标签及逐点平方差）。四组比例、均值、MSE贡献均保留；空组均值为NA，不写0。\n',
 '## Q6：Joint是否同时保住覆盖、条件优势、接口与相交？\n',
 '终点Joint−Geo四组的覆盖均值均下降，fixed-query MSE均值均上升；但验证两组的p95尾部下降，验证接口有收益，不能概括成所有量都变差。以下距离为原评分%L，fraction差只乘一次100换成百分点。CF/FF数目与受影响面比例同时保留在逐case表，未重跑相交扫描。\n']
 jc=[r for r in coverage if r['comparison']=='Joint-Geo']
 report.append(table(['划分','K比例','覆盖 pp','原双向RMSΔ %L','Gamma边RMSΔ %L','Gamma边p95Δ %L','CF受影响 pp','FF受影响 pp','小角面 pp'],[[r['split'],r['ratio'],100*r['coverage_fraction_delta'],r['remainder_symmetric_RMS_bbox_pct_delta'],r['Gamma_to_free_edges_RMS_bbox_pct_delta'],r['Gamma_to_free_edges_p95_bbox_pct_delta'],100*r['CF_proper_affected_free_fraction_delta'],100*r['FF_illegal_affected_free_fraction_delta'],100*r['free_minimum_angle_lt5_fraction_delta']] for r in jc]))
 report += ['验证40%：Joint接口边RMS改善0.157536%L、CF受影响比例下降0.941097个百分点、小角面比例下降0.938436个百分点，同时FF受影响比例增加1.22395个百分点，覆盖下降0.667318个百分点。收益和代价同时存在。三目标合并的验证严格列胜率，Native/Geo/Joint为41.67%/37.50%/41.67%；训练为50.00%/45.83%/50.00%。Joint不是所有条件优势都下降，但没有保住整套几何指标。\n',
 'A20是两划分各8父；A40为训练7父、验证6父。源面集合、free补集和适用父不同，因此20%与40%的差不是纯信息量因果效果。共有父表单独列出。起终点A比较只用真正共有48条A20第一噪声；48条中间生成仅验证A20，未补造训练或B中间观测。C原留出只作为背景复用原配对表：历史终点相交代价仍需保留，本轮没有重生成或重评分C。\n',
 '## Teacher与已有曲线：原始数值核验\n',
 '原teacher 288行对应72条(臂×父×t)输入轨迹。按相同case_id与input SHA在phase0/1000/2500/3000配对，t=.2/.5/.8分开；288项原汇总与独立复算吻合。共享生成曲线90项也与真实A20 noise0数值吻合。下面的相对变化为“各格均值之比−1”，不混作每父相对变化均值。\n']
 trows=[]
 for split in ('train','validation'):
  for t in (.2,.5,.8):
   rs=[r for r in tea if r['arm']=='Geo' and r['split']==split and r['t']==t];ix={r['phase_step']:r for r in rs}
   trows.append([split,t,*[ix[p]['FM_error'] for p in (0,1000,2500,3000)],100*(ix[3000]['FM_error']/ix[0]['FM_error']-1),100*(ix[3000]['D_edge']/ix[0]['D_edge']-1),100*(ix[3000]['L5']/ix[0]['L5']-1)])
 report.append(table(['Geo划分','t','FM phase0','1000','2500','3000','FM变化 %','D_edge变化 %','L5变化 %'],trows))
 report += ['FM的t=.2训练下降/验证上升、t=.8训练上升/验证下降在三臂均值上成立，t=.5曲线非单调；不能把FM走势套到全部loss或全部父。teacher的GT状态输入、t、patch与生成面板不同，没有相同完整输入hash可作跨任务逐例因果关联。未重新执行teacher，未据曲线调整门控。详见[原数值重汇总](teacher_recomputed_summary.csv)、[每输入轨迹](teacher_per_input_curves.csv)、[同输入跨臂配对](teacher_input_paired.csv)。\n',
 '## Q7：唯一优先下一步及检验假说\n',
 '**优先做“基准自由形状学习”的受控修复验证。** 本轮已在独立结果中纠正评分解释；接下来的科学优先级应放在alpha=1时自由部分的尺度/位置与完整形状，而不是只把平均self斜率推向1。依据是：验证全部24条曲线基准偏窄，Geo第一噪声响应改善未抵消基准恶化，撤销仿射后到GT的距离仍约7.7%L0。噪声不一致和接口/相交代价应保留为约束指标，不自动等同于同一训练病因。\n',
 '**检验假说：** 如果基准形状偏差是当前验证绝对控制误差的主要可修复瓶颈，那么下一次另行批准、固定父/噪声/预算的单一受控修复中，alpha=1的|p−g|/L0、位置误差与统一Q1到自由面的距离应共同降低；其绝对宽度收益应主要体现在base项减小，同时列条件优势、response项、噪声差及CF/FF质量不出现对应退化。若只改善宽度标量而表面距离不降，或收益依赖更强抵消，则该假说不获支持。这里没有提出GT部署校正，也没有将分解项自动改成新loss。\n',
 '本轮不启动该验证，不训练、不改loss、不扫描求解器/NFE/alpha、不追加seed、不换默认模型。单训练seed、B每划分4父和2噪声只支持描述性归因，不支持普遍性、显著性或唯一病因；四个训练父的新C不是未见物体。\n',
 '## 执行、可恢复性与交付\n',
 f'CPU按观测递增至{dispatch["actual_parallel_workers"]}个进程。首个已授权候选观测峰值RSS={dispatch["first_actual_candidate_peak_RSS"]/1024**2:.1f} MiB；逆仿射阶段并发墙钟{dispatch["seconds"]:.3f}秒（包含IO/import，不能与隔离算时直接比较）。资源快照记录RTX5080的总/用/空闲显存、利用率和进程，以及24核CPU与RAM；本轮GPU工作进程0。每个worker独立输出、固定surface RNG，按注册候选ID排序聚合。\n',
 '入口为 `experiments/mf_h1_local0/chair_control_attribution_v1/run_all.py`。所有新结果仅在 `native_t1/runs/chair_control_attribution_v1/`；旧源码、CSV、raw、报告保持只读。`source_manifest.json`登记所有读取来源与hash/schema；结束前逐项重新核对，结果见`source_hash_verification.json`。权重仅查原manifest/metadata身份，未反序列化。\n',
 '实现期间两次标量预检中断已记账：旧code.diff使用普通unified diff而非git diff头；原Post-scale重新点积与存档hat存在约1e−16级CPU舍入。只修正新目录中的解析与对账方式，未重试模型/生成/评分，未提高科学预算或放宽容差。恢复入口按源hash和完成阶段hash跳过已完成阶段；pullback按候选、距离类型保存完成标记，已完成评分不重复执行。失败/未提交的几何类型禁止静默重放。\n',
 '必要文件均已包含：216行逐输出、72行逐曲线、24行逐父噪声配对、648格原矩阵与216项方向审计、平方/符号分解、96行pullback、656行覆盖分区、teacher逐行表、原B三份CSV、实际评分源码小文件、源码hash、新增code.diff、8类CPU测试、执行账本、缺失/NA清单和恢复命令。另附原B终点144份小数组、固定查询点/距离向量及全部固定图，无大权重或训练轨迹。\n',
 '关键入口：[指标合同](metric_contract.json)、[来源清单](source_manifest.json)、[源码hash](source_code_hashes.json)、[缺失/NA](missing_and_NA.csv)、[账本](execution_ledger.jsonl)、[完整预算](execution_accounting.json)、[恢复命令](restart.txt)、[图件清单](figures/manifest.json)。评分/测试PASS只表示流程与数值核验通过，不代表模型质量通过。\n']
 (OUT/'ATTRIBUTION_REPORT.md').write_text('\n'.join(report),encoding='utf8')
 return dict(report='ATTRIBUTION_REPORT.md',language='zh',recommendation='baseline free-shape learning controlled repair verification; not started'),['ATTRIBUTION_REPORT.md']

def finalize():
 # Before/after of all frozen input evidence and public source, no checkpoint tensor reads.
 changed=[];verified=[]
 for r in js(OUT/'source_manifest.json')['sources']:
  h=digest(r['path']);ok=h==r['sha256'];verified.append(dict(path=r['path'],before=r['sha256'],after=h,unchanged=ok))
  if not ok:changed.append(r['path'])
 savej(OUT/'source_hash_verification.json',dict(status='UNCHANGED' if not changed else 'FAIL',files=len(verified),changed=changed,entries=verified))
 if changed:raise RuntimeError('Protected source changed')
 assert 'torch' not in sys.modules
 stages={p.stem:js(p) for p in sorted((OUT/'stages').glob('*.json'))}
 counts=dict(model_loads=0,model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,new_Gaussians=0,new_generated_outputs=0,CUDA_initializations=0,GPU_workers=0,
  B_raw_outputs_recomputed=216,B_curves=72,original_cross_cells_reused=648,new_original_cross_geometry_cells=0,no_response_analytic_groups_reused=72,original_analytic_candidates_verified=432,
  A_fixed_distance_vectors_reused=270,A_new_vectors_computed=0,A_new_pair_intersection_scans=0,A_pairs=164,coverage_partition_rows=656,
  teacher_rows_read=288,new_teacher_calls=0,pullback_candidates=96,pullback_distance_types_per_candidate=2,pullback_point_triangle_calls=288,fixture_families=8,fixture_point_triangle_calls=1,
  core_blocked_modules=0,original_A_geometry_NA_requests=18,stages={k:dict(seconds=v.get('seconds'),status=v['status'],result=v.get('result')) for k,v in stages.items()})
 savej(OUT/'execution_accounting.json',counts)
 restart='wsl -e /home/warpwang/miniforge3/envs/meshflow/bin/python -B /mnt/e/MeshFlow/experiments/mf_h1_local0/chair_control_attribution_v1/run_all.py\n'
 (OUT/'restart.txt').write_text(restart,encoding='utf8')
 versions=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__)
 for name in ('trimesh','matplotlib','scipy','psutil'):
  mod=__import__(name);versions[name]=getattr(mod,'__version__','NA')
 savej(OUT/'runtime_versions.json',versions)
 sourcefiles=sorted(CODE.glob('*.py'));diff=''
 for p in sourcefiles:
  rel=str(p.relative_to(ROOT));diff+=''.join(difflib.unified_diff([],p.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+rel))
 (OUT/'code.diff').write_text(diff,encoding='utf8')
 codedest=OUT/'analysis_code';codedest.mkdir(exist_ok=True)
 for p in sourcefiles:shutil.copyfile(p,codedest/p.name)
 savej(OUT/'analysis_code_hashes.json',[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in sourcefiles])
 (OUT/'README.md').write_text('''# CHAIR_CONTROL_ATTRIBUTION_REVIEW

先读 ATTRIBUTION_REPORT.md；全部数字从原chair_domain_control_v1逐例raw/CSV/评分JSON导出。

- width_per_output.csv=216行；width_per_curve.csv=72；noise_paired.csv=24。
- cross_matrix.csv=648个原单元；cross_direction_audit.csv保留旧行解释和正确列比较；no_response_matrix.csv只是原F1三行重复的解析正控，不是新评分。
- coverage_partition.csv=656行（164配对×4组）；coverage_paired.csv/coverage_per_parent.csv保留接口、角质量、pair数和受影响面比例。coverage_vectors和coverage_pair_vectors含实际固定Q坐标和逐点数值。标签0/1/2/3=both_in/enter/leave/both_out。
- pullback_per_case.csv=96个离线候选；pullback_cache每候选仅两类距离，包含真实采样点和平方距离。原模型输出未改写。
- evidence_arrays含144份原B终点free/C/GT/Q及hash。original_tables保存原B逐例三表、teacher、A、原协议和清单。
- original_scoring_code为实际相关评分源码全文及trimesh最近点依赖源码，不是伪代码；source_code_hashes.json提供来源和hash。analysis_code与code.diff仅为独立新增代码。
- source_manifest.json与source_hash_verification.json记录源前后hash；每个打包文件另列package_manifest.json。
- 恢复命令见restart.txt；源数据在本机原位置。完成阶段和候选会校验hash后复用，不重复冒充新证据。

CSV中的距离/L是fraction；原bbox_pct是%L；coverage fraction乘一次100才是百分点。NA不等于0。教师与生成输入不同，不能当逐状态因果配对。所有模型/训练/OT/CUDA调用为0。
''',encoding='utf8')
 resources('final_completion');event('fixed_task_complete',counts={k:v for k,v in counts.items() if k!='stages'},protected_source_files=len(verified))
 savej(OUT/'RUN_COMPLETE.json',dict(status='COMPLETE',protected_source_files=len(verified),all_source_hashes_unchanged=True,model_calls=0,core_blocked=0,complete_stages=list(stages),stop_reason='fixed authorized offline task finished; no next experiment launched'))
 # Archives exclude only their own bytes, mutable local lock, and transient helper files.
 include=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('CHAIR_CONTROL_ATTRIBUTION_REVIEW.zip','CHAIR_CONTROL_ATTRIBUTION_REVIEW.zip.sha256','package_manifest.json','archive_verification.json','attribution_cpu.lock') and '.tmp' not in p.name]
 pm=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=digest(p)) for p in include]
 savej(OUT/'package_manifest.json',pm);include.append(OUT/'package_manifest.json')
 zpath=OUT/'CHAIR_CONTROL_ATTRIBUTION_REVIEW.zip'
 with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in include:z.write(p,str(p.relative_to(OUT)))
 with zipfile.ZipFile(zpath) as z:
  assert z.testzip() is None
  for r in pm:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
  must=['ATTRIBUTION_REPORT.md','width_per_output.csv','width_per_curve.csv','noise_paired.csv','cross_matrix.csv','cross_direction_audit.csv','error_decomposition_summary.csv','pullback_per_case.csv','coverage_partition.csv','original_tables/B_cross_condition_scores.csv','code.diff','source_code_hashes.json']
  assert all(n in z.namelist() for n in must)
 h=digest(zpath);(OUT/'CHAIR_CONTROL_ATTRIBUTION_REVIEW.zip.sha256').write_text(h+'  '+zpath.name+'\n',encoding='ascii')
 savej(OUT/'archive_verification.json',dict(status='PASS',zip_sha256=h,zip_bytes=zpath.stat().st_size,files=len(include),CRC='PASS',all_member_SHA256='PASS',required_members='PASS'))
 print(json.dumps(dict(status='COMPLETE',zip=str(zpath),bytes=zpath.stat().st_size,sha256=h,source_files=len(verified)),ensure_ascii=False),flush=True)

def deliver():
 stage('09_supplemental',supplemental_checks)
 stage('10_report',write_report)
 finalize()
