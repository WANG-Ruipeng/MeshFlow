import json
from pathlib import Path
root=Path('output/a_bottle_condition14_once_20260925_01')
v=json.loads((root/'FINAL_VERIFICATION.json').read_text()); c=json.loads((root/'COMPLETE.json').read_text()); p=json.loads((root/'RUN_PROTOCOL.json').read_text())
arms=['fixed','correct_3k','mismatched_3k']; names={'fixed':'Fixed','correct_3k':'3k 正确条件','mismatched_3k':'3k 错配条件'}
def fmt(x):return '未达标' if x is None else f'{x:.3f}'
def yes(x):return '通过' if x else '未通过'
lines=['# 瓶类14对象：冻结3k初始化的单次验证','',
'42/42次完成，每对象三路各一次，没有重试、追加训练或换对象。正式结束于2026-09-25 15:02:33（北京时间）。',
'',
'**终点覆盖收益保留，但预先指定的4秒主指标没有显示正确条件3k优于Fixed；正确条件的额外价值也未得到本轮支持。**',
'',
'| 指标 | Fixed | 3k正确条件 | 3k错配条件 |','|---|---:|---:|---:|']
for b in [2,4,8]:
    lines.append('| '+str(b)+'秒当前审核通过 | '+' | '.join(f"{v['summary'][a]['current_pass'][str(b)]}/14 ({100*v['summary'][a]['current_pass'][str(b)]/14:.1f}%)" for a in arms)+' |')
lines.append('| 16步终点通过 | '+' | '.join(f"{v['summary'][a]['final_pass']}/14 ({100*v['summary'][a]['final_pass']/14:.1f}%)" for a in arms)+' |')
lines.append('| 完整耗时跨对象中位数（秒） | '+' | '.join(fmt(v['summary'][a]['total_seconds_median']) for a in arms)+' |')
lines.append('| 完整耗时最小–最大（秒） | '+' | '.join(f"{v['summary'][a]['total_seconds_min']:.3f}–{v['summary'][a]['total_seconds_max']:.3f}" for a in arms)+' |')
lines+=['',
'4秒是冻结主指标，2/8秒是辅助指标。每对象每路只有一次墙钟观察；表中的中位数是在14对象之间计算，不是三次计时的中位数，也不是成功子集的平均加速比。',
'',
'正确条件终点比Fixed增加2/14（14.29个百分点），同时4秒通过少1/14（7.14个百分点）。错配条件也达到14/14，并在本次4秒观察达到13/14。因此本轮不能归因成“正确的形状条件带来覆盖收益”，也不能宣传同质量稳定加速。',
'',
'正确条件与错配条件的14个共同成功对象，观测TTQ恰好7个更早、7个更晚。正确条件相对Fixed的12个共同成功对象，3个更早、9个更晚，另有2个只有正确条件在16步内达标。单次计时不能据此确认稳定逐对象速度排名。',
'',
'完整耗时先逐对象做配对差再取中位数：正确条件−Fixed为+0.060588秒；正确条件−错配为−0.005618秒。这些值仅描述本次观察。',
'',
'| 父对象UID | Fixed：TTQ / 完整秒 / 终点 | 正确3k：TTQ / 完整秒 / 终点 | 错配3k：TTQ / 完整秒 / 终点 | 4秒通过（Fixed/正确/错配） |',
'|---|---|---|---|---|']
for row in v['rows']:
    cells=[row['uid']]
    for a in arms:
        z=row['arms'][a];cells.append(f"{fmt(z['TTQ_seconds'])} / {z['total_seconds']:.3f} / {yes(z['final_pass'])}")
    cells.append(' / '.join('是' if row['arms'][a]['at_seconds']['4']['pass'] else '否' for a in arms))
    lines.append('| '+' | '.join(cells)+' |')
lines+=['',
'TTQ是0/1/2/4/8/16审核点中首次审核完成并通过联合合同的真实累计墙钟时间，包含此前所有审核，不是连续时间首次命中。4秒覆盖读取当时最近完成的审核结果；未完成审核或未达标仍留在14分母。全部任务跑满16步。',
'',
'本轮Fixed失败的两个对象是20c5…和e101…。其两次RMS改善分别只有1.420%/1.519%与0.071%/0.074%，都没有达到2%门槛；42个终点网格均为512面且通过精确有效性与双角度门槛。',
'',
'正确条件在20c5…的改善为2.267%/2.566%，距RMS门槛只有0.267/0.566个百分点；p05角度余量约2.41e−6度。这个新增成功接近约束边界，不能写成拥有宽裕质量余量。错配在同一对象的改善为17.456%/17.409%。e101…正确条件改善3.601%/3.470%，错配为9.394%/9.024%。',
'',
'[42份终点的四个门槛余量和绝对RMS](QUALITY_MARGINS_zh.md)保留了所有对象，未据余量改阈值。角度余量是实测角度减Fixed门槛，协议容差仍为1e−6度。41/42个终点的p05余量小于0.001度，表明优化结果靠近约束边界；本次没有测试数值扰动鲁棒性。',
'',
f"真实计时范围：三臂42次case累计{sum(v['summary'][a]['sum_case_seconds'] for a in arms):.3f}秒；driver墙钟{c['total_driver_wall_seconds']:.3f}秒（约{c['total_driver_wall_seconds']/60:.2f}分钟），包含资源等待/探测、落盘及逐次报告；一次性合成预热{c['one_time_warmup_seconds']:.3f}秒单列。case计时为暖CPU进程，从准备好的主场读取开始，计入每case数据/支持结构/权重/条件准备、初始化、恢复与优化、Fixed基准审核及六个观察点审核、最终目标半径查询；磁盘持久化及协议哈希预检在case外。225对象筛选79.214秒与原始网格转主场属于离线阶段，不计入case延迟。",
'',
'核验：35项相关测试通过；42次运行错误0；294次完整审核、714份迭代快照槽位；42次资源前后边界全部通过，但未认证连续资源隔离。独立从原始TASK复算通过率、TTQ、RMS改善，并逐项核对保存网格的目标主场半径；源码和输入前后哈希一致。',
'',
'范围：225个本地官方train瓶子按冻结几何规则只有14个合格（6.22%），全部纳入。结果仅覆盖这14个合格输入的同主场512面离散，不是原始瓶子资产保真度或完整生成质量；不代表其余211个输入。未重新证明原A预训练所有父对象的完整不重叠身份，因此不作所有方法均未见的测试集声明。',
'',
'研究判断：可以保留“已有学习初始化在合格瓶子上提高16步终点覆盖”的探索性正结果；本轮主时间指标和错配控制均不支持将贡献升级为条件自适应加速。14/14对象的正确/错配条件都产生了不同的代理点、网络特征和原始初始化，因此不能解释为网络没有读取条件。更好的通用起点可以保留为候选解释，普通扰动效应也尚未排除；后续需要固定通用布局或匹配幅度的非学习扰动等直接对照，不能由一个错配置换直接证明。当前不追加长训、不恢复官方基线长训。',
'',
'[独立核验和全部逐对象数字](FINAL_VERIFICATION.json) · [机器可读摘要](RESULT_NUMBERS.json) · [冻结运行协议](RUN_PROTOCOL.json) · [逐次原始记录](cases/) · [原始汇总](SUMMARY.json)',
'',
'复现入口（必须使用新的输出目录，脚本拒绝覆盖已开始的实验）：',
'```bash',
'python -m experiments.a_radial.time_quality_followup.bottle_ttq prepare --selection-run output/a_bottle_selection14_20260925_01 --old-run output/a_time_quality16_20260925_01 --study output/a_bottle_study_20260925_01/STUDY_PROTOCOL_v3.json --out output/NEW_BOTTLE_RUN',
'python -m experiments.a_radial.time_quality_followup.bottle_ttq gate --protocol output/NEW_BOTTLE_RUN/RUN_PROTOCOL.json --old-gate output/a_condition_wrapper_gate_20260925_01/REAL_PARITY.json --tests-xml output/a_bottle_adapter14_test_20260925_01/pytest.xml --out output/NEW_BOTTLE_RUN/TEST_GATE.json',
'python -m experiments.a_radial.time_quality_followup.bottle_ttq run --protocol output/NEW_BOTTLE_RUN/RUN_PROTOCOL.json --gate output/NEW_BOTTLE_RUN/TEST_GATE.json',
'```','']
(root/'RESULT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
q=['# 全14对象、三路终点质量余量','',
'RMS改善余量=改善率−2%，单位为百分点；角度余量=实测−Fixed门槛，单位度。绝对RMS单位为参考包围盒尺度百分比。原始未四舍五入数值见FINAL_VERIFICATION.json。',
'', '| UID | 路线 | 终点 | 两次绝对RMS | 两次RMS门槛余量（百分点） | 最小角余量（度） | p05角余量（度） |','|---|---|---|---|---|---|---|']
for row in v['rows']:
    for a in arms:
        z=row['arms'][a];m=z['final_margins'];q.append('| '+' | '.join([row['uid'],names[a],yes(z['final_pass']),'/'.join(f'{x:.6f}' for x in z['final_RMS_bbox_percent']),'/'.join(f'{x:.6f}' for x in m['rms_gain_margin_percentage_points']),f"{m['angle_margin_degrees']['min_triangle_angle_deg']:.9g}",f"{m['angle_margin_degrees']['triangle_min_angle_p05_deg']:.9g}"] )+' |')
(root/'QUALITY_MARGINS_zh.md').write_text('\n'.join(q)+'\n',encoding='utf-8')
n={k:v[k] for k in ['parents','attempts','timing_repeats','audit_calls','available_snapshot_slots','resource_flag_count','summary','paired','rows','protocol_sha256']}
n.update(driver_wall_seconds=c['total_driver_wall_seconds'],case_wall_sum_seconds=sum(v['summary'][a]['sum_case_seconds'] for a in arms),offline_screen225_seconds=79.214,source_screen_counts={'screened':225,'eligible':14,'topology_reject':149,'ray_reject':61,'exact_reject':1},interpretation='Endpoint coverage signal retained; predeclared4s primary and condition contribution not supported; single timing observations only')
(root/'RESULT_NUMBERS.json').write_text(json.dumps(n,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
statuspath=Path('output/a_bottle_study_20260925_01/STATUS.json');status=json.loads(statuspath.read_text(encoding='utf-8-sig'));status.update(status='COMPLETE_INDEPENDENT_VERIFICATION_PASS',formal_TTQ_cases_executed=42,model_inferences_on_bottles=28,formal_run=str(root),result=str(root/'RESULT_zh.md'),runtime_errors=0,resource_boundary_exceedances=0,primary4s_pass_counts={a:v['summary'][a]['current_pass']['4'] for a in arms},final_pass_counts={a:v['summary'][a]['final_pass'] for a in arms});statuspath.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
portal=Path('/mnt/e/MeshFlow/output/BOTTLE_CONDITION_STUDY_20260925.md')
portal.write_text('# A 瓶类冻结模型验证已完成（2026-09-25）\n\n**14对象×三路×单次，共42次全部完成。** 原225筛选14合格，使用全部14；不追加训练。\n\n| 指标 | Fixed | 正确3k | 错配3k |\n|---|---:|---:|---:|\n| 4秒主指标 | 12/14 | 11/14 | 13/14 |\n| 16步终点 | 12/14 | 14/14 | 14/14 |\n| 完整耗时跨对象中位数 | 5.654秒 | 5.839秒 | 5.828秒 |\n\n终点覆盖有提升，但4秒主指标和正确条件的额外价值均未获支持。单次计时不证明稳定加速。42次运行异常0、资源前后超限0；独立核验PASS。官方长训保持暂停。\n\n[完整14行账本与结论](../.worktrees/a-layout-value-20260924/output/a_bottle_condition14_once_20260925_01/RESULT_zh.md) · [质量余量](../.worktrees/a-layout-value-20260924/output/a_bottle_condition14_once_20260925_01/QUALITY_MARGINS_zh.md) · [独立核验](../.worktrees/a-layout-value-20260924/output/a_bottle_condition14_once_20260925_01/FINAL_VERIFICATION.json) · [225行筛选账本](../.worktrees/a-layout-value-20260924/output/a_bottle_study_20260925_01/SCREEN_LEDGER225.md)\n',encoding='utf-8')
print(json.dumps({'reports':'written','rows':len(v['rows']),'quality_rows':42,'driver_seconds':c['total_driver_wall_seconds']}))