from pathlib import Path
import hashlib, json, subprocess
import xml.etree.ElementTree as ET
import numpy as np
from experiments.a_radial.io_utils import array_hash

ROOT = Path.cwd()
PLAN = ROOT/'output/a_condition16_plan_20260925_01'
EXT = ROOT/'output/a_fixed_extended4_20260925_01'
GATE = ROOT/'output/a_condition_wrapper_gate_20260925_01'
DATA = ROOT/'output/a_condition_data_audit_20260925_01'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value):
    assert not p.exists(), str(p)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def md(p, value):
    assert not p.exists(), str(p)
    p.write_text(value, encoding='utf-8')

summary = read(EXT/'SUMMARY.json')
assert summary == read(EXT/'COMPLETE.json')
assert summary['recorded_cases'] == summary['prefix_pass_count'] == 4
assert summary['technical_or_runtime_error_cases'] == 0
assert summary['source_inputs_unchanged'] is True
assert summary['no_training'] is True
ext_inputs = {}
array_count = audits = snapshots = 0
for row in summary['rows']:
    task_path = EXT/'cases'/row['uid']/'TASK.json'
    task = read(task_path)
    assert task['status'] == 'completed' and task['completed_full_budget']
    assert task['prefix_parity']['status'] == 'PASS'
    assert task['completed_full_audit_calls'] == task['planned_full_audit_calls'] == 9
    assert len(task['snapshot_slots']) == 65
    assert all(s['available'] and s['iteration']==i for i,s in enumerate(task['snapshot_slots']))
    assert [e['iteration'] for e in row['endpoints']] == [16,32,64]
    assert all(not e['target_pass'] and e['qualified'] and e['exact_valid'] and e['actual_faces']==512 for e in row['endpoints'])
    assert len({e['geometry_hash'] for e in row['endpoints']}) == 1
    assert row['finite_stall_diagnostic']['accepted_after16'] == 0
    arrays_path = task_path.parent/task['arrays_file']
    assert sha(arrays_path) == task['arrays_sha256']
    with np.load(arrays_path, allow_pickle=False) as z:
        assert set(z.files) == set(task['array_hashes'])
        for k in z.files: assert array_hash(z[k]) == task['array_hashes'][k], k
        array_count += len(z.files)
    audits += task['completed_full_audit_calls']; snapshots += len(task['snapshot_slots'])
    ext_inputs[str(task_path.relative_to(ROOT))] = sha(task_path)
    ext_inputs[str(arrays_path.relative_to(ROOT))] = sha(arrays_path)
assert audits == 36 and snapshots == 260
ext_check = {'status':'PASS', 'scope':'Artifact readback and completed diagnostic consistency; not new optimization or timing repeat',
 'cases':4, 'audits':audits, 'snapshot_slots':snapshots, 'typed_array_hashes_checked':array_count,
 'prefix_parity_cases':4, 'all_step16_32_64_mesh_hashes_unchanged':True,
 'registered_failure_count_at_each_endpoint':4, 'new_training_updates':0,
 'input_sha256':ext_inputs, 'summary_sha256':sha(EXT/'SUMMARY.json'),
 'script_sha256':sha(Path(__file__))}
write(EXT/'FINAL_VERIFICATION.json', ext_check)

parity = read(GATE/'REAL_PARITY.json')
assert parity['status']=='PASS' and parity['correct_condition_exact_parity_count']==16
assert parity['real_mismatched_target_radius_check']['target_query_exact']
assert parity['real_mismatched_target_radius_check']['donor_radius_negative_control_rejected']
wrapper = ROOT/'experiments/a_radial/time_quality_followup/conditioned_deployment.py'
assert sha(wrapper) == parity['wrapper_sha256']
assert sha(GATE/'real_parity.py') == parity['script_sha256']
suite = ET.parse(GATE/'pytest.xml').getroot().find('testsuite')
assert int(suite.attrib['tests'])==16
assert all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped'))
write(GATE/'SUMMARY.json', {'status':'PASS','synthetic_tests':16,
 'real_correct_condition_bitwise_parity_old_objects':16,
 'real_mismatch_target_radius_control':True,
 'scope':'Implementation only on old development data; no new-parent efficacy validation',
 'new_training_updates':0, 'readonly_review':'Independent agent inspected wrapper and parity records; PASS',
 'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [wrapper,GATE/'real_parity.py',GATE/'REAL_PARITY.json',GATE/'pytest.xml']}})

lines = ['# Fixed 四对象连续64步诊断\n',
'本轮只延长旧16开发对象中 Fixed 未达联合质量合同的4个对象；每对象连续运行一次64步，保持同一优化器状态演化。它不属于新对象验证或三重复墙钟实验。\n',
'**结果：16、32、64步均为0/4达标；所有对象在这三个端点的网格哈希一致。** 4/4保持512面、有效性和Fixed双角度门槛，但两次同场RMS改善都未达到2%。\n',
'| UID | 两次RMS改善（16/32/64相同） | 距2%门槛的余量（百分点） | 最后接受步 | 16步后接受数 | 本次完整流程秒 |\n|---|---:|---:|---:|---:|---:|\n']
for row in summary['rows']:
    gain=row['endpoints'][-1]['gain_percent']; stall=row['finite_stall_diagnostic']
    lines.append(f"| {row['uid']} | {gain[0]:.6f}% / {gain[1]:.6f}% | {gain[0]-2:.6f} / {gain[1]-2:.6f} | {stall['last_accepted_iteration']} | {stall['accepted_after16']} | {row['total_case_seconds']:.6f} |\n")
lines += ['\n| UID前缀 | 最小角余量（度） | p05角余量（度） |\n|---|---:|---:|\n']
for row in summary['rows']:
    e=row['endpoints'][-1]
    lines.append(f"| {row['uid'][:8]} | {e['min_angle_margin_deg']:.9f} | {e['p05_angle_margin_deg']:.9f} |\n")
lines += [
'\n四条轨迹在最后接受步以后到实际第64步均未改变方向。相邻15/16、31/32、63/64步的提案诊断相同，每步14个提案全部因quality拒绝。这个有限轨迹支持“当前配方增加步数未能让这四个对象追上”，不能证明全局不可行、永久局部最优或其他优化器也会失败。\n',
'前0至16步与原TTQ repeat0逐项复核：当前轨迹、已审核网格、最终incumbent、提案、有效性和两次MC结果一致（排除墙钟字段）。本轮保存260个迭代快照槽，完成36次全审核，无运行错误。最终回读已检查所有存档数组哈希。\n',
f"本次4个任务的driver墙钟为{summary['driver_wall_seconds']:.6f}秒；表中每对象总时间包含加载、参考/基线准备、优化和全部审核，存档与原轨迹对账在case计时外。这是单次有界诊断，不据此给出速度排名，也不与旧三重复的TTQ直接合并。\n",
'主结果：[SUMMARY.json](SUMMARY.json)；协议：[RUN_PROTOCOL.json](RUN_PROTOCOL.json)；最终回读：[FINAL_VERIFICATION.json](FINAL_VERIFICATION.json)。\n',
'\n复现命令（必须使用新的空输出目录，旧结果不可覆盖）：\n\n```bash\npython -B -m experiments.a_radial.time_quality_followup.extended_fixed prepare --source-run output/a_time_quality16_20260925_01 --out output/a_fixed_extended4_REPLICATION\npython -B -u -m experiments.a_radial.time_quality_followup.extended_fixed run --run output/a_fixed_extended4_REPLICATION\n```\n']
md(EXT/'RESULT_zh.md', ''.join(lines))

head = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert head == 'bd7d8a138465c583694f9f86083ddd2979d10f87', head
assert subprocess.run(['git','diff','--quiet','HEAD']).returncode==0
plan=read(PLAN/'STUDY_PLAN.json')
assert plan['target_selection']['uids'] is None and plan['primary_time_seconds']==4
status = {'status':'PARTIAL_DELIVERY_NEW16_DATA_PENDING', 'independent16_cases_executed':0,
 'independent16_cases_planned':144, 'source_and_parent_manifest_frozen':False,
 'frozen_model_and_primary_metric':True, 'independent16_driver_integrated':False,
 'completed':['four_old_fixed_cases_continuous64','condition_wrapper_synthetic16_and_real_old16_parity','local_source_and_history_audit'],
 'strict_new_eligible_same_source_parents':0, 'missing_parent_count':16,
 'pending':'User source/scope answer; no automatic change to category, split or geometric rules',
 'training_updates_this_followup':0, 'official_training_remains_paused':True,
 'official_head':head, 'tracked_official_tree_matches_HEAD':True}
write(PLAN/'STATUS.json', status)
report = '''# A 小规模后续进展（2026-09-25）

已完成旧四对象扩展步数诊断、条件错配封装与本地数据来源审计。**新的16父对象、三臂三重复实验尚未开始（0/144）**；当前缺的是独立合格数据，不是GPU算力或训练权重。

| 已执行项目 | 结果 | 证据 |
|---|---|---|
| 旧4个 Fixed 失败对象连续到64步 | 16/32/64步均0/4；端点网格完全相同，16步后无新接受提案 | [四行诊断与余量](../a_fixed_extended4_20260925_01/RESULT_zh.md) |
| 条件错配封装 | 16项测试通过；旧16对象正确条件输出与原实现及旧原始输出逐位一致；目标半径负控通过 | [实现检查](../a_condition_wrapper_gate_20260925_01/SUMMARY.json) |
| 本地原始来源/历史接触核查 | 原train376中按旧规则合格的正是已用16；其他本地包没有新增speaker UID，严格新池0/16 | [来源审计](../a_condition_data_audit_20260925_01/README_zh.md) |

这次新增的几何证据是：对那四个失败对象，仅把现有Fixed优化器延长到64步未能追上原质量目标。不能把它扩大成任何优化器都追不上，也不能把旧16部署一致性测试当成新对象收益。

已固定3k v1 seed10模型（3000updates，tensor SHA `7c02c7926d11c76f22d520a024fa7fac5d4c08c69050cb3745c226a4e6c1b760`）、4秒当前审核达标覆盖率主指标、2/8秒辅助指标、16步、双RMS至少改善2%、512面及原双角度门槛。拟比较Fixed、正确条件3k、错配条件3k，每臂3次，共144任务。对象UID与置换尚未实例化，待数据来源确定后冻结；计划不是已启动的正式RUN_PROTOCOL。

错配实现只改变网络条件，最终半径重查当前目标主场。正确与错误条件均走相同的条件文件读取和proxy构建路径；网络/模型加载、准备、恢复、优化、历次审核等均应计入完整case时间。所有失败留在分母。报告保留四项门槛余量和资源超限，不选择更快的重复。

新的独立验证仍需16个有来源的新同类父对象，或用户明确改变验证类别/划分范围。已有376的几何筛查与16的模型/teacher/开发接触已分层登记；纯几何筛查不等于训练使用。41个官方test曾在跨包身份比较中读取数值数组，但没有接受资格筛选、模型、优化或MC，也未纳入新池。没有自动放宽几何筛选或用旋转/细分伪造新父对象。

本轮没有新增训练，官方MeshFlow仍暂停。官方固定提交和所有已跟踪源码未变。新144任务的端到端调度/汇总仍待数据来源确定后接入和执行，当前交付的是已验证条件封装与待执行计划。

[待执行计划](STUDY_PLAN.json) · [机器状态](STATUS.json) · [产物清单](DELIVERY_MANIFEST.json)
'''
md(PLAN/'PROGRESS_zh.md', report)
portal = Path('/mnt/e/MeshFlow/output/CONDITION_FOLLOWUP_20260925.md')
md(portal, '# A 后续验证进展\n\n已完成旧四对象64步诊断：0/4达标，延长后网格没有变化。条件错配封装通过16项测试和旧16对象真实一致性检查。\n\n**新16对象验证尚未运行：本地原来源没有额外合格父对象，缺16。** 3k权重与4秒主指标已固定，等待新的同类来源或验证范围调整。没有新增训练，官方MeshFlow仍暂停。\n\n[完整进展](../.worktrees/a-layout-value-20260924/output/a_condition16_plan_20260925_01/PROGRESS_zh.md) · [64步诊断](../.worktrees/a-layout-value-20260924/output/a_fixed_extended4_20260925_01/RESULT_zh.md) · [数据审计](../.worktrees/a-layout-value-20260924/output/a_condition_data_audit_20260925_01/README_zh.md)\n')
paths=[]
for directory in (PLAN,EXT,GATE,DATA):
    paths.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
paths.extend((ROOT/'experiments/a_radial/time_quality_followup').glob('*.py'))
write(PLAN/'DELIVERY_MANIFEST.json', {'scope':'This followup artifacts at handoff; excludes this manifest itself and mutable central portal',
 'files':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(paths))}})
print(json.dumps({'status':status['status'],'old4_final_verification':'PASS','audits':audits,'snapshot_slots':snapshots,'array_hashes':array_count,'new_cases':0,'artifacts_bound':len(set(paths))}))