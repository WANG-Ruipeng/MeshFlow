"""Summarize the fixed final fit-capacity diagnosis without selecting weights."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from ..io_utils import file_hash, write_json
from ..layout_endpoint.summarize import paired_values
from .evaluate import MODEL_NAMES, POLICIES, expected_slots, registered_uids
from .diagnose import endpoint_probe


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def med(x):return float(np.median(x)) if len(x) else None
def avg(x):return float(np.mean(x)) if len(x) else None
def fmt(x):return '-' if x is None else f'{x:.3f}'


def summarize(run):
    run=Path(run).resolve();p=read(run/'RUN_PROTOCOL.json');ledger=read(run/'results.json')
    rows=ledger['rows'];slots=expected_slots(p);keys=[(r['uid'],r['model'],r['policy']) for r in rows]
    if len(keys)!=len(set(keys)) or not set(keys)<=slots:raise ValueError('Duplicate or unregistered result')
    fixed={r['uid']:r for r in rows if r['model']=='fixed'}
    enriched=[]
    for r in rows:
        r={**r,**paired_values(r,fixed.get(r['uid']))}
        if r['uid'] not in registered_uids(p,r['model']):raise ValueError('Final model evaluated another parent')
        with np.load(run/'data'/f"{r['uid']}.npz",allow_pickle=False) as z:
            base,teacher,faces=z['base_u'],z['teacher_u'],z['faces']
        with np.load(run/r['mesh_file'],allow_pickle=False) as z:
            r['fit']=endpoint_probe(z['u'],base,teacher,faces,p['endpoint_scale_squared'])
        r['min_angle_margin_deg']=r['quality']['min_triangle_angle_deg']-r['quality_floor']['min_triangle_angle_deg']
        r['p05_angle_margin_deg']=r['quality']['triangle_min_angle_p05_deg']-r['quality_floor']['triangle_min_angle_p05_deg']
        enriched.append(r)
    groups=[]
    for name in MODEL_NAMES:
        for policy in POLICIES[name]:
            for split in ('all','train','internal_check'):
                planned=[u for u in registered_uids(p,name) if split=='all' or u in p['train_uids' if split=='train' else 'internal_check_uids']]
                if not planned:continue
                rr=[r for r in enriched if r['model']==name and r['policy']==policy and r['uid'] in planned]
                paired=[r for r in rr if r['paired_available']]
                endpoint=avg([r['fit']['endpoint_direction_mse_normalized'] for r in rr])
                zero=avg([r['fit']['zero_motion_direction_mse_normalized'] for r in rr])
                groups.append({'model':name,'policy':policy,'split':split,'planned':len(planned),'recorded':len(rr),
                    'missing_or_failed':len(planned)-len(rr),'valid':sum(r['audit']['valid'] for r in rr),
                    'qualified':sum(r['qualified'] for r in rr),'qualified_ge2pct_both':sum(r['qualified_ge2pct_both'] for r in paired),
                    'fallback':sum(r['fallback_to_fixed'] for r in rr),
                    'paired_gain_percent_median_two_measurements':[med([100*(1-r['rms_over_fixed_two_seeds'][i]) for r in paired]) for i in (0,1)],
                    'endpoint_mse_normalized_mean':endpoint,'zero_motion_mse_normalized_mean':zero,
                    'endpoint_mean_over_zero_mean':endpoint/zero if endpoint is not None and zero else None,
                    'min_angle_margin_median_deg':med([r['min_angle_margin_deg'] for r in rr]),
                    'p05_margin_median_deg':med([r['p05_angle_margin_deg'] for r in rr]),
                    'cpu_deployment_hot_seconds_median':med([r['hot_delivery_with_audit_seconds_median_3'] for r in rr if r['hot_delivery_with_audit_seconds_median_3'] is not None]),
                    'single_object_memorization_diagnostic':name.startswith('single_')})
    diag=read(run/'TRAIN_DIAGNOSTICS.json')
    if diag['protocol_sha256']!=file_hash(run/'RUN_PROTOCOL.json'):raise ValueError('Mechanism protocol mismatch')
    training={name:read(run/'training'/name/'TRAIN_REPORT.json') for name in p['jobs']}
    summary={'study':p['experiment'],'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),
        'source_closed_study_not_reinterpreted':True,'heldout_generalization':False,
        'attempts':ledger['attempt_count'],'planned_attempts':98,'rows':len(rows),'planned_rows':164,
        'errors':ledger['errors'],'missing_success_rows':ledger['missing_success_rows'],
        'groups':groups,'training':training,'mechanism_diagnostics':{'expected':156,'recorded':diag['attempt_count'],
            'errors':len(diag['errors']),'missing':len(diag['missing_success_slots']),
            'only_train_objects':diag['internal_check_objects_opened']==0,'groups':diag['groups']},
        'results_sha256':file_hash(run/'results.json'),'mechanism_sha256':file_hash(run/'TRAIN_DIAGNOSTICS.json'),
        'limitations':['All16 are previously used development objects.','Single parents have more exposures; not equal-exposure comparison.',
            'Bridge probes condition on teachers offline; actual CPU rollout never reads teachers.',
            'Only fixed 20000 final is final evaluation; intermediate train diagnosis cannot select weights.',
            'This is A fitting diagnosis, separate from full AE/FM and official MeshFlow comparison.']}
    write_json(run/'SUMMARY.json',summary,replace=(run/'SUMMARY.json').exists())
    write_json(run/'ANALYSIS_ROWS.json',{'rows':enriched},replace=(run/'ANALYSIS_ROWS.json').exists())
    columns=['uid','model','policy','split','qualified','qualified_ge2pct_both','fallback_to_fixed','gain1_pct','gain2_pct','endpoint_over_zero','movement_energy_over_teacher','cosine_to_teacher','min_angle_margin_deg','p05_angle_margin_deg']
    with (run/'PER_SAMPLE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in enriched:
            item={key:r.get(key) for key in columns}
            for i in (0,1):item[f'gain{i+1}_pct']=100*(1-r['rms_over_fixed_two_seeds'][i]) if r['paired_available'] else None
            item.update(endpoint_over_zero=r['fit']['endpoint_mse_over_zero_motion'],movement_energy_over_teacher=r['fit']['movement_energy_over_teacher'],cosine_to_teacher=r['fit']['displacement_cosine_to_teacher'])
            writer.writerow(item)
    lines=['A拟合能力诊断：固定20000更新','',
        f"最终评测 {len(rows)}/164 行、{ledger['attempt_count']}/98 次尝试，错误 {len(ledger['errors'])}；固定训练对象机制诊断 {diag['attempt_count']}/156 次，错误 {len(diag['errors'])}。",'',
        'joint两个模型完整恢复上一轮C0的1000更新状态，各新增19000次；single从对应v1模型重新建立优化器，各20000次。原网络、proxy256、桥速度损失、学习率1e-4及原CPU32部署均保留。所有中间诊断只用训练对象，不按中间结果选权重。','',
        '完整16对象最终原始输出：','',
        '|模型|有效|合格|合格且两次≥2%|配对RMS改善中位1/2 (%)|端点MSE均值/零运动均值|','|---|---:|---:|---:|---:|---:|']
    for g in groups:
        if g['split']=='all' and g['policy'] in ('raw','offline_teacher') and not g['single_object_memorization_diagnostic']:
            a,b=g['paired_gain_percent_median_two_measurements'];n=g['planned']
            lines.append(f"|{g['model']}|{g['valid']}/{n}|{g['qualified']}/{n}|{g['qualified_ge2pct_both']}/{n}|{fmt(a)} / {fmt(b)}|{fmt(g['endpoint_mean_over_zero_mean'])}|")
    lines += ['', '训练12与内部4分别报告：','', '|模型/划分|合格|合格且两次≥2%|配对RMS改善中位1/2 (%)|','|---|---:|---:|---:|']
    for g in groups:
        if g['model'].startswith(('parent_c0_','joint_')) and g['policy']=='raw' and g['split']!='all':
            a,b=g['paired_gain_percent_median_two_measurements'];n=g['planned']
            lines.append(f"|{g['model']} / {g['split']}|{g['qualified']}/{n}|{g['qualified_ge2pct_both']}/{n}|{fmt(a)} / {fmt(b)}|")
    lines += ['', '单对象记忆能力诊断（每项只评自身训练对象，不能作为类别泛化）：','', '|模型|合格|两次RMS改善 (%)|端点MSE/零运动|','|---|---:|---:|---:|']
    for g in groups:
        if g['split']=='all' and g['policy']=='raw' and g['single_object_memorization_diagnostic']:
            a,b=g['paired_gain_percent_median_two_measurements'];lines.append(f"|{g['model']}|{g['qualified']}/1|{fmt(a)} / {fmt(b)}|{fmt(g['endpoint_mean_over_zero_mean'])}|")
    lines += ['', '训练对象拟合轨迹（均值之比；用于机制，不替代逐对象统计）：','', '|任务/检查点|桥速度MSE/零速度MSE|自由端点MSE/零运动MSE|合格|','|---|---:|---:|---:|']
    for g in diag['groups']:
        v=g['velocity'];e=g['endpoint'];vm=v['mse']['mean'];vz=v['zero_velocity_mse']['mean'];em=e['endpoint_direction_mse_normalized']['mean'];ez=e['zero_motion_direction_mse_normalized']['mean']
        lines.append(f"|{g['job']} / {g['checkpoint']}|{fmt(vm/vz if vm is not None and vz else None)}|{fmt(em/ez if em is not None and ez else None)}|{g['qualified_count']}/{g['expected_parents']}|")
    lines += ['', '单对象每父对象新增曝光80000次；共同训练总计曝光80000次，分摊到12对象约6666/6667次，且继承不同训练历史。不能把差异直接当作相同数据量下的共享干扰证据。','',
        '每种方法保存raw与质量失败退回Fixed两行。回退不使用RMS，回退后合法不能计作学习收益；完整逐对象结果及失败见CSV/JSON。两次RMS使用新的592501/692501加UID索引，每方向8192面积采样。','',
        '该研究是在首轮未达到门槛后获用户授权进行的独立开发诊断，首轮结论未改写。未扩大模型、未改变教师或加入推理额外输入，也不是完整MeshFlow生成对比。','',
        '文件：[完整统计](SUMMARY.json)、[逐对象结果](PER_SAMPLE.csv)、[训练诊断](TRAIN_DIAGNOSTICS.json)、[协议](RUN_PROTOCOL.json)。']
    (run/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(rows),'errors':len(ledger['errors']),'mechanism_rows':len(diag['rows'])}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();summarize(args.run)