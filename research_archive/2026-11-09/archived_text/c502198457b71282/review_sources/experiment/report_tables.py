"""Presentation tables from completed registered CSVs; no scoring or model calls."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'native_t1/runs/chair_capped500_gradbudget_v1'
LABELS = ('START','HIST_TEACHER500','HIST_CAPPED500','T500','L500','G500','T1000','L1000','G1000')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def number(value):
    return None if value in (None, '', 'NA') else float(value)


def fmt(value, scale=1., digits=3, signed=False):
    x = number(value)
    if x is None:
        return 'NA'
    return format(x*scale, ('+' if signed else '')+'.'+str(digits)+'f')


def table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |', '| '+' | '.join(['---']*len(header))+' |'] +
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def build(out=OUT):
    out=Path(out); folder=out/'analysis'
    result=json.loads((folder/'results.json').read_text(encoding='utf-8'))
    if result['status']!='COMPLETE':
        return dict(status='NOT_RUN_ANALYSIS_INCOMPLETE',model_calls=0,scoring_calls=0)
    names=('checkpoint_profiles','summary','per_parent','per_noise','leave_one_parent_out',
           'drift_difference_in_differences_summary','original_joint_gates')
    data={name:read_csv(folder/(name+'.csv')) for name in names}
    hashes={name:hashlib.sha256((folder/(name+'.csv')).read_bytes()).hexdigest() for name in names}
    for name in ('summary','per_parent','per_noise','leave_one_parent_out'):
        assert hashes[name]==result['tables'][name]['sha256'], name
    assert hashes['drift_difference_in_differences_summary']==result['tables']['DID_summary']['sha256']
    profile={(r['panel'],r['label']):r for r in data['checkpoint_profiles']}
    summary={(r['panel'],r['candidate_label'],r['reference_label'],r['metric']):r for r in data['summary']}
    parent={(r['panel'],r['candidate_label'],r['reference_label'],r['metric'],r['uid']):r for r in data['per_parent']}
    did={(r['panel'],r['metric']):r for r in data['drift_difference_in_differences_summary']}
    def comp(panel, candidate, reference, metric):
        return summary[(panel,candidate,reference,metric)]
    text=[]
    def add(title, explanation, header, rows):
        text.extend([title,'',explanation,'',table(header,rows),''])
    for panel in ('OLD','FRESH'):
        rows=[]
        for label in LABELS:
            r=profile[(panel,label)]
            assert r['status']=='COMPLETE' and int(r['completed'])==16 and int(r['expected_outputs'])==16
            rows.append([label]+[fmt(r[k],100) for k in ('SeamCov','ConflictCov_C','ExposedRetain')]+
                        [fmt(r[k],1000) for k in ('fixed_query_MSE','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','D_sym2048')])
        add(panel+' 全标签绝对值','S/I/E 为 %；MSE 与 Dsym 均除以原 L0²，表内乘 1000。8 父等权，每父固定两噪声。',
            ['标签','S↑','I↓','E↑','GT→F512↓','GT→F2048↓','F→GT2048↓','Dsym↓'],rows)
    contrasts=(('HIST_CAPPED500','HIST_TEACHER500'),('HIST_CAPPED500','START'),
               ('L500','T500'),('L500','START'),('G500','L500'),('G500','T500'),('G500','START'))
    rows=[]
    for panel in ('OLD','FRESH'):
        for candidate,reference in contrasts:
            r=comp(panel,candidate,reference,'SeamCov')
            noises=sorted([q for q in data['per_noise'] if (q['panel'],q['candidate_label'],q['reference_label'],q['metric'])==
                           (panel,candidate,reference,'SeamCov')],key=lambda q:int(q['candidate']))
            lo=[number(q['delta']) for q in data['leave_one_parent_out'] if
                (q['panel'],q['candidate_label'],q['reference_label'],q['metric'])==(panel,candidate,reference,'SeamCov')]
            rows.append([panel,candidate+'−'+reference,fmt(r['delta'],100,signed=True),fmt(r['median_delta'],100,signed=True),
                r['improved_parents']+'/8',' / '.join(fmt(q['delta'],100,signed=True) for q in noises),
                r['all_noises_improve_parents']+'/8',r['opposite_noise_parents']+'/8',
                fmt(min(lo),100,signed=True)+' ~ '+fmt(max(lo),100,signed=True)])
    add('接缝配对分布','差值单位为百分点；两噪声按 OLD 0/1、FRESH 2/3 排列。留一父只是描述性敏感性，原门槛仍用全部8父。',
        ['面板','比较','均差','父中位差','改善父','两噪声均差','两噪声均改善父','噪声反向父','留一父均差范围'],rows)
    rows=[]
    for panel in ('OLD','FRESH'):
        for candidate,reference in contrasts:
            r=comp(panel,candidate,reference,'GT_to_F_MSE2048_L0')
            rows.append([panel,candidate+'−'+reference]+[fmt(comp(panel,candidate,reference,key)['percent_change'],signed=True) for key in
                ('fixed_query_MSE','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','D_sym2048','fixed_query_p95_L')]+
                [r['improved_parents']+'/8',fmt(comp(panel,candidate,reference,'ConflictCov_C')['delta'],100,signed=True),
                 fmt(comp(panel,candidate,reference,'ExposedRetain')['delta'],100,signed=True)])
    add('500步相对质量与控制代价','距离列为候选均值/参考均值−1的百分比，负值较好；I/E 为百分点。父改善数专指 GT→F2048。',
        ['面板','比较','GT512 %','GT2048 %','F→GT %','Dsym %','p95 %','GT2048改善父','I pp','E pp'],rows)
    rows=[]
    for panel in ('OLD','FRESH'):
        for arm in ('T','L','G'):
            rows.append([panel,arm+'1000−'+arm+'500']+[fmt(comp(panel,arm+'1000',arm+'500',key)['delta'],scale,signed=True) for key,scale in
                (('SeamCov',100),('ConflictCov_C',100),('ExposedRetain',100),('fixed_query_MSE',1000),('GT_to_F_MSE2048_L0',1000),('D_sym2048',1000))])
        rows.append([panel,'G变化−L变化']+[fmt(did[(panel,key)]['DID'],scale,signed=True) for key,scale in
            (('SeamCov',100),('ConflictCov_C',100),('ExposedRetain',100),('fixed_query_MSE',1000),('GT_to_F_MSE2048_L0',1000),('D_sym2048',1000))])
    add('持续训练变化及差分变化','前三列为百分点，后三列为归一化 MSE 差×1000。S/E 正值较好；I/MSE/Dsym 负值较好。DID不能代替先判断L是否实际回退。',
        ['面板','比较','S pp','I pp','E pp','GT512×1000','GT2048×1000','Dsym×1000'],rows)
    rows=[]
    for panel in ('OLD','FRESH'):
        for label in LABELS:
            r=profile[(panel,label)]
            rows.append([panel,label,fmt(r['fixed_query_p95_L'])]+[fmt(r[k],100) for k in ('coverage_fraction','CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction',
                'free_near_degenerate_fraction','free_exact_zero_area_fraction','free_minimum_angle_lt5_fraction')]+
                [fmt(r['free_total_surface_area']),fmt(r['free_area_over_GT_free_area']),fmt(r['mean_registered_free_faces'])])
    add('交叉、退化、面积与面数','p95为距离/L0；各比例为%；面积为原模型坐标平方的父均值，面积比是逐例比值的等父均值。CF/FF沿用原检测定义。平均free面数不代表每例相同，逐例N/K见原表。',
        ['面板','标签','p95/L0','coverage','CF受影响','FF受影响','近退化','零面积','角<5°','free面积','free/GT面积','free面数'],rows)
    rows=[]
    uids=sorted({r['uid'] for r in data['per_parent']})
    for uid in uids:
        values=[uid]
        for panel in ('OLD','FRESH'):
            for candidate,reference in (('L500','T500'),('G500','L500')):
                r=parent[(panel,candidate,reference,'SeamCov',uid)]
                noise=json.loads(r['noise_deltas'])
                values.append(fmt(r['delta'],100,signed=True)+' ['+'/'.join(fmt(noise[k],100,signed=True) for k in sorted(noise,key=int))+']')
                q=parent[(panel,candidate,reference,'GT_to_F_MSE2048_L0',uid)]
                cv,rv=number(q['candidate_value']),number(q['reference_value'])
                values.append(fmt(100*(cv/rv-1),signed=True) if cv is not None and rv not in (None,0) else 'NA')
        rows.append(values)
    add('完整8父主终点配对','S写为父均差 [两个噪声各自差]，单位百分点。GT写为该父两噪声均值的相对变化%。不挑最佳噪声。',
        ['父UID','OLD L−T S','OLD L/T GT%','OLD G−L S','OLD G/L GT%','FRESH L−T S','FRESH L/T GT%','FRESH G−L S','FRESH G/L GT%'],rows)
    rows=[]
    for r in data['original_joint_gates']:
        rows.append([r[k] for k in ('panel','candidate_label','matched_label','numeric_status','shape_quality_status','control_A_status','control_B_status','severe_cost_caveat','status')])
    add('原联合门槛','匹配对照与START双参考均保留；严重父代价按原实现单独flag，必须阅读逐父表和实际视觉审阅。数值条件、视觉判断与晋级建议分别解释。',
        ['面板','候选','匹配对照','数值','形状','控制A','控制B','严重父标记','含视觉状态'],rows)
    path=out/'report_tables.md';path.write_text('\n'.join(text),encoding='utf-8')
    receipt=dict(status='PASS',path=str(path),source_hashes=hashes,model_calls=0,scoring_calls=0,
                 report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'report_tables_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


if __name__=='__main__':
    print(json.dumps(build()))
