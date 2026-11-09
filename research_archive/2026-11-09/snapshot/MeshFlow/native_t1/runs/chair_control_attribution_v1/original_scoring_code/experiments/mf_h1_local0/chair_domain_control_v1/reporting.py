"""Deterministic report over real ledgers; CPU references are separately resumable."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import statistics
import numpy as np
from .common import atomic_json, read_json, digest
from .evaluation import ARMS, PHASES, DISPLAY, records_for_phase, arrays, resolve, scoring_bank, _bank_reference
from .width_analytics import DEFINITION

PRIMARY = [
    'remainder_symmetric_RMS_bbox_pct', 'fixed_R12_coverage_within_1pct_fraction',
    'Gamma_to_free_edges_RMS_bbox_pct', 'Gamma_to_free_edges_coverage_0.001_fraction',
    'CF_proper_affected_free_fraction', 'FF_illegal_affected_free_fraction',
    'free_minimum_angle_lt5_fraction', 'free_angle_p05_degrees',
]
WIDTH_METRICS = [
    'X_span_abs_error_L0', 'Y_span_abs_error_L0', 'Z_span_abs_error_L0',
    'X_median_abs_error_L0', 'Y_median_abs_error_L0', 'Z_median_abs_error_L0',
    'response_residual', 'response_abs_residual', 'slope', 'slope_abs_error',
    'monotonic', 'strict_monotonic', 'wrong_mean_minus_diagonal_RMS_L0',
    'diagonal_corresponding_RMS_L0', 'diagonal_no_worse_than_both_wrong',
    'diagonal_strictly_better_than_both_wrong',
]


def finite(value):
    return isinstance(value, (float, int, bool, np.number)) and math.isfinite(float(value))


def mean(values):
    values = [float(v) for v in values if finite(v)]
    return statistics.fmean(values) if values else None


def median(values):
    values = [float(v) for v in values if finite(v)]
    return statistics.median(values) if values else None


def fmt(value):
    return f'{float(value):.6g}' if finite(value) else 'NA'


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader()
        for row in rows:
            writer.writerow({k:('NA' if v is None else v) for k,v in row.items()})
    temp.replace(path)


def metric_specs():
    from experiments.mf_h1_local0.control_mvp48.eval_scoring import METRIC_SPECS
    specs = dict(METRIC_SPECS)
    for name, definition in list(specs.items()):
        if name.endswith('_bbox_pct'):
            specs[name.replace('_bbox_pct', '_L0_pct')] = dict(direction='lower', unit='% L0')
    for name in WIDTH_METRICS:
        direction = 'descriptive' if name in ('response_residual','slope') else (
            'higher' if name in ('monotonic','strict_monotonic','wrong_mean_minus_diagonal_RMS_L0',
                                'diagonal_no_worse_than_both_wrong','diagonal_strictly_better_than_both_wrong') else 'lower')
        specs[name] = dict(direction=direction, unit='ratio_or_fraction')
    specs['corresponding_GT_to_free_RMS_L0'] = dict(direction='lower', unit='fraction L0')
    specs['corresponding_GT_coverage_1pct_L0_fraction'] = dict(direction='higher', unit='fraction')
    return specs


def collect(out, panels):
    out = Path(out); rows=[]; teachers=[]; ledgers=[]
    specs = metric_specs()
    for phase in PHASES:
        for arm in ARMS:
            path = out/'evaluation'/f'phase{phase}'/arm/'ledger.json'
            ledger = read_json(path) if path.exists() else {}
            if ledger:
                ledgers.append(ledger)
            index = {r['case_id']:r for r in ledger.get('rows', [])}
            for record in records_for_phase(panels, phase):
                found = index.get(record['case_id'], {})
                row = dict(arm=arm, display_model=DISPLAY[arm], phase_step=phase,
                           conditional_total_step=1000+phase, table=record['table'],
                           case_id=record['case_id'], uid=record['uid'], split=record['split'],
                           task_id=record['task_id'], ratio=record['ratio'], alpha=record['alpha'],
                           noise_index=record['noise_index'], N=record['N'], K=record['K'],
                           C_area_fraction=record.get('C_area_fraction'), input_sha256=record.get('input_sha256'),
                           geometry_status=record['status'], reason=record.get('reason'),
                           max_old_training_jaccard=record.get('overlap',{}).get('max_old_training_jaccard'),
                           max_new_training_jaccard=record.get('overlap',{}).get('max_new_training_jaccard'),
                           status=found.get('status', 'NOT_APPLICABLE' if record['status']=='NOT_APPLICABLE' else 'NOT_RUN'),
                           score_status=found.get('score_status', 'NOT_RUN'),
                           raw_path=found.get('raw_path'), raw_sha256=found.get('raw_sha256'),
                           score_path=found.get('score_path'), score_sha256=found.get('score_sha256'))
                row.update({m:None for m in specs})
                if row['status']=='PASS':
                    if digest(resolve(out,row['raw_path'])) != row['raw_sha256']:
                        raise ValueError('Raw changed while reporting')
                    if not row['score_path'] or digest(resolve(out,row['score_path'])) != row['score_sha256']:
                        raise ValueError('Missing or changed score for completed output')
                    score = read_json(resolve(out,row['score_path']))
                    if row['score_status'] != score['status']:
                        raise ValueError('Score status mismatch')
                    row.update(score['metrics'])
                rows.append(row)
            tindex = {r['case_id']:r for r in ledger.get('teacher_rows', [])}
            for record in panels['teacher']:
                found = tindex.get(record['case_id'], {})
                row = dict(arm=arm, display_model=DISPLAY[arm], phase_step=phase,
                           conditional_total_step=1000+phase, split=record['split'], uid=record['uid'],
                           case_id=record['case_id'], t=record['t'], input_sha256=record['input_sha256'],
                           status=found.get('status','NOT_RUN'), FM_error=found.get('FM_error'),
                           D_edge=found.get('D_edge'), L5=found.get('L5'),
                           Yhat_free_area=found.get('geometry',{}).get('Yhat_free',{}).get('total_area'))
                teachers.append(row)
    return rows, teachers, ledgers


def aggregate(rows, group_keys, metrics, *, valid_key='score_status'):
    grouped=defaultdict(list)
    for r in rows:
        grouped[tuple(r.get(k) for k in group_keys)].append(r)
    parents=[]
    for group, members in sorted(grouped.items(),key=lambda p:str(p[0])):
        row=dict(zip(group_keys,group))
        row.update(requested_cases=len(members),
                   completed_cases=sum(r.get('status')=='PASS' or r.get('width_status')=='PASS' for r in members),
                   completed_candidates=sum(r.get('candidate') is not None and r.get('status')=='PASS' for r in members),
                   completed_generations=sum(r.get('status')=='PASS' and bool(r.get('raw_path')) and r.get('candidate') is None for r in members),
                   geometry_not_applicable_cases=sum(r.get('geometry_status')=='NOT_APPLICABLE' for r in members),
                   metric_valid_cases=sum(r.get(valid_key)=='PASS' for r in members))
        for metric in metrics:
            values=[r.get(metric) for r in members if r.get(valid_key)=='PASS']
            row[metric]=mean(values)
            row[metric+'_valid_cases']=sum(finite(v) for v in values)
        parents.append(row)
    return parents


def parent_summary(parents, group_keys, metrics):
    groups=defaultdict(list)
    for row in parents:
        groups[tuple(row.get(k) for k in group_keys)].append(row)
    result=[]
    for group, members in sorted(groups.items(),key=lambda p:str(p[0])):
        row=dict(zip(group_keys,group))
        row.update(requested_parents=len(members), valid_parents=sum(r['metric_valid_cases']>0 for r in members),
                   requested_cases=sum(r['requested_cases'] for r in members),
                   completed_generations=sum(r['completed_generations'] for r in members),
                   completed_cases=sum(r['completed_cases'] for r in members),
                   completed_candidates=sum(r['completed_candidates'] for r in members),
                   metric_valid_cases=sum(r['metric_valid_cases'] for r in members),
                   geometry_not_applicable_cases=sum(r['geometry_not_applicable_cases'] for r in members))
        for metric in metrics:
            vals=[r.get(metric) for r in members]
            row[metric]=mean(vals);row[metric+'_valid_parents']=sum(finite(v) for v in vals)
            row[metric+'_valid_cases']=sum(r[metric+'_valid_cases'] for r in members)
        result.append(row)
    return result


def paired_case_rows(before, after, comparison, metrics, *, metadata=None):
    index={r['case_id']:r for r in before}
    result=[]
    for a in after:
        b=index.get(a['case_id'])
        if b is None:
            continue
        same_input=a.get('input_sha256')==b.get('input_sha256') and a.get('input_sha256') is not None
        valid=same_input and a.get('score_status')=='PASS' and b.get('score_status')=='PASS'
        row=dict(comparison=comparison, table=a['table'], split=a['split'], uid=a['uid'],
                 ratio=a.get('ratio'), alpha=a.get('alpha'), case_id=a['case_id'], noise_index=a['noise_index'],
                 before_phase_step=b['phase_step'], after_phase_step=a['phase_step'],
                 same_input=same_input, status='PASS' if valid else 'NOT_COMPARABLE',
                 before_raw_sha256=b.get('raw_sha256'), after_raw_sha256=a.get('raw_sha256'))
        row.update(metadata or {})
        for m in metrics:
            x,y=a.get(m),b.get(m)
            row[m+'_before']=y;row[m+'_after']=x
            row[m+'_delta']=float(x)-float(y) if valid and finite(x) and finite(y) else None
        result.append(row)
    return result


def summarize_pairs(cases, metrics, group_keys):
    parent_groups=defaultdict(list)
    for row in cases:
        parent_groups[tuple(row.get(k) for k in group_keys)+ (row['uid'],)].append(row)
    parent_rows=[]
    for key, members in sorted(parent_groups.items(),key=lambda p:str(p[0])):
        row=dict(zip(group_keys+['uid'],key))
        row.update(requested_case_pairs=len(members), valid_case_pairs=sum(r['status']=='PASS' for r in members))
        for metric in metrics:
            row[metric+'_delta']=mean(r[metric+'_delta'] for r in members)
            row[metric+'_paired_valid_cases']=sum(finite(r[metric+'_delta']) for r in members)
        parent_rows.append(row)
    summaries=[];groups=defaultdict(list);spec=metric_specs()
    for row in parent_rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)
    for key, members in sorted(groups.items(),key=lambda p:str(p[0])):
        for metric in metrics:
            pairs=[r for r in members if finite(r[metric+'_delta'])]
            ds=[r[metric+'_delta'] for r in pairs]
            direction=spec[metric]['direction']
            worst=(max(pairs,key=lambda r:r[metric+'_delta']) if direction=='lower' else
                   min(pairs,key=lambda r:r[metric+'_delta'])) if pairs and direction!='descriptive' else None
            row=dict(zip(group_keys,key))
            row.update(metric=metric, direction=direction, unit=spec[metric]['unit'],
                       requested_parents=len(members), paired_valid_parents=len(pairs),
                       requested_case_pairs=sum(r['requested_case_pairs'] for r in members),
                       paired_valid_cases=sum(r[metric+'_paired_valid_cases'] for r in members),
                       mean_paired_parent_delta=mean(ds), median_paired_parent_delta=median(ds),
                       improved_parents=sum((v<0 if direction=='lower' else v>0) for v in ds) if direction!='descriptive' else None,
                       worst_parent=worst['uid'] if worst else None, worst_parent_delta=worst[metric+'_delta'] if worst else None)
            summaries.append(row)
    return parent_rows,summaries


def historical_holdout(source, history, out, panels):
    """Read scores only. Missing historical assets remain NA; never rescore."""
    source,history,out=Path(source),Path(history),Path(out)
    from experiments.mf_h1_local0.control_mvp48.eval_scoring import source_identity
    scorer_match=read_json(source/'registration/eval_protocol.json')['scorer_identity']==source_identity()
    old=read_json(source/'evaluation/gpu_run.json')
    oldindex={(r['case_id'],r['model']):r for r in old['rows']}
    result=[];checks=[]
    for arm in ARMS:
        for total in (1000,5000):
            newledgerpath=history/'evaluation'/'holdout'/f'{arm}_5000'/'ledger.json'
            prior=read_json(newledgerpath) if total==5000 and newledgerpath.exists() else {}
            index={r['case_id']:r for r in prior.get('rows',[])}
            checkpoint=(source/'checkpoints'/f'{arm}_step1000.pt' if total==1000 else
                        history/'checkpoints'/f'{arm}_step5000.pt')
            expected=(old['identities']['endpoints'][arm]['hashes']['checkpoint_sha256'] if total==1000 else
                      prior.get('identity',{}).get('checkpoint_sha256'))
            checkpoint_ok=checkpoint.exists() and expected is not None and digest(checkpoint)==expected
            checks.append(dict(arm=arm,conditional_total_step=total,checkpoint=str(checkpoint),
                               expected_sha256=expected,checkpoint_match=checkpoint_ok,scorer_match=scorer_match))
            for record in panels['C']:
                previous=oldindex.get((record['source_case_id'],arm),{}) if total==1000 else index.get(record['case_id'],{})
                base=source if total==1000 else history
                raw=resolve(base,previous.get('raw_path','MISSING'))
                scorepath=(source/'evaluation'/'scores'/(previous.get('request_id','MISSING')+'.json') if total==1000 else
                           resolve(history,previous.get('score_path','MISSING')))
                row=dict(arm=arm,display_model=DISPLAY[arm],conditional_total_step=total,case_id=record['case_id'],
                         uid=record['uid'],split='holdout',noise_index=record['noise_index'],ratio=.2,
                         table='C',phase_step=None,input_sha256=record['input_sha256'],
                         status='NOT_COMPARABLE',score_status='NOT_COMPARABLE',
                         reason='Missing or mismatched historical input/raw/checkpoint/scorer; no replay',
                         generation_replayed=False)
                oldinput=previous.get('input_file_sha256',previous.get('input_sha256'))
                ready=(scorer_match and checkpoint_ok and previous.get('status')=='PASS' and
                       raw.is_file() and scorepath.is_file() and oldinput==record['input_sha256'] and
                       digest(resolve(out,record['input_path']))==record['input_sha256'] and
                       digest(raw)==previous.get('raw_sha256'))
                if ready:
                    score=read_json(scorepath)
                    scorevalid=score.get('scoring_status',score.get('status'))=='PASS'
                    if total==1000:
                        scorevalid=scorevalid and score.get('raw_sha256')==previous.get('raw_sha256')
                    else:
                        scorevalid=scorevalid and digest(scorepath)==previous.get('score_sha256')
                    if scorevalid:
                        row.update(score['metrics'])
                        row.update(status='REUSED',score_status='PASS',reason=None,raw_path=str(raw),
                                   raw_sha256=previous['raw_sha256'],score_path=str(scorepath),score_sha256=digest(scorepath))
                result.append(row)
    atomic_json(out/'history_reuse.json',dict(status='PASS',checks=checks,requested_rows=192,
                strictly_reused_rows=sum(r['status']=='REUSED' for r in result),
                historical_missing_rows=sum(r['status']!='REUSED' for r in result),
                model_forwards=0,OT_calls=0,CPU_score_calls=0))
    return result


def figures(out, panels, rows, shared_curves, teacher_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);folder=out/'figures';folder.mkdir(exist_ok=True)
    records={r['case_id']:r for table in ('A','B','C') for r in panels[table]}
    index={(r['arm'],r['case_id']):r for r in rows if r['phase_step']==3000}
    files=[]
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,m in zip(axes.flat,PRIMARY[:6]):
        for arm,color in zip(ARMS,('#3974bc','#dc8a2c','#469b62')):
            for split,line in [('train','-'),('validation','--')]:
                data=sorted([r for r in shared_curves if r['arm']==arm and r['split']==split and finite(r.get(m))],key=lambda r:r['phase_step'])
                if data:ax.plot([r['phase_step'] for r in data],[r[m] for r in data],marker='o',linestyle=line,color=color,label=DISPLAY[arm]+' '+split)
        ax.set_title(m,fontsize=8);ax.set_xlabel('New phase updates');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6);fig.suptitle('Shared A20 first-noise requests only')
    fig.tight_layout();path=folder/'shared_input_generation_curves.png';fig.savefig(path,dpi=140);plt.close(fig);files.append(path)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,(split,t) in zip(axes.flat,[(s,t) for s in ('train','validation') for t in (.2,.5,.8)]):
        for arm,color in zip(ARMS,('#3974bc','#dc8a2c','#469b62')):
            data=sorted([r for r in teacher_summary if r['arm']==arm and r['split']==split and r['t']==t],key=lambda r:r['phase_step'])
            if data:ax.plot([r['phase_step'] for r in data],[r['FM_error'] for r in data],marker='o',color=color,label=DISPLAY[arm])
        ax.set_title(f'Fixed teacher {split} t={t}');ax.set_xlabel('New phase updates');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6);fig.tight_layout();path=folder/'fixed_teacher_curves.png';fig.savefig(path,dpi=140);plt.close(fig);files.append(path)
    from .fixed_figures import generate_fixed_figures
    fixed=generate_fixed_figures(out,panels)
    manifest=dict(status='PASS', files=[dict(path=str(p.relative_to(out)),sha256=digest(p)) for p in files]+fixed['files'],
                  fixed_view_status=fixed['status'])
    atomic_json(folder/'manifest.json',manifest)
    return manifest


def run(source,out,history):
    source,out,history=Path(source),Path(out),Path(history)
    panels=read_json(out/'panels.json')
    rows,teachers,ledgers=collect(out,panels)
    spec=metric_specs();geometry_metrics=[m for m in spec if m not in WIDTH_METRICS and not m.startswith('corresponding_')]
    if len(ledgers)!=12 or any(l.get('status')!='PASS' for l in ledgers):
        status=dict(status='NOT_RUN',reason='All 12 registered model-phase evaluations must finish before final analysis',
                    evaluation_workers_present=len(ledgers))
        atomic_json(out/'reporting_status.json',status)
        raise RuntimeError(status['reason'])
    from .width_analytics import run as analyze_width
    widths,cross,candidates,analysis=analyze_width(out,panels,rows)
    artifact_names=[]
    def save(name,data):
        write_csv(out/name,data);artifact_names.append(name)
    save('all_generation_per_case.csv',rows)
    save('A_context_per_case.csv',[r for r in rows if r['table']=='A' and r['phase_step']==3000])
    allparents=aggregate(rows,['table','arm','phase_step','split','ratio','uid'],geometry_metrics)
    summaries=parent_summary(allparents,['table','arm','phase_step','split','ratio'],geometry_metrics)
    save('generation_per_parent.csv',allparents);save('generation_summary.csv',summaries)
    save('A_context_per_parent.csv',[r for r in allparents if r['table']=='A' and r['phase_step']==3000])
    save('A_context_summary.csv',[r for r in summaries if r['table']=='A' and r['phase_step']==3000])
    # Preserve actual C information without selecting strata after model results.
    area_rows=[]
    for r in rows:
        if r['table']=='A' and r['phase_step']==3000:
            area=r.get('C_area_fraction')
            area_rows.append(dict(r,area_bin='NA' if not finite(area) else '<0.2' if area<.2 else '0.2-0.4' if area<.4 else '>=0.4'))
    area_parents=aggregate(area_rows,['arm','split','ratio','area_bin','uid'],geometry_metrics)
    save('A_context_area_strata.csv',parent_summary(area_parents,['arm','split','ratio','area_bin'],geometry_metrics))
    save('B_width_response.csv',widths);save('B_cross_condition_scores.csv',cross);save('B_analytic_baselines.csv',candidates)
    save('B_width_geometry_per_case.csv',[r for r in rows if r['table']=='B'])
    width_parents=aggregate(widths,['arm','phase_step','split','alpha','uid'],WIDTH_METRICS,valid_key='width_status')
    save('B_width_per_parent.csv',width_parents)
    width_summary=parent_summary(width_parents,['arm','phase_step','split','alpha'],WIDTH_METRICS)
    save('B_width_summary.csv',width_summary)
    candidate_metrics=geometry_metrics+['X_span_abs_error_L0','Y_span_abs_error_L0','Z_span_abs_error_L0',
                       'X_median_abs_error_L0','Y_median_abs_error_L0','Z_median_abs_error_L0',
                       'corresponding_GT_to_free_RMS_L0','corresponding_GT_coverage_1pct_L0_fraction',
                       'response_abs_residual','slope_abs_error']
    candidate_parents=aggregate(candidates,['candidate','arm','phase_step','split','alpha','uid'],candidate_metrics)
    save('B_analytic_per_parent.csv',candidate_parents)
    save('B_analytic_summary.csv',parent_summary(candidate_parents,['candidate','arm','phase_step','split','alpha'],candidate_metrics))
    tp=aggregate(teachers,['arm','phase_step','split','t','uid'],['FM_error','D_edge','L5','Yhat_free_area'],valid_key='status')
    ts=parent_summary(tp,['arm','phase_step','split','t'],['FM_error','D_edge','L5','Yhat_free_area'])
    save('teacher_per_case.csv',teachers);save('teacher_summary.csv',ts)
    # Shared A20 first-noise inputs only: no endpoint extra seed/C contaminates the curve.
    initial_cases={r['case_id'] for r in panels['A'] if 0 in r['phase_steps']}
    shared=[r for r in rows if r['table']=='A' and r['case_id'] in initial_cases]
    cp=aggregate(shared,['arm','phase_step','split','uid'],geometry_metrics)
    curves=parent_summary(cp,['arm','phase_step','split'],geometry_metrics)
    save('shared_input_generation_curves.csv',curves)
    comparisons=[]
    for arm in ARMS:
        before=[r for r in rows if r['arm']==arm and r['phase_step']==0]
        after=[r for r in rows if r['arm']==arm and r['phase_step']==3000]
        comparisons+=paired_case_rows(before,after,DISPLAY[arm]+'_endpoint_minus_start',geometry_metrics,metadata=dict(arm=arm,comparison_kind='same_input_start_endpoint'))
    save('shared_start_endpoint_per_case.csv',comparisons)
    pp,ps=summarize_pairs(comparisons,geometry_metrics,['comparison','table','split','ratio'])
    save('shared_start_endpoint_per_parent.csv',pp);save('shared_start_endpoint_summary.csv',ps)
    incremental=[]
    for phase in PHASES:
        for reference,arm in [('T1','Geo'),('Geo','Full'),('T1','Full')]:
            incremental+=paired_case_rows([r for r in rows if r['arm']==reference and r['phase_step']==phase],
                [r for r in rows if r['arm']==arm and r['phase_step']==phase],DISPLAY[arm]+'_minus_'+DISPLAY[reference],
                geometry_metrics,metadata=dict(phase_step=phase,comparison_kind='matched_budget_arm_increment'))
    ip,ins=summarize_pairs(incremental,geometry_metrics,['comparison','phase_step','table','split','ratio'])
    save('matched_budget_per_case.csv',incremental);save('matched_budget_per_parent.csv',ip);save('matched_budget_summary.csv',ins)
    # Width start/end pairs use only shared first-noise requests.
    widthjoined=[]
    rawidx={(r['arm'],r['phase_step'],r['case_id']):r for r in rows}
    for w in widths:
        widthjoined.append(dict(rawidx[(w['arm'],w['phase_step'],w['case_id'])],**w))
    wpairs=[]
    for arm in ARMS:
        wpairs+=paired_case_rows([r for r in widthjoined if r['arm']==arm and r['phase_step']==0],
            [r for r in widthjoined if r['arm']==arm and r['phase_step']==3000],DISPLAY[arm]+'_width_endpoint_minus_start',WIDTH_METRICS)
    wpp,wps=summarize_pairs(wpairs,WIDTH_METRICS,['comparison','table','split','ratio'])
    save('B_width_start_endpoint_pairs.csv',wpairs);save('B_width_start_endpoint_summary.csv',wps)
    # Same-current-condition comparisons against zero-forward edit references.
    analytic_pairs=[]
    model_analysis_metrics=list(dict.fromkeys(candidate_metrics))
    for phase in (0,3000):
        for arm in ARMS:
            after=[r for r in widthjoined if r['phase_step']==phase and r['arm']==arm]
            for candidate in ('No-response','Post-scale'):
                before=[]
                for c in candidates:
                    if c['phase_step']!=phase or c['arm']!=arm or c['candidate']!=candidate:continue
                    original=rawidx[(arm,phase,c['case_id'])]
                    before.append(dict(c,table='B',ratio=.2,input_sha256=original['input_sha256'],
                                       raw_sha256=c.get('candidate_raw_sha256')))
                analytic_pairs+=paired_case_rows(before,after,DISPLAY[arm]+'_minus_'+candidate,model_analysis_metrics,
                    metadata=dict(phase_step=phase,reference_candidate=candidate,
                                  comparison_scope='same_current_condition; analytic reference also knows previous C and output'))
    ap,aps=summarize_pairs(analytic_pairs,model_analysis_metrics,['comparison','phase_step','table','split','ratio'])
    save('B_model_minus_analytic_per_case.csv',analytic_pairs)
    save('B_model_minus_analytic_per_parent.csv',ap);save('B_model_minus_analytic_summary.csv',aps)
    width_increments=[]
    for phase in (0,3000):
        for reference,arm in [('T1','Geo'),('Geo','Full'),('T1','Full')]:
            width_increments+=paired_case_rows([r for r in widthjoined if r['arm']==reference and r['phase_step']==phase],
                [r for r in widthjoined if r['arm']==arm and r['phase_step']==phase],DISPLAY[arm]+'_minus_'+DISPLAY[reference],
                WIDTH_METRICS,metadata=dict(phase_step=phase))
    wip,wis=summarize_pairs(width_increments,WIDTH_METRICS,['comparison','phase_step','table','split','ratio'])
    save('B_width_matched_budget_per_case.csv',width_increments);save('B_width_matched_budget_summary.csv',wis)
    historical=historical_holdout(source,history,out,panels);save('C_historical_background.csv',historical)
    hidx={(r['arm'],r['conditional_total_step'],r['case_id']):r for r in historical}
    hold=[]
    for current in [r for r in rows if r['table']=='C']:
        r=dict(current)
        for total in (1000,5000):
            old=hidx[(r['arm'],total,r['case_id'])]
            r[f'old{total}_reuse_status']=old['status']
            r[f'old{total}_comparison_scope']='old1000_source_model_same_input' if total==1000 else 'historical_different_training_task_and_budget'
            for m in geometry_metrics:
                before=old.get(m);after=r.get(m)
                r[m+f'_old{total}']=before
                r[m+f'_new4000_minus_old{total}']=float(after)-float(before) if finite(after) and finite(before) else None
        hold.append(r)
    save('C_holdout_regression.csv',hold)
    hpairs=[]
    for arm in ARMS:
        after=[r for r in rows if r['table']=='C' and r['arm']==arm]
        for total in (1000,5000):
            before=[r for r in historical if r['arm']==arm and r['conditional_total_step']==total]
            hpairs+=paired_case_rows(before,after,DISPLAY[arm]+f'_new4000_minus_old{total}',geometry_metrics)
    hpp,hps=summarize_pairs(hpairs,geometry_metrics,['comparison','table','split','ratio'])
    save('C_holdout_paired_per_parent.csv',hpp);save('C_holdout_paired_summary.csv',hps)
    failures=[dict(scope='generation',**r) for r in rows if r['status']!='PASS' or r['score_status']!='PASS']
    failures += [dict(scope='teacher',**r) for r in teachers if r['status']!='PASS']
    save('failure_classification.csv',failures)
    figure_manifest=figures(out,panels,rows,curves,ts)
    completed=sum(l['sampling_counts']['complete_rollouts'] for l in ledgers)
    teacher_completed=sum(l['teacher_counts']['teacher_returns'] for l in ledgers)
    updates={}
    for arm in ARMS:
        p=out/'training'/(arm+'_status.json')
        status=read_json(p) if p.exists() else {}
        updates[arm]=status.get('counts',{}).get('optimizer_updates',status.get('new_updates'))
    from .finalize import finalize
    final_accounting=finalize(source,out)
    report=[
        '# CHAIR_DOMAIN_CONTROL_REPORT',
        '',
        f'实际新增训练更新：{updates}；实际完整生成 {completed}/582（登记 600，几何不适用 18）；固定 teacher {teacher_completed}/288。所有计数来自实际账本。正式终点累计条件步数 4000，新任务 phase_step=3000。',
        '',
        'Native=T1；Native+Geo=Geo；Native+Geo+Joint=Full。各臂从自己的旧 step1000 模型及完整优化器出发。研究主干选用 Native+Geo 不等于预先认定它整体最优。',
        '',
        '补全结果按父对象等权，并分训练父/已观察验证父、20%/40%面数条件及实际条件面积。A 表保留 3 个不可构造的40%条件，每臂6个生成格标为 NOT_APPLICABLE；不删除父对象或换对象凑数。',
        '',
        '| 模型 | 划分 | K档 | 有效父/请求父 | 有效case/请求 | remainder RMS (%L) | GT coverage | Gamma edge RMS (%L) | strict Gamma coverage | FF affected | small-angle fraction |',
        '| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in summaries:
        if r['table']=='A' and r['phase_step']==3000:
            report.append('| '+' | '.join([DISPLAY[r['arm']],r['split'],str(r['ratio']),f"{r['valid_parents']}/{r['requested_parents']}",f"{r['metric_valid_cases']}/{r['requested_cases']}"]+[fmt(r[m]) for m in PRIMARY[:4]+[PRIMARY[5],PRIMARY[6]]])+' |')
    report += [
        '',
        '几何响应只测自由部分三角角坐标的稳健跨度 q95−q05。B_width_response.csv 分列绝对宽度误差、相对自身alpha=1的残差、斜率、单调、Y/Z跨度和三轴中位坐标误差。极小基准宽度的比值/斜率为 UNDEFINED。C精确保留是采样构造性质，不计为学得的控制精度。',
        '',
        'B_cross_condition_scores.csv 是固定对应GT点到生成自由三角面的单向诊断，距离使用共同L0；与原面积加权双向RMS严格分列。原评分器仍使用对应alpha目标的L_alpha，新增L0列只转换距离单位，没有缩放输出坐标。',
        '',
        '起点已有多少响应、训练新增多少，只能查看 shared_start_endpoint_* 和 B_width_start_endpoint_* 的真正共有第一噪声配对。终点额外噪声与40%条件不混入起点配对；均值与中位数均在先算配对父差后汇总，不能拿两组各自中位数相减。',
        '',
        'No-response和Post-scale是零模型前向解析参照。Post-scale的alpha_hat只由编辑前后C对应X坐标拟合；它额外持有编辑前C与旧输出，输入合同不等同于只见当前C的生成器。两类候选使用当前C附接并对当前目标评分，CPU候选/评分/缓存命中单独记账，不算新增独立生成。即使模型响应接近理想斜率，也仅能称域内几何条件响应；是否优于Post-scale要同时看绝对几何、覆盖、接口与相交。',
        '',
        'Geo增量与Joint增量见 matched_budget_summary.csv：按相同phase、相同输入、相同父对象配对，报告均值、中位配对差、改善父数、最坏父与请求分母。Joint是否保住覆盖，必须同时读取 coverage、严格接口、CF/FF和角质量，不能仅用JEdge5或低RMS判断。',
        '',
        '20%/40%效果见 A_context_summary.csv 和 A_context_area_strata.csv；不可用40%格保留NA原因，不把缺失填零或解释为模型失败。训练父只表示新C组合未用于已知微调，完整物体和许多面已见；验证父与原留出均已被研究观察。',
        '',
        'C_holdout_regression.csv 是已观察留出回归。旧1000/5000只在checkpoint/raw/实际输入/评分器身份全匹配时读取原分数，缺失为NA。新累计4000与旧5000属于不同预算和训练任务的历史描述，不能当作同预算数据消融。',
        '',
        '本轮同时改变C任务分布和宽度训练，没有“仅更多C”或“仅宽度”独立训练臂；不能分离两者因果贡献。宽度alpha=.9/1/1.1为训练范围内部及边界，不是范围外推。无可信语义标签时只称规范X向、free整体几何响应，不声称部件姿态/连接重构或任意局部编辑。训练seed只有1，结果是开发性域内验证。',
        '',
        f"CPU控制分析：{analysis['counts']}。所有解析候选和3×3评分均为复用输出的CPU工作，零新增模型前向。图件见 figures/manifest.json；B的alpha之间使用同一原坐标镜头与共同范围，GT覆盖颜色来自实际评分。",
        '',
        '全局水密：NOT_RUN。评分有效不表示质量通过；严格接合、非法相交和角质量问题按原值保留。预算结束停止，不自动更换公共默认模型或启动下一轮。',
        '',
        '数据、代码、检查点与真实执行总账以 protocol.json、task_manifest.json、checkpoint_manifest.json、analysis_ledger.json、evaluation/ 与 training/ 目录为准。'
    ]
    from .conclusions import evidence_lines
    report += evidence_lines(summaries,ins,width_summary,wpairs,wps,aps,hps)
    report += ['', 'Inherited historical BF16 permutation FAIL: 10/1008 coordinates; max_abs=.064453125; atol=.015625; rtol=.03125. Not rerun, not reclassified PASS. Repeated complete generations: 0 (none claimed).']
    report += ['', '| B endpoint model | split | alpha | valid parents/requested | width abs error / L0 | slope | abs slope error | monotonic fraction | correct-column advantage / L0 |',
               '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in width_summary:
        if r['phase_step']==3000:
            report.append('| '+' | '.join([DISPLAY[r['arm']],r['split'],str(r['alpha']),f"{r['valid_parents']}/{r['requested_parents']}"]+[fmt(r.get(m)) for m in ['X_span_abs_error_L0','slope','slope_abs_error','monotonic','wrong_mean_minus_diagonal_RMS_L0']])+' |')
    report += ['', '| Paired endpoint increment | table | split | K fraction | metric | mean parent delta | median parent delta | improved/valid/requested parents | worst parent delta |',
               '| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: |']
    for r in ins:
        if r['phase_step']==3000 and r['metric'] in PRIMARY[:4] and r['table']=='A':
            report.append('| '+' | '.join([r['comparison'],r['table'],r['split'],str(r['ratio']),r['metric'],fmt(r['mean_paired_parent_delta']),fmt(r['median_paired_parent_delta']),f"{r['improved_parents']}/{r['paired_valid_parents']}/{r['requested_parents']}",fmt(r['worst_parent_delta'])])+' |')
    path=out/'CHAIR_DOMAIN_CONTROL_REPORT.md'
    path.write_text('\n'.join(report)+'\n',encoding='utf-8');artifact_names.append(path.name)
    artifact_names += ['checkpoint_manifest.json','execution_accounting.json','resource_summary.csv',
                       'protected_assets_after.json','restart.txt','analysis_ledger.json','history_reuse.json',
                       'figures/manifest.json','figures/fixed_manifest.json','code.diff',
                       'protocol.json','panels.json','train_plan.json','task_manifest.json','training_learning_curves.csv']
    status=dict(status='PASS',requested_generation_slots=len(rows),
                geometry_not_applicable_slots=sum(r['geometry_status']=='NOT_APPLICABLE' for r in rows),
                planned_applicable_generations=sum(r['geometry_status']=='READY' for r in rows),
                completed_generations=completed,score_valid_generations=sum(r['score_status']=='PASS' for r in rows),
                requested_teacher_slots=len(teachers),completed_teacher=teacher_completed,
                analytic_candidate_requests=analysis['analytic_candidate_rows'],
                analytic_candidate_complete=analysis['counts'].get('analytic_candidate_returns',0),
                analytic_score_cache_hits=analysis['counts'].get('analytic_score_cache_hits',0),
                cross_condition_requests=analysis['cross_condition_rows'],
                cross_condition_complete=analysis['counts'].get('cross_condition_group_returns',0)*9,
                historical_rows_reused=sum(r['status']=='REUSED' for r in historical),
                figures=len(figure_manifest['files']),
                artifacts=[dict(path=name,sha256=digest(out/name)) for name in artifact_names]+figure_manifest['files'])
    atomic_json(out/'reporting_status.json',status)
    return status


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True)
    p.add_argument('--out',required=True);p.add_argument('--history',required=True)
    a=p.parse_args();run(a.source,a.out,a.history)


if __name__=='__main__':
    main()
