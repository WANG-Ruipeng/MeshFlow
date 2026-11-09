"""Audit existing 3x3 CSV arithmetic only; no experiment/scoring imports."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse, csv, hashlib, json, math, statistics, time
from collections import defaultdict
from pathlib import Path
A=(.9,1.,1.1)
E='corresponding_GT_to_free_RMS_L0'
C='corresponding_GT_coverage_1pct_L0_fraction'
OLD=('wrong_mean_minus_diagonal_RMS_L0','diagonal_no_worse_than_both_wrong','diagonal_strictly_better_than_both_wrong')
KEY=('arm','phase_step','split','uid','noise_index')
METRICS=('diagonal_RMS_L0','wrong_mean_minus_diagonal_RMS_L0','wrong_min_minus_diagonal_RMS_L0','diagonal_no_worse_than_both_wrong','diagonal_strictly_better_than_both_wrong','diagonal_coverage_minus_wrong_mean')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def number(v):
    if v in (None,'','NA'): return None
    if v in (True,'True'): return 1.
    if v in (False,'False'): return 0.
    v=float(v)
    assert math.isfinite(v)
    return v
def equal(x,y): return x is None and y is None or x is not None and y is not None and abs(float(x)-float(y))<=1e-12
def key(r): return tuple(r[k] for k in KEY)
def mean(xs): return statistics.fmean(xs)
def write_csv(out,name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    p=out/name
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in rows:w.writerow({k:'NA' if v is None else v for k,v in r.items()})
    return p
def write_json(out,name,data):
    p=out/name;p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return p
def summarize(rows,groups,metrics):
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in groups)].append(r)
    result=[]
    for keys,rs in sorted(grouped.items()):
        base=dict(zip(groups,keys));parents=defaultdict(list)
        for r in rs:parents[r['uid']].append(r)
        for metric in metrics:
            pv={u:mean([float(r[metric]) for r in prs]) for u,prs in parents.items()}
            xs=list(pv.values())
            higher=metric!='diagonal_RMS_L0' and not metric.endswith('RMS_L0_delta') or metric.startswith('wrong_')
            worst=min(pv,key=pv.get) if higher else max(pv,key=pv.get)
            result.append(dict(base,metric=metric,requested_parents=len(parents),valid_parents=len(parents),
                requested_axis_records=len(rs),valid_axis_records=len(rs),parent_equal_mean=mean(xs),
                median_parent_value=statistics.median(xs),minimum_parent_value=min(xs),maximum_parent_value=max(xs),
                positive_parents=sum(x>0 for x in xs),negative_parents=sum(x<0 for x in xs),zero_parents=sum(x==0 for x in xs),
                worst_parent=worst,worst_parent_value=pv[worst],
                aggregation='Within-parent noise/axis average, then equal parent weights; alpha is not an independent replicate'))
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();source=Path(args.source).resolve();out=Path(args.out).resolve()
    assert out.name=='matrix_audit' and out.parent.name=='chair_control_followup_v1'
    out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[4];code=root/'experiments/mf_h1_local0/chair_domain_control_v1'
    started=time.time()
    paths=[source/'B_cross_condition_scores.csv',source/'B_width_response.csv',
        source/'all_generation_per_case.csv',source/'panels.json',source/'protocol.json',
        code/'width_analytics.py',code/'reporting.py',source/'CHAIR_DOMAIN_CONTROL_REPORT.md',
        out.parent/'cloud_feedback.txt']
    before={str(p):sha(p) for p in paths}
    manifest=json.loads((source/'reporting_status.json').read_text())
    manifest_hashes={r['path']:r['sha256'] for r in manifest['artifacts']}
    for name in ('B_cross_condition_scores.csv','B_width_response.csv','all_generation_per_case.csv','panels.json','protocol.json','CHAIR_DOMAIN_CONTROL_REPORT.md'):
        assert manifest_hashes[name]==sha(source/name), 'Source manifest mismatch: '+name
    if (out/'audit.json').exists():
        prev=json.loads((out/'audit.json').read_text())
        if prev.get('status')=='PASS' and prev.get('input_sha256')==before and prev.get('script_sha256')==sha(__file__):
            assert all(sha(out/r['path'])==r['sha256'] for r in prev['artifacts'])
            print(json.dumps({'status':'REUSED','matrices':72,'new_model_calls':0,'new_geometry_score_calls':0}));return
        raise RuntimeError('Existing audit has different identity; do not overwrite')
    matrix=read_csv(paths[0]);width=read_csv(paths[1]);generated=read_csv(paths[2])
    assert len(matrix)==648 and len(width)==216
    groups=defaultdict(dict);widx={}
    for r in matrix:
        a,b=float(r['alpha']),float(r['target_beta']);assert a in A and b in A
        assert r['status']=='PASS' and r['diagnostic_scope']=='ONE_WAY_CORRESPONDING_GT_POINTS'
        assert r['correct_condition']==str(a==b) and int(r['corresponding_point_count'])==512
        assert (a,b) not in groups[key(r)];groups[key(r)][(a,b)]=r
    for r in width:
        k=key(r)+(float(r['alpha']),);assert k not in widx;widx[k]=r
    assert len(groups)==72
    assert {k[0] for k in groups}=={'T1','Geo','Full'}
    assert {k[1] for k in groups}=={'0','3000'}
    assert {k[2] for k in groups}=={'train','validation'}
    for arm in ('T1','Geo','Full'):
        for split in ('train','validation'):
            parent_sets=[]
            for phase,noise in (('0','0'),('3000','0'),('3000','1')):
                ps={k[3] for k in groups if k[0]==arm and k[1]==phase and k[2]==split and k[4]==noise}
                assert len(ps)==4;parent_sets.append(ps)
            assert parent_sets[0]==parent_sets[1]==parent_sets[2]
    gidx={(r['arm'],r['phase_step'],r['case_id']):r for r in generated}
    source_gen={}
    for w in width:
        g=gidx[(w['arm'],w['phase_step'],w['case_id'])]
        assert g['score_status']=='PASS' and g['status']=='PASS'
        source_gen[key(w)+(float(w['alpha']),)]=g
    axis=[];evidence=[];matrows=[];slopes=[]
    for k,cells in sorted(groups.items()):
        assert set(cells)=={(a,b) for a in A for b in A}
        base=dict(zip(KEY,k));base['conditional_total_step']=1000+int(base['phase_step'])
        assert len({r['task_id'] for r in cells.values()})==1 and len({r['L0'] for r in cells.values()})==1
        base.update(task_id=next(iter(cells.values()))['task_id'],L0=number(next(iter(cells.values()))['L0']))
        for a in A:
            w=widx[k+(a,)];diag=number(cells[(a,a)][E]);diagcov=number(cells[(a,a)][C]);by_direction={}
            for direction in ('fixed_output_alpha','fixed_target_beta'):
                positions=[(a,b) for b in A if b!=a] if direction=='fixed_output_alpha' else [(b,a) for b in A if b!=a]
                off=[number(cells[p][E]) for p in positions];cov=[number(cells[p][C]) for p in positions]
                row=dict(base,direction=direction,axis_value=a,axis_name='alpha' if direction=='fixed_output_alpha' else 'beta',
                    source_case_id=w['case_id'],source_input_sha256=source_gen[k+(a,)]['input_sha256'],
                    diagonal_RMS_L0=diag,wrong_RMS_1_L0=off[0],wrong_RMS_2_L0=off[1],
                    wrong_mean_minus_diagonal_RMS_L0=mean(off)-diag,wrong_min_minus_diagonal_RMS_L0=min(off)-diag,
                    diagonal_no_worse_than_both_wrong=diag<=min(off),diagonal_strictly_better_than_both_wrong=diag<min(off),
                    diagonal_coverage_minus_wrong_mean=diagcov-mean(cov))
                axis.append(row);by_direction[direction]=row
            row=by_direction['fixed_output_alpha'];col=by_direction['fixed_target_beta']
            ev=dict(base,alpha_or_beta=a,case_id=w['case_id'],old_diagonal=number(w['diagonal_corresponding_RMS_L0']),
                recomputed_diagonal=diag,diagonal_match=equal(number(w['diagonal_corresponding_RMS_L0']),diag))
            for field in OLD:
                v=number(w[field]);rr=float(row[field]);cc=float(col[field])
                ev.update({f'old_{field}':v,f'fixed_output_{field}':rr,f'old_minus_fixed_output_{field}':v-rr,
                    f'old_matches_fixed_output_{field}':equal(v,rr),f'fixed_target_{field}':cc,
                    f'fixed_target_minus_old_{field}':cc-v,f'old_matches_fixed_target_{field}':equal(v,cc)})
            evidence.append(ev)
        ar=[r for r in axis if tuple(r[z] for z in KEY)==k]
        row=[r for r in ar if r['direction']=='fixed_output_alpha'];col=[r for r in ar if r['direction']=='fixed_target_beta']
        m=dict(base)
        for d,rs in [('row',row),('column',col)]:
            m.update({d+'_mean_advantage_L0':mean([r[OLD[0]] for r in rs]),
                d+'_diagonal_strict_win_fraction':mean([r[OLD[2]] for r in rs]),
                d+'_all_three_diagonal_strict_wins':all(r[OLD[2]] for r in rs)})
        m['mean_advantage_column_minus_row']=m['column_mean_advantage_L0']-m['row_mean_advantage_L0']
        assert abs(m['mean_advantage_column_minus_row'])<1e-12;matrows.append(m)
        ws=[widx[k+(a,)] for a in A]
        for name in ('slope','slope_abs_error','monotonic','strict_monotonic','W_09','W_10','W_11'):
            assert len({w[name] for w in ws})==1
        slopes.append(dict(base,**{f:ws[0][f] for f in ('slope','slope_abs_error','monotonic','strict_monotonic')},
            source_repeated_alpha_rows=3,independent_matrix_observations=1))
    assert len(axis)==432 and len(evidence)==216 and len(matrows)==72 and len(slopes)==72
    summary_noise=summarize(axis,['direction','arm','split','phase_step','noise_index','axis_value'],METRICS)
    summary=summarize(axis,['direction','arm','split','phase_step','axis_value'],METRICS)
    summary_all=summarize(axis,['direction','arm','split','phase_step'],METRICS)
    aix={(r['direction'],r['arm'],r['split'],r['uid'],r['noise_index'],r['phase_step'],r['axis_value']):r for r in axis}
    paired=[]
    for r in axis:
        if r['phase_step']!='3000' or r['noise_index']!='0':continue
        prev=aix[(r['direction'],r['arm'],r['split'],r['uid'],'0','0',r['axis_value'])]
        assert r['source_input_sha256']==prev['source_input_sha256'] and r['source_case_id']==prev['source_case_id']
        pair={k:r[k] for k in ('direction','arm','split','uid','noise_index','axis_value','axis_name','source_case_id','source_input_sha256')}
        pair.update(before_phase=0,after_phase=3000,true_shared_input=True)
        for m in METRICS:pair.update({m+'_before':prev[m],m+'_after':r[m],m+'_delta':float(r[m])-float(prev[m])})
        paired.append(pair)
    assert len(paired)==144
    deltas=[m+'_delta' for m in METRICS]
    ps=summarize(paired,['direction','arm','split','axis_value'],deltas)
    pall=summarize(paired,['direction','arm','split'],deltas)
    mpidx={(r['arm'],r['split'],r['uid'],r['noise_index'],r['phase_step']):r for r in matrows};mp=[]
    for r in matrows:
        if r['phase_step']!='3000' or r['noise_index']!='0':continue
        p=mpidx[(r['arm'],r['split'],r['uid'],'0','0')]
        row={z:r[z] for z in ('arm','split','uid','noise_index')}
        for f in ('row_mean_advantage_L0','column_mean_advantage_L0','row_diagonal_strict_win_fraction','column_diagonal_strict_win_fraction','row_all_three_diagonal_strict_wins','column_all_three_diagonal_strict_wins'):
            row.update({f+'_before':p[f],f+'_after':r[f],f+'_delta':float(r[f])-float(p[f])})
        mp.append(row)
    affected=[]
    for p in sorted(source.glob('*.csv')):
        if not p.name.startswith('B_'):continue
        rs=read_csv(p);n=0
        for r in rs:
            if any(r.get('metric')==f or any(k==f or k.startswith(f+'_') for k,v in r.items() if v not in ('','NA',None)) for f in OLD):n+=1
        if n:affected.append(dict(path=str(p),rows_containing_finite_or_named_direction_statistic=n,
            issue='Existing numbers are fixed-output rows, not fixed-target columns; preserve and relabel or use corrected audit files'))
    affected.append(dict(path=str(source/'CHAIR_DOMAIN_CONTROL_REPORT.md'),rows_containing_finite_or_named_direction_statistic=None,
        issue='The correct-column advantage header actually displays fixed-output-row advantage'))
    files=[]
    def save(name,rows):files.append(write_csv(out,name,rows))
    save('fixed_output_alpha_per_case.csv',[r for r in axis if r['direction']=='fixed_output_alpha'])
    save('fixed_target_beta_per_case.csv',[r for r in axis if r['direction']=='fixed_target_beta'])
    save('old_field_consistency_216.csv',evidence)
    save('matrix_level_72.csv',matrows)
    save('axis_summary_by_noise.csv',summary_noise)
    save('axis_summary_parent_equal.csv',summary)
    save('all_axes_parent_equal_summary.csv',summary_all)
    save('matched_noise0_start_endpoint_per_case.csv',paired)
    save('matched_noise0_start_endpoint_axis_summary.csv',ps)
    save('matched_noise0_start_endpoint_parent_summary.csv',pall)
    save('matched_noise0_matrix_pairs_24.csv',mp)
    save('slope_deduplicated_72.csv',slopes)
    save('affected_old_artifacts.csv',affected)
    facts=dict(source_matrix_cells=len(matrix),complete_matrices=len(groups),old_width_records=len(width),
        fixed_output_rows=216,fixed_target_rows=216,matched_axis_pairs=144,matched_matrix_pairs=24,
        slope_independent_matrix_records=72,source_repeated_slope_rows=216,
        diagonal_matches=sum(r['diagonal_match'] for r in evidence),
        old_fixed_output_margin_matches=sum(r['old_matches_fixed_output_'+OLD[0]] for r in evidence),
        old_fixed_output_no_worse_matches=sum(r['old_matches_fixed_output_'+OLD[1]] for r in evidence),
        old_fixed_output_strict_win_matches=sum(r['old_matches_fixed_output_'+OLD[2]] for r in evidence),
        old_fixed_target_margin_matches=sum(r['old_matches_fixed_target_'+OLD[0]] for r in evidence),
        old_fixed_target_no_worse_matches=sum(r['old_matches_fixed_target_'+OLD[1]] for r in evidence),
        old_fixed_target_strict_win_matches=sum(r['old_matches_fixed_target_'+OLD[2]] for r in evidence),
        changed_advantage_signs=sum((r['old_'+OLD[0]]>0)!=(r['fixed_target_'+OLD[0]]>0) for r in evidence),
        max_old_margin_vs_row_abs_difference=max(abs(r['old_minus_fixed_output_'+OLD[0]]) for r in evidence),
        max_all_alpha_mean_column_row_difference=max(abs(r['mean_advantage_column_minus_row']) for r in matrows))
    assert facts['old_fixed_output_margin_matches']==facts['old_fixed_output_no_worse_matches']==facts['old_fixed_output_strict_win_matches']==facts['diagonal_matches']==216
    files.append(write_json(out,'facts.json',facts))
    after={str(p):sha(p) for p in paths};assert before==after
    audit=dict(status='PASS',scope='Read existing CSV numeric fields and frozen source text only',input_sha256=before,
        source_hashes_unchanged=True,script_sha256=sha(__file__),CPU_workers=1,threads=1,
        new_training_updates=0,new_generations=0,new_model_forwards=0,new_model_backwards=0,new_OT_calls=0,
        new_geometry_score_calls=0,new_point_distance_calls=0,elapsed_seconds=time.time()-started,
        matrix_definition='E[alpha,beta]=one-way corresponding GT points Y_beta to predicted free surface F_alpha RMS / L0',
        old_direction='Fixed generated-output alpha row; target beta varies',
        corrected_direction='Fixed target beta column; generated-output alpha varies',
        tolerance='1e-12 for numerical equality only; strict wins use literal <; no near-tie quality threshold',
        findings=facts,artifacts=[dict(path=p.name,sha256=sha(p)) for p in files])
    write_json(out,'audit.json',audit)
    print(json.dumps(facts,indent=2))
if __name__=='__main__':main()
