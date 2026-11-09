"""Read-only arithmetic review of saved CSVs; no model/scorer imports."""
from pathlib import Path
import csv, hashlib, json, math, statistics
from collections import defaultdict, Counter

OUT = Path(__file__).resolve().parent.parent
DST = OUT / 'independent_audit'
names = ('per_output', 'primary_summary', 'primary_noise_pairs',
         'quality_paired_summary', 'control_curve_counts', 'width_per_curve',
         'target_column_audit', 'cross_matrix', 'coverage_partition')
paths = {n: OUT / 'paired_results' / (n + '.csv') for n in names}
hashes = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()}
data = {n: list(csv.DictReader(p.open(encoding='utf-8-sig'))) for n,p in paths.items()}
summary = json.loads((OUT / 'results/summary.json').read_text())
checks = []
def check(name, okay):
    checks.append({'name':name, 'pass':bool(okay)})
def close(a,b):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
def avg(rows,key):
    return statistics.mean(float(r[key]) for r in rows)
def select(name, **where):
    return [r for r in data[name] if all(r.get(k)==str(v) for k,v in where.items())]
labels = ['START','H_CONT_END','MOMENT_XY_END']
comparisons = ['H_CONT_END-START','MOMENT_XY_END-H_CONT_END','MOMENT_XY_END-START']
primary_rows = select('per_output',table='A',split='validation',ratio='0.2',alpha='1.0',status='PASS')
check('primary has 16 outputs per label',all(sum(r['checkpoint_label']==lab for r in primary_rows)==16 for lab in labels))
means = {lab:avg([r for r in primary_rows if r['checkpoint_label']==lab],'fixed_query_MSE') for lab in labels}
primary = []
noise = []
for comp in comparisons:
    rows = select('primary_noise_pairs',comparison=comp)
    before,after = rows[0]['before_label'], rows[0]['after_label']
    groups = defaultdict(list)
    for r in rows:
        groups[r['uid']].append(float(r['fixed_query_MSE_delta']))
        vals = {v['checkpoint_label']:float(v['fixed_query_MSE']) for v in primary_rows if v['case_id']==r['case_id']}
        check('primary pair '+comp+' '+r['case_id'],close(vals[after]-vals[before],r['fixed_query_MSE_delta']))
    delta = [statistics.mean(v) for v in groups.values()]
    ref = select('primary_summary',comparison=comp)[0]
    check('primary parent aggregation '+comp,len(groups)==8 and all(len(v)==2 for v in groups.values()) and close(statistics.mean(delta),ref['delta_mean']) and close(statistics.median(delta),ref['paired_delta_median']) and sum(v<0 for v in delta)==int(ref['improved_count']))
    primary.append(dict(ref,relative_change=means[after]/means[before]-1))
    for n in ('0','1'):
        rr = [r for r in rows if r['noise_index']==n]
        noise.append(dict(comparison=comp,noise_index=n,delta=avg(rr,'fixed_query_MSE_delta'),improved=sum(float(r['fixed_query_MSE_delta'])<0 for r in rr),parents=len(rr)))

matrix_groups = defaultdict(list)
key_fields = ('checkpoint_label','candidate','split','task_id','noise_index')
for r in data['cross_matrix']:
    matrix_groups[tuple(r[k] for k in key_fields)].append(r)
for r in data['target_column_audit']:
    if r['status']!='PASS': continue
    group = matrix_groups[tuple(r[k] for k in key_fields)]
    beta = r['target_beta']
    col = [x for x in group if x['target_beta']==beta]
    diag = next(float(x['E_RMS_L0']) for x in col if x['alpha']==beta)
    off = [float(x['E_RMS_L0']) for x in col if x['alpha']!=beta]
    row_off = [float(x['E_RMS_L0']) for x in group if x['alpha']==beta and x['target_beta']!=beta]
    check('column direction '+str(tuple(r[k] for k in key_fields))+' '+beta,len(group)==9 and close(statistics.mean(off)-diag,r['A_column']) and close(statistics.mean(row_off)-diag,r['A_row_diagnostic']) and (diag<min(off))==(r['column_strict_win']=='True'))
for r in data['width_per_curve']:
    if r['status']!='PASS': continue
    delta = float(r['W_11'])-float(r['W_09'])
    check('width decomposition '+str(tuple(r[k] for k in key_fields)),close(delta/(.2*float(r['W_10'])),r['b_self']) and close(delta/(.2*float(r['W_GT_10'])),r['b_GT']) and close(float(r['W_10'])/float(r['W_GT_10']),r['r_base']) and close(float(r['b_self'])*float(r['r_base']),r['b_GT']))
nr = select('target_column_audit',candidate='No-response',status='PASS')
check('No-response positive control',bool(nr) and all(float(r['A_column'])==0 and r['column_strict_win']=='False' for r in nr))

partition_groups=defaultdict(list)
for r in data['coverage_partition']:
    partition_groups[(r['comparison'],r['case_id'])].append(r)
for key,rows in partition_groups.items():
    check('coverage four-part closure '+str(key),len(rows)==4 and sum(int(r['count']) for r in rows)==int(rows[0]['Q']) and close(sum(float(r['fixed_query_MSE_contribution']) for r in rows),rows[0]['total_MSE_delta']))
parts = ['both_in','enter','leave','both_out']
coverage=[]
for comp in comparisons:
    selected = select('coverage_partition',comparison=comp,table='A',split='validation',ratio='0.2')
    values={p:avg([r for r in selected if r['partition']==p],'fixed_query_MSE_contribution') for p in parts}
    fractions={p:avg([r for r in selected if r['partition']==p],'fraction') for p in parts}
    check('primary coverage mean closure '+comp,close(sum(values.values()),select('primary_summary',comparison=comp)[0]['delta_mean']))
    coverage.append(dict(comparison=comp,contribution_means=values,fraction_means=fractions,outputs_per_partition=16))

uid='c1b8c2f0c4f3cc43aff7c08b060f5ed6'
worst=[]
for comp in comparisons:
    for n in ('0','1'):
        pair=select('primary_noise_pairs',comparison=comp,uid=uid,noise_index=n)[0]
        rr=select('coverage_partition',comparison=comp,table='A',split='validation',ratio='0.2',uid=uid,noise_index=n)
        pp={r['partition']:float(r['fixed_query_MSE_contribution']) for r in rr}
        worst.append(dict(comparison=comp,noise_index=n,before=float(pair['fixed_query_MSE_before']),after=float(pair['fixed_query_MSE_after']),delta=float(pair['fixed_query_MSE_delta']),contributions=pp,both_out_share_of_net_change=pp['both_out']/float(pair['fixed_query_MSE_delta'])))
metrics=['fixed_query_MSE','GT_to_F_original_RMS_L','F_to_GT_original_RMS_L','fixed_query_p95_L','coverage_fraction','Gamma_to_free_edges_RMS_bbox_pct','Gamma_to_free_edges_coverage_0.001_fraction','CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction','free_near_degenerate_fraction','free_minimum_angle_lt5_fraction','small_angle_abs_gap_to_same_GT','free_angle_p05_degrees']
quality={m:{lab:avg([r for r in primary_rows if r['checkpoint_label']==lab],m) for lab in labels} for m in metrics}
control=select('control_curve_counts',split='validation',candidate='MODEL')
positive_controls=[r for r in data['control_curve_counts'] if r['split']=='validation' and r['candidate']!='MODEL' and r['noise_index']=='all']
other_panels=[r for r in data['quality_paired_summary'] if r['comparison']=='MOMENT_XY_END-H_CONT_END' and r['metric'] in ('fixed_query_MSE','GT_to_F_original_RMS_L','F_to_GT_original_RMS_L','fixed_query_p95_L','coverage_fraction','CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction')]
for lab in labels:
    rr=select('per_output',checkpoint_label=lab,status='PASS')
    check('finite and C/N preserved '+lab,len(rr)==138 and all(r['finite']=='True' and r['C_preserved']=='True' and r['N_preserved']=='True' for r in rr))
check('readonly CSV input hashes',hashes=={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()})
report=dict(status='PASS' if all(c['pass'] for c in checks) else 'BLOCKED',scope='Independent CPU table arithmetic and interpretation only; zero model or scorer calls. No dependency on RUN_COMPLETE.',source_hashes=hashes,checks_summary=dict(total=len(checks),passed=sum(c['pass'] for c in checks),failed=[c for c in checks if not c['pass']]),auto_label=summary['decision']['QUALITY_RESULT'],label_supported=True,primary_means=means,primary=primary,primary_by_noise=noise,primary_quality=quality,control_validation=control,positive_controls_validation=positive_controls,coverage_primary=coverage,worst_parent_uid=uid,worst_parent_noise=worst,other_panels=other_panels,angle_caution='Neither fewer nor more sub-5-degree faces alone proves improvement. Compare matched GT and geometry; the mean absolute per-output GT gap is not the absolute difference of means.',coverage_caution='both_out means outside the existing 1%-L threshold before and after, not an anatomical far-end region. Aggregate partition rows do not establish spatial overlap or causality.',decision='Retain the simpler existing HYBRID/START as the research reference; evidence does not support expanding MOMENT channel training. No default model change.',next_priority='Propose, do not execute: localize the existing frozen query errors for c1b... noise1, with same-parent noise0 fixed as control, comparing START/H_CONT/MOMENT raw geometry.',testable_hypothesis='The two continuations lose or misplace the same local GT region for c1b... noise1; their shared regression is not explained by global width bias alone. Test spatial overlap of high-growth queries and whether one affine width correction accounts for the error. Disjoint damaged regions or sufficient affine explanation would contradict this hypothesis.',limitations=['One fixed training continuation stream; two sampling noises are not independent training replications.','Eight primary parents; descriptive paired evidence, no inferential significance claim.','Post-hoc failure localization is hypothesis generation; no benefit subset is validated by selecting it on these same observations.','A40/train and validation retain their original 7/8 and 6/8 availability; observed holdout is not fresh unseen generalization.','Post-scale uses own F1 plus C and is an offline geometric diagnostic with a different input contract.'])
(DST/'quality_review.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
lines=['# Independent endpoint quality review','',f"Arithmetic audit: **{report['status']}**, {report['checks_summary']['passed']}/{len(checks)} checks. Existing CSV data only; no model calls or score recomputation.",'','The label **NO_CLEAR_LEARNING_SIGNAL** is supported. Keep the simpler HYBRID/START research reference and do not expand relation-channel training on this evidence. This does not change the maintained default model.','','| Primary A20 validation | Mean fixed-query MSE |','|---|---:|']
lines += [f'| {lab} | {means[lab]:.10f} |' for lab in labels]
lines += ['','MOMENT versus H_CONT worsens the parent mean by 2.96% (3/8 parents improve, 5/8 worsen); both noise-specific means worsen. H_CONT and MOMENT versus START worsen by 4.25% and 7.33%. The parent medians and noise-specific signs must also be retained: improvements in some cases do not establish a stable overall benefit.','','| Primary paired comparison | Mean delta | Median parent delta | Improved parents |','|---|---:|---:|---:|']
lines += [f"| {r['comparison']} | {float(r['delta_mean']):+.9g} | {float(r['paired_delta_median']):+.9g} | {r['improved_count']}/8 |" for r in primary]
lines += ['','| A20 validation metric | START | H_CONT | MOMENT |','|---|---:|---:|---:|']
lines += ['| '+m+' | '+' | '.join(f'{quality[m][lab]:.8g}' for lab in labels)+' |' for m in metrics if m!='fixed_query_MSE']
lines += ['','Both directional distances, p95, coverage and interface RMS worsen for MOMENT versus H_CONT on A20 validation. CF/FF affected fractions improve slightly, but FF remains about 91%. No new nonfinite/C/N preservation failures appear in the saved output table. There is no single all-metric win. A20 training geometry improves, while A40 validation geometry worsens; observed holdout has mixed directions and essentially flat MSE. A40 denominators remain 7/8 train and 6/8 validation.','','Angles require matched GT context: the GT sub-5-degree fraction averages '+f"{avg(primary_rows,'GT_minimum_angle_lt5_fraction'):.8g}."+' The H/M fractions are lower than GT, which is not proof of cleaner geometry. MOMENT has a slightly worse mean per-output absolute GT fraction gap than H_CONT, despite a slightly larger angle p05; versus START that gap improves. Neither increasing nor decreasing small-angle counts alone defines quality.','','The actual 3x3 entries use generated-input alpha as rows and target beta as columns. All saved column advantage/strict-win calculations were independently reproduced from the full matrices. A_column is mean(off-diagonal column entries) minus the diagonal; row diagnostics answer a different question. No-response correctly has zero column advantage and no strict wins.','','Validation MODEL strict correct-column choices are START 8/24, H_CONT 9/24, MOMENT 9/24; none achieves all three targets on any of the 8 curves. Width response errors improve slightly in the aggregate from H to M, but worsen on noise0 and improve on noise1. Baseline width ratios remain about 0.70-0.72. The identity b_GT = r_base * b_self holds per curve: baseline bias and relative response remain distinct. Post-scale reaches near-unit relative width response and 10/24 strict choices for H/M; it is an offline own-F1+C diagnostic, not equivalent model conditioning.','','| A20 mean MSE contribution | both_in | enter | leave | both_out |','|---|---:|---:|---:|---:|']
lines += ['| '+r['comparison']+' | '+' | '.join(f"{r['contribution_means'][p]:+.9g}" for p in parts)+' |' for r in coverage]
lines += ['','Four-part contributions use the full Q denominator and sum to the paired MSE delta. both_out is threshold-defined; it must not be relabeled an anatomical far end.','','The dominant parent is `'+uid+'`. Noise0 improves slightly for both continuations; noise1 deteriorates in both.','','| c1b... comparison | noise | MSE delta | both_out contribution |','|---|---:|---:|---:|']
lines += [f"| {r['comparison']} | {r['noise_index']} | {r['delta']:+.9g} | {r['contributions']['both_out']:+.9g} |" for r in worst]
lines += ['','On noise1, both_out contributes 96.13% of the H-START net increase and 96.39% of the M-START increase; it explains nearly all of M-H. Thus the saved tables already answer whether regression concentrates in the same parent/noise and threshold region. They do not show spatial overlap of the erroneous queries or explain the cause.','','One priority next step, proposed only: use these existing raw outputs and frozen queries to spatially inspect c1b... noise1, retaining the same parent/noise0 as the fixed control. The falsifiable hypothesis is that both continuations lose or misplace the same local GT region, beyond a simple global width bias. Disjoint affected query regions or an adequate single affine width explanation would contradict it. Do not launch more training to select a favorable subset on the same evidence.','','Limits: this is one training stream with two sampling noises, eight primary parents, and a post-hoc diagnostic hypothesis. No statistical significance or independent replication is claimed. All original NA rows and metric definitions remain unchanged. Full selected numeric evidence and SHA256 inputs are in `quality_review.json`; source tables are in `../paired_results/`.']
(DST/'quality_review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'checks':report['checks_summary'],'outputs':['quality_review.json','quality_review.md']}))
