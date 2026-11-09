"""Existing-evidence CPU analysis. No model imports, surface scoring or new sampling."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import time
import numpy as np

ROOT = Path('/mnt/e/MeshFlow')
SOURCE = ROOT / 'native_t1/runs/chair_domain_control_v1'
OUT = Path(__file__).resolve().parent / 'width_analysis'
KEY = ['arm', 'phase_step', 'split', 'uid', 'noise_index']
QUALITY = ['remainder_symmetric_RMS_L0_pct', 'fixed_R12_coverage_within_1pct_fraction',
           'Gamma_to_free_edges_RMS_L0_pct', 'Gamma_to_free_edges_coverage_0.001_fraction',
           'CF_proper_affected_free_fraction', 'FF_proper_affected_free_fraction',
           'free_minimum_angle_lt5_fraction']
EPS = 1e-12


def read_csv(name):
    with (SOURCE / name).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    assert rows, name
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({k: 'NA' if v is None else v for k, v in r.items()} for r in rows)


def number(row, key):
    value = row[key]
    if value in ('', 'NA', None):
        return None
    return float(value)


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(values)) if values else None


def group(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    started = time.time()
    OUT.mkdir(exist_ok=True)
    rows = read_csv('B_width_response.csv')
    geometry = read_csv('B_width_geometry_per_case.csv')
    analytics = read_csv('B_analytic_baselines.csv')
    panels = json.loads((SOURCE / 'panels.json').read_text())
    bank_index = {r['case_id']: r for r in panels['B']}
    geo_index = {(r['arm'], r['phase_step'], r['case_id']): r for r in geometry}
    assert len(rows) == len(geometry) == len(geo_index) == 216
    assert len(analytics) == 432
    groups = group(rows, KEY)
    assert len(groups) == 72
    checked = {}
    arrays = {}
    raw_checks = []
    noise_by_group = {}
    for row in rows:
        ident = (row['arm'], row['phase_step'], row['case_id'])
        g = geo_index[ident]
        bank = bank_index[row['case_id']]
        for name, expected in [(g['raw_path'], g['raw_sha256']), (bank['input_path'], bank['input_sha256'])]:
            if name not in checked:
                actual = sha(SOURCE / name)
                assert actual == expected, name
                checked[name] = actual
        if bank['input_path'] not in arrays:
            with np.load(SOURCE / bank['input_path'], allow_pickle=False) as data:
                arrays[bank['input_path']] = {k: data[k] for k in ('C', 'gt_free', 'L0', 'z')}
        data = arrays[bank['input_path']]
        with np.load(SOURCE / g['raw_path'], allow_pickle=False) as raw:
            free = np.asarray(raw['raw'][len(data['C']):], dtype=np.float64).reshape(-1, 3)
            assert np.array_equal(raw['raw'][:len(data['C'])].reshape(-1, 3, 3), data['C'])
            assert np.array_equal(raw['noise'], data['z']), ident
            key = tuple(row[k] for k in KEY)
            if key in noise_by_group:
                assert np.array_equal(noise_by_group[key], raw['noise']), key
            else:
                noise_by_group[key] = raw['noise'].copy()
        gt = np.asarray(data['gt_free'], dtype=np.float64).reshape(-1, 3)
        L0 = float(data['L0'])
        predicted_span = np.quantile(free, .95, axis=0) - np.quantile(free, .05, axis=0)
        gt_span = np.quantile(gt, .95, axis=0) - np.quantile(gt, .05, axis=0)
        pred_median, gt_median = np.median(free, axis=0), np.median(gt, axis=0)
        residuals = []
        for axis, dim in zip('XYZ', range(3)):
            values = {axis+'_span':predicted_span[dim], axis+'_GT_span':gt_span[dim],
                      axis+'_median':pred_median[dim], axis+'_GT_median':gt_median[dim],
                      axis+'_span_abs_error_L0':abs(predicted_span[dim]-gt_span[dim])/L0,
                      axis+'_median_abs_error_L0':abs(pred_median[dim]-gt_median[dim])/L0}
            residuals += [abs(float(v)-number(row, k)) for k, v in values.items()]
        assert max(residuals) < 2e-12, ident
        raw_checks.append(dict(zip(['arm', 'phase_step', 'case_id'], ident),
                               scalar_values_checked=18, max_abs_difference=max(residuals),
                               raw_sha256=g['raw_sha256'], input_sha256=bank['input_sha256']))
    write_csv('raw_width_verification.csv', raw_checks)
    decomposition, curves = [], []
    for key, members in sorted(groups.items()):
        base = dict(zip(KEY, key))
        by_alpha = {float(r['alpha']):r for r in members}
        assert len(members) == 3 and set(by_alpha) == {.9, 1., 1.1}
        center = by_alpha[1.]
        p, y, L0 = (number(center, k) for k in ['X_span', 'X_GT_span', 'L0'])
        assert p > 1e-8 * L0 and y > 0
        slope = (number(by_alpha[1.1], 'X_span')-number(by_alpha[.9], 'X_span'))/(.2*p)
        local = []
        for alpha, row in sorted(by_alpha.items()):
            w, target = number(row, 'X_span'), number(row, 'X_GT_span')
            response = (w-alpha*p)/L0
            baseline = alpha*(p-y)/L0
            roundoff = (alpha*y-target)/L0
            total = (w-target)/L0
            residual = w/p-alpha
            assert abs(total-response-baseline-roundoff) < 1e-12
            assert abs(residual-number(row, 'response_residual')) < 1e-12
            assert abs(slope-number(row, 'slope')) < 1e-12
            assert abs(abs(total)-number(row, 'X_span_abs_error_L0')) < 1e-12
            assert abs(roundoff) < 1e-6
            g = geo_index[(row['arm'], row['phase_step'], row['case_id'])]
            derived = dict(base, case_id=row['case_id'], alpha=alpha, L0=L0,
                           predicted_base_width=p, GT_base_width=y, baseline_ratio=p/y,
                           actual_width=w, actual_GT_width=target,
                           response_signed_L0=response, baseline_signed_L0=baseline,
                           GT_transform_roundoff_L0=roundoff, total_signed_L0=total,
                           total_abs_L0=abs(total), response_abs_L0=abs(response),
                           baseline_abs_L0=abs(baseline), response_ratio=w/p,
                           response_residual=residual, response_abs_residual=abs(residual),
                           algebra_closure_abs_L0=abs(total-response-baseline-roundoff),
                           opposite_sign_terms=response*baseline < 0,
                           cancellation_L0=abs(response)+abs(baseline)-abs(response+baseline),
                           dominant_magnitude=('baseline' if abs(baseline)>abs(response)+EPS else
                                               'response' if abs(response)>abs(baseline)+EPS else 'tie'),
                           X_median_abs_error_L0=number(row,'X_median_abs_error_L0'),
                           Y_span_abs_error_L0=number(row,'Y_span_abs_error_L0'),
                           Z_span_abs_error_L0=number(row,'Z_span_abs_error_L0'),
                           raw_sha256=g['raw_sha256'], input_sha256=g['input_sha256'])
            derived.update({m:number(g,m) for m in QUALITY})
            decomposition.append(derived)
            local.append(derived)
        curve = dict(base, predicted_base_width=p, GT_base_width=y, L0=L0, baseline_ratio=p/y,
                     baseline_signed_L0=(p-y)/L0, baseline_abs_L0=abs(p-y)/L0,
                     slope=slope, slope_abs_error=abs(slope-1),
                     monotonic=number(by_alpha[.9],'X_span')<=p<=number(by_alpha[1.1],'X_span'),
                     base_X_median_abs_L0=number(center,'X_median_abs_error_L0'),
                     mean_X_median_abs_L0=mean(r['X_median_abs_error_L0'] for r in local),
                     base_Y_width_abs_L0=number(center,'Y_span_abs_error_L0'),
                     base_Z_width_abs_L0=number(center,'Z_span_abs_error_L0'),
                     mean_width_abs_L0=mean(r['total_abs_L0'] for r in local),
                     mean_response_abs_residual=mean(r['response_abs_residual'] for r in local),
                     edited_response_abs_residual=mean(r['response_abs_residual'] for r in local if r['alpha']!=1),
                     mean_response_abs_L0=mean(r['response_abs_L0'] for r in local),
                     mean_baseline_abs_L0=mean(r['baseline_abs_L0'] for r in local),
                     mean_cancellation_L0=mean(r['cancellation_L0'] for r in local),
                     response_ratio_09=local[0]['response_ratio'],response_ratio_11=local[2]['response_ratio'],
                     magnitude_label=('baseline_larger' if mean(r['baseline_abs_L0'] for r in local) >
                                                       mean(r['response_abs_L0'] for r in local)+EPS else 'response_larger_or_tie'))
        for m in QUALITY:
            curve['mean_'+m]=mean(r[m] for r in local)
            curve['base_'+m]=local[1][m]
            curve['edited_minus_base_'+m]=mean(r[m] for r in local if r['alpha']!=1)-local[1][m]
        curves.append(curve)
    write_csv('width_decomposition_per_case.csv', decomposition)
    write_csv('width_per_curve.csv', curves)
    metrics=['baseline_ratio','baseline_signed_L0','baseline_abs_L0','slope','slope_abs_error',
             'base_X_median_abs_L0','mean_X_median_abs_L0','base_Y_width_abs_L0','base_Z_width_abs_L0',
             'mean_width_abs_L0','mean_response_abs_residual','mean_response_abs_L0',
             'mean_baseline_abs_L0','mean_cancellation_L0','response_ratio_09','response_ratio_11']
    summary=[]
    for key,members in sorted(group(curves,['arm','phase_step','split','noise_index']).items()):
        row=dict(zip(['arm','phase_step','split','noise_index'],key),parents=len(members),curves=len(members))
        assert len(members)==4
        for m in metrics:
            vals=[r[m] for r in members]
            row[m+'_mean']=mean(vals); row[m+'_median']=float(np.median(vals))
            row[m+'_min']=min(vals); row[m+'_max']=max(vals)
        row['baseline_larger_count']=sum(r['magnitude_label']=='baseline_larger' for r in members)
        row['monotonic_count']=sum(r['monotonic'] for r in members)
        summary.append(row)
    write_csv('width_noise_summary.csv',summary)
    paired=[]
    paired_metrics=metrics+['mean_'+m for m in QUALITY]
    for key,members in sorted(group([r for r in curves if r['noise_index']=='0'],['arm','split','uid']).items()):
        by_phase={r['phase_step']:r for r in members}; assert set(by_phase)=={'0','3000'}
        before,after=by_phase['0'],by_phase['3000']
        row=dict(zip(['arm','split','uid'],key),noise_index=0)
        for metric in paired_metrics:
            row[metric+'_before']=before[metric]; row[metric+'_after']=after[metric]
            row[metric+'_delta']=after[metric]-before[metric]
        response_better=after['mean_response_abs_residual']<before['mean_response_abs_residual']-EPS
        row['response_improves']=response_better
        row['baseline_worsens']=after['baseline_abs_L0']>before['baseline_abs_L0']+EPS
        row['absolute_width_worsens']=after['mean_width_abs_L0']>before['mean_width_abs_L0']+EPS
        row['response_improves_with_baseline_and_absolute_worse']=response_better and row['baseline_worsens'] and row['absolute_width_worsens']
        row['FF_worsens']=after['mean_FF_proper_affected_free_fraction']>before['mean_FF_proper_affected_free_fraction']+EPS
        row['coverage_worsens']=after['mean_fixed_R12_coverage_within_1pct_fraction']<before['mean_fixed_R12_coverage_within_1pct_fraction']-EPS
        row['response_improves_with_FF_worse']=response_better and row['FF_worsens']
        paired.append(row)
    assert len(paired)==24
    write_csv('width_start_end_per_parent.csv',paired)
    pair_summary=[]
    for key,members in sorted(group(paired,['arm','split']).items()):
        row=dict(zip(['arm','split'],key),parents=len(members))
        for metric in paired_metrics:
            for suffix in ['before','after','delta']: row[metric+'_'+suffix]=mean(r[metric+'_'+suffix] for r in members)
            row[metric+'_decreased_parents']=sum(r[metric+'_delta'] < -EPS for r in members)
        for flag in ['response_improves','baseline_worsens','absolute_width_worsens',
                     'response_improves_with_baseline_and_absolute_worse','FF_worsens',
                     'coverage_worsens','response_improves_with_FF_worse']:
            row[flag+'_parents']=sum(r[flag] for r in members)
        pair_summary.append(row)
    write_csv('width_start_end_summary.csv',pair_summary)
    noise_pairs=[]
    for key,members in sorted(group([r for r in curves if r['phase_step']=='3000'],['arm','split','uid']).items()):
        by_noise={r['noise_index']:r for r in members}; assert set(by_noise)=={'0','1'}
        a,b=by_noise['0'],by_noise['1']; row=dict(zip(['arm','split','uid'],key))
        for m in metrics:
            row[m+'_noise0']=a[m]; row[m+'_noise1']=b[m]; row[m+'_noise1_minus_0']=b[m]-a[m]
        row['slope_straddles_one']=(a['slope']-1)*(b['slope']-1)<0
        row['slope_sign_changes']=a['slope']*b['slope']<0
        noise_pairs.append(row)
    write_csv('noise_pairs_per_parent.csv',noise_pairs)
    analytic=[]
    curve_index={tuple(r[k] for k in KEY):r for r in curves}
    for key,members in sorted(group(analytics,KEY).items()):
        row=dict(zip(KEY,key)); curve=curve_index[key]
        by_candidate=group(members,['candidate']); assert set(x[0] for x in by_candidate)=={'No-response','Post-scale'}
        for (candidate,), entries in by_candidate.items():
            assert len(entries)==3
            row[candidate+'_mean_abs_error_L0']=mean(number(r,'X_span_abs_error_L0') for r in entries)
            row[candidate+'_slope']=number(entries[0],'slope')
            row[candidate+'_mean_response_abs_residual']=mean(number(r,'response_abs_residual') for r in entries)
        row['No_response_minus_Post_scale_abs_error_L0']=row['No-response_mean_abs_error_L0']-row['Post-scale_mean_abs_error_L0']
        row['equal_within_1e-7_L0']=abs(row['No_response_minus_Post_scale_abs_error_L0'])<=1e-7
        ratio=curve['baseline_ratio']; row['baseline_ratio']=ratio
        row['GT_width_interval_region']='below_0.9' if ratio<.9 else 'above_1.1' if ratio>1.1 else 'inside_0.9_1.1'
        row['ideal_affine_mean_error_difference_L0']=mean(abs(curve['predicted_base_width']-a*curve['GT_base_width'])-a*abs(curve['predicted_base_width']-curve['GT_base_width']) for a in [.9,1.,1.1])/curve['L0']
        row['ideal_vs_actual_difference_residual_L0']=row['No_response_minus_Post_scale_abs_error_L0']-row['ideal_affine_mean_error_difference_L0']
        analytic.append(row)
    write_csv('analytic_equal_error_diagnostic.csv',analytic)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'T1':'#3875ad','Geo':'#d07817','Full':'#478b54'}
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained',sharey='row')
    for i,split in enumerate(['train','validation']):
        for j,arm in enumerate(['T1','Geo','Full']):
            ax=axes[i,j]
            members=[r for r in noise_pairs if r['arm']==arm and r['split']==split]
            for n,row in enumerate(members):
                ax.plot([0,1],[row['slope_noise0'],row['slope_noise1']],marker='o',label=row['uid'][:8])
            ax.axhline(1,color='black',linestyle='--',linewidth=.8); ax.axhline(0,color='gray',linewidth=.6)
            ax.set_xticks([0,1],['noise bank 0','noise bank 1']); ax.set_title(f'{arm} / {split} (4 parents)')
            ax.set_ylabel('width response slope'); ax.legend(fontsize=7)
    fig.suptitle('Endpoint response: all registered parent/noise pairs; ideal slope = 1')
    fig.savefig(OUT/'noise_response.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,split in zip(axes,['train','validation']):
        for arm in ['T1','Geo','Full']:
            members=[r for r in paired if r['split']==split and r['arm']==arm]
            ax.scatter([r['baseline_abs_L0_delta'] for r in members],[r['mean_response_abs_residual_delta'] for r in members],label=arm,color=colors[arm])
        ax.axhline(0,color='gray',linewidth=.6); ax.axvline(0,color='gray',linewidth=.6)
        ax.set_xlabel('Change in baseline absolute width error / L0 (lower better)')
        ax.set_ylabel('Change in mean absolute response residual (lower better)')
        ax.set_title(f'{split}: first-noise start to endpoint'); ax.legend()
    fig.savefig(OUT/'baseline_vs_response_changes.png',dpi=150); plt.close(fig)
    result={'status':'PASS','rows':216,'curves':72,'endpoint_curves':48,'paired_start_end_parents':24,
            'raw_files_verified':216,'unique_inputs_verified':len(arrays),'raw_scalar_values_checked':216*18,
            'max_raw_scalar_abs_difference':max(r['max_abs_difference'] for r in raw_checks),
            'max_algebra_closure_abs_L0':max(r['algebra_closure_abs_L0'] for r in decomposition),
            'max_GT_affine_roundoff_L0':max(abs(r['GT_transform_roundoff_L0']) for r in decomposition),
            'model_forwards':0,'optimizer_updates':0,'new_generations':0,'OT_calls':0,'geometry_scorer_calls':0,
            'existing_raw_width_recomputations':216,'source_files_sha256':checked,'seconds':time.time()-started,
            'counting_note':'Slopes are counted once per arm/phase/parent/noise curve, never once per alpha row.',
            'classification_note':'Magnitude labels describe the additive identity, not causal attribution. No calibrated success threshold invented.'}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_files_sha256'},indent=2))


if __name__=='__main__':
    run()
