"""Read-only analysis of saved motif-guidance evidence; no model/scorer/renderer.

Run after evaluate.collect, separately for calibration and final. Each command
writes a new immutable analysis receipt; it never updates the frozen evaluator.
I retains its original query denominator and its pre-registered parent domain.
An available-case mean is explicitly separate from a complete-domain mean.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

from . import prepare as P
from . import evaluate as E

VERSION = 'motif_guidance_saved_evidence_analysis_v1'
PRIMARY = ('CF_illegal_affected_free_fraction', 'FF_illegal_affected_free_fraction',
           'ConflictCov_C', 'CF_illegal_pair_count', 'FF_illegal_pair_count',
           'actual_free_faces', 'free_total_surface_area', 'deleted_free_faces',
           'free_minimum_angle_lt5_fraction', 'free_near_degenerate_fraction',
           'C_F_valid_edge_count', 'C_F_valid_edge_total_length',
           'hinge_diag_isolated_1face_edge_components')
GATES = {'CF_illegal_affected_free_fraction': (.02, .05),
         'FF_illegal_affected_free_fraction': (.02, .10), 'ConflictCov_C': (.02, .10)}


def number(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def mean(values):
    return sum(values)/len(values) if values else None


def parse(x):
    if x == '': return None
    if x == 'True': return True
    if x == 'False': return False
    try: return json.loads(x)
    except (ValueError, TypeError): return x


def csvread(path):
    with P.resolve(path).open(encoding='utf-8', newline='') as handle:
        return [{k: parse(v) for k, v in r.items()} for r in csv.DictReader(handle)]


def book(paths):
    return {str(P.resolve(p)): P.sha(p) for p in paths}


def verified(path, protected):
    path = P.resolve(path)
    result = P.read(path)
    protected[str(path)] = P.sha(path)
    return result


def sealed_file(path, receipt, protected):
    path = P.resolve(path)
    digest = next((h for p, h in receipt['files'].items() if P.resolve(p) == path), None)
    if digest is None or P.sha(path) != digest:
        raise ValueError('Unsealed or changed evidence: ' + str(path))
    protected[str(path)] = digest


def begin(mode, section):
    if 'torch' in sys.modules: raise RuntimeError('Use a fresh CPU process without torch')
    if mode not in ('calibration', 'final'): raise ValueError(mode)
    folder = P.OUT/'analysis_numeric'/mode/section
    receipt = folder/'complete.json'
    if receipt.exists():
        saved = P.read(receipt)
        if saved['source_sha256'] != P.sha(__file__): raise ValueError('Completed analysis source changed')
        E._verify(saved['files']); E._verify(saved['protected'])
        return folder, saved
    return folder, None


def finish(folder, data, files, protected):
    result = dict(status='COMPLETE', version=VERSION, source_sha256=P.sha(__file__),
                  new_model_calls=0, new_scoring_calls=0, new_render_calls=0,
                  **data, files=book(files), protected=protected)
    P.write_immutable(folder/'complete.json', result)
    return result


def cohort(q):
    if q['panel_role'] == 'calibration': return 'CALIBRATION'
    if q['split'] == 'DEV': return 'CONFIRMATION_DEV'
    if q['split'] in ('NEW_PARENT_HOLDOUT', 'HIGH_HOLDOUT'): return 'CONFIRMATION_HOLDOUT'
    if q['split'] in ('CURRENT_TRAIN', 'NEW_TRAIN', 'HIGH_TRAIN', 'NEW_WIDE_TRAIN'):
        return 'CONFIRMATION_EXPOSURE_TRAIN'
    raise ValueError('Unregistered exposure split: ' + str(q['split']))


def panel(protected):
    requests = P.load_requests()
    protected.update(book([P.OUT/'registration.json', P.REQUESTS, P.TASKS,
                           P.OUT/'baseline.json', P.OUT/'inputs/metric_fields.json']))
    by = {q['request_id']: q for q in requests}
    parents = defaultdict(set)
    for q in requests: parents[cohort(q)].add(q['parent_id'])
    if {k: len(v) for k, v in parents.items()} != {
        'CALIBRATION': 4, 'CONFIRMATION_EXPOSURE_TRAIN': 12,
        'CONFIRMATION_DEV': 4, 'CONFIRMATION_HOLDOUT': 3}:
        raise ValueError('Registered 4 + 12/4/3 parent panel changed')
    original_i = {}
    for t in P.read(P.OUT/'baseline.json'):
        path = P.resolve(t['score_paths']['RAW'])/'row.json'
        row = verified(path, protected)
        original_i[t['request_id']] = row['I_status']
    for role, expected in (('calibration', 6), ('confirmation', 32)):
        if sum(original_i[rid] == 'PASS' for rid, q in by.items() if q['panel_role'] == role) != expected:
            raise ValueError('Original I support domain changed')
    return by, original_i


def scope_names(q):
    c = cohort(q)
    result = [c]
    if q['panel_role'] == 'confirmation':
        result.append('CONFIRMATION_ALL')
        if c in ('CONFIRMATION_DEV', 'CONFIRMATION_HOLDOUT'):
            result.append('CONFIRMATION_ADAPTATION_UNSEEN')
    return result


def load_metrics(mode, protected):
    done = verified(P.OUT/'analysis'/mode/'complete.json', protected)
    if done['status'] != 'COMPLETE': raise ValueError('Run evaluate.collect first')
    path = P.OUT/'analysis'/mode/'full177_per_output.csv'
    sealed_file(path, done, protected)
    rows = csvread(path)
    if len(rows) != (48 if mode == 'calibration' else 706): raise ValueError('Unexpected collected row count')
    fields = P.read(P.OUT/'inputs/metric_fields.json')
    if len(fields) != 177 or any(set(fields)-set(r) for r in rows): raise ValueError('Original177 fields missing')
    return rows, fields


def method(row, mode):
    return row['label'] if mode == 'calibration' else row['arm']


def macro_metrics(rows, fields, requests, original_i, group_keys):
    """Equal-noise then equal-parent means; no missing values become zero."""
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        if row['request_id'] in groups[key]: raise ValueError('Duplicate metric request identity')
        groups[key][row['request_id']] = row
    parent_rows, macros = [], []
    for group, indexed in sorted(groups.items(), key=lambda x: str(x[0])):
        scopes = sorted({s for rid in indexed for s in scope_names(requests[rid])})
        for scope in scopes:
            expected = [rid for rid, q in requests.items() if scope in scope_names(q)]
            for field in fields:
                domain = [rid for rid in expected if field not in E.I_FIELDS or original_i[rid] == 'PASS']
                pids = sorted({requests[rid]['parent_id'] for rid in domain})
                values = []; missing = []; available_requests = 0
                for pid in pids:
                    ids = [rid for rid in domain if requests[rid]['parent_id'] == pid]
                    if len(ids) != 2: raise ValueError('Expected both frozen noises per eligible parent')
                    observations = []
                    reasons = []
                    for rid in ids:
                        r = indexed.get(rid)
                        if r is None: reasons.append(rid+':MISSING_ROW')
                        elif field in E.I_FIELDS and r.get('I_status') != 'PASS':
                            reasons.append(rid+':I_'+str(r.get('I_status')))
                        elif not number(r.get(field)): reasons.append(rid+':METRIC_NA')
                        else: observations.append(r[field])
                    available_requests += len(observations)
                    value = mean(observations) if not reasons else None
                    out = dict(zip(group_keys, group), scope=scope, parent_id=pid, metric=field,
                               expected_requests=2, valid_requests=len(observations), value=value,
                               status='COMPLETE' if not reasons else 'UNKNOWN', missing_reasons=reasons)
                    parent_rows.append(out)
                    if value is None: missing += reasons
                    else: values.append(value)
                macros.append(dict(zip(group_keys, group), scope=scope, metric=field,
                    expected_parents=len(pids), valid_parents=len(values), expected_requests=len(domain),
                    valid_requests=available_requests, complete_domain_mean=mean(values) if not missing else None,
                    available_parent_mean=mean(values), status='COMPLETE' if not missing else 'UNKNOWN',
                    missing_reasons=missing, denominator_policy='Both old noises then equal parents; original I PASS domain; no zero fill'))
    return parent_rows, macros


def metric_pairs(rows, fields, mode):
    main = [dict(r, method=method(r, mode)) for r in rows if r['main_panel']]
    # Existing arithmetic is reused; its output retains each original side.
    result = E._pairs(main, fields)
    return [r for r in result if (r['candidate'] == 'G3' or r['candidate'].startswith('G3_L'))
            and r['reference'] in ('G1', 'G2', 'R')]


def quality_gates(pairs, macros, requests, original_i):
    results = []
    for candidate, reference, stage in sorted({(r['candidate'], r['reference'], r['stage']) for r in pairs}):
        for scope in ('CONFIRMATION_ALL', 'CONFIRMATION_ADAPTATION_UNSEEN'):
            selected = [r for r in pairs if (r['candidate'], r['reference'], r['stage']) == (candidate, reference, stage)
                        and scope in scope_names(requests[r['request_id']])]
            if not selected: continue
            for field, (mean_limit, tail_limit) in GATES.items():
                macro = next(r for r in macros if (r['candidate'], r['reference'], r['stage'], r['scope'], r['metric'])
                             == (candidate, reference, stage, scope, field))
                tails = [dict(request_id=r['request_id'], parent_id=r['parent_id'], delta=r[field]) for r in selected
                         if number(r.get(field)) and r[field] > tail_limit
                         and (field not in E.I_FIELDS or original_i[r['request_id']] == 'PASS')]
                value = macro['complete_domain_mean']
                status = 'FAIL' if tails or (number(value) and value > mean_limit) else ('PASS' if number(value) else 'UNKNOWN')
                results.append(dict(candidate=candidate, reference=reference, stage=stage, scope=scope, metric=field,
                    status=status, parent_mean_delta=value, maximum_parent_mean_increase=mean_limit,
                    maximum_request_increase=tail_limit, expected_parents=macro['expected_parents'],
                    valid_parents=macro['valid_parents'], severe_tails=tails, missing_reasons=macro['missing_reasons'],
                    interpretation='Geometric screening only; this does not establish a visual/J improvement'))
    return results


def metrics(mode='final'):
    folder, cached = begin(mode, 'metrics')
    if cached: return cached
    protected = {}; requests, original_i = panel(protected)
    rows, fields = load_metrics(mode, protected)
    for r in rows:
        q = requests[r['request_id']]
        r.update(cohort=cohort(q), original_I_status=original_i[r['request_id']], method=method(r, mode))
    main = [r for r in rows if r['main_panel']]
    parents, macros = macro_metrics(main, fields, requests, original_i, ('method', 'stage'))
    pairs = metric_pairs(rows, fields, mode)
    pp, pm = macro_metrics(pairs, fields, requests, original_i, ('candidate', 'reference', 'stage'))
    gates = quality_gates(pairs, pm, requests, original_i)
    files = [E._csv(folder/'per_output_original177.csv', rows), E._csv(folder/'per_parent_metrics.csv', parents),
             E._csv(folder/'parent_macro_metrics.csv', macros), E._csv(folder/'paired_per_output.csv', pairs),
             E._csv(folder/'paired_per_parent.csv', pp), E._csv(folder/'paired_parent_macro.csv', pm)]
    if mode=='final':
        costpath=P.OUT/'analysis/final/stage_retention_and_costs.csv'
        receipt=P.read(P.OUT/'analysis/final/complete.json');sealed_file(costpath,receipt,protected)
        costs=csvread(costpath)
        for r in costs:r['cohort']=cohort(requests[r['request_id']])
        files.append(E._csv(folder/'stage_retention_and_costs.csv',costs))
    summary = dict(version=VERSION, mode=mode, original_fields=177, stage_rows=len(rows),
        primary_parent_means=[r for r in macros if r['metric'] in PRIMARY],
        primary_paired_deltas=[r for r in pm if r['metric'] in PRIMARY], quality_gates=gates,
        visual_claims='NOT_EVALUATED: numerical metrics do not imply J or visible fragments',
        scope_note='Confirmation 12 EXPOSURE-trained + 4 historical DEV + 3 adaptation holdout; official-pretraining exposure UNKNOWN',
        I_note='Original 2048 query-weight denominator unchanged; calibration 3 / confirmation 16 original PASS parents; extra missing support is UNKNOWN')
    path = folder/'numeric_summary.json'; P.write_immutable(path, summary); files.append(path)
    return finish(folder, dict(mode=mode, stage_rows=len(rows), original_fields=177), files, protected)


def selected_tasks(mode):
    return E._calibration_tasks() if mode == 'calibration' else P.load_tasks(resolved=True)


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix+key
        if isinstance(item, dict): result.update(flatten(item, name+'__'))
        elif not isinstance(item, list): result[name] = item
    return result


def guidance(mode='final'):
    folder, cached = begin(mode, 'guidance')
    if cached: return cached
    protected = book([P.OUT/'registration.json', P.TASKS]); steps = []; outputs = []
    selected = {t['cell_id'] for t in P.selected_tasks()} if mode == 'final' else None
    for task in selected_tasks(mode):
        dest = P.resolve(task['output_dir']); receipt = verified(dest/'receipt.json', protected)
        if receipt['status'] != 'COMPLETE': raise ValueError('Guidance diagnostics require completed RAW')
        audit_path = dest/'audit.json'; sealed_file(audit_path, receipt, protected); audit = P.read(audit_path)
        ds = []
        for step in range(50):
            path = dest/f'steps/step_{step:03}.json'; sealed_file(path, receipt, protected); d = P.read(path)
            if d['step'] != step or d['arm'] != task['arm']: raise ValueError('Diagnostic identity mismatch')
            guided = task['arm'] in ('G2','G3') and step >= 5
            if d['guided'] != guided or not 0 < d['scale'] <= 1: raise ValueError('Guidance window/scale mismatch')
            if d['correction']['max_corner'] > .005*P.L*(1+2e-6): raise ValueError('Registered cap exceeded')
            if task['arm'] == 'G2' and d['correction_F']['l2'] != 0: raise ValueError('G2 free correction nonzero')
            if not guided and d['correction']['l2'] != 0: raise ValueError('Correction outside registered window')
            ds.append(d)
            steps.append(dict(cell_id=task['cell_id'], request_id=task['request_id'], panel_role=task['panel_role'],
                              cohort=cohort(task['q']), **flatten(d)))
        active = [d for d in ds if d['guided']]
        row = dict(cell_id=task['cell_id'], arm=task['arm'], label=task['label'], lambda_value=task['lambda_value'],
            request_id=task['request_id'], parent_id=task['q']['parent_id'], noise_index=task['q']['noise_index'],
            panel_role=task['panel_role'], cohort=cohort(task['q']),
            main_panel=True if selected is None else task['cell_id'] in selected,
            guidance_steps=len(active), clip_steps=sum(d['clipped'] for d in active),
            clip_fraction_guided=mean([int(d['clipped']) for d in active]),
            scale_min=min((d['scale'] for d in active), default=None), scale_mean=mean([d['scale'] for d in active]),
            correction_S_rms_sum=sum(d['correction_S']['rms_corner'] for d in ds),
            correction_F_rms_sum=sum(d['correction_F']['rms_corner'] for d in ds),
            correction_corner_max_sum=sum(d['correction']['max_corner'] for d in ds),
            correction_over_euler_l2_mean=mean([d['correction_over_euler_l2'] for d in active if number(d['correction_over_euler_l2'])]),
            endpoint_rms_step0=ds[0]['endpoint_error']['rms_corner'],
            endpoint_rms_step49=ds[-1]['endpoint_error']['rms_corner'],
            shadow_terminal_corner_rms=audit['shadow_terminal_corner_rms'],
            shadow_terminal_corner_max=audit['shadow_terminal_corner_max'],
            model_seconds=audit['model_seconds'], VJP_seconds=audit['VJP_seconds'],
            max_cuda_peak_allocated_bytes=audit['max_cuda_peak_allocated_bytes'],
            max_cuda_peak_reserved_bytes=audit['max_cuda_peak_reserved_bytes'])
        if row['clip_steps'] != audit['clipped_steps']: raise ValueError('Clipping receipt mismatch')
        outputs.append(row)
    # Numerical sums above are path-length diagnostics, not net displacement.
    diagnostic_fields = [k for k in outputs[0] if k not in ('noise_index','lambda_value','main_panel')
                         and any(number(r.get(k)) for r in outputs)]
    parents = E._aggregate(outputs, ('arm','label','lambda_value','panel_role','cohort','parent_id'), diagnostic_fields)
    macros = E._aggregate(parents, ('arm','label','lambda_value','panel_role','cohort'), diagnostic_fields)
    distances = lambda_distances(protected)
    files = [E._csv(folder/'per_step.csv', steps), E._csv(folder/'per_output.csv', outputs),
             E._csv(folder/'per_parent.csv', parents), E._csv(folder/'parent_macro.csv', macros)]
    files.append(E._csv(folder/'calibration_lambda_RAW_distances.csv', distances))
    return finish(folder, dict(mode=mode, outputs=len(outputs), steps=len(steps),
        note='Correction sums are accumulated norms, not net movement. Shadow error is not a formal CF/I or visual success metric. Allocator peaks are not full-process VRAM.',
        cap_interpretation='For identical states and gradients, when the global cap is active the positive lambda scalar cancels from the applied full correction. Trajectories and floating-point arithmetic can still differ. Saved RAW distances and clip fractions diagnose closeness; they do not establish visual equivalence or failure.'), files, protected)


def lambda_distances(protected):
    """Same frozen request, three calibration lambdas; saved coordinates only."""
    import itertools
    import numpy as np
    groups=defaultdict(list)
    for t in E._calibration_tasks():
        if t['arm']=='G3':groups[t['request_id']].append(t)
    rows=[]
    for rid, tasks in sorted(groups.items()):
        if len(tasks)!=3:raise ValueError('Expected three frozen calibration lambdas')
        for a,b in itertools.combinations(sorted(tasks,key=lambda t:t['lambda_value']),2):
            meshes=[];audits=[]
            for t in (a,b):
                dest=P.resolve(t['output_dir']);receipt=verified(dest/'receipt.json',protected)
                sealed_file(t['raw_path'],receipt,protected);sealed_file(dest/'audit.json',receipt,protected)
                meshes.append(P.arrays(t['raw_path'])['raw']);audits.append(P.read(dest/'audit.json'))
            k=int(a['q']['K']);difference=meshes[1].astype(np.float64)-meshes[0].astype(np.float64)
            if not np.array_equal(meshes[0][:k],meshes[1][:k]):raise ValueError('Fixed C differs across lambda')
            d=difference[k:];norm=np.linalg.norm(d,axis=-1)
            rows.append(dict(request_id=rid,parent_id=a['q']['parent_id'],noise_index=a['q']['noise_index'],
                lambda_a=a['lambda_value'],lambda_b=b['lambda_value'],K=k,N=int(a['q']['N']),
                free_scalar_max_absolute_difference=float(np.abs(d).max()),
                free_scalar_mean_absolute_difference=float(np.abs(d).mean()),
                free_corner_max_distance=float(norm.max()),free_corner_mean_distance=float(norm.mean()),
                free_corner_rms_distance=float(np.sqrt(np.mean(norm**2))),
                free_corner_max_distance_over_L=float(norm.max()/P.L),
                free_byte_identical=bool(np.array_equal(meshes[0][k:],meshes[1][k:])),
                clipped_guided_steps_a=audits[0]['clipped_steps'],clipped_guided_steps_b=audits[1]['clipped_steps'],
                guided_steps_each=45,source='Saved RAW canonical coordinates; no generation or rescoring'))
    return rows


def jsonlines(path):
    return [json.loads(line) for line in P.resolve(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def calls(mode='final'):
    """Audit saved call journals and all receipt-file hashes; no network/model load."""
    folder, cached = begin(mode, 'calls')
    if cached: return cached
    protected = book([P.OUT/'registration.json', P.TASKS]); rows = []; errors = []
    scan_counts=Counter(RAW_full_pair_scans=0,BASE_cache_full_pair_scans=0,
                        FINAL_recomputed_full_pair_scans=0,candidate_full_geometry_scans=0,
                        HOLE_new_face_pair_scans=0,HOLE_full_old_pair_rescans=0)
    tasks = selected_tasks(mode)
    totals = Counter(vector_field=0, cfg_sample_equivalents=0, input_VJP=0, D0=0, GEO=0,
                     RAW_scores=0, BASE_scores=0, FINAL_scores=0, training_updates=0)
    selected = {t['cell_id'] for t in P.selected_tasks()} if mode == 'final' else set()
    for t in tasks:
        dest=P.resolve(t['output_dir']); out=dict(cell_id=t['cell_id'],arm=t['arm'],status='PASS',errors=[])
        rp=dest/'receipt.json'
        if not rp.is_file():
            out['errors'].append('MISSING_RAW_RECEIPT')
            journal=jsonlines(dest/'calls.jsonl') if (dest/'calls.jsonl').exists() else []
        else:
            receipt=verified(rp,protected)
            try: E._verify(receipt['files'])
            except (ValueError,OSError) as exc: out['errors'].append(str(exc))
            protected.update(receipt['files'])
            journal=jsonlines(dest/'calls.jsonl')
            expected=[item for k in range(50) for item in [('vector_field',k)]+([('input_VJP',k)] if t['arm'] in ('G2','G3') and k>=5 else [])]
            for event in ('CALL_BEGIN','CALL_END'):
                if [(r['kind'],r['step']) for r in journal if r['event']==event] != expected: out['errors'].append(event+'_SEQUENCE')
            if [r['step'] for r in journal if r['event']=='STEP_COMMIT'] != list(range(50)): out['errors'].append('STEP_COMMIT_SEQUENCE')
            if sum(r['event']=='INITIAL_STATE_COMMIT' for r in journal)!=1: out['errors'].append('INITIAL_STATE_COUNT')
            if len(list((dest/'states').glob('state_*.npz')))!=51:out['errors'].append('STATE_COUNT')
            if len(list((dest/'steps').glob('step_*.json')))!=50:out['errors'].append('STEP_DIAGNOSTIC_COUNT')
            for commit in (r for r in journal if r['event']=='STEP_COMMIT'):
                step=commit['step']
                if (commit.get('completed_updates'),commit.get('cumulative_generator_forwards'),commit.get('cumulative_input_VJP'))!=(step+1,step+1,max(0,step-4) if t['arm'] in ('G2','G3') else 0):
                    out['errors'].append('STEP_CUMULATIVE_COUNTS')
            for key,value in (('generator_forwards',50),('cfg_network_sample_equivalents',100),
                              ('input_VJP',45 if t['arm'] in ('G2','G3') else 0),('D0_forwards',0),('training_updates',0)):
                if receipt.get(key)!=value: out['errors'].append('RECEIPT_'+key)
            if receipt.get('noise_sha256') != t['q']['noise_sha256'] or receipt.get('shadow_noise_sha256') != t['q']['shadow_noise_sha256']:
                out['errors'].append('NOISE_IDENTITY')
        for kind in ('vector_field','input_VJP'):
            out[kind+'_begun']=sum(r['event']=='CALL_BEGIN' and r['kind']==kind for r in journal)
            out[kind+'_returned']=sum(r['event']=='CALL_END' and r['kind']==kind for r in journal)
            totals[kind]+=out[kind+'_returned']
        totals['cfg_sample_equivalents']+=2*out['vector_field_returned']
        out['committed_steps']=sum(r['event']=='STEP_COMMIT' for r in journal)
        out['failure_events']=[r['event'] for r in journal if 'FAILURE' in r['event']]
        if 'SCIENTIFIC_FAILURE' in out['failure_events']:out['errors'].append('SCIENTIFIC_FAILURE_REQUIRES_EXPLICIT_ACCOUNTING')
        score=P.OUT/'raw_scores'/t['cell_id']/'audit_receipt.json'
        if score.exists():
            sr=verified(score,protected); E._verify(sr['files'])
            protected.update(sr['files'])
            if (sr.get('status'),sr.get('full_quality_scans'),sr.get('model_calls'),sr.get('D0_calls')) != ('COMPLETE',1,0,0):
                out['errors'].append('RAW_SCORE_RECEIPT')
            totals['RAW_scores']+=1
            scan_counts['RAW_full_pair_scans']+=sr.get('full_quality_scans',0)
        else: out['errors'].append('MISSING_RAW_SCORE')
        dp=P.OUT/'post'/t['cell_id']/'receipt.json'
        dc=jsonlines(dp.parent/'calls.jsonl') if (dp.parent/'calls.jsonl').exists() else []
        out['D0_begun']=sum(r['event']=='D0_FORWARD_STARTED' for r in dc)
        out['D0_returned']=sum(r['event']=='D0_FORWARD_RETURNED' for r in dc)
        totals['D0']+=out['D0_returned']
        if t['cell_id'] in selected:
            if not dp.exists(): out['errors'].append('MISSING_D0_GEO')
            else:
                dr=verified(dp,protected); E._verify(dr['files'])
                protected.update(dr['files'])
                if Counter(r['event'] for r in dc)!=Counter(D0_FORWARD_STARTED=1,D0_FORWARD_RETURNED=1): out['errors'].append('D0_JOURNAL')
                if dr.get('D0_forwards')!=1 or dr.get('GEO_calls')!=1: out['errors'].append('D0_GEO_RECEIPT')
                totals['GEO']+=dr.get('GEO_calls',0)
            cp=P.OUT/'cpu'/t['cell_id']/'complete.json'
            if not cp.exists(): out['errors'].append('MISSING_CPU_CHAIN')
            else:
                cr=verified(cp,protected); E._verify(cr['files'])
                protected.update(cr['files'])
                for key,value in (('expanded_base_scores',1),('final_scores',1),('model_calls',0),('D0_calls',0),('GPU_calls',0),('new_noise',0),('new_queries',0)):
                    if cr.get(key)!=value: out['errors'].append('CPU_'+key)
                totals['BASE_scores']+=cr.get('expanded_base_scores',0);totals['FINAL_scores']+=cr.get('final_scores',0)
                cache=verified(P.OUT/'base_pair_caches'/t['cell_id']/'complete.json',protected)
                if cache.get('status')!='COMPLETE' or cache.get('full_pair_scans')!=1:out['errors'].append('BASE_PAIR_CACHE')
                scan_counts['BASE_cache_full_pair_scans']+=cache['full_pair_scans']
                for stage in ('BASE_POST','CLEAN_FINAL_POST'):
                    sc=verified(cp.parent/stage/'scores/score.json',protected)
                    out[stage+'_pair_origin']=sc.get('pair_origin')
                    if sc.get('pair_origin') not in ('RECOMPUTED_FULL_ORIGINAL_DETECTOR','INHERITED_COMPLETE_PAIR_CACHE_WITH_CURRENT_FACE_COUNT_VERIFIED'):
                        out['errors'].append('UNKNOWN_PAIR_ORIGIN_'+stage)
                    if stage=='CLEAN_FINAL_POST' and sc.get('pair_origin')=='RECOMPUTED_FULL_ORIGINAL_DETECTOR':
                        scan_counts['FINAL_recomputed_full_pair_scans']+=1
                for stage in ('HINGE_OPP015','ALPHA'):
                    sc=verified(cp.parent/stage/'summary.json',protected)
                    scan_counts['candidate_full_geometry_scans']+=sc.get('candidate_full_geometry_scans',0)
                sc=verified(cp.parent/'CLEAN_FINAL_POST/summary.json',protected)
                scan_counts['HOLE_new_face_pair_scans']+=sc.get('new_face_pair_scans',0)
                scan_counts['HOLE_full_old_pair_rescans']+=sc.get('full_old_pair_rescans',0)
        elif dp.exists() or dc or any((dp.parent/name).exists() for name in ('attempt.json','proposal.npz','proposal_receipt.json')):
            out['errors'].append('UNAUTHORIZED_D0_AUXILIARY_OR_CALIBRATION')
        if out['errors']: out['status']='FAIL';errors.append(dict(cell_id=t['cell_id'],errors=out['errors']))
        rows.append(out)
    expected=dict(vector_field=2000 if mode=='calibration' else 10000,
                  cfg_sample_equivalents=4000 if mode=='calibration' else 20000,
                  input_VJP=1080 if mode=='calibration' else 4860,
                  D0=0 if mode=='calibration' else 184,GEO=0 if mode=='calibration' else 184,
                  RAW_scores=40 if mode=='calibration' else 200,BASE_scores=0 if mode=='calibration' else 184,
                  FINAL_scores=0 if mode=='calibration' else 184,training_updates=0)
    if dict(totals)!=expected: errors.append(dict(error='TOTALS_DIFFER',expected=expected,actual=dict(totals)))
    incidents=[]
    for pattern in ('workers/*/failure.json','d0_workers/*/failure.json','engineering_incidents/*/receipt.json'):
        for path in sorted(P.OUT.glob(pattern)):
            incidents.append(dict(path=str(path),record=verified(path,protected)))
    known={P.resolve(t['output_dir']) for t in P.load_tasks()}
    unregistered=[str(p.parent) for p in (P.OUT/'raw').glob('*/calls.jsonl') if p.parent not in known]
    if unregistered:errors.append(dict(error='UNREGISTERED_CALL_JOURNALS',paths=unregistered))
    known_post={P.OUT/'post'/t['cell_id'] for t in P.load_tasks()}
    unregistered_post=[str(p.parent) for p in (P.OUT/'post').glob('*/calls.jsonl') if p.parent not in known_post]
    if unregistered_post:errors.append(dict(error='UNREGISTERED_D0_JOURNALS',paths=unregistered_post))
    barriers=[]
    if mode=='final':
        for path in sorted((P.OUT/'d0_workers').glob('*/all_RAW_barrier.json')):
            barrier=verified(path,protected);E._verify(barrier['files']);protected.update(barrier['files'])
            if barrier.get('outputs')!=200:errors.append(dict(error='D0_RAW_BARRIER',path=str(path)))
            protected_paths={P.resolve(p) for p in barrier['files']}
            if any(P.resolve(t[key]) not in protected_paths for t in tasks for key in ('raw_path','raw_receipt_path')):
                errors.append(dict(error='D0_BARRIER_MISSING_REGISTERED_RAW',path=str(path)))
            barriers.append(str(path))
        if not barriers:errors.append(dict(error='MISSING_D0_RAW_BARRIER'))
    files=[E._csv(folder/'per_output_calls.csv',rows)]
    audit=dict(status='PASS' if not errors else 'FAIL',mode=mode,expected=expected,actual=dict(totals),errors=errors,
               begun_calls={k:sum(r.get(k+'_begun',0) for r in rows) for k in ('vector_field','input_VJP','D0')},
               failed_call_attempts_included=True, reused_R_new_calls=0, frozen_free_noise_arrays_created=0,
               actual_pair_scan_counts=dict(scan_counts),historical_incidents=incidents,RAW_barrier_receipts=barriers,
               note='Returned calls and begun calls are separate; failure events remain disclosed even after authorized CPU-only recovery. 568 is new scored stage rows, not 568 full pair scans; inherited caches and HOLE new-face scans are separate.')
    path=folder/'call_audit.json';P.write_immutable(path,audit);files.append(path)
    return finish(folder,dict(mode=mode,audit_status=audit['status'],expected=expected,actual=dict(totals)),files,protected)


def visual(review_paths, mode='final'):
    """Summarize only supplied, image-authenticated actual reviews; never invent labels."""
    folder,cached=begin(mode,'visual')
    if cached:return cached
    protected={}; requests,original_i=panel(protected); rows,fields=load_metrics(mode,protected)
    available={(r['cell_id'],r['stage']):r for r in rows}
    tasks={t['cell_id']:t for t in P.load_tasks(resolved=(P.OUT/'lock.json').exists())+P.read(P.OUT/'baseline.json')}
    mapping_path=P.OUT/'calibration/hidden_method_mapping.json'
    mapping={(r['request_id'],r['display_label']):r['cell_id'] for r in P.read(mapping_path)['records']} if mapping_path.exists() else {}
    if mapping:protected.update(book([mapping_path]))
    reviews={}
    for path in review_paths:
        payload=verified(path,protected);records=payload['records'] if isinstance(payload,dict) else payload
        for source in records:
            r=E._normalize_review(source);cell=r.get('cell_id',mapping.get((r.get('request_id'),r.get('display_label'))))
            key=(cell,r['stage'])
            if key not in available:raise ValueError('Review outside this analysis panel: '+str(key))
            image=P.resolve(r['source_image'])
            if r.get('actually_viewed') is not True or r.get('reviewer')!='AI_PROXY' or P.sha(image)!=r['source_image_sha256']:
                raise ValueError('Actual image review evidence missing or changed')
            sealed=P.read(image.parent/'comparison.json')
            if sealed.get('image_sha256')!=r['source_image_sha256']:raise ValueError('Unsealed review image')
            identity=sealed['identity'];task=tasks[cell]
            if identity.get('request_id')!=task['request_id'] or identity.get('stage')!=r['stage']:
                raise ValueError('Review image request/stage differs')
            expected_mesh=P.sha(E._spec(task,r['stage'])['mesh'])
            if not any(c.get('mesh_sha256')==expected_mesh for c in identity['columns']):
                raise ValueError('Reviewed image does not contain the claimed output')
            if identity['mode'].startswith('calibration') and r.get('method_labels_hidden') is not True:
                raise ValueError('Calibration requires recorded hidden method labels')
            if not str(r.get('observation',r.get('note',''))).strip():raise ValueError('Review has no actual observation')
            if key in reviews:
                if any(reviews[key][f]!=r[f] for f in ('body','adoption','support','fragment_evidence','folding')):
                    raise ValueError('Conflicting actual reviews; preserve calibration labels')
                continue
            protected.update(book([image,image.parent/'comparison.json']))
            reviews[key]=dict(r,cell_id=cell,J=int(r['body']=='REASONABLE' and r['adoption']=='PLAUSIBLE' and r['support']=='PRESENT'))
    out=[];paired=[]
    for key,r in available.items():
        if r['stage']=='BASE_POST':continue
        review=reviews.get(key)
        out.append(dict(cell_id=key[0],stage=key[1],request_id=r['request_id'],parent_id=r['parent_id'],
            arm=r['arm'],label=r['label'],panel_role=r['panel_role'],cohort=cohort(requests[r['request_id']]),
            main_panel=r['main_panel'],method=method(r,mode),
            review_status='ACTUALLY_VIEWED' if review else 'NOT_REVIEWED',
            J=review['J'] if review else None,
            **{f:review[f] if review else None for f in ('body','adoption','support','fragment_evidence','folding')}))
    index={(method(r,mode),r['request_id'],r['stage']):r for r in out if r['main_panel']}
    for (m,rid,stage),r in index.items():
        if not (m=='G3' or m.startswith('G3_L')):continue
        for reference in ('G1','G2','R'):
            b=index.get((reference,rid,stage))
            if b is None:continue
            no_axes=(('body',('IMPAIRED','UNREASONABLE')),('adoption',('IMPLAUSIBLE',)),('support',('MISSING','IMPAIRED')))
            good=r['J']==1;strict_no=any(b[f] in values for f,values in no_axes)
            candidate_no=any(r[f] in values for f,values in no_axes)
            seen=r['review_status']==b['review_status']=='ACTUALLY_VIEWED'
            paired.append(dict(candidate=m,reference=reference,request_id=rid,parent_id=r['parent_id'],stage=stage,
                panel_role=r['panel_role'],cohort=r['cohort'],review_status='COMPLETE' if seen else 'UNKNOWN',
                candidate_J=r['J'],reference_J=b['J'],J_delta=r['J']-b['J'] if seen else None,
                strict_new=bool(good and strict_no) if seen else None,
                descriptive_uncertain_new=bool(good and b['J']==0 and not strict_no) if seen else None,
                retained=bool(good and b['J']==1) if seen else None,
                lost=bool(b['J']==1 and r['J']==0) if seen else None,
                explicit_task_loss=bool(b['J']==1 and candidate_no) if seen else None,
                uncertain_loss=bool(b['J']==1 and r['J']==0 and not candidate_no) if seen else None))
    files=[E._csv(folder/'actual_review_per_output.csv',out),E._csv(folder/'actual_review_paired.csv',paired)]
    pp,pm=macro_metrics(paired,('J_delta','strict_new','descriptive_uncertain_new','retained','lost','explicit_task_loss','uncertain_loss'),
                        requests,original_i,('candidate','reference','stage'))
    files.extend((E._csv(folder/'actual_review_paired_per_parent.csv',pp),
                  E._csv(folder/'actual_review_paired_parent_macro.csv',pm)))
    return finish(folder,dict(mode=mode,reviewed_cells=len(reviews),expected_non_BASE_cells=len(out),
        pending_cells=sum(r['review_status']=='NOT_REVIEWED' for r in out),ratings_inferred=0,
        verdict='No automatic final promotion; report must combine real reviews and fixed-domain geometric costs'),files,protected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('metrics','guidance','calls','visual','all'))
    parser.add_argument('--mode',choices=('calibration','final'),default='final')
    parser.add_argument('--reviews',nargs='+')
    args=parser.parse_args()
    if args.action=='all':result={name:fn(args.mode) for name,fn in (('metrics',metrics),('guidance',guidance),('calls',calls))}
    elif args.action=='visual':
        if not args.reviews:parser.error('visual requires explicit actual --reviews JSON paths')
        result=visual(args.reviews,args.mode)
    else:result=globals()[args.action](args.mode)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','protected')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
