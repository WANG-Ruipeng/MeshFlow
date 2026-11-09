"""Read completed receipts and logs only; never import a model or invoke OT.

preflight: independently audit the closed finite GPU preflight.
final: aggregate closed formal attempts; refuse a running/intermediate arm.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT = Path('/mnt/e/MeshFlow/experiments/r2_user_data_scaling_32_67_v1')
ARMS = ('USER_SMALL', 'USER_EXPANDED_67')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def lines(path):
    if not Path(path).exists():
        return
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def output(path):
    p = Path(path).absolute()
    if any(q.is_symlink() for q in (p, *p.parents)) or not p.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Independent experiment output only; symlinks forbidden')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path, value):
    p = output(path)
    if p.exists():
        raise FileExistsError('Preserve prior analysis; use a new --name: ' + str(p))
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')


def save_csv(path, rows):
    p = output(path)
    if p.exists():
        raise FileExistsError(p)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def scan_calls(path):
    pending = {}; seen = set(); attempted = collections.Counter(); done = collections.Counter()
    failed = collections.Counter(); successful_updates = []; failed_rows = []; contexts = []
    for row in lines(path):
        key = row['call_id']; kind = row['kind']; event = row['event']
        if event == 'CALL_BEGIN':
            if key in seen:
                raise ValueError('Duplicate reserved call: ' + str(path))
            seen.add(key); pending[key] = row; attempted[kind] += 1
        elif event in ('CALL_DONE', 'CALL_FAILED'):
            begin = pending.pop(key)
            if begin['kind'] != kind:
                raise ValueError('Call kind changed')
            if event == 'CALL_DONE':
                done[kind] += 1
                if kind == 'optimizer_update':
                    successful_updates.append(int(row['context']['step']))
                if kind == 'diagnostic_forward':
                    contexts.append(row['context'])
            else:
                failed[kind] += 1; failed_rows.append(row)
        else:
            raise ValueError('Unknown call event')
    return dict(attempted=dict(attempted), successful=dict(done), failed=dict(failed),
        pending_call_ids=sorted(pending), completed_updates=successful_updates,
        completed_diagnostic_contexts=contexts, failed_rows=failed_rows)


def audit_preflight():
    report = read(OUT/'preflight/report.json'); init = read(OUT/'initialization/audit.json')
    initial = read(OUT/'initialization/common_step0000.json'); lock = read(OUT/'training_budget_lock.json')
    plan = read(OUT/'plans/preflight/USER_EXPANDED_67.json'); steps = list(lines(OUT/'preflight/steps.jsonl'))
    calls = scan_calls(OUT/'preflight/calls.jsonl'); fixtures = read(OUT/'diagnostics/fixtures.json')
    expected = dict(optimizer_update=16, preflight_parity_forward=32, train_backward=128, train_forward=128)
    assert report['status'] == 'PASS' and calls['attempted'] == calls['successful'] == expected
    assert not calls['failed'] and not calls['pending_call_ids'] and calls['completed_updates'] == list(range(1,17))
    assert report['calls']['attempted'] == expected and report['calls']['successful'] == expected
    assert report['discard_preflight_state'] and not report['formal_state_reused']
    assert init['selected_state'] == 'ema' and init['fresh_AdamW'] and init['new_optimizer_step'] == 0
    assert init['official_file_sha256']=='bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d'
    assert init['official_backbone_sha256']=='b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222'
    assert init['role_nonzero'] == init['geo_exit_nonzero'] == 0
    assert init['all_parameters_FP32'] and init['all_parameters_trainable']
    assert not init['teacher_loaded'] and init['no_auxiliary_head_constructed']
    assert init['generator_state_sha256'] == initial['generator_state_sha256'] == report['lineage']['generator_initial_sha256']
    assert initial['sha256'] == report['lineage']['common_initialization_sha256'] == lock['shared_initialization_sha256']
    assert report['lineage']['fresh_AdamW'] and report['lineage']['new_optimizer_step'] == 0
    assert report['manifest_sha256'] == fixtures['manifest_sha256'] == sha(OUT/'data/train_manifest.json')
    assert len(report['parity']) == 16 and {r['fixture'] for r in report['parity']} == set(range(16))
    assert all(r['exact'] for r in report['parity'])
    assert report['changed_parameter_groups'] == {'Geo':True, 'backbone':True, 'role':True}
    assert [r['step'] for r in steps] == list(range(1,17))
    denominator_rows = []
    for row in steps:
        samples = plan['records'][(row['step']-1)*8:row['step']*8]
        denominator = sum(9*(s['N']-s['K']) for s in samples)
        assert len(samples) == len(row['sample_losses']) == 8 and len({s['uid'] for s in samples}) == 8
        assert sum(s['alpha'] == 1. for s in samples) == 4
        assert row['free_coordinate_denominator'] == denominator and all(s['batch_free_denominator'] == denominator for s in row['sample_losses'])
        assert row['generator_adamw_step'] == row['step'] and row['head_adamw_step'] == 0
        assert not row['alignment_active'] and row['aligned_samples'] == 0
        assert all(math.isfinite(row[k]) for k in ('FM','total_loss','generator_gradient_norm','total_step_seconds'))
        assert math.isclose(sum(s['FM_contribution'] for s in row['sample_losses']), row['FM'], rel_tol=1e-12)
        denominator_rows.append(dict(step=row['step'],free_scalar_denominator=denominator,FM=row['FM'],finite=True))
    hashes = {}
    for i in range(16):
        receipt = read(OUT/f'diagnostics/fixture_{i:02d}.json')
        assert receipt['plan_sha256'] == sha(OUT/'diagnostics/fixtures.json')
        assert sha(receipt['path']) == receipt['sha256']
        hashes[str(i)] = receipt['sha256']
    assert lock['preflight_sha256'] == sha(OUT/'preflight/report.json') and lock['status'] == 'LOCKED'
    value = dict(status='PASS_CLOSED_PREFLIGHT_RECEIPTS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Independent read-only receipt/log audit; no model calls, OT or checkpoint rehash/load',
        common_initialization_sha256=initial['sha256'],initial_generator_state_sha256=initial['generator_state_sha256'],
        fresh_AdamW=True,initial_optimizer_step=0,selected_official_state='ema',exact_core_fixture_parity=16,
        train_forward=128,parity_forward=32,total_forward=160,backward=128,discarded_optimizer_updates=16,
        formal_training_updates_in_this_audit=0,all_three_parameter_groups_changed=True,denominator_checks=denominator_rows,
        fixture_npz_hashes=hashes,report_sha256=sha(OUT/'preflight/report.json'),budget_updates_per_arm=lock['updates_per_arm'],
        timing=report['timing'],resources=report['resources'],
        caveat='Process VRAM is UNAVAILABLE; device used peak includes background processes. Conditioned-vs-unconditional equality is not claimed.')
    return value


def counts_ot():
    result = {}
    for phase in ('preflight','diagnostic','formal'):
        complete = reserved = 0; incomplete = []
        for path in sorted((OUT/'ot'/phase).glob('*/*.attempt.json')):
            ident = read(path); reserved += int(ident['sample_count'])
            receipt = path.with_name(path.name.replace('.attempt.json','.json'))
            if receipt.exists():
                r = read(receipt); assert r['status'] == 'PASS'; complete += int(r['actual_OT'])
            else:
                incomplete.append(str(path))
        result[phase] = dict(reserved_slots=reserved,completed_calls_receipted=complete,incomplete_batches=incomplete,
            incomplete_call_count='UNKNOWN' if incomplete else 0)
    return result


def aggregate_final(folder, allow_partial=False, plot=False):
    visual_path = OUT/'evaluation/gallery/output_review_receipt.json'
    visual = read(visual_path)
    if (visual.get('status') != 'COMPLETE' or visual.get('numeric_ranking_seen_before_review') is not False
        or visual.get('actually_reviewed') != 840 or sha(Path(visual['review_file'])) != visual['review_file_sha256']):
        raise RuntimeError('Final numerical training analysis requires completed, hash-verified actual image review')
    lock = read(OUT/'training_budget_lock.json'); target = int(lock['updates_per_arm'])
    status_paths = [OUT/'training'/a/'status.json' for a in ARMS]
    status_hashes = {str(p):sha(p) for p in status_paths}; statuses = {a:read(p) for a,p in zip(ARMS,status_paths)}
    allowed = {'COMPLETE'} | ({'TIME_CUTOFF','INTERRUPTED_SAFE','OOM','BLOCKED'} if allow_partial else set())
    if any(s['status'] not in allowed for s in statuses.values()):
        raise RuntimeError('Running/intermediate arm: refuse incomplete final aggregation')
    manifest = read(OUT/'data/train_manifest.json'); updates=[]; exposures=[]; parents=[]; diagnostics=[]; summaries=[]
    ledgers={}; input_hashes={str(visual_path):sha(visual_path)}; errors=[]; phases=[]
    for arm in ARMS:
        state=statuses[arm]; by_step={}; raw_exp=[]; seconds=[]
        attempts=sorted((OUT/'training'/arm/'attempts').iterdir())
        for attempt in attempts:
            final=attempt/'final_status.json'
            if not final.exists():
                raise RuntimeError('Open attempt: '+str(attempt))
            f=read(final); seconds.append(float(f['wall_seconds'])); input_hashes[str(final)]=sha(final)
            attempt_rows=list(lines(attempt/'training_log.jsonl'))
            for row in attempt_rows:
                if row['step'] in by_step:raise ValueError('Duplicate completed update log')
                by_step[row['step']]=dict(row,attempt=attempt.name)
            events=list(lines(attempt/'events.jsonl'))
            checkpoints=[r for r in events if r.get('event')=='CHECKPOINT']
            save_seconds=sum(r['seconds'] for r in checkpoints)
            step_seconds=sum(r['total_step_seconds'] for r in attempt_rows)
            resources=read(attempt/'resources.json')
            resource_summary=resources['summary']
            observed=resources['samples']
            phases.append(dict(arm=arm,attempt=attempt.name,status=f['status'],
                start_completed_step=min(r['step'] for r in attempt_rows)-1,
                stop_completed_step=max(r['step'] for r in attempt_rows),updates=len(attempt_rows),
                worker_wall_seconds=float(f['wall_seconds']),
                observer_first_unix=observed[0]['time_unix'],observer_last_unix=observed[-1]['time_unix'],
                update_loop_seconds=step_seconds,
                input_wait_seconds=sum(r['input_seconds'] for r in attempt_rows),
                model_update_seconds=sum(r['update_seconds'] for r in attempt_rows),
                full_checkpoint_save_seconds=save_seconds,checkpoint_events=len(checkpoints),
                residual_initialization_diagnostics_exports_other_seconds=float(f['wall_seconds'])-step_seconds-save_seconds,
                separate_inference_export_seconds='NA_NOT_SEPARATELY_TIMED',
                peak_allocated_bytes=max(r['cuda_peak_allocated_bytes'] for r in attempt_rows),
                peak_reserved_bytes=max(r['cuda_peak_reserved_bytes'] for r in attempt_rows),
                process_peak_mib=resource_summary['process_peak_mib'],
                process_memory_status=resource_summary['process_memory_status'],
                whole_device_peak_mib=resource_summary['total_device_used_peak_mib'],
                resource_sample_count=resource_summary['sample_count']))
            for tracked in ('training_log.jsonl','exposure.jsonl','events.jsonl','resources.json'):
                input_hashes[str(attempt/tracked)]=sha(attempt/tracked)
            for row in lines(attempt/'exposure.jsonl'):
                raw_exp.append(dict(row,attempt=attempt.name))
        calls=scan_calls(OUT/'training'/arm/'calls.jsonl');ledgers[arm]=calls
        completed=set(calls['completed_updates']); ordered=sorted(completed)
        if ordered != list(range(1,max(ordered,default=0)+1)):errors.append(arm+': nonconsecutive completed updates')
        if set(by_step)!=completed:errors.append(arm+': update logs differ from completed calls')
        if calls['pending_call_ids'] or calls['failed']:errors.append(arm+': pending/failed model calls preserved')
        expected_calls={'train_forward':8*len(completed),'train_backward':8*len(completed),'optimizer_update':len(completed)}
        if calls['attempted']!=expected_calls or calls['successful']!=expected_calls:errors.append(arm+': formal F/B/update accounting differs')
        if state['status']=='COMPLETE' and ordered!=list(range(1,target+1)):errors.append(arm+': target not reached')
        by_parent=collections.defaultdict(lambda:dict(attempted_exposures=0,completed_exposures=0,completed_free_scalars=0,clean_exposures=0,width_exposures=0))
        per_step=collections.defaultdict(list)
        for row in raw_exp:
            success=row['step'] in completed; row=dict(row,arm=arm,optimizer_update_completed=success)
            if row['free_coordinate_count']!=9*(row['N']-row['K']):raise ValueError('Exposure denominator differs')
            p=by_parent[row['uid']];p['attempted_exposures']+=1
            if success:
                p['completed_exposures']+=1;p['completed_free_scalars']+=row['free_coordinate_count']
                p['clean_exposures' if row['alpha']==1. else 'width_exposures']+=1
                per_step[row['step']].append(row)
            exposures.append(row)
        for step,row in sorted(by_step.items()):
            actual=per_step[step]; den=sum(r['free_coordinate_count'] for r in actual)
            if len(actual)!=8 or len({r['uid'] for r in actual})!=8 or sum(r['alpha']==1. for r in actual)!=4 or den!=row['free_coordinate_denominator']:
                errors.append(arm+': bad batch exposure at '+str(step))
            updates.append(dict(arm=arm,**{k:v for k,v in row.items() if k!='sample_losses'}))
        for uid in manifest['pools'][arm]:parents.append(dict(arm=arm,parent_id=uid,**by_parent[uid]))
        for path in sorted((OUT/'diagnostics'/arm).glob('step_*.json')):
            r=read(path);assert r['status']=='PASS' and r['identity']['arm']==arm
            for row in r['rows']:diagnostics.append(dict(arm=arm,step=r['identity']['step'],**row))
            input_hashes[str(path)]=sha(path)
        times=[r['total_step_seconds'] for r in by_step.values()]
        summaries.append(dict(arm=arm,status=state['status'],completed_updates=len(completed),expected_updates=target,
            parent_count=len(manifest['pools'][arm]),completed_exposures=sum(p['completed_exposures'] for p in by_parent.values()),
            completed_free_scalars=sum(p['completed_free_scalars'] for p in by_parent.values()),
            total_attempt_wall_seconds=sum(seconds),
            input_wait_seconds=sum(r['input_wait_seconds'] for r in phases if r['arm']==arm),
            full_checkpoint_save_seconds=sum(r['full_checkpoint_save_seconds'] for r in phases if r['arm']==arm),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in phases if r['arm']==arm),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in phases if r['arm']==arm),
            whole_device_peak_mib=max(r['whole_device_peak_mib'] for r in phases if r['arm']==arm),
            process_peak_mib=None,process_memory_status='NA_WDDM_NVIDIA_SMI',
            mean_update_including_input_seconds=statistics.mean(times) if times else None,
            median_update_including_input_seconds=statistics.median(times) if times else None,
            max_parent_exposure=max((p['completed_exposures'] for p in by_parent.values()),default=0),
            min_parent_exposure=min((p['completed_exposures'] for p in by_parent.values()),default=0)))
    dcalls=scan_calls(OUT/'diagnostics/calls.jsonl')
    if dcalls['attempted'].get('diagnostic_forward',0)>128:errors.append('diagnostic budget exceeded')
    if len(diagnostics)!=dcalls['successful'].get('diagnostic_forward',0):errors.append('diagnostic rows differ from successful calls; unclosed result may exist')
    if dcalls['pending_call_ids'] or dcalls['failed']:errors.append('diagnostic pending/failed calls preserved')
    grouped=collections.defaultdict(list)
    for r in diagnostics:grouped[(r['arm'],r['step'],r['group'])].append(r)
    groups=[dict(arm=a,step=s,group=g,count=len(v),mean_FM=statistics.mean(r['FM'] for r in v),
        free_weighted_FM=sum(r['FM']*r['free_scalar_count'] for r in v)/sum(r['free_scalar_count'] for r in v)) for (a,s,g),v in sorted(grouped.items())]
    joined=collections.defaultdict(dict)
    for r in diagnostics:joined[(r['step'],r['fixture'])][r['arm']]=r
    pairs=[]
    for (step,fixture),v in sorted(joined.items()):
        if set(v)==set(ARMS):
            a,b=(v[x] for x in ARMS)
            assert (a['uid'],a['condition_id'],a['group'],a['free_scalar_count'])==(b['uid'],b['condition_id'],b['group'],b['free_scalar_count'])
            pairs.append(dict(step=step,fixture=fixture,parent_id=a['uid'],condition_id=a['condition_id'],group=a['group'],
                SMALL_FM=a['FM'],EXPANDED_FM=b['FM'],expanded_minus_small_FM=b['FM']-a['FM']))
    if any(sha(Path(p))!=h for p,h in status_hashes.items()):raise RuntimeError('Training status changed during read; no final output written')
    missing_diagnostics=[{'arm':a,'step':step,'status':'NA_NOT_EXECUTED_OR_NOT_CLOSED'} for a in ARMS for step in (0,1000,3000,target) if not any(r['arm']==a and r['step']==step for r in diagnostics)]
    if all(s['status']=='COMPLETE' for s in statuses.values()) and missing_diagnostics:errors.append('Complete training lacks required fixed diagnostics')
    result=dict(status='PASS_CLOSED_TRAINING_ACCOUNTING' if not errors else 'ACCOUNTING_ISSUES_PRESERVED',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),errors=errors,budget_updates_per_arm=target,
        arms=summaries,phases=phases,
        timing_scope='Six serial worker attempt walls may be summed. Input wait and model update are children of update-loop time and must not be added again to wall. Input wait is exposed CPU preparation/OT wait, not summed OT thread work. Checkpoint save excludes inference export, which is not separately timed; residual includes initialization, diagnostics, export, status/log IO and observer shutdown. Whole-device VRAM includes unrelated applications and is not worker process memory.',
        formal_call_ledgers=ledgers,diagnostic_call_ledger=dcalls,missing_diagnostics=missing_diagnostics,OT=counts_ot(),
        preflight=dict(forwards=160,backwards=128,discarded_updates=16,receipt='preflight/report.json'),
        new_model_calls=0,input_receipt_hashes={**status_hashes,**input_hashes},
        interpretation='Fixed FM is a training diagnostic, not a ranking by unique-source reconstruction or a USER-control success metric.')
    folder=output(folder/'summary.json').parent
    save_csv(folder/'phases.csv',phases);save_csv(folder/'updates.csv',updates);save_csv(folder/'exposures.csv',exposures);save_csv(folder/'per_parent_exposure.csv',parents)
    save_csv(folder/'diagnostic_per_fixture.csv',diagnostics);save_csv(folder/'diagnostic_groups.csv',groups);save_csv(folder/'diagnostic_pairs.csv',pairs)
    save_json(folder/'summary.json',result)
    if plot:plot_training(folder,updates,groups)
    return result


def plot_training(folder, updates, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for arm in ARMS:
        rows=sorted((r for r in updates if r['arm']==arm),key=lambda r:r['step'])
        if rows:
            axes[0].plot([r['step'] for r in rows],[statistics.mean(q['FM'] for q in rows[max(0,i-99):i+1]) for i,r in enumerate(rows)],label=arm)
        for group in sorted({r['group'] for r in groups}):
            values=sorted((r for r in groups if r['arm']==arm and r['group']==group),key=lambda r:r['step'])
            axes[1].plot([r['step'] for r in values],[r['mean_FM'] for r in values],marker='o',label=arm+' / '+group)
    axes[0].set(title='Formal FM: trailing 100 updates',xlabel='Optimizer updates',ylabel='Batch free FM')
    axes[1].set(title='Fixed 16-fixture FM diagnostic',xlabel='Optimizer updates',ylabel='Macro mean FM')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=6)
    fig.savefig(output(folder/'training_and_fixed_FM.png'),dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('preflight','final'),required=True)
    p.add_argument('--name',default='final');p.add_argument('--allow-partial',action='store_true');p.add_argument('--plot',action='store_true')
    a=p.parse_args()
    if not a.name.replace('_','').replace('-','').isalnum():raise ValueError('Simple analysis name required')
    if a.stage=='preflight':
        r=audit_preflight();save_json(OUT/'training_analysis'/('preflight_audit_'+a.name+'.json'),r)
    else:r=aggregate_final(OUT/'training_analysis'/a.name,a.allow_partial,a.plot)
    print(json.dumps({k:v for k,v in r.items() if k in ('status','arms','errors','budget_updates_per_arm')},indent=2))


if __name__=='__main__':main()
