"""Read existing normal-forward logs only; no torch/model imports or scoring.

Run once after both Wave2 arms COMPLETE. PLAIN means the existing first1000
updates, not its final3000 endpoint. Missing clean/dirty diagnostics remain NA.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import csv
import hashlib
import json
import math
import statistics
import time
import numpy as np
from . import training_data as D

CATEGORIES=('all_free','seam_third','bulk_total','clean_bulk','dirty_bulk')
SOURCE_KEYS={'seam_third':'source_interface_third','bulk_total':'bulk',
             'clean_bulk':'quality_clean_bulk','dirty_bulk':'quality_dirty_bulk'}
ARMS=('PLAIN_S1000','CLEAN_BULK_FM','SURGICAL_LATE4')


def rows(path,maxstep=1000):
    result=[]
    with Path(path).open(encoding='utf-8-sig') as handle:
        for line in handle:
            if not line.strip():continue
            row=json.loads(line)
            if int(row.get('local_step',0))>maxstep:break
            result.append(row)
    return result


def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def csv_write(path,records):
    if not records:raise ValueError('Empty result table '+str(path))
    fields=list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open('x',encoding='utf-8',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=fields);w.writeheader();w.writerows(records)


def canonical(path):
    return str(D.resolve(path).resolve()).replace('\\','/').lower()


def category(row,name):
    if name=='all_free':
        return int(row['sample_free_coordinate_denominator']),float(row['all_free_velocity_sse'])
    key=SOURCE_KEYS[name]
    if key+'_scalar_count' not in row:return None,None
    return int(row[key+'_scalar_count']),float(row[key+'_velocity_sse'])


def pool(arm,records,scope,**identity):
    result=[]
    for name in CATEGORIES:
        parts=[category(r,name) for r in records]
        missing=any(n is None for n,s in parts)
        if missing:
            if not all(n is None for n,s in parts):raise ValueError('Partially missing diagnostic')
            count=sse=mse=None;status='NA_NOT_RECORDED_IN_OLD_PLAIN_FORWARD_LOGS'
        else:
            count=sum(n for n,s in parts);sse=math.fsum(s for n,s in parts)
            mse=sse/count if count else None;status='PASS' if count else 'NA_NO_COORDINATES'
        result.append(dict(arm=arm,scope=scope,category=name,samples=len(records),scalar_count=count,
            velocity_sse=sse,unweighted_velocity_mse=mse,status=status,
            aggregation='Sum of detached normal-forward SSE / sum of corresponding valid free scalars',**identity))
    return result


def verify_order(logs,diags,exposure):
    if [r['local_step'] for r in logs]!=list(range(1,1001)):
        raise ValueError('Exactly1000 update rows required')
    wanted=[(s,m) for s in range(1,1001) for m in range(8)]
    for records in (diags,exposure):
        if [(r['local_step'],r['microbatch']) for r in records]!=wanted:
            raise ValueError('Exactly8000 ordered normal-forward/exposure rows required')
    for r in logs+diags+exposure:
        if r['global_step']!=13000+r['local_step']:raise ValueError('Wrong global-step cursor')


def exposure_pairing(data,manifest):
    ledger=[];counter=Counter()
    for index,base in enumerate(data['PLAIN_S1000']['exposure']):
        step,micro=base['local_step'],base['microbatch']
        original=manifest['steps'][step-1]['samples'][micro]
        target=data['CLEAN_BULK_FM']['exposure'][index];surgical=data['SURGICAL_LATE4']['exposure'][index]
        for value in (target,surgical):
            for field in ('uid','alpha','t','N','K','sample_index','free_coordinate_count'):
                if value[field]!=base[field]:raise ValueError('Training input mismatch '+field)
            if canonical(value['path'])!=canonical(base['path']) or value['input_sha256']!=original['sha256']:
                raise ValueError('Cached input identity mismatch')
        if canonical(base['path'])!=canonical(original['path']) or base['uid']!=original['uid']:
            raise ValueError('Old PLAIN no longer matches frozen stream')
        counter[base['uid']]+=1
        ledger.append(dict(local_step=step,microbatch=micro,parent_id=base['uid'],N=base['N'],K=base['K'],
            alpha=base['alpha'],t=base['t'],source_sample_index=base['sample_index'],path=base['path'],
            original_cache_sha256=original['sha256'],three_arm_identity_match=True))
    return ledger,dict(status='PASS',examples=8000,updates=1000,parents=len(counter),
        identity_fields=['parent','N','K','alpha','time','source_sample_index','path','original_cache_sha256'],
        random_stream='Exact existing cache, no redraw or rematching',parent_exposures=dict(sorted(counter.items())),
        input_files_rehashed_here=False,hash_scope='Recorded identities; preparation verified cache bytes before training')


def numeric_stats(values):
    values=[float(v) for v in values if v is not None]
    if not values:return dict(count=0,mean=None,min=None,max=None,median=None)
    return dict(count=len(values),mean=statistics.fmean(values),min=min(values),max=max(values),median=statistics.median(values))


def resource_summary(folder,logs,exposure):
    lower=min(r['time_unix'] for r in exposure);upper=max(r['time_unix'] for r in logs)
    observers=[];all_observers=[];inputs=[]
    for path in sorted((folder/'attempts').glob('*/resources.json')):
        value=D.read(path);inputs.append(path)
        all_observers.extend(value['samples'])
        observers.extend(r for r in value['samples'] if lower<=r['time_unix']<=upper)
    process=[];device=[];compute=[]
    for row in observers:
        for line in row.get('processes',{}).get('stdout','').splitlines():
            parts=[p.strip() for p in line.split(',')]
            if len(parts)>=3 and parts[0]==str(row['pid']):
                try:process.append(float(parts[-1]))
                except ValueError:pass
        for line in row.get('gpu',{}).get('stdout','').splitlines():
            parts=[p.strip() for p in line.split(',')]
            if len(parts)>=5:
                try:device.append(float(parts[2]));compute.append(float(parts[4]))
                except ValueError:pass
    update_seconds=sum(r['update_seconds'] for r in logs)
    result=dict(scope='First1000 updates only; timestamp interval starts at first exposure and ends at update1000 log',
        active_interval_start_unix=lower,active_interval_end_unix=upper,
        active_interval_wall_seconds=upper-lower,updates=1000,examples=8000,
        sum_update_seconds=update_seconds,sum_input_seconds=sum(r['input_seconds'] for r in logs),
        sum_forward_seconds=sum(r['forward_seconds'] for r in logs),sum_backward_seconds=sum(r['backward_seconds'] for r in logs),
        sum_optimizer_seconds=sum(r['optimizer_seconds'] for r in logs),
        updates_per_second_update_scope=1000/update_seconds,examples_per_second_update_scope=8000/update_seconds,
        updates_per_second_active_interval=1000/(upper-lower),
        update_seconds_stats=numeric_stats(r['update_seconds'] for r in logs),
        torch_peak_allocated_MiB=max(r['cuda_peak_allocated_bytes'] for r in logs)/2**20,
        torch_peak_reserved_MiB=max(r['cuda_peak_reserved_bytes'] for r in logs)/2**20,
        process_peak_MiB=max(process) if process else None,
        process_memory_status='OBSERVED_MATCHING_PID' if process else 'NA_NVIDIA_SMI_PROCESS_MEMORY_UNAVAILABLE',
        total_device_peak_used_MiB=max(device) if device else None,device_utilization=numeric_stats(compute),
        observer_samples=len(observers),
        owner_lifetime_start_unix=min(r['time_unix'] for r in all_observers) if all_observers else None,
        owner_lifetime_end_unix=max(r['time_unix'] for r in all_observers) if all_observers else None,
        owner_scope='Observer lifetime includes model load, train, checkpoint I/O and final save; not continuous GPU compute.',
        attribution='Torch reserved is not process VRAM; total device includes unrelated processes/concurrent workers.',
        timing_limit='No isolated speedup claim: concurrent load and checkpoint I/O affect wall time; excludes final save.')
    return result,inputs


def concurrency_summary(data,phase):
    result=dict(status='OBSERVED_TIMESTAMPS',arms={})
    for arm in D.ARMS:
        other=next(a for a in D.ARMS if a!=arm)
        lower=phase[other]['owner_lifetime_start_unix'];upper=phase[other]['owner_lifetime_end_unix']
        if lower is None or upper is None:
            result['arms'][arm]=dict(status='NA_MISSING_OWNER_TIMESTAMPS');continue
        groups=defaultdict(list)
        for row in data[arm]['logs']:
            # Log-write time is just after the formal update; CPU diagnostic
            # writes are part of the update scope. No synthetic timing probes.
            end=row['time_unix'];start=end-row['update_seconds']
            overlap=max(0.,min(end,upper)-max(start,lower))
            kind='NO_OTHER_TRAINING_OWNER' if overlap==0 else 'OVERLAP_OTHER_TRAINING_OWNER' if overlap>=row['update_seconds']-1e-5 else 'OWNER_BOUNDARY_UPDATE'
            groups[kind].append(row)
        own0=phase[arm]['owner_lifetime_start_unix'];own1=phase[arm]['owner_lifetime_end_unix']
        phase_rows=[]
        for name,records in sorted(groups.items()):
            seconds=sum(r['update_seconds'] for r in records)
            phase_rows.append(dict(phase=name,updates=len(records),first_update=records[0]['local_step'],last_update=records[-1]['local_step'],
                summed_update_seconds=seconds,update_seconds=numeric_stats(r['update_seconds'] for r in records),
                updates_per_second_update_scope=len(records)/seconds,
                torch_peak_allocated_MiB=max(r['cuda_peak_allocated_bytes'] for r in records)/2**20,
                torch_peak_reserved_MiB=max(r['cuda_peak_reserved_bytes'] for r in records)/2**20))
        result['arms'][arm]=dict(status='PASS',owner_lifetime_seconds=own1-own0,
            overlapping_other_owner_seconds=max(0.,min(own1,upper)-max(own0,lower)),phases=phase_rows)
    result['interpretation']='Owner overlap is measured process-phase concurrency, not a claim both GPUs execute kernels simultaneously. Background user work remains unassigned; no isolated speedup estimate.'
    return result


def training_evidence(root,data):
    reg=D.read(root/'training_registration.json');evidence={};gradient=[];inputs=[]
    for arm in D.ARMS:
        folder=data[arm]['folder'];status=D.read(folder/'status.json')
        inputs.extend([folder/'status.json',folder/'first_formal_backward.json'])
        if status['status']!='COMPLETE' or status['local_step']!=1000 or status['global_step']!=14000:
            raise ValueError('Wave2 training incomplete')
        calls=status['calls']
        expected=dict(train_forward=8000,train_backward=8000,optimizer_update=1000)
        if calls['attempted']!=expected or calls['successful']!=expected or calls['failed'] or calls['pending_call_ids']:
            raise ValueError('Actual formal call budget differs')
        init_paths=sorted((folder/'attempts').glob('*/initialization.json'))
        if not init_paths:raise ValueError('Missing initialization evidence')
        inits=[D.read(p) for p in init_paths];inputs.extend(init_paths)
        source=[r for r in inits if r['global_step']==13000]
        if len(source)!=1:raise ValueError('Exactly one source initialization required')
        init=source[0]
        if init['generator_state_sha256']!=reg['source_generator_sha256'] or init['rng_sha256']!=reg['source_rng_sha256']:
            raise ValueError('Source generator/RNG differs across arms')
        if not init['optimizer_restore']['exact_retained_step_and_moments'] or init['head_forward_calls']!=0 or init['head_loaded']:
            raise ValueError('Restoration/head evidence differs')
        checks=jsonl(folder/'frozen_state_checks.jsonl');inputs.append(folder/'frozen_state_checks.jsonl')
        if not checks or checks[-1]['local_step']!=1000 or any(r['status']!='PASS' or r['initial']!=r['actual'] for r in checks):
            raise ValueError('Frozen state check record failed')
        if arm=='SURGICAL_LATE4' and any(r['initial']!=init['initial_frozen_state_sha256'] for r in checks):
            raise ValueError('Frozen initial identity changed')
        logs=data[arm]['logs']
        for prefix in ('role_embedding','context_encoder.','backbone.x_embedder.','backbone.t_embedder.','backbone.y_embedder.'):
            values=[r['preprocessing_gradient_norms'][prefix] for r in logs]
            if any(not np.isfinite(v) for v in values):raise ValueError('Nonfinite preprocessing gradient evidence')
            gradient.append(dict(arm=arm,parameter_group=prefix,updates=1000,**numeric_stats(values),
                nonzero_updates=sum(v>0 for v in values),source='Already-computed formal gradients, before clip',extra_backward_calls=0))
        evidence[arm]=dict(status='PASS_RECORDED_EVIDENCE',calls=calls,common_source_generator_sha256=init['generator_state_sha256'],
            common_source_rng_sha256=init['rng_sha256'],source_optimizer_sha256=init['source_optimizer_sha256'],
            retained_optimizer_states=init['optimizer_restore']['retained_parameter_states'],
            removed_frozen_states=init['optimizer_restore']['removed_frozen_parameter_states'],
            exact_retained_moments_and_step13000=init['optimizer_restore']['exact_retained_step_and_moments'],
            frozen_state_initial=init['initial_frozen_state_sha256'],frozen_state_checks=checks,
            final_global_step=status['global_step'],checkpoint=status['checkpoint'],inference_export=status['inference_export'],
            pretrained_forward_code_unchanged=init['private_seam_code_object_identical'] and init['private_native_code_object_identical'],
            frozen_blocks_kept_in_autograd=init['frozen_block_forward_in_autograd'],
            checkpoint_tensor_reloaded_here=False,
            evidence_boundary='Checks normal execution receipts; independent final audit verifies checkpoint tensors separately.')
    return evidence,gradient,inputs


def plot_curves(out,windows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    color={'PLAIN_S1000':'#555555','CLEAN_BULK_FM':'#0072B2','SURGICAL_LATE4':'#D55E00'}
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,name in zip(axes.flat,CATEGORIES):
        for arm in ARMS:
            selected=[r for r in windows if r['arm']==arm and r['category']==name and r['status']=='PASS']
            if selected:ax.plot([r['end_step'] for r in selected],[r['unweighted_velocity_mse'] for r in selected],label=arm,color=color[arm],marker='o',ms=3)
        ax.set_title(name.replace('_',' '));ax.set_xlabel('Continuation update (100-update windows)');ax.set_ylabel('Unweighted velocity MSE');ax.grid(alpha=.2)
    axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center')
    axes.flat[-1].text(.05,.12,'PLAIN clean/dirty: not recorded.\nTraining interpolation errors, not rollout metrics.',transform=axes.flat[-1].transAxes,fontsize=9)
    fig.savefig(out/'unweighted_learning_curves.png',dpi=180);fig.savefig(out/'unweighted_learning_curves.svg');plt.close(fig)


def run(root):
    root=D.resolve(root).resolve();out=root/'training_summary'
    if out.exists():raise FileExistsError('Summary exists; no overwrite or partial rerun')
    folders={'PLAIN_S1000':D.ROOT/'experiments/r2_seam_pool47_couple_v1/training/PLAIN3000',
             **{arm:root/'training'/arm for arm in D.ARMS}}
    for arm in D.ARMS:
        status=D.read(folders[arm]/'status.json')
        if status['status']!='COMPLETE':raise RuntimeError('Wait until both Wave2 arms complete')
    inputs=[];data={}
    for arm,folder in folders.items():
        paths=[folder/name for name in ('training_log.jsonl','geometry_diagnostics.jsonl','exposure.jsonl')]
        logs,diags,exposure=map(rows,paths);verify_order(logs,diags,exposure);inputs.extend(paths)
        data[arm]=dict(folder=folder,logs=logs,diags=diags,exposure=exposure)
    reg=D.read(root/'training_registration.json');manifest=D.read(D.resolve(reg['shared_data_manifest']))
    inputs.extend([root/'training_registration.json',D.resolve(reg['shared_data_manifest'])])
    pairing,identity=exposure_pairing(data,manifest)
    evidence,gradients,more=training_evidence(root,data);inputs.extend(more)
    updates=[];windows=[];times=[];joint=[];allpool=[];losses=[];phase={}
    for arm in ARMS:
        diagnostics=data[arm]['diags'];logs=data[arm]['logs']
        groups=defaultdict(list);timegroups=defaultdict(list);both=defaultdict(list)
        for r in diagnostics:
            groups[r['local_step']].append(r);timegroups[r['time_bin']].append(r)
            both[((r['local_step']-1)//100,int(r['time_bin']))].append(r)
        for step,records in sorted(groups.items()):updates.extend(pool(arm,records,'UPDATE',local_step=step,global_step=13000+step))
        for window in range(10):
            selected=[r for r in diagnostics if window*100<r['local_step']<=(window+1)*100]
            windows.extend(pool(arm,selected,'WINDOW100',start_step=window*100+1,end_step=(window+1)*100))
            samplelogs=logs[window*100:(window+1)*100]
            losses.append(dict(arm=arm,start_step=window*100+1,end_step=(window+1)*100,
                optimized_objective='QUALITY_WEIGHTED_FM' if arm=='CLEAN_BULK_FM' else 'ORIGINAL_UNWEIGHTED_FM',
                mean_optimized_loss=statistics.fmean(r['FM'] for r in samplelogs),
                normalization='Per-update whole effective batch denominator; mean over100 updates',
                mean_seam_coefficient_ratio=statistics.fmean(r.get('effective_unit_seam_coefficient_ratio_vs_plain',1.) for r in samplelogs),
                min_seam_coefficient_ratio=min(r.get('effective_unit_seam_coefficient_ratio_vs_plain',1.) for r in samplelogs),
                max_seam_coefficient_ratio=max(r.get('effective_unit_seam_coefficient_ratio_vs_plain',1.) for r in samplelogs),
                cross_objective_loss_comparison='Not a quality ranking: objectives and normalization differ'))
        for bi,records in sorted(timegroups.items()):times.extend(pool(arm,records,'TIME_BIN',time_bin=bi,t_lower=records[0]['time_bin_lower'],t_upper=records[0]['time_bin_upper']))
        for (window,bi),records in sorted(both.items()):joint.extend(pool(arm,records,'WINDOW100_TIME_BIN',
            start_step=window*100+1,end_step=(window+1)*100,time_bin=bi,t_lower=records[0]['time_bin_lower'],t_upper=records[0]['time_bin_upper']))
        allpool.extend(pool(arm,diagnostics,'ALL_1000_UPDATES'))
        phase[arm],more=resource_summary(data[arm]['folder'],logs,data[arm]['exposure']);inputs.extend(more)
    # Denominators for paired comparisons must agree even when A loss weights do not.
    compare=[]
    bywindow={(r['arm'],r['end_step'],r['category']):r for r in windows}
    for arm in D.ARMS:
        for end in range(100,1001,100):
            for categoryname in CATEGORIES:
                base=bywindow[('PLAIN_S1000',end,categoryname)];new=bywindow[(arm,end,categoryname)]
                if base['status']=='PASS' and new['status']=='PASS':
                    if base['scalar_count']!=new['scalar_count']:raise ValueError('Paired unweighted scalar denominators differ')
                    delta=new['unweighted_velocity_mse']-base['unweighted_velocity_mse'];status='PASS'
                else:delta=None;status='NA_PLAIN_CATEGORY_NOT_RECORDED'
                compare.append(dict(arm=arm,reference='PLAIN_S1000',end_step=end,category=categoryname,
                    reference_unweighted_mse=base['unweighted_velocity_mse'],arm_unweighted_mse=new['unweighted_velocity_mse'],
                    delta_unweighted_mse=delta,status=status,lower_is_better_for_training_fit=True,
                    interpretation='Paired training interpolation fit, not generalization or generation quality'))
    out.mkdir()
    for name,value in (('per_update_unweighted.csv',updates),('window100_unweighted.csv',windows),('time_bin_unweighted.csv',times),
        ('window100_time_bin_unweighted.csv',joint),('overall_unweighted.csv',allpool),('optimized_loss_separate.csv',losses),
        ('paired_window_deltas.csv',compare),('exposure_pairing.csv',pairing),('preprocessing_gradients.csv',gradients)):
        csv_write(out/name,value)
    D.write_new(out/'exposure_identity.json',identity);D.write_new(out/'training_evidence.json',evidence);D.write_new(out/'phase_resources.json',phase)
    concurrency=concurrency_summary(data,phase)
    decision=root/'training_second_worker_decision.json'
    if decision.exists():
        concurrency['dispatch_decision']=D.read(decision);inputs.append(decision)
    D.write_new(out/'concurrency_phases.json',concurrency)
    plot_curves(out,windows)
    final=[r for r in compare if r['end_step']==1000]
    statement=dict(status='COMPLETE',neural_network_constructions=0,neural_forwards=0,neural_backwards=0,
        geometry_scoring_calls=0,training_updates=0,source='Existing formal normal-forward logs only',
        training_fit_warning='These are interpolation-state errors. Endpoint rollout quality must be judged independently.',
        loss_warning='A weighted objective cannot be ranked against unweighted FM as if it were the same scalar.',
        seam_warning='Owner weights remain1 but wholebatch denominator raises their effective relative coefficient.',
        PLAIN_missing='Clean/dirty split was not recorded and remains NA; no additional forward was authorized.',
        matched_input_identity=identity['status'],final100_update_paired_deltas=final,
        source_hashes={str(p):D.digest(p) for p in dict.fromkeys(inputs)},summary_code_sha256=D.digest(__file__),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    D.write_new(out/'completion.json',statement)
    return statement


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='experiments/r2_seam_quality_wave1_wave2_v1')
    args=parser.parse_args();result=run(args.root)
    print(json.dumps(dict(status=result['status'],models=0,final100_update_paired_deltas=result['final100_update_paired_deltas']),indent=2))
