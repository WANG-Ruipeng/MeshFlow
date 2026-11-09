"""CPU-only, read-only log analysis; analysis/* are explicitly mutable outputs.

Usage: python -B analysis_training.py --partial
       python -B analysis_training.py  # requires completed 1500-update ledger
No model/checkpoint import, weight loading, GPU query, forward or re-scoring.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,csv,datetime,hashlib,io,json,math,re,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
ART=OUT/'analysis'
SOURCE_HASHES={}
READ_NOTES=[]

def snapshot(path):
    data=Path(path).read_bytes()
    SOURCE_HASHES[str(path)]=dict(sha256=hashlib.sha256(data).hexdigest(),snapshot_bytes=len(data))
    return data

def js(path):return json.loads(snapshot(path).decode('utf-8-sig'))

def jl(path):
    data=snapshot(path);lines=data.splitlines(keepends=True);rows=[]
    for i,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1 and not line.endswith(b'\n'):
                READ_NOTES.append(dict(path=str(path),ignored_incomplete_tail_bytes=len(line)))
            else:raise
    return rows

def write_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def descriptive(values):
    x=np.asarray(values,float)
    if not len(x):return None
    assert np.isfinite(x).all()
    return dict(n=len(x),sum=float(x.sum()),mean=float(x.mean()),median=float(np.median(x)),
                p90=float(np.percentile(x,90)),min=float(x.min()),max=float(x.max()))

def ratio(rows,num,den):
    if not all(num in r and den in r for r in rows):return None
    d=sum(r[den] for r in rows)
    return sum(r[num] for r in rows)/d if d else None

def window(rows,label):
    scalar_fields=('all_free','bulk','seam_tip')
    objfield='objective' if 'objective' in rows[0] else 'FM'
    out=dict(label=label,start_step=rows[0]['local_step'],end_step=rows[-1]['local_step'],updates=len(rows),
             complete_100_window=len(rows)==100,objective_mean=float(np.mean([r[objfield] for r in rows])),
             objective_field=objfield)
    for part in scalar_fields:
        out[part+'_velocity_mse']=ratio(rows,part+'_velocity_sse',part+'_scalar_count')
        out[part+'_scalar_count']=sum(r[part+'_scalar_count'] for r in rows)
        ep='endpoint_sse' if part=='all_free' else part+'_endpoint_sse'
        out[part+'_endpoint_mse']=ratio(rows,ep,part+'_scalar_count')
    for field in ('input_seconds','update_seconds','generator_gradient_norm'):
        out[field+'_mean']=float(np.mean([r[field] for r in rows]))
    out['step_seconds_mean']=out['input_seconds_mean']+out['update_seconds_mean']
    out['peak_allocated_bytes']=max(r['peak_allocated_bytes'] for r in rows)
    out['peak_reserved_bytes']=max(r['peak_reserved_bytes'] for r in rows)
    return out

def read_arm(path,label,expected=1500,partial=False):
    logs=jl(path/'training_log.jsonl')
    assert logs and [r['local_step'] for r in logs]==list(range(1,len(logs)+1)), 'Missing/repeated completed step'
    assert len(logs)<=expected
    status=js(path/'status.json')
    if not partial:assert len(logs)==expected and status['status']=='COMPLETE', 'Full analysis requires completed1500'
    fields=['all_free_velocity_sse','all_free_scalar_count','bulk_velocity_sse','bulk_scalar_count','seam_tip_velocity_sse','seam_tip_scalar_count']
    for r in logs:
        assert all(k in r and math.isfinite(float(r[k])) for k in fields)
        assert r['bulk_scalar_count']+r['seam_tip_scalar_count']==r['all_free_scalar_count']
        assert math.isclose(r['bulk_velocity_sse']+r['seam_tip_velocity_sse'],r['all_free_velocity_sse'],rel_tol=1e-6,abs_tol=1e-6)
        objective=r.get('objective',r.get('FM'))
        assert math.isclose(objective,r['all_free_velocity_sse']/r['all_free_scalar_count'],rel_tol=4e-6,abs_tol=1e-7)
    exp_all=jl(path/'exposure.jsonl');done=len(logs)
    exp=[r for r in exp_all if r['local_step']<=done]
    ids=[(r['local_step'],r['microbatch']) for r in exp]
    assert len(ids)==len(set(ids)) and set(ids)=={(s,mi) for s in range(1,done+1) for mi in range(8)}
    # Completed exposure statistics must aggregate to each completed batch log.
    grouped=collections.defaultdict(list)
    for r in exp:grouped[r['local_step']].append(r)
    for r in logs:
        for key in fields:
            assert math.isclose(sum(v[key] for v in grouped[r['local_step']]),r[key],rel_tol=1e-8,abs_tol=1e-7)
    resources=[]
    for p in sorted(path.glob('resources_step*.json')):
        step=int(re.search(r'step(\d+)',p.stem).group(1))
        if step<=done:resources.append(js(p))
    gpu=[];ram=[]
    for r in resources:
        for line in r.get('gpu',{}).get('stdout','').strip().splitlines():
            cols=[x.strip() for x in line.split(',')]
            if len(cols)==5:
                try:gpu.append(dict(name=cols[0],total_mib=float(cols[1]),used_mib=float(cols[2]),free_mib=float(cols[3]),utilization_percent=float(cols[4])))
                except ValueError:pass
        match=re.search(r'MemAvailable:\s+(\d+) kB',r.get('memory',''))
        if match:ram.append(int(match.group(1))*1024)
    condition=collections.Counter(r['condition_id'] for r in exp)
    parent=collections.Counter(r['parent_id'] for r in exp)
    part=collections.Counter(r['part_type'] for r in exp)
    summary=dict(label=label,completed_updates=done,status_at_read=status.get('status'),status_local_step=status.get('local_step'),
        completed_exposure_instances=len(exp),pending_exposure_rows_excluded=len(exp_all)-len(exp),
        metrics=window(logs,label),actual_parent_exposure=dict(sorted(parent.items())),actual_condition_exposure=dict(sorted(condition.items())),
        actual_part_exposure=dict(sorted(part.items())),actual_parents_seen=len(parent),actual_conditions_seen=len(condition),
        step_time=dict(input=descriptive([r['input_seconds'] for r in logs]),update=descriptive([r['update_seconds'] for r in logs]),
             input_plus_update=descriptive([r['input_seconds']+r['update_seconds'] for r in logs]),
             note='Batch timing excludes startup, checkpoint serialization and some status/resource overhead; no isolated-throughput claim.'),
        resource=dict(allocator_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in logs),
             allocator_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in logs),
             allocator_scope='Per real batch peaks include backward/AdamW; not full process VRAM.',
             periodic_device_used_peak_mib=max((r['used_mib'] for r in gpu),default=None),
             periodic_device_free_min_mib=min((r['free_mib'] for r in gpu),default=None),
             process_peak_mib=status.get('resources',{}).get('process_peak_mib'),
             process_peak_status=status.get('resources',{}).get('process_memory_status','NA_RUNNING_OR_UNAVAILABLE'),
             observer_summary=status.get('resources'),CPU_logical=resources[0].get('cpu_logical') if resources else None,
             sampled_available_RAM_min_bytes=min(ram,default=None),
             scope='Only already saved snapshots; device memory includes background/user processes. No new GPU query.'),
        reported_attempt_elapsed_seconds=status.get('elapsed_seconds'))
    return logs,exp,summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    ownlog,ownexp,own=read_arm(OUT/'training/COVERAGE_SEAM','COVERAGE',partial=args.partial)
    baselog,baseexp,baseline=read_arm(ROOT/'experiments/r2_bench_adaptation_strength_v1/training/A_COMMON_LR1E5','A_BASELINE')
    schedule=js(OUT/'data/schedule.json');manifest=js(OUT/'data/manifest.json');reg=js(OUT/'registration.json')
    assert reg['updates']==1500 and len(schedule['steps'])==1500
    planned={(s['step'],r['sample_index']):r for s in schedule['steps'] for r in s['samples']}
    for r in ownexp:
        p=planned[r['local_step'],r['microbatch']]
        for key in ('condition_id','parent_id','part_type','scale_xyz','noise_seed','epsilon_seed','time_seed','permutation_seed'):
            assert r[key]==p[key],key
    windows=[];done=len(ownlog)
    for start in range(0,done,100):
        end=min(start+100,done)
        windows.append(window(ownlog[start:end],'COVERAGE'))
        windows.append(window(baselog[start:end],'A_BASELINE'))
    # These fixed finite-sample quantities share source definitions. Endpoint
    # errors exist only in coverage logs and are never synthesized for A.
    comparable=dict(fields=['objective_mean','all_free_velocity_mse','bulk_velocity_mse','seam_tip_velocity_mse'],
        objective_mapping='Coverage objective and A FM both sum microbatch SSE / batch free scalar denominator.',
        mask_mapping='Both use free valid corners; seam fixes corners0,1 and scores corner2; bulk has no fixed corner.',
        source_definitions=['r2/pilots/bench_coverage_expansion_v1/training.py:statistics',
          'r2/pilots/bench_clean_multipart_seam_v1/training.py:stat'],
        caveat='Same quantity definitions, different parents/conditions/free masks and denominator distribution. Lower training loss does NOT rank generation quality.',
        endpoint_A='NA_NOT_LOGGED; no new forward or inferred A endpoint error',baseline_curve_scope=f'Matched prefix1..{done}; full A summary remains1500')
    source_rows={r['condition_id']:r for r in manifest['train']}
    topology=collections.Counter()
    for r in ownexp:
        src=source_rows[r['condition_id']]
        key='CLOSED_C' if src['boundary_edges']==0 else ('OPEN_NO_OWNER' if src['source_owner_count']==0 else 'OPEN_OWNER')
        topology[key]+=1
    own['actual_topology_exposure']=dict(topology)
    ART.mkdir(exist_ok=True)
    temp=ART/'training_windows.csv.tmp'
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)
    os.replace(temp,ART/'training_windows.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
    fields=[('objective_mean','Mean FM objective'),('bulk_velocity_mse','Bulk velocity MSE (sumSSE / sumScalars)'),
            ('seam_tip_velocity_mse','Seam-tip velocity MSE (sumSSE / sumScalars)'),('step_seconds_mean','Mean input + update seconds')]
    for ax,(field,title) in zip(axes.flat,fields):
        for label,color,style in [('COVERAGE','#0072B2','-'),('A_BASELINE','#D55E00','--')]:
            rs=[w for w in windows if w['label']==label]
            ax.plot([w['end_step'] for w in rs],[w[field] for w in rs],style,color=color,marker='o',ms=3,label=label)
        ax.set(title=title,xlabel='Completed local optimizer update');ax.grid(alpha=.22);ax.legend(fontsize=8)
        ax.set_xlim(0,max(100,done));ax.set_ylim(bottom=0)
    window_note='last window is partial' if done%100 else 'all windows complete'
    fig.suptitle(f'Bench coverage data experiment | {"PARTIAL" if args.partial else "COMPLETE"}: {done}/1500 updates\n'
                 f'100-update windows ({window_note}). Different input pools: loss is not a quality ranking.',fontsize=12)
    pngtemp=ART/'training_curve.tmp.png';fig.savefig(pngtemp,dpi=170);plt.close(fig);os.replace(pngtemp,ART/'training_curve.png')
    assert 'torch' not in sys.modules
    summary=dict(schema='BENCH_COVERAGE_READ_ONLY_TRAINING_ANALYSIS_V1',mode='PARTIAL' if args.partial else 'COMPLETE',
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),coverage=own,baseline_full=baseline,
        baseline_matched_prefix=window(baselog[:done],'A_BASELINE'),comparison=comparable,
        snapshot_notes=READ_NOTES,source_snapshots=SOURCE_HASHES,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validation=dict(completed_step_sequence=True,completed_exposure8_per_update=True,exposure_aggregates_match_batch=True,
            schedule_exact_for_completed_exposures=True,source_SSE_count_partitions=True,objective_SSE_denominator=True,
            full1500_required=not args.partial,no_torch_import=True),
        limits=dict(new_model_calls=0,new_GPU_calls=0,new_scoring=0,source_writes=0),
        interpretation='Training fit curve only. No plateau/quality/promotion conclusion is inferred automatically. Outputs in analysis/ may be refreshed.')
    write_json(ART/'training_summary.json',summary)
    print(json.dumps(dict(mode=summary['mode'],completed_updates=done,completed_instances=len(ownexp),
        observed_parents=own['actual_parents_seen'],observed_conditions=own['actual_conditions_seen'],
        last_window=windows[-2],outputs=['analysis/training_summary.json','analysis/training_windows.csv','analysis/training_curve.png'])))

if __name__=='__main__':main()
