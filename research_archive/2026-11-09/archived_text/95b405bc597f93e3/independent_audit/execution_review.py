"""CPU-only independent saved-ledger/identity/resource review."""
import collections, datetime, json, sys
from pathlib import Path
import numpy as np
from r2.pilots.seat_failure_diagnosis_v1 import common as C

r=C.load_registration();out=C.OUT/'independent_audit';prep=C.read(out/'preparation_review.json')
protected=dict(prep['protected_files']);protected.update(C.read(C.OUT/'execution/gpu_code_freeze.json')['files'])
for p,h in protected.items():assert C.sha(p)==h
qby={q['request_id']:q for q in r['requests']};rawidx=C.read(C.OUT/'evaluation/raw_index.json');postidx=C.read(C.OUT/'evaluation/post_index.json')
assert rawidx['status']==postidx['status']=='COMPLETE'
raws={x['request_id']:x for x in rawidx['rows']};posts={x['request_id']:x for x in postidx['rows']}
assert set(raws)==set(posts)==set(qby) and len(raws)==len(rawidx['rows'])==40
rows=[];postrows=[];hashes={};callcount=0;raw_intervals=collections.defaultdict(list);post_intervals=collections.defaultdict(list);events=[]
for p in (C.OUT/'execution').glob('worker_*_events.jsonl'):
    ev=[json.loads(s) for s in p.read_text().splitlines()];assert not any('STOPPED' in x['event'] for x in ev)
    for x in ev:x['_path']=str(p)
    events.extend(ev)

def verify(p,h):
    assert C.sha(p)==h,str(p);hashes[str(p)]=h

def stamp(t):return datetime.datetime.fromisoformat(t).timestamp()

for rid,q in qby.items():
    rr=raws[rid];pr=posts[rid];a=C.arrays(rr['raw_path']);b=C.arrays(q['input_path']);n,k=q['N'],q['K']
    for field in ('raw','ply','result'):verify(rr[field+'_path'],rr[field+'_sha256'])
    verify(pr['post_path'],pr['post_sha256'])
    for p,h in pr['files'].items():verify(p,h)
    path=a['trajectory'].reshape(51,n,9);raw=a['raw'].reshape(n,3,3)
    assert np.isfinite(path).all() and path.dtype==np.float32
    assert all(C.same(s[:k].reshape(k,3,3),b['C']) for s in path)
    assert C.same(path[-1].reshape(n,3,3),raw) and C.same(path[0,k:].reshape(1,n-k,9),b['z'][:,k:])
    assert C.same(a['C'],b['C']) and C.same(a['z'],b['z'])
    src=C.arrays(q['condition_source']);original=src['C'][np.arange(k)[:,None],np.argsort(src['C_source_corner_permutations'],axis=1)]
    po=C.arrays(pr['post_path']);assert C.same(po['output'][:k],original)
    proposal=C.read(pr['proposal_receipt']);assert proposal['checkpoint_sha256']==r['d0_checkpoint']['sha256'] and proposal['model_forwards']==1
    if q['action']=='CACHE':
        assert rr['new_rollouts']==pr['new_D0_F']==0
        for st in ('raw','post'):verify(q['cache_'+st+'_receipt'],q['cache_'+st+'_receipt_sha256'])
        continue
    folder=C.resolve(rr['raw_path']).parent;lines=[json.loads(x) for x in (folder/'calls.jsonl').read_text().splitlines()]
    assert len(lines)==103 and lines[0]['event']=='CELL_REGISTERED'
    for step in range(50):
        reserve,returned=lines[2+step*2:4+step*2]
        assert reserve['event']=='RESERVE' and reserve['kind']=='sampling_forward' and reserve['ordinal']==step+1
        context=reserve['context'];assert context['request_id']==rid and context['checkpoint_sha256']==r['checkpoint']['sha256'] and context['input_sha256']==q['input_sha256'] and context['euler_step']==step
        assert returned['event']=='RETURN' and returned['kind']=='sampling_forward' and returned['time_unix']>=reserve['time_unix']
    assert lines[1]['kind']=='rollout_attempt' and lines[-1]['event']=='ROLLOUT_COMPLETE'
    velocity=C.arrays(rr['velocities_path'])['velocity'];verify(rr['velocities_path'],rr['velocities_sha256'])
    assert velocity.shape==(50,n,9) and np.isfinite(velocity).all() and not np.any(velocity.view(np.uint32)&0xffff)
    assert all(C.same(path[s+1,k:],path[s,k:]+velocity[s,k:]*np.float32(.02)) for s in range(50))
    start,end=lines[1]['time_unix'],lines[-1]['time_unix'];worker=rr['worker'];raw_intervals[worker].append((start,end))
    audit=C.read(rr['result_path']);audit=audit.get('sampling',audit)
    rows.append(dict(request_id=rid,worker=worker,start_unix=start,end_unix=end,seconds=audit['seconds'],sampling_reserved=50,sampling_returned=50,peak_allocated_MiB=audit['cuda_peak_allocated_bytes']/2**20,peak_reserved_MiB=audit['cuda_peak_reserved_bytes']/2**20,all51_C_bitwise=True,euler_recurrence_bitwise=True))
    callcount+=50
    ev=[e for e in events if e.get('request_id')==rid]
    assert len(ev)==2 and {e['event'] for e in ev}=={'D0_FORWARD_RESERVED','D0_FORWARD_RETURNED'}
    es=next(x for x in ev if x['event']=='D0_FORWARD_RESERVED');ee=next(x for x in ev if x['event']=='D0_FORWARD_RETURNED');ps,pe=stamp(es['utc']),stamp(ee['utc']);assert pe>=ps
    post_intervals[worker].append((ps,pe))
    postrows.append(dict(request_id=rid,worker=worker,reserved_utc=es['utc'],returned_utc=ee['utc'],start_unix=ps,end_unix=pe,forward_seconds=proposal['seconds'],peak_allocated_MiB=proposal['peak_allocated_bytes']/2**20,peak_reserved_MiB=proposal['peak_reserved_bytes']/2**20,model_forwards=1,source_C_bitwise=True))
assert len(rows)==24 and len(postrows)==24 and callcount==1200 and len(events)==48
assert max(x['end_unix'] for x in rows)<min(x['start_unix'] for x in postrows)

snapshots=[]
for p in sorted((C.OUT/'resources').glob('*.json')):
    a=C.read(p);v=a['gpu']['stdout'].strip().splitlines()[0].split(',');assert a['gpu']['returncode']==0
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in a['memory'].splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    snapshots.append(dict(path=str(p),utc=a['utc'],unix=stamp(a['utc']),gpu_name=v[0].strip(),total_MiB=float(v[1]),used_MiB=float(v[2]),free_MiB=float(v[3]),utilization_percent=float(v[4]),CPU_logical=a['cpu_logical'],CPU_load1=a['loadavg'][0],MemAvailable_bytes=mem.get('MemAvailable'),process_query=a['processes']['stdout']))

intervals=[]
for stage,groups in (('GENERATOR',raw_intervals),('D0',post_intervals)):
    for worker,cells in groups.items():
        start=min(x[0] for x in cells);end=max(x[1] for x in cells)
        prefix='generator' if stage=='GENERATOR' else 'D0'
        before=next(x for x in snapshots if Path(x['path']).name.startswith(prefix+'_before_load_worker_'+str(worker)))
        finish=next(x for x in snapshots if Path(x['path']).name.startswith(prefix+'_finish_worker_'+str(worker)))
        intervals.append(dict(stage=stage,worker=worker,cells=len(cells),model_cell_interval_start_unix=start,model_cell_interval_end_unix=end,model_cell_span_seconds=end-start,before_load_utc=before['utc'],finish_utc=finish['utc'],before_load_to_finish_seconds=finish['unix']-before['unix']))
        identity=C.read(C.OUT/'execution'/f'{prefix}_worker_{worker}_of_2_identity.json');complete=C.read(C.OUT/'execution'/f'{prefix}_worker_{worker}_of_2_complete.json')
        if stage=='GENERATOR':assert identity['generator_state_sha256']==complete['generator_state_sha256'] and complete['new_sampling_F']==600
        else:assert identity['checkpoint_sha256']==r['d0_checkpoint']['sha256'] and complete['new_D0_F']==12

overlaps=[]
for stage in ('GENERATOR','D0'):
    a,b=sorted([x for x in intervals if x['stage']==stage],key=lambda x:x['worker'])
    overlap=max(0.,min(a['model_cell_interval_end_unix'],b['model_cell_interval_end_unix'])-max(a['model_cell_interval_start_unix'],b['model_cell_interval_start_unix']))
    overlaps.append(dict(stage=stage,two_configured_partitions=True,actual_model_cell_span_overlap_seconds=overlap,observed_execution='SERIAL_PARTITIONS' if overlap==0 else 'OVERLAPPING_PARTITIONS',speedup_claim='NOT_ESTABLISHED_NO_MATCHED_ISOLATED_BENCHMARK'))
assert all(x['actual_model_cell_span_overlap_seconds']==0 for x in overlaps)
for p,h in protected.items():assert C.sha(p)==h
assert 'torch' not in sys.modules
value=dict(status='PASS',utc=C.utc(),registration_sha256=C.sha(C.OUT/'registration.json'),logical_RAW=40,logical_POST=40,cached_RAW=16,cached_POST=16,new_rollouts=24,generator_sampling_reserved=1200,generator_sampling_returned=1200,new_D0_reserved=24,new_D0_returned=24,unresolved_calls=0,failed_calls=0,training=0,backwards=0,audit_model_forwards=0,all_RAW_before_first_D0=True,RAW_to_D0_gap_seconds=min(x['start_unix'] for x in postrows)-max(x['end_unix'] for x in rows),protected_source_code_files_checked=len(protected),protected_hashes_unchanged=True,artifact_hashes_checked=len(hashes),checkpoint_evidence='Actual workers loaded and SHA-verified the registered generator/D0; generator tensor state unchanged after each worker. This CPU review does not repeat hashing the 7.29GB D0 asset.',sampled_whole_card_peak_used_MiB=max(x['used_MiB'] for x in snapshots),sampled_whole_card_min_free_MiB=min(x['free_MiB'] for x in snapshots),sampled_whole_card_peak_utilization_percent=max(x['utilization_percent'] for x in snapshots),process_VRAM='NA: empty WSL/WDDM process query; card occupancy includes unrelated processes, PyTorch peaks cover their stated per-stage scope only.',headroom_MiB=max(2048.,.1*snapshots[0]['total_MiB']),generator_max_allocated_MiB=max(x['peak_allocated_MiB'] for x in rows),generator_max_reserved_MiB=max(x['peak_reserved_MiB'] for x in rows),D0_max_allocated_MiB=max(x['peak_allocated_MiB'] for x in postrows),D0_max_reserved_MiB=max(x['peak_reserved_MiB'] for x in postrows),worker_intervals=intervals,overlap_audit=overlaps,RAW_cells=rows,D0_cells=postrows,resources=snapshots,audit_source_sha256=C.sha(__file__))
C.write(out/'execution_review.json',value);C.csvwrite(out/'resource_timeline.csv',snapshots);C.csvwrite(out/'worker_time_intervals.csv',intervals)
text='''# Independent execution and resource review

PASS. Registered 24 new Euler50 outputs used exactly 1200 reserved/returned generator forwards; 24 new outputs each used exactly one official D0. There are also 16 verified historical RAW/POST pairs. Logical totals are 40 RAW and 40 POST. No unresolved/failed model calls or training updates were found.

All new RAW completed %.3f seconds before the first D0 reservation. Independent array checks verified all 51 C states and all 50 Euler recurrences for each new RAW; POST retained the registered source C corner order.

Both stages had two configured partitions of 12, but both actually ran sequentially: measured generator partition overlap = 0 seconds; D0 partition overlap = 0 seconds. This is not evidence of two concurrent GPU workers or speedup. CPU work may overlap independently.

Observed card peak used %.1f MiB and minimum free %.1f MiB, from discrete snapshots; these are not continuous/process memory peaks. Unrelated allocations are included. PyTorch generator max allocated/reserved = %.1f/%.1f MiB; D0 = %.1f/%.1f MiB. No process-specific VRAM was returned by WSL/WDDM.

%d protected preparation/source/code identities remained unchanged; %d recorded artifacts were SHA checked. Large checkpoint identity relies on the actual loaders' SHA checks and the generator after-state checks, rather than another large-weight hash pass.

See execution_review.json, resource_timeline.csv and worker_time_intervals.csv for exact times, counts and scopes.
'''%(value['RAW_to_D0_gap_seconds'],value['sampled_whole_card_peak_used_MiB'],value['sampled_whole_card_min_free_MiB'],value['generator_max_allocated_MiB'],value['generator_max_reserved_MiB'],value['D0_max_allocated_MiB'],value['D0_max_reserved_MiB'],len(protected),len(hashes))
C._write_bytes(out/'EXECUTION_REVIEW.md',text.encode('utf-8'))
C.write(out/'audit_execution_notes.json',dict(preparation_audit_first_attempt='Invoking audit by filesystem path produced ModuleNotFoundError r2 before data/model work. Resolved by module -m invocation; frozen code/registration unchanged.',successful_entrypoints=['python -B -m experiments.r2_seat_failure_diagnosis_v1.independent_audit.preparation_review','python -B -m experiments.r2_seat_failure_diagnosis_v1.independent_audit.execution_review'],extra_model_calls=0))
print(json.dumps({k:value[k] for k in ('status','new_rollouts','generator_sampling_returned','new_D0_returned','all_RAW_before_first_D0','protected_hashes_unchanged','audit_model_forwards')}))
print(json.dumps(overlaps))