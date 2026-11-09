"""Summarize saved receipts and scored rows; never load a neural model."""
from __future__ import annotations
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_chair_back_transfer_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
    manifest=read(OUT/'eval_manifest.json');qs=manifest['requests']
    calls=[];raw_seconds=[];d0_seconds=[];raw_peak=[];d0_peak=[]
    raw_ranges=[]
    for q in qs:
        b=OUT/'evaluation/S2000'/q['request_id']
        r=read(b/'raw/receipt.json');p=read(b/'post/proposal_receipt.json')
        raw_seconds.append(r['audit']['seconds']);d0_seconds.append(p['seconds'])
        raw_peak.append(r['audit']['peak_reserved_bytes']);d0_peak.append(p['peak_reserved_bytes'])
        ev=[json.loads(x) for x in (b/'raw/calls.jsonl').read_text().splitlines()]
        raw_ranges.append((min(x['unix'] for x in ev),max(x['unix'] for x in ev),q['request_id']))
        calls.extend(dict(request_id=q['request_id'],phase='RAW',**e) for e in ev)
        calls.extend(dict(request_id=q['request_id'],phase='D0',**json.loads(e)) for e in (b/'post/calls.jsonl').read_text().splitlines())
    for d in (OUT/'evaluation/S2000/layouts').iterdir():
        calls.extend(dict(condition_id=d.name,phase='HEAD',**json.loads(e)) for e in (d/'calls.jsonl').read_text().splitlines())
    snapshots=[read(p) for p in (OUT/'dispatch').glob('*/resources*.json')]
    used=[];free=[];utils=[]
    for s in snapshots:
        parts=s['gpu']['stdout'].splitlines()[0].split(',')
        used.append(float(parts[2]));free.append(float(parts[3]));utils.append(float(parts[4]))
    counter=defaultdict(int)
    for e in calls:counter[e['event']]+=1
    overlap=[(a[2],b[2],min(a[1],b[1])-max(a[0],b[0])) for i,a in enumerate(raw_ranges) for b in raw_ranges[i+1:] if min(a[1],b[1])>max(a[0],b[0])]
    stages=[dict(attempt=p.parent.name,**read(p)) for p in (OUT/'dispatch').glob('*/complete.json')]
    ledger=dict(status='COMPLETE_GPU',generator_calls_returned=counter['FORWARD_RETURNED'],
        head_calls_returned=counter['HEAD_FORWARD_RETURNED'],D0_calls_returned=counter['D0_FORWARD_RETURNED'],
        training_updates=0,backwards=0,expected=dict(generator=800,head=8,D0=16),event_counts=dict(counter),
        execution_attempts=stages,failed_worker_attempts=len(list((OUT/'dispatch').glob('*/failure.json'))),
        raw_rollout_seconds_sum=sum(raw_seconds),raw_rollout_seconds_range=[min(raw_seconds),max(raw_seconds)],
        D0_forward_seconds_sum=sum(d0_seconds),raw_max_reserved_MiB=max(raw_peak)/2**20,D0_max_reserved_MiB=max(d0_peak)/2**20,
        sampled_total_GPU_used_MiB_max=max(used),sampled_GPU_free_MiB_min=min(free),sampled_compute_utilization_percent_range=[min(utils),max(utils)],
        measurement_scope='Per-output allocator reset; sparse nvidia-smi whole-device snapshots include unrelated display use, not a continuous full-process peak. WSL process VRAM returned N/A.',
        raw_workers=2,D0_workers=1,raw_interval_overlap=overlap,
        scheduling='Start first authorized 8-request partition. After representative output GPU20%, used3148MiB/free12832MiB, permit second partition using conservative2600MiB estimate and2048MiB headroom. Startup latency may eliminate actual overlap. D0 reuses one official model for16 single forwards; CPU scoring/post uses2workers.',
        cpu_cores=snapshots[0]['cpu_cores'],ram_available_bytes_min=min(x['ram_available_bytes'] for x in snapshots),
        timing_warning='Worker wall times include loading, hashes and saves; do not claim isolated-compute or parallel speedup from them.')
    save(OUT/'execution_ledger.json',ledger)
    fields=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
       'C_F_has_valid_edge','C_F_valid_edge_count','C_F_valid_opposite_winding_edge_count',
       'free_only_reached_bulk_area_fraction','bulk_reach_ge_0_30_descriptive','retained_seam_faces',
       'legal_seam_bulk_edge_count','free_minimum_angle_lt5_fraction','free_q_gt10_fraction',
       'hinge_diag_isolated_1face_unfiltered_edge_components','hinge_diag_isolated_1face_edge_components',
       'nonmanifold_edges','deleted_free_faces','bridge_face_count','fixed_parent_FF_illegal_affected_fraction',
       'fixed_parent_CF_illegal_affected_fraction','ConflictCov_C_cross','ConflictCov_C_dup']
    allrows=[]
    for q in qs:
        for stage in ('RAW','BASE_POST','HINGE_OPP015','ALPHA','FINAL_POST'):
            p=OUT/'evaluation/scores/S2000'/q['request_id']/stage/'row.json'
            r=read(p)
            allrows.append({k:r.get(k) for k in ['request_id','condition_id','parent_id','split','stage','N','K',*fields]})
    digest=[]
    for split in ('FAMILIAR','DEV','ALL'):
        for stage in ('RAW','BASE_POST','HINGE_OPP015','ALPHA','FINAL_POST'):
            rr=[r for r in allrows if r['stage']==stage and (split=='ALL' or r['split']==split)]
            d=dict(split=split,stage=stage,outputs=len(rr))
            for f in fields:
                v=[r[f] for r in rr if isinstance(r[f],(int,float))]
                d[f]=sum(v)/len(v) if v else None
            d['legal_interface_outputs']=sum(r['C_F_has_valid_edge'] for r in rr)
            d['bulk_ge30_outputs']=sum(r['bulk_reach_ge_0_30_descriptive'] for r in rr)
            d['bulk_nonzero_outputs']=sum(r['free_only_reached_bulk_area_fraction']>0 for r in rr)
            digest.append(d)
    save(OUT/'metrics_digest.json',digest)
    with (OUT/'key_per_output.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    print(json.dumps(ledger,indent=2))
    print(json.dumps([r for r in digest if r['stage'] in ('BASE_POST','FINAL_POST')],indent=2))
if __name__=='__main__':main()
