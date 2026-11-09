"""Prepare/conditionally execute two source-target Bench D0 identity checks.

`prepare` imports no neural framework and performs no model/GEO/scoring calls.
`run` is gated by the root's completed full-panel acceptance decision. It uses
the existing Bench EMA loader and predict_proposal verbatim, then the original
C restoration / GEO_C / source-winding export. No training or new noise.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'r2/src'))
OUT = ROOT / 'experiments/r2_bench_geo_only_acceptance_v1'
DEST = OUT / 'identity'
OLD = ROOT / 'experiments/r2_bench_backend_counterfactual_v1'
D0 = ROOT / 'checkpoints/official_bench_denoiser/0021000.pt'
D0_SHA = '67765c728c503b61c21624c05997ea026c9de486d1529052fafa134be3336b14'
D0_BYTES = 7289979546
SCALE = 0.95 * 0.3762
L = float(np.sqrt(3.) * 2 / .3762)


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve(value):
    s = str(value).replace('\\','/')
    if os.name == 'nt' and s.startswith('/mnt/e/'):
        s = 'E:/' + s[7:]
    elif os.name != 'nt' and s.startswith('E:/'):
        s = '/mnt/e/' + s[3:]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    return x


def write_new(path, value):
    path = resolve(path); value = clean(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value: raise RuntimeError('Immutable identity artifact differs: ' + str(path))
        return
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def npz_new(path, value):
    with resolve(path).open('xb') as f:
        np.savez_compressed(f,**value);f.flush();os.fsync(f.fileno())


def event(kind, **fields):
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST/'calls.jsonl').open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(clean(dict(event=kind,utc=utc(),**fields)),ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())


def verify(book):
    for path,h in book.items():
        if sha(path) != h: raise RuntimeError('Protected identity asset changed: '+str(path))


def prepare():
    path = DEST/'registration.json'
    if path.exists():
        r=read(path);verify(r['protected_inputs']);verify(r['code_hashes']);return r
    manifest=read(OLD/'manifest.json');complete=read(OLD/'core_complete.json')
    if complete['status']!='COMPLETE' or complete['SOURCE_GEO']!=2: raise ValueError('Source GEO prerequisites incomplete')
    source_tasks=[t for t in manifest['tasks'] if t['kind']=='SOURCE_GEO']
    assert len(source_tasks)==2 and {t['parent_id'][:6] for t in source_tasks}=={'a8e987','27eb79'}
    protected={str(OLD/'manifest.json'):sha(OLD/'manifest.json'),str(OLD/'core_complete.json'):sha(OLD/'core_complete.json')}
    rows=[]
    for t in sorted(source_tasks,key=lambda x:x['condition_id']):
        original_dir=resolve(t['output_dir']);receipt=read(original_dir/'receipt.json');verify(receipt['files'])
        raw=arrays(t['new_raw_path']);case=arrays(t['case_path']);source=arrays(t['source_condition_path'])
        n,k=t['N'],t['K'];ids=np.r_[case['known_ids'],case['free_ids']]
        assert raw['raw'].dtype==np.float32 and raw['raw'].shape==(n,3,3) and np.isfinite(raw['raw']).all()
        assert np.array_equal(raw['raw'],case['full_target'][ids])
        assert np.array_equal(np.sort(ids),np.arange(n))
        assert np.array_equal(raw['C_vertices'][raw['C_faces_model']],raw['raw'][:k])
        assert np.array_equal(raw['C_vertices'][raw['C_faces_original']],source['C_original'])
        assert np.array_equal(raw['vertices'][raw['faces']],raw['raw'])
        assert not raw['raw_seam_slot_mask'].any() and (raw['raw_anchor_ids'][k:]<0).all()
        assert len(np.unique(raw['faces'][k:]))==3*(n-k)
        for p in (t['new_raw_path'],t['case_path'],t['source_condition_path'],original_dir/'receipt.json',original_dir/'post.npz',original_dir/'geometry.json'):
            protected[str(resolve(p))]=sha(p)
        rows.append(dict(parent_id=t['parent_id'],condition_id=t['condition_id'],N=n,K=k,
            raw_path=t['new_raw_path'],raw_sha256=sha(t['new_raw_path']),case_path=t['case_path'],
            source_condition_path=t['source_condition_path'],source_condition_sha256=sha(t['source_condition_path']),
            source_GEO_path=str(original_dir/'post.npz'),source_GEO_sha256=sha(original_dir/'post.npz'),
            query_descriptor=t['request']['query_descriptor'],
            output_dir=str(DEST/'outputs'/t['condition_id']),
            scope='Existing full source reference, C first then free soup; no generator or Gaussian sampling',
            source_known_limit='Prior source T-junction/compatibility findings retained; not a perfect-manifold certificate'))
    files=[Path(__file__),ROOT/'r2/pilots/bench_control_transfer_v1/postprocess.py',
           ROOT/'r2/src/meshflow_control/denoise_model.py',ROOT/'r2/pilots/seam_slot_c200_v1/postprocess.py',
           ROOT/'r2/src/meshflow_control/geo_c.py',ROOT/'r2/src/meshflow_control/geo_io.py',
           ROOT/'r2/src/meshflow_control/source_winding.py']
    assert D0.exists() and D0.stat().st_size==D0_BYTES
    r=dict(schema='BENCH_TWO_SOURCE_D0_IDENTITY_V1',status='PREPARED_NOT_EXECUTED',prepared_utc=utc(),
        gate_contract=dict(path=str(OUT/'acceptance_gate.json'),status='FAIL',run_source_identity=True,
                           meaning='Root concludes uniform GEO_ONLY is not accepted on complete frozen panel'),
        budget=dict(source_objects=2,D0_forwards=2,GEO_calls=2,orientation_solves=2,training=0,backwards=0,generator_forwards=0,new_noise=0,new_OT=0,score_calls=0),
        checkpoint=dict(path=str(D0),sha256=D0_SHA,bytes=D0_BYTES,branch='ema',parameters=455592963,
                        verification='Expected official Bench identity reused here; unchanged official loader revalidates SHA at actual gated load'),
        packing=dict(shape='N,3,3 FP32',source_order='C known_ids first, free_ids original order',extra_sort=False,padding=False,new_noise=False,
                     scale=SCALE,proposal='x=RAW*SCALE; residual=decode(pe_nerf(x),N); proposal=(residual.float()+x.float())/SCALE',
                     precision='FP32 parameters/residual addition/scaling/output; BF16 forward, math SDPA',
                     fixed_std_Bench=0.05,fixed_std_note='Identity-encoder checkpoint setting; predict_proposal adds no noise and does not call encoder posterior'),
        post_contract='Save unconstrained proposal first; restore original C; original GEO_C; source-C winding and fixed-C free orientation',
        raw_support_proxy=dict(min_Y='less than C_min_Y - 0.02L',centroid_Y='less than C_min_Y',L=L,meaning='Source below-C geometry proxy, not semantic support labels'),
        protected_inputs=protected,code_hashes={str(p):sha(p) for p in files},rows=rows,
        preparation_model_calls=0,preparation_GEO_calls=0)
    write_new(path,r)
    return r


def resource_snapshot():
    """Reuse only this new task's recorder; never write an old sealed run."""
    from . import resource_check
    before=set((OUT/'resources').glob('snapshot_*.json')) if (OUT/'resources').exists() else set()
    resource_check.run()
    added=set((OUT/'resources').glob('snapshot_*.json'))-before
    if len(added)!=1: raise RuntimeError('Resource recorder did not write exactly one new snapshot')
    path=added.pop()
    return path,read(path)


def gpu_guard(info):
    lines=info['gpu']['stdout'].strip().splitlines()
    if info['gpu']['returncode']!=0 or len(lines)<2: raise RuntimeError('Unavailable current GPU resource measurement')
    values=[x.strip() for x in lines[1].split(',')]
    total=float(values[1].split()[0]);free=float(values[3].split()[0]);headroom=max(2048.,.1*total)
    estimated_peak=3500.
    if free < headroom+1.2*estimated_peak:
        raise RuntimeError('Insufficient measured GPU headroom; no D0 model loaded')
    return dict(workers=1,total_MiB=total,free_MiB=free,headroom_MiB=headroom,
                estimated_screening_D0_process_peak_MiB=estimated_peak,
                observed_comparable_D0_process_peak_MiB=None,
                estimate_evidence='No sourced comparable process peak; 3500 MiB is only a conservative admission estimate, not a measurement or a weight-size inference',
                incremental_slots_floor=None,
                scope='Single already-authorized two-source D0 task; measure this real task, no extra probe and no concurrency expansion')


def quantities(soup):
    t=np.asarray(soup,np.float64)
    lengths=np.stack([np.linalg.norm(t[:,(i+1)%3]-t[:,i],axis=1) for i in range(3)],1)
    area=.5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    height=np.divide(2*area,lengths.max(1),out=np.zeros_like(area),where=lengths.max(1)>0)
    return area,lengths.min(1),height


def save_geometry(row,regsha):
    from r2.pilots.seam_slot_c200_v1.postprocess import geometry_from_proposal
    from meshflow_control.geo_io import write_indexed_ply
    dest=resolve(row['output_dir']);done=dest/'complete.json'
    if done.exists():
        old=read(done);verify(old['files']);return old
    if (dest/'geometry_attempt.json').exists(): raise RuntimeError('Incomplete identity GEO attempt; explicit review required')
    raw=arrays(row['raw_path']);source=arrays(row['source_condition_path']);proposal=arrays(dest/'proposal.npz')['official_unconstrained_proposal']
    pr=read(dest/'proposal_receipt.json');assert pr['D0_forwards']==1 and pr['proposal_sha256']==sha(dest/'proposal.npz')
    write_new(dest/'geometry_attempt.json',dict(utc=utc(),GEO_budget=1,registration_sha256=regsha))
    event('GEO_STARTED',condition_id=row['condition_id'],budget=1)
    pre,canonical,post,geometry,oriented,anchors=geometry_from_proposal(raw,proposal,source)
    event('GEO_RETURNED',condition_id=row['condition_id'],GEO_calls=1)
    n,k=row['N'],row['K'];sf,sc=post['source_face_ids'],post['source_corner_ids']
    post.update(identity_domain=np.asarray('SOURCE_IDENTITY_D0_REFERENCE_NOT_GENERATED'),
                source_target_face_ids=raw['source_target_face_ids'],
                retained_source_target_face_ids=raw['source_target_face_ids'][sf],
                retained_source_target_vertex_ids=raw['source_target_vertex_ids'][sf[:,None],sc],
                retained_source_original_corner_ids=raw['source_target_original_corner_ids'][sf[:,None],sc])
    npz_new(dest/'pre_geo.npz',dict(output=pre,C=raw['raw'][:k],known_faces=raw['known_faces'],raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
    npz_new(dest/'post_canonical.npz',canonical);npz_new(dest/'post.npz',post)
    write_indexed_ply(dest/'post.ply',post['vertices'],post['faces'])
    write_new(dest/'geometry.json',dict(stats=geometry['stats'],face_records=geometry['face_records']))
    write_new(dest/'anchor_identity.json',anchors);write_new(dest/'source_restoration.json',oriented['restoration_summary'])
    orientation=oriented['orientation_result']
    write_new(dest/'orientation.json',dict(summary=orientation['summary'],components=orientation['component_rows'],edges=orientation['edge_rows']))
    cmin=float(raw['raw'][:k,:,1].min());raw64=raw['raw'].astype(np.float64)
    selected=(np.arange(n)>=k)&(raw64[:,:,1].min(1)<cmin-.02*L)&(raw64[:,:,1].mean(1)<cmin)
    mapping=canonical['corner_target_input_ids'];merged=pre.reshape(-1,3)[mapping]
    assert np.array_equal(merged[canonical['source_face_ids']],canonical['output'])
    stages=[('SOURCE_RAW',raw['raw'],np.arange(n)),('D0_PROPOSAL',proposal,np.arange(n)),
            ('RESTORED_C',pre,np.arange(n)),('MERGED_BEFORE_CLEANUP',merged,np.arange(n)),
            ('POST',canonical['output'],canonical['source_face_ids'])]
    records=[];summary=[];source_area=quantities(raw['raw'])[0]
    for name,soup,ids in stages:
        a,s,h=quantities(soup);free=ids>=k;region=selected[ids]
        for i,slot in enumerate(ids):
            records.append(dict(stage=name,input_slot_id=int(slot),source_face_id=int(raw['source_target_face_ids'][slot]),is_C=bool(slot<k),raw_below_C_proxy=bool(selected[slot]),
                                area=float(a[i]),shortest_edge=float(s[i]),minimum_altitude=float(h[i]),kept=bool(canonical['source_face_kept'][slot]),reason=str(canonical['source_face_reason'][slot])))
        summary.append(dict(stage=name,present_free_faces=int(free.sum()),free_area=float(a[free].sum()),
                            raw_below_C_proxy_faces=int(region.sum()),raw_below_C_proxy_area=float(a[region].sum()),
                            proxy_area_ratio=float(a[region].sum()/source_area[selected].sum()) if selected.any() and source_area[selected].sum()>0 else None))
    with (dest/'per_face.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write_new(dest/'summary.json',dict(stages=summary,proxy_slots=np.flatnonzero(selected).tolist(),
        free_proposal_equals_restored_PRE=bool(np.array_equal(proposal[k:],pre[k:])),
        source_to_proposal_free_coordinate_RMS=float(np.sqrt(np.mean((proposal[k:].astype(np.float64)-raw64[k:])**2))),
        source_to_proposal_free_corner_displacement_max=float(np.linalg.norm(proposal[k:].astype(np.float64)-raw64[k:],axis=2).max()),
        limits='Identity reference is not a perfect manifold or unique open-generation target; areas are triangle sums, not surface unions'))
    result=dict(status='COMPLETE',condition_id=row['condition_id'],registration_sha256=regsha,D0_forwards=1,GEO_calls=1,orientation_solves=1,
                generator_forwards=0,backwards=0,updates=0,source_labels_used_for_generation=False,
                files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
    write_new(done,result);return result


def run(gate_path):
    reg=prepare();gate_path=resolve(gate_path);gate=read(gate_path)
    if gate.get('status')!='FAIL' or gate.get('run_source_identity') is not True:
        raise RuntimeError('Root full-panel failure gate is not open; no model or GEO is authorized by this entry')
    done=DEST/'COMPLETE.json'
    if done.exists():
        previous=read(done);verify(previous['files']);return previous
    verify(reg['protected_inputs']);verify(reg['code_hashes'])
    regsha=sha(DEST/'registration.json');gatehash=sha(gate_path)
    calls_file=DEST/'calls.jsonl'
    events=[json.loads(line) for line in calls_file.read_text(encoding='utf-8').splitlines()] if calls_file.exists() else []
    started={e['condition_id'] for e in events if e['event']=='D0_STARTED'}
    pending=[]
    for row in reg['rows']:
        dest=resolve(row['output_dir'])
        if (dest/'proposal_receipt.json').exists():
            pr=read(dest/'proposal_receipt.json')
            if pr['proposal_sha256']!=sha(dest/'proposal.npz') or pr['raw_sha256']!=row['raw_sha256']:
                raise RuntimeError('Saved identity proposal changed')
        else:
            if row['condition_id'] in started: raise RuntimeError('Started but incomplete D0; no silent model replay')
            pending.append(row)
    attempt=DEST/('attempt_'+str(time.time_ns()));attempt.mkdir()
    write_new(attempt/'registration.json',dict(gate_path=str(gate_path),gate_sha256=gatehash,registration_sha256=regsha,
        pending_D0=len(pending),source_objects=2,automatic_retry=False))
    model=None;calls=0;tick=time.perf_counter()
    try:
        if pending:
            resource_path,resource=resource_snapshot();guard=gpu_guard(resource)
            write_new(attempt/'dispatch.json',dict(**guard,resources_path=str(resource_path),resources_sha256=sha(resource_path)))
            import torch
            from r2.pilots.bench_control_transfer_v1.postprocess import load_bench_denoiser
            from meshflow_control.denoise_model import predict_proposal
            torch.set_num_threads(1)
            model,identity=load_bench_denoiser(D0)
            write_new(attempt/'model_identity.json',identity)
            resource_snapshot()
            versions=[p._version for p in model.parameters()]
            for row in pending:
                dest=resolve(row['output_dir']);dest.mkdir(parents=True,exist_ok=True)
                raw=arrays(row['raw_path']);assert sha(row['raw_path'])==row['raw_sha256']
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();forward_tick=time.perf_counter()
                event('D0_STARTED',condition_id=row['condition_id'],budget=1,raw_sha256=row['raw_sha256'])
                proposal=predict_proposal(model,raw['raw']);torch.cuda.synchronize();calls+=1
                event('D0_RETURNED',condition_id=row['condition_id'],calls=1)
                npz_new(dest/'proposal.npz',dict(official_unconstrained_proposal=proposal))
                write_new(dest/'proposal_receipt.json',dict(status='COMPLETE',condition_id=row['condition_id'],D0_forwards=1,
                    raw_sha256=row['raw_sha256'],proposal_sha256=sha(dest/'proposal.npz'),checkpoint_sha256=D0_SHA,
                    seconds=time.perf_counter()-forward_tick,allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                    reserved_peak_bytes=torch.cuda.max_memory_reserved(),peak_scope='Single real source D0 forward; resident model included; allocator is not full process VRAM'))
                resource_snapshot()
            assert versions==[p._version for p in model.parameters()]
            del model;model=None;torch.cuda.empty_cache()
        # All two proposals are saved before CPU geometry. Reuse saved proposal
        # after an engineering interruption; never re-execute a counted forward.
        completed=[save_geometry(row,regsha) for row in reg['rows']]
        verify(reg['protected_inputs']);verify(reg['code_hashes'])
        result=dict(status='COMPLETE',source_objects=2,D0_forwards_total=2,new_D0_forwards_this_invocation=calls,
            GEO_calls_total=2,orientation_solves_total=2,training=0,backwards=0,generator_forwards=0,new_noise=0,score_calls=0,
            seconds=time.perf_counter()-tick,gate_path=str(gate_path),gate_sha256=gatehash,
            files={str(resolve(row['output_dir'])/'complete.json'):sha(resolve(row['output_dir'])/'complete.json') for row in reg['rows']})
        write_new(done,result);return result
    except BaseException as exc:
        write_new(attempt/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),returned_D0=calls,automatic_retry=False))
        raise
    finally:
        if model is not None:
            import torch
            del model;torch.cuda.empty_cache()
        resource_snapshot()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--gate',default=str(OUT/'acceptance_gate.json'))
    a=p.parse_args();value=prepare() if a.action=='prepare' else run(a.gate)
    print(json.dumps({k:v for k,v in value.items() if k not in ('rows','protected_inputs','code_hashes','files')},ensure_ascii=False))
