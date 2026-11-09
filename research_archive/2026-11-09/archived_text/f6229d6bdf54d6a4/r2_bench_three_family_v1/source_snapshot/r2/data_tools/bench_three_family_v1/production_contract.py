"""Bounded CPU-only production-head layout and exact-source packing audit.

No generator, denoiser, backward, OT, calibration or source-derived input mask.
The only neural call is one explicitly requested frozen head call per eligible
open C. Layout is committed before the supervision case is read.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time, traceback
from pathlib import Path
from . import common as C
from r2.data_tools.bench_deployment_tasks_v1 import layout_contract as A
from r2.pilots.bench_parts_convergence_v1 import data as D
import numpy as np

OUT=C.OUT/'production_contract'
SOURCE=C.ROOT/'experiments/r2_bench_parts_convergence_v1/training/SEAM/generator_step15000.pt'
SOURCE_SHA='01e2e1834eb79834cfc47f88bd7080ff8cf166c6f243f890d36522be9c4d8023'
INPUT_KEYS={'C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations','N','K'}
REQUIRED_CASE={'full_target','source_faces','source_face_vertex_ids','known_ids','free_ids','N','K',
    'C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','edge_owner_face',
    'edge_source_triple','source_supported_labels','source_alias_labels','occupied_labels'}


def file_sha(path):
    h=hashlib.sha256()
    with C.resolve(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        x=v.detach().cpu().contiguous().numpy()
        h.update(k.encode());h.update(A.ahash(x).encode())
    return h.hexdigest()


def event(dest,kind,**kw):
    with (dest/'ledger.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(dict(time_unix=time.time(),event=kind,**kw),ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())


def code_hashes():
    return {str(p):file_sha(p) for p in (Path(__file__),Path(A.__file__),Path(D.__file__),
        C.ROOT/'r2/src/meshflow_control/seam/slot_data.py',
        C.ROOT/'r2/src/meshflow_control/seam/occupancy.py')}


def freeze_head():
    """Map old checkpoint metadata; copy only four tiny head tensors, no forward."""
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'old_s15000_head_only.pt';receipt=OUT/'HEAD_PROTOCOL.json'
    if receipt.exists():
        p=C.read(receipt);assert file_sha(target)==p['head_only_sha256']
        assert file_sha(SOURCE)==p['source_checkpoint_sha256']
        return p
    if target.exists() or (OUT/'head_extract_attempt.json').exists():
        raise RuntimeError('Unresolved head extraction; no silent overwrite/retry')
    assert file_sha(SOURCE)==SOURCE_SHA
    C.save(OUT/'head_extract_attempt.json',dict(source=str(SOURCE),sha256=SOURCE_SHA,forwards=0))
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    payload=torch.load(SOURCE,map_location='cpu',weights_only=True,mmap=True)
    head={k:v.detach().clone().cpu() for k,v in payload['head_state'].items()}
    del payload
    expected={'net.0.weight':(32,24),'net.0.bias':(32,),'net.2.weight':(1,32),'net.2.bias':(1,)}
    assert set(head)==set(expected)
    assert all(tuple(head[k].shape)==shape and head[k].dtype==torch.float32 for k,shape in expected.items())
    assert sum(v.numel() for v in head.values())==833
    assert not torch.cuda.is_initialized()
    with target.open('xb') as f:torch.save(dict(head_state=head,source_sha256=SOURCE_SHA),f)
    result=dict(status='FROZEN_BEFORE_HEAD_CALLS',source_checkpoint=str(SOURCE),source_checkpoint_sha256=SOURCE_SHA,
        head_only_path=str(target),head_only_sha256=file_sha(target),head_state_sha256=state_sha(head),parameters=833,
        architecture='OccupancyHead 24->32 SiLU->1',feature_schema='C_edge24_world_chair_v1',
        feature_function='r2.pilots.bench_parts_convergence_v1.data.edge_features',
        normalization='Only existing 24-feature model-scale/L construction; no new centering/scaler/augmentation',
        threshold_logit=0.0,equality_is_active=True,production_reference='OLD_S15000_HEAD, reused by RELATION/USER evaluation',
        original_precision='GPU BF16 autocast, FP32 parameters',execution_precision='CPU BF16 autocast, FP32 parameters, eval/no_grad',
        device_caveat='Same weights/features/threshold; no claim of CPU/GPU bitwise equality or threshold-stability certification',
        maximum_calls=6,calls_per_eligible_open_C=1,closed_C_calls=0,invalid_N_calls=0,
        no_source_labels_at_inference=True,layout_before_Y=True,retry_policy='Any unfinished forward attempt blocks replay',
        neural_forwards_so_far=0,generator_loaded=False,D0_loaded=False,
        checkpoint_access='CPU mmap metadata; only head_state tensors copied; generator tensor storage not materialized',
        main_protocol_sha256=file_sha(C.OUT/'PROTOCOL.json'),code_sha256=code_hashes(),
        production_sampling_source=str(C.ROOT/'r2/pilots/bench_parts_convergence_v1/sample.py'),
        production_sampling_sha256=file_sha(C.ROOT/'r2/pilots/bench_parts_convergence_v1/sample.py'))
    C.save(receipt,result);return result


def checked_input(a):
    assert set(a)<=INPUT_KEYS and {'C_vertices','C_faces_original','C_faces_model','N','K'}<=set(a),'C-only whitelist'
    n,k=int(a['N']),int(a['K'])
    if not 128<=n<=256:return None,dict(status='BLOCKED_N_OUTSIDE_128_256',N=n,K=k)
    assert len(a['C_faces_original'])==len(a['C_faces_model'])==k
    core={x:a[x] for x in ('C_vertices','C_faces_original','C_faces_model')};core['total_N']=n
    # Existing guard handles K, edge capacity, identity and winding; not relaxed.
    edges,features=D.edge_features(core['C_vertices'],core['C_faces_original'],n)
    empty=D.inference_layout(**core,occupied=np.zeros(len(edges),bool))
    if 'C' in a:assert np.array_equal(a['C'],a['C_vertices'][a['C_faces_model']])
    return (core,edges,features),dict(status='C_ONLY_INPUT_VALID',N=n,K=k,boundary_edges=len(edges))


def make_layout(inp,dest,protocol):
    """One attempt is durable before any call; a saved layout permits CPU resume."""
    identity=dict(input_sha256=file_sha(inp),head_protocol_sha256=file_sha(OUT/'HEAD_PROTOCOL.json'))
    receipt=dest/'layout_receipt.json'
    if receipt.exists():
        r=C.read(receipt);assert r['identity']==identity
        assert file_sha(dest/'layout.npz')==r['layout_sha256']
        return C.load(dest/'layout.npz'),r
    if (dest/'layout_attempt.json').exists():raise RuntimeError('Unresolved layout attempt; automatic replay prohibited')
    a=C.load(inp);prepared,check=checked_input(a)
    if prepared is None:return None,check
    core,edges,features=prepared
    C.save(dest/'layout_attempt.json',dict(identity=identity,expected_head_calls=int(bool(len(edges))),Y_read=False))
    if len(edges):
        # No generator/model adapter is imported or instantiated here.
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        from meshflow_control.seam.occupancy import OccupancyHead
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        payload=torch.load(C.resolve(protocol['head_only_path']),map_location='cpu',weights_only=True)
        head=OccupancyHead(24).eval().requires_grad_(False)
        head.load_state_dict(payload['head_state'],strict=True)
        assert state_sha(head.state_dict())==protocol['head_state_sha256'] and not torch.cuda.is_initialized()
        event(dest,'HEAD_FORWARD_STARTED',budget=1,device='cpu',Y_read=False)
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            logits=head(torch.from_numpy(features),torch.ones(len(edges),dtype=torch.bool)).float().numpy()
        event(dest,'HEAD_FORWARD_RETURNED',count=1)
        assert np.isfinite(logits).all() and logits.shape==(len(edges),)
        assert state_sha(head.state_dict())==protocol['head_state_sha256'] and not torch.cuda.is_initialized()
    else:logits=np.empty(0,np.float32)
    lay=D.inference_layout(**core,occupied=logits>=0)
    C.npz(dest/'layout.npz',{**lay,'occupancy_logits':logits})
    r=dict(status='FROZEN_C_ONLY_LAYOUT',identity=identity,layout_sha256=file_sha(dest/'layout.npz'),
        head_forward_calls=int(bool(len(edges))),Y_read=False,active=int(lay['occupied'].sum()),
        boundary_edges=len(edges),minimum_abs_logit=float(np.min(np.abs(logits))) if len(edges) else None,
        all_off_is_not_connection_success=bool(len(edges) and not lay['occupied'].any()))
    C.save(receipt,r);event(dest,'LAYOUT_FROZEN_BEFORE_Y',sha256=r['layout_sha256'])
    return C.load(dest/'layout.npz'),r


def verify_case(case,inp):
    assert REQUIRED_CASE<=set(case),sorted(REQUIRED_CASE-set(case))
    n,k=int(case['N']),int(case['K']);assert (n,k)==(int(inp['N']),int(inp['K']))
    for key in ('C_vertices','C_faces_original','C_faces_model'):
        assert np.array_equal(case[key],inp[key]),key
    known,free=case['known_ids'],case['free_ids']
    assert len(known)==k and len(free)==n-k and np.array_equal(np.sort(np.r_[known,free]),np.arange(n))
    assert np.array_equal(case['full_target'][known],inp['C_vertices'][inp['C_faces_model']])
    original,canonical=case['source_faces'],case['source_face_vertex_ids']
    assert original.shape==canonical.shape==(n,3) and np.array_equal(np.sort(original,1),np.sort(canonical,1))
    assert all(len(set(map(int,f)))==3 for f in original)
    b=len(D.boundary_from_C(inp['C_vertices'],inp['C_faces_original']))
    for key in ('source_supported_labels','source_alias_labels','occupied_labels'):
        assert case[key].dtype==np.bool_ and case[key].shape==(b,),key
    assert case['edge_owner_face'].shape==(b,) and case['edge_source_triple'].shape==(b,3)
    assert not np.any(case['source_alias_labels'] & case['occupied_labels'])
    assert np.array_equal(case['source_supported_labels'],case['source_alias_labels']|case['occupied_labels'])
    return n,k


def packing_audit(case,inp,occupied):
    n,k=verify_case(case,inp)
    pack,detail=A.compatible_endpoint(case,occupied)
    detail['all_off_is_not_connection_success']=bool(len(occupied) and not np.any(occupied))
    detail['source_supervision_read_after_layout']=True
    edges=D.boundary_from_C(inp['C_vertices'],inp['C_faces_original'])
    detail['active_edge_source_diagnostics']=[]
    for ei in np.flatnonzero(occupied):
        u,v=edges[ei,1:];su,sv=case['C_source_vertex_ids'][[u,v]]
        candidates=[int(fi) for fi in case['free_ids'] if su in case['source_face_vertex_ids'][fi] and sv in case['source_face_vertex_ids'][fi]]
        detail['active_edge_source_diagnostics'].append(dict(edge_id=int(ei),C_vertex_ids=[int(u),int(v)],
            source_endpoint_ids=[int(su),int(sv)],source_face=int(case['edge_owner_face'][ei]),
            source_legal_supported=bool(case['source_supported_labels'][ei]),source_alias=bool(case['source_alias_labels'][ei]),
            broader_source_incidence_face_ids=candidates))
    if pack is None:
        detail['failure_scope']='Original legal-source-owner packing fails; broader incidence candidates are diagnostic only, never repair the predicted mask.'
        return pack,detail,[]
    rows=[];restored=np.empty_like(case['full_target'])
    for slot,fi in enumerate(pack['source_face_ids']):
        cp=pack['source_corner_permutations'][slot]
        vid=case['source_face_vertex_ids'][fi,cp];orig=case['source_faces'][fi]
        source_restore=np.asarray([int(np.flatnonzero(vid==v).item()) for v in orig])
        can_to_orig=np.asarray([int(np.flatnonzero(case['source_face_vertex_ids'][fi]==v).item()) for v in orig])
        assert np.array_equal(pack['x1'][slot,source_restore],case['full_target'][fi,can_to_orig])
        restored[fi,cp]=pack['x1'][slot]
        rows.append(dict(slot=slot,source_face_id=int(fi),canonical_corner_permutation=cp.tolist(),
            source_winding_restore_permutation=source_restore.tolist(),source_vertex_ids=orig.tolist(),
            exact_source_winding_and_coordinates=True))
    assert np.array_equal(restored,case['full_target']) and len(set(pack['source_face_ids'].tolist()))==n
    detail.update(source_winding_roundtrip=True,face_count=n,unique_source_faces=n)
    return pack,detail,rows


def run(manifest_path):
    protocol=C.read(OUT/'HEAD_PROTOCOL.json')
    assert file_sha(protocol['head_only_path'])==protocol['head_only_sha256']
    for p,h in protocol['code_sha256'].items():assert file_sha(p)==h,p
    main=C.read(C.OUT/'PROTOCOL.json');manifest=C.read(manifest_path);rows=manifest['rows']
    allowed={p['parent_id'] for p in main['parents']}
    assert 0<len(rows)<=6 and len({r['condition_id'] for r in rows})==len(rows)
    assert {r['parent_id'] for r in rows}<=allowed
    C.save(OUT/'RUN_PROTOCOL.json',dict(manifest=str(C.resolve(manifest_path)),manifest_sha256=file_sha(manifest_path),
        HEAD_PROTOCOL_sha256=file_sha(OUT/'HEAD_PROTOCOL.json'),max_head_calls=len(rows),generator_calls=0,D0_calls=0,OT_calls=0))
    results=[]
    for row in rows:
        cid=row['condition_id'];assert Path(cid).name==cid and cid not in ('.','..')
        dest=OUT/'conditions'/cid;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'failure.json').exists():raise RuntimeError('Recorded failed attempt requires explicit intervention: '+cid)
        if (dest/'completion.json').exists():
            r=C.read(dest/'completion.json')
            for p,h in r['files'].items():assert file_sha(p)==h,p
            results.append(r);continue
        if not row.get('eligible_for_production_contract',False):
            r=dict(condition_id=cid,status='SKIP_NOT_GEOMETRY_AND_VISUAL_APPROVED',head_forward_calls=0,
                reasons=row.get('hold_reasons',[]),files={})
            C.save(dest/'completion.json',r);results.append(r);continue
        try:
            assert file_sha(row['input_path'])==row['input_sha256']
            lay,lr=make_layout(row['input_path'],dest,protocol)
            if lay is None:
                r=dict(condition_id=cid,head_forward_calls=0,files={},**lr)
                C.save(dest/'completion.json',r);results.append(r);continue
            # This is the first access to Y / source-owner labels in this condition.
            event(dest,'SUPERVISION_READ_STARTED_AFTER_LAYOUT')
            assert file_sha(row['case_path'])==row['case_sha256']
            case=C.load(row['case_path']);inp=C.load(row['input_path'])
            pack,detail,face_rows=packing_audit(case,inp,lay['occupied'])
            if pack is not None:C.npz(dest/'endpoint.npz',pack)
            C.save(dest/'packing.json',detail);C.save(dest/'per_face_roundtrip.json',dict(rows=face_rows))
            r=dict(condition_id=cid,status=detail['status'],head_forward_calls=lr['head_forward_calls'],
                active=lr['active'],boundary_edges=lr['boundary_edges'],all_off_is_not_connection_success=detail['all_off_is_not_connection_success'],
                files={str(p):file_sha(p) for p in dest.iterdir() if p.is_file()},
                input_sha256=row['input_sha256'],case_sha256=row['case_sha256'])
            C.save(dest/'completion.json',r);results.append(r)
        except Exception as e:
            C.save(dest/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',error=repr(e),traceback=traceback.format_exc()))
            raise
    total=sum(r['head_forward_calls'] for r in results)
    assert total<=6
    result=dict(status='COMPLETE',conditions=len(results),total_head_forward_calls=total,generator_calls=0,D0_calls=0,OT_calls=0,
        backward_calls=0,rows=results,endpoint_success_is_not_generation_or_connection_quality=True)
    C.save(OUT/'COMPLETE.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze-head','run']);parser.add_argument('--manifest')
    arg=parser.parse_args()
    result=freeze_head() if arg.action=='freeze-head' else run(arg.manifest)
    print(json.dumps(result,ensure_ascii=False,indent=2))
