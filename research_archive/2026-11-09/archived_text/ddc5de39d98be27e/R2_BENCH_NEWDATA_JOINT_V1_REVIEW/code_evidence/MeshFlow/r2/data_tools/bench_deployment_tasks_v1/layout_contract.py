"""Read-only CPU audit of deployment layouts and exact endpoint representability.

No head/model/OT calls. A source owner layout is a diagnostic reference, never a
substitute for a missing production prediction. All generated files live under
the independent experiment directory; immutable old assets are hash checked.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'r2/src'))
import numpy as np
from r2.pilots.bench_parts_convergence_v1.data import inference_layout, boundary_from_C

OUT = ROOT / 'experiments/r2_bench_deployment_task_data_v1/layout_contract'
MANIFEST = ROOT / 'experiments/r2_bench_relation_training_v1/data/manifest.json'
PANEL = ROOT / 'experiments/r2_bench_relation_training_v1/eval_manifest.json'
SOURCES = [
    'r2/pilots/bench_parts_convergence_v1/train.py',
    'r2/pilots/bench_parts_convergence_v1/data.py',
    'r2/pilots/bench_user_v2_training_v1/data.py',
    'r2/pilots/bench_relation_training_v1/data.py',
    'r2/pilots/bench_relation_training_v1/training.py',
    'r2/pilots/bench_relation_training_v1/eval_plan.py',
    'r2/pilots/bench_relation_training_v1/sample.py',
    'r2/pilots/seam_abc_v1/slot_data_extension.py',
    'r2/src/meshflow_control/seam/slot_data.py',
    'r2/src/meshflow_control/seam/model_adapter.py',
    'r2/pilots/bench_trajectory_diagnostic_v1/run.py',
]


def resolve(p):
    s = str(p).replace('\\', '/')
    if os.name == 'nt':
        s = s.replace('/mnt/e/', 'E:/')
    else:
        s = s.replace('E:/', '/mnt/e/')
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def sha(p):
    return hashlib.sha256(resolve(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(resolve(p).read_text(encoding='utf-8-sig'))


def arrays(p):
    with np.load(resolve(p), allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def write_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(p, rows):
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fields);w.writeheader();w.writerows(rows)


def ahash(x):
    a=np.ascontiguousarray(x)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def input_signature(a):
    return hashlib.sha256('|'.join(ahash(np.asarray(a[k])) for k in
        ('C_vertices','C_faces_original','C_faces_model','N')).encode()).hexdigest()


def layout(case, occupied):
    return inference_layout(case['C_vertices'],case['C_faces_original'],
                            case['C_faces_model'],int(case['N']),np.asarray(occupied,bool))


def compatible_endpoint(case, occupied):
    """One-to-one original-Y packing for a supplied, already fixed C-only mask.

    Y determines only the supervision assignment. No point, source face or N is
    changed. A legal alias edge may own its source face instead of the historical
    lowest-ID owner, provided that face is not instantiated twice. Inactive true
    owner faces return to bulk. Active Null or duplicate owners fail explicitly.
    Bulk is in deterministic source-ID order, NOT OT and NOT a training sample.
    """
    lay=layout(case,occupied)
    occupied=np.asarray(occupied,bool)
    supported=np.asarray(case['source_supported_labels'],bool)
    owners=np.asarray(case['edge_owner_face'],np.int64)
    active=np.flatnonzero(occupied)
    unsupported=active[~supported[active]]
    duplicates={int(fi):np.flatnonzero(occupied & (owners==fi)).tolist()
        for fi in owners[active[supported[active]]] if int(np.sum(occupied & (owners==fi)))>1}
    detail=dict(active=int(occupied.sum()),active_unsupported_edges=unsupported.tolist(),
        duplicated_source_face_edges=duplicates,
        turned_off_true_owners=np.flatnonzero(case['occupied_labels'] & ~occupied).tolist(),
        activated_alias_edges=np.flatnonzero(case['source_alias_labels'] & occupied).tolist())
    if len(unsupported) or duplicates:
        return None,dict(status='BLOCKED_NONREPRESENTABLE_LEGAL_SOURCE_LAYOUT',**detail)
    n,k=int(case['N']),int(case['K']);full=case['full_target']
    ids=np.full(n,-1,np.int64);corner=np.full((n,3),-1,np.int64)
    ids[:k]=case['known_ids'];corner[:k]=np.arange(3)
    used=set()
    for ei in active:
        fi=int(owners[ei]);ids[k+ei]=fi;used.add(fi)
        corner[k+ei]=[int(np.flatnonzero(case['source_face_vertex_ids'][fi]==v).item())
                      for v in case['edge_source_triple'][ei]]
    bulk=np.asarray([fi for fi in case['free_ids'] if int(fi) not in used],np.int64)
    assert len(bulk)==len(lay['bulk_slots'])
    ids[lay['bulk_slots']]=bulk;corner[lay['bulk_slots']]=np.arange(3)
    target=full[ids[:,None],corner]
    fixed=lay['anchor_ids']>=0
    assert np.array_equal(np.sort(ids),np.arange(n))
    assert np.array_equal(target[fixed],lay['C_vertices'][lay['anchor_ids'][fixed]])
    recovered=np.empty_like(full)
    for slot,fi in enumerate(ids):recovered[fi,corner[slot]]=target[slot]
    assert np.array_equal(recovered,full)
    package=dict(anchor_ids=lay['anchor_ids'],known_mask=lay['known_mask'],
        free_corner_mask=lay['free_corner_mask'],valid_mask=lay['valid_mask'],
        source_face_ids=ids,source_corner_permutations=corner,x1=target,
        occupied=occupied,free_coordinate_count=np.asarray(lay['free_coordinate_count']))
    return package,dict(status='EXACT_TARGET_BIJECTION_PASS',**detail,
        original_N_preserved=True,source_faces_copied=0,source_faces_deleted=0,
        introduced_zero_area_faces=0,coordinates_bitwise_preserved=True)


def compare(case, predicted):
    original=np.asarray(case['occupied_labels'],bool);predicted=np.asarray(predicted,bool)
    src,dep=layout(case,original),layout(case,predicted)
    same=np.array_equal(original,predicted)
    # Sorted endpoint identities distinguish mere face-slot permutations from
    # a different anchored edge even when the seam count happens to agree.
    def pairs(l):return sorted(tuple(sorted(map(int,l['anchor_ids'][s,:2]))) for s in l['seam_slots'])
    category='IDENTICAL_VISIBLE_LAYOUT' if same else (
        'EQUIVALENT_FACE_PERMUTATION' if pairs(src)==pairs(dep) else 'ROLE_ANCHOR_DIFFERENCE')
    result=dict(layout_comparison=category,source_owner_count=int(original.sum()),
        predicted_active_count=int(predicted.sum()),changed_edges=int(np.sum(original!=predicted)),
        source_free_scalars=int(src['free_coordinate_count']),
        deployment_free_scalars=int(dep['free_coordinate_count']),
        changed_corner_roles=int(np.sum((src['anchor_ids']>=0)!=(dep['anchor_ids']>=0))),
        C_face_mask_identical=np.array_equal(src['known_mask'],dep['known_mask']),
        source_only_masks=False)
    pack,comp=compatible_endpoint(case,predicted)
    return result,pack,comp


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    complete=OUT/'COMPLETE.json'
    if complete.exists():
        c=read(complete)
        for p,h in c['input_hashes'].items():assert sha(p)==h,p
        for p,h in c['output_hashes'].items():assert sha(p)==h,p
        return dict(status='PASS_HASH_VERIFIED',summary=read(OUT/'summary.json'))
    import psutil
    inputs={str(MANIFEST):sha(MANIFEST),str(PANEL):sha(PANEL),str(Path(__file__)):sha(__file__)}
    for rel in SOURCES:
        p=ROOT/rel
        if p.exists():inputs[str(p)]=sha(p)
    write_json(OUT/'resources.json',dict(cpu_count=psutil.cpu_count(),
        available_RAM_bytes=psutil.virtual_memory().available,cpu_percent=psutil.cpu_percent(.1),
        workers=1,numeric_threads=1,GPU_calls=0))
    manifest=read(MANIFEST);train=manifest['train'];assert len(train)==91
    # Collect only existing production cache files. Exact C/N signature is required;
    # matching condition names or old source IDs alone never permits cache reuse.
    paths=[]
    for directory in sorted((ROOT/'experiments').glob('r2_bench*')):
        if directory.name=='r2_bench_deployment_task_data_v1':continue
        paths.extend(sorted(directory.glob('**/layout.npz')))
    unique={};cache_by_sig=collections.defaultdict(list);inventory=[]
    for p in paths:
        h=sha(p)
        if h not in unique:
            a=arrays(p)
            if not all(k in a for k in ('occupied','occupancy_logits','C_vertices','C_faces_original','C_faces_model','N')):continue
            reconstructed=layout(a,a['occupied'])
            for k in ('anchor_ids','known_mask','free_corner_mask','edge_records','seam_slots','bulk_slots'):
                assert np.array_equal(a[k],reconstructed[k]),(str(p),k)
            unique[h]=a;cache_by_sig[input_signature(a)].append(dict(path=str(p),sha256=h,arrays=a))
        inventory.append(dict(path=str(p),sha256=h,condition_id=p.parent.name,
            input_signature=input_signature(unique[h])))
    used_cache={};rows=[];edges=[];details=[];train_cache_count=0
    for split,records in [('CURRENT_TRAIN',train),('FIXED_EVAL',manifest['dev']+[r for r in read(ROOT/'experiments/r2_bench_parts_convergence_v1/data/manifest.json')['train'] if r['condition_id'] in {q['condition_id'] for q in read(PANEL)['requests'] if q['group']=='TRAIN'}])]:
        if split=='FIXED_EVAL':
            wanted={q['condition_id'] for q in read(PANEL)['requests']}
            records=[r for r in records if r['condition_id'] in wanted]
        for row in records:
            p=resolve(row['case_path']);inputs[str(p)]=sha(p)
            if row.get('case_sha256'):assert inputs[str(p)]==row['case_sha256']
            case=arrays(p);cid=row['condition_id'];sig=input_signature(case)
            own=case['occupied_labels'];b=len(own)
            # Original oracle packing itself must reconstruct all N source faces.
            _,orig=compatible_endpoint(case,own);assert orig['status']=='EXACT_TARGET_BIJECTION_PASS'
            choices=cache_by_sig.get(sig,[])
            masks={ahash(c['arrays']['occupied']) for c in choices}
            base=dict(split=split,condition_id=cid,parent_id=row['parent_id'],
                part_type=row.get('part_type','UNKNOWN'),N=int(case['N']),K=int(case['K']),
                boundary_edges=b,source_owners=int(own.sum()),
                source_aliases=int(case['source_alias_labels'].sum()),
                source_null=int((~case['source_supported_labels']).sum()),
                source_endpoint_roundtrip='PASS',case_path=str(p),case_sha256=inputs[str(p)],
                cached_variants=len(choices),distinct_cached_masks=len(masks))
            if split=='CURRENT_TRAIN' and choices:train_cache_count+=1
            # Empty boundary has a unique C-only result and requires no head call.
            variants=choices if choices else ([dict(path='CLOSED_C_NO_HEAD',sha256='',arrays={'occupied':np.zeros(0,bool)})] if b==0 else [])
            if not variants:
                rows.append(dict(base,cache_status='BLOCKED_NO_EXACT_C_N_LAYOUT_CACHE',
                    layout_comparison='NA',deployment_endpoint_status='NA'))
                continue
            # Report all observed masks when a C had several historical heads;
            # never choose whichever mask best matches its source completion.
            for vi,cached in enumerate(variants):
                pred=cached['arrays']['occupied'];cmp,pack,comp=compare(case,pred)
                variant=hashlib.sha256(cached['sha256'].encode()).hexdigest()[:10] if cached['sha256'] else 'closed'
                if cached['sha256']:
                    used_cache[cached['path']]=cached['sha256'];inputs[cached['path']]=cached['sha256']
                d=dict(base,variant=variant,cache_status='EXACT_C_N_CACHE' if b else 'CLOSED_C_NO_HEAD',
                    cache_path=cached['path'],cache_sha256=cached['sha256'],**cmp,
                    deployment_endpoint_status=comp['status'],
                    active_null_edges=len(comp['active_unsupported_edges']),
                    duplicated_target_faces=len(comp['duplicated_source_face_edges']))
                rows.append(d);details.append(dict(condition_id=cid,variant=variant,**comp))
                for ei in range(b):
                    edges.append(dict(split=split,condition_id=cid,parent_id=row['parent_id'],variant=variant,
                        edge_id=ei,source_role='OWNER' if own[ei] else ('ALIAS' if case['source_alias_labels'][ei] else 'NULL'),
                        source_face_id=int(case['edge_owner_face'][ei]),predicted_active=bool(pred[ei]),
                        source_active=bool(own[ei]),C_u=int(case['edge_records'][ei,1]),C_v=int(case['edge_records'][ei,2])))
                if pack is not None:
                    dest=OUT/'endpoint_contract_examples'/cid/(variant+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                    with dest.open('wb') as f:np.savez_compressed(f,**pack)
    write_csv(OUT/'per_condition_layout.csv',rows);write_csv(OUT/'per_edge_layout.csv',edges)
    write_csv(OUT/'cache_inventory.csv',inventory);write_json(OUT/'compatibility_details.json',details)
    # A source oracle layout is a feasibility diagnostic only. ALL_ON is C-only,
    # but is an unadopted alternative policy, never relabelled as the cached head.
    all_on=[]
    for row in train:
        case=arrays(row['case_path']);_,c=compatible_endpoint(case,np.ones(len(case['occupied_labels']),bool))
        all_on.append(dict(condition_id=row['condition_id'],parent_id=row['parent_id'],
            boundary_edges=len(case['occupied_labels']),status=c['status'],
            active_null_edges=len(c['active_unsupported_edges']),duplicate_target_faces=len(c['duplicated_source_face_edges'])))
    write_csv(OUT/'all_on_representation_diagnostic.csv',all_on)
    summary=dict(status='AUDIT_COMPLETE_NOT_NEW_TRAINING_POOL',current_train_conditions=91,
        current_train_parents=len({r['parent_id'] for r in train}),current_train_exact_cache_conditions=train_cache_count,
        discovered_cache_files=len(paths),unique_cache_files=len(unique),
        current_train_closed_conditions=sum(r['boundary_edges']==0 for r in rows if r['split']=='CURRENT_TRAIN'),
        condition_variant_counts={s:dict(collections.Counter(r['deployment_endpoint_status'] for r in rows if r['split']==s)) for s in ('CURRENT_TRAIN','FIXED_EVAL')},
        layout_comparison_counts={s:dict(collections.Counter(r['layout_comparison'] for r in rows if r['split']==s)) for s in ('CURRENT_TRAIN','FIXED_EVAL')},
        all_on_diagnostic=dict(collections.Counter(r['status'] for r in all_on)),
        neural_forwards=0,GPU_calls=0,new_OT=0,new_noise=0,new_training_samples=0,
        source_owner_changes_model_input=True,source_owner_only_supervision_claim=False,
        cannot_infer_failure_causality_from_this_audit=True,
        pending='Missing C-only head caches require separately authorized head inference; incompatible masks cannot be repaired using Y without changing the deployment input contract.',
        elapsed_seconds=time.perf_counter()-start)
    write_json(OUT/'summary.json',summary)
    assert 'torch' not in sys.modules,'Audit unexpectedly imported torch'
    for p,h in inputs.items():assert sha(p)==h,p
    outputs={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p!=complete}
    write_json(complete,dict(status='COMPLETE',input_hashes=inputs,output_hashes=outputs,
        pure_CPU=True,torch_imported=False,neural_forwards=0,GPU_calls=0,new_OT=0,
        elapsed_seconds=time.perf_counter()-start))
    return summary


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2))
