"""Materialize at most six versioned C/Y tasks from frozen three-parent targets.

No head, neural model, OT, new noise, repair or simplification. Semantic bits
name already selected regions of the NEW target, never reuse old face IDs.
All production eligibility is false until a separate actual image review.
"""
from __future__ import annotations
import argparse, collections, json, sys
import numpy as np
from . import common as C
from . import production_contract as P
from r2.data_tools.bench_parts_v1.construct import partition
from r2.data_tools.bench_parts_v1.quality import SourceAuditor, topology

OUT=C.OUT/'tasks'
VERSION='BENCH_THREE_FAMILY_CY_V1'


def seed_verified_C_pairs(auditor,ids,geometry):
    """Reuse full-target pair verdicts without inventing their contact class.

Only C-C pairs are inserted. C/Y source-edge pairs are therefore evaluated by
the unchanged SourceAuditor detector on demand, and cached across the two C.
"""
    n=len(auditor.f)
    assert geometry['pair_count']==n*(n-1)//2
    exceptions={}
    for rows in (geometry['illegal_pairs'],geometry['unresolved_pairs']):
        for row in rows:
            key=tuple(sorted((int(row['face_i']),int(row['face_j']))))
            exceptions[key]=row['detector']
    inherited=0
    for pi,i in enumerate(ids):
        for j in ids[pi+1:]:
            key=tuple(sorted((int(i),int(j))))
            if key not in auditor.pair_cache:
                auditor.pair_cache[key]=exceptions.get(key,dict(
                    classification='REUSED_FULL_TARGET_VERIFIED_LEGAL',illegal=False,verified=True,
                    contact_subtype='NOT_RECONSTRUCTED_FROM_AGGREGATE_CACHE'))
                inherited+=1
    return inherited


def prepare_interface_cache(auditor,known):
    """Remove only anonymous inherited classifications needed as actual C/Y edges."""
    ids=np.asarray(known,np.int64);f=auditor.f;ks=set(ids.tolist())
    for tri in f[ids]:
        for j in range(3):
            e=tuple(sorted((int(tri[j]),int(tri[(j+1)%3]))))
            entries=auditor.book[e];inside=[x[0] for x in entries if x[0] in ks]
            outside=[x[0] for x in entries if x[0] not in ks]
            if len(entries)==2 and len(inside)==len(outside)==1:
                key=tuple(sorted((inside[0],outside[0])))
                if auditor.pair_cache.get(key,{}).get('classification')=='REUSED_FULL_TARGET_VERIFIED_LEGAL':
                    del auditor.pair_cache[key]


def one_task(parent,target_row,target,semantic_bits,old_task,auditor,geometry):
    recipe=old_task['recipe_id'];bit={'R1':1,'R2':2}[recipe]
    ids=np.flatnonzero((semantic_bits & bit)!=0).astype(np.int64)
    cid=parent['parent_id']+'__'+recipe+'__THREE_FAMILY_V1'
    dest=OUT/'conditions'/cid;dest.mkdir(parents=True,exist_ok=True)
    receipt=dest/'receipt.json'
    if receipt.exists():
        r=C.read(receipt)
        for p,h in r['files'].items():assert P.file_sha(p)==h,p
        return r['row']
    n=len(target['source_faces']);reasons=list(target_row.get('geometry_reasons',[]))
    if not target_row['geometry_pass'] and not reasons:reasons.append('TARGET_GEOMETRY_FAILED')
    if not 128<=n<=256:reasons.append('N_OUTSIDE_128_256')
    if not 0<len(ids)<n:
        row=dict(condition_id=cid,parent_id=parent['parent_id'],recipe_id=recipe,N=n,K=len(ids),
            status='HOLD_INVALID_SEMANTIC_PARTITION',hold_reasons=reasons+['C_EMPTY_OR_ENTIRE_M'],
            eligible_for_production_contract=False,pre_review_eligible=False,training_ready=False)
        C.save(receipt,dict(row=row,files={}));return row
    p=partition(target,ids)
    # The source-ID and coordinate partition is independent of all source labels.
    assert np.array_equal(np.sort(np.r_[p['known_ids'],p['free_ids']]),np.arange(n))
    restored=np.empty_like(target['full_target']);restored[ids]=p['C'];restored[p['free_ids']]=p['Y']
    assert np.array_equal(restored,target['full_target'])
    inherited=seed_verified_C_pairs(auditor,ids,geometry)
    prepare_interface_cache(auditor,ids)
    before=dict(auditor.calls)
    q,sup=auditor.candidate(ids,dict(candidate_id=cid,parent_id=parent['parent_id'],
        group=old_task['intended_eval_group'],proposed_part_type=old_task['proposed_region']))
    # Align explicitly to the existing Bench C-only boundary ordering.
    try:
        edge_order=P.D.boundary_from_C(p['C_vertices'],p['C_faces_original'])
        assert np.array_equal(edge_order,sup['edge_records'])
    except ValueError as e:
        reasons.append('BENCH_C_BOUNDARY_VALIDATION_FAILED');q['bench_boundary_error']=str(e)
        edge_order=sup['edge_records']
    for key in ('C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','known_ids','free_ids'):
        assert np.array_equal(sup[key],p[key]),key
    reasons.extend(q['reasons'])
    if len(edge_order)>len(p['free_ids']):reasons.append('BOUNDARY_SLOTS_EXCEED_FREE_BUDGET')
    reasons=sorted(set(reasons))
    case={**target,**p,**sup}
    inp={key:p[key] for key in P.INPUT_KEYS if key in p}
    P.verify_case(case,inp)
    C.npz(dest/'input_C_only.npz',inp);C.npz(dest/'paired_C_Y.npz',p);C.npz(dest/'case.npz',case)
    # Keep old auditor applicability text as historical policy, with explicit
    # current Bench override rather than mistaking closed C / alias for a bug.
    q.update(current_bench_closed_C_supported=True,current_bench_entry_consumable=not reasons,
        current_bench_entry_limitations=reasons,
        existing_alias_policy='source owner-only packing; production must independently predict a representable mask',
        new_C_internal_pair_tests=0,inherited_C_pair_verdicts=inherited,
        source_interface_new_detector_calls=int(auditor.calls['finite_detector_calls']-before.get('finite_detector_calls',0)),
        source_interface_new_pair_requests=int(auditor.calls['unique_pair_requests']-before.get('unique_pair_requests',0)),
        source_surface_changed_during_partition=False,model_calls=0,OT_calls=0,
        retained_target_geometry_path=target_row.get('geometry_path',str(C.resolve(target_row['target_case_path']).with_name('geometry.json'))),
        current_status='HOLD' if reasons else 'PASS_GEOMETRY_PENDING_ACTUAL_VISUAL_REVIEW')
    C.save(dest/'quality.json',q)
    selection=dict(condition_id=cid,parent_id=parent['parent_id'],recipe_id=recipe,version=VERSION,
        source_target_path=target_row['target_case_path'],source_target_sha256=target_row['target_case_sha256'],
        new_target_face_ids=ids.tolist(),new_face_selection='face_semantic_bits & '+str(bit),
        semantic_ancestry_condition_id=old_task['condition_id'],
        semantic_ancestry_region=old_task['proposed_region'],old_source_face_ids_reused=False,
        intended_eval_group=old_task['intended_eval_group'],future_training_pool_membership=recipe=='R1',
        actual_training_pool_membership=False,eligible_for_production_contract=False)
    C.save(dest/'selection.json',selection)
    row=dict(condition_id=cid,parent_id=parent['parent_id'],recipe_id=recipe,version=VERSION,
        N=n,K=len(ids),target_method=target_row['method'],proposed_region=old_task['proposed_region'],
        intended_eval_group=old_task['intended_eval_group'],future_training_pool_membership=recipe=='R1',
        semantic_ancestry_condition_id=old_task['condition_id'],
        target_case_path=target_row['target_case_path'],target_case_sha256=target_row['target_case_sha256'],
        source_case_path=target_row['target_case_path'],source_case_sha256=target_row['target_case_sha256'],
        source_face_ids=ids.tolist(),C_open_boundary_count=len(edge_order),
        source_owner_count=int(sup['occupied_labels'].sum()),source_alias_count=int(sup['source_alias_labels'].sum()),
        source_supported_count=int(sup['source_supported_labels'].sum()),
        closed_C_existing_Bench_layout_supported=not bool(len(edge_order)),
        pre_review_eligible=not reasons,eligible_for_production_contract=False,
        training_ready=False,training_pool_membership=False,
        status='HOLD' if reasons else 'PENDING_ACTUAL_VISUAL_REVIEW',hold_reasons=reasons,
        review_status='NOT_REVIEWED',C_Y_exact_complement=True,C_Y_same_target_transform=True,
        C_geometry_sha256=P.A.ahash(p['C']),model_calls=0,head_calls=0,OT_calls=0)
    for stem,filename in [('input','input_C_only.npz'),('paired','paired_C_Y.npz'),('case','case.npz'),('quality','quality.json'),('selection','selection.json')]:
        row[stem+'_path']=str(dest/filename);row[stem+'_sha256']=P.file_sha(dest/filename)
    C.save(receipt,dict(row=row,files={str(f):P.file_sha(f) for f in dest.iterdir() if f.is_file()}))
    return row


def run(selected_path):
    frozen=C.read(C.OUT/'PROTOCOL.json');selected=C.read(selected_path);rows=selected['rows']
    parents={r['parent_id']:r for r in frozen['parents']}
    assert len(rows)==3 and {r['parent_id'] for r in rows}==set(parents)
    OUT.mkdir(parents=True,exist_ok=True)
    protected={str(C.resolve(selected_path)):P.file_sha(selected_path),str(C.OUT/'PROTOCOL.json'):P.file_sha(C.OUT/'PROTOCOL.json')}
    result=[];pair_counts=collections.Counter()
    for row in rows:
        p=C.resolve(row['target_case_path']);assert P.file_sha(p)==row['target_case_sha256']
        a=C.load(p);sem=C.load(row['semantics_path'])['face_semantic_bits']
        assert sem.dtype.kind in 'iu' and sem.shape==(len(a['source_faces']),) and np.all((sem>=0)&(sem<=3))
        gp=C.resolve(row.get('geometry_path',str(p.with_name('geometry.json'))));g=C.read(gp)
        protected[str(p)]=P.file_sha(p);protected[str(C.resolve(row['semantics_path']))]=P.file_sha(row['semantics_path']);protected[str(gp)]=P.file_sha(gp)
        auditor=SourceAuditor(a,source_sha256=row['target_case_sha256'])
        par=parents[row['parent_id']]
        for task in sorted(par['tasks'],key=lambda x:x['recipe_id']):
            result.append(one_task(par,row,a,sem,task,auditor,g))
        pair_counts.update(auditor.calls)
    assert len(result)==6 and len({r['condition_id'] for r in result})==6
    # Explicit same-parent reserve protection, no new labels called TEST.
    for uid in parents:
        rs=[r for r in result if r['parent_id']==uid]
        if all('source_face_ids' in r for r in rs):
            assert rs[0]['source_face_ids']!=rs[1]['source_face_ids']
            assert not set(rs[1]['source_face_ids']).issubset(rs[0]['source_face_ids'])
    summary=dict(status='MATERIALIZED_PENDING_ACTUAL_REVIEW',schema=VERSION,parents=3,conditions=6,
        rows=result,train=[],training_ready=False,model_calls=0,head_calls=0,OT_calls=0,
        pre_review_eligible_conditions=sum(r['pre_review_eligible'] for r in result),
        production_eligible_conditions=0,source_interface_call_accounting=dict(pair_counts),
        protected_sources=protected,code_sha256={str(__file__):P.file_sha(__file__)},
        next='Actual target/C/Y image review must write reviewed_manifest.json; this manifest cannot call the head.')
    C.save(OUT/'manifest.json',summary)
    C.save(OUT/'production_manifest.json',dict(schema=VERSION,rows=result,eligible_conditions=0,
        manifest_sha256=P.file_sha(OUT/'manifest.json'),requires_separate_reviewed_manifest=True))
    for p,h in protected.items():assert P.file_sha(p)==h,p
    assert 'torch' not in sys.modules,'Materialization must not import a neural runtime'
    return dict(conditions=6,pre_review_eligible=summary['pre_review_eligible_conditions'],head_calls=0,model_calls=0,OT_calls=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selected',default=str(C.OUT/'selected_targets.json'))
    args=parser.parse_args();print(json.dumps(run(args.selected),ensure_ascii=False))
