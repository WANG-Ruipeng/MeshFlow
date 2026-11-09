"""Seven unique endpoint checks, no new prediction, OT, noise or geometry scoring."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json, time
from pathlib import Path
import numpy as np
from r2.data_tools.bench_three_family_v1.common import ROOT,resolve,read,sha,load,save,npz
from r2.data_tools.bench_deployment_tasks_v1 import layout_contract as A

OUT=ROOT/'experiments/r2_bench_null_layout_followup_v1'
PREV=ROOT/'experiments/r2_bench_three_family_v1'

def verify():
    c=read(OUT/'COMPLETE.json')
    for p,h in {**c['inputs'],**c['outputs']}.items():assert sha(p)==h,p
    print(json.dumps(c['summary'],ensure_ascii=False));return c

def main():
    if (OUT/'COMPLETE.json').exists():return verify()
    from r2.pilots.bench_adaptation_strength_v1.common import resources
    before=resources()
    train=read(ROOT/'experiments/r2_bench_relation_training_v1/data/manifest.json')['train']
    new=next(r for r in read(PREV/'tasks/reviewed_manifest.json')['rows'] if r['recipe_id']=='R1' and r['parent_id'].startswith('27eb79'))
    positive=sorted([r for r in train if r['source_owner_count']>0 and r['source_alias_count']==0],key=lambda r:r['condition_id'])[0]
    alias=sorted([r for r in train if r['source_alias_count']>0],key=lambda r:r['condition_id'])[0]
    rows=[dict(new,probe_role='NEW_OPEN_NULL'),dict(positive,probe_role='EXISTING_POSITIVE'),dict(alias,probe_role='EXISTING_ALIAS')]
    current_path=PREV/'production_contract/conditions'/new['condition_id']/'layout.npz'
    files=[Path(__file__),ROOT/'experiments/r2_bench_relation_training_v1/data/manifest.json',
        PREV/'tasks/reviewed_manifest.json',current_path,
        ROOT/'r2/data_tools/bench_deployment_tasks_v1/layout_contract.py',
        ROOT/'r2/pilots/bench_parts_convergence_v1/data.py',ROOT/'r2/pilots/seam_abc_v1/slot_data_extension.py',
        ROOT/'r2/src/meshflow_control/seam/slot_data.py']+[resolve(r['case_path']) for r in rows]
    # This metadata recount is not fresh geometry, labeling, or head evaluation.
    inputs={str(p):sha(p) for p in files}
    protocol=dict(task='Source-supervised endpoint versus frozen-head compatibility distinction',
        selected=[dict(condition_id=r['condition_id'],role=r['probe_role']) for r in rows],
        choice='Lexicographically first existing TRAIN positive without alias and first TRAIN alias; no output-quality selection',
        variants=dict(NEW_OPEN_NULL=['cached_current','source_owner','all_off_same_as_source'],
            EXISTING_POSITIVE=['source_owner','all_off'],EXISTING_ALIAS=['source_owner','one_alias_replaces_owner','owner_and_alias_duplicate']),
        max_unique_endpoint_checks=7,head_forwards=0,generator_forwards=0,training=0,OT=0,
        warning='All alternative masks are explicit offline diagnostics, never corrected deployment predictions or a new training admission.',
        resources=before,workers=1,CPU_threads=1,input_hashes=inputs)
    save(OUT/'PROTOCOL.json',protocol);started=time.perf_counter()
    result=[];forward=0;checks=0
    for row in rows:
        c=load(row['case_path']);labels=c['occupied_labels'].astype(bool);variants={'source_owner':labels}
        if row['probe_role']=='NEW_OPEN_NULL':
            assert not labels.any() and len(labels)==31
            variants={'cached_current':load(current_path)['occupied'],'source_owner':labels,'all_off_same_as_source':labels.copy()}
        elif row['probe_role']=='EXISTING_POSITIVE':variants['all_off']=np.zeros_like(labels)
        else:
            ei=int(np.flatnonzero(c['source_alias_labels'])[0]);fi=int(c['edge_owner_face'][ei])
            owner=int(np.flatnonzero(labels & (c['edge_owner_face']==fi))[0])
            swap=labels.copy();swap[owner]=False;swap[ei]=True
            duplicate=labels.copy();duplicate[ei]=True
            variants.update(one_alias_replaces_owner=swap,owner_and_alias_duplicate=duplicate)
        cached={}
        for name,mask in variants.items():
            key=mask.tobytes();dest=OUT/'endpoints'/row['probe_role']/name
            if key in cached:
                prior=cached[key];result.append(dict(prior,variant=name,reused_identical_mask=prior['variant']));continue
            lay=A.layout(c,mask);pack,detail=A.compatible_endpoint(c,mask);checks+=1
            n,k=int(c['N']),int(c['K']);wind=False
            if pack is not None:
                for si,fi in enumerate(pack['source_face_ids']):
                    cp=pack['source_corner_permutations'][si];vid=c['source_face_vertex_ids'][fi,cp]
                    orig=c['source_faces'][fi]
                    perm=np.asarray([int(np.flatnonzero(vid==v).item()) for v in orig])
                    expected=np.asarray([int(np.flatnonzero(c['source_face_vertex_ids'][fi]==v).item()) for v in orig])
                    assert np.array_equal(pack['x1'][si,perm],c['full_target'][fi,expected])
                wind=True;npz(dest/'endpoint.npz',pack)
            npz(dest/'diagnostic_mask.npz',dict(occupied=mask))
            save(dest/'packing.json',detail)
            r=dict(condition_id=row['condition_id'],probe_role=row['probe_role'],variant=name,N=n,K=k,
                boundary_edges=len(mask),active_edges=int(mask.sum()),seam_slots=len(lay['seam_slots']),bulk_slots=len(lay['bulk_slots']),
                fixed_free_corners=int((lay['anchor_ids'][k:]>=0).sum()),free_scalars=int(lay['free_coordinate_count']),
                status=detail['status'],source_winding_roundtrip=wind,
                coordinates_and_faces_preserved=bool(pack is not None),deployment_policy_changed=False,
                reused_identical_mask=None,detail_path=str(dest/'packing.json'))
            result.append(r);cached[key]=r
    assert checks==7,checks
    assert sum(r['status']=='EXACT_TARGET_BIJECTION_PASS' for r in result)==6
    save(OUT/'endpoint_results.json',dict(rows=result,unique_checks=checks,logical_variants=len(result)))
    stats=[]
    for r in train+[new]:
        c=load(r['case_path']);inputs[str(resolve(r['case_path']))]=sha(r['case_path'])
        b=len(c['occupied_labels']);owners=int(c['occupied_labels'].sum());aliases=int(c['source_alias_labels'].sum())
        support=int(c['source_supported_labels'].sum());assert support==owners+aliases
        stats.append(dict(condition_id=r['condition_id'],parent_id=r['parent_id'],B=b,owner=owners,alias=aliases,
            true_null=b-support,all_null_open=bool(b and support==0),closed=b==0,
            group='NEW_SUPERVISION_CANDIDATE_ONLY' if r['condition_id']==new['condition_id'] else 'EXISTING91'))
    save(OUT/'source_role_inventory.json',dict(rows=stats,
        scope='Existing label metadata only; no new geometry classification. Added candidate not activated.',
        existing=dict(conditions=91,parents=len({r['parent_id'] for r in train}),
            edges=sum(r['B'] for r in stats[:-1]),owner=sum(r['owner'] for r in stats[:-1]),
            alias=sum(r['alias'] for r in stats[:-1]),true_null=sum(r['true_null'] for r in stats[:-1]),
            all_null_open=sum(r['all_null_open'] for r in stats[:-1]),closed=sum(r['closed'] for r in stats[:-1])),
        new=stats[-1]))
    summary=dict(status='CPU_CONTRACT_CHECK_COMPLETE',unique_endpoint_checks=checks,logical_variants=len(result),
        neural_forwards=0,training_updates=0,new_noise=0,OT=0,geometry_rescoring=0,
        new_supervised_candidate_parents=1,active_training_pool_added=0,
        fixed_old_head_compatible_train_candidates=0,
        elapsed_seconds=time.perf_counter()-started)
    save(OUT/'summary.json',summary);save(OUT/'resources_after.json',resources())
    for p,h in inputs.items():assert sha(p)==h,p
    outputs={str(p):sha(p) for p in OUT.rglob('*') if p.is_file()}
    save(OUT/'COMPLETE.json',dict(summary=summary,inputs=inputs,outputs=outputs))
    return verify()

if __name__=='__main__':main()
