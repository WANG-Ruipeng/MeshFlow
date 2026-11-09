"""Immutable CPU seat candidates from five retained low-face targets.

No target repair, relabeling of old gates, model inference, or training.
Candidate acceptance and train/holdout split remain a root decision.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from r2.data_tools.bench_user_v2 import build as b
from r2.data_tools.bench_parts_v1 import quality as Q
from r2.data_tools.bench_parts_v1.construct import partition
from r2.data_tools.bench_seat_coverage_v1.inspect import ROOT,OUT,sources

def loadjson(p):return json.loads(Path(p).read_text('utf-8-sig'))
def selection(r,a):
    uid=r['parent_id'];desc=loadjson(OUT/'inspection'/(uid+'_bounds.json'))
    if uid.startswith('3e'):
        ids=[74,76,77,81,83,84,85,87,88,89,90,91,92,93,94,95]
        note='All original upward seat-plane faces; retain real source rear notches. Identical to old semantic-HOLD C; no new target or repaired opening.'
        prior='Old V2 semantic HOLD: irregular notch / C-alone meaning uncertain; independent review required.'
    elif uid.startswith('a8'):
        ids=list(range(150,189));note='Complete top of all seven interior seat slats plus perimeter end/front/back rails. Original shared-point contacts retained; no slat or difficult face removed.'
        prior='Old RELATION SLATTED_TOP_FULL K39 held for nonmanifold C vertex links; old geometry decision preserved.'
    elif uid.startswith('d9'):
        ids=[102,106];note='All original horizontal seat-level upper faces between end uprights and below the continuous back; two triangles. No back-top ledge included.'
        prior='Identical old V2 K2 seat band, classified partial in FULL course. New full-target review must settle extent; not automatically accepted.'
    elif uid.startswith('df'):
        ids=next(x['face_ids'] for x in desc['regions'] if len(x['face_ids'])==38)
        note='Complete pair of upper seating surfaces on both sides of central back divider, including original end connection. No split/duplication of source vertices.'
        prior='Old V2E seat K38 held for one nonmanifold C vertex link; target itself has accepted geometry/fidelity/visual receipt.'
    else:
        old=loadjson(ROOT/'experiments/r2_bench_three_family_v1/tasks/conditions'/f'{uid}__R1__THREE_FAMILY_V1'/'selection.json')['new_target_face_ids']
        ids=[x['id'] for x in desc['faces'] if x['id'] in old and x['center'][1]<.3 and x['normal'][1]>.2]
        note='Complete upper seating bowl with rounded front lip, ending at the source bend into the rising back. Existing upper faces only; side-thickness and underside remain Y. Bend boundary is a curation choice, not a source semantic label.'
        prior='New C subset of accepted 255-face 27 target; old R1 seat-back shell/source-Null metadata not reused as this C layout.'
    return sorted(ids),note,prior

def vertex_link_details(case,q):
    details=[];f=case['C_faces_original'];v=case['C_vertices'];sid=case['C_source_vertex_ids'];known=case['known_ids']
    for item in q['topology']['nonmanifold_vertices']:
        i=int(item['vertex_id']);local=np.flatnonzero((f==i).any(1));tris=case['C_vertices'][f[local]]
        details.append(dict(old_detector_item=item,C_vertex_id=i,source_vertex_id=int(sid[i]),position=v[i].tolist(),C_face_ids=local.tolist(),target_face_ids=known[local].tolist(),incident_triangles=tris.tolist(),interpretation='Finite triangle contact and semantic component explanation must be reviewed; do not infer target self-intersection from C vertex link alone.'))
    return details

def build():
    if (OUT/'candidate_manifest.json').exists():
        m=loadjson(OUT/'candidate_manifest.json')
        for r in m['conditions']:
            assert b.sha(r['case_path'])==r['case_sha256']
        print('ALREADY_CANDIDATES_READY');return
    from r2.data_tools.bench_user_relation_rebuild_v1.run import resources
    started=time.perf_counter();b.save(OUT/'resources_before_build.json',resources())
    old=loadjson(ROOT/'experiments/r2_bench_full_seat_curriculum_v1/data/manifest.json');rows=[];protected={}
    for pr in sources():
        uid=pr['parent_id'];a=b.load(pr['target_case_path']);ids,note,prior=selection(pr,a)
        cid=uid+'__COMPLETE_SEAT_CANDIDATE__COVERAGE_V1';dest=OUT/'conditions'/cid
        protected[str(b.resolve(pr['target_case_path']))]=b.sha(pr['target_case_path'])
        pair=partition(a,ids);q,sup=Q.SourceAuditor(a,pr['source_target_sha256']).candidate(np.array(ids,dtype=np.int64),dict(candidate_id=cid,parent_id=uid,group='UNASSIGNED',proposed_part_type='SEAT_SURFACE'))
        case=dict(a,**pair);case.update(sup);cs=set(case['C_source_vertex_ids'].tolist())
        case['source_third_in_C']=np.array([bool(ok and tr[2] in cs) for ok,tr in zip(case['source_supported_labels'],case['edge_source_triple'])])
        inp={k:case[k] for k in ('C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations','N','K','target_version_id')}
        for name,data in [('case',case),('input',inp),('paired',pair),('supervision_only',sup)]:b.npz(dest/(name+'.npz'),data)
        b.save(dest/'quality.json',q);b.save(dest/'vertex_links.json',vertex_link_details(case,q))
        b.save(dest/'selection.json',dict(target_face_ids=ids,description=note,prior_gate=prior,source_target_path=pr['target_case_path'],source_target_sha256=pr['source_target_sha256'],target_unchanged=True,C_old_source_ID_preserved=True))
        n,k=int(case['N']),int(case['K']);iv=pr['inventory']
        row=dict(condition_id=cid,parent_id=uid,group='UNASSIGNED',replay_group='NEW',official_split=iv['official_split'],N=n,K=k,part_type='SEAT_SURFACE',region_name='COMPLETE_SEAT_CANDIDATE',extent_description=note,prior_gate=prior,source_face_ids=ids,target_face_ids=ids,target_version_id=str(case['target_version_id'].item()),
          boundary_edges=len(sup['edge_records']),source_owner_count=int(sup['occupied_labels'].sum()),source_alias_count=int(sup['source_alias_labels'].sum()),source_null_count=int((~sup['source_supported_labels']).sum()),
          geometry_status=q['geometry_status'],geometry_reasons=q['reasons'],native_array_compatible=128<=n<=256 and 2<=k<n,review_status='PENDING_ROOT_ACTUAL_REVIEW',eligibility='CANDIDATE_ONLY',
          current_course22_parent=uid in old['train_parents'],COMMON12000_exposed=iv['in_COMMON64']=='True',historical_RELATION26_exposed=iv['in_current_RELATION26']=='True',historical_NEWDATA27_exposed=True,source_initialization_exposed=iv['in_COMMON64']=='True',official_pretraining_exposure=iv['actual_official_checkpoint_parent_exposure'],old_DEV_regression=iv['old_DEV_REGRESSION']=='True',
          source_target_path=pr['target_case_path'],source_target_sha256=pr['source_target_sha256'],source_target_mesh_path=pr['target_mesh_path'],full_target_geometry_recomputed=False,source_parent_receipt=pr,
          model_input_policy='C,N and reversible C metadata only; source labels/Y used solely for supervision; no old C-only head gate')
        for field,name in [('case','case.npz'),('input','input.npz'),('paired','paired.npz')]:row[field+'_path']=str(dest/name);row[field+'_sha256']=b.sha(dest/name)
        for field,name in [('quality','quality.json'),('supervision','supervision_only.npz')]:row[field+'_file']=str(dest/name);row[field+'_sha256']=b.sha(dest/name)
        rows.append(row)
    b.save(OUT/'protected_source_hashes.json',protected)
    b.save(OUT/'candidate_manifest.json',dict(schema='BENCH_COMPLETE_SEAT_COVERAGE_CANDIDATES_V1',conditions=rows,train=[],heldout=[],old_manifest=str(ROOT/'experiments/r2_bench_full_seat_curriculum_v1/data/manifest.json'),old_manifest_sha256=b.sha(ROOT/'experiments/r2_bench_full_seat_curriculum_v1/data/manifest.json'),status='NEEDS_ROOT_REVIEW_AND_SPLIT',training_updates=0,model_forwards=0))
    b.csvout(OUT/'candidate_conditions.csv',[{k:v for k,v in r.items() if k!='source_parent_receipt'} for r in rows]);b.save(OUT/'build_receipt.json',dict(seconds=time.perf_counter()-started,parent_count=len(rows),C_geometry_checks=len(rows),model_forwards=0,full_target_reaudit=0))
    print(json.dumps([{k:r[k] for k in ('condition_id','N','K','boundary_edges','geometry_reasons','COMMON12000_exposed','historical_RELATION26_exposed')} for r in rows],indent=2))
def render():
    from r2.pilots.bench_full_seat_curriculum_v1.data import _render
    m=loadjson(OUT/'candidate_manifest.json');rows=[]
    for r in m['conditions']:
        target=OUT/'gallery'/(r['condition_id']+'.png')
        if target.exists():rows.append(dict(image_path=str(target),image_sha256=b.sha(target),condition_id=r['condition_id']));continue
        a=b.load(r['case_path']);d=_render(r,a,target);d['condition_id']=r['condition_id'];rows.append(d)
    b.save(OUT/'gallery_manifest.json',rows);print(json.dumps(rows))
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','render']);args=p.parse_args();globals()[args.stage]()
if __name__=='__main__':main()
