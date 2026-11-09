"""CPU-only candidate preparation, actual-review gate and frozen pool publication.

No semantic decisions are synthesized. Run candidates, review saved pictures,
then supply visual_reviews.json and run publish. Original assets stay read-only.
"""
from __future__ import annotations
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,csv,hashlib,io,json,tarfile,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import data as D
from r2.pilots.bench_control_transfer_v1.data_prepare import preprocess,geometry_hash
from r2.data_tools.bench_parts_v1 import construct as C,quality as Q
ROOT,OUT=D.ROOT,D.OUT
OLD=ROOT/'experiments/r2_bench_parts_convergence_v1/data/manifest.json'
INITIAL=ROOT/'experiments/r2_bench_control_transfer_v1/data/manifest.json'
INVENTORY=ROOT/'experiments/r2_bench_data_chain_audit_20261021/geometry/archive_supply/per_parent.csv'
ARCHIVE=ROOT/'downloaded_data/archives/shapenet-cls.tar.gz'

def save_npz(path,values):D.old._save_npz(path,values)
def write(path,obj):D.write_new(path,obj)
def _render_job(item):
    from r2.data_tools.bench_parts_v1 import render as renderer
    # Renderer has an experiment-root write guard. Bind that guard only inside
    # this isolated CPU worker; no historical renderer/source is modified.
    renderer.OUT=OUT
    parent,rows=item;dest=OUT/'gallery/data/candidates';dest.mkdir(parents=True,exist_ok=True)
    return renderer.candidate_sheet(parent,rows,dest)

def candidates():
    start=time.time();path=OUT/'candidates/manifest.json'
    if path.exists():return D.read(path)
    OUT.mkdir(parents=True,exist_ok=True)
    resource=D.old.resources();write(OUT/'preparation'/('candidate_dispatch_'+str(time.time_ns())+'.json'),dict(resources=resource,
        CPU_workers_geometry=1,CPU_workers_render=2,numeric_threads_per_worker=1,GPU_compute=False,
        reason='Small47condition audit sequential, independent15parent raster pages two processes; no neural probes'))
    inv=list(csv.DictReader(INVENTORY.open(encoding='utf-8-sig')))
    extra=[r for r in inv if r['N_bin']=='128_256' and r['existing64_8_membership']=='UNSELECTED']
    original=D.read(INITIAL);forbidden={r['parent_id'] for r in original['train']+original['dev']}
    old_geometries={r['exact_geometry_sha256']:r['parent_id'] for r in original['train']+original['dev']}
    protected={str(p):D.sha(p) for p in (OLD,INITIAL,INVENTORY,ARCHIVE,Path(__file__),Path(D.__file__),Path(C.__file__),Path(Q.__file__))}
    rows=[];blocked=[];parents=[]
    with tarfile.open(ARCHIVE) as tar:
        split={}
        for s in ('train','test'):
            with np.load(io.BytesIO(tar.extractfile('shapenet-02828884/split/'+s+'.npz').read()),allow_pickle=True) as z:split.update({str(v['uid']):s for v in z['npz_list']})
        for r in extra:
            pid=r['parent_id'];assert pid not in forbidden
            if split.get(pid)!='train':blocked.append(dict(parent_id=pid,reason='NOT_OFFICIAL_TRAIN'));continue
            member=tar.extractfile(r['archive_member']).read();assert hashlib.sha256(member).hexdigest()==r['member_sha256']
            with np.load(io.BytesIO(member),allow_pickle=False) as z:a=preprocess({k:z[k].copy() for k in z.files})
            gh=geometry_hash(a)
            if gh in old_geometries:blocked.append(dict(parent_id=pid,reason='EXACT_GEOMETRY_ALREADY_SELECTED',representative=old_geometries[gh]));continue
            old_geometries[gh]=pid
            sourcepath=OUT/'candidates/sources'/(pid+'.npz');save_npz(sourcepath,a);sourcehash=D.sha(sourcepath)
            found,missing=C.propose(a);blocked.extend(dict(parent_id=pid,**v) for v in missing)
            parents.append(dict(parent_id=pid,official_split=split[pid],source_archive_member=r['archive_member'],source_archive_member_sha256=r['member_sha256'],
                source_case_path=str(sourcepath),source_case_sha256=sourcehash,exact_geometry_sha256=gh,N=len(a['source_faces']),
                source_duplicate_excess=int(r['source_ID_duplicate_excess_faces']),source_exact_zero_area=int(r['exact_zero_area_faces_original_float64']),
                proposal_count=len(found),full_source_geometry_repaired=False))
            auditor=Q.SourceAuditor(a,sourcehash);v,f,t,area,n,cent=C.geom(a)
            for kind,i,ids,evidence,uncertainty in found:
                cid=pid+'__'+kind+'_'+str(i+1);pp=OUT/'candidates/paired'/(cid+'.npz');save_npz(pp,C.partition(a,ids))
                qp=OUT/'candidates/quality'/(cid+'.json');sp=OUT/'candidates/supervision'/(cid+'.npz')
                if qp.exists() and sp.exists():
                    report=D.read(qp);sup=D.arrays(sp)
                    if report['source_sha256']!=sourcehash or report['source_face_ids']!=ids:raise ValueError('Existing candidate audit identity differs')
                    assert np.array_equal(sup['known_ids'],ids)
                else:
                    report,sup=auditor.candidate(np.asarray(ids,np.int64),dict(candidate_id=cid,parent_id=pid,group='TRAIN',proposed_part_type=kind))
                    write(qp,report);save_npz(sp,sup)
                pts=t[ids].reshape(-1,3)
                rows.append(dict(candidate_id=cid,condition_id=cid,parent_id=pid,group='TRAIN',official_split='train',source_face_ids=ids,
                    source_case_path=str(sourcepath),source_case_sha256=sourcehash,case_path=str(pp),case_sha256=D.sha(pp),
                    proposed_part_type=kind,part_index=i+1,construction_evidence=evidence,uncertainty=uncertainty,
                    semantic_authority='GEOMETRIC_PROPOSAL_PENDING_ACTUAL_REVIEW',status='PENDING_ACTUAL_REVIEW',labels_fed_to_model=False,
                    quality_file=str(qp),quality_sha256=D.sha(qp),supervision_file=str(sp),supervision_sha256=D.sha(sp),
                    geometry_status=report['geometry_status'],geometry_reasons=report['reasons'],seam_applicability=report['seam_applicability'],seam_reasons=report['seam_reasons'],
                    source_owner_count=report['source_owner_count'],source_alias_count=report['source_alias_count'],source_null_unsupported_count=report['source_null_unsupported_count'],source_target_risks=report['source_target_risks'],
                    N=len(f),K=len(ids),face_fraction=len(ids)/len(f),area=float(area[ids].sum()),area_fraction=float(area[ids].sum()/area.sum()),
                    area_units='official_normalized_object_units_squared',area_denominator='sum_all_original_source_faces_including_duplicates',
                    bbox_normalized=[pts.min(0).tolist(),pts.max(0).tolist()],span_normalized=np.ptp(pts,axis=0).tolist(),region_normal_not_single_plane=bool(np.linalg.norm(np.std(n[ids],axis=0))>.1),
                    geometry_unchanged=True,source_partition_bitwise=True,**C.topology(a,ids)))
    groups=collections.defaultdict(list)
    for r in rows:groups[r['parent_id']].append(r)
    with ProcessPoolExecutor(max_workers=2) as pool:pages=list(pool.map(_render_job,sorted(groups.items())))
    for p,h in protected.items():assert D.sha(p)==h
    result=dict(schema='BENCH_COVERAGE_EXPANSION_CANDIDATES_V1',rows=rows,parents=parents,blocked=blocked,pages=pages,
        protected_hashes=protected,geometry_status_counts=dict(collections.Counter(r['geometry_status'] for r in rows)),
        no_model_calls=True,no_source_mutations=True,wall_seconds=time.time()-start)
    write(path,result)
    print(json.dumps(dict(candidates=len(rows),parents_with_candidates=len(groups),geometry_status_counts=result['geometry_status_counts'],manifest=str(path)),indent=2))
    return result

def publish():
    cand=D.read(OUT/'candidates/manifest.json');reviews=D.read(OUT/'preparation/visual_reviews.json')['rows']
    rv={r['candidate_id']:r for r in reviews}
    assert len(rv)==len(reviews) and set(rv)=={r['candidate_id'] for r in cand['rows']}
    old=D.read(OLD);assert len(old['train'])==277 and len(old['dev'])==39
    protected=dict(cand['protected_hashes']);protected[str(OUT/'preparation/visual_reviews.json')]=D.sha(OUT/'preparation/visual_reviews.json')
    oldrows=old['rows'];new=[];excluded=[];checks=[]
    for r in oldrows:
        for key,hkey in (('case_path','case_sha256'),('input_path','input_sha256'),('quality_file','quality_sha256')):
            assert D.sha(r[key])==r[hkey];protected[r[key]]=r[hkey]
        review=r['review'];assert review['decision']=='ACCEPT' and review['human_plausible_input']=='YES'
        assert D.sha(review['image_path'])==review['image_sha256'];protected[review['image_path']]=review['image_sha256']
    for row in cand['rows']:
        review=rv[row['candidate_id']]
        if not review.get('actually_viewed') or not review['reviewer'].startswith('AI_PROXY'):raise ValueError('Actual review required')
        assert D.sha(review['image_path'])==review['image_sha256']
        reason=list(row['geometry_reasons'])
        if review['decision']!='ACCEPT':reason.append('SEMANTIC_'+review['decision'])
        if review['human_plausible_input']!='YES':reason.append('PLAUSIBILITY_'+review['human_plausible_input'])
        if reason:excluded.append(dict(row,review=review,exclusion_reasons=reason));continue
        a=D.arrays(row['source_case_path']);pair=D.arrays(row['case_path']);sup=D.arrays(row['supervision_file']);cid=row['candidate_id']
        edges,features=D.old.edge_features(pair['C_vertices'],pair['C_faces_original'],int(pair['N']))
        if len(edges)>int(pair['N'])-int(pair['K']):excluded.append(dict(row,review=review,exclusion_reasons=['BOUNDARY_SLOTS_EXCEED_FREE']));continue
        assert np.array_equal(edges,sup['edge_records'])
        case={**a,**pair,**sup,'edge_features':features}
        cp=OUT/'data/cases'/(cid+'.npz');save_npz(cp,case)
        inp={k:pair[k] for k in ('C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations','N','K')}
        ip=OUT/'data/inputs'/(cid+'.npz');save_npz(ip,inp)
        obj=OUT/'data/inputs'/(cid+'.obj');text='# Original C vertices and source winding\n'+''.join('v '+' '.join(format(float(x),'.9g') for x in v)+'\n' for v in inp['C_vertices'])+''.join('f '+' '.join(str(int(x)+1) for x in f)+'\n' for f in inp['C_faces_original'])
        if obj.exists():assert obj.read_text('ascii')==text
        else:obj.write_text(text,encoding='ascii')
        r=dict(row,review=review,part_type=review['actual_part'],extent_description=review['extent_description'],status='ACTIVE_REVIEWED_SOURCE_CONDITION',
            semantic_authority='ACTUAL_NONBLIND_AI_PROXY',original_candidate_case_path=row['case_path'],original_candidate_case_sha256=row['case_sha256'],
            case_path=str(cp),case_sha256=D.sha(cp),paired_path=row['case_path'],paired_sha256=row['case_sha256'],input_path=str(ip),input_sha256=D.sha(ip),input_obj_path=str(obj),input_obj_sha256=D.sha(obj),
            adapter_semantics='CLOSED_NATIVE_NO_SEAM' if not len(edges) else 'OWNER_ONLY_OR_ALL_BULK',current_bench_entry_consumable=True,current_bench_entry_limitations=[])
        assert np.array_equal(case['full_target'][case['known_ids']],case['C']) and np.array_equal(case['full_target'][case['free_ids']],case['Y'])
        assert np.array_equal(np.sort(np.r_[case['known_ids'],case['free_ids']]),np.arange(int(case['N'])))
        assert np.array_equal(case['C_vertices'][case['C_faces_model']],case['C'])
        new.append(r);checks.append(dict(condition_id=cid,partition_bitwise=True,C_roundtrip=True,N_unchanged=int(case['N'])==len(a['source_faces']),inference_Y_keys=False))
    train=sorted(old['train']+new,key=lambda r:(r['parent_id'],r['part_type'],r['condition_id']))
    rows=train+old['dev'];assert not {r['parent_id'] for r in train}&{r['parent_id'] for r in old['dev']}
    schedule=D.make_schedule(train);write(OUT/'data/schedule.json',schedule)
    summary=dict(original_train_conditions=277,original_train_parents=64,new_conditions=len(new),new_parents=len({r['parent_id'] for r in new}),
        train_conditions=len(train),train_parents=len({r['parent_id'] for r in train}),dev_conditions=39,dev_parents=8,
        closed_train=sum(r['boundary_edges']==0 for r in train),open_no_owner_train=sum(r['boundary_edges']>0 and r['source_owner_count']==0 for r in train),
        train_part_counts=dict(collections.Counter(r['part_type'] for r in train)),new_part_counts=dict(collections.Counter(r['part_type'] for r in new)),excluded_new=len(excluded))
    result=dict(schema='BENCH_COVERAGE_EXPANSION_FROZEN_DATA_V1',rows=rows,train=train,dev=old['dev'],new_train=new,
        summary=summary,source_manifest=str(OLD),source_manifest_sha256=D.sha(OLD),schedule=str(OUT/'data/schedule.json'),schedule_sha256=D.sha(OUT/'data/schedule.json'),
        data_policy='Restore277reviewedconditions/64TRAINparents plus all qualified new official-TRAIN candidates. No source cleanup; preserved closed/noowner/native cases. C-only inference unchanged.',
        source_labels_for_training_only=True,part_labels_not_model_inputs=True,protected_hashes=protected)
    for p,h in protected.items():assert D.sha(p)==h
    write(OUT/'data/manifest.json',result);write(OUT/'data/excluded_new.json',dict(rows=excluded));write(OUT/'data/preparation_receipt.json',dict(status='COMPLETE',summary=summary,new_case_checks=checks,all_protected_hashes_unchanged=True,model_calls=0,training_updates=0))
    print(json.dumps(summary,ensure_ascii=False,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['candidates','publish']);a=p.parse_args()
    candidates() if a.stage=='candidates' else publish()
if __name__=='__main__':main()
