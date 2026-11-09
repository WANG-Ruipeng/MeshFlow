"""Read-only CPU proposal feasibility; does not publish any training condition."""
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
from pathlib import Path
import csv,json,hashlib,tarfile,io,time,subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'r2/src')]
from r2.pilots.bench_control_transfer_v1.data_prepare import preprocess
from r2.data_tools.bench_parts_v1.construct import propose,planar_regions,topology

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    start=time.time()
    archive=ROOT/'downloaded_data/archives/shapenet-cls.tar.gz'
    inv=ROOT/'experiments/r2_bench_data_chain_audit_20261021/geometry/archive_supply/per_parent.csv'
    records=list(csv.DictReader(inv.open(encoding='utf-8-sig')))
    chosen=[r for r in records if r['N_bin']=='128_256' and r['existing64_8_membership']=='UNSELECTED']
    protected=[archive,inv,ROOT/'r2/pilots/bench_control_transfer_v1/data_prepare.py',ROOT/'r2/data_tools/bench_parts_v1/construct.py']
    before={str(p):sha(p) for p in protected}
    parents=[];proposals=[]
    with tarfile.open(archive) as tar:
        official={}
        for split in ('train','test'):
            with np.load(io.BytesIO(tar.extractfile('shapenet-02828884/split/'+split+'.npz').read()),allow_pickle=True) as z:
                official.update({str(r['uid']):split for r in z['npz_list']})
        for r in chosen:
            member=tar.extractfile(r['archive_member']).read()
            assert hashlib.sha256(member).hexdigest()==r['member_sha256']
            with np.load(io.BytesIO(member),allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
            full=preprocess(raw)
            found,blocked=propose(full)
            parents.append(dict(parent_id=r['parent_id'],official_split=official.get(r['parent_id'],'UNKNOWN'),N=r['N'],
                source_duplicate_excess=r['source_ID_duplicate_excess_faces'],source_exact_zero_area=r['exact_zero_area_faces_original_float64'],
                planar_region_count=len(planar_regions(full)),proposal_count=len(found),blocked=blocked,
                status='PROPOSAL_ONLY_NO_GEOMETRY_OR_VISUAL_QUALIFICATION',source_member=r['archive_member']))
            for kind,i,ids,evidence,uncertainty in found:
                proposals.append(dict(parent_id=r['parent_id'],official_split=official.get(r['parent_id'],'UNKNOWN'),proposed_part_type=kind,index=i+1,
                    K=len(ids),N=int(r['N']),source_face_ids=ids,construction_evidence=evidence,uncertainty=uncertainty,
                    **topology(full,ids),status='PENDING_FINITE_GEOMETRY_AND_ACTUAL_AI_PROXY_REVIEW'))
    after={str(p):sha(p) for p in protected}
    assert before==after
    doc={'status':'READ_ONLY_FEASIBILITY_COMPLETE','parents':parents,'proposals':proposals,
        'summary':{'unselected_parents':len(parents),'parents_with_proposals':sum(r['proposal_count']>0 for r in parents),
            'proposals':len(proposals),'new_training_rows':0,'new_model_calls':0,'GPU_computation':False,'source_mutations':0,
            'wall_seconds':time.time()-start},'source_hashes_unchanged':before,
        'warning':'Do not call geometric proposals semantic data; requires frozen detector qualification and actual visual review before publication.'}
    (HERE/'candidate_probe.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(doc['summary'],indent=2));print(json.dumps(parents,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
