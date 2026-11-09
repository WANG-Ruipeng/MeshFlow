"""One actual production CPU Native OT per candidate, no model forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,time,sys
from pathlib import Path
import numpy as np
from r2.data_tools.bench_seat_coverage_v1.inspect import ROOT,OUT
sys.path.insert(0,str(ROOT/'r2/src'))
from r2.data_tools.bench_user_v2 import build as b
from r2.pilots.bench_full_seat_curriculum_v1.data import check_case
from r2.pilots.bench_parts_convergence_v1.data import supervised_sample
from r2.data_tools.bench_user_relation_rebuild_v1.run import resources
def main():
    m=b.read(OUT/'candidate_manifest.json');results=[];start=time.perf_counter()
    for row in m['conditions']:
        dest=OUT/'validation'/row['condition_id'];rp=dest/'receipt.json'
        if rp.exists():
            r=b.read(rp);assert r['case_sha256']==b.sha(row['case_path']);results.append(r);continue
        dest.mkdir(parents=True,exist_ok=True);b.save(dest/'attempt.json',dict(case_sha256=row['case_sha256'],started=time.time(),resources=resources(),purpose='Candidate Native CPU packing, not training or semantic acceptance'))
        identity,case=check_case(row);n,k=int(case['N']),int(case['K']);seed=int(hashlib.sha256(('BENCH_SEAT_COVERAGE_CPU_ONLY:'+row['condition_id']).encode()).hexdigest()[:8],16)
        z=np.random.RandomState(seed).normal(size=(n-k,3,3)).astype(np.float32);eps=np.random.RandomState(seed^0xA37).normal(size=z.shape).astype(np.float32)
        b.npz(dest/'validation_noise_only.npz',dict(z=z,epsilon=eps,seed=np.asarray(seed)))
        t=time.perf_counter();s=supervised_sample(case,z,eps,.5,seed^0xCA7,'PLAIN');wall=time.perf_counter()-t
        fixed=s['anchor_ids']>=0;known=s['known_mask'];ids=s['source_face_ids'];cp=s['source_corner_permutations']
        assert np.array_equal(np.sort(ids),np.arange(n))
        assert np.array_equal(s['x1'],case['full_target'][ids[:,None],cp])
        assert fixed[known].all() and not fixed[~known].any()
        assert np.array_equal(s['u'],s['x1']-s['x0'])
        assert np.array_equal(s['xt'],np.float32(.5)*s['x0']+np.float32(.5)*s['x1'])
        assert np.array_equal(s['xt'][fixed],s['x1'][fixed]) and not s['u'][fixed].any()
        assert int(s['free_coordinate_count'])==9*(n-k)
        order=s['face_permutation'];mixed=np.float32(np.sqrt(.75))*z+np.float32(.5)*eps
        x0=np.zeros_like(s['x0']);x0[k:]=mixed;orig_anchor=np.empty_like(s['anchor_ids']);orig_anchor[order]=s['anchor_ids'];fm=orig_anchor>=0;x0[fm]=s['C_vertices'][orig_anchor[fm]]
        assert np.array_equal(s['x0'],x0[order]);assert np.isfinite(s['xt']).all()
        b.npz(dest/'native_cpu_sample.npz',s)
        r=dict(condition_id=row['condition_id'],parent_id=row['parent_id'],status='PASS',case_sha256=row['case_sha256'],identity=identity,production_CPU_OT=1,CPU_noise_arrays=2,model_forwards=0,GPU_compute=0,training_updates=0,N=n,K=k,free_FM_scalars=9*(n-k),target_faces_once=True,C_fixed=True,OT_then_HYBRID025=True,wall_seconds=wall,semantic_acceptance=False)
        b.save(rp,r);results.append(r)
    for p,h in b.read(OUT/'protected_source_hashes.json').items():assert b.sha(p)==h
    b.save(OUT/'validation'/'summary.json',dict(status='PASS_NATIVE_ARRAYS_NOT_SEMANTIC_ACCEPTANCE',conditions=results,total_CPU_OT=sum(r['production_CPU_OT'] for r in results),model_forwards=0,training_updates=0,seconds=time.perf_counter()-start))
    print(json.dumps(dict(pass_count=len(results),CPU_OT=len(results),model_forwards=0)))
if __name__=='__main__':main()
