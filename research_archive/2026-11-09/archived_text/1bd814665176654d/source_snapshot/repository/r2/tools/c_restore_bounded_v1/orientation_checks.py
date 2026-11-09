"""Pure-CPU fixed-C XOR tests; no real meshes, models, or quality scoring."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.tools.c_restore_bounded_v1 import orientation as O
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_c_restore_bounded_v1/orientation_checks'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(p):
    p=Path(p).absolute()
    if any(q.is_symlink() for q in (p,*p.parents)):raise ValueError('Symlink output')
    p=p.resolve()
    if p==OUT.resolve() or not p.is_relative_to(OUT.resolve()):raise ValueError('Outside owned checks')
    return p
def atomic(p,fn,binary=False):
    p=checked(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb' if binary else 'w',**({} if binary else {'encoding':'utf-8'})) as f:
        fn(f);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def write(p,o):atomic(p,lambda f:json.dump(o,f,indent=2,allow_nan=False))
def save(p,**a):atomic(p,lambda f:np.savez_compressed(f,**a),True)
def resources():
    def run(args):
        r=subprocess.run(args,text=True,capture_output=True,timeout=20);return dict(code=r.returncode,stdout=r.stdout,stderr=r.stderr)
    return dict(cpu_cores=os.cpu_count(),RAM=Path('/proc/meminfo').read_text(),loadavg=list(os.getloadavg()),
        gpu=run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        gpu_processes=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        cpu_processes=run(['ps','-eo','pid,pcpu,pmem,comm','--sort=-pcpu']))

def fixtures():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,2,0],[0,0,1]],np.float32)
    rows=[dict(name='same_directed_CF_interface',vertices=v,faces=np.array([[0,1,2],[1,2,3]],np.int64),K=1),
          dict(name='two_C_anchors_conflict_through_free',vertices=v,faces=np.array([[0,1,2],[3,2,4],[1,2,3]],np.int64),K=2)]
    # A three-quad triangulated Mobius strip. The cycle, not fixed C, is
    # contradictory. These synthetic positions only avoid zero-area faces.
    mv=[]
    for i in range(3):
        theta=2*np.pi*i/3
        for w in (-.25,.25):
            r=1+w*np.cos(theta/2);mv.append([r*np.cos(theta),r*np.sin(theta),w*np.sin(theta/2)])
    mf=[]
    for i in range(3):
        a,b=2*i,2*i+1;an,bn=(2*(i+1),2*(i+1)+1) if i<2 else (1,0)
        mf.extend([[a,b,an],[b,bn,an]])
    rows.append(dict(name='nonorientable_cycle_conflict',vertices=np.array(mv,np.float32),faces=np.array(mf,np.int64),K=0))
    rows.append(dict(name='nonmanifold_three_incident_faces',vertices=v,faces=np.array([[0,1,2],[1,0,3],[0,1,5]],np.int64),K=1))
    uv=np.concatenate((v[:4],v[:4]+np.array([3,0,0],np.float32)))
    rows.append(dict(name='disconnected_unanchored_component',vertices=uv,faces=np.array([[0,1,2],[1,2,3],[4,5,6],[5,6,7]],np.int64),K=1))
    rows.append(dict(name='degenerate_face_exclusion',vertices=v,faces=np.array([[0,1,2],[0,0,1]],np.int64),K=1))
    sv=v.copy();sv[0,0]=np.float32(-0.0)
    rows.append(dict(name='C_byte_order_and_signed_zero_preservation',vertices=sv,faces=np.array([[2,0,1],[1,2,3]],np.int32),K=1))
    return rows

def check(name,r,faces,K):
    s=r['summary'];fl=r['flips']
    if name in ('same_directed_CF_interface','C_byte_order_and_signed_zero_preservation'):
        assert fl.tolist()==[False,True] and s['opposite_C_F_edges_before']==0 and s['opposite_C_F_edges_after']==1
    elif name=='two_C_anchors_conflict_through_free':
        assert s['parity_conflict_components']==0 and s['fixed_C_anchor_conflict_components']==1
        assert not fl.any() and np.array_equal(r['faces'],faces)
    elif name=='nonorientable_cycle_conflict':
        assert s['parity_conflict_components']==1 and s['fixed_C_anchor_conflict_components']==0
        assert s['fixed_C_anchor_status_not_evaluated_due_to_parity_components']==1
        assert not fl.any() and np.array_equal(r['faces'],faces)
    elif name=='nonmanifold_three_incident_faces':
        assert s['nonmanifold_edges']==1 and s['eligible_two_incidence_edges']==0
        assert s['whole_indexed_edge_orientation_status']=='NA_NONMANIFOLD_OR_DEGENERATE'
        assert all(c['touches_nonmanifold_edge'] for c in r['component_rows'])
    elif name=='disconnected_unanchored_component':
        assert fl.tolist()==[False,True,False,True] and s['unanchored_nondegenerate_components']==1
        c=next(c for c in r['component_rows'] if c['C_face_count']==0)
        assert c['root_face']==2 and c['gauge']==0 and not c['outward_orientation_certified']
    elif name=='degenerate_face_exclusion':
        assert s['degenerate_faces']==1 and not fl.any()
        assert any('DEGENERATE_INCIDENT_FACE' in e['exclusion_reasons'] for e in r['edge_rows'])
    assert r['faces'][:K].tobytes()==faces[:K].tobytes() and not fl[:K].any()

def main():
    completion=OUT/'completion.json';sources={str(Path(O.__file__)):sha(O.__file__),str(Path(__file__)):sha(__file__)}
    if completion.exists():
        old=json.loads(completion.read_text());assert old['status']=='PASS' and old['source_hashes']==sources
        for p,h in old['files'].items():assert sha(p)==h
        print(json.dumps(dict(status='VERIFIED_REUSE',cases=7,model_calls=0)));return
    if (OUT/'registration.json').exists():raise RuntimeError('Incomplete prior checks; no automatic replay')
    before=resources();start=time.perf_counter();data=fixtures()
    protected={str(p):sha(p) for p in (ROOT/'AGENTS.md',ROOT/'WORKSPACE.md',ROOT/'native_t1/docs/control_metrics.md')}
    write(OUT/'registration.json',dict(status='REGISTERED',source_hashes=sources,definition=O.definition(),
        protected=protected,cases=[r['name'] for r in data],CPU_workers=1,threads=2,
        rationale='Small deterministic analytic orientation fixtures, no GPU/model/real-mesh work'))
    write(OUT/'resources_before.json',before)
    results=[]
    for case in data:
        name=case['name'];v,f,K=case['vertices'],case['faces'],case['K'];vbits,fbits=v.tobytes(),f.tobytes()
        write(OUT/name/'started.json',dict(status='STARTED'))
        r=O.orient_fixed_C(v,f,K);check(name,r,f,K)
        assert v.tobytes()==vbits and f.tobytes()==fbits
        assert v[r['faces'][:K]].tobytes()==v[f[:K]].tobytes()
        assert np.array_equal(np.sort(r['faces'],axis=1),np.sort(f,axis=1))
        # Determinism and idempotence on these seven small authorized fixtures
        # are arithmetic checks, not extra model or real-mesh evaluation.
        repeat=O.orient_fixed_C(v,f,K);assert np.array_equal(repeat['faces'],r['faces']) and repeat['component_rows']==r['component_rows']
        again=O.orient_fixed_C(v,r['faces'],K);assert np.array_equal(again['faces'],r['faces'])
        save(OUT/name/'arrays.npz',vertices=v,original_faces=f,repaired_faces=r['faces'],flips=r['flips'],face_component_ids=r['face_component_ids'],face_corner_permutation=r['face_corner_permutation'])
        write(OUT/name/'result.json',dict(summary=r['summary'],component_rows=r['component_rows'],edge_rows=r['edge_rows']))
        results.append(dict(case=name,status='PASS',flipped_free_faces=r['summary']['flipped_free_faces'],parity_conflict_components=r['summary']['parity_conflict_components'],fixed_C_anchor_conflict_components=r['summary']['fixed_C_anchor_conflict_components'],nonmanifold_edges=r['summary']['nonmanifold_edges'],C_bytes_preserved=True,deterministic=True,idempotent=True))
    for p,h in protected.items():assert sha(p)==h
    files={str(p):sha(p) for p in OUT.rglob('*') if p.is_file()}
    done=dict(status='PASS',cases=results,source_hashes=sources,files=files,protected_unchanged=True,
        model_calls=0,real_mesh_operations=0,new_queries=0,seconds=time.perf_counter()-start,peak_CPU_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(completion,done);print(json.dumps({k:v for k,v in done.items() if k not in ('files','source_hashes')}))
if __name__=='__main__':main()
