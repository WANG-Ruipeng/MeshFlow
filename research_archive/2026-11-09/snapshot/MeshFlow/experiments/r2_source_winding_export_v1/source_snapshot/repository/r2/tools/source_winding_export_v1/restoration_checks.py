"""Eight finite CPU arithmetic checks; no real POST export or orientation solve."""
from pathlib import Path
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib, itertools, json, resource, subprocess, sys, time, traceback
import numpy as np
from . import restoration as R

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_source_winding_export_v1/restoration_checks'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def checked(path):
    p=Path(path).absolute()
    if any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('Symlink output')
    p=p.resolve()
    if p==OUT.resolve() or not p.is_relative_to(OUT.resolve()):raise ValueError('Output outside owned checks directory')
    return p

def write(path,value):
    p=checked(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(p)

def resources():
    def run(args):
        p=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return dict(code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    return dict(CPU_cores=os.cpu_count(),loadavg=list(os.getloadavg()),memory=Path('/proc/meminfo').read_text(),
        GPU=run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        GPU_processes=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        CPU_processes=run(['ps','-eo','pid,pcpu,pmem,comm','--sort=-pcpu']))

def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def fixture():
    # Six known triangles cover all S3 permutations; distinct post vertex
    # numbering and nontrivial known source-face order prevent accidental IDs.
    vertex=[]
    for i in range(8):vertex += [[i*2.,-0.,0.],[i*2.+1.,0.,0.],[i*2.,1.,.25]]
    vertex += [vertex[0].copy()]  # same-coordinate different source ID
    pv=np.asarray(vertex,np.float32);original=np.arange(24,dtype=np.int64).reshape(8,3)
    P=np.asarray(list(itertools.permutations(range(3)))+[(0,1,2),(0,1,2)],np.int64)
    ids=R.permute_corner_data(original,P);M=pv[ids]
    parent=dict(full_target=M.reshape(8,9).copy(),model_vertices=pv,original_source_faces=original.copy(),
        retained_to_original_face=np.arange(8,dtype=np.int64),source_faces=original.copy(),
        source_face_vertex_ids=ids,source_corner_permutations=P)
    chosen=np.asarray([4,1,5,0,3,2],np.int64)
    condition=dict(C=M[chosen].copy(),source_face_ids=chosen.copy(),original_source_face_ids=chosen.copy(),
        C_source_vertex_ids=ids[chosen].copy(),C_source_corner_permutations=P[chosen].copy(),
        K=np.asarray(6,np.int64),N=np.asarray(8,np.int64))
    vorder=np.arange(len(pv)-1,-1,-1);v=pv[vorder].copy();remap=np.argsort(vorder)
    f=remap[np.concatenate((ids[chosen],ids[6:]))].astype(np.int32)
    sf=np.arange(8,dtype=np.int64);sc=np.tile(np.arange(3,dtype=np.int64),(8,1))
    return v,f,6,condition,parent,sf,sc

def run_fixture(data):
    v,f,K,c,p,sf,sc=data
    return R.restore_source_order(v,f,K,c,p,source_face_ids=sf,source_corner_ids=sc)

def expect_rejection(data,substring):
    try:run_fixture(data)
    except ValueError as e:
        assert substring in str(e),(substring,str(e))
        return dict(expected_exception='ValueError',message=str(e))
    raise AssertionError('Expected source contract rejection')

def all_six():
    data=fixture();v,f,K,c,p,sf,sc=data
    original={name:x.tobytes() for name,x in [('vertices',v),('faces',f),('sf',sf),('sc',sc)]}
    r=run_fixture(data);q=run_fixture(data)
    expected=p['model_vertices'][p['original_source_faces'][c['original_source_face_ids']]]
    assert bits(r['C'],expected) and np.array_equal(r['C_original_source_vertex_ids'],p['original_source_faces'][c['original_source_face_ids']])
    assert bits(r['faces'][K:],f[K:]) and bits(r['source_face_ids'],sf)
    assert bits(r['faces'],q['faces']) and r['face_rows']==q['face_rows']
    for name,x in [('vertices',v),('faces',f),('sf',sf),('sc',sc)]:assert x.tobytes()==original[name]
    assert r['summary']['C_rows_reordered']==5 and r['summary']['C_winding_reversed_from_canonical']==3
    return dict(permutations=6,exact_source_vertex_IDs=True,nontrivial_source_face_order_retained=True,
        input_arrays_unchanged=True,free_rows_unchanged=True,deterministic=True,summary=r['summary'])

def cyclic_inverse():
    data=fixture();c=data[3];r=run_fixture(data);rows=[]
    for j,P in enumerate(c['C_source_corner_permutations']):
        if P.tolist() not in ([1,2,0],[2,0,1]):continue
        Q=r['P_inverse'][j];assert not np.array_equal(P,Q)
        assert np.array_equal(P[Q],np.arange(3)) and np.array_equal(Q[P],np.arange(3))
        assert bits(c['C'][j][Q],r['C'][j]);rows.append(dict(P=P.tolist(),P_inverse=Q.tolist()))
    assert len(rows)==2
    return dict(three_cycles=rows,forward_permutation_is_not_its_inverse=True)

def parity_normals():
    data=fixture();c=data[3];r=run_fixture(data)
    def normals(t):
        t=t.astype(np.float64);return np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    nm,nr=normals(c['C']),normals(r['C']);odd=R.permutation_parity(c['C_source_corner_permutations'])
    assert np.array_equal(nr,np.where(odd[:,None],-nm,nm))
    assert int(odd.sum())==3 and int((~odd).sum())==3
    return dict(odd_reversals=3,even_preservations=3,normal_sign_matches_parity=True)

def composed_attributes():
    data=fixture();v,f,K,c,p,sf,sc=data;r=run_fixture(data)
    second=np.tile(np.arange(3),(len(f),1));second[K:]=[0,2,1]
    final=R.compose_corner_permutations(r['face_corner_permutation'],second)
    attrs=np.arange(len(f)*3*2,dtype=np.float32).reshape(len(f),3,2);attrs[0,0,0]=np.float32(-0.)
    sequential=R.permute_corner_data(R.permute_corner_data(attrs,r['face_corner_permutation']),second)
    composed=R.permute_corner_data(attrs,final);assert bits(sequential,composed)
    final_sc=R.permute_corner_data(r['source_corner_ids'],second)
    assert np.array_equal(final_sc,R.permute_corner_data(sc,final))
    assert bits(final[:K],r['face_corner_permutation'][:K])
    assert np.signbit(composed[0][np.flatnonzero(final[0]==0)[0],0])
    return dict(trailing_attributes_bitwise=True,signed_zero_preserved=True,source_corner_IDs_composed=True,C_mapping_unchanged_by_free_stage=True)

def coincident_ID_mismatch():
    data=fixture();v,f,K,c,p,sf,sc=data
    where=np.argwhere(c['C_source_vertex_ids']==0)[0];i,j=map(int,where)
    assert bits(p['model_vertices'][0],p['model_vertices'][-1])
    c['C_source_vertex_ids'][i,j]=len(p['model_vertices'])-1
    return expect_rejection(data,'Condition source vertex ID mismatch')

def parent_lineage_mismatch():
    data=fixture();data[4]['retained_to_original_face'][[0,1]]=[1,0]
    return expect_rejection(data,'Parent retained face lineage mismatch')

def invalid_permutation():
    data=fixture();data[3]['C_source_corner_permutations'][0]=[0,0,2]
    return expect_rejection(data,'must contain each of 0,1,2')

def noncanonical_POST():
    data=fixture();data[1][0]=data[1][0,[0,2,1]]
    return expect_rejection(data,'POST C is not in exact registered canonical coordinate order')

CASES=[('all_six_permutations_exact_identity',all_six),('three_cycle_inverse_not_forward',cyclic_inverse),
       ('even_and_odd_parity_normals',parity_normals),('per_corner_composition_signed_zero',composed_attributes),
       ('coincident_coordinate_source_ID_mismatch_rejected',coincident_ID_mismatch),
       ('parent_original_face_lineage_mismatch_rejected',parent_lineage_mismatch),
       ('malformed_saved_permutation_rejected',invalid_permutation),('noncanonical_POST_order_rejected',noncanonical_POST)]

def main():
    sources={str(Path(R.__file__)):sha(R.__file__),str(Path(__file__)):sha(__file__)}
    completion=OUT/'completion.json'
    if completion.exists():
        done=json.loads(completion.read_text());assert done['status']=='PASS' and done['source_hashes']==sources
        for p,h in done['files'].items():assert sha(p)==h
        print(json.dumps(dict(status='VERIFIED_REUSE',cases=8,model_calls=0)));return
    if (OUT/'registration.json').exists():raise RuntimeError('Incomplete attempted checks; no silent retry')
    measured=resources();start=time.perf_counter()
    protected={str(p):sha(p) for p in (ROOT/'AGENTS.md',ROOT/'WORKSPACE.md',ROOT/'native_t1/docs/control_metrics.md',ROOT/'r2/tools/c_restore_bounded_v1/orientation.py')}
    write(OUT/'registration.json',dict(status='REGISTERED',source_hashes=sources,protected=protected,
        cases=[name for name,_ in CASES],definition=R.definition(),CPU_workers=1,CPU_threads=2,GPU_workers=0,
        real_POST_exports=0,orientation_solver_calls=0,model_calls=0))
    write(OUT/'resources_before.json',measured);results=[]
    try:
        for name,fn in CASES:
            write(OUT/name/'started.json',dict(status='STARTED'))
            detail=fn();value=dict(case=name,status='PASS',**detail);write(OUT/name/'result.json',value);results.append(value)
        for p,h in protected.items():assert sha(p)==h
        files={str(p):sha(p) for p in OUT.rglob('*') if p.is_file()}
        done=dict(status='PASS',cases=results,source_hashes=sources,protected_before=protected,
            protected_after={p:sha(p) for p in protected},protected_unchanged=True,files=files,
            seconds=time.perf_counter()-start,peak_CPU_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            real_POST_exports=0,orientation_solver_calls=0,model_calls=0,GEO_calls=0,quality_calls=0,
            torch_imported='torch' in sys.modules)
        write(completion,done);print(json.dumps({k:v for k,v in done.items() if k not in ('files','source_hashes','protected_before','protected_after')},indent=2))
    except BaseException:
        write(OUT/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),completed_cases=results,
            model_calls=0,real_POST_exports=0,orientation_solver_calls=0))
        raise

if __name__=='__main__':main()
