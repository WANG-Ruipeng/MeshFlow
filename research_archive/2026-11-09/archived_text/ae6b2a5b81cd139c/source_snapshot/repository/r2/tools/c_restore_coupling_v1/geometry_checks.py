"""Four fixed CPU-only analytic controls; no model/quality/GEO calls."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.tools.c_restore_coupling_v1 import geometry as G
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_c_restore_coupling_v1/geometry_checks'
L=9.208138264587333

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(p):
    p=Path(p).absolute()
    if any(q.is_symlink() for q in (p,*p.parents)):raise ValueError('Symlink write')
    p=p.resolve()
    if p==OUT.resolve() or not p.is_relative_to(OUT.resolve()):raise ValueError('Write outside owned geometry checks')
    return p
def atomic(p,fn,binary=False):
    p=checked(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb' if binary else 'w',**({} if binary else {'encoding':'utf-8'})) as f:
        fn(f);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def write(p,obj):atomic(p,lambda f:json.dump(obj,f,indent=2,allow_nan=False))
def save(p,**a):atomic(p,lambda f:np.savez_compressed(f,**a),True)
def resources():
    def cmd(a):
        r=subprocess.run(a,capture_output=True,text=True,timeout=20);return dict(code=r.returncode,stdout=r.stdout,stderr=r.stderr)
    return dict(cpu_cores=os.cpu_count(),RAM=Path('/proc/meminfo').read_text(),loadavg=list(os.getloadavg()),
        gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        cpu_processes=cmd(['ps','-eo','pid,pcpu,pmem,comm','--sort=-pcpu']))

def cases():
    c=(L*np.array([[[0,0,0],[.2,0,0],[0,.2,0]]])).astype(np.float32);ids=np.array([[0,1,2]],np.int64)
    shift=np.array([.01*L,0,0],np.float32)
    prop=np.concatenate((c+shift,c+shift),axis=0)
    common=dict(case='common_translation',C=c,proposal=prop,ids=ids,reference_free=c.copy(),
        expectation='Matching C/free proposal corners jointly translated; ridge leaves measurable residual, no exact-weld promise')
    free=(c+np.array([0,0,.03*L],np.float32));identity=dict(case='identity_zero_error',C=c,proposal=np.concatenate((c,free)),ids=ids,
        expectation='b=0 implies exactly zero field and bitwise unchanged free')
    outside=dict(case='outside_compact_support',C=c,proposal=np.concatenate((c+shift,free+np.array([0,0,2*L],np.float32))),ids=ids,
        expectation='Every free corner beyond radius from all proposal anchors; zero field and bitwise unchanged free')
    dup_c=np.concatenate((c,(L*np.array([[[0,0,0],[-.2,0,0],[0,-.2,0]]])).astype(np.float32)))
    dup_ids=np.array([[0,1,2],[0,3,4]],np.int64);dup_prop=dup_c.copy()
    dup_prop[0,0]=[.001*L,0,0];dup_prop[1,0]=[-.001*L,0,0]
    # Two distinct source targets have identical proposal centers: deliberately
    # incompatible displacement observations; same-source copies also split.
    dup_prop[0,1]=[.15*L,0,0];dup_prop[1,1]=dup_prop[0,1]
    dup_free=np.array([[[.15*L,0,0],[.001*L,0,0],[-.001*L,0,0]]],np.float32)
    duplicate=dict(case='duplicate_split_and_conflicting_targets',C=dup_c,proposal=np.concatenate((dup_prop,dup_free)),ids=dup_ids,
        expectation='Keep all 6 observations, source ID0 total weight1; conflicting coincident centers cannot fit exactly; output finite and capped')
    return [common,identity,outside,duplicate]

def main():
    completion=OUT/'completion.json'
    sources={str(Path(G.__file__)):sha(G.__file__),str(Path(__file__)):sha(__file__)}
    if completion.exists():
        done=json.loads(completion.read_text());assert done['status']=='PASS' and done['source_hashes']==sources
        for p,h in done['files'].items():assert sha(p)==h
        print(json.dumps(dict(status='VERIFIED_REUSE',checks=4,model_calls=0)));return
    if (OUT/'registration.json').exists():raise RuntimeError('Attempt already registered without completion; no silent replay')
    before=resources();start=time.perf_counter()
    protected={str(p):sha(p) for p in (ROOT/'AGENTS.md',ROOT/'WORKSPACE.md',ROOT/'native_t1/docs/control_metrics.md')}
    controls=cases()
    write(OUT/'registration.json',dict(status='REGISTERED',definition=G.definition(),L=L,source_hashes=sources,protected=protected,
        cases=[dict(case=c['case'],expectation=c['expectation']) for c in controls],budget=dict(CPU_cases=4,model_calls=0,new_noise=0),
        concurrency=1,threads=2,reason='Four tiny dependent evidence writes; pure analytic CPU arithmetic, no GPU allocation'))
    write(OUT/'resources_before.json',before)
    rejected=[]
    for p in (ROOT/'AGENTS.md',ROOT/'experiments/r2_postprocess_avmr_pmr_v1/registration.json'):
        try:checked(p)
        except ValueError:rejected.append(str(p))
        else:raise AssertionError('Output guard accepted old file')
    write(OUT/'write_guard_check.json',dict(status='PASS',rejected_before_open=rejected))
    results=[]
    for item in controls:
        name=item['case'];c,p,ids=item['C'],item['proposal'],item['ids'];K=len(c)
        before_arrays=(c.tobytes(),p.tobytes(),ids.tobytes())
        write(OUT/name/'started.json',dict(status='STARTED',case=name))
        output,diag,details=G.coordinate_restore(c,p,ids,L=L)
        assert before_arrays==(c.tobytes(),p.tobytes(),ids.tobytes())
        assert output[:K].tobytes()==c.tobytes() and np.isfinite(output).all()
        assert np.linalg.norm(details['free_actual_displacement_FP32'],axis=1).max()<=G.FREE_CAP_L*L
        if name in ('identity_zero_error','outside_compact_support'):
            assert output[K:].tobytes()==p[K:].tobytes()
            assert np.count_nonzero(details['free_displacement_unclipped'])==0
        if name=='common_translation':
            err=np.linalg.norm(output[K:].astype(float)-item['reference_free'],axis=-1)
            base=np.linalg.norm(p[K:].astype(float)-item['reference_free'],axis=-1)
            assert np.all(err<base)
            diag['known_common_translation_error']=dict(mean=float(err.mean()),max=float(err.max()),exact_bits=output[K:].tobytes()==item['reference_free'].tobytes(),baseline_mean=float(base.mean()))
        if name=='duplicate_split_and_conflicting_targets':
            assert diag['anchor_corners']==6 and diag['source_vertices']==5
            assert next(x for x in diag['duplicates'] if x['source_vertex_id']==0)['total_weight']==1
            assert diag['anchor_fit_residual_norm']['max']>0 and diag['split_source_vertex_groups']==1
            assert diag['clipped_free_corners']>0
        save(OUT/name/'input_and_output.npz',C=c,proposal=p,C_source_vertex_ids=ids,output=output,**details)
        write(OUT/name/'diagnostic.json',diag)
        results.append(dict(case=name,status='PASS',scope='Arithmetic/cap/provenance controls, not mesh attachment success',
            anchor_residual_max=diag['anchor_fit_residual_norm']['max'],clipped_free_corners=diag['clipped_free_corners'],system_condition_number_2=diag['system_condition_number_2']))
    for p,h in protected.items():assert sha(p)==h
    files={str(p):sha(p) for p in OUT.rglob('*') if p.is_file()}
    done=dict(status='PASS',cases=results,source_hashes=sources,protected_unchanged=True,files=files,model_calls=0,new_noise=0,
        seconds=time.perf_counter()-start,peak_CPU_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(completion,done);print(json.dumps({k:v for k,v in done.items() if k not in ('files','source_hashes')}))
if __name__=='__main__':main()
