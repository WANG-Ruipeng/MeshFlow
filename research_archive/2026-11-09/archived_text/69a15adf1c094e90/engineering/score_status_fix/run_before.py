"""Resumable CPU-only START post-processing experiment. No model dispatch."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, subprocess, time, traceback
from pathlib import Path
from datetime import datetime, timezone
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'native_t1/runs/chair_start_postprocess_v1'
SOURCE=ROOT/'native_t1/runs/chair_capped500_half_v1'
CODE=Path(__file__).parent
RADIUS=.015/(.95*.3762)
START_SHA='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.partial')
    temp.write_text(json.dumps(clean(v),ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    temp.replace(p)
def freeze(p,v):
    if Path(p).exists():
        if read(p)!=clean(v):raise ValueError('Frozen file differs: '+str(p))
    else:write(p,v)
def event(kind,**v):
    p=OUT/'ledger/events.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:
        f.write(json.dumps(clean(dict(utc=datetime.now(timezone.utc).isoformat(),event=kind,**v)),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save_npz(p,**v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        old=arrays(p)
        if set(old)!=set(v) or any(old[k].dtype!=v[k].dtype or old[k].shape!=v[k].shape or old[k].tobytes()!=v[k].tobytes() for k in old):
            raise ValueError('Frozen NPZ differs: '+str(p))
    else:np.savez_compressed(p,**v)
def copy(src,dst):
    src,dst=Path(src),Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if digest(src)!=digest(dst):raise ValueError('Frozen copy differs')
    else:shutil.copyfile(src,dst)
def csv_write(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    keys=sorted({k for r in rows for k in r})
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple,np.ndarray)) else clean(v) for k,v in r.items()})

def resources(tag):
    import psutil
    commands={
        'gpu':'nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv',
        'gpu_processes':'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv',
        'cpu_processes':'ps -eo pid,pcpu,pmem,comm --sort=-pcpu'}
    snap=dict(utc=datetime.now(timezone.utc).isoformat(),cpu_logical=psutil.cpu_count(),cpu_available_affinity=len(os.sched_getaffinity(0)),
        memory=psutil.virtual_memory()._asdict(),cpu_percent=psutil.cpu_percent(interval=.2),self_rss=psutil.Process().memory_info().rss)
    for key,cmd in commands.items():
        q=subprocess.run(cmd.split(),capture_output=True,text=True);snap[key]=dict(returncode=q.returncode,stdout=q.stdout[:16000],stderr=q.stderr[:2000])
    write(OUT/'resources'/f'{tag}.json',snap)
    return snap

def prepare():
    if (OUT/'cases.json').exists():
        verify_protected();return dict(status='REUSED',cases=len(read(OUT/'cases.json')))
    OUT.mkdir(parents=True,exist_ok=True);resources('before_experiment')
    protected={};cases=[];ids=set()
    def protect(p,expected=None):
        p=Path(p);actual=digest(p)
        if expected and actual!=expected:raise ValueError('Source hash changed: '+str(p))
        protected[str(p)]=actual
    for panel in ('OLD','FRESH'):
        mp=SOURCE/'manifests'/f'{panel}.json';bp=SOURCE/'evaluation/START'/panel/'per_output.json'
        protect(mp);protect(bp);m=read(mp)['rows'];base={r['request_id']:r for r in read(bp)}
        if len(m)!=16 or len(base)!=16:raise ValueError('Expected sixteen frozen pairs per panel')
        for request in m:
            rid=request['request_id'];row=base[rid];cid=request['condition_id'];K=int(request['K'])
            if row['status']!='PASS' or row['checkpoint_sha256']!=START_SHA:raise ValueError('Not the exact START cohort')
            for key,hkey in (('condition_npz','condition_sha256'),('bank_npz','bank_sha256')):protect(request[key],request[hkey])
            for key in ('raw','result','vectors','control_vectors'):
                protect(row[key+'_path'],row[key+'_sha256'])
            protect(row['legacy_score_path'])
            raw=arrays(row['raw_path']);c=arrays(request['condition_npz'])
            output=raw['output'].reshape(-1,3,3)
            if output.dtype!=np.float32 or output.shape!=(request['N'],3,3) or output[:K].tobytes()!=c['C'].tobytes():raise ValueError('RAW/C contract differs')
            if raw['path'][:,:K].reshape(51,K,3,3).tobytes()!=np.broadcast_to(c['C'],(51,K,3,3)).copy().tobytes():raise ValueError('Original path C differs')
            localc=OUT/'inputs'/f'{cid}.npz';copy(request['condition_npz'],localc)
            localraw=OUT/'inputs/raw'/f'{rid}.npz';save_npz(localraw,output=output)
            for name in ('control.json','control.npz','quality.json','quality.npz'):
                p=SOURCE/'references'/cid/name;protect(p);copy(p,OUT/'references'/cid/name)
            if rid in ids:raise ValueError('Duplicate case')
            ids.add(rid)
            cases.append(dict(id=rid,parent=request['uid'],noise=request['candidate'],panel=panel,K=K,N=int(request['N']),
                C_path=str(localc),GT_path=str(localc),queries_path=str(localc),raw_path=str(localraw),original_raw_path=row['raw_path'],
                original_raw_sha256=row['raw_sha256'],geo_path=str(OUT/'outputs/GEO_C'/f'{rid}.npz'),runroot=str(OUT),
                request=request,baseline_row=row,baseline_per_output_path=str(bp),radius=RADIUS,L0=float(c['L0'])))
    if len(cases)!=32 or len({c['parent'] for c in cases})!=8:raise ValueError('Panel incomplete')
    for folder in (ROOT/'native_t1',ROOT/'experiments/mf_h1_local0'):
        for p in folder.rglob('*.py'):
            rel=p.relative_to(folder)
            if p.is_relative_to(CODE) or any(x in ('runs','maintenance','outputs','__pycache__') for x in rel.parts):continue
            protect(p)
    for p in (ROOT/'datasets/mesh_dataset.py',ROOT/'gradio_demo.py',ROOT/'AGENTS.md',ROOT/'WORKSPACE.md'):protect(p)
    protocol=dict(task='CHAIR_START_POSTPROCESS_COMPARISON_V1',status='FROZEN',labels=['RAW','GEO_C'],parents=8,outputs_per_label=32,
        budget=dict(training_updates=0,model_forwards=0,new_generator_samples=0,new_geometry_outputs=32,geometry_recipes=1,threshold_scan=0),
        denoiser=dict(status='NOT_RUN_ASSET_UNAVAILABLE',reason='No verified learned mesh-denoiser code/weight identity in maintained local assets or checked published README.'),
        algorithm=dict(radius=RADIUS,reference_radius=.015,reference_scale_from_model=.95*.3762,
            units='x_model=2*x_unitbox/.3762; reference convention x_ref=1.9*x_unitbox=.95*.3762*x_model. Fixed globally; no output-bbox normalization.',
            limitation='Paper-inspired nominal [-.95,.95] reference-space radius, not a verified reproduction of the authors full denoiser/welding runtime units.',
            C='All known triangles, corner order, winding and FP32 bits retained; distinct known anchors never merged.',
            CF='Exactly one distinct C anchor within radius: snap free corner; multiple: reject and keep corner immutable.',
            FF='Lexicographically ordered immutable original-point representatives; nearest existing within radius, no transitive chaining or averaging.',
            delete='Only free exact repeated-vertex / collinear triangles, and unordered exact duplicates with C preferred.',
            disabled=['hole_filling','smoothing','coordinate_quantization','clipping','learned_denoising','threshold_tuning','normal_winding_repair']),
        evaluation=dict(raw='32 identity-verified frozen score rows; original vectors retained',
            geometry='Original S/I/E and quality formulas applied to variable free-face count. Original N and final N both reported.',
            fixed_queries='Original C/GT/Gamma/directions files copied byte-for-byte; no new seeds or target queries.',
            predicted_surface_queries='Original fixed seeds and area-sampling algorithm applied to each output surface; positions change with surface, as required by original F-to-GT definition.',
            topology='True exported vertex-index connectivity plus exact-coordinate RAW connectivity diagnostic; distinct from geometric seam coverage.',
            inference='Descriptive on already observed eight DEV parents/four noises; no new-parent generalization or promotion.',
            hypothesis='Small bounded changes and connectivity cleanup may improve local seams/intersections without sacrificing C or main-shape fidelity.'),
        runtime=dict(device='CPU',threads_per_process=2,GPU_workers=0,initial_score_workers=1,additional_workers='Measured CPU/RAM after first registered case; cap four to leave headroom.'),
        stop='Exactly one fixed cleanup of the 32 retained START outputs; all evidence delivered, no default switch, training, generation, commit or push.',
        sources=['https://arxiv.org/html/2606.23489v2#S4.SS4','https://arxiv.org/html/2606.23489v2#A2.SS3'],
        source_cohort=str(SOURCE),start_checkpoint_sha256=START_SHA)
    freeze(OUT/'protocol.json',protocol);freeze(OUT/'protected_before.json',protected);freeze(OUT/'cases.json',cases)
    event('PREPARED',cases=32,protected_files=len(protected),protocol_sha256=digest(OUT/'protocol.json'))
    return dict(status='PASS',cases=32,protected_files=len(protected),radius=RADIUS)

def verify_protected():
    protected=read(OUT/'protected_before.json');bad=[]
    for p,h in protected.items():
        if not Path(p).exists() or digest(p)!=h:bad.append(p)
    result=dict(status='PASS' if not bad else 'FAIL',checked=len(protected),changed=bad)
    write(OUT/'audit/protected_after.json',result)
    if bad:raise ValueError('Protected sources changed')
    return result

def export_ply(path,vertices,faces,K):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    lines=['ply','format ascii 1.0',f'element vertex {len(vertices)}','property float x','property float y','property float z',
        f'element face {len(faces)}','property list uchar int vertex_indices','property uchar red','property uchar green','property uchar blue','end_header']
    lines += [' '.join(format(float(x),'.9g') for x in v) for v in vertices]
    lines += ['3 '+' '.join(str(int(x)) for x in f)+(' 45 125 220' if i<K else ' 235 145 55') for i,f in enumerate(faces)]
    text='\n'.join(lines)+'\n'
    if p.exists() and p.read_text()!=text:raise ValueError('Frozen PLY differs')
    if not p.exists():p.write_text(text,encoding='ascii')
    a=text.split('end_header\n',1)[1].splitlines()
    v=np.asarray([[float(x) for x in s.split()] for s in a[:len(vertices)]],dtype=np.float32)
    f=np.asarray([[int(x) for x in s.split()[1:4]] for s in a[len(vertices):]],dtype=np.int64)
    if v[f].tobytes()!=vertices[faces].tobytes():raise ValueError('PLY full geometry roundtrip failed')

def geometry_stage():
    from .geometry import process
    src={str(p):digest(p) for p in CODE.glob('*.py') if p.name in ('geometry.py','test_geometry.py','run.py')}
    freeze(OUT/'registration/geometry_source_identity_v2.json',src)
    tick=time.perf_counter();rows=[]
    for c in read(OUT/'cases.json'):
        marker=OUT/'geometry'/f"{c['id']}.json"
        if marker.exists():
            row=read(marker)
            if digest(c['geo_path'])!=row['geo_sha256']:raise ValueError('Geometry cache differs')
            rows.append(row);continue
        event('GEOMETRY_BEGIN',id=c['id'])
        raw=arrays(c['raw_path'])['output'];t=time.perf_counter();v=process(raw,c['K'],RADIUS)
        npvalues={k:x for k,x in v.items() if isinstance(x,np.ndarray)}
        save_npz(c['geo_path'],**npvalues)
        export_ply(OUT/'meshes/GEO_C'/f"{c['id']}.ply",v['vertices'],v['faces'],c['K'])
        rawv=raw.reshape(-1,3);rawf=np.arange(len(rawv),dtype=np.int64).reshape(-1,3)
        export_ply(OUT/'meshes/RAW'/f"{c['id']}.ply",rawv,rawf,c['K'])
        stats=v.get('stats',{})
        row=dict(id=c['id'],parent=c['parent'],noise=c['noise'],panel=c['panel'],seconds=time.perf_counter()-t,
            geo_sha256=digest(c['geo_path']),stats=stats,face_records=v.get('face_records',[]))
        write(marker,row);rows.append(row);event('GEOMETRY_DONE',id=c['id'],seconds=row['seconds'],geo_sha256=row['geo_sha256'])
    csv_write(OUT/'geometry/per_output_geometry.csv',[dict(id=r['id'],parent=r['parent'],noise=r['noise'],panel=r['panel'],**r['stats']) for r in rows])
    return dict(status='PASS',outputs=len(rows),wall_seconds=time.perf_counter()-tick,model_calls=0)

def scoring_job(case):
    import resource
    import torch
    def forbidden(*a,**kw):raise RuntimeError('Model execution/weight loading forbidden in CPU postprocess experiment')
    torch.nn.Module._call_impl=forbidden;torch.load=forbidden
    from .scoring import score_case
    marker=OUT/'status/scores'/f"{case['id']}.json"
    if marker.exists():return read(marker)
    geo=read(OUT/'geometry'/f"{case['id']}.json");case=dict(case,geometry_stats=geo['stats'],geometry_stats_path=str(OUT/'geometry'/f"{case['id']}.json"))
    tick=time.perf_counter();event('SCORE_BEGIN',id=case['id'])
    try:
        value=score_case(case,OUT)
        value.update(id=case['id'],wall_seconds=time.perf_counter()-tick,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if value.get('status')!='PASS':raise RuntimeError('Score returned non-PASS: '+str(value))
        write(marker,value);event('SCORE_DONE',id=case['id'],wall_seconds=value['wall_seconds'],maxrss_kib=value['maxrss_kib'])
        return value
    except Exception as e:
        failure=dict(id=case['id'],status='BLOCKED',reason=repr(e),traceback=traceback.format_exc())
        write(OUT/'failures'/f"score_{case['id']}.json",failure);event('SCORE_FAILED',**failure);raise

def scoring_stage():
    from concurrent.futures import ProcessPoolExecutor
    import psutil
    cases=read(OUT/'cases.json');tick=time.perf_counter()
    with ProcessPoolExecutor(max_workers=1) as pool:first=pool.submit(scoring_job,cases[0]).result()
    snap=resources('after_first_score')
    peak=max(first.get('maxrss_kib',0)*1024,512*1024*1024)
    slots=min(4,max(1,len(os.sched_getaffinity(0))//4),max(1,int((psutil.virtual_memory().available-4*1024**3)/(1.5*peak))))
    plan=dict(workers=slots,threads_per_worker=2,first_case=cases[0]['id'],first_maxrss_bytes=peak,
        cpu_available=len(os.sched_getaffinity(0)),available_memory=psutil.virtual_memory().available,
        rationale='One registered score measured before dispatch; cap uses CPU affinity, 1.5x observed RSS and 4GiB RAM headroom, at most four for interactive headroom.',
        first_wall_seconds=first['wall_seconds'],GPU_workers=0)
    write(OUT/'resources/score_dispatch.json',plan)
    with ProcessPoolExecutor(max_workers=slots) as pool:results=list(pool.map(scoring_job,cases[1:]))
    results=[first]+results;resources('after_scoring')
    rows=[r for value in results for r in value['rows']]
    write(OUT/'evaluation/per_output.json',rows);csv_write(OUT/'evaluation/per_output.csv',rows)
    return dict(status='PASS',cases=len(results),rows=len(rows),wall_seconds=time.perf_counter()-tick,workers=slots,new_model_calls=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','geometry','score','visualize','analyze','verify','all'));a=p.parse_args()
    selected=('prepare','geometry','score','visualize','analyze','verify') if a.stage=='all' else (a.stage,)
    for stage in selected:
        tick=time.perf_counter();event('STAGE_BEGIN',stage=stage)
        try:
            if stage=='prepare':result=prepare()
            elif stage=='geometry':result=geometry_stage()
            elif stage=='score':result=scoring_stage()
            elif stage=='visualize':
                from .visualization import run
                result=run(OUT)
            elif stage=='analyze':
                from .report import run
                result=run(OUT)
            else:result=verify_protected()
            receipt=dict(stage=stage,status='PASS',wall_seconds=time.perf_counter()-tick,result=result)
            write(OUT/'status'/f'{stage}.json',receipt);event('STAGE_DONE',**receipt);print(json.dumps(clean(receipt),ensure_ascii=False),flush=True)
        except Exception as e:
            event('STAGE_FAILED',stage=stage,reason=repr(e));raise

if __name__=='__main__':main()
