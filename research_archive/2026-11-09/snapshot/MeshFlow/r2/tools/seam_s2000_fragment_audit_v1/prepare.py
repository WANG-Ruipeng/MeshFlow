"""Freeze a CPU-only, saved-geometry diagnostic; no model imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import json, hashlib, subprocess, time
from pathlib import Path
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'experiments/r2_seam_quality_wave1_wave2_v1'
OUT=ROOT/'experiments/r2_seam_s2000_fragment_audit_v1'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    resources=dict(unix=time.time(),cpu_logical=psutil.cpu_count(),memory=psutil.virtual_memory()._asdict(),
        loadavg=os.getloadavg(),cpu_percent_sample=psutil.cpu_percent(interval=1),gpu_scientific_workers=0,
        processes=[dict(pid=p.pid,name=p.info['name'],rss=p.info['memory_info'].rss)
                   for p in psutil.process_iter(['name','memory_info']) if 'python' in (p.info['name'] or '').lower()])
    for key,cmd in [('gpu',['nvidia-smi']),('gpu_processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'])]:
        q=subprocess.run(cmd,capture_output=True,text=True); resources[key]=dict(code=q.returncode,stdout=q.stdout,stderr=q.stderr)
    write(OUT/'resources_initial.json',resources)
    qs=json.loads((SRC/'protocol.json').read_text())['evaluation']['requests']
    items=[]; hashes={}
    for q in qs:
        folder=SRC/'evaluation/PLAIN_S2000'/q['request_id']
        paths=dict(RAW=folder/'raw/raw.npz',D0=folder/'post/proposal.npz',RESTORED=folder/'post/pre_geo.npz',
            POST=folder/'post/post.npz',CANONICAL=folder/'post/post_canonical.npz',geometry=folder/'post/geometry.json')
        missing=[str(p) for p in paths.values() if not p.exists()]
        rec={k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','N','K','split')}
        rec.update(paths={k:str(p) for k,p in paths.items()},missing=missing,status='BLOCKED' if missing else 'READY')
        for p in paths.values():
            if p.exists(): hashes[str(p)]=sha(p)
        items.append(rec)
    protocol=dict(version='s2000_saved_stage_fragment_audit_v1',status='FROZEN',model='PLAIN_S2000',
        requests=items,L=9.208138264587333,stages=['RAW','D0','RESTORED','POST'],
        scientific_scope='CPU-only saved geometry shape/topology/deletion diagnostics and real visual review; no resampling or model calls',
        small_angle_degrees=5,area_over_L2_thresholds=[1e-6,1e-5,1e-4],primary_tiny_area_over_L2=1e-5,
        component_contract='POST actual indexed mesh, edge-connected and vertex-connected components separately; including C in graph; report C-free tiny detached components with <=3 faces and <=0.001 total free area. Descriptive, not automatic defects.',
        face_identity='generated RAW slot, via source_face_ids/source_corner_ids; no unknown GT',
        geometry_contract='No mesh edits, welds, deletion or D0 reruns; angles and area computed in FP64. Existing FF/I scores reused only.',
        visual_contract='All28 four-stage panels, two fixed historical views plus diagnostic POST highlights; nonblind AI_PROXY, uncertainty allowed. A white line or open edge alone is not a certified unwanted crack.',
        limits=dict(training=0,model_loads=0,model_forwards=0,new_noise=0,new_generation=0,new_GEO=0,commit=0,push=0),
        dispatch=dict(cpu_analysis_workers=1,cpu_render_workers_max=2,threads_per_worker=1,
                      reason='CPU geometry is small; overlap independent renderer with analysis. Renderer starts one real panel and measures RSS before second worker.'),
        source_hashes=hashes,
        closest_prior='Old SEAM200 D0 restoration/geometry-guard diagnostics; this evaluates saved S2000 small faces and visible fragments, not a new method.')
    p=OUT/'protocol.json'
    if p.exists():
        old=json.loads(p.read_text()); assert old==protocol,'Frozen protocol mismatch'
    else: write(p,protocol)
    write(OUT/'preparation_events.json',dict(events=[
        'Initial sandbox WSL resource query denied before analysis; escalation used existing environment.',
        'First escalated inline Python resource query failed shell quoting; no scientific work occurred. Replaced by file-based entry.'],
        model_calls=0))
    one=next(x for x in items if x['status']=='READY')
    keys={}
    for name,p in one['paths'].items():
        if str(p).endswith('.npz'):
            with np.load(p,allow_pickle=False) as z: keys[name]={k:dict(shape=z[k].shape,dtype=str(z[k].dtype)) for k in z.files}
    write(OUT/'array_inventory.json',keys)
    print(json.dumps(dict(status='FROZEN',requests=len(items),ready=sum(x['status']=='READY' for x in items),
        cpu=resources['cpu_logical'],RAM_GiB=resources['memory']['available']/2**30,load=resources['loadavg'],array_keys=keys),default=str))
if __name__=='__main__': main()
