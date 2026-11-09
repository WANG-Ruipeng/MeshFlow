"""Four registered START samples from two connected-triangle rectangles."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,threading,time,traceback
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LP_NUM_THREADS']:os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def event(kind,**v):
    p=OUT/'ledger.jsonl'
    with p.open('a') as f:f.write(json.dumps(dict(event=kind,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**v))+'\n')
def command(args):
    r=subprocess.run(args,capture_output=True,text=True)
    return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
def resources(label):
    import psutil
    result=dict(label=label,time=time.time(),CPU_cores=os.cpu_count(),CPU_available=psutil.cpu_count(),
        CPU_utilization=psutil.cpu_percent(.1),RAM_available=psutil.virtual_memory().available,
        GPU=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        GPU_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),
        CPU_processes=command(['ps','-eo','pid,comm,pcpu,pmem,args','--sort=-pcpu']))
    write(OUT/'audit'/('resources_'+label+'.json'),result);return result
def obj(path,tri):
    tri=np.asarray(tri).reshape(-1,3,3);vertices=[];faces=[];lookup={}
    for t in tri:
        face=[]
        for v in t:
            key=tuple(float(x) for x in v)
            if key not in lookup:lookup[key]=len(vertices)+1;vertices.append(key)
            face.append(lookup[key])
        faces.append(face)
    Path(path).write_text('\n'.join(['v '+' '.join(f'{x:.9g}' for x in v) for v in vertices]+['f '+' '.join(map(str,f)) for f in faces])+'\n')
def prepare():
    if (OUT/'plan.json').exists():raise FileExistsError('Plan already frozen; use --sample to resume unattempted requests')
    (OUT/'inputs').mkdir(exist_ok=True)
    requests=[];conditions=[]
    for width,condition in [(2.4,'seat_w240'),(3.0,'seat_w300')]:
        d=2.4;y=-.55
        vertices=np.array([[-width/2,y,-d/2],[width/2,y,-d/2],[width/2,y,d/2],[-width/2,y,d/2]],np.float32)
        C=vertices[[[0,3,2],[0,2,1]]]
        assert len(np.unique(C.reshape(-1,3),axis=0))==4
        normals=np.cross(C[:,1]-C[:,0],C[:,2]-C[:,0]);areas=np.linalg.norm(normals,axis=1)/2
        assert np.all(normals[:,1]>0) and np.all(areas>0)
        edges=np.array([np.linalg.norm(C[:,(i+1)%3]-C[:,i],axis=1) for i in range(3)]).T
        angles=np.degrees(np.arccos(np.clip((edges**2+np.roll(edges,1,1)**2-np.roll(edges,2,1)**2)/(2*edges*np.roll(edges,1,1)),-1,1)))
        np.save(OUT/'inputs'/(condition+'.npy'),C);obj(OUT/'inputs'/(condition+'.obj'),C)
        conditions.append(dict(condition=condition,width=width,depth=d,seat_y=y,K=2,unique_vertices=4,
            coplanar=True,shared_edges=1,boundary_edges=4,connected_components=1,normal='+Y',
            triangle_area=areas.tolist(),minimum_angle_degrees=float(angles.min()),maximum_edge_ratio=float((edges.max(1)/edges.min(1)).max()),
            C_sha256=sha(OUT/'inputs'/(condition+'.npy'))))
        for seed in [20261006,20261007]:
            z=np.zeros((1,160,9),np.float32);z[:,2:]=np.random.RandomState(seed).randn(158,3,3).astype(np.float32).reshape(1,158,9)
            np.savez_compressed(OUT/'inputs'/('noise_'+str(seed)+'.npz'),initial_gaussian=z)
            identity=condition+'_'+str(seed)
            requests.append(dict(id=identity,condition=condition,seed=seed,C='inputs/'+condition+'.npy',
                input_npz='inputs/noise_'+str(seed)+'.npz',noise_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
                N=160,K=2,width=width,depth=d,seat_y=y,output_dir='outputs/'+identity))
    # Paired width variants are adjacent, both use byte-identical saved free noise.
    requests.sort(key=lambda r:(r['seed'],r['width']))
    source_files=list((ROOT/'native_t1').glob('*.py'))+[ROOT/'native_t1/recipes/chair_hybrid_start.json']
    plan=dict(name='CHAIR_TWO_TRIANGLE_CONTROL_V1',conditions=conditions,requests=requests,
        budget=dict(training_updates=0,teacher_calls=0,new_samples=4,steps_each=50,max_model_forwards=200,retries=0),
        checkpoint='native_t1/checkpoints/chair_hybrid_start_total5000.pt',
        checkpoint_sha256='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80',
        definition='A deliberately constructed connected two-triangle horizontal seat; width-only intervention, shared Gaussian by seed. No target/GT supplied to model.',
        rationale='Seat height -0.55 and depth2.4 follow the existing original-scale chair examples; rounded user-authored dimensions, not a retessellated GT mesh.',
        warnings=['K2 is API-supported but substantially sparser than the earlier 20%-context panels; generation quality is unverified.',
            'Exact C preservation does not imply connectivity, exposed retention, or general control success.'],
        frozen_sources={str(p.relative_to(ROOT)):sha(p) for p in source_files},
        input_file_hashes={str(p.relative_to(OUT)):sha(p) for p in (OUT/'inputs').iterdir()})
    write(OUT/'plan.json',plan);event('PLAN_FROZEN',sha256=sha(OUT/'plan.json'),jobs=4,forward_budget=200)
    print('Prepared 2 connected rectangles, 2 paired noises, 4 frozen requests.')

class Monitor:
    def __init__(self):self.stop=threading.Event();self.samples=[];self.thread=threading.Thread(target=self.loop,daemon=True)
    def loop(self):
        p=OUT/'audit/gpu_telemetry.jsonl'
        with p.open('a') as f:
            while not self.stop.is_set():
                v=dict(time=time.time(),device=command(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
                    processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']))
                self.samples.append(v);f.write(json.dumps(v)+'\n');f.flush();self.stop.wait(.25)
    def start(self):self.thread.start()
    def close(self):self.stop.set();self.thread.join(5)

def sample():
    plan=read(OUT/'plan.json');before=resources('before_model')
    for path,digest in plan['input_file_hashes'].items():assert sha(OUT/path)==digest
    for path,digest in plan['frozen_sources'].items():assert sha(ROOT/path)==digest
    from native_t1.runtime import configure_stable_runtime
    import torch
    from native_t1.chair_checkpoint import load_chair_start
    from native_t1.chair_sampling import make_noise,clamped_sample
    from native_t1.sampling import new_counts
    from native_t1.artifacts import state_sha256
    torch.set_num_threads(2);configure_stable_runtime()
    rows=[];pending=[]
    for job in plan['requests']:
        r=OUT/job['output_dir']/'run.json'
        if r.exists():
            prior=read(r)
            if prior['status']!='PASS':raise RuntimeError('Prior incomplete/failed attempt is not automatically retried: '+job['id'])
            assert sha(OUT/job['output_dir']/'raw.npz')==prior['raw_sha256'];rows.append(prior)
        else:pending.append(job)
    if not pending:print('All four requests already complete; no forwards.');return
    gpu=before['GPU']['stdout'].split(',');total=float(gpu[1]);free=float(gpu[3]);headroom=max(2048,total*.1)
    if free<headroom+1024:raise RuntimeError('Insufficient free VRAM for first authorized observation; no dispatch')
    monitor=Monitor();monitor.start();started=time.perf_counter();counts=new_counts();phase='load';active=None
    try:
        torch.cuda.init();torch.cuda.reset_peak_memory_stats()
        model,load=load_chair_start(device='cuda');torch.cuda.synchronize()
        load.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            peak_scope='Model construction/device load before any forward; separate from per-rollout peaks')
        write(OUT/'audit/model_load.json',load);event('MODEL_LOADED',seconds=load['seconds'],sha256=load['checkpoint_sha256'])
        for ji,job in enumerate(pending):
            phase=job['id'];active=dict(status='RUNNING',**job,counts=new_counts(),checkpoint_sha256=load['checkpoint_sha256'])
            dest=OUT/job['output_dir'];dest.mkdir(parents=True,exist_ok=True);write(dest/'run.json',active)
            snapshot=resources('before_'+job['id'])
            remaining_free=float(snapshot['GPU']['stdout'].split(',')[3])
            if remaining_free<headroom:raise RuntimeError('Headroom exhausted; no further dispatch')
            C=np.load(OUT/job['C'],allow_pickle=False);z=make_noise(job['seed'],job['N'],job['K'])
            with np.load(OUT/job['input_npz']) as saved:assert np.array_equal(z.numpy(),saved['initial_gaussian'])
            event('SAMPLE_BEGIN',request=job['id'],max_forwards=50,worker=1)
            raw,path,audit=clamped_sample(model,z.cuda(),torch.from_numpy(C).cuda(),active['counts'])
            np.savez_compressed(dest/'raw.npz',output=raw,path=path,initial_gaussian=z.numpy(),C=C)
            # A plain triangle-soup OBJ preserves every original generated coordinate and face.
            tri=raw.reshape(-1,3,3);lines=[]
            for v in tri.reshape(-1,3):lines.append('v '+' '.join(f'{float(x):.9g}' for x in v))
            lines+=['f '+str(3*i+1)+' '+str(3*i+2)+' '+str(3*i+3) for i in range(len(tri))]
            (dest/'raw.obj').write_text('\n'.join(lines)+'\n')
            active.update(status='PASS',sampling=audit,raw_sha256=sha(dest/'raw.npz'))
            write(dest/'run.json',active);rows.append(active)
            for k in counts:counts[k]+=active['counts'][k]
            event('SAMPLE_END',request=job['id'],seconds=audit['seconds'],counts=active['counts'])
            resources('after_'+job['id'])
            if ji==0:
                utilization=[float(x['device']['stdout'].split(',')[-1]) for x in monitor.samples if x['device']['returncode']==0]
                decision=dict(actual_workers=1,remaining_jobs=len(pending)-1,load_seconds=load['seconds'],first_sampling_seconds=audit['seconds'],
                    observed_global_GPU_utilization_max=max(utilization,default=None),first_peak_reserved=audit['peak_cuda_reserved'],
                    headroom_MiB=headroom,process_VRAM_note='Use telemetry process values if reported; global device occupancy includes unrelated desktop processes.',
                    reason='Four short jobs reuse one resident model. Only three remain after representative measurement; extra worker model loading and strict identity validation add startup before they can contribute. CPU analysis/rendering overlap when independent; no extra calibration samples.')
                write(OUT/'audit/dispatch_decision.json',decision)
        after=state_sha256(model);assert after==load['state_sha256']
        for path,digest in plan['input_file_hashes'].items():assert sha(OUT/path)==digest
        for path,digest in plan['frozen_sources'].items():assert sha(ROOT/path)==digest
        summary=dict(status='PASS',requests=rows,total_model_forwards=sum(r['counts']['model_forward_attempts'] for r in rows),
            new_samples_this_invocation=len(pending),training_updates=0,backwards=0,optimizer_updates=0,
            state_before=load['state_sha256'],state_after=after,source_hashes_unchanged=True,
            wall_seconds=time.perf_counter()-started,plan_sha256=sha(OUT/'plan.json'),worker_count=1)
        assert summary['total_model_forwards']==200
        write(OUT/'results.json',summary);event('PILOT_DONE',forwards=200,samples=4,wall_seconds=summary['wall_seconds'])
        print('PASS: 4 samples, 200 forwards, zero training; immutable C in all51 states.')
    except BaseException as error:
        if active is not None:
            active.update(status='STOPPED',error=repr(error));write(OUT/active['output_dir']/'run.json',active)
        write(OUT/'failure.json',dict(status='STOPPED',phase=phase,error=repr(error),traceback=traceback.format_exc(),completed=len(rows)))
        event('STOPPED',phase=phase,error=repr(error));raise
    finally:monitor.close();resources('after_model')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--sample',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.sample:sample()
    else:p.error('Choose --prepare or --sample')
