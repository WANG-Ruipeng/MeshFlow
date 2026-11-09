#!/usr/bin/env python3
"""Fixed 40% DEV panel, quality-only scoring and complete RAW galleries.

Standalone new experiment adapter. Historical inputs/scorers/results remain read-only.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse, hashlib, html, importlib, json, sys, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from meshflow_control.evaluation.io import arrays, array_sha, digest, read_json, write_json, write_csv, save_npz, resolve
LABELS = ('START5000','HIST_FM6000','HIST_FREE_ALIGN6000','FM_ONLY','EARLY','LATE')
NS = 'CHAIR_EARLY_ALIGNMENT_SCHEDULE_40_V1'
ROOT_NAME = 'chair_early_alignment_schedule40_v1'
L_CANON = np.sqrt(3.) * 2. / .3762
VIEWS = (('front_3q',20,-60),('back_3q',20,120),('front',0,-90),('side',0,0),('wire',20,-60))
CONTRACTS = ('C_output_bitwise','C_all51_bitwise','actual_Gaussian_bitwise','NPZ_C_bitwise','PLY_C_bitwise','N_preserved','finite')

def root(p):
    p=resolve(p).resolve()
    if p.name != ROOT_NAME or p.parent.name != 'experiments': raise ValueError('Wrong experiment directory')
    p.mkdir(parents=True,exist_ok=True)
    return p

def seed(uid): return int.from_bytes(hashlib.sha256((NS+'/eval40/'+uid+'/0').encode()).digest()[:4], 'little')
def requests(out): return read_json(root(out)/'evaluation/requests.json')['rows']
def byte_equal(a,b): return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def expand_condition(C,F,K):
    """Keep old C verbatim; append nearest free centroids, lexicographic source-order ties.
    No vertex movement, repair, topology guarantee, or reference-shape inference input.
    """
    C=np.asarray(C); F=np.asarray(F)
    if C.dtype!=np.float32 or F.dtype!=np.float32: raise ValueError('Preserve existing FP32 triangles without casting')
    C=C.reshape(-1,3,3); F=F.reshape(-1,3,3)
    if not len(C)<=K<len(C)+len(F): raise ValueError('Invalid expansion budget')
    c=C.astype(np.float64).mean(1); f=F.astype(np.float64).mean(1)
    d=((f[:,None]-c[None])**2).sum(-1).min(1)
    idx=np.lexsort((np.arange(len(F)),d))[:K-len(C)]
    return np.concatenate((C,F[idx])),idx

def prepare(out,art='/mnt/e/MeshFlow-Control-artifacts',legacy='/mnt/e/MeshFlow'):
    out=root(out); art=resolve(art); legacy=resolve(legacy)
    old=read_json(art/'data/evaluation/requests.json')['rows']
    parents=sorted({q['parent_id'] for q in old if q['panel']=='P1'})
    if len(parents)!=8: raise ValueError('Expected eight existing DEV parents')
    domain=legacy/'native_t1/runs/chair_domain_control_v1'; panel=read_json(domain/'panels.json')
    train={p['uid'] for p in read_json(art/'data/train_manifest.json')['parents']}
    if set(parents)&train: raise ValueError('DEV parent leaked into conditional training')
    records=[]; audit=[]
    for uid in parents:
        row=next(r for r in panel['A'] if r['parent_id']==uid and r['ratio']==.4 and r['noise_index']==0)
        N=int(row['N']); K=int(round(.4*N))
        if row['status']=='READY':
            path=domain/row['input_path']
            if digest(path)!=row['input_sha256']: raise ValueError('Original A40 input changed')
            a=arrays(path); C=a['C'].reshape(-1,3,3)
            source_ids=a.get('source_face_ids'); method='EXISTING_A40_CONNECTED_CONDITION'
            detail=dict(source_path=str(path),source_sha256=digest(path),source_face_ids=None if source_ids is None else source_ids.tolist())
        else:
            q=next(q for q in old if q['parent_id']==uid and q['panel']=='P1')
            meta=read_json(q['reference_path']); path=resolve(meta['condition_path'])
            if digest(path)!=meta['condition_sha256']: raise ValueError('Old condition changed')
            a=arrays(path); C,extra=expand_condition(a['C'],a['GT_free'],K)
            method='OLD_C_PLUS_NEAREST_FREE_CENTROIDS_EXACT_COORDINATES'
            detail=dict(source_path=str(path),source_sha256=digest(path),old_K=len(a['C']),appended_free_indices=extra.tolist(),old_A40_status=row['status'],old_A40_reason=row['reason'],trusted_interface='NOT_ASSUMED')
        if C.dtype!=np.float32 or C.shape!=(K,3,3) or not np.isfinite(C).all(): raise ValueError('Invalid C40')
        s=seed(uid); z=np.zeros((1,N,9),np.float32)
        z[:,K:]=np.random.RandomState(s).randn(N-K,3,3).astype(np.float32).reshape(1,N-K,9)
        rid=uid+'__C40__n0'; dst=out/'evaluation/inputs'/(rid+'.npz')
        bank=dict(C=C,N=np.array(N,dtype=np.int64),K=np.array(K,dtype=np.int64),z=z)
        if dst.exists():
            saved=arrays(dst)
            if set(saved)!=set(bank) or not all(byte_equal(saved[k],bank[k]) for k in bank): raise ValueError('Frozen input differs')
        else: save_npz(dst,bank)
        records.append(dict(request_id=rid,parent_id=uid,condition_id=uid+'_C40',panel='DEV40',cohort='SEEN_DEV',candidate=0,N=N,K=K,known_face_fraction=K/N,condition_construction=method,noise_seed=s,C_sha256=array_sha(C),Gaussian_sha256=array_sha(z),input_path=str(dst),input_sha256=digest(dst)))
        audit.append(dict(parent_id=uid,construction=method,**detail))
    manifest=dict(schema=NS,rows=records,labels=LABELS,expected_outputs=48,rollouts_per_model=8,steps=50,scale_source='public canonical cube diagonal sqrt(3)*2/.3762',L_canonical=float(L_CANON),source_reconstruction_scores=False,DEV_previously_observed=True,source_used_only_to_construct_C=True)
    write_json(out/'evaluation/requests.json',manifest,freeze=True)
    write_json(out/'evaluation/condition_construction.json',dict(rows=audit,original_panel_sha256=digest(domain/'panels.json'),original_requests_sha256=digest(art/'data/evaluation/requests.json')),freeze=True)
    return dict(status='FROZEN',requests=8,outputs=48,existing_A40=sum(r['condition_construction'].startswith('EXISTING') for r in records),expanded_old_C=sum(not r['condition_construction'].startswith('EXISTING') for r in records),model_calls=0)

class SampleBudget:
    def __init__(self,path,context): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.context=context; self.counts={}
    def event(self,event,**kw):
        with self.path.open('a') as f:
            f.write(json.dumps(dict(time_unix=time.time(),event=event,**kw),allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    def reserve(self,kind,context=None):
        count=self.counts.get(kind,0)+1
        if count>dict(rollout_attempt=1,sampling_forward=50)[kind]: raise RuntimeError('Registered sample budget exceeded')
        self.counts[kind]=count; self.event('RESERVE',kind=kind,context=context or self.context,ordinal=count)
    def call(self,kind,fn,context=None):
        self.reserve(kind,context); t=time.perf_counter()
        try: result=fn()
        except BaseException as e: self.event('ERROR',kind=kind,error=repr(e),context=context); raise
        self.event('RETURN',kind=kind,seconds=time.perf_counter()-t,context=context); return result

def sample_label(out,label,checkpoint,attempt='initial'):
    if label not in LABELS: raise ValueError('Unregistered label')
    from meshflow_control import runtime
    import torch
    from meshflow_control.sampling import sample
    from meshflow_control.checkpoints import load_start
    from meshflow_control.training.context_checkpoint import load_generator as load_historical
    from meshflow_control.training.schedule40 import load_generator as load_new
    out=root(out); checkpoint=resolve(checkpoint); checkpoint_hash=digest(checkpoint)
    folder=out/'sampling'/label; folder.mkdir(parents=True,exist_ok=True)
    log=folder/(attempt+'.jsonl'); start=time.perf_counter(); completed=[]
    torch.set_num_threads(2); runtime.configure_stable_runtime()
    try:
        if label=='START5000': model,_,identity=load_start(checkpoint,device='cuda',trainable=False)
        elif label.startswith('HIST_'): model,identity=load_historical(checkpoint,device='cuda')
        else: model,identity=load_new(checkpoint,device='cuda')
        for q in requests(out):
            dst=out/'outputs'/label/q['request_id']; receipt=dst/'receipt.json'
            binding=dict(label=label,request_id=q['request_id'],input_sha256=q['input_sha256'],checkpoint_sha256=checkpoint_hash)
            if receipt.exists():
                row=read_json(receipt)
                if any(row.get(k)!=v for k,v in binding.items()) or any(digest(row[k+'_path'])!=row[k+'_sha256'] for k in ('raw','ply','result')): raise ValueError('Completed sample differs')
                completed.append(row); continue
            if digest(q['input_path'])!=q['input_sha256']: raise ValueError('Input changed')
            if (dst/'started.json').exists() and attempt=='initial': raise RuntimeError('Unreceipted prior sampling requires explicit recovery')
            dst.mkdir(parents=True,exist_ok=True)
            write_json(dst/('started_'+attempt+'.json'),dict(**binding,attempt=attempt,time_unix=time.time()),freeze=True)
            write_json(dst/'started.json',dict(**binding,attempt=attempt,time_unix=time.time()))
            b=arrays(q['input_path']); budget=SampleBudget(log,dict(**binding,attempt=attempt))
            raw,trajectory,audit=sample(model,b['C'],q['N'],q['noise_seed'],b['z'],budget=budget,out=dst,context=dict(**binding,attempt=attempt))
            row=dict(q,**{k:v for k,v in binding.items() if k not in q},status='COMPLETE',attempt=attempt,raw_path=str(dst/'raw.npz'),raw_sha256=digest(dst/'raw.npz'),ply_path=str(dst/'raw.ply'),ply_sha256=digest(dst/'raw.ply'),result_path=str(dst/'run.json'),result_sha256=digest(dst/'run.json'),seconds=audit['seconds'],cuda_peak_allocated_bytes=audit['cuda_peak_allocated_bytes'],cuda_peak_reserved_bytes=audit['cuda_peak_reserved_bytes'])
            write_json(receipt,row,freeze=True); completed.append(row)
            write_json(folder/'progress.json',dict(status='RUNNING',label=label,attempt=attempt,completed=len(completed),elapsed_seconds=time.perf_counter()-start,peak_scope='single real Euler50 rollout including resident generator',cuda_peak_allocated_bytes=audit['cuda_peak_allocated_bytes'],cuda_peak_reserved_bytes=audit['cuda_peak_reserved_bytes']))
        write_json(folder/'completed.json',dict(status='COMPLETE',label=label,attempt=attempt,rows=completed,seconds=time.perf_counter()-start))
    except BaseException as exc:
        oom=isinstance(exc,torch.cuda.OutOfMemoryError)
        write_json(folder/(attempt+'_failure.json'),dict(status='OOM' if oom else 'FAILED',attempt=attempt,error=repr(exc),traceback=traceback.format_exc(),completed=len(completed),seconds=time.perf_counter()-start))
        raise
    return dict(status='COMPLETE',label=label,outputs=len(completed))

def score_one(args):
    out,q,label=args; out=root(out); dst=out/'evaluation/rows'/label/(q['request_id']+'.json')
    from meshflow_control.evaluation.geometry import scan_geometry,topology
    from meshflow_control.evaluation.shape import shape_quality
    from meshflow_control.mesh_io import read_ply
    row=dict(q,label=label,metrics={},contracts={},status='BLOCKED')
    receipt=out/'outputs'/label/q['request_id']/'receipt.json'
    if not receipt.exists(): row['reason']='MISSING_REGISTERED_RAW'; write_json(dst,row); return row
    r=read_json(receipt); sources={p.name:digest(p) for p in Path(importlib.import_module('meshflow_control.evaluation.geometry').__file__).parent.glob('*.py')}; identity=dict(raw_sha256=r['raw_sha256'],ply_sha256=r['ply_sha256'],result_sha256=r['result_sha256'],input_sha256=q['input_sha256'],adapter_sha256=digest(__file__),scorer_sources=sources,L_canonical=float(L_CANON))
    if dst.exists():
        old=read_json(dst)
        if old.get('identity')==identity:
            if any(digest(r[k+'_path'])!=r[k+'_sha256'] for k in ('raw','ply','result')) or digest(q['input_path'])!=q['input_sha256']: raise ValueError('Scored asset changed')
            return old
        if old['status']=='PASS': raise ValueError('Completed scoring identity changed')
    t=time.perf_counter()
    try:
        if any(digest(r[k+'_path'])!=r[k+'_sha256'] for k in ('raw','ply','result')) or digest(q['input_path'])!=q['input_sha256']: raise ValueError('Scoring input changed')
        a=arrays(r['raw_path']); b=arrays(q['input_path']); tri=a['raw'].reshape(-1,3,3); C=b['C']; K=q['K']; N=q['N']
        ply,_=read_ply(r['ply_path']); path=a['trajectory'].reshape(51,N,3,3)
        facts=dict(C_output_bitwise=byte_equal(tri[:K],C),C_all51_bitwise=all(byte_equal(x[:K],C) for x in path),actual_Gaussian_bitwise=byte_equal(a['z'],b['z']),NPZ_C_bitwise=byte_equal(a['C'],C),PLY_C_bitwise=byte_equal(ply[:K],C),N_preserved=len(tri)==N,finite=bool(np.isfinite(tri).all()),trajectory_final_is_RAW=byte_equal(path[-1],tri),PLY_full_bitwise=byte_equal(ply,tri))
        if not all(facts.values()): raise ValueError('Output contract failed '+str(facts))
        metrics,cf,ff=scan_geometry(C,tri[K:],L_CANON); metrics.update(topology(tri,K))
        cq=shape_quality(C,L_CANON); wq=shape_quality(tri,L_CANON)
        metrics.update(C_total_surface_area=cq['total_area'],C_near_degenerate_count=cq['near_degenerate_area_count'],C_exact_zero_area_count=cq['exact_zero_area_count'],whole_total_surface_area=wq['total_area'],whole_near_degenerate_count=wq['near_degenerate_area_count'],whole_minimum_angle_lt5_fraction=wq['min_angle_lt5_fraction'])
        # Names below retain CF proper and FF union definitions; no synthetic global intersection score.
        pairfile=out/'evaluation/pairs'/label/(q['request_id']+'.json'); write_json(pairfile,dict(CF=cf,FF=ff))
        row.update(status='PASS',metrics=metrics,contracts=facts,pair_audit=str(pairfile),pair_sha256=digest(pairfile),raw_path=r['raw_path'],condition_construction=q['condition_construction'])
    except Exception as exc: row.update(status='SCORING_BLOCKED',reason=repr(exc),traceback=traceback.format_exc())
    row.update(identity=identity,seconds=time.perf_counter()-t); write_json(dst,row); return row

def score(out,workers=4):
    out=root(out); qs=requests(out); todo=[(str(out),q,l) for q in qs for l in LABELS]
    tick=time.perf_counter()
    if workers==1: rows=[score_one(x) for x in todo]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool: rows=list(pool.map(score_one,todo))
    write_json(out/'evaluation/per_output.json',dict(rows=rows))
    write_csv(out/'evaluation/per_output.csv',[dict(**{k:r.get(k) for k in ('label','parent_id','request_id','N','K','condition_construction','status','reason')},**r['metrics'],**r['contracts']) for r in rows])
    names=sorted({k for r in rows for k,v in r['metrics'].items() if isinstance(v,(int,float)) and not isinstance(v,bool)})
    summaries=[]; paired=[]
    for label in LABELS:
        good=[r for r in rows if r['label']==label and r['status']=='PASS']
        s=dict(label=label,registered=8,scored=len(good),status='COMPLETE' if len(good)==8 else 'PARTIAL_NOT_FULL_PANEL')
        for key in names:
            v=[r['metrics'][key] for r in good if key in r['metrics'] and r['metrics'][key] is not None]
            s[key+'_scored_count']=len(v)
            if v: s[key+'_mean']=float(np.mean(v)); s[key+'_median']=float(np.median(v))
        summaries.append(s)
    by={(r['label'],r['parent_id']):r for r in rows}
    for a,b in [('EARLY','LATE'),('EARLY','FM_ONLY'),('LATE','FM_ONLY'),('FM_ONLY','HIST_FM6000'),('EARLY','HIST_FREE_ALIGN6000'),('LATE','HIST_FREE_ALIGN6000')]:
        for q in qs:
            ra=by[a,q['parent_id']]; rb=by[b,q['parent_id']]
            if ra['status']!='PASS' or rb['status']!='PASS':
                paired.append(dict(candidate=a,reference=b,parent_id=q['parent_id'],status='BLOCKED',reason=str((ra['status'],rb['status'])))); continue
            paired.append(dict(candidate=a,reference=b,parent_id=q['parent_id'],status='PASS',**{k+'_delta':ra['metrics'][k]-rb['metrics'][k] for k in names if k in ra['metrics'] and k in rb['metrics'] and ra['metrics'][k] is not None and rb['metrics'][k] is not None}))
    write_csv(out/'evaluation/summary.csv',summaries); write_csv(out/'evaluation/paired_parent_deltas.csv',paired)
    receipt=dict(status='COMPLETE' if all(r['status']=='PASS' for r in rows) else 'PARTIAL',registered=48,scored=sum(r['status']=='PASS' for r in rows),wall_seconds=time.perf_counter()-tick,workers=workers,source_reconstruction_evaluated=False,visual_fragment_count='NOT_INFERRED_FROM_RAW_TOPOLOGY')
    write_json(out/'evaluation/scoring_receipt.json',receipt); return receipt

def render_page(path,columns,bounds,title,neutral):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    fig=plt.figure(figsize=(15,2.8*len(columns)),dpi=110,facecolor='white'); lo,hi=map(np.asarray,bounds); center=(lo+hi)/2; span=max(float((hi-lo).max()),.01)*1.08
    for i,(label,tri,K) in enumerate(columns):
        for j,(view,elev,azim) in enumerate(VIEWS):
            ax=fig.add_subplot(len(columns),5,i*5+j+1,projection='3d'); draw=tri[:,:,(0,2,1)]
            colors=['#b4b8bc']*len(tri) if neutral else ['#3a82c5']*K+['#e7994c']*(len(tri)-K)
            if view=='wire':
                artist=Line3DCollection(draw[:,((0,1),(1,2),(2,0)),:].reshape(-1,2,3),colors=np.repeat(colors,3),linewidths=.38)
            else: artist=Poly3DCollection(draw,facecolors=colors,edgecolors='#444444',linewidths=.10,alpha=1)
            ax.add_collection3d(artist); cc=center[(0,2,1),]
            ax.set_xlim(cc[0]-span/2,cc[0]+span/2); ax.set_ylim(cc[1]-span/2,cc[1]+span/2); ax.set_zlim(cc[2]-span/2,cc[2]+span/2)
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=elev,azim=azim); ax.set_axis_off(); ax.set_title(label+' / '+view,fontsize=9,pad=0)
    fig.suptitle(title,fontsize=12,y=.994); fig.subplots_adjust(left=0,right=1,top=1-.65/(len(columns)*2.8),bottom=0,wspace=0,hspace=.02)
    path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,facecolor='white'); plt.close(fig)

def render(out):
    out=root(out); pages=[]; cameras={}; aliases={l:'Method '+str(i+1) for i,l in enumerate(sorted(LABELS,key=lambda l:hashlib.sha256((NS+'/blind/'+l).encode()).hexdigest()))}; ordered=sorted(LABELS,key=lambda l:aliases[l])
    write_json(out/'gallery/method_mapping.json',aliases,freeze=True)
    for q in requests(out):
        b=arrays(q['input_path']); C=b['C']; loaded={}; hashes={}
        for label in LABELS:
            rp=out/'outputs'/label/q['request_id']/'receipt.json'
            if not rp.exists(): raise RuntimeError('Cannot freeze complete camera before all48 RAW')
            r=read_json(rp)
            if digest(r['raw_path'])!=r['raw_sha256']: raise ValueError('RAW changed')
            loaded[label]=arrays(r['raw_path'])['raw'].reshape(-1,3,3); hashes[label]=r['raw_sha256']
        pp=np.concatenate([C.reshape(-1,3)]+[x.reshape(-1,3) for x in loaded.values()]); bounds=(pp.min(0),pp.max(0))
        columns=[('Input C',C,q['K'])]+[(aliases[l],loaded[l],q['K']) for l in ordered]
        files={}
        for kind in ('colored','neutral'):
            path=out/'gallery'/kind/(q['request_id']+'.png')
            render_page(path,columns,bounds,q['parent_id']+' | fixed40% | all registered outputs | '+kind,kind=='neutral')
            files[kind]=str(path); files[kind+'_sha256']=digest(path)
        pages.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],AI_VISUAL_REVIEW='PENDING',**files))
        cameras[q['request_id']]=dict(bounds=[x.tolist() for x in bounds],views=VIEWS,source_hashes=hashes,scope='C and all six RAW; same camera, no cropping/smoothing/repair; source completion hidden')
    write_json(out/'gallery/camera_manifest.json',cameras,freeze=True); write_json(out/'gallery/manifest.json',dict(status='COMPLETE',pages=pages,outputs=48,AI_VISUAL_REVIEW='PENDING',human_usability='NOT_RUN'),freeze=True)
    lines=['<!doctype html><meta charset="utf-8"><title>Schedule40 complete RAW gallery</title><h1>Fixed40%: six models, eight DEV objects</h1><p>All48 outputs. Four views plus wire. No source target, repair, cropping or per-model scaling. Anonymous labels; AI review separate.</p>']
    for p in pages:
        lines.append('<h2>'+p['parent_id']+'</h2>')
        for k in ('neutral','colored'): lines.append('<p><a href="'+Path(p[k]).relative_to(out).as_posix()+'">'+k+'</a></p>')
    lines.append('<a href="gallery/method_mapping.json">Method mapping</a>'); (out/'OPEN_GALLERY.html').write_text('\n'.join(lines),encoding='utf8')
    return dict(status='COMPLETE',pages=len(pages)*2,outputs=48)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['prepare','sample','score','render'],required=True); p.add_argument('--out',required=True); p.add_argument('--label',choices=LABELS); p.add_argument('--checkpoint'); p.add_argument('--attempt',default='initial'); p.add_argument('--workers',type=int,default=4); a=p.parse_args()
    if a.stage=='prepare': value=prepare(a.out)
    elif a.stage=='sample': value=sample_label(a.out,a.label,a.checkpoint,a.attempt)
    elif a.stage=='score': value=score(a.out,a.workers)
    else: value=render(a.out)
    print(json.dumps(value,allow_nan=False),flush=True)
