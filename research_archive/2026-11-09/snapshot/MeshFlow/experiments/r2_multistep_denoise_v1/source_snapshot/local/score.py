"""USER_MIX D1/D3/D5 evaluation, strictly CPU and independent output directory.
Reuse immutable RAW/D1 evidence; score only 24 newly postprocessed endpoints.
Run after all D2..D5 model forwards have completed. No intermediate GEO_C.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, concurrent.futures, importlib.util, json, multiprocessing, resource, sys, time
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'experiments/r2_multistep_denoise_v1'
OLD=REPO/'experiments/r2_postprocess_avmr_pmr_v1'
RAW_AVMR=REPO/'experiments/r2_avmr_pmr_offline_v1'
LABEL='R2_USER_MIX'
STAGES=('RAW','D1','D3','D5')
META=('request_id','condition_id','parent_id','region_type','tier','candidate','N','K')
sys.path.insert(0,str(REPO))
def module(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
# Import only definitions; none of the old main/prepare/write entrypoints run.
P=module('_multistep_frozen_post_primitives',REPO/'r2/tools/postprocess_avmr_pmr_v1/run.py')
A=P.A
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
Q=module('_multistep_frozen_quality',REPO/'r2/tools/postprocess_avmr_pmr_v1/quality.py')
sha,read,csv_read=A.sha,A.read,A.csv_read
VERSION='r2_multistep_denoise_fixed_endpoints_cpu_v1'

def checked(path):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):raise ValueError('Refusing output outside this experiment: '+str(path))
    return path

def write(path,value):return A.write(checked(path),value)
def csv_write(path,rows):return A.csv_write(checked(path),rows)
def event(kind,**fields):
    path=checked(OUT/'score_ledger.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**fields),allow_nan=False)+'\n')

def registration(check_hashes=False):
    reg=read(OUT/'registration.json');rows=reg['records']
    if len(rows)!=12 or len({r['request_id'] for r in rows})!=12 or any(r['label']!=LABEL for r in rows):raise ValueError('Expected exactly the fixed 12 USER_MIX outputs')
    old=read(OLD/'registration.json')
    originals={r['request_id']:r for r in old['records'] if r['label']==LABEL}
    extra={'D1_denoised_path','D1_denoised_sha256','D1_receipt_path','D1_post_path','D1_score_path'}
    if {r['request_id'] for r in rows}!=set(originals):raise ValueError('Original USER_MIX request set changed')
    for r in rows:
        original=originals[r['request_id']]
        if any(r.get(k)!=v for k,v in original.items()) or set(r)-set(original)-extra:raise ValueError('Original USER_MIX record identity or allowed D1 schema changed')
        if 'D1_denoised_path' in r and sha(r['D1_denoised_path'])!=r['D1_denoised_sha256']:raise ValueError('Registered reused D1 denoised file changed')
    if reg['settings']!=old['settings']:raise ValueError('Frozen AVMR settings changed')
    if check_hashes:
        for p,h in reg['protected'].items():
            if sha(p)!=h:raise ValueError('Protected file changed: '+p)
    if 'torch' in sys.modules:raise AssertionError('CPU scorer imported torch')
    return reg

def denoise_identity(r,stage):
    roundnum=int(stage[1:]);path=OUT/'denoised'/stage/r['request_id']/'denoised.npz';receipt=read(path.with_name('receipt.json'))
    wanted=dict(request_id=r['request_id'],raw_sha256=r['raw_sha256'],round=roundnum,status='PASS')
    if any(receipt.get(k)!=v for k,v in wanted.items()):raise ValueError('Denoiser receipt identity mismatch: '+str(path))
    if sha(path)!=receipt['output_sha256']:raise ValueError('Denoiser artifact hash mismatch')
    return path,receipt

def validate_post(r):
    den,dr=denoise_identity(r,r['stage']);post=Path(r['post_path']);pp=read(post.with_name('postprocess.json'))
    if sha(r['raw_path'])!=r['raw_sha256'] or sha(post)!=r['post_sha256']:raise ValueError('RAW or POST changed')
    if pp['status']!='PASS' or pp['input']['file_sha256']!=dr['output_sha256']:raise ValueError('GEO_C ancestry mismatch')
    for name,detail in pp['files'].items():
        if sha(post.parent/name)!=detail['sha256']:raise ValueError('GEO_C artifact changed')
    return dr,pp

def old_receipt(r):
    p=read(OLD/'scores'/LABEL/r['request_id']/'score.json')
    if p['status']!='PASS' or p['identity']['raw_sha256']!=r['raw_sha256']:raise ValueError('Old D1 score identity mismatch')
    for path,h in p['files'].items():
        if sha(path)!=h:raise ValueError('Frozen D1 score detail changed')
    return p

def postprocess_all():
    reg=registration(True);index=[];tick=time.perf_counter()
    oldindex={r['request_id']:r for r in csv_read(OLD/'post_index.csv') if r['label']==LABEL}
    for r in reg['records']:
        rid=r['request_id'];K=int(r['K']);old=oldindex[rid];p=old_receipt(r)
        if sha(old['post_path'])!=p['identity']['post_sha256']:raise ValueError('Old D1 post changed')
        with np.load(r['raw_path'],allow_pickle=False) as a:raw=a['raw'].reshape(-1,3,3)
        for stage in STAGES:
            if stage=='RAW':
                index.append(dict(r,stage=stage,post_path=r['raw_path'],post_sha256=r['raw_sha256'],post_ply='',denoised_path='',output_N=int(r['N']),reused=True))
                continue
            if stage=='D1':
                index.append(dict(r,stage=stage,post_path=old['post_path'],post_sha256=old['post_sha256'],post_ply=old['post_ply'],denoised_path=old['denoised_path'],output_N=int(old['output_N']),reused=True))
                continue
            den,dr=denoise_identity(r,stage);dest=checked(OUT/'outputs'/stage/rid);ppath=dest/'postprocess.json'
            with np.load(den,allow_pickle=False) as a:
                if a['output'].dtype!=np.float32 or a['output'].shape!=raw.shape or not A.bit_equal(a['output'][:K],raw[:K]):raise ValueError('Denoising changed N/C/dtype')
            if ppath.exists():
                pp=read(ppath)
                if pp['status']!='PASS' or pp['input']['file_sha256']!=dr['output_sha256']:raise ValueError('Invalid GEO_C resume')
                for name,detail in pp['files'].items():
                    if sha(dest/name)!=detail['sha256']:raise ValueError('Existing GEO_C file changed')
            else:
                event('GEO_C_STARTED',stage=stage,request_id=rid)
                pp=P.PC.run(den,K,dest,input_key='output')
                event('GEO_C_COMPLETE',stage=stage,request_id=rid,seconds=pp['seconds'],deleted=pp['stats']['deleted_free_faces'])
            with np.load(dest/'post.npz',allow_pickle=False) as a:
                if not A.bit_equal(a['output'][:K],raw[:K]) or not A.bit_equal(a['C'],raw[:K]):raise ValueError('POST C changed')
            index.append(dict(r,stage=stage,post_path=str(dest/'post.npz'),post_sha256=sha(dest/'post.npz'),post_ply=str(dest/'post.ply'),denoised_path=str(den),output_N=pp['output_faces'],reused=False))
    csv_write(OUT/'post_index.csv',index)
    write(OUT/'postprocess_completion.json',dict(status='PASS',new_postprocess=24,reused_RAW_D1=24,all_C_unchanged=True,wall_seconds=time.perf_counter()-tick,intermediate_GEO_C=0))
    return index

def distances_stats(x,y,K):
    d=np.linalg.norm(x[K:].astype(np.float64)-y[K:].astype(np.float64),axis=-1).ravel()
    return dict(mean=float(d.mean()) if len(d) else None,p95=float(np.quantile(d,.95)) if len(d) else None,max=float(d.max()) if len(d) else None)

def score_one(r):
    tick=time.perf_counter();stage=r['stage'];rid=r['request_id'];K=int(r['K'])
    if stage not in ('D3','D5'):raise ValueError('Only new endpoints may be scored')
    dr,pp=validate_post(r);dest=checked(OUT/'scores'/stage/rid);receipt=dest/'score.json'
    identity=dict(label=LABEL,stage=stage,request_id=rid,raw_sha256=r['raw_sha256'],post_sha256=r['post_sha256'],registration_sha256=sha(OUT/'registration.json'),score_source_sha256=sha(__file__))
    if receipt.exists():
        saved=read(receipt)
        if saved['identity']!=identity or saved['status']!='PASS':raise ValueError('Cannot reuse mismatched score')
        for path,h in saved['files'].items():
            if sha(path)!=h:raise ValueError('Scoring detail changed')
        return dict(reused=True,stage=stage,request_id=rid,path=str(receipt))
    with np.load(r['raw_path'],allow_pickle=False) as a:raw=a['raw'].reshape(-1,3,3)
    with np.load(r['post_path'],allow_pickle=False) as a:
        post=a['output'];vertices=a['vertices'];faces=a['faces'];source_ids=a['source_face_ids']
    with np.load(r['condition_source'],allow_pickle=False) as a:
        C=a['C'];bid,pts,boundary_audit=A.topology(C,a['C_source_vertex_ids'])
    if not A.bit_equal(C,raw[:K]) or not A.bit_equal(C,post[:K]):raise ValueError('Source/RAW/POST C differs')
    ds,nearest=A.nearest(pts,post[K:]);vrows=[]
    for vid,p,d,j in zip(bid,pts,ds,nearest):
        v=post[K:].reshape(-1,3)[j] if j>=0 else [None]*3
        vrows.append(dict(source_vertex_id=vid,cx=float(p[0]),cy=float(p[1]),cz=float(p[2]),distance_model=float(d) if np.isfinite(d) else None,distance_L_fraction=float(d/A.L) if np.isfinite(d) else None,
            nearest_free_face=int(j)//3 if j>=0 else None,nearest_free_corner=int(j)%3 if j>=0 else None,nearest_original_source_face=int(source_ids[K+int(j)//3]) if j>=0 else None,
            fx=float(v[0]) if j>=0 else None,fy=float(v[1]) if j>=0 else None,fz=float(v[2]) if j>=0 else None))
    metrics=[dict(setting=n,threshold_model=t,scale=s,tau=v,**A.metric(ds,t)) for n,t,s,v in A.SETTINGS]
    quality=Q.score_quality(r,r['post_path'],checked(dest/'quality'))
    with np.load(r['denoised_path'],allow_pickle=False) as a:den=a['output']
    d1path=OLD/'denoised'/LABEL/rid/'denoised.npz'
    with np.load(d1path,allow_pickle=False) as a:d1=a['output']
    displacements=dict(RAW_to_denoised_free_model=distances_stats(raw,den,K),D1_to_denoised_free_model=distances_stats(d1,den,K))
    original_kept=raw[source_ids];den_kept=den[source_ids]
    displacements['RAW_to_POST_retained_free_model']=distances_stats(original_kept,post,K)
    displacements['GEO_C_retained_free_model']=distances_stats(den_kept,post,K)
    result=dict(status='PASS',version=VERSION,identity=identity,metadata={k:r[k] for k in META},stage=stage,C_unchanged=True,metrics=metrics,vertices=vrows,boundary_audit=boundary_audit,
        topology_POST_INDEXED=P.topology_stats(vertices,faces,K),quality=quality,postprocess_stats=pp['stats'],denoise=dr,displacements=displacements,seconds=time.perf_counter()-tick,files=quality['files'])
    write(receipt,result)
    if 'torch' in sys.modules:raise AssertionError('CPU evaluation unexpectedly imported torch')
    return dict(reused=False,stage=stage,request_id=rid,path=str(receipt),seconds=result['seconds'],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def index_rows():
    reg=registration();rows=csv_read(OUT/'post_index.csv')
    expected={(r['request_id'],s) for r in reg['records'] for s in STAGES}
    if len(rows)!=48 or {(r['request_id'],r['stage']) for r in rows}!=expected:raise ValueError('48-stage output index incomplete or duplicated')
    return rows

def score_all(workers=2):
    registration(True);rows=[r for r in index_rows() if r['stage'] in ('D3','D5')];tick=time.perf_counter()
    before=A.resources();before['dispatch']=dict(CPU_workers=1,BLAS_threads_per_worker=2,GPU_workers=0,reason='First authorized real score measured before parallel dispatch')
    write(OUT/'resources_before_cpu_scoring.json',before)
    first=score_one(rows[0]);event('CPU_SCORE_COMPLETE',**first)
    snapshot=A.resources();snapshot['dispatch']=dict(CPU_workers=workers,BLAS_threads_per_worker=2,GPU_workers=0,first_actual_score=first,reason='24 independent frozen-geometry endpoint audits; two CPU workers bounded by registered budget, no GPU or extra probes')
    write(OUT/'resources_before_parallel_scoring.json',snapshot)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs={pool.submit(score_one,r):r for r in rows[1:]}
        for f in concurrent.futures.as_completed(jobs):
            r=jobs[f]
            try:event('CPU_SCORE_COMPLETE',**f.result())
            except Exception as exc:
                event('CPU_SCORE_FAILED',stage=r['stage'],request_id=r['request_id'],error=repr(exc));raise
    write(OUT/'score_completion.json',dict(status='PASS',new_scored_outputs=24,reused_RAW_D1_outputs=24,CPU_workers=workers,BLAS_threads_per_worker=2,wall_seconds=time.perf_counter()-tick,model_forwards=0,new_queries=0))
    summarize()

def parse(v):
    if v in ('','None','null',None):return None
    try:return float(v)
    except (TypeError,ValueError):return v

def numeric_summary(rows,exclude):
    fields=set.intersection(*(set(r) for r in rows))-set(exclude);out={}
    for k in sorted(fields):
        vals=[r[k] for r in rows if isinstance(r[k],(int,float)) and r[k] is not None]
        if vals or all(r[k] is None for r in rows):
            out[k]=float(np.mean(vals)) if vals else None;out[k+'_valid_outputs']=len(vals)
    return out

def flatten_numeric(prefix,data):
    result={}
    for k,v in data.items():
        name=prefix+k
        if isinstance(v,dict):result.update(flatten_numeric(name+'_',v))
        elif isinstance(v,(str,int,float,bool)) or v is None:result[name]=v
    return result

def summarize():
    reg=registration();rows=index_rows();per=[];verts=[];quals=[];topos=[];effects=[]
    rawq={(r['request_id'],r['stage']):r for r in csv_read(OLD/'quality_per_output.csv') if r['label']==LABEL}
    by_record={r['request_id']:r for r in reg['records']}
    for r in rows:
        stage=r['stage'];rid=r['request_id'];meta={k:r[k] for k in META};meta.update(label=LABEL,stage=stage)
        d1=old_receipt(by_record[rid])
        if stage=='RAW':
            p=read(RAW_AVMR/'scores'/LABEL/(rid+'.json'))
            if p['status']!='PASS' or p['identity']['raw_sha256']!=r['raw_sha256']:raise ValueError('RAW score receipt mismatch')
            metrics=p['metrics'];vertex=p['vertices'];quality={k:parse(v) for k,v in rawq[(rid,'RAW')].items() if k not in META and k not in ('label','stage')};topo=d1['topology_RAW_EXACT_COORD_INDEX'];toponame='RAW_EXACT_COORD_INDEX'
        else:
            if stage=='D1':p=d1
            else:
                score_one(r);p=read(OUT/'scores'/stage/rid/'score.json')
            metrics=p['metrics'];vertex=p['vertices'];quality=dict(p['quality']['metrics'],free_face_denominator=p['quality']['free_face_denominator']);topo=p['topology_POST_INDEXED'];toponame='POST_INDEXED'
            effects.append(dict(meta,**flatten_numeric('',p['postprocess_stats']),**flatten_numeric('denoise_',p['denoise']),**flatten_numeric('displacement_',p.get('displacements',{}))))
        per.extend(dict(meta,**v) for v in metrics);verts.extend(dict(meta,**v) for v in vertex);quals.append(dict(meta,**quality));topos.append(dict(meta,index_definition=toponame,**topo))
    csv_write(OUT/'per_output.csv',per);csv_write(OUT/'per_vertex_distances.csv',verts);csv_write(OUT/'quality_per_output.csv',quals);csv_write(OUT/'topology_per_output.csv',topos);csv_write(OUT/'modifications_per_output.csv',effects)
    summary=[];qsummary=[];tsummary=[]
    for stage in STAGES:
        for group in ('ALL','seat','back'):
            for setting,_,_,_ in A.SETTINGS:
                sub=[r for r in per if r['stage']==stage and r['setting']==setting and (group=='ALL' or r['region_type']==group)]
                valid=[r for r in sub if r['status']=='PASS']
                summary.append(dict(label=LABEL,stage=stage,group=group,setting=setting,outputs=len(sub),valid_outputs=len(valid),A_VMR=float(np.mean([r['VMR'] for r in valid])) if valid else None,PMR=float(np.mean([r['perfect'] for r in valid])) if valid else None))
            qs=[r for r in quals if r['stage']==stage and (group=='ALL' or r['region_type']==group)]
            ts=[r for r in topos if r['stage']==stage and (group=='ALL' or r['region_type']==group)]
            qsummary.append(dict(label=LABEL,stage=stage,group=group,outputs=len(qs),**numeric_summary(qs,(*META,'label','stage'))))
            tsummary.append(dict(label=LABEL,stage=stage,group=group,outputs=len(ts),**numeric_summary(ts,(*META,'label','stage')),outputs_with_C_free_shared_edge=sum(r['C_open_edges_with_free']>0 for r in ts)))
    csv_write(OUT/'summary.csv',summary);csv_write(OUT/'quality_summary.csv',qsummary);csv_write(OUT/'topology_summary.csv',tsummary)
    lookup={(r['request_id'],r['stage'],r['setting']):r for r in per};ql={(r['request_id'],r['stage']):r for r in quals};tl={(r['request_id'],r['stage']):r for r in topos}
    paired=[];qpaired=[];tpaired=[]
    for r in reg['records']:
        rid=r['request_id'];meta={k:r[k] for k in META};meta['label']=LABEL
        for stage in ('RAW','D3','D5'):
            for setting,_,_,_ in A.SETTINGS:
                a,b=lookup[(rid,'D1',setting)],lookup[(rid,stage,setting)]
                paired.append(dict(meta,reference='D1',stage=stage,setting=setting,D1_VMR=a['VMR'],endpoint_VMR=b['VMR'],delta_VMR=b['VMR']-a['VMR'],D1_perfect=a['perfect'],endpoint_perfect=b['perfect'],delta_matched_count=b['matched_count']-a['matched_count']))
            for dst,source in ((qpaired,ql),(tpaired,tl)):
                a,b=source[(rid,'D1')],source[(rid,stage)];fields=set(a)&set(b)-set(META)-{'label','stage'};vals={}
                for k in sorted(fields):
                    if isinstance(a[k],(int,float)) and isinstance(b[k],(int,float)):
                        vals['D1_'+k]=a[k];vals['endpoint_'+k]=b[k];vals['delta_'+k]=b[k]-a[k]
                dst.append(dict(meta,reference='D1',stage=stage,**vals))
    csv_write(OUT/'paired_vs_D1.csv',paired);csv_write(OUT/'quality_paired_vs_D1.csv',qpaired);csv_write(OUT/'topology_paired_vs_D1.csv',tpaired)
    parent=[]
    for stage in STAGES:
        for cid in sorted({r['condition_id'] for r in reg['records']}):
            samples=[r for r in quals if r['stage']==stage and r['condition_id']==cid]
            parent.append(dict(label=LABEL,stage=stage,condition_id=cid,parent_id=samples[0]['parent_id'],region_type=samples[0]['region_type'],noise_outputs=len(samples),**numeric_summary(samples,(*META,'label','stage'))))
    csv_write(OUT/'per_condition_noise_quality.csv',parent)
    after={p:sha(p) for p in reg['protected']}
    if after!=reg['protected']:raise ValueError('Protected inputs changed')
    write(OUT/'scoring_protected_after.json',dict(status='PASS',count=len(after),files=after))
    write(OUT/'scoring_aggregation.json',dict(status='PASS',stage_outputs=48,new_scored_outputs=24,threshold_rows=len(per),vertex_rows=len(verts),quality_rows=len(quals),topology_rows=len(topos),settings=len(A.SETTINGS),stages=list(STAGES),torch_imported='torch' in sys.modules))
    event('AGGREGATED',stage_outputs=48,threshold_rows=len(per),quality_rows=len(quals));print(json.dumps(dict(status='PASS',new_scores=24,stage_outputs=48,summary=str(OUT/'summary.csv'))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('postprocess','score','summarize','all'),default='all');p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    if a.stage in ('postprocess','all'):postprocess_all()
    if a.stage in ('score','all'):score_all(a.workers)
    if a.stage=='summarize':summarize()
if __name__=='__main__':main()