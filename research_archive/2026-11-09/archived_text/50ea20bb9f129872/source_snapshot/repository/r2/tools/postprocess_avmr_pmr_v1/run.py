"""Fixed 48-output RAW -> official DENOISE_C -> GEO_C -> CPU scoring controller.
Old scoring files and raw assets are read-only. No training or new rollouts.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, collections, concurrent.futures, importlib.util, json, sys, time, resource, multiprocessing
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'experiments/r2_postprocess_avmr_pmr_v1'
OLD=REPO/'experiments/r2_avmr_pmr_offline_v1'
MIX=REPO/'experiments/r2_user_mix_from_official_v1'
sys.path.insert(0,str(REPO))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
A=module('_post_frozen_avmr',REPO/'r2/tools/avmr_pmr_v1/run.py')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
from native_t1 import postprocess_cli as PC
from native_t1.postprocessing import FIXED_CONFIG
sha,read,csv_read=A.sha,A.read,A.csv_read

def checked_output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):raise ValueError('Refusing any output outside experiment root: '+str(path))
    return path

def write(path,value):return A.write(checked_output(path),value)
def csv_write(path,rows):return A.csv_write(checked_output(path),rows)
VERSION='r2_official_denoise_c_geo_c_evaluation_v1'
STAGE='DENOISE_C_GEO_C'
META=('request_id','condition_id','parent_id','region_type','tier','candidate','N','K')

def event(kind,**fields):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**fields))+'\n')

def records():
    oldreg=read(MIX/'evaluation/registration.json')
    req={q['request_id']:q for q in oldreg['requests']}
    rows=[dict(req[r['request_id']],**{k:v for k,v in r.items() if k not in req[r['request_id']]}) for r in csv_read(MIX/'evaluation/raw_index.csv')]
    assert len(rows)==48 and len({(r['label'],r['request_id']) for r in rows})==48
    return sorted(rows,key=lambda r:(A.LABELS.index(r['label']),r['request_id']))

def quality_module():
    return module('_post_quality_adapter',Path(__file__).with_name('quality.py'))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'registration.json'
    if path.exists():
        reg=read(path)
        for p,h in reg['protected'].items():
            if sha(p)!=h:raise ValueError('Protected source changed: '+p)
        return reg
    q=quality_module()
    protected=dict(read(OLD/'registration.json')['protected'])
    for protected_path,expected in protected.items():
        if sha(protected_path)!=expected:raise ValueError('Original frozen hash mismatch: '+protected_path)
    q.source_hashes()
    for folder in (OLD,):
        for p in folder.rglob('*'):
            if p.is_file():protected[str(p)]=sha(p)
    extra=q.protected_paths()+[str(Path(__file__)),str(Path(__file__).with_name('quality.py')),
      str(REPO/'native_t1/postprocessing.py'),str(REPO/'native_t1/postprocess_cli.py'),str(REPO/'r2/tools/avmr_pmr_v1/run.py')]
    for p in extra:
        actual=sha(p)
        if str(p) in protected and protected[str(p)]!=actual:raise ValueError('Conflicting protection hash')
        protected[str(p)]=actual
    reg=dict(version=VERSION,stage=STAGE,outputs=48,records=records(),protected=protected,
      avmr_definition=read(OLD/'registration.json'),postprocess_config=FIXED_CONFIG,
      denoise_contract='See denoise_registration.json; one official frozen forward per existing RAW; restore original C exactly before GEO_C',
      quality_definition=q.definition(),quality_source_hashes=q.source_hashes(),status='REGISTERED',budget=dict(final_post_outputs=48,new_generator_rollouts=0,training_updates=0,
        denoiser_forwards=48,quality_outputs=48),settings=read(OLD/'registration.json')['settings'],
      topology='Exact-coordinate indexed RAW versus actual exported indexed POST; C-open edges descriptive, no required attachment domain',
      C_protection='C coordinates, order, multiplicity and winding fixed; no claim of arbitrary original vertex-ID graph preservation',
      exclusions=['No threshold sweep','No new Gaussian','No GT reconstruction score','No S_join','No default model change'])
    if path!=OUT/'registration.json':raise AssertionError('Registration destination identity changed')
    write(path,reg);write(OUT/'resources_before_cpu.json',A.resources())
    for p in extra:
        src=Path(p);dst=OUT/'source_snapshot/local'/src.relative_to(REPO) if src.is_relative_to(REPO) else OUT/'source_snapshot/external'/src.name
        dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
    event('REGISTERED',outputs=48,geometry_config=FIXED_CONFIG,model_loads=0)
    return reg

def exact_index(tri):
    vertices=[];mapping={};faces=[]
    for face in tri:
        row=[]
        for p in face:
            key=p.tobytes()
            if key not in mapping:mapping[key]=len(vertices);vertices.append(p)
            row.append(mapping[key])
        faces.append(row)
    return np.asarray(vertices,np.float32).reshape(-1,3),np.asarray(faces,np.int64).reshape(-1,3)

def topology_stats(vertices,faces,K):
    edges=collections.defaultdict(list);ce=collections.Counter();fe=collections.Counter()
    for i,f in enumerate(faces):
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            key=tuple(sorted((int(a),int(b))));edges[key].append((i,int(a),int(b)))
            (ce if i<K else fe)[key]+=1
    cb=[e for e,n in ce.items() if n==1]
    attached=[e for e in cb if fe[e]>0]
    valid=[e for e in attached if len(edges[e])==2 and edges[e][0][1:]==edges[e][1][1:][::-1]]
    cb_length=sum(float(np.linalg.norm(vertices[a].astype(float)-vertices[b])) for a,b in cb)
    att_length=sum(float(np.linalg.norm(vertices[a].astype(float)-vertices[b])) for a,b in attached)
    parent=list(range(len(faces)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for info in edges.values():
        for item in info[1:]:parent[find(item[0])]=find(info[0][0])
    return dict(vertices=len(vertices),faces=len(faces),C_open_edges=len(cb),C_open_edges_with_free=len(attached),
       C_open_edge_attached_fraction=len(attached)/len(cb) if cb else None,
       C_open_length_attached_fraction=att_length/cb_length if cb_length else None,
       C_free_two_face_opposite_winding_edges=len(valid),
       boundary_edges=sum(len(v)==1 for v in edges.values()),nonmanifold_edges=sum(len(v)>2 for v in edges.values()),
       all_edges_two_faces=bool(edges) and all(len(v)==2 for v in edges.values()),
       shared_edge_components=len({find(i) for i in range(len(faces))}),
       interpretation='Descriptive indexed connectivity; no self-intersection/manifold/watertight certificate')

def postprocess_all():
    reg=prepare();index=[]
    for r in reg['records']:
        label,rid,K=r['label'],r['request_id'],int(r['K'])
        den=OUT/'denoised'/label/rid/'denoised.npz'
        denreceipt=read(den.with_name('receipt.json'))
        if any(denreceipt[k]!=v for k,v in dict(label=label,request_id=rid,raw_sha256=r['raw_sha256'],denoise_registration_sha256=sha(OUT/'denoise_registration.json')).items()):raise ValueError('Denoise RAW/request/contract identity mismatch')
        if denreceipt['status']!='PASS' or sha(den)!=denreceipt['output_sha256']:raise ValueError('Denoise receipt invalid')
        dest=OUT/'outputs'/label/rid
        receipt=dest/'postprocess.json'
        if receipt.exists():
            p=read(receipt)
            if p['status']!='PASS' or p['input']['file_sha256']!=sha(den):raise ValueError('Cannot resume failed/mismatched POST')
            for f,v in p['files'].items():
                if sha(dest/f)!=v['sha256']:raise ValueError('POST hash mismatch')
        else:
            event('GEO_C_STARTED',label=label,request_id=rid)
            p=PC.run(den,K,dest,input_key='output')
            event('GEO_C_COMPLETE',label=label,request_id=rid,seconds=p['seconds'],deleted=p['stats']['deleted_free_faces'])
        with np.load(r['raw_path'],allow_pickle=False) as a:raw=a['raw'].reshape(-1,3,3)
        with np.load(dest/'post.npz',allow_pickle=False) as a:
            if not A.bit_equal(a['C'],raw[:K]) or not A.bit_equal(a['output'][:K],raw[:K]):raise ValueError('RAW-to-POST C changed')
        index.append({**r,'post_path':str(dest/'post.npz'),'post_sha256':sha(dest/'post.npz'),
                      'post_ply':str(dest/'post.ply'),'denoised_path':str(den),'stage':STAGE,'output_N':p['output_faces']})
    csv_write(OUT/'post_index.csv',index)
    write(OUT/'postprocess_completion.json',dict(status='PASS',outputs=len(index),stage=STAGE,all_C_unchanged=True))
    return index

def score_one(r):
    start=time.perf_counter();K=int(r['K']);dest=OUT/'scores'/r['label']/r['request_id'];receipt=dest/'score.json'
    identity=dict(label=r['label'],request_id=r['request_id'],raw_sha256=r['raw_sha256'],post_sha256=r['post_sha256'],
                  registration_sha256=sha(OUT/'registration.json'))
    if sha(r['raw_path'])!=r['raw_sha256'] or sha(r['post_path'])!=r['post_sha256']:raise ValueError('Geometry hash mismatch before cache reuse')
    dr=read(Path(r['denoised_path']).with_name('receipt.json'))
    if any(dr[k]!=v for k,v in dict(label=r['label'],request_id=r['request_id'],raw_sha256=r['raw_sha256'],denoise_registration_sha256=sha(OUT/'denoise_registration.json')).items()):raise ValueError('Denoise identity mismatch')
    if sha(r['denoised_path'])!=dr['output_sha256']:raise ValueError('Denoised file changed')
    pp=read(Path(r['post_path']).with_name('postprocess.json'))
    if pp['input']['file_sha256']!=dr['output_sha256'] or pp['files']['post.npz']['sha256']!=r['post_sha256'] or pp['status']!='PASS':raise ValueError('POST ancestry mismatch')
    if receipt.exists():
        saved=read(receipt)
        if saved['identity']!=identity or saved['status']!='PASS':raise ValueError('Existing score identity/status mismatch')
        for p,h in saved['files'].items():
            if sha(p)!=h:raise ValueError('Score detail changed')
        return {'reused':True,'path':str(receipt),'label':r['label'],'request_id':r['request_id']}
    if sha(r['raw_path'])!=r['raw_sha256'] or sha(r['post_path'])!=r['post_sha256']:raise ValueError('Geometry hash mismatch')
    with np.load(r['raw_path'],allow_pickle=False) as a:raw=a['raw'].reshape(-1,3,3)
    with np.load(r['post_path'],allow_pickle=False) as a:
        post=a['output'];vertices=a['vertices'];faces=a['faces'];source_ids=a['source_face_ids']
    if not A.bit_equal(raw[:K],post[:K]):raise ValueError('C changed')
    with np.load(r['condition_source'],allow_pickle=False) as a:
        C=a['C'];bid,pts,ba=A.topology(C,a['C_source_vertex_ids'])
    if not A.bit_equal(C,raw[:K]):raise ValueError('Source C differs')
    ds,nearest=A.nearest(pts,post[K:]);vs=[]
    for vid,p,d,j in zip(bid,pts,ds,nearest):
        v=post[K:].reshape(-1,3)[j] if j>=0 else [None]*3
        vs.append(dict(source_vertex_id=vid,cx=float(p[0]),cy=float(p[1]),cz=float(p[2]),
           distance_model=float(d) if np.isfinite(d) else None,distance_L_fraction=float(d/A.L) if np.isfinite(d) else None,
           nearest_free_face=int(j)//3 if j>=0 else None,nearest_free_corner=int(j)%3 if j>=0 else None,
           nearest_original_source_face=int(source_ids[K+int(j)//3]) if j>=0 else None,
           fx=float(v[0]) if j>=0 else None,fy=float(v[1]) if j>=0 else None,fz=float(v[2]) if j>=0 else None))
    vm=[dict(setting=n,threshold_model=t,scale=s,tau=v,**A.metric(ds,t)) for n,t,s,v in A.SETTINGS]
    rv,rf=exact_index(raw)
    topo_raw=topology_stats(rv,rf,K);topo_post=topology_stats(vertices,faces,K)
    q=quality_module().score_quality(r,r['post_path'],dest/'quality')
    pp=read(Path(r['post_path']).with_name('postprocess.json'))
    dr=read(Path(r['denoised_path']).with_name('receipt.json'))
    details=dict(status='PASS',identity=identity,metadata={k:r[k] for k in META},stage=STAGE,
         metrics=vm,vertices=vs,topology_RAW_EXACT_COORD_INDEX=topo_raw,topology_POST_INDEXED=topo_post,
         quality=q,postprocess_stats=pp['stats'],denoise=dr,seconds=time.perf_counter()-start,
         C_unchanged=True,files={})
    details['files']=dict(q.get('files',{}))
    dest.mkdir(parents=True,exist_ok=True);write(receipt,details)
    return dict(reused=False,path=str(receipt),label=r['label'],request_id=r['request_id'],seconds=details['seconds'],
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def score_all(workers=2):
    reg=prepare();rows=csv_read(OUT/'post_index.csv')
    if len(rows)!=48 or len({(r['label'],r['request_id']) for r in rows})!=48 or {(r['label'],r['request_id']) for r in rows}!={(r['label'],r['request_id']) for r in reg['records']}:raise ValueError('Incomplete or duplicated POST panel')
    tick=time.perf_counter();first=score_one(rows[0]);event('CPU_SCORE_COMPLETE',**first)
    snapshot=A.resources();snapshot['dispatch']=dict(CPU_workers=workers,BLAS_threads_per_worker=2,GPU_workers=0,
         first_actual_score=first,reason='48 independent geometry audits, previous 1-3 sec/output; 24 CPU cores and >40GiB host RAM observed; cap two workers')
    write(OUT/'resources_before_parallel_cpu.json',snapshot)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs={pool.submit(score_one,r):r for r in rows[1:]}
        for f in concurrent.futures.as_completed(jobs):
            r=jobs[f]
            try:result=f.result();event('CPU_SCORE_COMPLETE',**result)
            except Exception as e:
                event('CPU_SCORE_FAILED',label=r['label'],request_id=r['request_id'],error=repr(e));raise
    write(OUT/'score_completion.json',dict(status='PASS',outputs=48,CPU_workers=workers,
          wall_seconds=time.perf_counter()-tick,model_forwards=0))
    summarize()

def summarize():
    reg=prepare();per=[];vrows=[];qrows=[];trows=[];effects=[]
    indexed=csv_read(OUT/'post_index.csv')
    if len(indexed)!=48 or len({(r['label'],r['request_id']) for r in indexed})!=48:raise ValueError('POST index mismatch')
    for r in indexed:
        if not (OUT/'scores'/r['label']/r['request_id']/'score.json').exists():raise ValueError('Missing score, run score stage first')
        score_one(r)
    old_quality={(r['label'],r['request_id']):r for r in csv_read(MIX/'evaluation/per_output.csv')}
    for r in reg['records']:
        p=read(OUT/'scores'/r['label']/r['request_id']/'score.json')
        old=read(OLD/'scores'/r['label']/(r['request_id']+'.json'))
        meta={k:r[k] for k in META};meta['label']=r['label']
        for stage,vals in [('RAW',old['metrics']),(STAGE,p['metrics'])]:
            for v in vals:per.append(dict(meta,stage=stage,**v))
        for v in p['vertices']:vrows.append(dict(meta,stage=STAGE,**v))
        for name,t in [('RAW_EXACT_COORD_INDEX',p['topology_RAW_EXACT_COORD_INDEX']),('POST_INDEXED',p['topology_POST_INDEXED'])]:
            trows.append(dict(meta,stage=name,**t))
        qrows.append(dict(meta,stage=STAGE,**p['quality']['metrics'],free_face_denominator=p['quality']['free_face_denominator']))
        original=old_quality[(r['label'],r['request_id'])]
        old_values={}
        for k,v in p['quality']['metrics'].items():
            if k not in original:raise ValueError('Original quality field absent: '+k)
            value=original[k]
            if value in ('','None','null'):old_values[k]=None
            else:
                try:old_values[k]=float(value)
                except ValueError:old_values[k]=value
        qrows.append(dict(meta,stage='RAW',**old_values,free_face_denominator=int(r['N'])-int(r['K'])))
        effects.append(dict(meta,**{k:v for k,v in p['postprocess_stats'].items() if isinstance(v,(str,int,float,bool))},
                            denoise_max_free_displacement=p['denoise'].get('max_free_displacement_model'),
                            denoise_C_proposal_max_displacement=p['denoise'].get('C_proposal_max_displacement_model')))
    csv_write(OUT/'per_output.csv',per);csv_write(OUT/'per_vertex_distances_POST.csv',vrows)
    csv_write(OUT/'quality_per_output.csv',qrows);csv_write(OUT/'topology_per_output.csv',trows);csv_write(OUT/'modifications_per_output.csv',effects)
    summary=[];paired=[]
    for stage in ('RAW',STAGE):
        for label in A.LABELS:
            for setting,_,_,_ in A.SETTINGS:
                for group in ('ALL','seat','back'):
                    sub=[v for v in per if v['label']==label and v['stage']==stage and v['setting']==setting and (group=='ALL' or v['region_type']==group)]
                    summary.append(dict(stage=stage,label=label,setting=setting,group=group,outputs=len(sub),
                        A_VMR=float(np.mean([v['VMR'] for v in sub])),PMR=float(np.mean([v['perfect'] for v in sub]))))
    by={(r['label'],r['request_id'],r['setting'],r['stage']):r for r in per}
    for r in reg['records']:
        for name,_,_,_ in A.SETTINGS:
            a,b=(by[(r['label'],r['request_id'],name,s)] for s in ('RAW',STAGE))
            paired.append(dict(label=r['label'],request_id=r['request_id'],parent_id=r['parent_id'],region_type=r['region_type'],setting=name,
                RAW_VMR=a['VMR'],POST_VMR=b['VMR'],delta_VMR=b['VMR']-a['VMR'],RAW_perfect=a['perfect'],POST_perfect=b['perfect'],
                delta_matched_count=b['matched_count']-a['matched_count']))
    csv_write(OUT/'summary.csv',summary);csv_write(OUT/'paired_RAW_POST.csv',paired)
    qsummary=[]
    for stage in ('RAW',STAGE):
        for label in A.LABELS:
            for group in ('ALL','seat','back'):
                sub=[v for v in qrows if v['label']==label and v['stage']==stage and (group=='ALL' or v['region_type']==group)]
                fields=set.intersection(*(set(v) for v in sub))-set(META)-{'label','stage'}
                vals={}
                for k in fields:
                    values=[v[k] for v in sub if isinstance(v[k],(int,float)) and v[k] is not None]
                    if values or all(v[k] is None for v in sub):
                        vals[k]=float(np.mean(values)) if values else None
                        vals[k+'_valid_outputs']=len(values)
                qsummary.append(dict(stage=stage,label=label,group=group,outputs=len(sub),**vals))
    csv_write(OUT/'quality_summary.csv',qsummary)
    after={p:sha(p) for p in reg['protected']}
    if after!=reg['protected']:raise ValueError('Protected source changed during execution')
    write(OUT/'protected_after.json',dict(status='PASS',count=len(reg['protected']),files=after))
    event('AGGREGATED',outputs=48,threshold_rows=len(per))
    print(json.dumps(dict(status='PASS',outputs=48,summary=str(OUT/'summary.csv'))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','postprocess','score','summarize','all'],default='all');p.add_argument('--workers',type=int,choices=[1,2],default=2);a=p.parse_args()
    if a.stage=='prepare':prepare()
    if a.stage in ('postprocess','all'):postprocess_all()
    if a.stage in ('score','all'):score_all(a.workers)
    if a.stage=='summarize':summarize()
if __name__=='__main__':main()