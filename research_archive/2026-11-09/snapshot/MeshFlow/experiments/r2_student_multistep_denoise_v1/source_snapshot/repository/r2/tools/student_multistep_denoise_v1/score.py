"""CPU-only, frozen-math S3/S5 evaluation; cached S1/identity/D1 references.

Exactly 24 new GEO_C and 24 quality jobs. Never calls a model, samples queries,
or writes to an earlier experiment. Per-job receipts avoid concurrent JSONL.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,concurrent.futures,csv,json,multiprocessing,resource,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.tools.student_multistep_denoise_v1 import common as C
from r2.pilots.anchor_aware_denoiser_v1 import evaluation as E

OUT=C.OUT
SOURCE=Path(__file__).resolve()
STAGES=('IDENTITY_GEO','D1','S1','S3','S5')
REUSED={'IDENTITY_GEO':'IDENTITY_GEO','D1':'D0_GEO','S1':'STUDENT_GEO'}
VERSION='student_multistep_frozen_geometry_v1'

def readcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,obj):C.write(p,E.clean(obj))
def csvwrite(p,rows):
    if rows:C.csvwrite(p,E.clean(rows))
    else:C.atomic(p,lambda f:csv.writer(f).writerow(('stage','request_id','status')))
def meta(r):return dict(E.meta(E.canonical_identity(r)),stage=r['stage'])
def verify(mapping):
    for p,h in mapping.items():assert C.sha(p)==h,p
def event(kind,r,**kw):
    p=OUT/'score_events'/r['stage']/r['request_id']/(kind+'.json')
    if p.exists():raise RuntimeError('Duplicate job event: '+str(p))
    write(p,dict(event=kind,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**meta(r),**kw))
def job_open(kind,r):
    directory=OUT/'score_events'/r['stage']/r['request_id']
    if list(directory.glob(kind+'_*.json')):raise RuntimeError('Unfinished attempted job; no automatic replay')
    event(kind+'_STARTED',r)
def payload(p):return E.payload(p)

def prepare():
    master=C.read(OUT/'registration.json');C.verify(master)
    target=OUT/'score_registration.json'
    if target.exists():
        reg=C.read(target);assert reg['parent_registration_sha256']==C.sha(OUT/'registration.json')
        assert reg['score_source_sha256']==C.sha(SOURCE)
        verify(reg['protected']);return reg
    A,Q,PC,config=E.helpers()
    assert master['postprocess_config']==config
    assert master['settings']==[dict(name=n,threshold_model=t,scale=s,tau=v) for n,t,s,v in A.SETTINGS]
    old=C.read(C.DIST/'evaluation/registration.json')
    protected={}
    def protect(p):protected[str(p)]=C.sha(p)
    for p in (Path(E.__file__),C.ROOT/'r2/tools/postprocess_avmr_pmr_v1/run.py',
              C.ROOT/'r2/tools/postprocess_avmr_pmr_v1/quality.py',C.ROOT/'r2/tools/avmr_pmr_v1/run.py',
              C.ROOT/'native_t1/postprocessing.py',C.ROOT/'native_t1/postprocess_cli.py',
              C.DIST/'evaluation/post_index.csv',C.DIST/'evaluation/registration.json',*Q.protected_paths()):protect(p)
    available={(r['method'],r['request_id']):r for r in readcsv(C.DIST/'evaluation/post_index.csv') if r['label']=='R2_USER_MIX'}
    baselines=[]
    for orig in master['records']:
        for stage,method in REUSED.items():
            oldrow=available[method,orig['request_id']]
            for key in ('label','parent_id','condition_id','raw_sha256','Gaussian_sha256','C_sha256','checkpoint_sha256'):
                assert oldrow[key]==orig[key],key
            for key in ('N','K','candidate'):assert int(oldrow[key])==int(orig[key])
            d=C.read(oldrow['diagnostic_path']);assert d['status']=='PASS'
            assert d['identity']['post_sha256']==C.sha(oldrow['post_path'])==oldrow['post_sha256']
            assert d['quality']['status']=='PASS'
            assert d['quality']['identity']['processed_sha256']==oldrow['post_sha256']
            for mapping in (d['files'],d['quality']['files']):
                E.verify_files(mapping)
                for p in mapping:protect(p)
            for key in ('diagnostic_path','post_path','post_ply','denoised_path'):protect(oldrow[key])
            protect(Path(oldrow['post_path']).with_name('postprocess.json'))
            row=dict(orig)
            row.update({k:oldrow[k] for k in ('post_path','post_ply','post_sha256','denoised_path','pre_GEO_key','diagnostic_path','denoiser_checkpoint_sha256','output_N')})
            row.update(stage=stage,method=stage,denoiser=stage,reused=True)
            baselines.append(row)
    assert len(baselines)==36
    reg=dict(version=VERSION,status='REGISTERED',parent_registration_sha256=C.sha(OUT/'registration.json'),
        score_source_sha256=C.sha(SOURCE),records=master['records'],baselines=baselines,protected=protected,
        settings=master['settings'],postprocess_config=config,L=A.L,quality_definition=Q.definition(),
        quality_source_hashes=Q.source_hashes(),topology_definition=old['topology_definition'],
        reachable_area_definition=old['reachable_area_definition'],
        budget=dict(new_GEO=24,new_quality=24,reused_POST=36,primary_POST=60,RAW_reference=12,model_calls=0,new_queries=0),
        main_comparisons=['S3_minus_S1','S5_minus_S1','S5_minus_S3'],CPU_workers_max=2,threads_per_worker=2,
        legacy_components='Original postprocess topology_stats shared-edge components; descriptive only, no fragment or manifold certificate')
    write(target,reg);return reg

def check_inference(r,stage):
    den=OUT/'denoised'/stage/r['request_id']/'denoised.npz';rec=C.read(den.with_name('receipt.json'))
    assert rec['status']=='PASS' and rec['output_sha256']==C.sha(den)
    assert rec['request_id']==r['request_id'] and int(rec['round'])==int(stage[1:])
    assert rec['raw_sha256']==r['raw_sha256'] and rec['checkpoint_sha256']==C.EXPECTED
    assert rec['registration_sha256']==C.sha(OUT/'registration.json')
    assert C.sha(rec['input_path'])==rec['input_sha256']
    a=payload(den);raw=payload(r['raw_path'])['raw'].reshape(-1,3,3);K=int(r['K'])
    assert a['output'].shape==raw.shape and a['output'].dtype==np.float32 and np.isfinite(a['output']).all()
    assert E.same(a['output'][:K],raw[:K]) and E.same(a['C'],raw[:K])
    assert E.same(a['proposal_model'][K:],a['output'][K:])
    return den

def postprocess():
    reg=prepare();completion=C.read(OUT/'denoise_completion.json');assert completion['status']=='PASS'
    _,_,PC,_=E.helpers();rows=list(reg['baselines'])
    for stage in ('S3','S5'):
        for orig in reg['records']:
            row=dict(orig,stage=stage,method=stage,denoiser=stage,reused=False)
            den=check_inference(row,stage);dest=C.checked(OUT/'outputs'/stage/row['request_id'])
            if (dest/'postprocess.json').exists():
                pp=C.read(dest/'postprocess.json');assert pp['status']=='PASS' and pp['input']['file_sha256']==C.sha(den)
                for name,v in pp['files'].items():assert C.sha(dest/name)==v['sha256']
            else:
                staging=C.checked(dest.with_name(dest.name+'.staging'))
                if dest.exists() or staging.exists():raise RuntimeError('Incomplete GEO staging; no automatic replay')
                job_open('GEO',row);pp=PC.run(den,int(row['K']),staging,input_key='output')
                write(staging/'postprocess.json',E.rewrite_paths(pp,staging,dest));os.replace(staging,dest)
                event('GEO_COMPLETE',row,seconds=pp['seconds'])
            post=dest/'post.npz';a=payload(post);raw=payload(row['raw_path'])['raw'].reshape(-1,3,3)
            assert E.same(a['output'][:int(row['K'])],raw[:int(row['K'])])
            assert E.same(a['vertices'][a['faces']],a['output'])
            v,f=PC.read_indexed_ply(dest/'post.ply');assert E.same(v,a['vertices']) and E.same(f,a['faces'])
            row.update(denoised_path=str(den),pre_GEO_key='output',post_path=str(post),post_sha256=C.sha(post),
                post_ply=str(dest/'post.ply'),output_N=len(a['output']),denoiser_checkpoint_sha256=C.EXPECTED,
                diagnostic_path=str(OUT/'diagnostics'/stage/row['request_id']/'diagnostic.json'))
            rows.append(row)
    rows += [dict(r,stage='RAW',method='RAW',post_path=r['raw_path'],post_sha256=r['raw_sha256'],reused=True) for r in reg['records']]
    assert len(rows)==72
    csvwrite(OUT/'post_index.csv',rows)
    write(OUT/'post_completion.json',dict(status='PASS',primary_POST=60,RAW_reference=12,new_GEO=24,reused_POST=36,C_and_PLY_exact=True))
    return rows

def score_one(row):
    row=E.canonical_identity(row);diag=Path(row['diagnostic_path']);post=Path(row['post_path']);den=Path(row['denoised_path'])
    ident=dict(meta(row),post_sha256=C.sha(post),input_sha256=C.sha(den),score_registration_sha256=C.sha(OUT/'score_registration.json'))
    if diag.exists():
        d=C.read(diag);assert d['status']=='PASS' and d['identity']==ident
        E.verify_files(d['files']);E.verify_files(d['quality']['files']);return str(diag)
    start=time.perf_counter();A,Q,_,_=E.helpers();dest=C.checked(OUT/'scores'/row['stage']/row['request_id'])
    if (dest/'score.json').exists():
        quality=C.read(dest/'score.json');assert quality['status']=='PASS' and quality['identity']['processed_sha256']==C.sha(post)
        E.verify_files(quality['files'])
    else:
        staging=C.checked(dest.with_name(dest.name+'.staging'))
        if dest.exists() or staging.exists():raise RuntimeError('Unfinished quality job; no automatic replay')
        job_open('QUALITY',row);quality=Q.score_quality(row,post,C.checked(staging/'quality'))
        quality=E.rewrite_paths(quality,staging,dest);write(staging/'score.json',quality);os.replace(staging,dest)
        E.verify_files(quality['files']);event('QUALITY_COMPLETE',row,seconds=quality['seconds'])
    a=payload(post);K=int(row['K']);before=payload(den)['output']
    topology,edges,reached=E.actual_valid_graph(a['vertices'],a['faces'],K,C.read(quality['pair_audit_path']))
    verts,metrics=E.boundary_values(row,a['output'],A);pv,pm=E.boundary_values(row,before,A)
    shape=Q._old().cpu_math()['shape'].shape_quality(before[K:],A.L)
    out=C.checked(diag.parent);csvwrite(out/'valid_edge_details.csv',edges)
    write(out/'pre_GEO_shape_per_face.json',shape)
    d=dict(status='PASS',identity=ident,quality=quality,topology=topology,vertices=verts,metrics=metrics,
        pre_GEO_vertices=pv,pre_GEO_metrics=pm,pre_GEO_shape=dict(free_faces=len(before)-K,
            free_total_surface_area=shape['total_area'],free_minimum_angle_lt5_fraction=shape['min_angle_lt5_fraction'],
            free_q_gt10_fraction=shape['q_actual_gt10_fraction'],free_exact_zero_area_count=shape['exact_zero_area_count'],
            free_near_degenerate_count=shape['near_degenerate_area_count']),
        reached_free_output_face_ids=reached,source_face_ids=a['source_face_ids'].tolist(),
        postprocess=C.read(post.with_name('postprocess.json')),seconds=time.perf_counter()-start,
        peak_cpu_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        files={str(p):C.sha(p) for p in out.iterdir() if p.is_file()})
    write(diag,d);return str(diag)

def scores():
    reg=prepare();rows=[r for r in readcsv(OUT/'post_index.csv') if r['stage'] in ('S3','S5')];assert len(rows)==24
    start=time.perf_counter();write(OUT/'resources_before_score.json',C.resources())
    # First authorized real item provides an observed CPU-memory and time scope.
    t=time.perf_counter();first=score_one(rows[0]);seconds=time.perf_counter()-t
    after=C.resources();write(OUT/'resources_after_first_score.json',after)
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    workers=2 if os.cpu_count()>=4 and available>max(4<<30,3*peak) else 1
    write(OUT/'CPU_dispatch.json',dict(status='PASS',first_real_item=first,first_seconds=seconds,
        first_process_peak_RSS_bytes=peak,available_RAM_bytes=available,workers=workers,threads_per_worker=2,
        reason='Two bounded CPU workers if four cores and measured RSS headroom permit; GPU use zero',
        timing_scope='First authorized real quality plus diagnostic; remaining concurrent wall time is not an isolated model benchmark'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=[first,*pool.map(score_one,rows[1:],chunksize=1)]
    verify(reg['protected']);C.verify(C.read(OUT/'registration.json'))
    for kind in ('GEO','QUALITY'):
        for suffix in ('STARTED','COMPLETE'):assert len(list((OUT/'score_events').glob('*/*/'+kind+'_'+suffix+'.json')))==24
    write(OUT/'score_completion.json',dict(status='PASS',new_quality=24,new_GEO=24,reused_POST=36,receipts=receipts,
        model_calls=0,new_queries=0,workers=workers,threads_per_worker=2,seconds=time.perf_counter()-start))

def edge_keys(post,d):
    a=payload(post);_,edges,_=E.actual_valid_graph(a['vertices'],a['faces'],int(d['identity']['K']),C.read(d['quality']['pair_audit_path']))
    return {tuple(sorted(a['vertices'][i].tobytes().hex() for i in (e['vertex_u'],e['vertex_v']))):e for e in edges if e['valid_edge'] and e['is_C_F']}

def source_events(r,d,br,bd):
    m=meta(r);K=int(r['K']);a=payload(r['post_path']);b=payload(br['post_path']);events=[];faces=[];edges=[]
    old=E.pair_source_map(C.read(bd['quality']['pair_audit_path']),bd['source_face_ids'],K)
    new=E.pair_source_map(C.read(d['quality']['pair_audit_path']),d['source_face_ids'],K)
    oi={p for p,c in old.items() if c in E.ILLEGAL};ni={p for p,c in new.items() if c in E.ILLEGAL};kept=set(d['source_face_ids'])
    for p in sorted(oi|ni):
        retained=all(i in kept for i in p);nc=new.get(p,'disjoint')
        fate=('S1_ILLEGAL_FACE_REMOVED' if not retained else 'PERSISTENT_ILLEGAL' if p in ni else 'BOTH_RETAINED_RESOLVED_NONILLEGAL' if nc=='disjoint' or nc in E.LEGAL else 'BOTH_RETAINED_UNKNOWN') if p in oi else 'NEW_ILLEGAL'
        events.append(dict(**m,raw_source_face_a=p[0],raw_source_face_b=p[1],S1_classification=old.get(p,'disjoint'),new_classification=nc if retained else 'FACE_REMOVED',fate=fate))
    for i in range(K,len(a['source_face_kept'])):
        x,y=bool(a['source_face_kept'][i]),bool(b['source_face_kept'][i])
        faces.append(dict(**m,raw_source_face=i,method_kept=x,S1_kept=y,method_reason=str(a['source_face_reason'][i]),S1_reason=str(b['source_face_reason'][i]),fate='BOTH_KEPT' if x and y else 'METHOD_ONLY_DELETION' if y else 'S1_ONLY_DELETION' if x else 'BOTH_DELETED'))
    ne,oe=edge_keys(r['post_path'],d),edge_keys(br['post_path'],bd)
    for e in sorted(set(ne)|set(oe)):
        edges.append(dict(**m,C_endpoint_a_bits=e[0],C_endpoint_b_bits=e[1],fate='RETAINED' if e in ne and e in oe else 'S1_VALID_EDGE_LOST' if e in oe else 'NEW_VALID_EDGE',method_winding_opposite=ne.get(e,{}).get('winding_opposite'),S1_winding_opposite=oe.get(e,{}).get('winding_opposite')))
    counts=collections.Counter(e['fate'] for e in events)
    assert counts['PERSISTENT_ILLEGAL']+counts['NEW_ILLEGAL']==len(ni)
    summary=dict(**m,S1_illegal_pairs=len(oi),new_illegal_pairs=len(ni),**{k:counts[k] for k in ('S1_ILLEGAL_FACE_REMOVED','PERSISTENT_ILLEGAL','BOTH_RETAINED_RESOLVED_NONILLEGAL','BOTH_RETAINED_UNKNOWN','NEW_ILLEGAL')})
    return summary,events,faces,edges

def numeric(table):return sorted({k for r in table for k,v in r.items() if k not in (*E.META,'stage') and isinstance(v,(float,int))})
def summarize(table,base,nums):
    out=[]
    for name,extra in (('ALL',()),('REGION',('region_type',)),('PARENT',('parent_id',)),('NOISE',('candidate',)),('PARENT_NOISE',('parent_id','candidate'))):
        out.extend(dict(grouping=name,**r) for r in E.summaries(table,base+extra,nums))
    return out

def aggregate():
    reg=prepare();assert C.read(OUT/'score_completion.json')['status']=='PASS'
    rows=[E.canonical_identity(r) for r in readcsv(OUT/'post_index.csv') if r['stage']!='RAW'];assert len(rows)==60
    ds={(r['stage'],r['request_id']):C.read(r['diagnostic_path']) for r in rows};rs={(r['stage'],r['request_id']):r for r in rows}
    P=E.module('_student_multistep_frozen_post',C.ROOT/'r2/tools/postprocess_avmr_pmr_v1/run.py')
    quality=[];top=[];legacy=[];vm=[];supported=[];vertices=[];distance=[];mods=[];preshape=[];fates=[];ff=[];faces=[];edges=[]
    for r in rows:
        d=ds[r['stage'],r['request_id']];m=meta(r);K=r['K'];a=payload(r['post_path'])
        assert d['status']=='PASS' and E.same(a['output'][:K],payload(r['raw_path'])['raw'].reshape(-1,3,3)[:K])
        quality.append(dict(**m,**d['quality']['metrics'],free_face_denominator=d['quality']['free_face_denominator']))
        top.append(dict(**m,**d['topology']));legacy.append(dict(**m,**P.topology_stats(a['vertices'],a['faces'],K)))
        preshape.append(dict(**m,**d['pre_GEO_shape']))
        for phase,mk,vk in (('POST','metrics','vertices'),('PRE_GEO_C_RESTORED','pre_GEO_metrics','pre_GEO_vertices')):
            for v in d[mk]:(vm if v['domain']=='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY' else supported).append(dict(**m,phase=phase,**v))
            vertices.extend(dict(**m,phase=phase,**v) for v in d[vk])
            vv=[v['distance_model'] for v in d[vk]];valid=bool(vv) and all(x is not None for x in vv)
            distance.append(dict(**m,phase=phase,boundary_vertices=len(vv),status='PASS' if valid else 'NA_EMPTY_BOUNDARY_OR_FREE',distance_mean_model=float(np.mean(vv)) if valid else None,distance_median_model=float(np.median(vv)) if valid else None,distance_p95_model=float(np.quantile(vv,.95)) if valid else None,distance_max_model=float(max(vv)) if valid else None))
        pre=payload(r['denoised_path'])[r['pre_GEO_key']].reshape(-1,3,3);raw=payload(r['raw_path'])['raw'].reshape(-1,3,3)
        s1=payload(r['S1_denoised_path'])['output'];drift=np.linalg.norm(pre[K:].astype(float)-raw[K:],axis=-1);delta=np.linalg.norm(pre[K:].astype(float)-s1[K:],axis=-1)
        mods.append(dict(**m,raw_free_faces=len(raw)-K,POST_free_faces=len(a['output'])-K,deleted_free_faces=len(raw)-len(a['output']),deleted_free_fraction=(len(raw)-len(a['output']))/(len(raw)-K),denoise_free_corner_displacement_RAW_mean=float(drift.mean()),denoise_free_corner_displacement_RAW_p95=float(np.quantile(drift,.95)),denoise_free_corner_displacement_RAW_max=float(drift.max()),denoise_free_corner_displacement_S1_mean=float(delta.mean()),denoise_free_corner_displacement_S1_p95=float(np.quantile(delta,.95)),denoise_free_corner_displacement_S1_max=float(delta.max()),GEO_free_corner_displacement_mean=float(a['corner_displacement'][K:].mean()),GEO_free_corner_displacement_max=float(a['corner_displacement'][K:].max()),POST_free_surface_area=d['quality']['metrics']['free_total_surface_area']))
        if r['stage'] in ('S3','S5'):
            summary,f,fa,e=source_events(r,d,rs['S1',r['request_id']],ds['S1',r['request_id']]);fates.append(summary);ff.extend(f);faces.extend(fa);edges.extend(e)
    tables=dict(quality_per_output=quality,topology_per_output=top,legacy_indexed_topology_per_output=legacy,per_output=vm,source_supported_reference_per_output=supported,per_boundary_vertex=vertices,boundary_distance_per_output=distance,pre_GEO_shape_per_output=preshape,modifications_per_output=mods,FF_fates_per_output=fates,FF_pair_fates=ff,GEO_source_face_events=faces,GEO_valid_edge_events=edges)
    for name,table in tables.items():csvwrite(OUT/(name+'.csv'),table)
    for name,table,keys,nums in (('quality',quality,('stage','label'),numeric(quality)),('topology',top,('stage','label'),numeric(top)),('modifications',mods,('stage','label'),numeric(mods)),('AVMR',vm,('stage','label','phase','setting'),('VMR','perfect'))):csvwrite(OUT/(name+'_summary.csv'),summarize(table,keys,nums))
    qmap={(r['stage'],r['request_id']):r for r in quality};tmap={(r['stage'],r['request_id']):r for r in top};lmap={(r['stage'],r['request_id']):r for r in legacy};paired=[];avpaired=[]
    for new,old in (('S3','S1'),('S5','S1'),('S5','S3')):
        for r in [r for r in rows if r['stage']==new]:
            p=dict(E.meta(r),comparison=new+'_minus_'+old)
            for table,mapping in ((quality,qmap),(top,tmap),(legacy,lmap)):
                for k in numeric(table):
                    a,b=mapping[new,r['request_id']].get(k),mapping[old,r['request_id']].get(k)
                    p[('legacy_' if table is legacy else '')+k+'_delta']=float(a)-float(b) if a is not None and b is not None else None
            paired.append(p)
        refs={(r['request_id'],r['phase'],r['setting']):r for r in vm if r['stage']==old}
        for r in [v for v in vm if v['stage']==new]:
            b=refs[r['request_id'],r['phase'],r['setting']]
            avpaired.append(dict(E.meta(r),comparison=new+'_minus_'+old,phase=r['phase'],setting=r['setting'],**{k+'_delta':r[k]-b[k] if r[k] is not None and b[k] is not None else None for k in ('VMR','perfect')}))
    csvwrite(OUT/'paired_per_output.csv',paired);csvwrite(OUT/'paired_summary.csv',summarize(paired,('comparison','label'),[k for k in paired[0] if k.endswith('_delta')]))
    csvwrite(OUT/'AVMR_paired_per_output.csv',avpaired);csvwrite(OUT/'AVMR_paired_summary.csv',summarize(avpaired,('comparison','label','phase','setting'),('VMR_delta','perfect_delta')))
    ids={r['request_id'] for r in reg['records']}
    for src,dst in (('per_output.csv','RAW_AVMR_reference.csv'),('quality_per_output.csv','RAW_quality_reference.csv')):
        oldrows=readcsv(C.MULTI/src);csvwrite(OUT/dst,[r for r in oldrows if r['stage']=='RAW' and r['request_id'] in ids])
        csvwrite(OUT/('teacher_D3_D5_background_'+src),[dict(r,scope='BACKGROUND_PRIOR_TEACHER_EXPERIMENT',topology_scope='Legacy topology excluded from current valid-edge comparisons') for r in oldrows if r['stage'] in ('D3','D5') and r['request_id'] in ids])
    assert len(vm)==960 and len(paired)==36 and len(avpaired)==576
    verify(reg['protected']);C.verify(C.read(OUT/'registration.json'))
    write(OUT/'score_aggregate_completion.json',dict(status='PASS',primary_POST=60,RAW_reference=12,new_GEO=24,new_quality=24,reused_POST=36,AVMR_POST_rows=480,AVMR_PRE_rows=480,per_boundary_vertices=len(vertices),paired_outputs=36,FF_events=len(ff),valid_edge_events=len(edges),model_calls=0,new_queries=0,protected_files=len(reg['protected'])))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','post','score','aggregate','all'),default='all');args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='post':postprocess()
    elif args.stage=='score':scores()
    elif args.stage=='aggregate':aggregate()
    else:postprocess();scores();aggregate()
if __name__=='__main__':main()
