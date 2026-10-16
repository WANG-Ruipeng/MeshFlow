"""CPU-only fixed-panel evaluation and rendering for C restoration coupling.

score(): 24 new GEO_C + 24 new complete POST quality scans; D0/D3 reused.
PRE: inherited all-boundary AVMR, exact-coordinate edge-local classification,
shape only. Complete PRE CF/FF/I is NA_NOT_BUDGETED. No model or new queries.
Call this module in a separate CPU process; inherited helpers hide CUDA.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,concurrent.futures,csv,html,importlib.util,json,multiprocessing,resource,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.tools.c_restore_coupling_v1 import common as C
from r2.pilots.anchor_aware_denoiser_v1 import evaluation as E

OUT=C.OUT
SOURCE=Path(__file__).resolve()
METHODS=('D0_BASE','RBF','FEEDBACK','D3')
NEW=('RBF','FEEDBACK')
VERSION='c_restore_coupling_frozen_geometry_v1'
DOMAIN='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY'

def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,value):C.write(path,E.clean(value))
def atomic(path,writer,binary=False):
    path=C.checked(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=C.checked(path.with_name(path.name+'.tmp'))
    with tmp.open('wb' if binary else 'w',**({} if binary else {'encoding':'utf-8','newline':''})) as f:
        writer(f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def csvwrite(path,rows):
    rows=E.clean(rows);fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    def writer(f):
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    atomic(path,writer)
def verify(mapping):
    for path,h in mapping.items():assert C.sha(path)==h,path
def meta(row):return dict(E.meta(E.canonical_identity(row)),method=row['method'])
def event(kind,row,**fields):
    p=OUT/'score_events'/row['method']/row['request_id']/(kind+'.json')
    if p.exists():raise RuntimeError('Duplicate completed event: '+str(p))
    write(p,dict(event=kind,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**meta(row),**fields))
def job_open(kind,row):
    d=OUT/'score_events'/row['method']/row['request_id']
    if list(d.glob(kind+'_*.json')):raise RuntimeError('Unfinished attempted job; no automatic replay')
    event(kind+'_STARTED',row)

def prepare():
    master=C.read(OUT/'registration.json');C.verify(master,weights=False);target=OUT/'score_registration.json'
    if target.exists():
        reg=C.read(target)
        assert reg['parent_registration_sha256']==C.sha(OUT/'registration.json')
        assert reg['source_sha256']==C.sha(SOURCE)
        verify(reg['protected']);return reg
    A,Q,PC,config=E.helpers()
    records=master['records']
    assert len(records)==12 and len({r['request_id'] for r in records})==12
    assert all(r['label']=='R2_USER_MIX' for r in records)
    assert master['postprocess_config']==config
    assert master['settings']==[dict(name=n,threshold_model=t,scale=s,tau=v) for n,t,s,v in A.SETTINGS]
    protected={}
    def protect(p):protected[str(p)]=C.sha(p)
    for p in [Path(E.__file__),C.ROOT/'r2/tools/avmr_pmr_v1/run.py',C.ROOT/'r2/tools/postprocess_avmr_pmr_v1/quality.py',
              C.ROOT/'native_t1/postprocessing.py',C.ROOT/'native_t1/postprocess_cli.py',
              C.ROOT/'r2/tools/student_multistep_denoise_v1/gallery.py',*Q.protected_paths()]:protect(p)
    baselines=[]
    for r in records:
        for method,prefix,base in [('D0_BASE','D0',C.OLD),('D3','D3',C.MULTI)]:
            rid=r['request_id']
            den=Path(r[prefix+'_denoised_path']);post=Path(r[prefix+'_post_path'])
            sp=(base/'scores'/'R2_USER_MIX'/rid/'score.json') if method=='D0_BASE' else (base/'scores'/'D3'/rid/'score.json')
            s=C.read(sp);q=s['quality'];assert s['status']=='PASS' and q['status']=='PASS'
            dr=C.read(den.with_name('receipt.json'));pp=C.read(post.with_name('postprocess.json'))
            assert dr['status']=='PASS' and dr['request_id']==rid and dr['raw_sha256']==r['raw_sha256']
            assert dr['output_sha256']==C.sha(den)
            assert pp['status']=='PASS' and pp['input']['file_sha256']==C.sha(den)
            for filename,desc in pp['files'].items():assert C.sha(post.parent/filename)==desc['sha256']
            assert s['identity']['raw_sha256']==r['raw_sha256']
            assert s['identity']['post_sha256']==C.sha(post)==q['identity']['processed_sha256']
            assert q['identity']['raw_sha256']==r['raw_sha256']
            for mapping in (s['files'],q['files']):
                verify(mapping)
                for path in mapping:protect(path)
            for path in [den,den.with_name('receipt.json'),post,post.with_name('post.ply'),post.with_name('postprocess.json'),sp,r['raw_path'],r['condition_source']]:protect(path)
            baselines.append(dict(r,method=method,reused=True,denoised_path=str(den),post_path=str(post),
                post_ply=str(post.with_name('post.ply')),post_sha256=C.sha(post),quality_cache_path=str(sp),
                output_N=len(E.payload(post)['output']),pre_GEO_key='output',
                diagnostic_path=str(OUT/'diagnostics'/method/rid/'diagnostic.json')))
    reg=dict(version=VERSION,status='REGISTERED',parent_registration_sha256=C.sha(OUT/'registration.json'),
        source_sha256=C.sha(SOURCE),records=records,baselines=baselines,protected=protected,
        settings=master['settings'],postprocess_config=config,quality_definition=Q.definition(),
        quality_source_hashes=Q.source_hashes(),L=A.L,methods=METHODS,
        budget=dict(new_GEO=24,new_POST_quality=24,reused_POST_quality=24,POST_outputs=48,PRE_outputs=48,
            new_model_calls=0,new_queries=0,full_PRE_quality=0),
        PRE_quality='NA_NOT_BUDGETED',boundary_domain=DOMAIN,required_interface='NOT_REGISTERED; descriptive, not must-attach',
        PRE_topology='FP32 bitwise coordinate index only; no tolerance weld. Only actual shared-edge incident pairs classified with original verified pair detector. Not original source-ID topology.',
        valid_edge='Inherited actual_valid_graph: exactly two incident nonzero/nonduplicate faces, resolved legal pair; winding separately reported.',
        reachable_area='Inherited valid-edge BFS seeded with all C faces; reached free area / whole free area; not semantic body identification.',
        comparisons=['RBF_minus_D0_BASE','FEEDBACK_minus_D0_BASE','FEEDBACK_minus_RBF','D3_minus_D0_BASE'],
        aggregation='Per-output then equal noise within six parents; no independent12-parent CI.',CPU_workers_max=2,threads_per_worker=2)
    # Preserve the exact small scoring sources independently from old read-only files.
    snapshot={}
    for path in sorted(set(protected)|{str(SOURCE),str(Path(C.__file__))}):
        src=Path(path)
        if src.suffix!='.py':continue
        h=C.sha(src);dst=OUT/'scoring_code'/(h[:12]+'__'+src.name)
        data=src.read_bytes()
        if dst.exists():assert dst.read_bytes()==data
        else:atomic(dst,lambda f,data=data:f.write(data),True)
        snapshot[str(src)]=dict(sha256=h,snapshot=str(dst),snapshot_sha256=C.sha(dst))
    reg['actual_scoring_source_files']=snapshot
    write(OUT/'scoring_definition.json',{k:reg[k] for k in ('version','source_sha256','settings','postprocess_config','quality_definition','quality_source_hashes','L','budget','PRE_quality','boundary_domain','required_interface','PRE_topology','valid_edge','reachable_area','aggregation','actual_scoring_source_files')})
    write(target,reg);return reg

def new_input(row):
    den=OUT/'pre_geo'/row['method']/row['request_id']/'denoised.npz'
    rec=C.read(den.with_name('receipt.json'));ident=rec.get('identity',rec)
    assert rec['status']=='PASS' and rec.get('output_sha256',ident.get('output_sha256'))==C.sha(den)
    for k,v in [('method',row['method']),('request_id',row['request_id']),('raw_sha256',row['raw_sha256']),('registration_sha256',C.sha(OUT/'registration.json'))]:assert ident[k]==v,(k,ident.get(k),v)
    a=E.payload(den);raw=E.payload(row['raw_path'])['raw'].reshape(-1,3,3);K=int(row['K'])
    assert C.sha(row['raw_path'])==row['raw_sha256']
    assert a['output'].shape==raw.shape and a['output'].dtype==np.float32 and np.isfinite(a['output']).all()
    assert E.same(a['output'][:K],raw[:K]) and E.same(a['C'],raw[:K])
    return den

def postprocess():
    reg=prepare();_,_,PC,_=E.helpers();rows=list(reg['baselines'])
    for method in NEW:
        for r in reg['records']:
            row=dict(r,method=method,reused=False);den=new_input(row)
            dest=C.checked(OUT/'outputs'/method/r['request_id']);receipt=dest/'postprocess.json'
            if receipt.exists():
                pp=C.read(receipt);assert pp['status']=='PASS' and pp['input']['file_sha256']==C.sha(den)
                for name,v in pp['files'].items():assert C.sha(dest/name)==v['sha256']
            else:
                staging=C.checked(dest.with_name(dest.name+'.staging'))
                if dest.exists() or staging.exists():raise RuntimeError('Incomplete GEO; inspect retained state')
                job_open('GEO',row);pp=PC.run(den,int(r['K']),staging,input_key='output')
                write(staging/'postprocess.json',E.rewrite_paths(pp,staging,dest));os.replace(staging,dest)
                event('GEO_COMPLETE',row,seconds=pp['seconds'])
            post=dest/'post.npz';a=E.payload(post);raw=E.payload(r['raw_path'])['raw'].reshape(-1,3,3)
            assert E.same(a['output'][:int(r['K'])],raw[:int(r['K'])])
            assert E.same(a['vertices'][a['faces']],a['output'])
            v,f=PC.read_indexed_ply(dest/'post.ply');assert E.same(v,a['vertices']) and E.same(f,a['faces'])
            rows.append(dict(row,denoised_path=str(den),pre_GEO_key='output',post_path=str(post),post_ply=str(dest/'post.ply'),
                post_sha256=C.sha(post),output_N=len(a['output']),diagnostic_path=str(OUT/'diagnostics'/method/r['request_id']/'diagnostic.json')))
    assert len(rows)==48;csvwrite(OUT/'post_index.csv',rows)
    write(OUT/'post_completion.json',dict(status='PASS',new_GEO=24,reused_POST=24,outputs=48,C_and_PLY_exact=True))
    return rows

def boundary(row,tri,A):
    source=E.payload(row['condition_source']);K=int(row['K']);assert E.same(source['C'],tri[:K])
    ids,points,audit=A.topology(source['C'],source['C_source_vertex_ids']);ds,near=A.nearest(points,tri[K:])
    verts=[dict(source_vertex_id=int(v),x=float(p[0]),y=float(p[1]),z=float(p[2]),
        distance_model=float(d) if np.isfinite(d) else None,distance_L_fraction=float(d/A.L) if np.isfinite(d) else None,
        nearest_free_face=int(n)//3 if n>=0 else None,nearest_free_corner=int(n)%3 if n>=0 else None)
        for v,p,d,n in zip(ids,points,ds,near)]
    metrics=[dict(domain=DOMAIN,setting=n,threshold_model=t,scale=s,tau=v,**A.metric(ds,t)) for n,t,s,v in A.SETTINGS]
    return verts,metrics,audit

def exact_pre_graph(tri,K,Q,L):
    flat=np.ascontiguousarray(tri,dtype=np.float32).reshape(-1,3)
    _,first,inverse=np.unique(flat.view(np.uint32).reshape(-1,3),axis=0,return_index=True,return_inverse=True)
    vertices=flat[first];faces=inverse.reshape(-1,3);edge_faces=collections.defaultdict(list)
    for i,f in enumerate(faces):
        for u,v in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):edge_faces[tuple(sorted((int(u),int(v))))].append(i)
    pairs={'CF':{'contacts':[]},'FF':{'contacts':[]}};tested=set();geometry=Q._old().cpu_math()['geometry']
    for inc in edge_faces.values():
        if len(inc)!=2 or inc[0]==inc[1] or max(inc)<K:continue
        i,j=sorted(inc)
        if (i,j) in tested:continue
        tested.add((i,j))
        result=geometry.pair_scan(tri[i:i+1],tri[j:j+1],L,independent=True,return_disjoint=True)
        item=result['contacts'][0];key='CF' if i<K else 'FF'
        pair=[i,j-K] if i<K else [i-K,j-K]
        pairs[key]['contacts'].append(dict(item,pair=pair))
    top,edges,reached=E.actual_valid_graph(vertices,faces,K,pairs)
    top.update(topology_domain='EXACT_FP32_COORDINATE_INDEX_NO_WELD',edge_local_tested_pairs=len(tested),full_pair_quality_status='NA_NOT_BUDGETED')
    return top,edges,reached,pairs

def opposite_reach(tri,K,edges):
    adjacent=collections.defaultdict(set)
    for e in edges:
        if e['valid_edge'] and e['winding_opposite']:
            i,j=e['incident_face_ids'];adjacent[i].add(j);adjacent[j].add(i)
    seen=set(range(K));todo=list(seen)
    while todo:
        i=todo.pop()
        for j in adjacent[i]-seen:seen.add(j);todo.append(j)
    reached=sorted(i for i in seen if i>=K)
    area=.5*np.linalg.norm(np.cross(tri[:,1].astype(float)-tri[:,0],tri[:,2].astype(float)-tri[:,0]),axis=-1)
    total=float(area[K:].sum())
    return dict(opposite_winding_reached_free_faces=len(reached),opposite_winding_reached_free_area=float(area[reached].sum()),
        opposite_winding_reached_free_area_fraction=float(area[reached].sum()/total) if total>0 else None)

def score_one(row):
    row=E.canonical_identity(row);diag=Path(row['diagnostic_path']);post=Path(row['post_path']);den=Path(row['denoised_path'])
    ident=dict(**meta(row),post_sha256=C.sha(post),input_sha256=C.sha(den),score_registration_sha256=C.sha(OUT/'score_registration.json'))
    if diag.exists():
        d=C.read(diag);assert d['status']=='PASS' and d['identity']==ident
        verify(d['files']);verify(d['quality']['files']);return str(diag)
    started=time.perf_counter();A,Q,_,_=E.helpers();dest=C.checked(OUT/'scores'/row['method']/row['request_id'])
    if row['method'] not in NEW:
        old=C.read(row['quality_cache_path']);quality=old['quality'];assert old['status']=='PASS'
        assert quality['identity']['processed_sha256']==C.sha(post);verify(quality['files'])
    elif (dest/'score.json').exists():
        quality=C.read(dest/'score.json');assert quality['status']=='PASS' and quality['identity']['processed_sha256']==C.sha(post);verify(quality['files'])
    else:
        staging=C.checked(dest.with_name(dest.name+'.staging'))
        if dest.exists() or staging.exists():raise RuntimeError('Incomplete quality; no implicit replay')
        job_open('QUALITY',row);quality=Q.score_quality(row,post,C.checked(staging/'quality'))
        quality=E.rewrite_paths(quality,staging,dest);write(staging/'score.json',quality);os.replace(staging,dest)
        event('QUALITY_COMPLETE',row,seconds=quality['seconds'])
    a=E.payload(post);pre=E.payload(den)['output'];K=int(row['K'])
    assert E.same(a['output'][:K],pre[:K])
    top,edges,reached=E.actual_valid_graph(a['vertices'],a['faces'],K,C.read(quality['pair_audit_path']))
    top.update(opposite_reach(a['output'],K,edges),topology_domain='ACTUAL_POST_INDEXED')
    pret,preedges,prereached,prepairs=exact_pre_graph(pre,K,Q,A.L);pret.update(opposite_reach(pre,K,preedges))
    vertices,metrics,ba=boundary(row,a['output'],A);pv,pm,pba=boundary(row,pre,A)
    shape=Q._old().cpu_math()['shape'].shape_quality(pre[K:],A.L)
    out=C.checked(diag.parent);csvwrite(out/'valid_edge_details_POST.csv',edges);csvwrite(out/'valid_edge_details_PRE.csv',preedges)
    write(out/'PRE_edge_local_pairs.json',prepairs);write(out/'PRE_shape_per_face.json',shape)
    d=dict(status='PASS',identity=ident,quality=quality,quality_reused=row['method'] not in NEW,topology=top,pre_GEO_topology=pret,
        reached_free_output_face_ids=reached,pre_GEO_reached_free_face_ids=prereached,vertices=vertices,metrics=metrics,boundary_audit=ba,
        pre_GEO_vertices=pv,pre_GEO_metrics=pm,pre_GEO_quality_status='NA_NOT_BUDGETED',
        pre_GEO_shape=dict(free_faces=len(pre)-K,free_total_surface_area=shape['total_area'],free_minimum_angle_lt5_fraction=shape['min_angle_lt5_fraction'],
            free_q_gt10_fraction=shape['q_actual_gt10_fraction'],free_exact_zero_area_count=shape['exact_zero_area_count'],free_near_degenerate_count=shape['near_degenerate_area_count']),
        source_face_ids=a['source_face_ids'].tolist(),postprocess=C.read(post.with_name('postprocess.json')),seconds=time.perf_counter()-started,
        peak_cpu_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,files={str(p):C.sha(p) for p in out.iterdir() if p.is_file()})
    write(diag,d)
    if 'torch' in sys.modules:raise RuntimeError('CPU module unexpectedly imported torch')
    return str(diag)

def numeric(rows):return sorted({k for r in rows for k,v in r.items() if k not in E.META and isinstance(v,(int,float)) and not isinstance(v,bool)})
def summary(rows,base,nums):
    result=[]
    for name,extra in [('ALL',()),('REGION',('region_type',)),('PARENT',('parent_id',)),('NOISE',('candidate',)),('PARENT_NOISE',('parent_id','candidate'))]:
        result.extend(dict(grouping=name,**r) for r in E.summaries(rows,base+extra,nums))
    return result

def aggregate(rows):
    quality=[];top=[];vm=[];vertices=[];distance=[];mods=[];shape=[];prequality=[];edges=[]
    for row in rows:
        row=E.canonical_identity(row);d=C.read(row['diagnostic_path']);m=meta(row);K=int(row['K']);a=E.payload(row['post_path'])
        quality.append(dict(**m,phase='POST',**d['quality']['metrics'],free_face_denominator=d['quality']['free_face_denominator']))
        prequality.append(dict(**m,phase='PRE_GEO',status='NA_NOT_BUDGETED',CF=None,FF=None,I=None))
        shape.append(dict(**m,phase='PRE_GEO',**d['pre_GEO_shape']))
        for phase,tk,mk,vk in [('POST','topology','metrics','vertices'),('PRE_GEO','pre_GEO_topology','pre_GEO_metrics','pre_GEO_vertices')]:
            top.append(dict(**m,phase=phase,**d[tk]));vm.extend(dict(**m,phase=phase,**v) for v in d[mk]);vertices.extend(dict(**m,phase=phase,**v) for v in d[vk])
            ds=[v['distance_model'] for v in d[vk]];good=bool(ds) and all(x is not None for x in ds)
            distance.append(dict(**m,phase=phase,boundary_vertices=len(ds),status='PASS' if good else 'NA_EMPTY_BOUNDARY_OR_FREE',
                distance_mean_model=float(np.mean(ds)) if good else None,distance_median_model=float(np.median(ds)) if good else None,
                distance_p95_model=float(np.quantile(ds,.95)) if good else None,distance_max_model=max(ds) if good else None))
            ep=Path(row['diagnostic_path']).parent/('valid_edge_details_'+('PRE' if phase=='PRE_GEO' else 'POST')+'.csv')
            edges.extend(dict(**m,phase=phase,**v) for v in readcsv(ep))
        raw=E.payload(row['raw_path'])['raw'].reshape(-1,3,3);pre=E.payload(row['denoised_path'])['output']
        base=E.payload(row['D0_denoised_path'])['output'];dr=np.linalg.norm(pre[K:].astype(float)-raw[K:],axis=-1);db=np.linalg.norm(pre[K:].astype(float)-base[K:],axis=-1)
        mods.append(dict(**m,raw_free_faces=len(raw)-K,POST_free_faces=len(a['output'])-K,free_retained_fraction=(len(a['output'])-K)/(len(raw)-K),
            deleted_free_faces=len(raw)-len(a['output']),deleted_free_fraction=(len(raw)-len(a['output']))/(len(raw)-K),
            pre_GEO_free_displacement_RAW_mean=float(dr.mean()),pre_GEO_free_displacement_RAW_p95=float(np.quantile(dr,.95)),pre_GEO_free_displacement_RAW_max=float(dr.max()),
            pre_GEO_free_displacement_D0_mean=float(db.mean()),pre_GEO_free_displacement_D0_p95=float(np.quantile(db,.95)),pre_GEO_free_displacement_D0_max=float(db.max()),
            GEO_free_corner_displacement_mean=float(a['corner_displacement'][K:].mean()),GEO_free_corner_displacement_max=float(a['corner_displacement'][K:].max())))
    allrows={}
    for r in quality+top+distance+mods:
        key=(r['method'],r['request_id']);allrows.setdefault(key,dict(meta(r)))
        prefix=('PRE_' if r.get('phase')=='PRE_GEO' else 'POST_' if r.get('phase')=='POST' else '')
        allrows[key].update({prefix+k:v for k,v in r.items() if k not in (*E.META,'method','phase')})
    paired=[]
    for left,right in [('RBF','D0_BASE'),('FEEDBACK','D0_BASE'),('FEEDBACK','RBF'),('D3','D0_BASE')]:
        for rid in sorted({r['request_id'] for r in rows}):
            x,y=allrows[left,rid],allrows[right,rid];p=dict(meta(x),comparison=left+'_minus_'+right)
            for k in numeric([x,y]):p[k+'_delta']=float(x[k])-float(y[k]) if isinstance(x.get(k),(int,float)) and isinstance(y.get(k),(int,float)) else None
            paired.append(p)
    vpairs=[];vv={(r['method'],r['request_id'],r['phase'],r['setting']):r for r in vm}
    for left,right in [('RBF','D0_BASE'),('FEEDBACK','D0_BASE'),('FEEDBACK','RBF'),('D3','D0_BASE')]:
        for x in [r for r in vm if r['method']==left]:
            y=vv[right,x['request_id'],x['phase'],x['setting']]
            vpairs.append(dict(**meta(x),phase=x['phase'],setting=x['setting'],comparison=left+'_minus_'+right,
                A_VMR_delta=float(x['VMR'])-float(y['VMR']) if x['VMR'] is not None and y['VMR'] is not None else None,
                PMR_delta=float(x['perfect'])-float(y['perfect']) if x['perfect'] is not None and y['perfect'] is not None else None))
    geovm=[]
    for x in [r for r in vm if r['phase']=='POST']:
        y=vv[x['method'],x['request_id'],'PRE_GEO',x['setting']]
        geovm.append(dict(**meta(x),setting=x['setting'],threshold_model=x['threshold_model'],
            A_VMR_POST_minus_PRE=float(x['VMR'])-float(y['VMR']) if x['VMR'] is not None and y['VMR'] is not None else None,
            PMR_POST_minus_PRE=float(x['perfect'])-float(y['perfect']) if x['perfect'] is not None and y['perfect'] is not None else None))
    geo=[]
    for method,rid in allrows:
        r=allrows[method,rid];v=dict(meta(r))
        for k in list(r):
            if k.startswith('POST_') and 'PRE_'+k[5:] in r:
                x,y=r[k],r['PRE_'+k[5:]]
                if isinstance(x,(int,float)) and isinstance(y,(int,float)):v[k[5:]+'_POST_minus_PRE']=float(x)-float(y)
        geo.append(v)
    for name,table,base in [('quality_per_output',quality,('method','phase')),('topology_per_output',top,('method','phase')),('boundary_distance_per_output',distance,('method','phase')),
                            ('modifications_per_output',mods,('method',)),('pre_GEO_shape_per_output',shape,('method','phase')),('paired_per_output',paired,('comparison',)),('GEO_changes_per_output',geo,('method',))]:
        csvwrite(OUT/(name+'.csv'),table);csvwrite(OUT/(name.replace('per_output','summary')+'.csv'),summary(table,base,numeric(table)))
    csvwrite(OUT/'per_output.csv',vm);csvwrite(OUT/'AVMR_PMR_summary.csv',summary(vm,('method','phase','setting'),['VMR','perfect','matched_count','boundary_vertex_count']))
    csvwrite(OUT/'per_boundary_vertex.csv',vertices);csvwrite(OUT/'actual_edge_details.csv',edges);csvwrite(OUT/'PRE_full_quality_status.csv',prequality)
    csvwrite(OUT/'AVMR_paired_per_output.csv',vpairs);csvwrite(OUT/'AVMR_paired_summary.csv',summary(vpairs,('comparison','phase','setting'),['A_VMR_delta','PMR_delta']))
    csvwrite(OUT/'GEO_AVMR_changes_per_output.csv',geovm);csvwrite(OUT/'GEO_AVMR_changes_summary.csv',summary(geovm,('method','setting'),['A_VMR_POST_minus_PRE','PMR_POST_minus_PRE']))
    csvwrite(OUT/'combined_per_output.csv',list(allrows.values()))
    write(OUT/'score_aggregate_completion.json',dict(status='PASS',POST_quality_rows=len(quality),PRE_POST_topology_rows=len(top),AVMR_rows=len(vm),boundary_rows=len(vertices),paired_rows=len(paired),model_calls=0,new_queries=0,full_PRE_quality='NA_NOT_BUDGETED'))

def score():
    reg=prepare();start=time.perf_counter();C.event('CPU_EVALUATION_STARTED',score_registration_sha256=C.sha(OUT/'score_registration.json'));rows=postprocess()
    write(OUT/'resources_before_score.json',C.resources())
    new=[r for r in rows if r['method'] in NEW]
    tick=time.perf_counter();first=score_one(new[0]);elapsed=time.perf_counter()-tick
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    workers=2 if (os.cpu_count() or 1)>=4 and available>max(4<<30,3*peak) else 1
    write(OUT/'CPU_dispatch.json',dict(workers=workers,threads_per_worker=2,first_authorized_item=first,first_seconds=elapsed,process_peak_RSS_bytes=peak,available_RAM_bytes=available,reason='First authorized POST job measured; bounded two CPU workers if RAM/core headroom permits',GPU_workers=0))
    remaining=[r for r in rows if (r['method'],r['request_id'])!=(new[0]['method'],new[0]['request_id'])]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:receipts=[first,*pool.map(score_one,remaining,chunksize=1)]
    verify(reg['protected'])
    for kind in ('GEO','QUALITY'):
        for suffix in ('STARTED','COMPLETE'):assert len(list((OUT/'score_events').glob('*/*/'+kind+'_'+suffix+'.json')))==24
    aggregate(rows)
    write(OUT/'score_completion.json',dict(status='PASS',new_GEO=24,new_POST_quality=24,reused_POST_quality=24,outputs=48,receipts=receipts,workers=workers,seconds=time.perf_counter()-start,new_model_calls=0,new_queries=0))
    C.event('CPU_EVALUATION_COMPLETE',score_completion_sha256=C.sha(OUT/'score_completion.json'),outputs=48,new_GEO=24,new_POST_quality=24)
    return C.read(OUT/'score_completion.json')

def render():
    assert C.read(OUT/'score_completion.json')['status']=='PASS'
    C.event('CPU_RENDER_STARTED',source_sha256=C.sha(SOURCE))
    G=E.module('_c_restore_inherited_renderer',C.ROOT/'r2/tools/student_multistep_denoise_v1/gallery.py')
    import matplotlib.pyplot as plt
    from PIL import Image
    start=time.perf_counter();rows=[E.canonical_identity(r) for r in readcsv(OUT/'post_index.csv')]
    registration=prepare();gdir=C.checked(OUT/'gallery');gdir.mkdir(parents=True,exist_ok=True)
    source_sha=C.sha(SOURCE);definition={str(OUT/'registration.json'):C.sha(OUT/'registration.json'),str(OUT/'post_index.csv'):C.sha(OUT/'post_index.csv')}
    receipts=[];pending=[]
    for r in registration['records']:
        rid=r['request_id'];K=int(r['K']);group={x['method']:x for x in rows if x['request_id']==rid};assert set(group)==set(METHODS)
        meshes={m:G.load_mesh(group[m]['post_path'],K,'POST') for m in METHODS}
        raw=E.payload(r['raw_path'])['raw'].reshape(-1,3,3);assert all(E.same(a[:K],raw[:K]) for a in meshes.values())
        xyz=np.concatenate([a.reshape(-1,3) for a in meshes.values()]).astype(float);lo,hi=xyz.min(0),xyz.max(0)
        bounds=dict(center_model_xyz=((lo+hi)/2).tolist(),span=float((hi-lo).max())*1.12,bounds_model_xyz=[lo.tolist(),hi.tolist()])
        source_hashes={group[m]['post_path']:C.sha(group[m]['post_path']) for m in METHODS}
        for vi,camera in enumerate(((22.,-60.),(22.,120.))):
            png=gdir/(rid+'__v'+str(vi)+'.png');jp=png.with_suffix('.jpg');rp=png.with_suffix('.json')
            identity=dict(request_id=rid,parent_id=r['parent_id'],condition_id=r['condition_id'],region_type=r['region_type'],label=r['label'],K=K,
                methods=METHODS,N_by_method={m:len(meshes[m]) for m in METHODS},camera_elev_azim=list(camera),view_id=vi,bounds=bounds,
                source_hashes=source_hashes,definition_hashes=definition,source_code_sha256=source_sha,pixels=[2400,1600],C_bitwise_unchanged_all_stages=True,topology_claim_from_picture=False)
            # JSON roundtrip normalizes method tuples before identity comparison.
            identity=json.loads(json.dumps(identity))
            if rp.exists():
                rec=C.read(rp);assert rec['status']=='COMPLETE' and rec['identity']==identity
                assert C.sha(png)==rec['sha256'] and C.sha(jp)==rec['jpg_sha256']
            else:
                fig=plt.figure(figsize=(16,10.6666666667),dpi=150)
                for i,m in enumerate(METHODS):
                    ax=fig.add_subplot(2,2,i+1,projection='3d');G.draw(ax,meshes[m],K,bounds,camera)
                    ax.set_title(m+' | POST N='+str(len(meshes[m]))+' C='+str(K)+' F='+str(len(meshes[m])-K),fontsize=13)
                fig.suptitle('R2_USER_MIX | '+r['parent_id']+' | '+r['region_type']+' '+r['tier']+' | n'+str(r['candidate'])+'\nC restoration coupling | fixed camera '+str(vi+1),fontsize=15)
                fig.text(.5,.018,'BLUE = unchanged C | GRAY = free | Same camera/bounds | Pictures do not certify attachment',ha='center',fontsize=11)
                fig.subplots_adjust(left=.01,right=.99,top=.91,bottom=.04,wspace=.0,hspace=.02)
                atomic(png,lambda f:fig.savefig(f,format='png',dpi=150),True);plt.close(fig)
                with Image.open(png) as im:atomic(jp,lambda f:im.convert('RGB').save(f,format='JPEG',quality=85,optimize=True),True)
                rec=dict(status='COMPLETE',identity=identity,image=str(png),png=str(png),sha256=C.sha(png),jpg=str(jp),jpg_sha256=C.sha(jp));write(rp,rec)
            receipts.append(rec)
        for m in METHODS:
            pending.append(dict(request_id=rid,parent_id=r['parent_id'],region_type=r['region_type'],method=m,review_kind='AI_PROXY',review_status='PENDING',blind=False,
                overall_shape='NOT_REVIEWED',C_use='NOT_REVIEWED',boundary_continuation='NOT_REVIEWED',thin_structure='NOT_REVIEWED',
                change_vs_D0='NOT_REVIEWED',uncertainty='NOT_REVIEWED',observations='NOT_REVIEWED',figure_v0=rid+'__v0.png',figure_v1=rid+'__v1.png'))
    assert len(receipts)==24 and len(pending)==48
    if not (gdir/'review_pending.csv').exists():csvwrite(gdir/'review_pending.csv',pending)
    cards=[]
    for r in registration['records']:
        rid=r['request_id'];cards.append('<section><h2>'+html.escape(rid)+'</h2>'+''.join('<a href="'+rid+'__v'+str(v)+'.png"><img style="width:49%" src="'+rid+'__v'+str(v)+'.jpg"></a>' for v in range(2))+'</section>')
    page='<!doctype html><meta charset="utf-8"><title>C restoration coupling</title><h1>All12 fixed USER_MIX requests</h1><p>Nonblind AI_PROXY pending. Blue C unchanged. No image proves shared topology.</p>'+''.join(cards)
    atomic(gdir/'index.html',lambda f:f.write(page))
    write(gdir/'gallery_manifest.json',dict(status='COMPLETE',requests=12,PNG_count=24,JPEG_count=24,POST_outputs=48,methods=METHODS,source_code_sha256=source_sha,
        definition_hashes=definition,receipts=receipts,wall_seconds=time.perf_counter()-start,model_calls=0,new_scores=0,visual_review_status='PENDING'))
    C.event('CPU_RENDER_COMPLETE',gallery_manifest_sha256=C.sha(gdir/'gallery_manifest.json'),PNG_count=24)
    return C.read(gdir/'gallery_manifest.json')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','postprocess','score','render','all'],required=True)
    a=parser.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='postprocess':postprocess()
    elif a.stage=='score':score()
    elif a.stage=='render':render()
    else:score();render()

if __name__=='__main__':main()
