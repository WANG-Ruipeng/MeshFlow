"""CPU, read-only Bench relation-coverage investigation. No model or OT calls.

Commands: analyze / render / verify. All writes are confined to a new audit.
The descriptors and nearest-neighbour ranking are exploratory data summaries,
not generation metrics, semantic truth, or a proof of insufficient coverage.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv, hashlib, io, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_relation_coverage_v1'
DATA=ROOT/'experiments/r2_bench_user_data_v2_expansion_v1'
TRAIN=ROOT/'experiments/r2_bench_user_v2_training_v1'
L=float(np.sqrt(3)*2/.3762)
SOURCES={}

def path(p):
    p=str(p).replace('\\','/')
    if p.startswith('E:/'): p='/mnt/e/'+p[3:]
    q=Path(p)
    return q if q.is_absolute() else ROOT/q
def sha(p): return hashlib.sha256(path(p).read_bytes()).hexdigest()
def track(p, expected=None):
    p=path(p);h=sha(p)
    if expected: assert h==expected, str(p)
    SOURCES[str(p)]=h
    return p
def read(p, expected=None): return json.loads(track(p,expected).read_text('utf-8-sig'))
def arr(p, expected=None):
    with np.load(track(p,expected),allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,Path):return str(x)
    return x
def put(p, b):
    p=path(p);assert p.resolve().is_relative_to(OUT.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==b,'Immutable output differs: '+str(p)
    else:
        with p.open('xb') as f:f.write(b)
def save(p,x):put(p,(json.dumps(clean(x),ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode())
def csvout(name,rows):
    s=io.StringIO(newline='');fields=list(dict.fromkeys(k for r in rows for k in r))
    w=csv.DictWriter(s,fieldnames=fields);w.writeheader()
    w.writerows([{k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple,np.ndarray)) else v for k,v in r.items()} for r in rows])
    put(OUT/name,('\ufeff'+s.getvalue()).encode())
def event(stage,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(clean(dict(time_unix=time.time(),stage=stage,**kw)),ensure_ascii=False)+'\n')
def resources():
    def cmd(args):
        r=subprocess.run(args,capture_output=True,text=True);return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    return dict(cpu_logical=psutil.cpu_count(),cpu_physical_reported_by_WSL=psutil.cpu_count(logical=False),
        available_RAM=psutil.virtual_memory().available,total_RAM=psutil.virtual_memory().total,load=psutil.getloadavg(),
        gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
        gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv']),
        python_processes=[dict(pid=p.pid,name=p.info['name'],cmd=p.info['cmdline']) for p in psutil.process_iter(['name','cmdline'])
                          if (p.info['name'] or '').startswith('python')])

def moments(t):
    t=np.asarray(t,np.float64)
    cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);a=np.linalg.norm(cross,axis=1)/2
    mass=float(a.sum());assert mass>0
    cent=t.mean(1);c=(cent*a[:,None]).sum(0)/mass
    ss=t.sum(1)
    second=(np.einsum('fvi,fvj->fij',t,t)+np.einsum('fi,fj->fij',ss,ss))/12
    cov=(second*a[:,None,None]).sum(0)/mass-np.outer(c,c)
    eig=np.maximum(np.linalg.eigvalsh(cov),0)
    n=np.divide(cross,2*a[:,None],out=np.zeros_like(cross),where=a[:,None]>0)
    nt=(np.einsum('fi,fj->fij',n,n)*a[:,None,None]).sum(0)/mass
    return dict(area=mass,centroid=c,extent=np.ptp(t.reshape(-1,3),axis=0),eig=eig,normal_tensor=nt,
                plane_rms=float(np.sqrt(eig[0])),horizontal_normal_area=float((a*np.abs(n[:,1])).sum()/mass)),a,cent

def topology(f):
    edges=defaultdict(list)
    for i,t in enumerate(f):
        for j in range(3):edges[tuple(sorted((int(t[j]),int(t[(j+1)%3]))))].append(i)
    parent=list(range(len(f)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for ids in edges.values():
        for i in ids[1:]:parent[find(i)]=find(ids[0])
    return dict(C_edge_components=len({find(i) for i in range(len(f))}),C_boundary_edges=sum(len(v)==1 for v in edges.values()),
                C_nonmanifold_edges=sum(len(v)>2 for v in edges.values()))

def descriptor(r,split):
    a=arr(r['case_path'],r['case_sha256']);c=np.asarray(a['C'],float);y=np.asarray(a['Y'],float);m=np.concatenate([c,y])
    assert len(c)==r['K'] and len(m)==r['N']
    cm,ca,cc=moments(c);mm,ma,mc=moments(m);ym,ya,yc=moments(y)
    lo=m.reshape(-1,3).min(0);hi=m.reshape(-1,3).max(0)
    d=dict(condition_id=r['condition_id'],parent_id=r['parent_id'],split=split,part_type=r['part_type'],
           extent_description=r.get('extent_description',r.get('review',{}).get('extent_description','UNKNOWN')),
           K=len(c),N=len(m),C_area=cm['area'],M_area=mm['area'],C_area_fraction=cm['area']/mm['area'],
           C_plane_rms_L=cm['plane_rms']/L,C_horizontal_normal_area=cm['horizontal_normal_area'],
           C_centroid_Y_relative_M_bbox=float((cm['centroid'][1]-lo[1])/(hi[1]-lo[1])),
           C_Y_centroid_distance_L=float(np.linalg.norm(cm['centroid']-ym['centroid'])/L),
           C_source_owner_count=int(a['occupied_labels'].sum()),C_source_alias_count=int(a['source_alias_labels'].sum()),
           case_path=r['case_path'],case_sha256=r['case_sha256'],source_relation_policy='Source M is one reference completion, not unique desired output',
           **topology(a['C_faces_original']))
    for j,ax in enumerate('xyz'):
        d['C_centroid_'+ax]=float(cm['centroid'][j]);d['C_extent_'+ax]=float(cm['extent'][j])
        d['C_M_extent_ratio_'+ax]=float(cm['extent'][j]/mm['extent'][j]) if mm['extent'][j]>0 else None
    for i in range(3):
        for j in range(3):d[f'C_normal_tensor_{i}{j}']=float(cm['normal_tensor'][i,j])
    # Triangle-centroid area distribution: explicitly not exact clipped halfspace area.
    if cm['horizontal_normal_area']>=.99 and cm['plane_rms']/L<=.005:
        dy=yc[:,1]-cm['centroid'][1]
        d['Y_centroid_area_below_C_1pctL']=float(ya[dy<-.01*L].sum()/ya.sum())
        d['Y_centroid_area_above_C_1pctL']=float(ya[dy>.01*L].sum()/ya.sum())
    else:
        d['Y_centroid_area_below_C_1pctL']=None;d['Y_centroid_area_above_C_1pctL']=None
    return d

def difference(a,b):
    center=sum((a['C_centroid_'+ax]-b['C_centroid_'+ax])**2 for ax in 'xyz')/L**2
    extent=sum((a['C_extent_'+ax]-b['C_extent_'+ax])**2 for ax in 'xyz')/L**2
    area=(np.sqrt(a['C_area'])-np.sqrt(b['C_area']))**2/L**2
    norm=sum((a[f'C_normal_tensor_{i}{j}']-b[f'C_normal_tensor_{i}{j}'])**2 for i in range(3) for j in range(3))
    return dict(C_descriptor_distance=float(np.sqrt(center+extent+area+.01*norm)),
                centroid_distance_L=float(np.sqrt(center)),extent_distance_L=float(np.sqrt(extent)),
                sqrt_area_difference_L=float(np.sqrt(area)),normal_tensor_distance=float(np.sqrt(norm)))

def analyze():
    start=time.monotonic();res=resources();event('CPU_ANALYSIS_START',resources=res)
    if (OUT/'analysis_complete.json').exists():verify();return
    protocol=dict(version='BENCH_RELATION_COVERAGE_EXPLORATORY_V1',scope='81 TRAIN +39 DEV saved conditions; 184 existing parent admission records; actual 1500-update exposure logs',
        model_calls=0,training_updates=0,new_noise=0,new_OT=0,old_asset_writes=0,L=L,
        descriptor='Exact triangle surface moments, axis extents, unsigned area normal tensor, source vertex-ID edge components; source Y height uses triangle-centroid area approximation',
        nearest='C-only sqrt(center_distance^2/L^2+extent_distance^2/L^2+sqrt_area_difference^2/L^2+0.01*normal_tensor_Frobenius^2); fixed exploratory screen, not learned/validated semantic metric',
        comparisons='All 39 DEV against 81 TRAIN; exclude same parent for four familiar diagnostic C; report all-C and same-curation-part rankings separately; top3 distinct parents for review',
        no_target_in_ranking=True,selection='Eight pre-existing diagnostic conditions, not selected by new descriptor values',
        source_M_usage='Curation/relation illustration only; alternative coherent completion remains valid',
        dispatch='1 CPU metadata/array worker; 2 independent CPU render workers, one numeric thread each; 0 GPU workers',
        resources=res)
    if (OUT/'protocol.json').exists():
        old=json.loads((OUT/'protocol.json').read_text())
        assert {k:v for k,v in old.items() if k!='resources'}=={k:v for k,v in protocol.items() if k!='resources'}
        event('CPU_RESUME',reason='Initial metadata pass encountered a null historical review field; guarded as UNKNOWN. Partial immutable tables preserved; no scientific definition change.')
    else:save(OUT/'protocol.json',protocol)
    ready=read(DATA/'DATASET_READY.json');m=read(DATA/'manifest.json',ready['manifest_sha256'])
    reg=read(TRAIN/'registration.json');read(reg['manifest'],reg['manifest_sha256'])
    schedule=read(reg['input_schedule'],reg['input_schedule_sha256'])
    diag=read('experiments/r2_bench_trajectory_crossover_v1/registration.json')
    labels_path=track('experiments/r2_bench_early_fm_v1/analysis/adoption_paired.csv')
    labels=list(csv.DictReader(labels_path.open(encoding='utf-8-sig')))
    case_rows=m['train']+m['dev'];idx={r['condition_id']:r for r in case_rows};assert len(idx)==120
    rows=[descriptor(r,'TRAIN') for r in m['train']]+[descriptor(r,'DEV') for r in m['dev']]
    byid={r['condition_id']:r for r in rows};assert len(rows)==120
    # Actual completed forwards, matched to the frozen schedule (not expected exposures alone).
    exposure=track(TRAIN/'training/USER_V2_SEAM/exposure.jsonl')
    logs=[json.loads(s) for s in exposure.read_text().splitlines()]
    logp=track(TRAIN/'training/USER_V2_SEAM/training_log.jsonl');updates=[json.loads(s)['local_step'] for s in logp.read_text().splitlines()]
    assert updates==list(range(1,1501)) and len(logs)==12000
    expected={(s['step'],r['sample_index']):r for s in schedule['steps'] for r in s['samples']}
    seen=set();counts=Counter();parts=defaultdict(lambda:Counter());scalar=Counter();aug=defaultdict(list)
    for r in logs:
        key=(r['local_step'],r['sample_index']);assert key not in seen;seen.add(key)
        e=expected[key];assert r['condition_id']==e['condition_id'] and r['scale_xyz']==e['scale_xyz']
        counts[r['condition_id']]+=1;scalar[r['condition_id']]+=r['all_free_scalar_count'];aug[r['condition_id']].append(r['scale_xyz'])
        parts[r['part_type']]['samples']+=1;parts[r['part_type']]['free_scalars']+=r['all_free_scalar_count']
    assert len(seen)==len(expected)
    for r in rows:
        cid=r['condition_id'];r['actual_training_samples']=counts[cid];r['actual_free_scalar_exposure']=scalar[cid]
        if cid in aug:
            a=np.array(aug[cid]);r['scale_xyz_min']=a.min(0).tolist();r['scale_xyz_max']=a.max(0).tolist()
        else:r['scale_xyz_min']=None;r['scale_xyz_max']=None
    csvout('condition_descriptors.csv',rows)
    grouped=[]
    for part,counts0 in sorted(parts.items()):
        rr=[r for r in rows if r['split']=='TRAIN' and r['part_type']==part]
        grouped.append(dict(part=part,parents=len({r['parent_id'] for r in rr}),conditions=len(rr),**counts0,
                     sample_fraction=counts0['samples']/len(logs),scalar_fraction=counts0['free_scalars']/sum(scalar.values())))
    csvout('part_exposure.csv',grouped)
    parent_samples=Counter()
    for r in rows:parent_samples[r['parent_id']]+=r['actual_training_samples']
    csvout('parent_exposure.csv',[dict(parent_id=p,samples=n) for p,n in sorted(parent_samples.items()) if n])
    anchors=list(dict.fromkeys(q['condition_id'] for q in diag['requests']))
    queryids=sorted(set(anchors)|{r['condition_id'] for r in m['dev']})
    nearest=[];top={}
    for cid in queryids:
        q=byid[cid]
        for domain in ('ALL_C','SAME_CURATED_PART'):
            pool=[r for r in rows if r['split']=='TRAIN' and r['parent_id']!=q['parent_id'] and (domain=='ALL_C' or r['part_type']==q['part_type'])]
            scored=sorted([(difference(q,r),r) for r in pool],key=lambda t:(t[0]['C_descriptor_distance'],t[1]['condition_id']))
            visited=set();rank=0
            for dif,r in scored:
                if r['parent_id'] in visited:continue
                visited.add(r['parent_id']);rank+=1
                item=dict(query_condition=cid,query_parent=q['parent_id'],query_split=q['split'],query_part=q['part_type'],domain=domain,rank=rank,
                          neighbor_condition=r['condition_id'],neighbor_parent=r['parent_id'],neighbor_part=r['part_type'],**dif)
                nearest.append(item)
                if cid in anchors and domain=='ALL_C' and rank<=3:top.setdefault(cid,[]).append(r['condition_id'])
    csvout('C_neighbor_ranking.csv',nearest)
    anchorrow=[]
    for cid in anchors:
        rr=[r for r in labels if r['request_id'].rsplit('__n',1)[0]==cid]
        anchorrow.append(dict(byid[cid],RAW_labels=[r['USER_RAW'] for r in rr],FINAL_labels=[r['USER_FINAL'] for r in rr],neighbors=top[cid]))
    csvout('diagnostic_conditions.csv',anchorrow)
    decisions=read(DATA/'all_admission_decisions.json');pp=decisions['parents'];cc=decisions['conditions']
    accepted={p['parent_id'] for p in m['parents']}
    decisionrows=[]
    for p in pp:
        decisionrows.append(dict(parent_id=p['parent_id'],accepted=p['parent_id'] in accepted,N_original=p.get('N_original'),N_target=p.get('N'),
             reasons=p.get('reasons'),illegal_pairs=p.get('illegal_pairs'),unresolved_pairs=p.get('unresolved_pairs'),
             status=p.get('status'),eligibility=p.get('eligibility'),review_status=p.get('review_status'),
             review_notes=(p.get('review') or {}).get('notes'),target_mesh_path=p.get('target_mesh_path'),target_case_path=p.get('target_case_path')))
    csvout('parent_admission_audit.csv',decisionrows)
    held=[r for r in cc if r.get('review_status') not in ('ACCEPT',) or r['parent_id'] not in accepted]
    csvout('held_condition_records.csv',[dict(parent_id=r['parent_id'],condition_id=r['condition_id'],part_type=r['part_type'],
            decision=r.get('review_status'),extent=r.get('extent_description'),notes=(r.get('review') or {}).get('notes'),
            admission_reasons=r.get('admission_reasons'),quality_file=r.get('quality_file')) for r in held])
    parentidx={r['parent_id']:r for r in pp}
    special_prefixes=('9e429b','943dde','df6fd3','8967e6','dbd069','f35697')
    special=[p for p in pp if any(p['parent_id'].startswith(k) for k in special_prefixes)]
    for r in held:
        if any(r['parent_id'].startswith(k) for k in special_prefixes) and r.get('quality_file'):
            read(r['quality_file'],r.get('quality_sha256'))
    # Track/render only accepted25 plus explicitly relevant existing held whole targets.
    parent_render=list(m['parents'])+[p for p in special if p['parent_id'] not in accepted]
    for p in parent_render:
        track(p['target_mesh_path'],p.get('target_mesh_sha256'))
    reasons=Counter(x for p in pp for x in p.get('reasons',[]))
    summary=dict(conditions=len(rows),train_parents=len(m['parents']),train_conditions=len(m['train']),dev_conditions=len(m['dev']),
        actual_samples=len(logs),updates=len(updates),parent_exposure_min=min(n for n in parent_samples.values() if n),
        parent_exposure_max=max(parent_samples.values()),part_exposure=grouped,admission_proposals=len(pp),
        geometry_reason_counts=dict(reasons),geometry_pass=sum(not p.get('reasons') for p in pp),
        visual_only_held=[p['parent_id'] for p in pp if not p.get('reasons') and p['parent_id'] not in accepted],
        augmentation='CLEAN and anisotropic XYZ[.92,1.08], origin-fixed and jointly applied to C/Y; no translation or independent new relations',
        source_families='UNKNOWN: source-ID uniqueness is not style-family independence',
        held_condition_decisions=dict(Counter(r.get('review_status','UNKNOWN') for r in held)))
    save(OUT/'summary.json',summary)
    save(OUT/'render_plan.json',dict(anchors=anchors,neighbors=top,conditions=idx,parents=parent_render,
         special_conditions=[r for r in cc if any(r['parent_id'].startswith(k) for k in special_prefixes)],
         views=[[-60,22],[120,22]],image_status='NOT_REVIEWED'))
    assert 'torch' not in sys.modules
    save(OUT/'input_hashes.json',SOURCES)
    save(OUT/'analysis_complete.json',dict(wall_seconds=time.monotonic()-start,source_files=len(SOURCES),rows=len(rows),model_calls=0))
    event('CPU_ANALYSIS_COMPLETE',wall_seconds=time.monotonic()-start,resources=resources())
    print(json.dumps(clean(summary),ensure_ascii=False))

def render_one(job):
    from PIL import Image,ImageDraw
    from r2.data_tools.bench_user_v2.render import cell,basis,font,shared_bounds,text_lines
    kind,name,entries=job
    cameras=[basis(-60,22),basis(120,22)];size=350
    if kind=='parent_grid':
        width=size*5;rows=(len(entries)+4)//5
        page=Image.new('RGB',(width,rows*(size+60)+50),'white');draw=ImageDraw.Draw(page)
        draw.text((12,10),name+' | source targets, not generated outputs',font=font(20),fill='black')
        for i,p in enumerate(entries):
            a=arr(p['target_mesh_path'],p.get('target_mesh_sha256'));t=a['vertices'][a['faces']]
            cen,span=shared_bounds([t],cameras);x=(i%5)*size;y=50+(i//5)*(size+60)
            draw.text((x+8,y),p['parent_id'][:8]+' N='+str(len(t)),font=font(17),fill='black')
            im,_=cell(t,0,cameras[0],cen,span,size);page.paste(im,(x,y+30))
        frame='Each whole target fit independently; not for absolute size comparison'
    else:
        meshes=[]
        for r in entries:
            a=arr(r['case_path'],r['case_sha256']);meshes.append((a['C'],a['Y']))
        cen,span=shared_bounds([np.concatenate(t) for t in meshes],cameras)
        page=Image.new('RGB',(size*4,80+len(entries)*(size+100)),'white');draw=ImageDraw.Draw(page)
        draw.text((12,8),name+' | blue C / gray source Y; one reference completion',font=font(20),fill='black')
        draw.text((12,38),'Columns: M=C+Y (view A) | C only (view A) | M=C+Y (view B) | C only (view B)',font=font(17),fill='black')
        for i,(r,(c,y)) in enumerate(zip(entries,meshes)):
            yy=80+i*(size+100)
            label=('QUERY' if i==0 and kind=='neighbors' else 'TRAIN' if kind=='neighbors' else str(r.get('review_status','UNKNOWN')))
            draw.text((12,yy),label+' '+r['condition_id']+' | '+r['part_type'],font=font(16),fill='black')
            text_lines(draw,(12,yy+25),r.get('extent_description','UNKNOWN'),size*4-30,16,2)
            for j,(tri,k,cam) in enumerate([(np.concatenate([c,y]),len(c),cameras[0]),(c,len(c),cameras[0]),
                                           (np.concatenate([c,y]),len(c),cameras[1]),(c,len(c),cameras[1])]):
                im,_=cell(tri,k,cam,cen,span,size);page.paste(im,(size*j,yy+80))
        frame=dict(center=cen.tolist(),span=span,contract='Shared original model frame across entire page; no C-only recentering')
    b=io.BytesIO();page.save(b,format='PNG');p=OUT/'gallery'/f'{name}.png';put(p,b.getvalue())
    assert 'torch' not in sys.modules
    return dict(path=str(p),sha256=sha(p),entries=[r['parent_id'] if kind=='parent_grid' else r['condition_id'] for r in entries],
                kind=kind,frame=frame,review_status='NOT_REVIEWED',size=list(page.size))

def render():
    start=time.monotonic();r=read(OUT/'render_plan.json');res=resources()
    assert res['available_RAM']>2*1024**3
    event('RENDER_DISPATCH',workers=2,numeric_threads=1,resources=res)
    jobs=[('parent_grid','accepted_targets_25',[p for p in r['parents'] if p.get('eligibility')=='ACCEPTED_TARGET'])]
    for cid in r['anchors']:
        entries=[r['conditions'][x] for x in [cid]+r['neighbors'][cid]]
        jobs.append(('neighbors','neighbors_'+cid[:6],entries))
    for prefix in ('9e429b','943dde','df6fd3','8967e6','dbd069','f35697'):
        es=[v for v in r['special_conditions'] if v['parent_id'].startswith(prefix)]
        if es:jobs.append(('held','candidate_'+prefix,es))
    with ProcessPoolExecutor(max_workers=2) as ex:pages=list(ex.map(render_one,jobs))
    save(OUT/'gallery_manifest.json',dict(pages=pages))
    html='<meta charset="utf-8"><title>Bench relation coverage</title><h1>Bench data relation audit</h1><p>Blue C / gray source Y. Exploratory; source M is not unique GT. See separate visual_review.json.</p>'
    for p in pages:
        name=Path(p['path']).name;html+=f'<h2>{name}</h2><a href="gallery/{name}"><img src="gallery/{name}" style="max-width:1400px;width:100%"></a>'
    put(OUT/'index.html',html.encode())
    event('RENDER_COMPLETE',pages=len(pages),workers=2,wall_seconds=time.monotonic()-start,resources=resources())
    print(json.dumps(dict(pages=len(pages),wall_seconds=time.monotonic()-start)))

def verify():
    sources=json.loads((OUT/'input_hashes.json').read_text());bad=[p for p,h in sources.items() if sha(p)!=h]
    if (OUT/'gallery_manifest.json').exists():
        g=json.loads((OUT/'gallery_manifest.json').read_text());bad +=[p['path'] for p in g['pages'] if sha(p['path'])!=p['sha256']]
    for name in ('source_hashes.json','delivery_hashes.json'):
        if (OUT/name).exists():
            hashes=json.loads((OUT/name).read_text());bad +=[p for p,h in hashes.items() if sha(p)!=h]
    assert not bad,bad
    event('VERIFY_PASS',source_files=len(sources),model_calls=0);print('PASS: unchanged source hashes',len(sources))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['analyze','render','verify']);args=parser.parse_args()
    try:globals()[args.stage]()
    except Exception as exc:
        event('CPU_STAGE_FAILED',requested_stage=args.stage,error=repr(exc),model_calls=0)
        raise
