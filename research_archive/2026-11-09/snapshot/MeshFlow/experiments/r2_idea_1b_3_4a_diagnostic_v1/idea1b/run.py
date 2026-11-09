"""Registered single-start CPU affine alignment and unchanged B2 decoder."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,io,json,sys,time,traceback
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import minimize
from r2.tools.partial_edge_binding_v1 import assets as A,engine as E,metrics as Q
KEYS=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','free_q_gt10_fraction')
L=E.L
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,set)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(clean(obj),indent=2,sort_keys=True,ensure_ascii=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf8')!=data:raise ValueError('No overwrite: '+str(path))
    else:path.write_text(data,encoding='utf8')
def table(path,rows):
    path=Path(path);rows=clean(rows)
    if path.exists():raise ValueError('No overwrite: '+str(path))
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def event(name,**kw):
    with (OUT/'execution.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(clean(dict(event=name,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)))+'\n')
def rows():
    return A.read(A.OUT/'input_manifest.json')['rows']
def freeze():
    p=OUT/'registration.json'
    if p.exists():return A.read(p)
    rr=rows();protected={k:v for r in rr for k,v in r['protected_files'].items()}
    protected.update(Q.definition_hashes())
    for p0 in (A.OUT/'input_manifest.json',Path(A.__file__),Path(E.__file__),ROOT/'r2/src/meshflow_control/geo_c.py',ROOT/'r2/src/meshflow_control/winding_solver.py'):
        protected[str(p0)]=A.sha(p0)
    for r in rr:
        cp=A.OUT/'cells/B_020'/r['short_id']/'complete.json'
        protected[str(cp)]=A.sha(cp);protected.update(A.read(cp)['files'])
        bp=A.OUT/'baseline'/r['short_id']/'connectivity.json';protected[str(bp)]=A.sha(bp)
        pair=A.resolve(r['score_path']).parent/'pairs.json';protected[str(pair)]=A.sha(pair)
    reg=dict(status='FROZEN_BEFORE_EXECUTION',version='idea1b_terminal_affine_v1',requests=24,
        inputs=[r['request_id'] for r in rr],L=L,source_sha256={str(Path(__file__)):A.sha(__file__)},
        protected_files=protected,CPU_processes=1,threads=1,GPU_calls=0,models=0,
        transform='All F positive diagonal affine about C bbox center; C untouched, shared C vertex identities not split.',
        scales=[.95,1.05],translation_per_axis_L=[-.02,.02],max_affine_displacement_L=.035,
        max_displacement_domain='All retained POST F vertices and all original pre_geo free corners including dropped source faces.',
        anchored_axes='>=2 distinct anchor coordinates: scale=1,translation=0; single coordinate: shift=-scale_delta*(anchor-center); no anchor: independent scale and translation.',
        objective='32 equal midpoint samples per original C open edge; original edge-length weighted mean distance to finite original POST F-open segments. No GT/reward selection.',
        optimizer=dict(method='Powell',starts=1,start='identity',maximum_objective_evaluations=300,xtol=1e-4,ftol=1e-6,maxiter=10000,
            feasibility_penalty='distance/L + 1000*max(0,max_affine_displacement/(.035L)-1); best feasible only retained; budget exhaustion is not infeasibility proof'),
        B2='Unmodified original B2 engine using affine POST representatives and affine pre_geo; original corner clusters; 2%L relative to transformed pre_geo. Total displacement from original pre_geo separately recorded, NOT promised <=2%.',
        evaluation='Two new full scores/request (48 max): AFFINE and AFFINE_B2. Original POST and original B2 caches only. Exact original scorers and C queries.',
        screening='Inherited RAW YES AND new valid CF. This is screening only; no automatic new POST adoption label.',
        rendering='All 608/610; any new/lost legal CF vs original; each stage worst positive CF/FF/I delta. Deduplicated before/AFFINE/AFFINE_B2 same-camera CPU pages.',
        theory='Invertible uniform F affine preserves exact FF intersections/topology; FP32/predicate tolerance changes measured and disclosed. Binding effects separately paired.')
    write(OUT/'registration.json',reg);return reg
def domains(a):
    k=a['row']['K'];f=a['faces'];v=a['vertices']
    ci=E.incidence(f[:k]);fi=E.incidence(f,k)
    ce=np.asarray([e for e,o in sorted(ci.items()) if len(o)==1],np.int64)
    fe=np.asarray([e for e,o in sorted(fi.items()) if len(o)==1 and e[0]!=e[1]],np.int64)
    lengths=np.linalg.norm(v[ce[:,1]].astype(float)-v[ce[:,0]],axis=1)
    t=(np.arange(32)+.5)/32
    points=(v[ce[:,0],None]*(1-t)[None,:,None]+v[ce[:,1],None]*t[None,:,None]).reshape(-1,3).astype(float)
    weights=np.repeat(lengths/32,32);weights/=weights.sum()
    return ce,fe,points,weights
def distance(points,weights,vertices,edges):
    seg=vertices[edges].astype(float);a=seg[:,0];d=seg[:,1]-a;den=(d*d).sum(1)
    diff=points[:,None,:]-a
    t=np.clip(np.divide(np.einsum('pei,ei->pe',diff,d),den[None],out=np.zeros(diff.shape[:2]),where=den[None]>0),0,1)
    dd=diff-t[:,:,None]*d
    nearest=np.sqrt((dd*dd).sum(2).min(1))
    return float(weights@nearest)
def affine(a,p,center):
    v=a['vertices'].copy();k=a['row']['K']
    fv=np.unique(a['faces'][k:]);cv=np.unique(a['faces'][:k])
    moved=(v[fv].astype(float)-center)*(1+p[:3])+center+p[3:]*L
    v[fv]=moved.astype(np.float32);v[cv]=a['vertices'][cv]
    pre=a['pre_geo'].copy();pre[k:]=((pre[k:].astype(float)-center)*(1+p[:3])+center+p[3:]*L).astype(np.float32)
    dv=float(np.linalg.norm(v[fv].astype(float)-a['vertices'][fv],axis=1).max())
    dp=float(np.linalg.norm(pre[k:].reshape(-1,3).astype(float)-a['pre_geo'][k:].reshape(-1,3),axis=1).max())
    return v,pre,dv,dp
def optimize(a,dest):
    ce,fe,pts,w=domains(a);v=a['vertices'];f=a['faces'];k=a['row']['K']
    if not len(fe):raise ValueError('No free open edges; no registered objective')
    cpoints=v[np.unique(f[:k])].astype(float);center=(cpoints.min(0)+cpoints.max(0))/2
    anchors=np.intersect1d(np.unique(f[:k]),np.unique(f[k:]))
    columns=[];axis_audit=[]
    for j in range(3):
        vals=np.unique(v[anchors,j]) if len(anchors) else np.asarray([])
        if len(vals)>=2:axis_audit.append(dict(axis=j,kind='FIXED_TWO_OR_MORE_ANCHOR_COORDINATES',distinct=len(vals)));continue
        if len(vals)==1:
            coef=-(float(vals[0])-center[j])/L
            scale=min(.05,.02/abs(coef)) if coef else .05
            col=np.zeros(6);col[j]=scale;col[3+j]=coef*scale;columns.append(col)
            axis_audit.append(dict(axis=j,kind='ANCHOR_LINKED_SCALE_TRANSLATION',distinct=1))
        else:
            col=np.zeros(6);col[j]=.05;columns.append(col)
            col=np.zeros(6);col[3+j]=.02;columns.append(col)
            axis_audit.append(dict(axis=j,kind='FREE_SCALE_AND_TRANSLATION',distinct=0))
    basis=np.stack(columns,1) if columns else np.zeros((6,0))
    base=distance(pts,w,v,fe);history=[];best=dict(value=base,p=np.zeros(6),index=0)
    rejected=Counter()
    def objective(x):
        if len(history)>=300:raise RuntimeError('Objective hard cap violated')
        p=basis@x
        vv,pp,dv,dp=affine(a,p,center)
        anchor_error=float(np.max(np.abs(((v[anchors].astype(float)-center)*p[:3]+p[3:]*L)))) if len(anchors) else 0.
        excess=max(0,max(dv,dp)/(.035*L)-1)
        feasible=excess<=1e-12 and anchor_error<=1e-12 and np.max(abs(p[:3]))<=.05+1e-12 and np.max(abs(p[3:]))<=.02+1e-12
        value=distance(pts,w,vv,fe)
        if not feasible:rejected['AFFINE_DISPLACEMENT_CAP' if excess>1e-12 else 'ANCHOR_OR_PARAMETER_CONSTRAINT']+=1
        if feasible and value<best['value']-1e-14:best.update(value=value,p=p.copy(),index=len(history))
        history.append(dict(index=len(history),parameters=p,value_model=value,feasible=feasible,
                            max_post_displacement=dv,max_pre_geo_displacement=dp,anchor_affine_error=anchor_error))
        return value/L+1000*excess+(0 if anchor_error<=1e-12 else 1e6)
    objective(np.zeros(basis.shape[1]))
    if basis.shape[1]:
        opt=minimize(objective,np.zeros(basis.shape[1]),method='Powell',bounds=[(-1,1)]*basis.shape[1],
            options=dict(maxfev=299,maxiter=10000,xtol=1e-4,ftol=1e-6))
        result=dict(status=int(opt.status),success=bool(opt.success),message=str(opt.message),scipy_nfev=int(opt.nfev),nit=int(opt.nit))
    else:result=dict(status='NO_FEASIBLE_DOF',success=True,message='Shared C anchors fix all six affine parameters',scipy_nfev=0,nit=0)
    vv,pp,dv,dp=affine(a,best['p'],center)
    assert len(history)<=300 and max(dv,dp)<=.035*L+1e-10
    assert np.array_equal(vv[np.unique(f[:k])],v[np.unique(f[:k])])
    audit=dict(**result,actual_nfev=len(history),best_evaluation=best['index'],rejected_feasibility=dict(rejected),
        parameters=best['p'],scales=1+best['p'][:3],translation_model=best['p'][3:]*L,center=center,
        reduced_dof=basis.shape[1],basis=basis,anchor_vertex_ids=anchors,axes=axis_audit,
        baseline_distance_pctL=100*base/L,affine_distance_pctL=100*best['value']/L,
        max_post_displacement_pctL=100*dv/L,max_pre_geo_displacement_pctL=100*dp/L)
    write(dest/'optimization.json',audit);table(dest/'objective_history.csv',history)
    np.savez_compressed(dest/'affine_pre_geo.npz',output=pp,known_faces=np.asarray(k))
    return vv,pp,audit
def endpoints(a,vertices,pre):
    f=a['faces'];k=a['row']['K'];sf=a['source_face_ids'];cv=a['corner_vertex_ids'].ravel()
    ci=E.incidence(f[:k]);fi=E.incidence(f,k);ai=E.incidence(f)
    boundary=[e for e,o in sorted(ci.items()) if len(o)==1]
    op=[(e,o[0]) for e,o in sorted(fi.items()) if len(o)==1 and e[0]!=e[1]]
    flat=pre.reshape(-1,3).astype(float);mins=[];inside=0;clusters=0;unoccupied=0;raw=[]
    for edge,(i,j,l) in op:
        v0,v1=map(int,f[i,[j,l]]);axis=vertices[v1].astype(float)-vertices[v0]
        for b in boundary:
            axis2=vertices[b[1]].astype(float)-vertices[b[0]]
            den=np.linalg.norm(axis)*np.linalg.norm(axis2)
            if not den or abs(axis@axis2/den)<=np.cos(np.pi/6):continue
            sid=int(sf[i]);c0,c1=3*sid+j,3*sid+l
            dd,t0,t1=min((max(np.linalg.norm(flat[c0]-vertices[t0]),np.linalg.norm(flat[c1]-vertices[t1])),t0,t1) for t0,t1 in (b,b[::-1]))
            mins.append(float(dd));inside+=int(dd<=.02*L)
            ids0=np.flatnonzero((cv==v0)&(np.arange(len(cv))>=3*k));ids1=np.flatnonzero((cv==v1)&(np.arange(len(cv))>=3*k))
            dc=max(np.linalg.norm(flat[ids0]-vertices[t0],axis=1).max(),np.linalg.norm(flat[ids1]-vertices[t1],axis=1).max())
            clusters+=int(dd<=.02*L and dc<=.02*L)
            free=not any(ii>=k for ii,_,_ in ai[b])
            unoccupied+=int(dd<=.02*L and dc<=.02*L and free)
    return dict(axis_pass_pairs=len(mins),best_endpoint_max_pctL=min(mins)/L*100 if mins else None,
                endpoint_pairs_within2=inside,cluster_pairs_within2=clusters,unoccupied_cluster_pairs_within2=unoccupied)
def make_score(row,a,output,label,dest,base_ledger,base_pairs,aff_ledger=None,aff_pairs=None):
    dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'post.npz',**output)
    k=row['K'];score=Q.score_arrays(output['vertices'],output['faces'],k,a['C_canonical'],
          a['C_canonical_source_vertex_ids'],row['query_descriptor'],row['N'],output['source_face_ids'])
    vectors=score.pop('vectors');np.savez_compressed(dest/'vectors.npz',**vectors);write(dest/'score.json',score)
    nl=Q.connectivity_ledger(output['vertices'],output['faces'],k,score['pairs'],output['source_face_ids'],output['source_corner_ids'])
    write(dest/'connectivity.json',nl)
    ledger=Q.compare_ledgers(base_ledger,nl,base_pairs,score['pairs'],set(map(int,output['source_face_ids'][k:])))
    write(dest/'vs_original_ledger.json',ledger)
    for field in ('baseline_edge_fates','new_edges','illegal_pair_fates'):table(dest/(field+'.csv'),ledger[field])
    if aff_ledger is not None:
        incremental=Q.compare_ledgers(aff_ledger,nl,aff_pairs,score['pairs'],set(map(int,output['source_face_ids'][k:])))
        write(dest/'vs_affine_ledger.json',incremental)
    else:incremental=None
    m=score['metrics'];g=score['topology']
    oldB=A.read(A.OUT/'cells/B_020'/row['short_id']/'score.json')
    rr=dict(label=label,request_id=row['request_id'],short_id=row['short_id'],parent_id=row['parent_id'],split=row['split'],
        raw_adoption_inherited=row['raw_adoption'],new_POST_review='NOT_REVIEWED',
        baseline_joint=int(row['baseline_joint']),screening_joint=int(row['raw_adoption']=='YES' and g['C_F_valid_edge_count']>0),
        valid_cf_edges=g['C_F_valid_edge_count'],valid_cf_length=g['C_F_valid_edge_total_length'],
        old_B2_valid_cf_edges=oldB['topology']['C_F_valid_edge_count'],
        old_B2_valid_cf_length=oldB['topology']['C_F_valid_edge_total_length'],
        **ledger['summary'])
    for key in KEYS:
        rr[key]=m[key];rr['original_'+key]=float(row['score_table_row'][key])
        rr['delta_original_pp_'+key]=100*(m[key]-rr['original_'+key])
        rr['delta_old_B2_pp_'+key]=100*(m[key]-oldB['metrics'][key])
    if incremental is not None:rr.update({'incremental_B2_'+k:v for k,v in incremental['summary'].items()})
    return rr,nl,score['pairs']
def worker(row):
    dest=OUT/'requests'/row['short_id'];done=dest/'complete.json'
    if done.exists():
        rec=A.read(done);A.verify(rec['files']);return rec['rows']
    if dest.exists():raise RuntimeError('Unfinished request; no automatic retry: '+str(dest))
    dest.mkdir(parents=True);tick=time.perf_counter();event('REQUEST_START',short_id=row['short_id'])
    try:
        a=A.load_asset(row);k=row['K'];origpairs=A.read(A.resolve(row['score_path']).parent/'pairs.json')
        bl=A.read(A.OUT/'baseline'/row['short_id']/'connectivity.json')
        vv,pre,opt=optimize(a,dest)
        base=dict(a['canonical']);base.update(vertices=vv,output=vv[a['faces']])
        affine_out=dict(vertices=vv,faces=a['restored_faces'],output=vv[a['restored_faces']],known_faces=np.asarray(k),
                        source_face_ids=a['source_face_ids'],source_corner_ids=a['post']['source_corner_ids'])
        original_ep=endpoints(a,a['vertices'],a['pre_geo']);after_ep=endpoints(a,vv,pre)
        write(dest/'endpoint_reachability.json',dict(original=original_ep,affine=after_ep,warning='Geometry candidates are not all legal; actual B2 acceptance and scored edge results separate.'))
        rr1,al,ap=make_score(row,a,affine_out,'AFFINE',dest/'AFFINE',bl,origpairs)
        inv=np.argsort(a['source']['C_source_corner_permutations'],axis=1)
        out,candidates,deleted,binding=E.decode(pre,base,k,inv,'B',.02,Q.classify_cf)
        write(dest/'binding.json',binding);write(dest/'candidate_records.json',candidates);table(dest/'candidate_records.csv',candidates)
        table(dest/'deleted_faces.csv',deleted)
        rr2,nl,npairs=make_score(row,a,out,'AFFINE_B2',dest/'AFFINE_B2',bl,origpairs,al,ap)
        assert rr2['incremental_B2_CF_new_illegal_pairs']==0
        ids=out['assignment_input_corner_ids'];targets=out['assignment_target_C_vertex_ids']
        total_bound=float(np.linalg.norm(a['pre_geo'].reshape(-1,3)[ids].astype(float)-out['vertices'][targets],axis=1).max()) if len(ids) else 0.
        total_retained=float(np.linalg.norm(out['output'].astype(float)-a['pre_geo'][out['source_face_ids']][np.arange(len(out['source_face_ids']))[:,None],out['source_corner_ids']],axis=2)[k:].max())
        displacement=dict(affine_POST_max_pctL=opt['max_post_displacement_pctL'],affine_pre_geo_max_pctL=opt['max_pre_geo_displacement_pctL'],
            B2_bound_max_from_transformed_pre_pctL=binding['max_bound_corner_displacement']/L*100,
            B2_bound_total_from_original_pre_pctL=total_bound/L*100,
            final_retained_max_from_original_pre_pctL=total_retained/L*100,
            total_contract='No total <=2% claim; original GEO representative consolidation also present for unbound corners.')
        write(dest/'displacements.json',displacement)
        for rr in (rr1,rr2):
            rr.update(optimizer_actual_nfev=opt['actual_nfev'],optimizer_success=opt['success'],
                reduced_dof=opt['reduced_dof'],baseline_distance_pctL=opt['baseline_distance_pctL'],
                affine_distance_pctL=opt['affine_distance_pctL'],original_best_endpoint_pctL=original_ep['best_endpoint_max_pctL'],
                affine_best_endpoint_pctL=after_ep['best_endpoint_max_pctL'],
                original_cluster_candidates_2pct=original_ep['unoccupied_cluster_pairs_within2'],
                affine_cluster_candidates_2pct=after_ep['unoccupied_cluster_pairs_within2'],
                B2_accepted=binding['accepted'],**displacement)
        files={str(p):A.sha(p) for p in dest.rglob('*') if p.is_file()}
        write(done,dict(status='COMPLETE',rows=[rr1,rr2],files=files,seconds=time.perf_counter()-tick,model_calls=0,GPU_calls=0))
        event('REQUEST_COMPLETE',short_id=row['short_id'],seconds=time.perf_counter()-tick,nfev=opt['actual_nfev'],
              affine_edges=rr1['valid_cf_edges'],bound_edges=rr2['valid_cf_edges'],accepted=binding['accepted'])
        print(json.dumps(dict(request=row['short_id'],nfev=opt['actual_nfev'],distance=[opt['baseline_distance_pctL'],opt['affine_distance_pctL']],CF=[rr1['valid_cf_edges'],rr2['valid_cf_edges']],seconds=round(time.perf_counter()-tick,2))),flush=True)
        return [rr1,rr2]
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),status='FAILED_NO_AUTOMATIC_RETRY'))
        event('REQUEST_FAILED',short_id=row['short_id']);raise
def aggregate(rr):
    table(OUT/'per_output.csv',rr);summary=[]
    for label in ('AFFINE','AFFINE_B2'):
        for split in ('ALL','TRAIN','DEV'):
            ss=[r for r in rr if r['label']==label and (split=='ALL' or r['split']==split)]
            s=dict(label=label,split=split,outputs=len(ss),interfaces=sum(r['valid_cf_edges']>0 for r in ss),
                   screening_joint=sum(r['screening_joint'] for r in ss),new_screening_joint=sum(r['screening_joint'] and not r['baseline_joint'] for r in ss),
                   lost_screening_joint=sum(r['baseline_joint'] and not r['screening_joint'] for r in ss))
            for key in ('valid_cf_edges','valid_cf_length','CF_lost_legal_edges','CF_new_legal_edges','FF_lost_legal_edges','FF_new_illegal_pairs','FF_illegal_pairs_removed','deleted_source_faces'):
                s[key]=sum(r[key] for r in ss)
            for key in KEYS:s['delta_original_pp_'+key]=float(np.mean([r['delta_original_pp_'+key] for r in ss]));s['delta_old_B2_pp_'+key]=float(np.mean([r['delta_old_B2_pp_'+key] for r in ss]))
            summary.append(s)
    table(OUT/'summary.csv',summary);write(OUT/'summary.json',summary)
    return summary
def gallery(rr):
    from PIL import Image,ImageDraw
    from r2.tools.partial_edge_binding_v1.render import _font,VIEWS,SIZE
    from r2.pilots.regular_patch_control_v1.depth_gallery import basis,raster
    selected={r['short_id'] for r in rr if r['short_id'].startswith(('608','610')) or r['CF_new_legal_edges'] or r['CF_lost_legal_edges']}
    for label in ('AFFINE','AFFINE_B2'):
        for key in KEYS[:3]:
            ss=sorted((r for r in rr if r['label']==label),key=lambda r:(-r['delta_original_pp_'+key],r['short_id']))
            if ss[0]['delta_original_pp_'+key]>0:selected.add(ss[0]['short_id'])
    byid={r['short_id']:r for r in rows()};images=[];dest=OUT/'gallery';dest.mkdir(exist_ok=True)
    for short in sorted(selected):
        row=byid[short];paths=[A.resolve(row['post_path'])]+[OUT/'requests'/short/lab/'post.npz' for lab in ('AFFINE','AFFINE_B2')]
        meshes=[A.arrays(p)['output'] for p in paths]
        if any(x[:row['K']].tobytes()!=meshes[0][:row['K']].tobytes() for x in meshes):raise ValueError('C render contract')
        points=np.concatenate([x.reshape(-1,3) for x in meshes]);center=(points.min(0)+points.max(0))/2
        cameras=[basis(az,el) for _,az,el in VIEWS];span=max(float(np.ptp((points-center)@b,axis=0)[:2].max()) for b in cameras)
        size=360;page=Image.new('RGB',(3*size,3*size+100),'white');draw=ImageDraw.Draw(page)
        draw.text((8,8),short+' '+row['split']+' | blue C | original / affine / affine+B2',font=_font(16),fill='black')
        for col,(mesh,lab) in enumerate(zip(meshes,('Original POST','AFFINE','AFFINE + B2'))):
            draw.text((col*size+8,33),lab,font=_font(15),fill='black')
            for j,((name,_,_),camera) in enumerate(zip(VIEWS,cameras)):
                im,_=raster(mesh,row['K'],camera,center,span,size=size);page.paste(im,(col*size,58+j*size))
                draw.text((col*size+5,62+j*size),name,font=_font(12),fill='black')
        draw.text((8,3*size+70),'Same cameras/scale; depth ties favor C. Actual image review required; no semantic labels inferred.',font=_font(11),fill='black')
        p=dest/(short+'.jpg');page.save(p,quality=92,subsampling=0)
        images.append(dict(short_id=short,path=str(p),sha256=A.sha(p),source_paths=list(map(str,paths)),review_status='NOT_REVIEWED'))
    write(dest/'manifest.json',dict(status='COMPLETE',images=images,selection=freeze()['rendering'],automatic_labels=0))
    (dest/'index.html').write_text('<meta charset="utf-8"><h1>Idea1B fixed selection</h1>'+''.join('<h2>'+r['short_id']+'</h2><img width="1080" src="'+Path(r['path']).name+'">' for r in images),encoding='utf8')
    return images
def report(summary,rr,seconds):
    parts=['# Idea1B fixed terminal affine CPU diagnostic','',
        'All 24 frozen FULL_S6000 requests completed; 48 original scorer calls. No model, GPU, training or new noise.','',
        'Inherited RAW adoption is used only for screening. New POST usage/body labels remain NOT_REVIEWED.','',
        '|Stage|Split|Interfaces|Screening joint|New/lost screened|Legal CF edges|Lost FF edges|New illegal FF pairs|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:parts.append(f"|{s['label']}|{s['split']}|{s['interfaces']}/{s['outputs']}|{s['screening_joint']}/{s['outputs']}|{s['new_screening_joint']}/{s['lost_screening_joint']}|{s['valid_cf_edges']}|{s['FF_lost_legal_edges']}|{s['FF_new_illegal_pairs']}|")
    parts+=['','The objective is fixed finite-edge distance, not the reward. One identity-start bounded Powell run uses at most 300 evaluations/request; a budget limit is not proof of no better affine solution. Shared C anchors eliminate infeasible affine degrees of freedom.','',
        'Affine displacement is capped at 3.5%L on retained F representatives and all original pre_geo F corners. B2 is exactly the old decoder with 2%L relative to transformed pre_geo; the separate total displacement includes original GEO representative consolidation and is NOT claimed <=2%L.','',
        'An invertible global affine preserves exact F/F intersections. Classification changes under AFFINE alone may reflect FP32 geometry/predicate tolerances; all actual pair differences are retained rather than called self-intersection repair. Additional B2 changes are separately paired in vs_affine_ledger.json.','',
        f'Execution wall time before report: {seconds:.2f}s, one CPU process, one BLAS/OMP thread. Full rows: per_output.csv; before/after endpoint reachability and objective trace per request; fixed-selection CPU images in gallery/. No old assets were modified.']
    (OUT/'REPORT.md').write_text('\n'.join(parts)+'\n',encoding='utf8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('freeze','run'),required=True);args=p.parse_args()
    reg=freeze()
    if args.stage=='freeze':print(json.dumps(dict(status=reg['status'],requests=24,protected=len(reg['protected_files']))));return
    if (OUT/'complete.json').exists():
        done=A.read(OUT/'complete.json');A.verify(done['files']);print('COMPLETE_CACHE_VERIFIED');return
    A.verify(reg['source_sha256']);A.verify(reg['protected_files'])
    if 'torch' in sys.modules:raise RuntimeError('Unexpected model import')
    start=time.perf_counter();event('EXECUTION_START',pid=os.getpid(),CPU_processes=1,threads=1)
    results=[]
    for r in rows():results.extend(worker(r))
    summaries=aggregate(results);images=gallery(results);report(summaries,results,time.perf_counter()-start)
    A.verify(reg['source_sha256']);A.verify(reg['protected_files'])
    if 'torch' in sys.modules:raise RuntimeError('Model framework imported')
    files={str(p):A.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='execution.jsonl'}
    event('EXECUTION_COMPLETE',seconds=time.perf_counter()-start,full_scores=48)
    write(OUT/'complete.json',dict(status='COMPLETE',requests=24,full_scores=48,model_calls=0,GPU_calls=0,
          seconds=time.perf_counter()-start,images=len(images),protected_files=len(reg['protected_files']),files=files))
    print(json.dumps(dict(status='COMPLETE',seconds=time.perf_counter()-start,summary=summaries)),flush=True)
if __name__=='__main__':main()
