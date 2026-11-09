"""CPU-only local-repair gallery, preserving inherited views and finite cuts.

Rendering is gated on completed target evaluation. It does not create targets,
score geometry, call a model, or assign visual-success labels.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse, copy, html, io, json, resource, sys, time, traceback
from pathlib import Path
import numpy as np
from . import common as C
from . import assets as A
from r2.pilots.bench_backend_relative_deformation_v1 import render as OLD_RENDER
G,D,H=OLD_RENDER.G,OLD_RENDER.D,OLD_RENDER.H
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT,OUT,L=C.ROOT,C.OUT,C.L
AXES=OLD_RENDER.AXES
STATE_NAME={0:'UNKNOWN',1:'KEEP',2:'EDIT',3:'C'}
STATE_COLOR={0:'#91979f',1:'#248951',2:'#eb8e25',3:'#1686c0'}
VERSION='bench_raw_local_repair_fixed_views_states_support_cuts_all_accepted_v1'

def specs(task):
    rid=task['request_id'];assets=task['assets']
    return [
        ('RAW','COVERAGE_FINAL','RAW',assets['RAW']['path']),
        ('TARGET','LOCAL_REPAIR','TARGET',OUT/'targets'/rid/'target.npz'),
        ('GEO_TARGET','LOCAL_REPAIR','BASE_POST',OUT/'geometry'/rid/'post.npz'),
        ('FINAL_TARGET','LOCAL_REPAIR','CLEAN_FINAL_POST',OUT/'cpu/LOCAL_REPAIR'/rid/'CLEAN_FINAL_POST/post.npz'),
        ('SHAPE_FINAL','SHAPE_D0','CLEAN_FINAL_POST',assets['SHAPE_FINAL']['path']),
        ('RELDEF_FINAL','RELDEF_D0','CLEAN_FINAL_POST',assets['RELDEF_FINAL']['path']),
    ]

def _mesh(path,kind,k):
    a=C.arrays(path)
    if kind=='RAW':
        # Repair tensors use original canonical RAW order. The saved `output`
        # may already restore C winding, with a nonidentity source_corner_ids.
        # Display the exact raw tensor and use its identity corner domain.
        tri=np.asarray(a['raw']);sf=np.asarray(a['raw_slot_ids'],np.int64)
        sc=np.tile(np.arange(3,dtype=np.int64),(len(tri),1))
    elif kind=='TARGET':
        tri=np.asarray(a['target']);sf=a['raw_slot_ids'];sc=a['source_corner_ids']
    else:
        tri=np.asarray(a.get('output',a.get('raw')))
        sf=np.asarray(a.get('parent_source_raw_slot',a.get('raw_slot_ids',a.get('source_face_ids'))),np.int64)
        sc=a['source_corner_ids']
    if tri.ndim!=3 or tri.shape[1:]!=(3,3) or len(tri)<k or not np.isfinite(tri).all():raise ValueError('Invalid mesh input')
    if sf.shape!=(len(tri),) or sc.shape!=(len(tri),3):raise ValueError('Wrong source-slot/corner mapping domain')
    role=np.asarray(a.get('face_role',['C']*k+['BULK']*(len(tri)-k))).astype('<U6')
    if role.shape!=(len(tri),) or set(role)-{'C','BULK','BRIDGE'}:raise ValueError('Unexpected Native roles')
    return dict(tri=tri,sf=sf,sc=sc,role=role,arrays=a)

def _project(points,az,center,span,size=440):
    v=(np.asarray(points,float)-np.asarray(center))@D.basis(az,23)
    return np.stack([size*.5+v[...,0]*size*.86/span,size*.5-v[...,1]*size*.86/span],axis=-1)

def _raster(ax,mesh,k,az,depth,title,states=None,labels=False):
    tri=mesh['tri'];role=mesh['role']
    raster,scope=G.role_raster(tri,role,k,D.basis(az,23),np.asarray(depth['center']),depth['span'])
    ax.imshow(raster);ax.set_axis_off();ax.set_title(title,fontsize=9)
    if states is not None:
        p=_project(tri,az,depth['center'],depth['span'])
        # Overlay is explicitly an x-ray identity annotation, not z-buffer visibility.
        for state in (0,1,3,2):
            at=states==state
            ax.scatter(p[...,0][at],p[...,1][at],s=7 if state!=2 else 18,c=STATE_COLOR[state],alpha=.7 if state!=2 else 1,edgecolors='none',zorder=10+state)
        if labels:
            for fi in np.flatnonzero((states==2).any(1)):
                mid=p[fi].mean(0);ax.text(mid[0],mid[1],str(int(mesh['sf'][fi])),fontsize=6,color='#864c00',zorder=20,
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.3))
    # X-ray annotations outside a local crop must not autoscale that crop.
    ax.set_xlim(-.5,439.5);ax.set_ylim(439.5,-.5)
    return scope

def _mapped_states(mesh,states):
    sf,sc=mesh['sf'],mesh['sc'];out=np.zeros((len(sf),3),np.uint8)
    real=(sf>=0)&(sf<len(states))
    out[real]=states[sf[real,None],sc[real]]
    return out

def _section(ax,mesh,k,co,bounds,view,index,task,name,stage,rows,page,states=None):
    tri=mesh['tri'];sf=mesh['sf'];g=dict(C=tri[:k],F=tri[k:],C_original=co,tri=tri)
    selected=[] if states is None else np.flatnonzero((states[k:]==2).any(1)).tolist()
    draw_bounds=bounds;draw_index=index
    if index>2:
        # Legacy drawer has only three title fractions, even though the plane
        # itself is supplied by bounds. Route this extra display-only plane
        # through its middle slot; ledger and final title retain the real plane.
        draw_bounds=copy.deepcopy(bounds);draw_index=1
        key='front_positions' if view=='front' else 'side_positions'
        draw_bounds[key]=list(draw_bounds[key]);draw_bounds[key][1]=bounds[key][index]
    H.draw_section(ax,g,selected,draw_bounds,L,view,draw_index)
    axis,proj=(2,(0,1)) if view=='front' else (0,(2,1))
    pos=float(bounds['front_positions' if view=='front' else 'side_positions'][index])
    for kind,part,offset in (('C',tri[:k],0),('FREE',tri[k:],k)):
        segments,ids=H.finite_sections(part,axis,pos,1e-12*L)
        for seg,li in zip(segments,ids):
            fi=int(li)+offset;sid=int(sf[fi]);mid=seg[:,proj].mean(0)
            if kind=='FREE':
                tag='bridge'+str(fi) if sid<0 else 'f'+str(sid)
                edit=states is not None and (states[fi]==2).any()
                ax.text(mid[0],mid[1],tag,fontsize=5,color='#ab5e08' if edit else '#303640',zorder=8,
                    bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=.3))
            row=dict(request_id=task['request_id'],page=page,column=name,stage=stage,view=view,plane_axis=axis,plane_position=pos,
                     output_face_id=fi,raw_slot_id=sid,role=str(mesh['role'][fi]))
            for end,p in zip(('start','end'),seg):
                for d,x in zip('xyz',p):row['segment_'+end+'_'+d]=float(x)
            rows.append(row)
    ax.set_title(name+f' | {view}={pos:.5f}',fontsize=8)

def _save(fig,path):
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=140)
    with path.open('xb') as f:f.write(buffer.getvalue())

def _support_bounds(raw,k,bounds):
    """Extra cuts determined once from RAW; identical for every compared stage."""
    out=copy.deepcopy(bounds);pts=raw.reshape(-1,3).astype(float);lo,hi=pts.min(0),pts.max(0)
    margin=.025*L;out['local_low']=(lo-margin).tolist();out['local_high']=(hi+margin).tolist()
    c_y=float(np.median(raw[:k,:,1]));centers=raw[k:].mean(1)
    lower=centers[centers[:,1]<c_y-.05*L]
    found=[]
    if len(lower):
        middle=float(np.median(pts[:,0]))
        for at in (lower[:,0]<=middle,lower[:,0]>middle):
            group=lower[at]
            if len(group):
                # A centroid plane intersects an actual RAW lower free triangle.
                order=np.argsort(group[:,0],kind='stable');found.append(float(group[order[len(order)//2],0]))
    cuts=list(bounds['side_positions'])
    while len(found)<2:found.append(float(cuts[0 if not found else -1]))
    return out,found

def _states_figure(task,meshes,states,depth,dest,plt):
    fig,axes=plt.subplots(2,2,figsize=(11,10),dpi=140)
    for col,name in enumerate(('RAW','TARGET')):
        for row,az in enumerate((-60,120)):
            _raster(axes[row,col],meshes[name],task['K'],az,depth,name+f' corner states | az={az}',states,labels=True)
    fig.suptitle(task['request_id']+'\nC blue / KEEP green / EDIT orange / UNKNOWN gray; numbers are EDIT RAW slots',fontsize=9)
    fig.text(.5,.02,'X-ray corner annotation: hidden corners are deliberately shown. UNKNOWN coordinates are placeholders, not repair supervision.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94));p=dest/'corner_states.png';_save(fig,p);plt.close(fig)
    return p

def _patch_figures(task,meshes,states,patches,dest,plt):
    chosen=[p for p in patches if p['decision']=='ACCEPTED']
    chosen.extend([p for p in patches if p['decision']=='REJECTED'][:2])
    # Every accepted edit is shown; rejected examples follow original log order.
    if not chosen:
        fallback=np.flatnonzero((states==2).any(1)).tolist()
        if not fallback:fallback=np.flatnonzero((states==0).any(1)).tolist()
        chosen=[dict(patch_id='MASK_OVERVIEW_NO_PATCH_LOG',decision='NA',raw_slots=fallback,reason='No accepted/rejected patch available',proposal_path=None)]
    records=[];images=[];raw=meshes['RAW']['tri'];k=task['K']
    for start in range(0,len(chosen),2):
        page=chosen[start:start+2]
        fig,axes=plt.subplots(len(page),4,figsize=(18,4.3*len(page)),dpi=140,squeeze=False)
        p=dest/('patch_zoom.png' if start==0 else f'patch_zoom_{start//2:03d}.png')
        for row,patch in enumerate(page):
            slots=sorted(set(map(int,patch['raw_slots'])))
            if any(i<0 or i>=len(raw) for i in slots):raise ValueError('Patch RAW slot out of range')
            prop=meshes['TARGET']['tri']
            if patch.get('proposal_path'):
                pa=C.arrays(patch['proposal_path']);prop=pa['output']
                if prop.shape!=raw.shape:raise ValueError('Patch proposal does not preserve RAW slot count')
            pts=np.concatenate([raw[slots].reshape(-1,3),prop[slots].reshape(-1,3)]) if slots else raw.reshape(-1,3)
            center=(pts.min(0)+pts.max(0))/2
            span=max(max(np.ptp((pts-center)@D.basis(az,23),axis=0)[:2]) for az in (-60,120))
            depth=dict(center=center,span=max(float(span)*1.25,.04*L))
            pmesh=dict(meshes['RAW'],tri=prop)
            unmappable=any(int(x)<0 for x in patch.get('cluster_ids',[])) or 'UNMAPPABLE' in str(patch.get('reason','')).upper()
            tag=' UNMAPPABLE' if unmappable else ''
            for col,(name,mesh,az) in enumerate((('RAW',meshes['RAW'],-60),('PATCH_PROPOSAL',pmesh,-60),('RAW',meshes['RAW'],120),('PATCH_PROPOSAL',pmesh,120))):
                _raster(axes[row,col],mesh,k,az,depth,f"{patch['patch_id']} {patch['decision']}{tag} | {name} {az}",states,labels=True)
            records.append(dict(patch_id=patch['patch_id'],decision=patch['decision'],raw_slots=slots,reason=patch.get('reason',''),
                proposal_path=patch.get('proposal_path'),display_center=center.tolist(),display_span=depth['span'],image_path=str(p),image_row=row,unmappable=unmappable))
        fig.suptitle(task['request_id']+f'\nAll ACCEPTED + first two REJECTED, original log order | page {start//2+1}; rejected proposals NOT committed',fontsize=9)
        fig.text(.5,.015,'Same local frame for RAW/proposal. Orange labels refer to committed EDIT state; all affected slots are listed in patch_display.json.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.95));_save(fig,p);plt.close(fig);images.append(p)
    C.write(dest/'patch_display.json',dict(display_policy='allaccepted+first2rejected_two_rows_per_page',rows=records,images=[str(p) for p in images]))
    return images

def _barrier(task):
    ready=OUT/'targets_READY.json';complete=OUT/'scores/COMPLETE.json'
    if not ready.exists() or not complete.exists():raise RuntimeError('Targets/evaluation not ready; renderer may not run early')
    reg=C.read(complete)
    if reg.get('status')!='COMPLETE' or reg.get('new_scores')!=48 or set(reg.get('labels',[]))!={'RAW_IDENTITY','LOCAL_REPAIR'}:
        raise RuntimeError('Full target + RAW identity 48-row scoring barrier incomplete')
    rid=task['request_id'];paths=[OUT/'geometry'/rid/'complete.json',OUT/'cpu/LOCAL_REPAIR'/rid/'complete.json',OUT/'cpu/RAW_IDENTITY'/rid/'complete.json']
    book={str(ready):C.sha(ready),str(complete):C.sha(complete)}
    for p in paths:
        r=C.read(p)
        if r.get('status')!='COMPLETE':raise RuntimeError('Per-request geometry/cascade incomplete')
        C.verify(r['files']);book[str(p)]=C.sha(p)
    return book

def build(task,manifest_sha256):
    barrier=_barrier(task);rid=task['request_id'];dest=OUT/'gallery'/rid;done=dest/'complete.json'
    ss=specs(task);target_path=OUT/'targets'/rid/'target.npz';patchpath=OUT/'targets'/rid/'patches.json'
    patchbook=C.read(patchpath);patches=patchbook['rows']
    sources={str(C.resolve(p)):C.sha(p) for _,_,_,p in ss};sources[str(patchpath)]=C.sha(patchpath)
    identity_final=OUT/'cpu/RAW_IDENTITY'/rid/'CLEAN_FINAL_POST/post.npz'
    sources[str(identity_final)]=C.sha(identity_final)
    for p in patches:
        if p.get('proposal_path'):sources[str(C.resolve(p['proposal_path']))]=C.sha(p['proposal_path'])
    identity=dict(version=VERSION,request_id=rid,manifest_sha256=manifest_sha256,sources=sources,barriers=barrier,
        source_hashes={str(Path(mod.__file__)):C.sha(mod.__file__) for mod in (OLD_RENDER,G,D,H)},renderer_sha256=C.sha(__file__))
    if done.exists():
        r=C.read(done)
        if r['identity']!=identity:raise ValueError('Rendered identity changed')
        C.verify(r['files']);return r
    if dest.exists():raise RuntimeError('Incomplete render retained; explicit recovery required')
    dest.mkdir(parents=True);C.write(dest/'attempt.json',dict(identity=identity,time=time.time(),model_calls=0))
    plt=H._mpl();tick=time.perf_counter();fig=None
    try:
        q=task['request'];k=task['K'];n=task['N'];bounds=q['gallery_bounds'];depth=q['gallery_depth_bounds']
        inp=C.arrays(task['assets']['INFERENCE']['path']);co=inp['C_vertices'][inp['C_faces_original']]
        tar=C.arrays(target_path);states=tar['corner_state'];assert states.shape==(n,3) and set(np.unique(states))<={0,1,2,3}
        assert np.all(states[:k]==3) and np.all(states[k:]!=3)
        meshes={name:_mesh(path,stage,k) for name,_,stage,path in ss};raw=meshes['RAW']['tri']
        assert np.array_equal(tar['raw'],raw) and np.array_equal(tar['target'][:k],raw[:k])
        sections=[];catalog=[];cells=[]
        fig=plt.figure(figsize=(32.1,13.8),dpi=140,facecolor='white')
        grid=fig.add_gridspec(4,6,left=.025,right=.99,bottom=.07,top=.9,wspace=.18,hspace=.32)
        for col,(name,label,stage,path) in enumerate(ss):
            mesh=meshes[name];tri=mesh['tri'];sf=mesh['sf'];mapped=_mapped_states(mesh,states)
            if any(sorted(map(tuple,x))!=sorted(map(tuple,y)) for x,y in zip(tri[:k],co)):raise ValueError('C geometry changed')
            facts=dict(cell_id=rid+'__'+name,column=col,label=label,stage=stage,path=str(C.resolve(path)),faces=len(tri),views=[])
            for fi,(face,sid) in enumerate(zip(tri,sf)):
                r=dict(request_id=rid,column=name,stage=stage,output_face_id=fi,raw_slot_id=int(sid),role=str(mesh['role'][fi]))
                for vi in range(3):
                    r[f'corner{vi}_raw_id']=int(mesh['sc'][fi,vi]);r[f'corner{vi}_state']=STATE_NAME[int(mapped[fi,vi])]
                    for ci,axis in enumerate('xyz'):r[f'v{vi}_{axis}']=float(face[vi,ci])
                catalog.append(r)
            for row,az in enumerate((-60,120)):
                ax=fig.add_subplot(grid[row,col]);scope=_raster(ax,mesh,k,az,depth,name+f' | az={az}\nN={len(tri)} / free={len(tri)-k}')
                facts['views'].append(dict(azimuth=az,**scope))
            for row,view in ((2,'front'),(3,'side')):
                _section(fig.add_subplot(grid[row,col]),mesh,k,co,bounds,view,1,task,name,stage,sections,'comparison',mapped if name in ('RAW','TARGET','GEO_TARGET','FINAL_TARGET') else None)
            cells.append(facts)
        fig.suptitle(rid+' | '+task['eval_group']+'\nRAW / local TARGET / GEO(TARGET) / FINAL(TARGET) / SHAPE FINAL / RELDEF FINAL\nBlue C, gray free; inherited cameras and central cuts. Orange sections mark faces containing committed EDIT corners.',fontsize=10)
        fig.text(.5,.025,'TARGET is a local repair proposal, not certified complete ground truth. UNKNOWN coordinates remain placeholders.\nCentral sections can miss supports; see support_sections.png and corner/patch maps. Images do not certify connectivity or absence of intersections.',ha='center',fontsize=8)
        main=dest/'comparison.png';_save(fig,main);plt.close(fig);fig=None
        # Inherited lateral cuts plus two RAW-defined lower-support centroid cuts.
        support,extra=_support_bounds(raw,k,bounds)
        support['side_positions']=list(bounds['side_positions'])+extra
        cuts=[('side',0,'inherited left'),('side',2,'inherited right'),('side',3,'RAW lower left'),('side',4,'RAW lower right')]
        fig=plt.figure(figsize=(30,12),dpi=140,facecolor='white');grid=fig.add_gridspec(4,6,left=.025,right=.99,bottom=.06,top=.91,wspace=.2,hspace=.45)
        for col,(name,label,stage,path) in enumerate(ss):
            mesh=meshes[name];mapped=_mapped_states(mesh,states)
            for row,(view,idx,cutname) in enumerate(cuts):
                ax=fig.add_subplot(grid[row,col]);_section(ax,mesh,k,co,support,view,idx,task,name,stage,sections,'support_sections',mapped if col<4 else None)
                ax.set_title(name+' | '+cutname+f" X={support['side_positions'][idx]:.5f}",fontsize=8)
        fig.suptitle(rid+'\nSupport cuts: original side planes plus two planes through lower RAW free-face centroids; full RAW extent, same bounds for all stages',fontsize=10)
        fig.text(.5,.02,'Extra planes are display-only and derived from fixed RAW before candidate inspection. A missing cut alone does not prove a support is missing.',ha='center',fontsize=8)
        support_image=dest/'support_sections.png';_save(fig,support_image);plt.close(fig);fig=None
        C.write(dest/'display_contract.json',dict(original_bounds=bounds,original_depth=depth,support_bounds=support,support_cut_specs=cuts,extra_cut_origin='RAW lower free-face centroid; same across all stages',corner_states=STATE_NAME,geometry_changes=0,new_scoring_queries=0))
        states_image=_states_figure(task,meshes,states,depth,dest,plt)
        patch_images=_patch_figures(task,meshes,states,patches,dest,plt)
        # Matched no-D0 control: same original RAW, same GEO + final cascade.
        # This separates removing D0 from the incremental local target edits.
        identity_mesh=_mesh(identity_final,'CLEAN_FINAL_POST',k)
        matched_specs=[('RAW_IDENTITY_FINAL','RAW_IDENTITY',identity_mesh,identity_final),
                       ('LOCAL_REPAIR_FINAL','LOCAL_REPAIR',meshes['FINAL_TARGET'],C.resolve(ss[3][3]))]
        fig=plt.figure(figsize=(11.8,13.8),dpi=140,facecolor='white')
        grid=fig.add_gridspec(4,2,left=.06,right=.97,bottom=.07,top=.9,wspace=.22,hspace=.32)
        matched_cells=[]
        for col,(name,label,mesh,path) in enumerate(matched_specs):
            mapped=_mapped_states(mesh,states);tri=mesh['tri'];views=[]
            for row,az in enumerate((-60,120)):
                scope=_raster(fig.add_subplot(grid[row,col]),mesh,k,az,depth,name+f' | az={az}\nN={len(tri)} / free={len(tri)-k}')
                views.append(dict(azimuth=az,**scope))
            for row,view in ((2,'front'),(3,'side')):
                _section(fig.add_subplot(grid[row,col]),mesh,k,co,bounds,view,1,task,name,'CLEAN_FINAL_POST',sections,'matched_control',mapped if col==1 else None)
            for fi,(face,sid) in enumerate(zip(tri,mesh['sf'])):
                r=dict(request_id=rid,column=name,stage='CLEAN_FINAL_POST',output_face_id=fi,raw_slot_id=int(sid),role=str(mesh['role'][fi]))
                for vi in range(3):
                    r[f'corner{vi}_raw_id']=int(mesh['sc'][fi,vi]);r[f'corner{vi}_state']=STATE_NAME[int(mapped[fi,vi])]
                    for ci,axis in enumerate('xyz'):r[f'v{vi}_{axis}']=float(face[vi,ci])
                catalog.append(r)
            matched_cells.append(dict(column=col,label=label,stage='CLEAN_FINAL_POST',path=str(path),faces=len(tri),views=views))
        fig.suptitle(rid+'\nMatched no-D0 control: RAW identity FINAL vs local repair TARGET FINAL\nSame input noise, original GEO_C and final cascade; only local target edits differ',fontsize=9)
        fig.text(.5,.025,'Use this pair to assess edit benefit separately from bypassing D0. Original GEO-only was BASE, not this FINAL control.\nBlue C / gray free; fixed cameras and central sections. Orange cut segments contain committed EDIT corners.',ha='center',fontsize=8)
        matched_image=dest/'matched_control.png';_save(fig,matched_image);plt.close(fig);fig=None
        C.csvwrite(dest/'sections_face_ids.csv',sections);C.csvwrite(dest/'face_catalog.csv',catalog)
        if 'torch' in sys.modules:raise RuntimeError('CPU renderer imported torch')
        result=dict(status='RENDERED_PENDING_ACTUAL_REVIEW',identity=identity,request_id=rid,parent_id=task['parent_id'],condition_id=task['condition_id'],noise_index=task['noise_index'],eval_group=task['eval_group'],
            path=str(main),image_sha256=C.sha(main),supplements=[dict(path=str(p),sha256=C.sha(p)) for p in [matched_image,support_image,states_image]+patch_images],cells=cells,
            matched_control=dict(path=str(matched_image),sha256=C.sha(matched_image),cells=matched_cells,interpretation='Both no-D0 with identical GEO/final cascade; local editing increment only'),
            seconds=time.perf_counter()-tick,model_calls=0,scoring_calls=0,geometry_edits=0,new_noise=0,new_queries=0,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
        C.write(done,result);print('RENDERED',rid,round(result['seconds'],2),flush=True);return result
    except BaseException as exc:
        C.write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:
        if fig is not None:plt.close(fig)

def run(request_id=None):
    mp=OUT/'data/manifest.json';m=A.load_manifest(True);mh=C.sha(mp);tasks=m['tasks']
    if request_id is not None:
        selected=[t for t in tasks if t['request_id']==request_id]
        if len(selected)!=1:raise ValueError('Request is not among fixed12')
        return build(selected[0],mh)
    done=OUT/'gallery/COMPLETE.json'
    if done.exists():r=C.read(done);C.verify(r['files']);return r
    receipts=[build(t,mh) for t in tasks];pending=[]
    body='<meta charset="utf-8"><h1>Bench RAW local repair: fixed 12</h1><p>Actual cached meshes. TARGET is a partial repair proposal; UNKNOWN is not supervision. No automatic success labels.</p>'
    for r in receipts:
        body+='<h2>'+html.escape(r['request_id'])+'</h2>'
        for p in [r['path']]+[s['path'] for s in r['supplements']]:
            rel=C.resolve(p).relative_to(OUT/'gallery').as_posix();body+='<a href="'+rel+'"><img loading="lazy" style="width:100%" src="'+rel+'"></a>'
        for cell in r['cells']:
            pending.append(dict(**{key:r[key] for key in ('request_id','parent_id','condition_id','noise_index','eval_group')},
                **{key:cell[key] for key in ('cell_id','column','label','stage')},image_path=r['path'],image_sha256=r['image_sha256'],supplements=r['supplements'],reviewer=None,reviewer_kind='AI_PROXY',blind=False,actually_viewed=False,viewed_at_utc=None,**{key:None for key in AXES},note=None))
    index=OUT/'gallery/index.html'
    with index.open('x',encoding='utf-8') as f:f.write(body)
    C.write(OUT/'gallery/review_manifest.json',dict(status='PENDING_TEMPLATES_NOT_OBSERVATIONS',pages=12,cells=72,axes=AXES,cells_to_review=pending,
        protocol='Actual comparison, support cuts, corner states and local accepted/rejected patches must be read. Never infer ratings from metric counts.'))
    result=dict(status='RENDERED_PENDING_ACTUAL_REVIEW',pages=12,cells=72,workers=1,images=receipts,model_calls=0,scoring_calls=0,geometry_edits=0,new_noise=0,new_queries=0,
        files={str(p):C.sha(p) for p in (OUT/'gallery').rglob('*') if p.is_file()})
    C.write(done,result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request');args=parser.parse_args()
    r=run(args.request);print(json.dumps({k:v for k,v in r.items() if k not in ('images','files','cells')}))
