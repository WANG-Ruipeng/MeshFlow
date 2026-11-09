"""Five-column fixed-frame backend calibration gallery; no automatic labels."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,html,io,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from .evaluate import ROOT,OUT,OLD,LABEL,read,resolve,sha,write_new,arrays,csvwrite,tasks_from,verify,locations
from r2.pilots.chair_multipart_seam_v1 import gallery as G
D,H=G.D,G.H
L=float(np.sqrt(3.)*2/.3762)
AXES={
    'overall_shape':['REASONABLE','INCOMPLETE','UNCERTAIN'],
    'C_use':['PLAUSIBLE','REPLACED_OR_IGNORED','UNCERTAIN'],
    'support_retention':['PRESENT','MISSING_OR_DAMAGED','UNCERTAIN'],
    'visible_fragment_or_gap':['YES','NO_OBVIOUS','UNCERTAIN'],
    'layering_or_fold':['YES','NO_OBVIOUS','UNCERTAIN']}

def specs(t):
    o=t['original'];loc=locations(t)
    return [('COVERAGE_RAW','COVERAGE_FINAL','RAW',o['raw_path']),
        ('COVERAGE_FINAL_POST','COVERAGE_FINAL','CLEAN_FINAL_POST',o['final_post_path']),
        ('COVERAGE_GEO_ONLY','COVERAGE_FINAL','GEO_ONLY',t['new_post_path']),
        ('SHAPE_D0_BASE',LABEL,'BASE_POST',loc['base']),
        ('SHAPE_D0_FINAL',LABEL,'CLEAN_FINAL_POST',loc['final'])]

def one(t,mh):
    dest=OUT/'gallery'/t['request_id'];done=dest/'complete.json';ss=specs(t)
    identity=dict(request_id=t['request_id'],manifest_sha256=mh,render_source_sha256=sha(__file__),
        sources={str(resolve(p)):sha(p) for _,_,_,p in ss})
    if done.exists():
        r=read(done)
        if r['identity']!=identity:raise ValueError('Render identity changed')
        verify(r['files']);return r
    if dest.exists():raise RuntimeError('Partial render retained; no silent retry: '+str(dest))
    dest.mkdir(parents=True);write_new(dest/'attempt.json',dict(identity=identity,time=time.time(),model_calls=0))
    tick=time.perf_counter();plt=H._mpl();fig=None
    try:
        q=t['request'];K=int(q['K']);bound=q['gallery_bounds'];depth=q['gallery_depth_bounds']
        inp=arrays(q['inference_npz']);co=inp['C_vertices'][inp['C_faces_original']]
        fig=plt.figure(figsize=(26.75,13.8),dpi=140,facecolor='white')
        grid=fig.add_gridspec(4,5,left=.025,right=.99,bottom=.07,top=.90,wspace=.18,hspace=.32)
        sections=[];face_rows=[];cells=[]
        for col,(name,label,stage,path) in enumerate(ss):
            a=arrays(path);tri=np.asarray(a['output']).reshape(-1,3,3)
            if not np.isfinite(tri).all() or len(tri)<K:raise ValueError('Invalid stage geometry')
            ids=np.asarray(a.get('parent_source_raw_slot',a.get('source_face_ids',np.arange(len(tri)))),np.int64)
            role=np.asarray(a.get('face_role',['C']*K+['BULK']*(len(tri)-K))).astype('<U6')
            if len(ids)!=len(tri) or len(role)!=len(tri):raise ValueError('Wrong role/provenance domain')
            if set(role)-{'C','BULK','BRIDGE'}:raise ValueError('Native gallery cannot introduce SEAM')
            if not np.array_equal(tri[:K],co):
                # RAW retains the source model order; unordered coordinates must match.
                if stage!='RAW' or any(sorted(map(tuple,x))!=sorted(map(tuple,y)) for x,y in zip(tri[:K],co)):raise ValueError('C geometry changed')
            g=dict(C=tri[:K],F=tri[K:],C_original=co,tri=tri)
            facts=dict(cell_id=t['request_id']+'__'+name,column=col,label=label,stage=stage,path=str(resolve(path)),faces=len(tri),views=[])
            for fi,(face,sid) in enumerate(zip(tri,ids)):
                row=dict(request_id=t['request_id'],column=name,stage=stage,output_face_id=fi,raw_slot_id=int(sid),role=str(role[fi]))
                for vi in range(3):
                    for ci,axis in enumerate('xyz'):row[f'v{vi}_{axis}']=float(face[vi,ci])
                face_rows.append(row)
            for viewrow,az in enumerate((-60,120)):
                raster,scope=G.role_raster(tri,role,K,D.basis(az,23),np.asarray(depth['center']),depth['span'])
                ax=fig.add_subplot(grid[viewrow,col]);ax.imshow(raster);ax.set_axis_off()
                ax.set_title(name+f' | az={az}\nN={len(tri)} / free={len(tri)-K}',fontsize=9)
                facts['views'].append(dict(azimuth=az,**scope))
            for viewrow,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[viewrow,col]);H.draw_section(ax,g,[],bound,L,view,1)
                axis,proj=(2,(0,1)) if view=='front' else (0,(2,1))
                pos=bound['front_positions' if view=='front' else 'side_positions'][1]
                for kind,part,offset in (('C',tri[:K],0),('FREE',tri[K:],K)):
                    segments,localids=H.finite_sections(part,axis,pos,1e-12*L)
                    for segment,li in zip(segments,localids):
                        fi=int(li)+offset;mid=segment[:,proj].mean(0)
                        tag=('bridge'+str(fi)) if ids[fi]<0 else ('C' if kind=='C' else 'f')+str(int(ids[fi]))
                        if kind=='FREE':ax.text(mid[0],mid[1],tag,fontsize=5,color='#187448' if role[fi]=='BRIDGE' else '#303640',
                            zorder=8,bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=.3))
                        sections.append(dict(request_id=t['request_id'],column=name,stage=stage,view=view,plane_axis=axis,
                            plane_position=float(pos),output_face_id=fi,raw_slot_id=int(ids[fi]),role=str(role[fi]),
                            segment_start_x=float(segment[0,0]),segment_start_y=float(segment[0,1]),segment_start_z=float(segment[0,2]),
                            segment_end_x=float(segment[1,0]),segment_end_y=float(segment[1,1]),segment_end_z=float(segment[1,2])))
                ax.set_title(name+f' | {view}={pos:.5f}',fontsize=9)
            cells.append(facts)
        fig.suptitle(t['request_id']+' | '+q['eval_group']+'\nFrozen Coverage RAW: official D0 / GEO-only / shape-preservation candidate\n'
            +'Blue C / gray free / green existing BRIDGE; same inherited cameras and central section positions',fontsize=11)
        fig.text(.5,.025,'The central sections may miss end legs; absence of a cut segment is not evidence that a support is absent.\n'
            +'Surface views and sections must be reviewed together. Section endpoints alone do not certify cracks. Face IDs are in the accompanying CSV.',ha='center',fontsize=8)
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=140);image=dest/'comparison.png'
        with image.open('xb') as f:f.write(buffer.getvalue())
        csvwrite(dest/'sections_face_ids.csv',sections);csvwrite(dest/'face_catalog.csv',face_rows)
        r=dict(status='RENDERED_PENDING_ACTUAL_REVIEW',identity=identity,request_id=t['request_id'],parent_id=q['parent_id'],
            condition_id=q['condition_id'],noise_index=q['noise_index'],eval_group=q['eval_group'],
            path=str(image),image_sha256=sha(image),cells=cells,seconds=time.perf_counter()-tick,
            model_calls=0,scoring_calls=0,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
        if 'torch' in sys.modules:raise RuntimeError('CPU renderer imported torch')
        write_new(done,r);print('RENDERED',q['request_id'],round(r['seconds'],2),flush=True);return r
    except BaseException as exc:
        write_new(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:
        if fig is not None:plt.close(fig)

def run(manifest_path=None):
    manifest_path=resolve(manifest_path or OLD/'manifest.json');m,tasks=tasks_from(manifest_path);done=OUT/'gallery/COMPLETE.json'
    if done.exists():
        r=read(done);verify(r['files']);return r
    for t in tasks:
        loc=locations(t)
        for p in (loc['geometry']/'complete.json',loc['cpu']/'complete.json'):
            r=read(p)
            if r['status']!='COMPLETE':raise RuntimeError('CPU geometry/cascade barrier incomplete')
            verify(r['files'])
    tick=time.perf_counter();mh=sha(manifest_path);receipts=[one(t,mh) for t in tasks]
    body='<meta charset="utf-8"><h1>Bench backend shape preservation: complete frozen30</h1><p>Real mesh rasterizations; paired fixed views and cuts. RAW / official D0 FINAL / GEO-only / candidate BASE / candidate FINAL. Actual five-axis AI_PROXY review is separate.</p>'
    pending=[]
    for r in receipts:
        rel=Path(r['path']).relative_to(OUT/'gallery').as_posix()
        body+='<h2>'+html.escape(r['request_id'])+'</h2><a href="'+rel+'"><img style="width:100%" loading="lazy" src="'+rel+'"></a>'
        for cell in r['cells']:
            pending.append(dict(request_id=r['request_id'],parent_id=r['parent_id'],condition_id=r['condition_id'],noise_index=r['noise_index'],
                eval_group=r['eval_group'],cell_id=cell['cell_id'],column=cell['column'],label=cell['label'],stage=cell['stage'],
                image_path=r['path'],image_sha256=r['image_sha256'],reviewer=None,reviewer_kind='AI_PROXY',blind=False,
                actually_viewed=False,actual_view_source=None,viewed_at_utc=None,**{key:None for key in AXES},note=None))
    index=OUT/'gallery/index.html'
    if index.exists():
        if index.read_text(encoding='utf-8')!=body:raise ValueError('Gallery index changed')
    else:
        with index.open('x',encoding='utf-8') as f:f.write(body)
    write_new(OUT/'gallery/review_manifest.json',dict(status='PENDING_TEMPLATES_NOT_OBSERVATIONS',pages=30,cells=150,
        manifest_sha256=mh,axes=AXES,protocol='Review every cell using two surface views and two original sections. Never infer ratings from metric values or copy old labels. Source style is not unique success. Central cuts may miss supports.',cells_to_review=pending))
    r=dict(status='RENDERED_PENDING_ACTUAL_REVIEW',pages=30,cells=150,workers=1,images=receipts,
        seconds=time.perf_counter()-tick,model_calls=0,scoring_calls=0,failures=0,
        files={str(p):sha(p) for p in (OUT/'gallery').rglob('*') if p.is_file()})
    write_new(done,r);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(OLD/'manifest.json'))
    a=p.parse_args();r=run(a.manifest);print(json.dumps({k:v for k,v in r.items() if k not in ('images','files')}))
