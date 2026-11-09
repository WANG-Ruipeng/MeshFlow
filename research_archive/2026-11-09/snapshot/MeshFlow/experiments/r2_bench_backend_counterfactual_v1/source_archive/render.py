"""CPU raster/finite-section comparison of saved backend stages; no AI images."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,html,io,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from .evaluate import ROOT,OUT,read,resolve,sha,write_new,arrays,csvwrite
from r2.pilots.chair_multipart_seam_v1 import gallery as G
D,H=G.D,G.H
L=float(np.sqrt(3.)*2/.3762)


def mesh_arrays(path,K):
    data=arrays(path)
    key=next((k for k in ('output','official_unconstrained_proposal','denoised','proposal','raw','triangles','full') if k in data),None)
    if key is None:raise ValueError('No triangle array in '+str(path)+': '+str(list(data)))
    tri=np.asarray(data[key]).reshape(-1,3,3)
    if not np.isfinite(tri).all() or len(tri)<K:raise ValueError('Invalid stage geometry')
    ids=np.asarray(data.get('source_face_ids',np.arange(len(tri))),np.int64)
    if ids.shape!=(len(tri),):raise ValueError('Invalid stage face-identity map')
    return tri,ids,key


def stage_specs(task):
    if task['kind']=='SOURCE_GEO':
        return [('SOURCE_C_PLUS_Y',task['new_raw_path']),('SOURCE_GEO',task['new_post_path'])]
    o=task['original']
    return [('RAW',o['raw_path']),('D0_FULL_PROPOSAL',o['proposal_path']),
            ('RESTORED_C_PRE_GEO',o['pre_geo_path']),('BASE_POST',o['base_post_path']),('GEO_RAW',task['new_post_path'])]


def one(task,manifest_sha):
    dest=OUT/'gallery'/task['job_id'];done=dest/'complete.json'
    specs=stage_specs(task)
    identity=dict(job_id=task['job_id'],manifest_sha256=manifest_sha,
        sources={str(resolve(p)):sha(p) for _,p in specs},render_source_sha256=sha(__file__))
    if done.exists():
        result=read(done)
        if result['identity']!=identity:raise ValueError('Saved rendering identity changed')
        for p,h in result['files'].items():
            if sha(p)!=h:raise ValueError('Saved rendering changed')
        return result
    if dest.exists():raise RuntimeError('Incomplete rendering attempt; no silent retry: '+str(dest))
    dest.mkdir(parents=True);write_new(dest/'attempt.json',dict(identity=identity,time=time.time(),model_calls=0))
    tick=time.perf_counter();plt=H._mpl();fig=None
    try:
        q=task['request'];K=int(task['K']);bound=q['gallery_bounds'];depth=q['gallery_depth_bounds']
        inp=arrays(q['inference_npz']);co=inp['C_vertices'][inp['C_faces_original']]
        ncols=len(specs)
        fig=plt.figure(figsize=(5.35*ncols,13.8),dpi=140,facecolor='white')
        grid=fig.add_gridspec(4,ncols,left=.025,right=.99,bottom=.07,top=.90,wspace=.18,hspace=.32)
        sections=[];faces=[];cells=[]
        for col,(stage,path) in enumerate(specs):
            tri,ids,key=mesh_arrays(path,K)
            role=np.full(len(tri),'BULK',dtype='<U6');role[:K]='C'
            g=dict(C=tri[:K],F=tri[K:],C_original=co,tri=tri)
            facts=dict(stage=stage,path=str(resolve(path)),array_key=key,faces=len(tri),views=[])
            for fi,(face,sid) in enumerate(zip(tri,ids)):
                record=dict(job_id=task['job_id'],stage=stage,output_face_id=fi,raw_slot_id=int(sid),role='C' if fi<K else 'FREE')
                for vi in range(3):
                    for ci,axis in enumerate('xyz'):record[f'v{vi}_{axis}']=float(face[vi,ci])
                faces.append(record)
            for row,az in enumerate((-60,120)):
                raster,scope=G.role_raster(tri,role,K,D.basis(az,23),np.asarray(depth['center']),depth['span'])
                ax=fig.add_subplot(grid[row,col]);ax.imshow(raster);ax.set_axis_off()
                ax.set_title(stage+f' | az={az}\nN={len(tri)} / free={len(tri)-K}',fontsize=9)
                if stage=='D0_FULL_PROPOSAL':ax.text(.02,.02,'Blue is proposed C (not preserved)',fontsize=7,color='#126b91',transform=ax.transAxes)
                facts['views'].append(dict(azimuth=az,**scope))
            for row,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[row,col]);H.draw_section(ax,g,[],bound,L,view,1)
                axis,project=(2,(0,1)) if view=='front' else (0,(2,1))
                pos=bound['front_positions' if view=='front' else 'side_positions'][1]
                for kind,part,offset in (('C',tri[:K],0),('FREE',tri[K:],K)):
                    segments,localids=H.finite_sections(part,axis,pos,1e-12*L)
                    for segment,localid in zip(segments,localids):
                        fi=int(localid)+offset;mid=segment[:,project].mean(0)
                        label=('C' if kind=='C' else 'f')+str(int(ids[fi]))
                        # Every intersected free face has a readable identifier
                        # in the accompanying CSV, including overlapping labels.
                        if kind=='FREE':
                            ax.text(mid[0],mid[1],label,fontsize=5,color='#303640',zorder=8,
                                bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=.3))
                        sections.append(dict(job_id=task['job_id'],stage=stage,view=view,plane_axis=axis,plane_position=float(pos),
                            output_face_id=fi,raw_slot_id=int(ids[fi]),role=kind,label=label,
                            segment_start_x=float(segment[0,0]),segment_start_y=float(segment[0,1]),segment_start_z=float(segment[0,2]),
                            segment_end_x=float(segment[1,0]),segment_end_y=float(segment[1,1]),segment_end_z=float(segment[1,2])))
                ax.set_title(stage+f' | {view}={pos:.5f}',fontsize=9)
            cells.append(facts)
        fig.suptitle(task['label']+' | '+task['request_id']+'\n'
            +'Saved CPU backend counterfactual | fixed cameras, source frame and original central section planes\n'
            +'Blue C / gray free; D0 full-proposal column temporarily contains moved proposed C',fontsize=11)
        fig.text(.5,.025,'Section IDs f<RAW-slot> are listed in sections_face_ids.csv and face_catalog.csv. Cut line ends alone do not certify a physical crack.\n'
            +'Central cuts retain inherited C-local bounds; full surface views show the complete source-fixed frame. Source pages are references, not a required generated style.',ha='center',fontsize=8)
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=140)
        image_path=dest/'comparison.png'
        with image_path.open('xb') as f:f.write(buffer.getvalue())
        csvwrite(dest/'sections_face_ids.csv',sections);csvwrite(dest/'face_catalog.csv',faces)
        result=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_VIEW',identity=identity,job_id=task['job_id'],
            path=str(image_path),image_sha256=sha(image_path),cells=cells,section_rows=len(sections),
            face_catalog_rows=len(faces),seconds=time.perf_counter()-tick,model_calls=0,scoring_calls=0,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
        if 'torch' in sys.modules:raise RuntimeError('CPU renderer imported torch')
        write_new(done,result);print('RENDERED',task['job_id'],round(result['seconds'],2),flush=True)
        return result
    except BaseException as exc:
        write_new(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False))
        raise
    finally:
        if fig is not None:plt.close(fig)


def run(manifest_path):
    manifest_path=resolve(manifest_path);m=read(manifest_path)
    if len(m['tasks'])!=10:raise ValueError('Expected 8 output and 2 source pages')
    tick=time.perf_counter();receipts=[one(t,sha(manifest_path)) for t in m['tasks']]
    target=OUT/'gallery/index.html'
    if not target.exists():
        body='<meta charset="utf-8"><h1>Saved backend comparison: RAW versus D0/GEO</h1><p>Actual geometry rasterization and finite sections; no model calls. Fixed inherited cameras and planes. Source references are separate from generated outputs.</p>'
        for r in receipts:
            relative=Path(r['path']).relative_to(OUT/'gallery').as_posix()
            base=relative.rsplit('/',1)[0]
            body+='<h2>'+html.escape(r['job_id'])+'</h2><a href="'+base+'/sections_face_ids.csv">Section face IDs</a> · <a href="'+base+'/face_catalog.csv">All face coordinates</a><br><a href="'+relative+'"><img style="width:100%" loading="lazy" src="'+relative+'"></a>'
        target.write_text(body,encoding='utf-8')
    result=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_VIEW',pages=len(receipts),workers=1,images=receipts,
        seconds=time.perf_counter()-tick,model_calls=0,scoring_calls=0,failures=0)
    done=OUT/'gallery/COMPLETE.json'
    if not done.exists():write_new(done,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    a=p.parse_args();r=run(a.manifest);print(json.dumps({k:v for k,v in r.items() if k!='images'}))
