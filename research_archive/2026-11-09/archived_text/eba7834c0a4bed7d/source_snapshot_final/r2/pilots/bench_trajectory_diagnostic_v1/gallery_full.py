"""Fixed-camera CPU galleries of all completed, saved diagnostic outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import numpy as np,html,argparse,concurrent.futures,multiprocessing
from r2.pilots.chair_multipart_seam_v1 import gallery as G
from r2.pilots.seam_slot_c200_v1.generate import raw_arrays
from r2.pilots.bench_parts_convergence_v1 import data as B
C,D,H=G.C,G.D,G.H

def frame(q):
    src=arrays(OUT/'source_reference'/q['condition_id']/'source.npz');k=q['K'];c=src['output'][:k]
    if q['split']=='DEV':
        old=read(ROOT/'experiments/r2_bench_parts_convergence_v1/gallery/NATIVE_S21000__SEAM_S21000_full/definition.json')
        return old['bounds'][q['condition_id']],old['depth_bounds'][q['condition_id']],c
    g=dict(C=c,F=src['output'][k:],C_original=c,tri=src['output']);bounds=H.bounds_for_condition([g],None,C.L)
    pts=src['output'].reshape(-1,3);center=(pts.min(0)+pts.max(0))/2
    span=max(np.ptp((pts-center)@D.basis(az,23),axis=0)[:2].max() for az in (-60,120))*1.25
    return bounds,dict(center=center.tolist(),span=float(span)),c

def cells(q,kind):
    rid=q['request_id'];k=q['K'];result=[]
    if kind=='main':
        for label in LABELS:
            for stage,p in [('RAW',OUT/'evaluation'/label/rid/'raw/raw.npz'),('FINAL',OUT/'cpu'/label/rid/'CLEAN_FINAL_POST/post.npz')]:
                result.append((label+' '+stage,arrays(p),[p]))
    elif kind in LABELS:
        base=OUT/'evaluation'/kind/rid;raw=arrays(base/'raw/raw.npz');source=arrays(q['source_npz']);lay=arrays(OUT/'layouts'/q['condition_id']/'layout.npz')
        result.append((kind+' RAW',raw,[base/'raw/raw.npz']))
        for stage,key,p in [('D0 proposal (C movable)','official_unconstrained_proposal',base/'post/proposal.npz'),('PRE_GEO (anchors restored)','output',base/'post/pre_geo.npz')]:
            bank=arrays(p);tri=bank[key].copy()
            # Reorder only the C corners for display; no geometry is transformed.
            perm=np.asarray(source['C_source_corner_permutations']) if 'C_source_corner_permutations' in source else None
            if perm is not None:tri[:k]=np.take_along_axis(tri[:k],np.argsort(perm,axis=1)[...,None],axis=1)
            display={**raw,'output':tri};result.append((stage,display,[p]))
        for stage,p in [('GEO',base/'post/post.npz'),('FINAL',OUT/'cpu'/kind/rid/'CLEAN_FINAL_POST/post.npz')]:result.append((stage,arrays(p),[p]))
    elif kind=='oracle':
        p=OUT/'evaluation/USER'/rid/'raw/raw.npz';o=OUT/'execution/USER/fixed'/rid/'oracle_raw.npz'
        src=OUT/'source_reference'/q['condition_id']/'source.npz'
        result=[('SOURCE: one completion',arrays(src),[src]),('USER production layout',arrays(p),[p]),('USER source layout'+(' (same)' if not o.exists() else ''),arrays(o if o.exists() else p),[o if o.exists() else p])]
    elif kind=='continuation':
        s=arrays(q['fixed_sample_npz']);core=arrays(q['inference_npz']);case=arrays(q['case_path']);source=arrays(q['source_npz'])
        lay=B.inference_layout(core['C_vertices'],core['C_faces_original'],core['C_faces_model'],q['N'],case['occupied_labels'].astype(bool));inv=np.argsort(s['face_permutation'])
        for t in manifest()['continuations']:
            p=OUT/'execution/USER/fixed'/rid/f'continuation_t{t}.npz';tr=arrays(p)
            for stage,i in [('START',0),('END',-1)]:
                arr=raw_arrays(tr['states'][i][inv],lay,case['occupied_labels'].astype(np.float32),source)
                result.append((f't={t} '+stage,arr,[p]))
    return result

def one(job):
    q,kind=job;rid=q['request_id'];dest=OUT/'gallery_full'/kind;dest.mkdir(parents=True,exist_ok=True);png=dest/(rid+'.png')
    if png.with_suffix('.json').exists():return read(png.with_suffix('.json'))
    bounds,depth,c=frame(q);cc=cells(q,kind);k=q['K'];plt=H._mpl();fig=plt.figure(figsize=(4.5*len(cc),12),dpi=100,facecolor='white')
    grid=fig.add_gridspec(4,len(cc),top=.92,bottom=.06,hspace=.28,wspace=.18);info=[]
    for col,(label,mesh,paths) in enumerate(cc):
        tri,role=G.roles(mesh,k);g=dict(C=tri[:k],F=tri[k:],C_original=c,tri=tri);scopes=[]
        for r,az in enumerate((-60,120)):
            raster,scope=G.role_raster(tri,role,k,D.basis(az,23),np.array(depth['center']),depth['span']);ax=fig.add_subplot(grid[r,col]);ax.imshow(raster);ax.axis('off');ax.set_title(label+f' | {az}',fontsize=9);scopes.append(scope)
        for r,view in [(2,'front'),(3,'side')]:
            ax=fig.add_subplot(grid[r,col]);H.draw_section(ax,g,np.flatnonzero(role[k:]=='SEAM'),bounds,C.L,view,1);ax.set_title(label+' | '+view,fontsize=9)
        info.append(dict(label=label,files={str(p):sha(p) for p in paths},scopes=scopes))
    fig.suptitle(rid+'\n'+q['split']+' '+q['part_type']+f' | K={k} N={q["N"]} | '+kind,fontsize=10)
    fig.text(.5,.02,'Blue C / orange seam / gray bulk / green bridge. Fixed scale and sections; section endpoints do not prove cracks. Source-informed diagnostics are not deployable generation.',ha='center',fontsize=8)
    fig.savefig(png);plt.close(fig)
    rec=dict(request_id=rid,kind=kind,split=q['split'],part=q['part_type'],path=str(png),sha256=sha(png),bounds=bounds,depth=depth,cells=info)
    save(png.with_suffix('.json'),rec);print('RENDERED',kind,rid,flush=True);return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--kinds',nargs='+',default=['main','COMMON','USER','oracle','continuation']);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    qs=read(OUT/'evaluation_inputs/manifest.json')['requests'];jobs=[(q,k) for k in a.kinds for q in qs if k not in ('oracle','continuation') or q['split']=='CURRENT_TRAIN']
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:pages=list(pool.map(one,jobs))
    dest=OUT/'gallery_full';save(dest/('complete_'+'_'.join(a.kinds)+'.json'),dict(status='RENDERED',pages=len(pages),model_calls=0,workers=a.workers))
    allpages=[read(p) for k in ('main','COMMON','USER','oracle','continuation') for p in sorted((dest/k).glob('*.json'))]
    (dest/'index.html').write_text('<meta charset="utf-8"><h1>Bench trajectory: complete saved evidence</h1><p>Fixed cameras / sections. Nonblind AI_PROXY review recorded separately.</p>'+''.join('<p>'+html.escape(p['kind']+' '+p['request_id'])+'</p><a href="'+p['kind']+'/'+Path(p['path']).name+'"><img loading="lazy" style="width:100%" src="'+p['kind']+'/'+Path(p['path']).name+'"></a>' for p in allpages),encoding='utf-8')
if __name__=='__main__':main()
