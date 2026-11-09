import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import numpy as np, argparse, concurrent.futures, multiprocessing, html
from r2.pilots.chair_multipart_seam_v1 import gallery as G
C,D,H=G.C,G.D,G.H

def one(q):
    rid=q['request_id'];dest=OUT/'gallery';png=dest/(rid+'.png')
    if png.with_suffix('.json').exists():return read(png.with_suffix('.json'))
    frame=read(q['frame_json']);bounds=frame['bounds'];depth=frame['depth'];k=q['K']
    core=arrays(q['inference_npz']);c=core['C_vertices'][core['C_faces_original']]
    plt=H._mpl();fig=plt.figure(figsize=(18,12),dpi=115,facecolor='white')
    grid=fig.add_gridspec(4,4,top=.92,bottom=.06,hspace=.28,wspace=.18);cells=[]
    for col,combo in enumerate(COMBOS):
        p=raw_path(combo,rid);mesh=arrays(p);tri,role=G.roles(mesh,k)
        g=dict(C=tri[:k],F=tri[k:],C_original=c,tri=tri);scopes=[]
        for row,az in enumerate((-60,120)):
            raster,scope=G.role_raster(tri,role,k,D.basis(az,23),np.asarray(depth['center']),depth['span'])
            ax=fig.add_subplot(grid[row,col]);ax.imshow(raster);ax.axis('off');ax.set_title(combo+' RAW | '+str(az),fontsize=10);scopes.append(scope)
        for row,view in ((2,'front'),(3,'side')):
            ax=fig.add_subplot(grid[row,col]);H.draw_section(ax,g,np.flatnonzero(role[k:]=='SEAM'),bounds,C.L,view,1);ax.set_title(combo+' RAW | '+view,fontsize=10)
        cells.append(dict(combo=combo,raw=str(p),sha256=sha(p),scopes=scopes))
    fig.suptitle(rid+'\n'+q['split']+' '+q['part_type']+f' | K={k} N={q["N"]} | C=COMMON U=USER; prefix / suffix at t=0.5',fontsize=10)
    fig.text(.5,.02,'Blue input C / orange seam / gray bulk. All RAW, zero D0. Inherited fixed cameras & section positions. Section endpoints alone do not certify cracks.',ha='center',fontsize=9)
    dest.mkdir(parents=True,exist_ok=True);fig.savefig(png);plt.close(fig)
    rec=dict(request_id=rid,split=q['split'],part=q['part_type'],path=str(png),sha256=sha(png),bounds=bounds,depth=depth,cells=cells)
    save(png.with_suffix('.json'),rec);print('RENDERED',rid,flush=True);return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pages=list(pool.map(one,registration()['requests']))
    save(OUT/'gallery/complete.json',dict(status='RENDERED',pages=16,workers=a.workers,model_calls=0))
    (OUT/'gallery/index.html').write_text('<meta charset="utf-8"><h1>Bench actual-trajectory crossover</h1><p>CC / CU / UC / UU, switch t=0.5. AI_PROXY review is separate.</p>'+''.join('<p>'+html.escape(r['request_id'])+'</p><a href="'+Path(r['path']).name+'"><img loading="lazy" width="100%" src="'+Path(r['path']).name+'"></a>' for r in pages),encoding='utf-8')

if __name__=='__main__':main()
