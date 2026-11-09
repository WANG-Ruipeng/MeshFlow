import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import numpy as np,io,html
from r2.pilots.chair_multipart_seam_v1 import gallery as G
C,D,H=G.C,G.D,G.H

def main():
    dest=OUT/'gallery';dest.mkdir(exist_ok=True);qs=read(OUT/'evaluation_inputs/manifest.json')['requests'];pages=[]
    historic=read(ROOT/'experiments/r2_bench_parts_convergence_v1/gallery/NATIVE_S21000__SEAM_S21000_full/definition.json')
    for q in qs:
        rid=q['request_id'];common=OUT/'evaluation/COMMON'/rid/'raw/raw.npz'
        if not common.exists():continue
        png=dest/(rid+'.png')
        if png.exists():pages.append(read(png.with_suffix('.json')));continue
        source=OUT/'source_reference'/q['condition_id']/'source.npz';src=arrays(source)
        cells=[('SOURCE reference only',source),('COMMON RAW',common)]
        if q['split']=='DEV':
            old=V2/'evaluation/USER_V2_S1500'/rid/'raw';receipt=read(old/'receipt.json')
            assert receipt['checkpoint_sha256']==sha(manifest()['checkpoints']['USER']) and receipt['noise_sha256']==q['noise_sha256']
            assert sha(old/'raw.npz')==receipt['raw_sha256']
            priorlay=arrays(V2/'evaluation/USER_V2_S1500/layouts'/q['condition_id']/'layout.npz');lay=arrays(OUT/'layouts'/q['condition_id']/'layout.npz')
            assert np.array_equal(priorlay['anchor_ids'],lay['anchor_ids'])
            cells.append(('USER RAW (verified cache)',old/'raw.npz'))
        k=q['K'];c=src['output'][:k];f=src['output'][k:];g=dict(C=c,F=f,C_original=c,tri=src['output'])
        if q['split']=='DEV':bounds=historic['bounds'][q['condition_id']];depth=historic['depth_bounds'][q['condition_id']]
        else:
            bounds=H.bounds_for_condition([g],None,C.L);pts=src['output'].reshape(-1,3);center=(pts.min(0)+pts.max(0))/2
            span=max(np.ptp((pts-center)@D.basis(az,23),axis=0)[:2].max() for az in (-60,120))*1.25
            depth=dict(center=center.tolist(),span=float(span))
        plt=H._mpl();fig=plt.figure(figsize=(5*len(cells),12),dpi=100,facecolor='white');grid=fig.add_gridspec(4,len(cells),top=.91,bottom=.06,hspace=.28,wspace=.18)
        infos=[]
        for col,(label,path) in enumerate(cells):
            mesh=arrays(path);tri,role=G.roles(mesh,k);g=dict(C=tri[:k],F=tri[k:],C_original=c,tri=tri);scopes=[]
            for r,az in enumerate((-60,120)):
                raster,scope=G.role_raster(tri,role,k,D.basis(az,23),np.array(depth['center']),depth['span']);ax=fig.add_subplot(grid[r,col]);ax.imshow(raster);ax.axis('off');ax.set_title(label+f' | {az}',fontsize=9);scopes.append(scope)
            for r,view in [(2,'front'),(3,'side')]:
                ax=fig.add_subplot(grid[r,col]);H.draw_section(ax,g,np.flatnonzero(role[k:]=='SEAM'),bounds,C.L,view,1);ax.set_title(label+' | '+view,fontsize=9)
            infos.append(dict(label=label,path=str(path),sha256=sha(path),scopes=scopes))
        fig.suptitle(rid+'\n'+q['split']+' '+q['part_type']+f' | K={k} N={q["N"]}\nBlue C / orange seam / gray bulk. Source is ONE possible completion, not a style target.',fontsize=11)
        fig.text(.5,.02,'PARTIAL DIAGNOSTIC: GPU stopped after12 COMMON outputs; current TRAIN USER generation unavailable. No postprocessing comparison here.',ha='center',fontsize=8)
        fig.savefig(png);plt.close(fig)
        page=dict(request_id=rid,split=q['split'],part=q['part_type'],path=str(png),sha256=sha(png),cells=infos,bounds=bounds,depth=depth,review='PENDING_ACTUAL_AI_PROXY')
        save(png.with_suffix('.json'),page);pages.append(page)
    save(dest/'complete.json',dict(status='RENDERED_PARTIAL',pages=pages,model_calls=0))
    (dest/'index.html').write_text('<meta charset="utf-8"><h1>Bench diagnostic: partial evidence</h1>'+''.join('<p>'+html.escape(p['request_id'])+'</p><img style="width:100%" src="'+Path(p['path']).name+'">' for p in pages),encoding='utf-8')
    print('GALLERY',len(pages))
if __name__=='__main__':main()
