"""Same historical cameras and scales; CPU renders for real AI_PROXY review."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,multiprocessing,time,html
from .protocol import ROOT,OUT,WAVE1,WAVE2
from r2.pilots.seam_abc_v1 import eval_common as C
from r2.pilots.seam_abc_v1.render import measured_workers
from r2.pilots.seam_followup_v1 import gallery as old

def mesh(label,rid,stage='POST'):
    if label=='B_LONG_LAYOUT':
        return ROOT/'experiments/r2_seam_followup_v1/layout/evaluation/B_LONG_LAYOUT'/rid/'post/post.npz'
    if stage=='BOUND015':
        wave='wave1' if label in WAVE1 else 'wave2'
        return OUT/'binding'/wave/'cells'/(label+'_BIND_015')/rid/'post.npz'
    return OUT/'evaluation'/label/rid/stage.lower()/(stage.lower()+'.npz')

def manifest(kind):
    p=OUT/'gallery_manifests'/(kind+'.json')
    if p.exists():return C.read(p)
    qs=C.read(OUT/'protocol.json')['evaluation']['requests'];pages=[]
    if kind=='wave1_POST':labels=('B_LONG_LAYOUT',*WAVE1)
    elif kind=='wave2_POST':labels=('PLAIN_S1000',*WAVE2)
    else:labels=()
    for q in qs:
        common={k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','K','N','split')}
        if labels:
            pages.append(dict(**common,page_id=q['request_id']+'_POST',stage='POST',kind=kind,
                cells=[dict(label=l,path=str(mesh(l,q['request_id']))) for l in labels]))
        else:
            # Preregistered critical cases, plus newly changed >=30% success IDs
            # can be appended in an explicit supplementary manifest later.
            if q['request_id'] not in ('2d08a64e_n3','608b46b0_n3','610b1621_n1','610b1621_n2',
                                      '73b369b9_n1','73b369b9_n2','3f741759_n2','85b16941_n1'):continue
            wave=WAVE1 if kind=='wave1_BOUND' else WAVE2
            for l in wave:
                pages.append(dict(**common,page_id=q['request_id']+'_'+l,stage='POST_BOUND015',kind=kind,
                    cells=[dict(label=l+'_'+s,path=str(mesh(l,q['request_id'],s))) for s in ('POST','BOUND015')]))
    result=dict(pages=pages,model_calls=0,selection='All28POST per new model; eight preregistered critical BOUND cases')
    C.write(p,result);return result

def run(kind):
    m=manifest(kind);pages=m['pages'];out=OUT/'gallery'/kind;out.mkdir(parents=True,exist_ok=True)
    done=out/'completion.json'
    if done.exists():
        r=C.read(done)
        for i in r['images']:assert C.sha(i['path'])==i['image_sha256']
        print('COMPLETE_VERIFIED',kind,len(r['images']));return r
    original=ROOT/'experiments/r2_seam_pool47_couple_v1/gallery/training/definition.json'
    prior=C.read(original);bounds=prior['bounds'];db=prior['depth_bounds']
    hashes={cell['path']:C.sha(cell['path']) for page in pages for cell in page['cells']}
    definition=dict(version='seam_quality_wave1_wave2_gallery_v1',code_sha256=C.sha(__file__),
        manifest_sha256=C.sha(OUT/'gallery_manifests'/(kind+'.json')),historical_frame_path=str(original),
        historical_frame_sha256=C.sha(original),bounds=bounds,depth_bounds=db,geometry_hashes=hashes,
        scale='Exact previous pool47 comparison camera centers/spans and section bounds; no per-output rescale',GT_display=False)
    dp=out/'definition.json'
    if dp.exists():assert C.read(dp)==definition
    else:C.write(dp,definition)
    dh=C.sha(dp);jobs=[(p,str(out),dh,bounds[p['condition_id']],db[p['condition_id']]) for p in pages]
    tick=time.perf_counter();first=old.one(jobs[0]);snap=C.resources();workers,policy=measured_workers(snap,first)
    workers=min(workers,4)
    C.write(out/('dispatch_'+str(time.time_ns())+'.json'),dict(resources=snap,policy=policy,actual_workers=workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        images=[first,*list(pool.map(old.one,jobs[1:]))]
    (out/'index.html').write_text('<meta charset="utf-8"><h1>'+html.escape(kind)+'</h1>'+''.join(
        '<h2>'+html.escape(r['page_id'])+'</h2><a href="'+r['page_id']+'.png"><img style="width:100%" src="'+r['page_id']+'.png"></a>' for r in images),encoding='utf-8')
    r=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',images=images,workers=workers,wall_seconds=time.perf_counter()-tick)
    C.write(done,r);print('RENDER_COMPLETE',kind,len(images));return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['wave1_POST','wave2_POST','wave1_BOUND','wave2_BOUND']);run(p.parse_args().kind)
