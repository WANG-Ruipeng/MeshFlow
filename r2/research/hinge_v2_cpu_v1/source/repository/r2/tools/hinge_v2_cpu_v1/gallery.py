"""Fixed original exterior and section views, deterministic CPU renderer."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import concurrent.futures
import multiprocessing
import html
import time
from pathlib import Path
from r2.pilots.seam_followup_v1 import gallery as R
from . import run as S
C,OUT=S.C,S.OUT/'gallery'

def run():
    done=OUT/'completion.json'
    if done.exists():
        r=C.read(done)
        for x in r['images']:assert C.sha(x['path'])==x['image_sha256']
        return r
    OUT.mkdir(parents=True,exist_ok=True);old=C.read(S.O.OUT/'gallery/definition.json');pages=[]
    # Every request, independent of improvement or failure, is displayed.
    for q in S.O.OLD.requests():
        cells=[dict(label='HINGE_OPP015',path=str(S.basepath(q)/'post.npz'))]
        cells += [dict(label=arm,path=str(S.OUT/'cells'/arm/q['request_id']/'post.npz')) for arm in S.ARMS]
        pages.append(dict(**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','K','N','split')},
            page_id=q['request_id'],kind='Original output and four fixed CPU arms',stage='HINGE V2',cells=cells))
    definition=dict(version='hinge_v2_same_previous_exterior_section_v1',previous_definition_path=str(S.O.OUT/'gallery/definition.json'),
        previous_definition_sha256=C.sha(S.O.OUT/'gallery/definition.json'),bounds=old['bounds'],depth_bounds=old['depth_bounds'],
        mesh_hashes={c['path']:C.sha(c['path']) for p in pages for c in p['cells']},code_sha256=C.sha(__file__),GPU_calls=0)
    C.write(OUT/'definition.json',definition);C.write(OUT/'manifest.json',dict(pages=pages));dh=C.sha(OUT/'definition.json')
    jobs=[(p,str(OUT),dh,old['bounds'][p['condition_id']],old['depth_bounds'][p['condition_id']]) for p in pages]
    C.resources(OUT/'resources_before.json');tick=time.perf_counter();first=R.one(jobs[0]);snap=C.resources(OUT/'resources_after_first.json')
    mem=int(next(x.split()[1] for x in snap['memory'].splitlines() if x.startswith('MemAvailable:')))*1024
    peak=first['peak_RSS_KiB']*1024;workers=max(1,min(3,int(max(0,mem-(4<<30))/(peak*1.2)),int(max(1,snap['cpu_logical']-snap['loadavg'][0]-3))))
    C.write(OUT/'dispatch.json',dict(workers=workers,first_peak_RSS_bytes=peak,available_RAM_bytes=mem,reason='CPU raster only; preserve cores for score/audit'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        images=[first,*list(pool.map(R.one,jobs[1:]))]
    page='<meta charset="utf-8"><h1>HINGE V2: frozen exterior and sections</h1><p>All 28 requests. Blue C / gray free. BRIDGE is new geometry; topology tables remain authoritative. No GT reconstruction grading.</p>'
    page+=''.join('<h2>'+html.escape(r['request_id'])+'</h2><a href="'+Path(r['path']).name+'"><img loading="lazy" style="width:100%" src="'+Path(r['path']).name+'"></a>' for r in images)
    (OUT/'index.html').write_text(page,encoding='utf-8')
    result=dict(status='RENDERED_PENDING_ACTUAL_VIEW',images=images,workers=workers,wall_seconds=time.perf_counter()-tick,GPU_calls=0)
    C.write(done,result);return result

if __name__=='__main__':print(run()['status'])
