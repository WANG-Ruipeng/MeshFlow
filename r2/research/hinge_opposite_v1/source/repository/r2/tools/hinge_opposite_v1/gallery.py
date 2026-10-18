"""Four-column CPU renders with the exact inherited camera/section contract."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import html,json
from pathlib import Path
from r2.pilots.seam_followup_v1 import gallery as R
from r2.pilots.seam_hinge_quality_v1.evaluation import dispatch_cpu
from . import run as RUN

C=RUN.C
OUT=RUN.OUT/'gallery'


def run():
    done=OUT/'completion.json'
    if done.exists():
        saved=C.read(done)
        for row in saved['images']:assert C.sha(row['path'])==row['image_sha256']
        for path,sha in C.read(OUT/'definition.json')['mesh_hashes'].items():assert C.sha(path)==sha
        return saved
    qs=RUN.OLD.requests();pages=[]
    for q in qs:
        rid=q['request_id'];cells=[
            dict(label='S2000 POST',path=str(RUN.OLD.paths('PLAIN_S2000',q)['post'])),
            dict(label='old HINGE015',path=str(RUN.OLD.OUT/'cells/PLAIN_S2000_HINGE015'/rid/'post.npz'))]
        for suffix in ('015','025'):
            folder=RUN.OUT/'cells'/('HINGE_OPP'+suffix)/rid
            receipt=C.read(folder/'complete.json');assert receipt['status']=='COMPLETE'
            RUN.B.verify(receipt['files'])
            cells.append(dict(label='opposite HINGE'+suffix,path=str(folder/'post.npz')))
        pages.append(dict(**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','K','N','split')},
            page_id=rid,kind='same original POST, new all-induced-edge winding guard',stage='POST variants',cells=cells))
    source=RUN.OLD.OLD/'gallery/wave1_POST/definition.json';historic=C.read(source)
    sources={str(Path(x.__file__)):C.sha(x.__file__) for x in (R,R.D,R.H)}
    sources[str(Path(__file__))]=C.sha(__file__)
    definition=dict(version='hinge_opposite_same_historical_camera_and_section_v1',
        original_definition_path=str(source),original_definition_sha256=C.sha(source),
        bounds=historic['bounds'],depth_bounds=historic['depth_bounds'],
        mesh_hashes={cell['path']:C.sha(cell['path']) for page in pages for cell in page['cells']},
        source_hashes=sources,columns=['original_POST','old_HINGE015','new_OPP015','new_OPP025'],
        rows=['exterior_azimuth_minus60_elevation23','exterior_azimuth120_elevation23','original_front_finite_section','original_side_finite_section'],
        scale='Exactly inherited centers, spans, section plane coordinates; no per-output refitting',
        images_are_not_topology='Finite sections can miss off-plane connections; only actual viewing may be labelled AI_PROXY',
        clipping='Display only; all source meshes remain read-only',model_calls=0,GPU_calls=0)
    path=OUT/'definition.json';C.write(path,definition);C.write(OUT/'manifest.json',dict(pages=pages))
    jobs=[(p,str(OUT),C.sha(path),historic['bounds'][p['condition_id']],historic['depth_bounds'][p['condition_id']]) for p in pages]
    images,workers,seconds=dispatch_cpu(jobs,R.one,OUT,workers=2)
    images=sorted(images,key=lambda r:r['request_id'])
    text='<meta charset="utf-8"><h1>S2000 HINGE opposite guard</h1><p>Original POST / old HINGE015 / new OPP015 / new OPP025. Exact original cameras and finite sections; AI_PROXY review is recorded separately.</p>'
    text+=''.join('<h2>'+html.escape(row['request_id'])+'</h2><a href="'+Path(row['path']).name+'"><img loading="lazy" style="width:100%" src="'+Path(row['path']).name+'"></a>' for row in images)
    (OUT/'index.html').write_text(text,encoding='utf-8')
    result=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',images=images,workers=workers,wall_seconds=seconds,
        definition_sha256=C.sha(path),model_calls=0,GPU_calls=0)
    C.write(done,result)
    return result


if __name__=='__main__':
    r=run();print(json.dumps(dict(status=r['status'],images=len(r['images']),seconds=r['wall_seconds'])),flush=True)
