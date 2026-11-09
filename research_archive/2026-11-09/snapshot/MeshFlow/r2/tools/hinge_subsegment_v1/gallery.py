"""CPU exterior/section views of all requests with a registered split trial."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import html
import json
import time
from pathlib import Path
from r2.pilots.seam_followup_v1 import gallery as R
from . import census as S

C = S.C
OUT = S.OUT / 'gallery'


def run():
    done = OUT / 'completion.json'
    if done.exists():
        saved = C.read(done)
        for image in saved['images']:
            assert C.sha(image['path']) == image['image_sha256']
        return saved
    trials = C.read(S.OUT / 'trials/all_trials.json')
    ids = {r['request_id'] for r in trials}
    pages = []
    for q in S.O.OLD.requests():
        rid = q['request_id']
        if rid not in ids:
            continue
        cells = [dict(label='baseline HINGE 1.5%', path=str(S.paths(q)['mesh']))]
        for stage in ('SPLIT015', 'SPLIT025'):
            root = S.OUT / 'trials' / stage / rid
            rec = C.read(root / 'complete.json')
            cells.append(dict(label=stage + f" (accepted {rec['accepted']})", path=str(root / 'post.npz')))
        pages.append(dict(**{key:q[key] for key in ('request_id','condition_id','parent_id','noise_index','K','N','split')},
                          page_id=rid, kind='registered trials; final outputs only, rejected trials rolled back',
                          stage='CPU seam-side split', cells=cells))
    source = S.O.OUT / 'gallery/definition.json'
    old = C.read(source)
    definition = dict(version='subsegment_same_previous_camera_and_section_v1',
                      previous_definition_path=str(source), previous_definition_sha256=C.sha(source),
                      bounds=old['bounds'], depth_bounds=old['depth_bounds'],
                      mesh_hashes={cell['path']:C.sha(cell['path']) for page in pages for cell in page['cells']},
                      code_sha256=C.sha(__file__), actual_review_required=True,
                      no_new_camera_fit=True, GPU_calls=0, model_calls=0)
    C.write(OUT / 'definition.json', definition)
    C.write(OUT / 'manifest.json', dict(pages=pages))
    C.resources(OUT / 'resources_before.json')
    start = time.perf_counter()
    # Only four small figures, one CPU thread; independent audits may overlap.
    images = [R.one((p, str(OUT), C.sha(OUT/'definition.json'),
                     old['bounds'][p['condition_id']], old['depth_bounds'][p['condition_id']])) for p in pages]
    text = '<meta charset="utf-8"><h1>Finite subsegment split test</h1><p>Baseline HINGE 1.5% / split 1.5% / split 2.5%. Same exterior and section positions. Rejected trials are absent from final meshes.</p>'
    text += ''.join('<h2>'+html.escape(r['request_id'])+'</h2><img style="width:100%" src="'+Path(r['path']).name+'">' for r in images)
    (OUT/'index.html').write_text(text, encoding='utf-8')
    result = dict(status='RENDERED_PENDING_ACTUAL_VIEW', images=images, workers=1,
                  wall_seconds=time.perf_counter()-start, GPU_calls=0, model_calls=0)
    C.write(done, result)
    print(json.dumps(dict(images=len(images), seconds=result['wall_seconds'])), flush=True)
    return result


if __name__ == '__main__':
    run()
