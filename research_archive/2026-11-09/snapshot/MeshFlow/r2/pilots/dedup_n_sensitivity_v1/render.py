"""Four-label count/weight comparison of saved meshes, CPU raster only."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,html
from pathlib import Path
from . import common as C
from . import evaluate as E

ORDER=('ORIGINAL_ORIG_N','ORIGINAL_LOW_N','DEDUP_ORIG_N','DEDUP_LOW_N')

def run():
    E.check_code()
    R=C.module('r2.pilots.full_seat_learning_curve_v1._dedup_n_gallery_private',
        C.ROOT/'r2/pilots/full_seat_learning_curve_v1/render.py')
    R.C=C;R.LABELS=ORDER;R.TRAIN_LABELS=E.TRAIN_LABELS;R.REFERENCES=E.REFERENCES
    R.registration=E.registration;R.verify=E.verify;R.VERSION='dedup_n_sensitivity_cpu_gallery_v1'
    result=R.run(ORDER);rows=[];per_label={label:{q['request_id']:q for q in E.requests(label)} for label in ORDER}
    for page in result['images']:
        for label in ORDER:
            q=per_label[label][page['request_id']]
            for stage in ('RAW','POST'):
                rows.append(dict(request_id=page['request_id'],parent_id=page['parent_id'],split=q['split'],noise=q['candidate'],label=label,stage=stage,
                    N=q['N'],K=q['K'],source_arm=E.SOURCE_ARM[label],reviewer='UNREVIEWED',review_status='UNREVIEWED',
                    C_usage='UNKNOWN',body_complete='UNKNOWN',visible_fragments='UNKNOWN',visible_cracks='UNKNOWN',
                    visible_extra_surfaces='UNKNOWN',visible_join='UNKNOWN',notes='',image_path=page['path'],image_sha256=page['sha256']))
    dest=C.OUT/'evaluation/gallery';C.csvwrite(dest/'review_template.csv',rows)
    title='Fixed weights × original/lower N: preserved C and exact Gaussian prefix'
    prose='<meta charset="utf-8"><h1>'+title+'</h1><p>Four frozen endpoints, same saved C, two noise lengths. N changes the N embedding bin and the number of free slots. Single official D0/GEO_C and source winding for all. Columns are original/high, original/low, dedup/high, dedup/low. Both views and RAW/POST share one scale across the entire C panel.</p>'
    prose+=''.join('<h2>'+html.escape(x['parent_id'][:8]+'_n'+str(x['candidate']))+'</h2><a href="'+Path(x['path']).name+'"><img style="width:100%" src="'+Path(x['path']).name+'" loading="lazy"></a>' for x in result['images'])
    C._write_bytes(dest/'COMPARE.html',prose.encode())
    C.write(dest/'experiment_context.json',dict(version=E.VERSION,labels=ORDER,description=title,training_updates=0,
        queries=24,RAW_POST_separate=True,figure_generation_is_not_review=True,render_adapter_sha256=C.sha(__file__),
        review='Actual NON_BLIND_AI_PROXY or named human; visual role is not historical strict adoption or topological certification.',
        comparison='N decreases use the exact prefix of old free Gaussian arrays, not identical full-dimensional initial states. No newly drawn noise.'))
    return dict(status='COMPLETE',images=len(result['images']),review_rows=len(rows),model_calls=0)

if __name__=='__main__':
    argparse.ArgumentParser().parse_args();print(json.dumps(run()))
