"""CPU gallery adapter for the fixed target-cleanup panel, without model calls."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json
from pathlib import Path
from . import common as C
from . import evaluate as E

def run():
    E.check_code()
    R=C.module('r2.pilots.full_seat_learning_curve_v1._target_cleanup_gallery_private',
        C.ROOT/'r2/pilots/full_seat_learning_curve_v1/render.py')
    R.C=C;R.LABELS=('FULL_S6000',)+E.TRAIN_LABELS;R.TRAIN_LABELS=E.TRAIN_LABELS;R.REFERENCES=E.REFERENCES
    R.registration=E.registration;R.verify=E.verify;R.VERSION='target_cleanup_c200_cpu_gallery_v1'
    result=R.run(R.LABELS)
    # Original rasterization and common bounds remain unchanged. A separate
    # experiment description prevents inherited gallery prose implying training
    # a learning-curve or independent blind/human review.
    rows=[]
    for page in result['images']:
        q=next(q for q in E.registration()['requests'] if q['request_id']==page['request_id'])
        for label in R.LABELS:
            for stage in ('RAW','POST'):
                rows.append(dict(request_id=page['request_id'],parent_id=page['parent_id'],split=q['split'],noise=q['candidate'],label=label,stage=stage,
                    reviewer='UNREVIEWED',review_status='UNREVIEWED',C_usage='UNKNOWN',body_complete='UNKNOWN',visible_fragments='UNKNOWN',visible_cracks='UNKNOWN',
                    visible_extra_surfaces='UNKNOWN',visible_join='UNKNOWN',notes='',image_path=page['path'],image_sha256=page['sha256']))
    dest=C.OUT/'evaluation/gallery'
    C.csvwrite(dest/'review_template.csv',rows)
    C.write(dest/'experiment_context.json',dict(version=E.VERSION,labels=R.LABELS,
        task='FULL_S6000 versus equal 200-update original/duplicate-cleaned training targets',
        new_training_arms=2,queries=24,RAW_POST_separate=True,
        figure_generation_is_not_review=True,reviewer_required='AI_PROXY or actual named human, never auto-derived from scores',
        rendering_source=C.sha(Path(__file__)),panel_scope='Previously seen six FULL conditions; three historical TRAIN and three historical DEV. Continued-train exposure reported separately.'))
    return dict(status='COMPLETE',images=len(result['images']),review_rows=len(rows),model_calls=0)

if __name__=='__main__':
    argparse.ArgumentParser().parse_args();print(json.dumps(run()))
