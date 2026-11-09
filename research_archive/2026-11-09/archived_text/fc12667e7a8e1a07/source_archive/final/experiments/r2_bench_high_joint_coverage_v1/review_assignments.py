"""Freeze disjoint review ownership; this does not generate labels or view images."""
import json
from r2.pilots.bench_high_joint_coverage_v1.common import OUT,read,write_new,sha
ASSIGNMENTS={
    'high':('NEW_WIDE_TRAIN','HIGH_TRAIN','HIGH_HOLDOUT'),
    'old':('OLD',),
    'regression':('FOCUS','DEV','HOLDOUT'),
}
plan=read(OUT/'gallery/manifest.json');all_ids=[]
for owner,groups in ASSIGNMENTS.items():
    pages=[p for p in plan['pages'] if p['report_group'] in groups]
    ids=[p['page_id'] for p in pages];all_ids+=ids
    write_new(OUT/'visual_review/assignments'/(owner+'.json'),dict(
        schema='DISJOINT_ACTUAL_AI_PROXY_REVIEW_ASSIGNMENT_V1',owner=owner,groups=list(groups),
        gallery_manifest_sha256=sha(OUT/'gallery/manifest.json'),assigned_page_ids=ids,
        pages=[dict(page_id=p['page_id'],path=str(OUT/'gallery'/(p['page_id']+'.png')),
                    request_id=p['request_id'],group=p['report_group']) for p in pages],
        labels=['B','W'],reviewer_kind='AI_PROXY',blind=False,status='PENDING_ACTUAL_VIEW',
        policy='Actually view all two-view/two-section pages. Five axes separately, concrete note for each model; uncertain is allowed. No copying historical labels or deriving labels from scores.',
        ratings_generated=0,model_calls=0))
assert len(all_ids)==len(set(all_ids))==138
print(json.dumps(dict(pages=138,cells=276,ratings_generated=0,assignments={k:len([p for p in plan['pages'] if p['report_group'] in v]) for k,v in ASSIGNMENTS.items()})))
