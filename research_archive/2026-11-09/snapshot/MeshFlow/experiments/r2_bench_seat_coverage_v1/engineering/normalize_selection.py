"""Adapt data selection metadata to the evaluation API; no data/noise change."""
from r2.pilots.bench_seat_coverage_v1.common import OUT, SOURCE_ROOT, read, sha, write_new

m=read(OUT/'data/manifest.json')
s=read(OUT/'data/eval_selection.json')
review=read(OUT/'data/root_input_review.json')
assert review['reviewer_kind']=='AI_PROXY' and all(r['actually_viewed'] for r in review['records'])
new=[dict(r,eval_group='NEW_TRAIN',full_seat_review='ACCEPT') for r in m['train'] if r['replay_group']=='NEW']
new += [dict(r,eval_group='NEW_PARENT_HOLDOUT',full_seat_review='ACCEPT') for r in m['dev']]
assert len(new)==3 and {r['condition_id'] for r in new}=={r['condition_id'] for r in s['new_requests']}
for r in new:
    assert all(k in r for k in ('input_path','input_sha256','paired_path','paired_sha256','case_path','case_sha256','boundary_edges'))
write_new(OUT/'eval_selection.json',dict(status='APPROVED_INPUT_REVIEW',schema='EVAL_API_SELECTION_V1',
    data_selection=str(OUT/'data/eval_selection.json'),data_selection_sha256=sha(OUT/'data/eval_selection.json'),
    old_manifest=str(SOURCE_ROOT/'eval_manifest.json'),new_conditions=new,
    history_scope=s['history_scope']))
write_new(OUT/'engineering/selection_api_adapter.json',dict(status='PASS',
    initial_prepare_error='Data builder status/key names differed from evaluation API; failed before writing any panel/noise',
    source_selection_unchanged=True,scientific_selection_unchanged=True,model_calls=0,noise_arrays_created=0))
print({'status':'PASS','new_conditions':len(new)})
