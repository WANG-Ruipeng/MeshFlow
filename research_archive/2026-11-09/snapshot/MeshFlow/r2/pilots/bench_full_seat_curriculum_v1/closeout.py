"""Close the completed bounded experiment, archiving code without model calls."""
from __future__ import annotations
import datetime
import difflib
import json
from pathlib import Path
from . import summarize_results as S
from . import visual_review as V


def main():
    out = S.OUT
    review = V.validate()
    assert review['status'] == 'COMPLETE_ACTUAL_AI_PROXY_REVIEW'
    controller = S.read(out/'controller/status.json')
    assert controller['status'] == 'READY_REVIEW'
    jobs = ('native','seam','raw-common','raw-native','raw-seam','d0-common','d0-native',
            'd0-seam','score-raw','score-post','collect','gallery','pipeline')
    phase_rows = []
    for job in jobs:
        state = S.read(out/'workflow'/job/'status.json')
        assert state['status'] == 'COMPLETE' and state['exit_code'] == 0, job
        phase_rows.append(dict(job=job, **state))
    train = S.read(out/'results/training_summary.json')
    assert train['status'] == 'COMPLETE'
    for stage, steps in zip(train['stages'], (6000,3000)):
        assert stage['completed_prefix_updates'] == steps
        calls = stage['calls']
        assert not any(calls['failed'].values()) and not calls['pending_call_ids']
        assert len(calls['attempt_ids']) == 1
    required = ['REPORT.md','METHODS_AND_LIMITATIONS.md','RUNBOOK.md','PROTOCOL.md',
        'registration.json','training_registration.json','eval_manifest.json',
        'results/completion.json','results/analysis_v1/completion.json',
        'results/training_summary.json','results/PRIMARY_TABLES.md',
        'results/report_tables_receipt.json','results/source_archive/receipt.json',
        'visual_review/summary_v1/completion.json','visual_review/root_supplement.json',
        'visual_review/root_supplement_2.json','visual_review/root_supplement_3.json',
        'gallery/completion.json','gallery/index.html','audits/final_training_9000.json',
        'engineering/all72_raw_cache_audit.json','engineering/all72_d0_cache_audit.json',
        'engineering/collected_evidence_audit.json',
        'engineering/visual_review_iso_fraction_compat.json']
    for path in required:
        assert (out/path).is_file() and (out/path).stat().st_size > 0, path
    score_registration = S.read(out/'evaluation/score_registration.json')
    training_registration = S.read(out/'training_registration.json')
    protected = {**training_registration['training_code_hashes'], **score_registration['source_hashes']}
    sources = []
    for path, expected in sorted(protected.items()):
        p = S.resolve(path)
        assert S.digest(p) == expected, path
        if p.is_relative_to(S.ROOT):
            rel = Path('workspace')/p.relative_to(S.ROOT)
        else:
            # The inherited external scorers are read-only and archived as text.
            parts = p.parts
            assert 'MeshFlow-Control' in parts, path
            rel = Path('MeshFlow-Control').joinpath(*parts[parts.index('MeshFlow-Control')+1:])
        dst = out/'results/final_code_archive'/rel
        S.immutable_bytes(dst, p.read_bytes())
        sources.append(dict(source=str(p), archive=str(dst.relative_to(out)), sha256=expected,
                            role='FROZEN_TRAINING_OR_SCORER'))
    diff = []
    for p in sorted(Path(__file__).parent.glob('*.py')):
        dst = out/'results/final_code_archive/current_pilot'/p.name
        S.immutable_bytes(dst,p.read_bytes())
        sources.append(dict(source=str(p),archive=str(dst.relative_to(out)),sha256=S.digest(p),role='CURRENT_PILOT'))
        diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(keepends=True),
            fromfile='/dev/null',tofile='b/'+str(p.relative_to(S.ROOT))))
    S.immutable_bytes(out/'results/code.diff',''.join(diff).encode())
    S.immutable_bytes(out/'results/final_code_archive/manifest.json',
        (json.dumps(dict(status='COMPLETE',files=sources,
            diff_scope='Additive current pilot source files, not a claim of a single-factor patch against Chair.',
            weights_copied=0,original_files_modified=False),indent=2)+'\n').encode())
    required += ['results/code.diff','results/final_code_archive/manifest.json']
    records = {p:S.digest(out/p) for p in required}
    result = dict(status='COMPLETE',experiment='R2_BENCH_FULL_SEAT_CURRICULUM_V1',
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training=dict(native_updates=6000,seam_updates=3000,total_updates=9000,
            train_forward=72000,train_backward=72000,head_updates=0,extra_probes=0,retries=0),
        evaluation=dict(fixed_requests=24,models=3,raw_outputs=72,sampling_forward=3600,
            C_only_head_forward=12,D0_forward=72,score_rows=216,gallery_pages=72,
            actual_visual_review_cells=216,reviewer_kind='AI_PROXY',blind=False),
        review_validation=review,workflow_phases=phase_rows,artifacts_sha256=records,
        retained_checkpoints='training/NATIVE/checkpoint_step6000.pt and training/SEAM/checkpoint_step3000.pt; actual export loading authenticated separately',
        no_automatic_training_extension=True,default_changed=False,commit=False,push=False,
        weights_deleted=False,completion_does_not_mean_model_promotion=True,
        own_model_calls=0,own_scoring_calls=0)
    S.immutable_bytes(out/'COMPLETE.json',(json.dumps(result,indent=2,ensure_ascii=False)+'\n').encode())
    print(json.dumps(dict(status='COMPLETE',code_files=len(sources),artifacts=len(records),
        training_updates=9000,score_rows=216,actual_visual_review_cells=216)))


if __name__ == '__main__':
    main()
