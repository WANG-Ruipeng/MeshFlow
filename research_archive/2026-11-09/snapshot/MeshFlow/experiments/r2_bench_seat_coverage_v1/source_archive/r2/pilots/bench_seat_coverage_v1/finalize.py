"""Seal completed evidence and record final resources; never starts experiment jobs."""
from datetime import datetime, timezone
import os, re
import psutil
from .common import ROOT, OUT, read, sha, write_new, resources

def run():
    if (OUT/'COMPLETE.json').exists():
        old=read(OUT/'COMPLETE.json')
        for path,h in old['artifacts_sha256'].items():
            assert sha(path)==h, path
        return old
    audits=[(p,read(p)) for p in sorted((OUT/'audits').glob('closeout_check_*.json'))]
    audit_path,audit=next((p,r) for p,r in reversed(audits) if r['status']=='AUDIT_PASS')
    assert read(OUT/'results/report_tables_v1/completion.json')['status']=='COMPLETE'
    assert read(OUT/'results/existing_geo_deletion_ledger/completion.json')['status']=='COMPLETE'
    assert read(OUT/'training/NATIVE/status.json')['status']=='COMPLETE'
    assert read(OUT/'workflow/pipeline/status.json')['status']=='COMPLETE'
    archive=read(OUT/'source_archive/receipt.json')
    for relative,h in archive['sources'].items():
        assert sha(ROOT/relative)==h, relative
    for path,h in archive['archived_files'].items():
        assert sha(path)==h, path
    live=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            if p.pid==os.getpid() or 'python' not in (p.info['name'] or '').lower(): continue
            args=p.info['cmdline'] or []
            if any(x.startswith('r2.pilots.bench_seat_coverage_v1.') and x.rsplit('.',1)[-1] in ('train','sample','post','controller','run','evaluate','gallery') for x in args):
                live.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    assert not live, live
    final_resources=OUT/'audits/final_resources.json'
    write_new(final_resources,dict(status='EXPERIMENT_WORKERS_EXITED',live_experiment_python_workers=live,
        resources=resources(),attribution='Other user processes are not stopped or attributed to this experiment.'))
    required=['REPORT.md','PROTOCOL.md','DATA_REPORT.md','RUNBOOK.md','budget.json',
        'data/manifest.json','data/schedule.json','data/cpu_gate.json','data/root_input_review.json',
        'training_registration.json','eval_manifest.json','training/NATIVE/status.json',
        'training/NATIVE/checkpoint_step1500.json','training/NATIVE/generator_step1500.json',
        'results/completion.json','results/per_output.csv','results/paired_per_parent.csv',
        'results/report_tables_v1/completion.json','results/learning_curve.png',
        'results/existing_geo_deletion_ledger/completion.json','gallery/completion.json','gallery/index.html',
        'visual_review/summary_v1/completion.json','visual_review/summary_v1/actual_observations.csv',
        'visual_review/summary_v1/historical_label_changes.csv','visual_review/root_supplementary_notes.json',
        'source_archive/receipt.json','source_archive/code.diff','audits/numeric_crosscheck_v1/receipt.json',
        'audits/numeric_crosscheck_v1/READOUT.md','audits/final_resources.json']
    paths=[OUT/p for p in required]+[audit_path]
    # All local links in the report must resolve before completion.
    links=re.findall(r'\]\(([^)]+)\)',(OUT/'REPORT.md').read_text('utf-8'))
    for s in links:
        if '://' not in s: assert (OUT/s).exists(), s
    weights=[]
    for name in ('checkpoint_step1500','generator_step1500'):
        p=OUT/'training/NATIVE'/(name+'.pt'); side=read(p.with_suffix('.json'))
        recorded=audit['verified_file_sha256'][str(p.resolve())]
        assert recorded==side['sha256'] and p.stat().st_size==side['bytes']
        weights.append(dict(path=str(p),bytes=p.stat().st_size,sha256=recorded,
            verification='Full SHA256 from completed final audit, final stat and sidecar cross-check; no duplicate large-file hash'))
    result=dict(status='COMPLETE',experiment='R2_BENCH_SEAT_COVERAGE_V1',completed_utc=datetime.now(timezone.utc).isoformat(),
        decision='KEEP_PRIOR_SEAM3000_REFERENCE; ARCHIVE_LOCAL_RAW_GAINS_AND_POST_SUPPORT_LOSS; DO_NOT_PROMOTE_COVERAGE_ENDPOINT',
        updates=1500,train_forwards=12000,train_backwards=12000,head_calls=0,probe_calls=0,replayed_updates=0,
        new_RAW=36,generator_sampling_forwards=1800,new_D0=36,reused_outputs=48,total_scoring_rows=252,
        new_scoring_rows=108,reused_scoring_rows=144,numeric_fields=177,actual_visual_cells=252,gallery_pages=90,
        weights=weights,audit_status='AUDIT_PASS',audit_receipt=str(audit_path),
        artifacts_sha256={str(p):sha(p) for p in paths},
        limits=['Two densely exposed new training parents; one held-out parent',
                'No equal-budget OLD-only continuation control','Nonblind AI_PROXY, historical label changes retained',
                'Source T-junction Native admission does not certify SEAM compatibility'],
        default_changed=False,commit=False,push=False,old_assets_deleted=False,automatic_next_training=False)
    write_new(OUT/'COMPLETE.json',result)
    return result
if __name__=='__main__':
    print({k:v for k,v in run().items() if k not in ('artifacts_sha256','weights','limits')})
