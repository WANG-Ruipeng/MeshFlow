"""Seal completed evidence; later invocations verify only, never run experiments."""
import argparse
import datetime
import difflib
import json
import subprocess
import sys
from . import eval_common as C


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deep',action='store_true');args=parser.parse_args()
    root=C.OUT;done=root/'completion.json'
    if done.exists():
        record=C.read(done)
        for p,h in record['files'].items():
            if C.sha(C.ROOT/p)!=h:raise ValueError('Changed sealed artifact: '+p)
        if args.deep:
            subprocess.run([sys.executable,'-B','-m','r2.pilots.seam_quality_four_v1.final_audit'],check=True,cwd=C.ROOT)
        print(json.dumps(dict(status='COMPLETE_VERIFIED',files=len(record['files']),deep=args.deep,
            model_calls=0,new_scores=0,updates=0)));return
    checks={
        'execution':root/'audit/final_execution.json',
        'training':root/'training_analysis/final/completion.json',
        'visual':root/'evaluation/visual_review_receipt.json',
        'scores':root/'evaluation/score_completion.json',
        'render':root/'evaluation/gallery/render_completion.json',
        'anti_flatten':root/'antiflatten/FINAL_STATUS.json',
        'redirection':root/'redirection/completion.json'}
    for key,p in checks.items():
        r=C.read(p)
        if r['status'] not in ('PASS','COMPLETE','COMPLETE_PENDING_ACTUAL_REVIEW'):
            raise ValueError((key,r['status']))
    visual=C.read(checks['visual'])
    if visual['actual_view_record_count']!=128 or visual['page_count']!=32:
        raise ValueError('Actual review incomplete')
    source=sorted(p for p in (C.ROOT/'r2/pilots/seam_quality_four_v1').rglob('*') if p.suffix in ('.py','.md'))
    hashes={str(p.relative_to(C.ROOT)):C.sha(p) for p in source}
    C.write(root/'source_hashes.json',hashes)
    with (root/'code.diff').open('x',encoding='utf-8') as f:
        for p in source:
            f.writelines(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(keepends=True),
                fromfile='/dev/null',tofile=str(p.relative_to(C.ROOT)),n=3))
    artifacts=list(checks.values())+source+[root/'REPORT.md',root/'PROTOCOL.md',root/'registration.json',
        root/'eval_manifest.json',root/'report_tables.json',root/'source_hashes.json',root/'code.diff',
        root/'execution_freeze.json',root/'training_execution_freeze.json',root/'evaluation/summary_completion.json',
        root/'training_analysis/historical_curve_definition_audit.json',root/'training_analysis/time_bin_window_summary.csv']
    for folder in ('evaluation','evaluation/reviews','evaluation/gallery','training_analysis/final','audit'):
        artifacts += [p for p in (root/folder).iterdir() if p.is_file() and p.suffix in ('.csv','.md','.png','.json','.html')]
    files={str(p.relative_to(C.ROOT)):C.sha(p) for p in sorted(set(artifacts))}
    C.write(done,dict(status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        fixed_budget_finished=True,training_updates=2400,endpoint_RAW=48,endpoint_D0=48,new_endpoint_scores=96,
        CPU_antiflatten_POST=16,CPU_redirection_POST=16,actual_visual_pages=32,actual_visual_model_records=128,
        default_changed=False,commit_or_push=False,cleanup=False,model_work_stopped=True,files=files,
        weight_hashes_and_all_inputs='audit/final_execution.json verified_files; training status/checkpoint receipts',
        recovery='python -B -m r2.pilots.seam_quality_four_v1.closeout [--deep]',
        limits='Four seen TRAIN conditions, reused noises; bounded findings, no DEV or stable-upgrade claim'))
    print(json.dumps(dict(status='COMPLETE',sealed_files=len(files))))


if __name__=='__main__':main()
