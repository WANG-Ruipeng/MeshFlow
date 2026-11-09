"""Review artifacts only. Never call a model, scorer, optimizer or renderer."""
import argparse,csv,difflib,json,shutil,zipfile
from pathlib import Path
from .common import *


def sources(out=OUT):
    out=Path(out);code=Path(__file__).parent;old=code.with_name('chair_control_feedback_auto_v1')
    dest=out/'review_sources';dest.mkdir(exist_ok=True)
    records={};patch=[]
    for path in sorted(code.glob('*.py')):
        target=dest/'experiment'/path.name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(path,target);records[str(target.relative_to(out))]=dict(source=str(path),sha256=digest(path))
        previous=old/path.name
        before=previous.read_text(encoding='utf-8-sig').splitlines(True) if previous.exists() else []
        after=path.read_text(encoding='utf-8-sig').splitlines(True)
        patch.extend(difflib.unified_diff(before,after,fromfile=str(previous.relative_to(ROOT)) if previous.exists() else '/dev/null',
            tofile=str(path.relative_to(ROOT))))
    originals=[ROOT/'experiments/mf_h1_local0/evaluate.py',
        ROOT/'experiments/mf_h1_local0/control_mvp48/eval_scoring.py',
        ROOT/'experiments/mf_h1_local0/chair_boundary_attachment_v1/control_metrics.py',
        ROOT/'experiments/mf_h1_local0/chair_control_usability_quick_v1/scoring.py',
        old/'aggregation.py',old/'evaluation.py',old/'replay.py',old/'loss_balance.py']
    originals.extend(old.glob('*.py'))
    originals.extend(ROOT/'experiments/mf_h1_local0'/name for name in (
        'native_inpainting_metrics.py','native_patch_seams.py','native_patch_metrics.py',
        'guided_native_geometry_audit.py','guided_native_geometry_geometry.py','gpu_guidance_evaluation.py'))
    originals.append(ROOT/'native_t1/runs/edge_factorial_20261001_123000/source_snapshot/experiments/mf_h1_local0/edge_factorial_pilot/edge_metric.py')
    for path in sorted(set(originals)):
        target=dest/'original'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);records[str(target.relative_to(out))]=dict(source=str(path),sha256=digest(path))
    (out/'code.diff').write_text(''.join(patch),encoding='utf-8')
    atomic_json(out/'SOURCE_HASHES.json',records)
    return records


def finalize(out=OUT):
    out=Path(out);sources(out)
    assert read_json(out/'audit/runtime_final.json')['status']=='PASS'
    assert read_json(out/'training_audit/final/summary.json')['status']=='PASS'
    assert (out/'CAPPED500_REPLICATION_GRADIENT_BUDGET_REPORT.md').is_file()
    inventory={}
    include=[]
    exact=['CAPPED500_REPLICATION_GRADIENT_BUDGET_REPORT.md','REPRODUCE.md','LITERATURE_SCOPE.md',
        'protocol.json','hypotheses.json','task_contract.txt','SOURCE_HASHES.json','code.diff','accounting.json',
        'report_tables.md','report_tables_receipt.json',
        'audit/runtime_final.json','audit/checkpoint_manifest.json','audit/checkpoint_manifest.csv',
        'audit/runtime_final_outputs.csv','audit/analysis_contract_review.json','registration/receipt.json','registration/checkpoints.json',
        'registration/noise_identity_audit.json','registration/auxiliary_queue_plan.json',
        'preflight/real_complete.json','preflight/gradient_cpu_tests.json','preflight/gradient_cpu_accounting.json',
        'ledger/gpu_occupancy.jsonl','ledger/orchestrator.jsonl','registration/conditions.json',
        'registration/training_plan.json','registration/source_files.json','registration/historical_cache.json',
        'registration/future_ot_precompute/receipt.json','registration/future_ot_precompute/completion_summary.json']
    for name in exact:
        p=out/name
        if p.exists():include.append(p)
    for folder,patterns in [('recipes',['*.json']),('review_sources',['*.py']),
        ('analysis',['*.csv','*.json']),('training_audit/final',['*.csv','*.json']),
        ('training_diagnostics/final',['*.csv','*.json','*.png']),('report_figures',['*.png','*.svg','*.csv','*.json']),
        ('visuals/reviews',['*.json','*.md']),('engineering',['*.json','*.py','*.diff','*.log']),
        ('preflight',['*.json']),('logs',['*.log']),
        ('manifests',['*.json']),('resources',['*_summary.json']),('failures',['*.json'])]:
        for pattern in patterns:include.extend((out/folder).rglob(pattern))
    for label in LABELS:
        for panel in ('OLD','FRESH'):
            path=out/'evaluation'/label/panel/'per_output_metrics.csv'
            assert path.is_file(), 'Missing required per-output CSV: '+str(path)
            with path.open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.DictReader(handle))
            expected={r['request_id'] for r in read_json(out/'manifests'/(panel+'.json'))['rows']}
            assert len(rows)==16 and {r['request_id'] for r in rows}==expected, str(path)
            include.append(path)
    for panel in ('OLD','FRESH'):
        pointer=read_json(out/'visuals'/('latest_'+panel+'.json'))
        include.append(Path(pointer['contact_sheet']))
    include=sorted(set(p for p in include if p.is_file()))
    for p in include:
        record=dict(bytes=p.stat().st_size,sha256=digest(p))
        if p.suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as handle:
                rows=csv.reader(handle);header=next(rows);record['data_rows']=sum(1 for _ in rows);record['columns']=len(header)
        inventory[str(p.relative_to(out))]=record
    atomic_json(out/'REVIEW_INVENTORY.json',dict(files=inventory,
        full_local_artifacts='Full RAW, weights, full-resolution all-parent gallery, complete call journals remain in this run.',
        included='All registered output CSVs, paired scientific analysis tables, step-gradient/update tables, actual code and audit results; contact sheets.'))
    include.append(out/'REVIEW_INVENTORY.json')
    folder=out/'delivery';folder.mkdir(exist_ok=True);target=folder/'CAPPED500_REPLICATION_GRADIENT_BUDGET_REVIEW.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in include:z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        import hashlib
        for p in include:assert hashlib.sha256(z.read(p.relative_to(out).as_posix())).hexdigest()==digest(p)
    result=dict(status='PASS',path=str(target),bytes=target.stat().st_size,sha256=digest(target),files=len(include),
        under_10MB=target.stat().st_size<10_000_000,raw_and_weights_included=False,full_resolution_gallery_included=False)
    atomic_json(folder/'receipt.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources-only',action='store_true');a=p.parse_args()
    print(json.dumps(dict(status='PASS',files=len(sources())) if a.sources_only else finalize()))
