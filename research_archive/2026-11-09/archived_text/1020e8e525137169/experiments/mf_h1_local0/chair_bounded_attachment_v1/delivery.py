"""Close only this registered experiment; no model or geometry calls."""
from pathlib import Path
import argparse
import csv
import difflib
import json
import shutil
import subprocess
import time
from .common import *


def _csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def run(out=OUT):
    from . import protocol,metrics_adapter
    from .ledger import Budget
    out=Path(out);tick=time.perf_counter();protocol.verify(out)
    accounting=Budget(out).snapshot()
    if accounting['pending'] or accounting['failed']:raise RuntimeError('Unresolved actual calls')
    assert accounting['training_updates']==accounting['model_backward']==accounting['optimizer_updates']==0
    requests=read_json(out/'requests.json');gate=read_json(out/'phase1_gate.json')
    scores=read_json(out/'scoring/per_output.json');by_score={r['request_id']:r for r in scores}
    assets=[];counts={label:0 for label in LABELS};missing=[]
    for request in requests:
        path=result_path(out,request);r=read_json(path)
        if r['status'].startswith('NOT_RUN'):
            assert request['label']=='TRAJECTORY' and gate['status']!='PASS'
            assert r['status']=='NOT_RUN_NO_PHASE1_SIGNAL'
            assert not r.get('raw_path')
            missing.append(request['request_id']);continue
        assert r['status'] in ('CACHED','PROJECTED','SAMPLED')
        for pkey,hkey in [('raw_path','raw_sha256'),('mesh_path','mesh_sha256')]:
            p=resolve(r[pkey]);sha=digest(p)
            if sha!=r[hkey]:raise RuntimeError('Saved output hash changed: '+str(p))
            assets.append(dict(request_id=request['request_id'],label=request['label'],kind=pkey,path=str(p),sha256=sha,bytes=p.stat().st_size))
        if request['label']!='RAW':
            p=resolve(r['intervention_log_path'])
            assert digest(p)==r['intervention_log_sha256']
        assert by_score[request['request_id']]['status']=='PASS'
        counts[request['label']]+=1
    assert counts['RAW']==counts['ENDPOINT']==16
    assert counts['TRAJECTORY']==(16 if gate['status']=='PASS' else 0)
    if gate['status']!='PASS':
        assert len(missing)==16 and accounting['total_model_forwards']==0
    independent=read_json(out/'audit/endpoint_independent.json')
    assert independent['status']=='PASS'
    raw_checks=[read_json(p) for p in sorted((out/'scoring/raw_identity').glob('*.json'))]
    assert len(raw_checks)==16 and all(r['status']=='PASS' for r in raw_checks)
    gallery=read_json(out/'gallery/summary.json')
    assert gallery['status']=='COMPLETE' and gallery['actual_rows']==16 and gallery['rendered_cells']==sum(counts.values())
    _csv(out/'output_asset_index.csv',assets)

    # Recheck only the actual pre-registered input/code dependencies.
    originals=read_json(out/'registration/source_hashes.json')
    expected={str(resolve(p)):h for p,h in originals.items()}
    expected[str(START)]=START_SHA
    for relative,sha in metrics_adapter.source_identity().items():
        if 'chair_bounded_attachment_v1' not in relative:expected[str(resolve(relative))]=sha
    for relative,row in read_json(out/'tests/project/source_freeze.json')['sources'].items():
        if row['role']=='read_only_classifier_dependency':expected[str(resolve(relative))]=row['sha256']
    preserved=[]
    for path,sha in sorted(expected.items()):
        p=Path(path);actual=digest(p)
        preserved.append(dict(path=path,before_sha256=sha,after_sha256=actual,unchanged=sha==actual,bytes=p.stat().st_size))
    protection=dict(status='PASS' if all(r['unchanged'] for r in preserved) else 'FAIL',
        checked_files=len(preserved),scope='Only actual registered dependencies, metric/classifier sources and retained START weight',rows=preserved)
    atomic_json(out/'audit/protected_dependencies.json',protection)
    if protection['status']!='PASS':raise RuntimeError('Protected dependency changed')

    sources={};diff=[]
    for p in sorted(CODE.glob('*.py')):
        relative=p.relative_to(ROOT);target=out/'source_snapshots/new'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        sources[str(relative)]=dict(sha256=digest(p),bytes=p.stat().st_size,snapshot=str(target.relative_to(out)),role='new_experiment')
        lines=p.read_text(encoding='utf-8-sig').splitlines(keepends=True)
        diff.extend(difflib.unified_diff([],lines,fromfile='/dev/null',tofile='b/'+relative.as_posix()))
    dependency_paths={resolve(relative) for relative in metrics_adapter.source_identity() if 'chair_bounded_attachment_v1' not in relative}
    dependency_paths.update(resolve(p) for p in originals if str(p).endswith('.py'))
    for relative,row in read_json(out/'tests/project/source_freeze.json')['sources'].items():
        if row['role']=='read_only_classifier_dependency':dependency_paths.add(resolve(relative))
    for p in sorted(dependency_paths):
        relative=p.relative_to(ROOT);target=out/'source_snapshots/dependencies'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        sources[str(relative)]=dict(sha256=digest(p),bytes=p.stat().st_size,snapshot=str(target.relative_to(out)),role='unchanged_actual_dependency')
    (out/'code.diff').write_text(''.join(diff),encoding='utf-8')
    atomic_json(out/'source_hashes.json',dict(sources=sources,code_diff_sha256=digest(out/'code.diff'),
        note='New additions only. Prior files are byte-identical. Engineering recovery diffs remain under engineering/.'))

    git=subprocess.run(['git','-c','core.autocrlf=true','diff','--stat'],cwd=ROOT,capture_output=True,text=True)
    atomic_json(out/'audit/git_readonly.json',dict(command='git -c core.autocrlf=true diff --stat',
        returncode=git.returncode,stdout=git.stdout,stderr=git.stderr,
        note='Command-local autocrlf matches existing Windows global setting; no git config/index/commit change. Protected SHA256 is byte authority.'))
    entry='experiments.mf_h1_local0.chair_bounded_attachment_v1.execute'
    (out/'README_RESUME.md').write_text(
        '# Fixed bounded attachment pilot\n\n'
        'Main results: [report](BOUNDED_ATTACHMENT_CORRECTION_REPORT.md), [all outputs](per_output_metrics.csv), '
        '[parent pairs](aggregation/paired_parent.csv), [noise pairs](aggregation/paired_noise.csv), '
        '[16-row gallery](gallery/index.html), [mesh index](output_asset_index.csv).\n\n'
        'From E:\\MeshFlow in PowerShell:\n\n```powershell\n'
        'wsl -e /home/warpwang/miniforge3/envs/meshflow/bin/python -B -u -m '+entry+'\n```\n\n'
        'This entry validates frozen identities and reuses completed projection/scoring/rendering outputs. '
        'It never silently replays a failed or unfinished geometric/model call. '
        'Actual calls are in ledger/calls.jsonl and accounting.json; finite CPU fixtures are separately counted in tests/project/attempt1/results.json. '
        'The sixteen original Gaussians are preserved under inputs/inference/.\n\n'
        'The executed phase-1 gate did not authorize phase 2. All 16 TRAJECTORY cells explicitly say NOT_RUN_NO_PHASE1_SIGNAL; '
        'no trajectory mesh or score is fabricated. Do not modify the gate or expand this cohort. '
        'This delivery does not implement or dispatch an unused GPU experiment after a failed gate.\n\n'
        'Old model/raw/scorers/reports were read only. code.diff contains isolated new files. '
        'source_snapshots/dependencies/ includes the actual reused scoring/classifier code. '
        'No training, default switch, commit or push occurred.\n',encoding='utf-8')
    report=out/'BOUNDED_ATTACHMENT_CORRECTION_REPORT.md'
    assert report.is_file() and report.stat().st_size>1000
    excluded={'artifact_manifest.json','final_completion.json','controller_status.json','delivery_summary.json'}
    inventory=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.relative_to(out).as_posix() in excluded:continue
        if '__pycache__' in p.parts:continue
        inventory.append(dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    atomic_json(out/'artifact_manifest.json',dict(files=inventory,excluded_dynamic_files=sorted(excluded),
        source_code_and_report_included=True,model_weights_copied=False))
    value=dict(status='COMPLETE',phase1_gate=gate['status'],phase2_status='NOT_RUN_NO_PHASE1_SIGNAL' if gate['status']!='PASS' else 'EVALUATED',
        output_counts=counts,trajectory_not_run_records=len(missing),actual_model_forwards=accounting['total_model_forwards'],
        actual_endpoint_projections=accounting['attempts'].get('projection_endpoint',0),
        CPU_fixture_configurations=8,CPU_fixture_project_calls=12,
        RAW_original_score_identity_checks=16,protected_dependencies=protection['checked_files'],
        artifact_files=len(inventory),artifact_bytes=sum(r['bytes'] for r in inventory),
        report_sha256=digest(report),artifact_manifest_sha256=digest(out/'artifact_manifest.json'),
        source_hashes_sha256=digest(out/'source_hashes.json'),seconds=time.perf_counter()-tick,
        no_automatic_commit=True,no_default_model_change=True,fixed_task_stopped=True)
    atomic_json(out/'delivery_summary.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));a=p.parse_args()
    print(json.dumps(run(a.out),indent=2))
