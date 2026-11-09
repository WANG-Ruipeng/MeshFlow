"""Repair the V2 sampling-time presentation after the controller terminates.

The frozen evaluator writes metadata.seconds. The frozen reporting adapter only
looks for wall_seconds / elapsed_seconds. This standalone CPU presentation fix
reads the actual saved seconds, preserves every original report/table, replaces
only the two sampling-cost display tables, and invokes the unchanged renderer.
It never imports or invokes training, sampling, model, or scoring entrypoints.

Usage: python -B presentation/final_report_review.py --out <V2 output directory>
       python -B presentation/final_report_review.py --out <V2 output directory> --smoke
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

LABELS=('START','H_CONT_END','MOMENT_XY_END')
DEFAULT_OUT=Path(__file__).resolve().parents[1]
REPORT_NAME='CHAIR_MOMENT_CHANNEL_V2_REPORT.md'
COST_NAMES=('sampling_cost_per_output','cost_summary')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def csv_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for row in rows for k in row})
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:'NA' if v is None else json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    os.replace(tmp,path)


def finite_seconds(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0


def measured_seconds(row):
    """Never treat historical cache latency as a fresh model sampling time."""
    if row.get('status')!='SAMPLED':return None,None
    meta=row.get('metadata',row.get('audit',{}))
    for field in ('seconds','wall_seconds','elapsed_seconds'):
        value=meta.get(field)
        if finite_seconds(value):return float(value),'metadata.'+field
    value=row.get('sampling_seconds')
    return (float(value),'generated_row.sampling_seconds') if finite_seconds(value) else (None,None)


def rebuild(original_rows,original_summary,generated):
    index={}
    for label,rows in generated.items():
        for row in rows:
            key=(label,row['case_id'])
            if key in index:raise ValueError('Duplicate generated-row identity: '+str(key))
            index[key]=row
    if {(r['checkpoint_label'],r['case_id']) for r in original_rows}!=set(index):
        raise ValueError('Original sampling-cost table and actual generated records differ')
    corrected=[];changes=[]
    for row in original_rows:
        item=dict(row);key=(row['checkpoint_label'],row['case_id']);source=index[key]
        if source['status']!=row['status']:raise ValueError('Original generation status changed')
        value,field=measured_seconds(source)
        item['sampling_seconds']=value
        # This provenance field is presentation metadata, not a model metric.
        item['sampling_seconds_source']=field
        corrected.append(item)
        changes.append(dict(checkpoint_label=key[0],case_id=key[1],status=row['status'],
            old_sampling_seconds=row.get('sampling_seconds'),corrected_sampling_seconds=value,
            source_field=field,missing_actual_sample_time=source['status']=='SAMPLED' and value is None))
    summaries=[]
    for row in original_summary:
        item=dict(row);label=row['checkpoint_label']
        subset=[r for r in corrected if r['checkpoint_label']==label and r['status']=='SAMPLED']
        values=[r['sampling_seconds'] for r in subset if finite_seconds(r['sampling_seconds'])]
        if row['sampled']!=len(subset):raise ValueError('Original actual sample count changed')
        item.update(sampling_timing_count=len(values),sampling_seconds_mean=statistics.mean(values) if values else None,
            sampling_seconds_median=statistics.median(values) if values else None,
            missing_sampled_seconds=len(subset)-len(values),
            sampling_timing_source='Actual generated_rows metadata.seconds; explicit legacy timing fallback only if present; no cache timing or invented values.')
        summaries.append(item)
    # All unrelated fields are exactly preserved, including quality-independent
    # training summaries, checkpoints, sampled/cache/NA counts, and runtime data.
    for before,after in zip(original_rows,corrected):
        assert {k:v for k,v in after.items() if k not in ('sampling_seconds','sampling_seconds_source')}=={k:v for k,v in before.items() if k!='sampling_seconds'}
    allowed={'sampling_timing_count','sampling_seconds_mean','sampling_seconds_median','missing_sampled_seconds','sampling_timing_source'}
    for before,after in zip(original_summary,summaries):
        assert {k:v for k,v in before.items() if k not in allowed}=={k:v for k,v in after.items() if k not in allowed}
    return corrected,summaries,changes


def smoke(out):
    """Finite synthetic schema check; does not load real score or model files."""
    source={'START':[dict(case_id='cache',status='CACHED',metadata=dict(seconds=999))],
        'H_CONT_END':[dict(case_id='a',status='SAMPLED',metadata=dict(seconds=2.5)),dict(case_id='b',status='SAMPLED',metadata={}),dict(case_id='na',status='NOT_APPLICABLE',metadata={})],
        'MOMENT_XY_END':[dict(case_id='a',status='SAMPLED',metadata=dict(seconds=0.)),dict(case_id='b',status='SAMPLED',metadata=dict(seconds=4.))]}
    old=[dict(checkpoint_label=label,case_id=r['case_id'],status=r['status'],sampling_seconds=None,metadata=r['metadata']) for label,rows in source.items() for r in rows]
    summaries=[dict(checkpoint_label=label,sampled=sum(r['status']=='SAMPLED' for r in rows),cached=sum(r['status']=='CACHED' for r in rows),NA=sum(r['status']=='NOT_APPLICABLE' for r in rows),
        sampling_timing_count=0,sampling_seconds_mean=None,sampling_seconds_median=None,training_summary={'preserve':True}) for label,rows in source.items()]
    corrected,summary,changes=rebuild(old,summaries,source)
    assert summary[0]['sampling_timing_count']==0 and summary[0]['sampling_seconds_mean'] is None
    assert summary[1]['sampling_timing_count']==1 and summary[1]['missing_sampled_seconds']==1
    assert next(r for r in corrected if r['checkpoint_label']=='H_CONT_END' and r['case_id']=='b')['sampling_seconds'] is None
    assert summary[2]['sampling_timing_count']==2 and summary[2]['sampling_seconds_mean']==2. and summary[2]['sampling_seconds_median']==2.
    assert all(r['training_summary']=={'preserve':True} for r in summary)
    result=dict(status='PASS',metadata_seconds_supported=True,zero_seconds_is_valid=True,missing_stays_NA=True,
        caches_excluded=True,unrelated_fields_preserved=True,model_calls=0,scoring_calls=0,report_rendered=False)
    write(Path(out)/'presentation/timing_fix_smoke.json',result);return result


def final_controller_guard(out):
    markers=[out/'RUN_COMPLETE.json',out/'RUN_STOPPED.json']
    if not any(p.exists() for p in markers):raise RuntimeError('Final controller completion marker is absent; wait for finalization')
    running=[]
    for path in (out/'controller').glob('*.json'):
        if read(path).get('status')=='RUNNING':running.append(str(path))
    if running:raise RuntimeError('A controller stage is still RUNNING: '+str(running))


def preserve(out,backup,relative):
    source=out/relative;target=backup/relative
    if not source.exists():raise FileNotFoundError(source)
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    return target


def main(out):
    out=Path(out).resolve();final_controller_guard(out)
    folder=out/'presentation/sampling_timing_correction';folder.mkdir(parents=True,exist_ok=True)
    backup=folder/'originals';paired=out/'paired_results'
    protected=[paired/'per_output.json',paired/'per_parent.json',paired/'primary_summary.json',paired/'results.json',out/'results/summary.json']
    protected_hashes={str(p.relative_to(out)):digest(p) for p in protected}
    generated={};sources={}
    for label in LABELS:
        path=out/'evaluation'/label/'generated_rows.json'
        if path.exists():
            value=read(path);generated[label]=value if isinstance(value,list) else value['rows'];sources[str(path.relative_to(out))]=digest(path)
        else:generated[label]=[]
    originals={}
    for name in COST_NAMES:
        for ext in ('json','csv'):
            relative=Path('paired_results')/(name+'.'+ext);originals[str(relative)]=preserve(out,backup,relative)
    originals[REPORT_NAME]=preserve(out,backup,Path(REPORT_NAME))
    manifest=paired/'artifact_manifest.json'
    if manifest.exists():originals['paired_results/artifact_manifest.json']=preserve(out,backup,Path('paired_results/artifact_manifest.json'))
    old_rows=read(originals['paired_results/sampling_cost_per_output.json']);old_summary=read(originals['paired_results/cost_summary.json'])
    corrected,summary,changes=rebuild(old_rows,old_summary,generated)
    for name,rows in (('sampling_cost_per_output',corrected),('cost_summary',summary)):
        write(folder/(name+'.json'),rows);csv_rows(folder/(name+'.csv'),rows)
        shutil.copyfile(folder/(name+'.json'),paired/(name+'.json'))
        shutil.copyfile(folder/(name+'.csv'),paired/(name+'.csv'))
    write(folder/'sampling_timing_changes.json',changes);csv_rows(folder/'sampling_timing_changes.csv',changes)
    # Import the frozen renderer by its package. Importing its score utilities
    # does not invoke them. No scorer entrypoint or raw geometry computation runs.
    root=Path(__file__).resolve().parents[4]
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    renderer=importlib.import_module('experiments.mf_h1_local0.chair_moment_channel_v2.report')
    package=Path(renderer.__file__).parent
    frozen_before={p.name:digest(p) for p in package.glob('*.py')}
    renderer.render(out,read(paired/'results.json'))
    frozen_after={p.name:digest(p) for p in package.glob('*.py')}
    if frozen_before!=frozen_after:raise AssertionError('Frozen experiment source changed')
    for relative,expected in protected_hashes.items():
        if digest(out/relative)!=expected:raise AssertionError('Scientific result changed: '+relative)
    for relative,expected in sources.items():
        if digest(out/relative)!=expected:raise AssertionError('Generated records changed: '+relative)
    before=originals[REPORT_NAME].read_text(encoding='utf-8');after=(out/REPORT_NAME).read_text(encoding='utf-8')
    report_diff=''.join(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile='original_automatic/'+REPORT_NAME,tofile='corrected_presentation/'+REPORT_NAME))
    (folder/'report.diff').write_text(report_diff,encoding='utf-8')
    code=Path(__file__).read_text(encoding='utf-8')
    (folder/'code.diff').write_text(''.join(difflib.unified_diff([],code.splitlines(keepends=True),fromfile='/dev/null',tofile='b/presentation/final_report_review.py')),encoding='utf-8')
    summary_diff=''.join(difflib.unified_diff(originals['paired_results/cost_summary.json'].read_text(encoding='utf-8').splitlines(keepends=True),
        (paired/'cost_summary.json').read_text(encoding='utf-8').splitlines(keepends=True),fromfile='original_automatic/cost_summary.json',tofile='corrected_presentation/cost_summary.json'))
    (folder/'cost_summary.diff').write_text(summary_diff,encoding='utf-8')
    write(manifest,[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(paired.glob('*')) if p.is_file() and p.name!='artifact_manifest.json'])
    note='''# Sampling timing presentation correction

The frozen evaluator saved each actual rollout duration in `generated_rows.json -> rows[].metadata.seconds`.
The frozen reporting adapter searched `wall_seconds` / `elapsed_seconds`, so its automatic sampling timing count was zero despite saved timing measurements.

This standalone, post-controller CPU script reads the existing actual `metadata.seconds` first, preserves explicit legacy time fields only as a fallback, and leaves missing times as NA. Cached START rows never enter model timing statistics. Counts and summary mean/median use only actual SAMPLED rows with finite nonnegative measured durations, with the measured count and missing count disclosed.

Original report, both cost tables (CSV/JSON), and the original paired artifact manifest are retained byte-for-byte under `originals/`. Corrected tables remain independently available here and replace the corresponding presentation tables under `paired_results/`. The unchanged frozen `report.render` regenerates the main report using those corrected tables. This does not modify raw arrays, generated-record metadata, scoring, grouping, primary endpoints, quality decisions, training math, or frozen source.

`sampling_timing_changes.csv` records every row's original and corrected time and source field. `report.diff` is the actual main-report display change; `cost_summary.diff` is the summary-table display change; `code.diff` records only this newly added independent script, not a patch to frozen experiment code. `review_receipt.json` contains before/after hashes and preservation checks. Global delivery hashes should be refreshed after this script and figure rendering complete.

Sampling durations retain their original `sample_case` measurement scope; they are not an isolated forward benchmark, a per-step NFE estimate, or historical cache latency. No model or scoring calls are made by this correction.
'''
    (folder/'ENGINEERING_CORRECTION.md').write_text(note,encoding='utf-8')
    outputs={str(p.relative_to(out)):digest(p) for p in (out/REPORT_NAME,paired/'cost_summary.json',paired/'cost_summary.csv',paired/'sampling_cost_per_output.json',paired/'sampling_cost_per_output.csv',manifest)}
    value=dict(status='COMPLETE',correction='sampling timing metadata.seconds compatibility',model_calls=0,scoring_calls=0,
        frozen_source_unchanged=True,frozen_source_sha256=frozen_before,protected_result_hashes_unchanged=protected_hashes,
        generated_record_hashes_unchanged=sources,originals={relative:dict(path=str(path.relative_to(out)),sha256=digest(path)) for relative,path in originals.items()},
        corrected_files=outputs,script_sha256=digest(__file__),summary=[{k:r.get(k) for k in ('checkpoint_label','sampled','cached','NA','sampling_timing_count','sampling_seconds_mean','sampling_seconds_median','missing_sampled_seconds')} for r in summary],
        renderer_source_sha256=digest(renderer.__file__),global_delivery_hash_refresh_required=True)
    write(folder/'review_receipt.json',value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=DEFAULT_OUT);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();result=smoke(args.out) if args.smoke else main(args.out)
    print(json.dumps(result,ensure_ascii=False,indent=2))
