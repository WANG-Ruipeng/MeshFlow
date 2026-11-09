"""Read-only evidence validation and isolated final archive; never dispatch work.

Default --check reports missing stages without writing. --finalize requires all
registered outputs and all three real-review groups, then writes a new dated
archive. No project modules, neural frameworks, scorers or renderers are imported.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'experiments/r2_chair_multipart_seam_v1'
CODE=REPO/'r2/pilots/chair_multipart_seam_v1'
GALLERY='S2000__MULTIPART_S1500_same_pipeline_roles_v2'
GROUPS={'seat_part_01':12,'seat_part_02':16,'BACK_FULL':16}
DOCUMENTS=(OUT/'REPORT.md',REPO/'experiments/r2_unified_rescue_post_v1/REPORT.md',
           REPO/'experiments/r2_unified_rescue_post_v1/report_ready.json',
           *sorted((REPO/'experiments/r2_unified_rescue_post_v1').glob('report_revision_*.json')))

def resolve(path):
    text=str(path)
    if text.startswith('/mnt/e/MeshFlow/'):
        return REPO/text[len('/mnt/e/MeshFlow/'):]
    if text.lower().startswith('e:\\meshflow\\'):
        return REPO/Path(text[12:].replace('\\','/'))
    p=Path(text)
    return p if p.is_absolute() else REPO/p

def sha(path):
    h=hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(resolve(path).read_text(encoding='utf-8-sig'))
def jsonl(path):return [json.loads(x) for x in resolve(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

class Check:
    def __init__(self):self.files={};self.missing=[];self.errors=[];self.stages={}
    def need(self,path):
        p=resolve(path)
        if not p.is_file():self.missing.append(str(p));return None
        self.files[str(p)]=sha(p)
        return read(p)
    def verify(self,files):
        for path,expected in files.items():
            p=resolve(path)
            if not p.is_file():self.missing.append(str(p));continue
            actual=sha(p);self.files[str(p)]=actual
            if actual!=expected:self.errors.append('HASH_MISMATCH '+str(p))
    def require(self,truth,message):
        if not truth:self.errors.append(message)

def collect():
    c=Check();manifest=c.need(OUT/'eval_manifest.json')
    for p in DOCUMENTS:
        if p.is_file():c.files[str(p)]=sha(p)
        else:c.missing.append(str(p))
    step1ready=c.need(REPO/'experiments/r2_unified_rescue_post_v1/report_ready.json')
    if step1ready and DOCUMENTS[1].is_file():
        c.require(step1ready['report_sha256']==sha(DOCUMENTS[1]),'Step1 latest report_ready does not bind current report')
    if DOCUMENTS[0].is_file() and 'BACK16页实际图审摘要将在最终收口补入' in DOCUMENTS[0].read_text(encoding='utf-8-sig'):
        c.missing.append('ROOT_REPORT_FINALIZATION_PENDING')
    requests=manifest['requests'] if manifest else []
    ids={q['request_id'] for q in requests}
    c.require(len(requests)==len(ids)==44,'Expected 44 unique fixed requests')
    reg=c.need(OUT/'training_registration.json');status=c.need(OUT/'training/status.json')
    if reg:c.verify(reg['training_code_hashes'])
    if status:
        c.require((status['status'],status['local_step'],status['global_step'],status['head_step'])==('COMPLETE',1500,15500,1500),'Training endpoint is not fixed local1500/global15500')
        c.stages['training']=dict(status=status['status'],updates=status['local_step'],elapsed_seconds=status.get('elapsed_seconds'),checkpoint=status['checkpoint'])
        # The already-completed CPU payload validation is authoritative; this
        # collector does not load tensor payloads or repeat the model audit.
    summaries=[]
    for p in sorted((OUT/'training_summary').glob('*/completion.json')):
        r=read(p)
        if r.get('final') and r.get('local_step')==1500:summaries.append(p)
    train_account={};calls=[]
    if summaries:
        folder=summaries[-1].parent;c.need(summaries[-1]);train_account=c.need(folder/'accounting.json')
        payload=c.need(folder/'final_checkpoint_check.json')
        if payload:
            c.require(payload['status']=='PASS','Final checkpoint CPU audit not PASS')
            c.verify({payload['path']:payload['sha256']})
        source=c.need(folder/'source_hashes.json')
        if source:c.verify({x['path']:x['sha256'] for x in source})
        c.stages['training_accounting']=train_account
    else:c.missing.append(str(OUT/'training_summary/<final1500>/completion.json'))
    callpath=OUT/'training/calls.jsonl'
    if callpath.is_file():
        calls=jsonl(callpath);c.files[str(callpath)]=sha(callpath)
        begin=collections.Counter(r['kind'] for r in calls if r['event']=='CALL_BEGIN')
        done=collections.Counter(r['kind'] for r in calls if r['event']=='CALL_DONE')
        expected={k:12000 for k in ('train_forward','train_backward','head_forward','head_backward')}
        expected.update(optimizer_update=1500,head_optimizer_update=1500)
        c.require(dict(begin)==dict(done)==expected,'Call ledger does not match 1500 fixed updates')
        c.require(all(r['event'] in ('CALL_BEGIN','CALL_DONE') for r in calls),'Unexpected failed/partial training calls')
    data=c.need(OUT/'data/completion.json')
    if data:c.require(data['summary']['conditions']==78 and data['summary']['samples']==12000 and data['summary']['DEV_parent_leakage']==0,'Prepared data budget/identity differs')
    inf=c.need(OUT/'inference_acceptance.json')
    if inf:
        c.require((inf['status'],inf['requests'],inf['conditions'],inf['generator_forwards'],inf['head_forwards'],inf['D0_forwards'])==('PASS',44,15,2200,15,44),'Inference budget/acceptance mismatch')
        c.require({r['request_id'] for r in inf['rows']}==ids,'Inference requests differ')
        c.require(inf['all_RAW_before_D0'] and inf['extra_training_updates']==0 and inf['extra_model_forwards']==0,'Inference extra calls or ordering mismatch')
        c.stages['inference']={k:inf[k] for k in ('status','requests','conditions','generator_forwards','head_forwards','D0_forwards','all_RAW_before_D0')}
    score=c.need(OUT/'evaluation/completions/RAW__BASE_POST.json')
    if score:
        c.require(score['status']=='COMPLETE' and score['rows']==88,'Expected 88 original scores');c.verify(score['files'])
        for p in score['files']:
            r=c.need(p)
            if r:c.verify(r['files'])
        c.stages['original_scores']={k:score[k] for k in ('status','rows','workers','seconds')}
    aggregate=c.need(OUT/'evaluation/tables/completion.json')
    if aggregate:c.verify(aggregate['files']);c.require(aggregate['rows']==88,'Original aggregate rows mismatch')
    unified=c.need(OUT/'unified/completion.json')
    if unified:
        c.require(unified['status']=='COMPLETE' and unified['requests']==44,'Expected 44 unified cells');c.verify(unified['files'])
        seen=set()
        for p in unified['files']:
            r=c.need(p)
            if r:
                seen.add(r['request_id']);c.verify(r['files'])
                c.require(r['model_calls']==r['GPU_calls']==r['new_D0']==0 and r['BASE_rescored'] is False,'Unified unexpectedly ran neural/base work')
        c.require(seen==ids,'Unified request IDs differ')
        c.stages['unified']={k:unified[k] for k in ('status','requests','workers','seconds','source_execution_revision')}
    compare=c.need(OUT/'comparison/completion.json')
    if compare:
        c.require(compare['status']=='COMPLETE' and compare['requests']==44 and compare['per_output_rows']==396,'Comparison must preserve 396 rows')
        c.verify(compare['files']);c.verify(compare['source_files'])
        c.stages['comparison']={k:compare[k] for k in ('status','requests','per_output_rows','metrics')}
    layout=c.need(OUT/'layout_diagnostic/prediction_completion.json')
    if layout:
        c.require(layout['status']=='COMPLETE' and len(layout['conditions'])==15,'Layout diagnostics incomplete')
        c.verify(layout['files']);c.verify(layout['source_receipts'])
    numerical=c.need(OUT/'analysis/numerical_summary.json')
    if numerical:
        c.require(numerical['status']=='COMPLETE_CACHE_ONLY' and numerical['requests']==44,'Numerical summary incomplete')
        c.verify(numerical['files'])
        if compare:c.require(numerical['source_comparison_sha256']==sha(OUT/'comparison/completion.json'),'Numerical comparison identity mismatch')
    gallery=c.need(OUT/'gallery'/GALLERY/'completion.json')
    if gallery:
        c.require(gallery['requests']==44 and {r['request_id'] for r in gallery['images']}==ids,'Gallery incomplete or different IDs')
        for r in gallery['images']:c.verify({r['path']:r['image_sha256']})
        c.stages['gallery']={k:gallery[k] for k in ('status','requests','workers','wall_seconds')}
    viewed=[]
    for group,count in GROUPS.items():
        rr=c.need(OUT/'visual_review'/group/'completion.json')
        if not rr:continue
        c.require(rr['status']=='COMPLETE_ACTUAL_AI_PROXY_REVIEW' and rr['requests']==count,'Actual review group incomplete '+group)
        c.require(rr['evaluator']=='AI_PROXY' and rr['nonblind'] and rr['topology_by_image']=='UNKNOWN','Review evidence boundary invalid '+group)
        c.verify(rr['files']);rv=c.need(OUT/'visual_review'/group/'review.json')
        if rv:viewed.extend(r['request_id'] for r in rv['requests'])
        c.stages['review_'+group]={k:rr[k] for k in ('requests','stage_observations','old_new_comparisons','processing_comparisons','evaluator','nonblind','topology_by_image')}
    if len(viewed)==44:c.require(len(set(viewed))==44 and set(viewed)==ids,'Actual review union has duplicates/missing IDs')
    dispatch=[]
    for p in sorted((OUT/'dispatch').glob('*/complete.json')):
        r=c.need(p);dispatch.append(dict(path=str(p),**r))
    cpu_checks=c.need(OUT/'training_audit/cpu_trainer_checks.json')
    compute=dict(formal_training=train_account.get('completed',{}),inference=c.stages.get('inference',{}),
        recorded_full_score_outputs=dict(original_RAW_and_BASE=(score or {}).get('rows'),
            unified_GUARD_and_FINAL=2*unified['requests'] if unified else None,
            candidate_intersection_checks='NA_NOT_SEPARATELY_COUNTED; retained per-cell candidate ledgers'),
        data_CPU_validation={k:data.get(k) for k in ('OT_calls','head_forwards','head_backwards','training_updates','generator_forwards','denoiser_forwards','GPU_jobs')} if data else None,
        trainer_CPU_checks=cpu_checks,
        totals_including_data_head_checks=dict(
            generator_forwards=train_account.get('completed',{}).get('train_forward',0)+(inf or {}).get('generator_forwards',0)+(data or {}).get('generator_forwards',0),
            head_forwards=train_account.get('completed',{}).get('head_forward',0)+(inf or {}).get('head_forwards',0)+(data or {}).get('head_forwards',0),
            denoiser_forwards=(inf or {}).get('D0_forwards',0)+(data or {}).get('denoiser_forwards',0),
            generator_backwards=train_account.get('completed',{}).get('train_backward',0),
            head_backwards=train_account.get('completed',{}).get('head_backward',0)+(data or {}).get('head_backwards',0),
            generator_updates=train_account.get('completed',{}).get('optimizer_update',0),head_updates=train_account.get('completed',{}).get('head_optimizer_update',0)),
        collector_model_calls=0,collector_scoring_calls=0,collector_geometry_calls=0,
        scope='Formal Step2 plus recorded 4 CPU head fixture forwards/1 backward; inherited Step1 and historical training excluded. CPU trainer fixture receipt retained separately.')
    timing=dict(training_elapsed_seconds=status.get('elapsed_seconds') if status else None,
        formal_call_elapsed_seconds=(max(r['time_unix'] for r in calls)-min(r['time_unix'] for r in calls)) if calls else None,
        dispatch_shards=dispatch,original_score_seconds=score.get('seconds') if score else None,
        unified_wall_seconds=unified.get('seconds') if unified else None,gallery_wall_seconds=gallery.get('wall_seconds') if gallery else None,
        interpretation='Shard wall seconds are overlapping process timings, not summed experiment wall time or isolated throughput. Unrecorded phase timing remains NA.')
    return c,compute,timing

def archive(c,compute,timing):
    assert not c.missing and not c.errors,'Closeout requires complete verified caches and actual reviews'
    assert all(sha(p)==c.files[str(p)] for p in DOCUMENTS),'Report changed during closeout validation'
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    dest=OUT/'closeout'/stamp;dest.mkdir(parents=True,exist_ok=False)
    sources={p.resolve() for p in CODE.glob('*.py')}
    # Freeze actual code mentioned by scientific/execution registrations without
    # importing these modules. Historical versions remain in their old archives.
    def source_paths(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if isinstance(k,str) and k.endswith('.py') and resolve(k).is_file():sources.add(resolve(k))
                source_paths(v)
        elif isinstance(x,list):
            for v in x:source_paths(v)
    for p in (OUT/'training_registration.json',OUT/'inference_registration.json',OUT/'evaluation/score_registration.json',OUT/'unified/registration.json'):
        source_paths(read(p))
    snapshots=[];diff=[]
    git=['git','-c','safe.directory='+str(REPO)]
    head=subprocess.run([*git,'rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
    for p in sorted(sources):
        rel=p.relative_to(REPO);out=dest/'code'/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(p.read_bytes())
        snapshots.append(dict(source=str(p),snapshot=str(out),sha256=sha(p)))
        prior=subprocess.run([*git,'show','HEAD:'+rel.as_posix()],cwd=REPO,capture_output=True)
        old=prior.stdout.decode('utf-8',errors='replace') if prior.returncode==0 else ''
        new=p.read_text(encoding='utf-8-sig')
        diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+rel.as_posix() if prior.returncode==0 else '/dev/null',tofile='b/'+rel.as_posix()))
    (dest/'code.diff').write_text(''.join(diff),encoding='utf-8')
    documents=[]
    for p in DOCUMENTS:
        out=dest/'documents'/p.relative_to(REPO);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(p.read_bytes())
        documents.append(dict(source=str(p),snapshot=str(out),sha256=sha(p)))
    write(dest/'document_hashes.json',documents)
    write(dest/'source_hashes.json',dict(git_HEAD=head,diff_base='HEAD, including full additions for files absent from HEAD',snapshots=snapshots))
    write(dest/'evidence_hashes.json',c.files);write(dest/'compute_accounting.json',compute);write(dest/'phase_timing.json',timing)
    instructions='''# Evidence and recovery index

This directory verifies completed Step2 outputs. It does not select a new default, rerun jobs, publish, or alter historical evidence.

- Full live optimizer checkpoint: `../../training/checkpoint_step1500.pt` (local1500/global15500); generator/head exports: `../../inference/`.
- Training curves and CPU payload audit: `../../training_summary/`.
- Original 44 fixed requests and noise identities: `../../eval_manifest.json`.
- RAW and single-D0 BASE: `../../evaluation/MULTIPART_S1500/`; original 88 scores: `../../evaluation/scores/`.
- Unified GUARD_GEO/FINAL and per-cell ledger: `../../unified/cells/`.
- All 396 old/new metric rows and paired tables: `../../comparison/`.
- Numerical interpretation and layout diagnostics: `../../analysis/`, `../../layout_diagnostic/`.
- All 44 galleries and three actual AI_PROXY review groups: `../../gallery/`, `../../visual_review/`.
- Source snapshots, SHA256, HEAD-relative code.diff, actual calls and phase timing are local to this archive.

Read-only status: `python -B -m r2.pilots.chair_multipart_seam_v1.resume`.
Read-only complete-stage check: `python -B -m r2.pilots.chair_multipart_seam_v1.closeout --check`.
New immutable closeout: `python -B -m r2.pilots.chair_multipart_seam_v1.closeout --finalize`.

No stage is automatically run on missing evidence. Explicit resume requires the existing frozen entrypoint and its ownership/failure checks; unresolved neural attempts must never be silently repeated. Do not retrain completed step1500. Gallery review is nonblind AI_PROXY; exact topology is UNKNOWN from images. New recipe combines training pool, head and additional updates, so matched output comparisons do not isolate one change. Source owner labels are diagnostic only.
'''
    (dest/'INDEX_AND_RECOVERY.md').write_text(instructions,encoding='utf-8')
    result=dict(status='COMPLETE_EVIDENCE_CLOSEOUT',stages=c.stages,verified_evidence_files=len(c.files),source_files=len(snapshots),
        actual_review_requests=44,actual_review_stage_observations=264,model_calls=0,scoring_calls=0,geometry_calls=0,
        no_default_change=True,no_git_write=True,files={str(p):sha(p) for p in dest.rglob('*') if p.is_file()})
    write(dest/'completion.json',result)
    return dict(path=str(dest),status=result['status'],verified_evidence_files=len(c.files),source_files=len(snapshots))

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--finalize',action='store_true');args=p.parse_args()
    c,compute,timing=collect()
    result=dict(status='READY' if not c.missing and not c.errors else 'PENDING' if not c.errors else 'FAILED',missing=c.missing,errors=c.errors,stages=c.stages,read_only=True,model_calls=0,scoring_calls=0)
    if args.finalize:result=archive(c,compute,timing)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if c.errors:sys.exit(2)

if __name__=='__main__':main()
