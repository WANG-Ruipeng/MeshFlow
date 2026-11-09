"""Validate real budgets/evidence and archive a small reproducible review bundle."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import shutil,difflib,zipfile,sys,json,collections
from .analyze import STAGES,LABELS,scorepath,readcsv

def main():
    final=OUT/'COMPLETE.json'
    if final.exists():
        r=read(final);verify(r['files']);verify(read(OUT/'delivery_hashes.json')['files']);verify(registration()['protected_files'])
        print('COMPLETE_VERIFIED_ZERO_NEW_CALLS');return
    reg=registration();verify(reg['protected_files']);verify(reg['training_code_hashes'])
    er=read(OUT/'evaluation/registration.json');verify(er['source_hashes']);verify(er['files'])
    status=read(OUT/'training'/ARM/'status.json');assert status['status']=='COMPLETE' and status['local_step']==1500
    assert status['calls']['attempted']==status['calls']['successful']=={'optimizer_update':1500,'train_backward':12000,'train_forward':12000}
    assert not status['calls']['failed'] and not status['calls']['pending_call_ids']
    for k in ('checkpoint','inference_export'):
        r=status[k];assert sha(r['path'])==r['sha256']
    training=read(OUT/'analysis/training_integrity.json');assert training['paired_exposures']==12000
    assert training['first_eight_forward_statistics_bit_equal']
    callrows=[dict(phase='training',started=12000,returned=12000,backward_started=12000,
        backward_returned=12000,optimizer_updates=1500,seconds=status['elapsed_seconds'],
        peak_scope='Full formal training including backward and AdamW; device usage includes background',
        device_peak_mib=status['resources']['total_device_used_peak_mib'],
        process_peak_mib=status['resources']['process_peak_mib'],
        process_memory_status=status['resources']['process_memory_status'])]
    for shard in (0,1):
        d=OUT/'execution'/f'raw_shard{shard}';r=read(d/'complete.json');verify(r['files'])
        events=[json.loads(s) for s in (d/'calls.jsonl').read_text().splitlines()]
        starts=[e for e in events if e['stage']=='STARTED'];returns=[e for e in events if e['stage']=='RETURNED']
        assert len(starts)==len(returns)==400
        callrows.append(dict(phase='sampling',shard=shard,started=400,returned=400,seconds=r['seconds'],
            first_start=min(e['time'] for e in starts),last_return=max(e['time'] for e in returns),
            peak_allocated_bytes=r['peak_allocated_bytes'],peak_reserved_bytes=r['peak_reserved_bytes']))
    for q in reg['requests']:
        d=OUT/'evaluation/EARLY'/q['request_id']
        for kind in ('raw','post'):
            r=read(d/kind/'receipt.json');verify(r['files'])
            assert r.get('generator_forwards',50)==50 and r.get('D0_forwards',1)==1
        events=[json.loads(s) for s in (d/'post/calls.jsonl').read_text().splitlines()]
        assert len(events)==2 and 'D0_FORWARD_STARTED' in str(events[0]) and 'D0_FORWARD_RETURNED' in str(events[1])
    post_summary=read(OUT/'evaluation/EARLY/post_complete_full.json')
    assert post_summary['new_D0_forwards']==post_summary['new_GEO_calls']==16
    callrows.append(dict(phase='D0',started=16,returned=16,seconds=post_summary['seconds']))
    for mode in ('raw','post'):verify(read(OUT/'scoring'/mode/'complete.json')['files'])
    summary=read(OUT/'analysis/summary.json');assert summary['status']=='ANALYZED' and summary['scored_rows']==144
    review=readcsv(OUT/'analysis/visual_review.csv');assert len(review)==64
    assert len({r['image'] for r in review})==16 and all(r['reviewer']=='AI_PROXY' for r in review)
    assert (OUT/'REPORT.md').exists() and (OUT/'REPORT.md').stat().st_size>1000
    assert 'torch' not in sys.modules
    csvwrite(OUT/'analysis/call_ledger.csv',callrows)
    sources={**reg['training_code_hashes'],**er['source_hashes']}
    sources.update({str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
    verify(sources);diff=[]
    for path,digest in sorted(sources.items()):
        p=resolve(path)
        if p.is_relative_to(ROOT):rel=p.relative_to(ROOT)
        elif p.is_relative_to(ROOT.parent):rel=Path('external')/p.relative_to(ROOT.parent)
        else:rel=Path('external_absolute')/str(p).lstrip('/').replace(':','')
        target=OUT/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==digest
        else:shutil.copy2(p,target)
        if p.is_relative_to(Path(__file__).parent):
            diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile=str(rel)))
    save(OUT/'source_hashes.json',dict(files=sources,old_sources_modified=False))
    text=''.join(diff);p=OUT/'code.diff'
    if p.exists():assert p.read_text(encoding='utf-8')==text
    else:p.write_text(text,encoding='utf-8')
    # Small original scorer artifacts, no huge pair lists or model tensors.
    evidence=[]
    for q in reg['requests']:
        for label in LABELS:
            for stage in STAGES:
                source=scorepath(label,q,stage);dest=OUT/'score_evidence'/label/q['request_id']/stage
                for name in ('row.json','score.json','valid_edges.csv','seam_bulk_faces.csv','illegal_pairs_by_role.csv'):
                    p=source/name
                    if not p.exists():continue
                    target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():assert sha(target)==sha(p)
                    else:shutil.copy2(p,target)
                    evidence.append(target)
    if not (OUT/'resources_final.json').exists():save(OUT/'resources_final.json',resources())
    save(OUT/'WEIGHTS.json',dict(training_checkpoint=status['checkpoint'],inference_export=status['inference_export'],
        official_and_old_checkpoints_unchanged=True,model_promoted_to_default=False))
    selected=[]
    for folder in ('analysis','gallery','source_snapshot','engineering','score_evidence'):
        selected.extend(p for p in (OUT/folder).rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.png','.html','.py','.md'))
    selected.extend(OUT/p for p in ('REPORT.md','PROTOCOL.md','RUNBOOK.md','registration.json','training_registration.json',
        'cpu_gate.json','resource_initial.json','resources_final.json','WEIGHTS.json','source_hashes.json','code.diff',
        'training_change.diff','frozen_time_weights.csv','visual_review_notes.json','evaluation/registration.json','training_dispatch_go.json'))
    selected.extend(p for p in (OUT/'execution').rglob('*.json') if p.name!='failure.json')
    selected.extend(p for p in (OUT/'evaluation/EARLY').glob('postprocess_attempt_*/resources_*.json'))
    selected.extend(OUT/p for p in ('evaluation/EARLY/post_complete_full.json','evaluation/EARLY/raw_complete_full.json',
        'scoring/raw/complete.json','scoring/post/complete.json','training/EARLY_FM/status.json'))
    selected=sorted(set(selected));assert all(p.is_file() for p in selected)
    save(OUT/'delivery_hashes.json',dict(files={str(p):sha(p) for p in selected}))
    archive=OUT/'BENCH_EARLY_FM_REVIEW.zip'
    if archive.exists():raise RuntimeError('Archive exists without completion; audit before replacing')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for p in selected+[OUT/'delivery_hashes.json']:z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    assert archive.stat().st_size<10_000_000,archive.stat().st_size
    calls=status['calls']
    save(final,dict(status='COMPLETE',experiment=EXPERIMENT,training_updates=1500,training_calls=calls,
        expected_train_forward=12000,expected_train_backward=12000,new_sampling_forwards=800,new_D0=16,new_head=0,
        new_evaluation_noise=0,new_RAW=16,new_POST=16,new_score_rows=48,all_score_rows=144,AI_PROXY_review_pages=16,AI_PROXY_review_cells=64,
        GPU_replay=0,zip_bytes=archive.stat().st_size,no_default_change=True,no_commit_push=True,
        files={str(p):sha(p) for p in (archive,OUT/'delivery_hashes.json',OUT/'source_hashes.json',OUT/'WEIGHTS.json',
            OUT/'analysis/summary.json',OUT/'analysis/training_integrity.json')}))
    print(json.dumps(read(final),ensure_ascii=False))

if __name__=='__main__':main()
