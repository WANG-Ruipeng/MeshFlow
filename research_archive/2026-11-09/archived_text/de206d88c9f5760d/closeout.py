"""Read-only evidence authentication after all registered work; no model calls."""
from pathlib import Path
import collections, datetime, json, sys, time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from r2.pilots.bench_exposure_allocation_v1 import evaluation as E

OUT = Path(__file__).resolve().parent

def main():
    if (OUT/'COMPLETE.json').exists():
        done = E.read(OUT/'COMPLETE.json')
        E.verify(done['checkpoint_hashes']); E.verify(done['evidence_hashes'])
        print('COMPLETE: retained artifacts verified; no work repeated.')
        return
    tick = time.perf_counter()
    reg = E.read(OUT/'registration.json')
    E.verify(reg['training_code_hashes'])
    E.verify({reg['source_checkpoint']: reg['source_checkpoint_sha256']})
    E.verify(E.read(OUT/'evaluation/score_registration.json')['source_hashes'])
    p0 = E.read(OUT/'audits/P0_cache_audit.json')
    E.verify(p0['files']); E.verify(p0['protected'])
    totals = collections.Counter()
    checkpoints = {}
    call_files = {}
    for arm in ('A_50_50', 'B_75_25'):
        s = E.read(OUT/'training'/arm/'status.json')
        assert s['calls']['failed'] == {} and s['calls']['pending_call_ids'] == []
        assert s['calls']['successful'] == {'optimizer_update':1500, 'train_backward':12000, 'train_forward':12000}
        totals.update(s['calls']['successful'])
        attempts = list((OUT/'training'/arm/'attempts').glob('attempt_*'))
        assert len(attempts) == 1
        cp = OUT/'training'/arm/'calls.jsonl'
        events = [json.loads(line) for line in cp.read_text('utf-8').splitlines() if line.strip()]
        begin = [e for e in events if e['event']=='CALL_BEGIN']
        end = [e for e in events if e['event']=='CALL_DONE']
        assert len(events)==2*25500 and len(begin)==len(end)==25500
        assert len({e['call_id'] for e in begin})==25500
        assert {(e['call_id'],e['kind']) for e in begin} == {(e['call_id'],e['kind']) for e in end}
        assert collections.Counter(e['kind'] for e in end)==s['calls']['successful']
        call_files[str(cp)] = E.sha(cp)
        for kind in ('checkpoint_step1500', 'generator_step1500'):
            r = E.read(OUT/'training'/arm/(kind+'.json'))
            assert r['global_step'] == 21000 and r['local_step'] == 1500
            E.verify({r['path']:r['sha256']}); checkpoints[r['path']] = r['sha256']
    for label in ('A','B'):
        _, receipts = E.verify_raw(label)
        assert len(receipts) == 30
        for q in E.requests():
            raw = E.producer(label,q,'raw'); post = E.producer(label,q,'post')
            rc = raw/'calls.jsonl'; pc = post/'calls.jsonl'
            re = [json.loads(line) for line in rc.read_text('utf-8').splitlines() if line.strip()]
            pe = [json.loads(line) for line in pc.read_text('utf-8').splitlines() if line.strip()]
            assert len(re)==100
            for event in ('FORWARD_STARTED','FORWARD_COMPLETE'):
                selected=[e for e in re if e['event']==event]
                assert len(selected)==50 and {e['step'] for e in selected}==set(range(50))
            assert [e['event'] for e in pe]==['D0_FORWARD_STARTED','D0_FORWARD_RETURNED']
            call_files.update({str(rc):E.sha(rc),str(pc):E.sha(pc)})
    r = E.read(OUT/'results/completion.json')
    assert (r['rows'],r['new_score_rows'],r['reused_score_rows'],len(r['metric_fields'])) == (270,180,90,177)
    E.verify(r['files']); E.verify(r['source_rows'])
    for rel in ('results/analysis_v1/completion.json','visual_review/summary_v1/completion.json','visual_review/intersection_v1/completion.json','gallery/completion.json'):
        a = E.read(OUT/rel); E.verify(a['files'])
        if 'sources' in a: E.verify(a['sources'])
    visual = E.read(OUT/'visual_review/summary_v1/completion.json')
    assert visual['actual_observations'] == 270
    archive = E.read(OUT/'final_source_archive/manifest.json')
    for row in archive['files']:
        E.verify({str(OUT/'final_source_archive'/row['relative_path']):row['sha256']})
    assert (OUT/'REPORT.md').is_file()
    evidence = dict(call_files)
    for folder in ('results','visual_review','training_closeout','data_expansion','audits','dispatch','workflow','gallery','final_source_archive'):
        for p in (OUT/folder).rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.json','.jsonl','.csv','.md','.py','.diff','.txt','.html','.png'):
                evidence[str(p)] = E.sha(p)
    for name in ('REPORT.md','RUNBOOK.md','RECOVERY_README.md','PROTOCOL.md','registration.json','training_registration.json','closeout.py','audit_result_tables.py','derive_visual_intersection.py','package_review.py'):
        evidence[str(OUT/name)] = E.sha(OUT/name)
    result = dict(status='COMPLETE', experiment='R2_BENCH_EXPOSURE_ALLOCATION_V1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_training=dict(totals), new_RAW=60, new_generator_sampling_forwards=3000,
        new_D0_forwards=60, new_score_rows=180, reused_P0_score_rows=90,
        full_numeric_fields=177, gallery_pages=90, actual_AI_PROXY_observations=270,
        scientific_failures=0, replays=0, new_training_ready_parents=0,
        data_status='Two source C/Y candidates; both blocked by N128..256 contract',
        original_Coverage_and_P0_and_registered_code_unchanged=True,
        checkpoint_hashes=checkpoints, evidence_hashes=evidence,
        auto_promotion=False, commit=False, push=False, task_stopped=True,
        audit_model_calls=0, seconds=time.perf_counter()-tick)
    E.immutable(OUT/'COMPLETE.json',result)
    assert 'torch' not in sys.modules
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_hashes','checkpoint_hashes')},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
