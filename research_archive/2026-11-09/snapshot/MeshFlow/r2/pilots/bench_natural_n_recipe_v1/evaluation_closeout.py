"""Read-only CPU evidence audits; never dispatch, load tensors, or score meshes.

preflight records the already-completed input/layout contract before inference.
complete verifies actual call ledgers, original metric values and real reviews.
It writes a bounded evaluation receipt, not a scientific promotion or report.
"""
from __future__ import annotations
import argparse
import ast
import collections
import datetime
import json
import sys
import time
from pathlib import Path
from . import evaluation as E

OUT=E.OUT
AXES={
    'overall_shape':('REASONABLE','INCOMPLETE','UNCERTAIN'),
    'C_use':('PLAUSIBLE','REPLACED_OR_IGNORED','UNCERTAIN'),
    'support_retention':('PRESENT','MISSING_OR_DAMAGED','UNCERTAIN'),
    'visible_fragment_or_gap':('YES','NO_OBVIOUS','UNCERTAIN'),
    'layering_or_fold':('YES','NO_OBVIOUS','UNCERTAIN'),
}


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def preflight():
    dest=OUT/'audits/pre_evaluation_audit.json'
    if dest.exists():
        r=E.read(dest);E.verify(r['files']);return r
    from . import report_results as R
    cache=E.read(OUT/'audits/P0_cache_audit.json')
    contract=E.read(OUT/'audits/eval_cpu_contract.json')
    assert cache['status']==contract['status']=='PASS'
    assert contract['requests']==38 and contract['model_calls']==0 and contract['scoring_calls']==0
    assert contract['manifest_sha256']==E.sha(E.MANIFEST)
    expected=dict(requests=38,new_RAW=84,new_D0=84,generator_forwards=4200,
                  new_score_rows=252,reused_score_rows=90,total_score_rows=342)
    assert E.budget()==expected
    routes=collections.Counter()
    for label in E.LABELS:
        assert len(E.new_requests(label))=={'P0':8,'A':38,'B':38}[label]
        for q in E.requests():
            for stage in E.STAGES:
                cell=R.cell(q,label,stage)
                root=E.OLD if label=='P0' and E.cached_p0(q) else E.OUT
                stored='B' if root==E.OLD else label
                assert str(root) in cell['path'] and '/'+stored+'/' in cell['path']
                routes['REUSED' if root==E.OLD else 'NEW']+=1
    assert routes=={'REUSED':90,'NEW':252}
    assert R.groups()==('OLD','FOCUS','DEV','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT')
    denoiser=E.ROOT/'r2/src/meshflow_control/denoise_model.py'
    text=denoiser.read_text(encoding='utf-8')
    assert '1 <= len(raw) <= 800' in text
    assert 'reshape(1, number, 9)' in text
    actual_functions={n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    for name in ('_validate_raw','predict_proposal'):
        assert '256' not in actual_functions[name]
    files={str(p):E.sha(p) for p in (Path(__file__),Path(E.__file__),Path(R.__file__),
        E.MANIFEST,OUT/'audits/P0_cache_audit.json',OUT/'audits/eval_cpu_contract.json',
        OUT/'gallery/manifest.json',OUT/'visual_review/protocol.json',denoiser)}
    raw_code=E.inference_code_hashes({str(Path(E.__file__)):E.sha(E.__file__)},'RAW')
    d0_code=E.inference_code_hashes({str(Path(E.__file__)):E.sha(E.__file__)},'D0')
    files.update(raw_code);files.update(d0_code)
    result=dict(status='PASS',checked_at_utc=utc(),budget=expected,routing_cells=dict(routes),
        new_requests_per_label={'P0':8,'A':38,'B':38},groups=list(R.groups()),
        P0_old30='Original exposure B request/noise/query/camera and outputs authenticated; never regenerated or rescored',
        high_N=[308,375,384,724],D0_shape_contract='1..800, dynamically reshaped; no N256 restriction in validate/predict',
        D0_evidence='Static source inspection only; this is not a GPU execution or high-N quality guarantee',
        gallery_pages=114,expected_actual_visual_cells=342,
        source_versions=dict(generator_runtime='bench_natural_n_native128_799_v1',
            evaluation=E.VERSION,raw_dependency_files=len(raw_code),D0_dependency_files=len(d0_code)),
        files=files,model_calls=0,scoring_calls=0,tensor_loads=0,new_noise=0)
    E.immutable(dest,result);return result


class Evidence:
    def __init__(self):self.files={};self.cache={}
    def add(self,path,expected=None):
        p=E.resolve(path);stat=p.stat();key=(str(p),stat.st_size,stat.st_mtime_ns)
        digest=self.cache.get(key)
        if digest is None:digest=E.sha(p);self.cache[key]=digest
        if expected is not None and digest!=expected:raise ValueError('Evidence changed: '+str(p))
        self.files[str(p)]=digest
        return digest
    def verify(self,book):
        for path,digest in book.items():self.add(path,digest)
    def read(self,path):self.add(path);return E.read(path)
    def events(self,path):
        self.add(path)
        return [json.loads(line) for line in E.resolve(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _train_audit(book):
    totals=collections.Counter();rows=[]
    for arm in E.ARMS.values():
        folder=OUT/'training'/arm
        status=book.read(folder/'status.json')
        expected={'train_forward':12000,'train_backward':12000,'optimizer_update':1500}
        assert status['calls']['successful']==expected
        assert not status['calls']['failed'] and not status['calls']['pending_call_ids']
        calls=book.events(folder/'calls.jsonl');begins={};dones={}
        for row in calls:
            assert row['event'] in ('CALL_BEGIN','CALL_DONE'), 'Scientific failure needs explicit review'
            target=begins if row['event']=='CALL_BEGIN' else dones
            assert row['call_id'] not in target,'Duplicate call identity'
            target[row['call_id']]=row
        assert set(begins)==set(dones) and len(calls)==51000
        assert all(begins[k]['kind']==dones[k]['kind'] and begins[k]['context']==dones[k]['context'] for k in begins)
        count=collections.Counter(r['kind'] for r in dones.values());assert count==expected
        steps=[r['context']['global_step'] for r in dones.values() if r['kind']=='optimizer_update']
        assert steps==list(range(21001,22501))
        for name in ('checkpoint_step1500','generator_step1500'):
            receipt=book.read(folder/(name+'.json'))
            assert receipt['local_step']==1500 and receipt['global_step']==22500
            book.add(receipt['path'],receipt['sha256'])
        attempts={r['attempt_id'] for r in calls}
        rows.append(dict(arm=arm,actual=dict(count),attempts=len(attempts),pending=0,failed=0,replayed_updates=0))
        totals.update(count)
    return rows,dict(totals)


def complete():
    dest=OUT/'audits/evaluation_closeout.json'
    if dest.exists():
        result=E.read(dest);E.verify(result['protected_files']);return result
    tick=time.perf_counter();book=Evidence();book.add(__file__)
    reg=book.read(E.registration_path());book.verify(reg['training_code_hashes'])
    book.add(reg['source_checkpoint'],reg['source_checkpoint_sha256'])
    book.add(E.MANIFEST);book.verify(E.read(E.MANIFEST)['protected'])
    train_rows,train_totals=_train_audit(book)
    inference=[];counts=collections.Counter();latest_raw=0.;earliest_d0=float('inf')
    for label in E.LABELS:
        raw_complete=book.read(E.raw_completion(label));book.verify(raw_complete['files'])
        new=E.new_requests(label)
        assert raw_complete['outputs']==38 and raw_complete['total_new_rollouts']==len(new)
        assert raw_complete['total_new_generator_forwards']==50*len(new)
        assert raw_complete['total_new_head_forwards']==0
        assert raw_complete['reused_rollouts']==38-len(new)
        post_complete=book.read(OUT/'evaluation'/label/'post_complete_full.json');book.verify(post_complete['files'])
        assert post_complete['outputs']==len(new) and post_complete['new_D0_forwards']==len(new) and post_complete['new_GEO_calls']==len(new)
        for kind in ('RAW','D0'):
            registration=book.read(OUT/'evaluation/inference_registrations'/(label+'__'+kind+'.json'))
            book.verify(registration['code'])
            if 'protected' in registration:book.verify(registration['protected'])
            if kind=='D0':
                assert len(registration['RAW_barrier'])==84
                book.verify(registration['RAW_barrier'])
        seen=[]
        for q in new:
            seen.append(q['request_id']);base=OUT/'evaluation'/label/q['request_id']
            raw=book.read(base/'raw/receipt.json');book.verify(raw['files'])
            assert raw['status']=='COMPLETE' and raw['label']==label and raw['request_id']==q['request_id']
            assert raw['generator_forwards']==50 and raw['D0_forwards']==0 and raw['noise_sha256']==q['noise_sha256']
            book.add(base/'raw/raw.npz',raw['raw_sha256'])
            events=book.events(base/'raw/calls.jsonl')
            assert len(events)==100 and collections.Counter(r['event'] for r in events)=={'FORWARD_STARTED':50,'FORWARD_COMPLETE':50}
            for event in ('FORWARD_STARTED','FORWARD_COMPLETE'):
                assert [r['step'] for r in events if r['event']==event]==list(range(50))
            latest_raw=max(latest_raw,max(r['unix'] for r in events))
            post=book.read(base/'post/receipt.json');book.verify(post['files'])
            assert post['status']=='COMPLETE' and post['label']==label and post['request_id']==q['request_id']
            assert post['D0_forwards']==post['GEO_calls']==1
            book.add(base/'post/post.npz',post['post_sha256'])
            events=book.events(base/'post/calls.jsonl')
            assert [r['event'] for r in events]==['D0_FORWARD_STARTED','D0_FORWARD_RETURNED']
            earliest_d0=min(earliest_d0,datetime.datetime.fromisoformat(events[0]['utc']).timestamp())
            counts.update(RAW=1,generator_forward=50,D0=1,GEO=1)
        folder=OUT/'evaluation'/label
        actual={p.name for p in folder.iterdir() if p.is_dir() and ((p/'raw').exists() or (p/'post').exists())}
        assert actual==set(seen),'Unregistered inference outputs'
        for layout in (folder/'layouts').glob('*/receipt.json'):
            rr=book.read(layout);book.verify(rr['files']);assert rr['occupancy_forward_calls']==0
        failures=list(folder.glob('**/failure.json'));assert not failures,'Science attempt failures require review'
        inference.append(dict(label=label,new_outputs=len(new),reused_outputs=38-len(new),
            sampling_attempts=len(list(folder.glob('sampling_attempt_*'))),D0_attempts=len(list(folder.glob('postprocess_attempt_*')))))
    assert counts==dict(RAW=84,generator_forward=4200,D0=84,GEO=84)
    assert latest_raw<earliest_d0,'All84 RAW before any D0 barrier violated'
    score_reg=book.read(OUT/'evaluation/score_registration.json');book.verify(score_reg['source_hashes'])
    done=book.read(OUT/'results/completion.json');book.verify(done['files']);book.verify(done['source_rows'])
    assert (done['rows'],done['new_score_rows'],done['reused_score_rows'])==(342,252,90)
    metrics=done['metric_fields'];old=E.read(E.OLD/'results/completion.json')
    assert metrics==old['metric_fields'] and len(metrics)==len(set(metrics))==177
    rows=book.read(OUT/'results/metric_rows.json');by={(r['label'],r['stage'],r['request_id']):r for r in rows}
    assert len(by)==len(rows)==342
    status_counts=collections.Counter();na_values=0
    for label in E.LABELS:
        for q in E.requests():
            for stage in E.STAGES:
                source,score,folder=E.authenticate_score(label,q,stage)
                row=by[label,stage,q['request_id']]
                assert row['report_group']==q['report_group']
                assert all(row[m]==source[m] for m in metrics),'Original metric/NA changed'
                assert score['seam_to_bulk']['status']=='NA_NOT_SEAM_REPRESENTATION'
                na_values+=sum(row[m] is None for m in metrics)
                status_counts[row['result_origin']]+=1
    assert status_counts=={'NEW_REGISTERED':252,'REUSED_READ_ONLY':90}
    for rel in ('results/analysis_v1/completion.json','visual_review/summary_v1/completion.json','gallery/completion.json'):
        rr=book.read(OUT/rel);book.verify(rr['files'])
        if 'sources' in rr:book.verify(rr['sources'])
    gallery=E.read(OUT/'gallery/completion.json');assert gallery['sheets']==114 and gallery['cells']==342
    protocol=book.read(OUT/'visual_review/protocol.json')
    expected={(r['request_id'],r['label'],r['stage']):r for r in protocol['expected_units']}
    reviews=[]
    for path in sorted((OUT/'visual_review/shards').glob('*.json')):reviews.extend(book.read(path)['rows'])
    assert len(reviews)==342 and len({(r['request_id'],r['label'],r['stage']) for r in reviews})==342
    for r in reviews:
        want=expected[r['request_id'],r['label'],r['stage']]
        for key in ('parent_id','condition_id','noise_index','report_group'):assert r[key]==want[key]
        assert r['reviewer_kind']=='AI_PROXY' and r['blind'] is False and r['actually_viewed'] is True and r['views_and_sections_checked'] is True
        for key in ('reviewer','view_event_id','viewed_at_utc','note'):assert str(r[key]).strip()
        for key,allowed in AXES.items():assert r[key] in allowed
        png=OUT/'gallery'/(want['page_id']+'.png');receipt=book.read(png.with_suffix('.json'))
        book.add(png,r['image_sha256']);book.add(png.with_suffix('.json'),r['image_receipt_sha256'])
        assert r['gallery_definition_sha256']==receipt['definition_sha256']
    resources={}
    for folder in ('dispatch','workflow'):
        for path in (OUT/folder).rglob('*.json'):
            book.add(path)
            if path.name=='status.json' and folder=='workflow':
                rr=E.read(path);resources[rr.get('job',str(path))]={k:rr[k] for k in ('status','seconds','exit_code') if k in rr}
    for path in (OUT/'evaluation').glob('*/**/*resource*.json'):book.add(path)
    for path in (OUT/'training').glob('*/attempts/*/*resource*.json'):book.add(path)
    if 'torch' in sys.modules:raise RuntimeError('Closeout must not import neural framework')
    result=dict(status='PASS',checked_at_utc=utc(),training=train_rows,training_totals=train_totals,
        actual_inference=dict(counts),inference_attempts=inference,head_forwards=0,
        scientific_failed_calls=0,pending_calls=0,replayed_updates=0,
        RAW_barrier=dict(last_RAW_returned_unix=latest_raw,first_D0_started_unix=earliest_d0),
        scoring=dict(cells=342,new=252,reused=90,original_fields=177,preserved_null_values=na_values,
                     Native_seam_status='NA_NOT_SEAM_REPRESENTATION'),
        gallery_pages=114,actual_AI_PROXY_observations=342,blind=False,
        original_P0_cache_unchanged=True,workflow=resources,protected_files=book.files,
        model_calls=0,scoring_calls=0,tensor_loads=0,auto_promotion=False,seconds=time.perf_counter()-tick)
    E.immutable(dest,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('preflight','complete'))
    result={'preflight':preflight,'complete':complete}[parser.parse_args().action]()
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','protected_files')},ensure_ascii=False,indent=2))
