"""Read preserved arrays, receipts and scores; never invoke geometry or scoring."""
from .. import runtime as R
import argparse
import json


def equal_fields(left,right,fields):
    return all(json.dumps(left[f],sort_keys=True,allow_nan=True)==
               json.dumps(right[f],sort_keys=True,allow_nan=True) for f in fields)


def run(jobs_path):
    reg,rh,jobs=R.load_jobs(jobs_path);folder=R.OUT/'audits/execution';done=folder/'complete.json'
    if done.exists():
        old=R.read(done)
        if old['registration_sha256']!=rh:raise ValueError('Audit registration changed')
        R.verify(old['inputs']);return old
    result_rows=[];inputs={str(R.resolve(jobs_path)):rh};fields=reg['metric_fields']
    for job in reg['jobs']:
        jid=job['job_id'];dest=R.job_dir(jid);cp=dest/'complete.json'
        receipt=R.verify_result(R.read(cp));inputs[str(cp)]=R.sha(cp)
        if receipt['registration_sha256']!=rh or receipt['job_sha256']!=R.canonical_hash(job):
            raise AssertionError('Job identity differs: '+jid)
        identity,raw,source,x=R.proposal_contract(job)
        if identity!=receipt['scientific_identity']:raise AssertionError('Scientific identity differs')
        k=int(job['request']['K']);a=receipt['artifacts']
        if R.arrays(a['pre'])['output'].tobytes()!=x.tobytes():raise AssertionError('Physical proposal differs')
        for name in ('base','final'):
            post=R.arrays(a[name])
            if post['output'][:k].tobytes()!=source['C_original'].tobytes():
                raise AssertionError('Source-oriented C differs: '+jid+'/'+name)
        events=[json.loads(line) for line in (dest/'events.jsonl').read_text().splitlines() if line.strip()]
        calls=[e['kind'] for e in events if e['event']=='CALL_STARTED']
        finished=[e['kind'] for e in events if e['event']=='CALL_COMPLETE']
        physical=receipt['execution']=='NEW'
        expected=['GEO_SOURCE_EXPORT','BASE_PAIR_DETECTOR','FULL_CPU_CASCADE'] if physical else []
        if calls!=expected or finished!=expected:raise AssertionError('Scientific event budget mismatch: '+jid)
        if any(e['event']=='FAILED' for e in events) or (dest/'failure.json').exists():
            raise AssertionError('Failure or replay needs explicit audit')
        for key in ('model_calls','D0_calls','new_noise','new_queries'):
            if receipt[key]!=0:raise AssertionError('Unexpected scientific call')
        for key,value in [('GEO_calls',int(physical)),('base_pair_detector_calls',int(physical)),
                          ('full_cascade_calls',int(physical)),('new_score_rows',2*int(physical))]:
            if receipt[key]!=value:raise AssertionError('Count receipt differs')
        rows=R.read(a['metric_rows'])
        if len(rows)!=2 or {r['stage'] for r in rows}!={'BASE_POST','CLEAN_FINAL_POST'}:
            raise AssertionError('Two-stage score identity differs')
        if any(any(f not in r for f in fields) for r in rows):raise AssertionError('Original177 field missing')
        result_rows.append(dict(job_id=jid,execution=receipt['execution'],C_exact=True,
            physical_proposal_exact=True,new_GEO_calls=int(physical),new_full_cascades=int(physical),
            new_score_rows=2*int(physical),new_base_pair_detector_calls=int(physical),
            seconds=receipt['seconds'],peak_RSS_KiB=receipt['peak_RSS_KiB'],
            CPU_complete=a['cpu_complete'],metric_rows=a['metric_rows']))
    collected=R.read(R.OUT/'results/complete.json');R.verify(collected['files'])
    inputs[str(R.OUT/'results/complete.json')]=R.sha(R.OUT/'results/complete.json')
    rows=R.read(R.OUT/'results/metric_rows.json')
    if len(rows)!=72:raise AssertionError('Expected 60 candidate plus 12 baseline rows')
    for row in rows:
        original=R.read(row['physical_metric_source'])
        candidates=[v for v in original if v['request_id']==row['request_id'] and v['stage']==row['stage']]
        if len(candidates)!=1 or not equal_fields(candidates[0],row,fields):
            raise AssertionError('Collected original177 values differ')
    baselines=[r for r in rows if r['label']=='M_D0']
    if len(baselines)!=12:raise AssertionError('Original baseline count differs')
    for row in baselines:
        task=next(t for t in reg['references'] if t['request']['request_id']==row['request_id'])
        original=R.read(task['paths'][row['stage']+'_row'])
        if not equal_fields(original,row,fields):raise AssertionError('Original baseline177 differs')
    physical=sum(r['execution']=='NEW' for r in result_rows)
    if physical!=reg['budget']['physical_jobs'] or 2*physical!=reg['budget']['new_score_rows']:
        raise AssertionError('Frozen physical budget differs')
    R.assert_cpu();R.verify(reg['protected']);R.verify(reg['source_hashes'])
    folder.mkdir(parents=True,exist_ok=True)
    R.csvwrite(folder/'per_job.csv',result_rows)
    result=dict(status='PASS',registration_sha256=rh,helper_sha256=R.sha(__file__),
        logical_jobs=30,new_physical_jobs=physical,reused_jobs=30-physical,new_GEO_calls=physical,
        new_full_cascades=physical,new_base_pair_detector_calls=physical,new_full_score_rows=2*physical,
        detector_scope='Counts explicit initial scan only; internal cascade legality scans remain in original cascade records',
        candidate_rows=60,original_baseline_rows=12,original_fields=177,
        all_C_bytes_and_proposal_exact=True,all_177_values_match_original_sources=True,
        aliases_have_zero_new_scientific_calls=True,failures=0,replays=0,
        model_calls=0,D0_calls=0,new_noise=0,new_queries=0,inputs=inputs,
        files={str(folder/'per_job.csv'):R.sha(folder/'per_job.csv')},utc=R.utc())
    R.write_new(done,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',required=True);a=p.parse_args()
    r=run(a.jobs);print({k:v for k,v in r.items() if k not in ('files','inputs')})
