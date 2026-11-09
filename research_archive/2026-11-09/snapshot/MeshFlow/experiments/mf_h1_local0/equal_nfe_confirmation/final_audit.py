"""No-neural-call final audit of registration, arithmetic, accounting and protected assets."""
from pathlib import Path
import argparse, hashlib, json, subprocess, time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def bits(a,b):
    a=np.ascontiguousarray(a);b=np.ascontiguousarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def run(out):
    out=Path(out).resolve();tick=time.perf_counter();protocol=read(out/'registration/protocol.json');checks=[]
    def check(name,passed,**details):checks.append(dict(name=name,passed=bool(passed),**details))
    check('immutable_protocol',sha(out/'registration/protocol.json')==(out/'registration/protocol.sha256').read_text().strip())
    before=read(out/'registration/protection_before.json');changed=[]
    for name,digest in before['hashes'].items():
        p=ROOT/name
        if not p.is_file() or sha(p)!=digest:changed.append(name)
    check('protected_hashes',not changed,count=len(before['hashes']),changed=changed)
    metadata=[]
    for name,expected in before['prior_file_metadata'].items():
        p=ROOT/name
        if not p.is_file():metadata.append(dict(path=name,change='missing'))
        elif [p.stat().st_size,p.stat().st_mtime_ns]!=expected:metadata.append(dict(path=name,change='size/mtime'))
    check('existing_history_and_assets_metadata',not metadata,count=len(before['prior_file_metadata']),changed=metadata)
    source_changes=[name for name,digest in protocol['source_sha256'].items() if sha(ROOT/name)!=digest]
    check('registered_sources',not source_changes,changed=source_changes)
    check('HEAD_unchanged',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==protocol['HEAD'])
    ledger=read(out/'forward_ledger.json');events=[json.loads(line) for line in (out/'forward_ledger.jsonl').read_text().splitlines()]
    attempted=sum(e['action']=='attempt' for e in events);returned=sum(e['action']=='return' for e in events)
    check('durable_ledger',attempted==ledger['total_attempts'] and returned==ledger['total_returns'],attempts=attempted,returns=returned)
    check('budget',attempted<=6320 and ledger['attempts_by_kind']['formal']<=6000 and ledger['attempts_by_kind']['regression']<=200 and ledger['attempts_by_kind']['repeat']<=100 and ledger['attempts_by_kind']['warmup']<=20,by_kind=ledger['attempts_by_kind'])
    rows={};workers=[]
    for group in protocol['groups']:
        p=out/'workers'/(group['group_id']+'.json')
        if p.exists():
            worker=read(p);workers.append(worker)
            for row in worker['rows']:
                if row['task_id'] in rows:raise RuntimeError('Duplicate task')
                rows[row['task_id']]=row
    trajectories=[];formals=[t for g in protocol['groups'] for t in g['tasks'] if t['kind']=='formal']
    for task in [t for g in protocol['groups'] for t in g['tasks']]:
        row=rows.get(task['task_id'])
        if row is None or row.get('status')!='PASS':continue
        p=out/row['path'];okay=sha(p)==row['sha256'];errors=[]
        with np.load(p,allow_pickle=False) as z:
            s=z['states'];v=z['stage_velocities'];q=z['stage_states'];C=z['C'].reshape(12,9)
            known=z['known_slots'];free=z['free_slots'];M=25 if task['solver']=='Midpoint25' else 50
            okay=okay and s.shape==(M+1,112,9) and q.shape==v.shape==(50,112,9)
            okay=okay and s.dtype==q.dtype==v.dtype==np.float32 and np.isfinite(s).all() and np.isfinite(q).all() and np.isfinite(v).all()
            okay=okay and bits(z['raw'],s[-1]) and bits(s[:,known],np.broadcast_to(C,s[:,known].shape).copy()) and bits(q[:,known],np.broadcast_to(C,q[:,known].shape).copy())
            ideal=[(k/50)**2 for k in range(51)] if task['solver']=='E_quad50' else [k/M for k in range(M+1)]
            dt=[1/50]*50 if task['solver']=='Euler50' else [ideal[k+1]-ideal[k] for k in range(M)]
            okay=okay and bits(z['accepted_times_FP32'],np.asarray(ideal,np.float32)) and bits(z['step_sizes_FP32'],np.asarray(dt,np.float32))
            query=[]
            for k in range(M):
                n=2*k if M==25 else k;query.append(ideal[k])
                if not bits(q[n],s[k]):errors.append('start'+str(k))
                if M==25:
                    mid=s[k,free]+v[n,free]*np.float32(dt[k]*.5)
                    if not bits(q[n+1,free],mid):errors.append('midpoint'+str(k))
                    query.append((ideal[k]+ideal[k+1])*.5)
                expect=s[k,free]+v[n+1 if M==25 else n,free]*np.float32(dt[k])
                if not bits(s[k+1,free],expect):errors.append('accept'+str(k))
            okay=okay and bits(z['stage_times_FP32'],np.asarray(query,np.float32)) and not errors
            task_ledger=ledger['tasks'][task['task_id']]
            okay=okay and task_ledger['attempts']==task_ledger['returns']==50
        trajectories.append(dict(task_id=task['task_id'],passed=bool(okay),equation_mismatches=errors))
    check('saved_trajectory_exact_arithmetic',all(r['passed'] for r in trajectories),audited=len(trajectories),failures=[r for r in trajectories if not r['passed']])
    formal_success=sum(rows.get(t['task_id'],{}).get('status')=='PASS' for t in formals)
    check('formal_matrix_complete',len(formals)==120 and formal_success==120,requested=len(formals),successful=formal_success)
    check('regression_and_repeat',all(r.get('regression',{}).get('status')=='PASS' for r in rows.values() if r['kind'] in ('regression','repeat')) and sum(r['kind'] in ('regression','repeat') for r in rows.values())==6)
    check('shared_lock_before_CUDA',all(w.get('CUDA_initialized_before_locked_runtime') is False and w.get('lock_held_until_process_exit') is True for w in workers),worker_count=len(workers))
    results=read(out/'results.json') if (out/'results.json').exists() else {}
    check('formal_scoring_complete',results.get('status')=='COMPLETE_SCORED_MATRIX' and results.get('formal_scored')==120)
    check('scalar_schema_54',len(results.get('metric_names54',[]))==54)
    binding=results.get('binding',[])
    check('all_binding_groups_complete',len(binding)==60 and all(r.get('status')=='PASS' for r in binding))
    score_rows=[];score_groups=[]
    for group in protocol['groups']:
        if not any(t['kind']=='formal' for t in group['tasks']):continue
        path=out/'scores'/(group['group_id']+'.json')
        if path.exists():
            scored=read(path);score_groups.append(scored.get('status'));score_rows.extend(scored.get('rows',[]))
    check('independent_score_manifest',len(score_groups)==20 and all(s=='PASS' for s in score_groups) and len(score_rows)==120 and {r['task_id'] for r in score_rows}=={t['task_id'] for t in formals})
    import csv
    counts={}
    for name,expected in [('per_output.csv',120),('binding.csv',60)]:
        p=out/name
        if p.exists():
            with p.open(newline='') as f:counts[name]=len(list(csv.DictReader(f)))
        else:counts[name]=0
        check(name+'_denominator',counts[name]==expected,rows=counts[name])
    plots=sorted((out/'figures').glob('*.png'))
    check('fixed_views',len(plots)>=10,count=len(plots))
    diff=[]
    for p in sorted(Path(__file__).parent.glob('*.py')):
        relative=str(p.relative_to(ROOT));content=p.read_text().splitlines()
        diff.extend([f'diff --git a/{relative} b/{relative}','new file mode 100644','--- /dev/null',f'+++ b/{relative}',f'@@ -0,0 +1,{len(content)} @@']+['+'+x for x in content])
    (out/'code.diff').write_text('\n'.join(diff)+'\n')
    result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL_OR_INCOMPLETE',checks=checks,model_forwards=0,CPU_trajectory_audits=len(trajectories),
        seconds=time.perf_counter()-tick,HEAD=protocol['HEAD'],git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        code_diff_sha256=sha(out/'code.diff'),actual_forward_attempts=attempted,actual_forward_returns=returned,
        artifacts={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in [out/'results.json',out/'per_output.csv',out/'paired_effects.csv',out/'binding.csv',out/'registration/protocol.json'] if p.exists()})
    (out/'final_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('status','seconds','actual_forward_attempts','CPU_trajectory_audits')}),flush=True)
    if result['status']!='PASS':
        print(json.dumps([c for c in checks if not c['passed']],indent=2),flush=True)
        raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
