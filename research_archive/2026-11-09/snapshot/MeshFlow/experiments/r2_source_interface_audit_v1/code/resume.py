"""Read-only completion validation; resume only previously unstarted registered jobs."""
from pathlib import Path
import importlib.util,json,sys,time,concurrent.futures,multiprocessing
P=Path(__file__).with_name('audit.py')
spec=importlib.util.spec_from_file_location('source_audit_frozen',P)
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
def main():
    reg=a.prepare();finished=[];pending=[];blocked=[]
    for job in reg['jobs']:
        d=a.OUT/'conditions'/job['condition_id'];p=d/'result.json'
        if p.exists():
            r=a.read(p)
            assert r['identity']==dict(job=job,entry_sha256=a.sha(P))
            if r['status']=='PASS':finished.append(job)
            else:blocked.append(dict(condition_id=job['condition_id'],status=r['status'],reason='Explicit review required; no retry'))
        elif d.exists():blocked.append(dict(condition_id=job['condition_id'],status='STARTED_WITHOUT_RESULT',reason='No implicit replay'))
        else:pending.append(job)
    if blocked:
        print(json.dumps(dict(status='BLOCKED_NO_RETRY',finished=len(finished),pending=len(pending),items=blocked)));return 2
    if not pending:
        manifest=a.OUT/'MANIFEST.json'
        if manifest.exists():
            for r in a.read(manifest)['files']:assert a.sha(a.OUT/r['path'])==r['sha256'],r['path']
        print(json.dumps(dict(status='ALREADY_COMPLETE_VERIFIED',conditions=len(finished),protected_files=len(reg['protected']),new_geometry_calls=0,models=0)))
        return 0
    stamp=str(time.time_ns());attempt=a.OUT/'recovery'/stamp
    res=a.resources()
    a.write(attempt/'resources_before.json',res)
    a.event('RECOVERY_REGISTERED',already_complete=len(finished),unstarted=len(pending),resource_record=str(attempt))
    first=a.score(pending[0]);a.write(attempt/'first.json',first)
    if first['status']!='PASS':return 2
    # Preserve original measured dispatch; choose at most its count after fresh resource check.
    max_workers=a.read(a.OUT/'protocol/dispatch.json')['CPU_workers']
    workers=min(max_workers,2 if res['affinity']>=4 and res['available_RAM']>max(4<<30,4*first['peak_RSS']) else 1)
    receipts=[first]
    if workers==1:receipts.extend(a.score(j) for j in pending[1:])
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as p:
            receipts.extend(p.map(a.score,pending[1:]))
    a.write(attempt/'completion.json',dict(receipts=receipts,workers=workers,previously_complete=len(finished)))
    if all(r['status']=='PASS' for r in receipts):
        # Existing differing aggregate files are refused by the frozen writer.
        a.aggregate()
    print(json.dumps(dict(status='RECOVERY_RECORDED',receipt=str(attempt/'completion.json'))))
    return 0
if __name__=='__main__':raise SystemExit(main())
