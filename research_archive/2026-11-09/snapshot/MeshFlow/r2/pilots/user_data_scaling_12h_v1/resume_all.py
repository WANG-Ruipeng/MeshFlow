"""Finite local controller. Default is read-only status; no model imports here.

All scientific work is delegated to registered child entrypoints, one GPU
process at a time. DATA_SCALE_BLOCKED and the original clock are not overridable.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OUT = ROOT / 'experiments/r2_user_data_scaling_12h_v1'
TASK = 'R2_USER_DATA_SCALING_12H_V1'
HIST = 'HIST_USER_MIX_FROZEN'
DEFAULT_D0 = ROOT / 'experiments/r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_0030000.pt'
HIST_WEIGHT = ROOT / 'experiments/r2_user_mix_from_official_v1/inference/R2_USER_MIX_generator_step6000.pt'
SCRIPT_NAMES = ('training.py', 'training_data.py', 'training_diagnostics.py', 'evaluation.py', 'scoring.py', 'rendering.py')


class Stop(RuntimeError):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason
        super().__init__(status + ': ' + reason)


def read(path, default=None):
    p = Path(path)
    if not p.exists() and default is not None:
        return default
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def checked(path):
    p = Path(path).absolute()
    if any(x.is_symlink() for x in (p, *p.parents)) or not p.resolve().is_relative_to(OUT.resolve()) or p.resolve() == OUT.resolve():
        raise Stop('OUTPUT_BLOCKED', 'Output must be a new artifact below the registered OUT, without symlinks')
    return p


def write(path, value, replace=False):
    p = checked(path)
    if p.exists() and not replace:
        if read(p) != value:
            raise Stop('IDENTITY_BLOCKED', 'Frozen artifact differs: ' + str(p))
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = checked(p.with_name(p.name + '.tmp'))
    if temp.exists():
        raise Stop('RECOVERY_REQUIRED', 'Incomplete JSON write: ' + str(temp))
    with temp.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, p)


def event(name, **fields):
    p = checked(OUT / 'controller/ledger.jsonl'); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf8') as f:
        f.write(json.dumps(dict({'event':name, 'utc':utc()}, **fields), ensure_ascii=False, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def registration():
    r = read(OUT / 'registration.json')
    if r.get('task', r.get('experiment')) != TASK or r.get('experiment', TASK) != TASK:
        raise Stop('IDENTITY_BLOCKED', 'Wrong experiment registration')
    return r


def clock_check(training=False):
    r = registration()
    limit = r['training_deadline_unix'] if training else r['deadline_unix']
    if time.time() >= float(limit):
        raise Stop('TRAINING_TIME_CUTOFF' if training else 'TASK_DEADLINE_REACHED', 'No new scientific work after registered cutoff')
    return r


def pointer(freeze, key, expected):
    ref = freeze.get(key)
    if not isinstance(ref, dict) or set(('path', 'sha256')) - set(ref):
        raise Stop('DATA_SCALE_BLOCKED', 'Missing frozen ' + key + ' identity')
    p = Path(ref['path'])
    if not p.is_absolute():
        p = OUT / p
    if p.resolve() != expected.resolve() or not p.is_file() or sha(p) != ref['sha256']:
        raise Stop('DATA_SCALE_BLOCKED', 'Frozen ' + key + ' path/hash mismatch')
    return p


def data_gate():
    registration()
    freeze_path = OUT / 'data/data_freeze.json'
    if not freeze_path.exists():
        raise Stop('DATA_SCALE_BLOCKED', 'No final data_freeze.json; inventory/proposals are not a training pool')
    f = read(freeze_path)
    if f.get('status') != 'FROZEN':
        raise Stop('DATA_SCALE_BLOCKED', 'Explicit data status is ' + str(f.get('status', 'MISSING')))
    tp = pointer(f, 'train_manifest', OUT / 'data/train_manifest.json')
    pp = pointer(f, 'panel_manifest', OUT / 'data/panel_manifest.json')
    train, panel = read(tp), read(pp)
    if train.get('status') != 'FROZEN' or train.get('schema') != 'r2_user_scaling_data_v1':
        raise Stop('DATA_SCALE_BLOCKED', 'Training manifest is not a frozen scaling manifest')
    pools = train.get('pools', {}); large = [x for x in pools if x.startswith('USER_LARGE_')]
    if len(large) != 1 or set(pools) != {'USER_SMALL', large[0]}:
        raise Stop('DATA_SCALE_BLOCKED', 'Exactly SMALL and one actual LARGE are required')
    arm = large[0]; small, big = pools['USER_SMALL'], pools[arm]
    if len(small) != 32 or not 128 <= len(big) <= 256 or not set(small) < set(big) or any(len(v) != len(set(v)) for v in pools.values()):
        raise Stop('DATA_SCALE_BLOCKED', 'Need 32 nested SMALL and 128..256 independent LARGE parents')
    if arm != 'USER_LARGE_' + str(len(big)) or f.get('large_label') != arm or f.get('train_parent_count') != len(big):
        raise Stop('DATA_SCALE_BLOCKED', 'LARGE name/count does not match the frozen actual pool')
    conditions = panel.get('conditions', [])
    counts = collections.Counter(x['parent_id'] for x in conditions)
    if len(conditions) != 64 or len(counts) != 32 or set(counts.values()) != {2} or len({x['condition_id'] for x in conditions}) != 64 or f.get('dev_parent_count') != 32:
        raise Stop('DATA_SCALE_BLOCKED', 'Need a new 32-parent / 64-distinct-C panel')
    if set(counts) & set(big):
        raise Stop('DATA_SCALE_BLOCKED', 'Training and DEV parents overlap')
    if any(not 128 <= int(x['N']) <= 256 or not 0 < int(x['K']) < int(x['N']) for x in conditions):
        raise Stop('DATA_SCALE_BLOCKED', 'Invalid DEV N/K')
    # The original training adapter performs the remaining exact C/Y partition
    # and source-byte checks. This stdlib gate cannot import a model framework.
    return dict(freeze=f, train=train, panel=panel, large=arm, train_path=tp, panel_path=pp)


def status():
    r = registration(); f = read(OUT / 'data/data_freeze.json', {})
    try:
        g = data_gate(); gate = dict(status='PASS', large=g['large'])
    except Stop as exc:
        gate = dict(status=exc.status, reason=exc.reason)
    return dict(task=TASK, utc=utc(), registration_status=r.get('status'), data_status=f.get('status', 'NOT_FROZEN'),
        data_gate=gate, train_parent_count=f.get('train_parent_count'), dev_parent_count=f.get('dev_parent_count'),
        deadline_utc=r['deadline_utc'], training_cutoff_utc=r['training_cutoff_utc'],
        seconds_to_deadline=float(r['deadline_unix']) - time.time(),
        budget=read(OUT / 'training_budget_lock.json', {'status': 'NOT_LOCKED'}),
        training={p.parent.name: read(p) for p in sorted((OUT / 'training').glob('*/status.json'))},
        controller=read(OUT / 'controller/status.json', {'status': 'NOT_DISPATCHED'}),
        model_imported=False, model_calls=0, default_changed=False, commit_or_push=False)


def verify():
    """Bounded CPU check. Large weights are deliberately not read here."""
    s = status(); paths = {p: None for p in CODE.glob('*.py')}
    paths[OUT / 'registration.json'] = None
    freeze = read(OUT / 'data/data_freeze.json', {})
    if freeze:
        paths[OUT / 'data/data_freeze.json'] = None
    for key in ('train_manifest', 'panel_manifest'):
        ref = freeze.get(key, {})
        if isinstance(ref, dict) and ref.get('path'):
            p = Path(ref['path']); p = p if p.is_absolute() else OUT / p
            paths[p] = ref.get('sha256')
    # Verify only declared source dependencies, never a recursive asset crawl.
    for rp in (OUT / 'evaluation/implementation_check.json', OUT / 'evaluation/registration.json'):
        if rp.exists():
            for p, h in read(rp).get('source_hashes', {}).items():
                paths[Path(p)] = h
    rows=[]; total=0
    for p, expected in sorted(paths.items(), key=lambda x: str(x[0])):
        if not p.is_file():
            rows.append(dict(path=str(p), status='MISSING')); continue
        size=p.stat().st_size
        if size > 16 * 1024**2 or total + size > 256 * 1024**2:
            rows.append(dict(path=str(p), bytes=size, status='SKIPPED_BOUNDED_AUDIT')); continue
        actual=sha(p); total += size
        rows.append(dict(path=str(p), bytes=size, sha256=actual, expected_sha256=expected,
            status='PASS' if expected is None or actual == expected else 'HASH_MISMATCH'))
    report=dict(status='HASH_MISMATCH' if any(x['status'] in ('MISSING','HASH_MISMATCH') for x in rows) else 'PASS_LIMITED_CPU',
        utc=utc(), bytes_hashed=total, files=rows, data_gate=s['data_gate'],
        scope='Source and frozen manifest files only; no weights, model imports, generation, OT or new scores',model_calls=0)
    write(OUT / 'controller' / ('verify_' + str(time.time_ns()) + '.json'), report)
    return report


@contextlib.contextmanager
def file_lock(name):
    if os.name == 'nt':
        raise Stop('ENVIRONMENT_BLOCKED', 'Scientific dispatch uses the registered WSL Python, not Windows Python')
    import fcntl
    p=checked(OUT / name); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Stop('RESOURCE_BLOCKED', 'Another owned controller/worker holds ' + name)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def resources(name, estimate_mib=0):
    def query(args):
        q=subprocess.run(args, capture_output=True, text=True, timeout=20)
        return dict(returncode=q.returncode, stdout=q.stdout, stderr=q.stderr)
    record=dict(utc=utc(), gpu=query(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        processes=query(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']),
        cpu_logical=os.cpu_count(), loadavg=list(os.getloadavg()), memory=Path('/proc/meminfo').read_text(),
        disk_free_bytes=__import__('shutil').disk_usage(OUT).free, owned_gpu_workers=1,
        estimated_increment_MiB=estimate_mib, headroom='max(2048 MiB, 15% total)',
        policy='One GPU worker fixed by task; initial authorized preflight measures full AdamW peak, no extra probe')
    write(OUT / 'controller/resources' / (name + '_' + str(time.time_ns()) + '.json'),record)
    if record['gpu']['returncode']:
        raise Stop('RESOURCE_BLOCKED', 'nvidia-smi unavailable')
    fields=record['gpu']['stdout'].strip().splitlines()[0].split(','); total, free=float(fields[1]),float(fields[3])
    if free - max(2048., .15 * total) < estimate_mib:
        raise Stop('RESOURCE_BLOCKED', 'Measured available memory does not retain required headroom')
    return record


def dispatch(name, script, args, *, gpu=False, training=False, child_owns_gpu=False, estimate_mib=0):
    clock_check(training); folder=OUT / 'controller/dispatch' / name; rp=folder / 'receipt.json'
    identity=dict(script=str(CODE / script), script_sha256=sha(CODE / script), args=list(args),
        data_freeze_sha256=sha(OUT / 'data/data_freeze.json'))
    if rp.exists():
        prior=read(rp)
        if prior.get('identity') != identity or prior.get('status') != 'COMPLETE':
            raise Stop('RECOVERY_REQUIRED', 'Existing dispatch failed/interrupted or identity changed: ' + name)
        return prior
    if folder.exists():
        raise Stop('RECOVERY_REQUIRED', 'Incomplete dispatch; never silently replay ' + name)
    folder=checked(folder); folder.mkdir(parents=True)
    before=resources(name,estimate_mib) if gpu else None
    command=[sys.executable,'-B',str(CODE / script),*map(str,args)]
    env=os.environ.copy(); env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='2'
    identity['args']=list(args)
    write(folder/'started.json',dict(identity=identity,utc=utc(),command=command,gpu=gpu))
    event('DISPATCH_BEGIN',name=name,command=command,gpu=gpu); start=time.monotonic()
    cm=contextlib.nullcontext() if not gpu or child_owns_gpu else file_lock('gpu_owner.lock')
    expired=False
    with cm, (folder/'stdout.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        next_print=time.monotonic()+30
        try:
            while child.poll() is None:
                # A training child handles the nine-hour safe checkpoint cutoff.
                # Twelve hours always terminates this owned process group.
                if time.time() >= float(registration()['deadline_unix']):
                    expired=True; os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL); child.wait()
                    break
                if time.monotonic() >= next_print:
                    print(json.dumps(dict(stage=name,pid=child.pid,elapsed_seconds=round(time.monotonic()-start),log=str(folder/'stdout.log'))),flush=True)
                    next_print=time.monotonic()+30
                time.sleep(1)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL); child.wait()
            raise
    result=dict(identity=identity,status='TASK_DEADLINE_REACHED' if expired else 'COMPLETE' if child.returncode==0 else 'FAILED',
        returncode=child.returncode,seconds=time.monotonic()-start,utc=utc(),stdout_path=str(folder/'stdout.log'))
    write(rp,result); event('DISPATCH_END',name=name,**{k:result[k] for k in ('status','returncode','seconds')})
    if gpu: resources(name+'_after',0)
    if result['status'] != 'COMPLETE':
        raise Stop(result['status'], 'See preserved child log; no automatic retry: ' + str(folder))
    return result


def lock_budget(gate, init_receipt):
    p=OUT/'training_budget_lock.json'; pre=read(OUT/'preflight/report.json')
    if pre.get('status')!='PASS' or pre.get('manifest_sha256')!=sha(gate['train_path']):
        raise Stop('PREFLIGHT_BLOCKED','Real finite update preflight did not pass frozen data')
    if p.exists():
        lock=read(p)
        if lock.get('status')!='LOCKED' or lock.get('updates_per_arm') not in (4000,6000) or lock.get('preflight_sha256')!=sha(OUT/'preflight/report.json') or lock.get('manifest_sha256')!=sha(gate['train_path']):
            raise Stop('BUDGET_BLOCKED','Budget lock does not match real preflight and data')
        return lock
    if list((OUT/'training').glob('*/calls.jsonl')):
        raise Stop('BUDGET_BLOCKED','Cannot first choose budget after formal work began')
    r=clock_check(True); rate=float(pre['timing']['conservative_seconds_per_update'])
    if not math.isfinite(rate) or rate<=0 or pre.get('updates')!=16 or not pre.get('discard_preflight_state'):
        raise Stop('PREFLIGHT_BLOCKED','Missing sixteen real measured/discarded updates')
    # No extra save benchmark: disclose conservative allowances derived from
    # measured common initialization/load/save wall time, not weight file size.
    initial=float(init_receipt['seconds']); save_allowance=max(10.,3*initial); reload_allowance=max(10.,initial)
    diagnostic_allowance=float(pre['wall_seconds'])
    estimates={}
    for steps in (6000,4000):
        saves=2*(steps//500); segments=2*(steps//2000)
        estimates[str(steps)]=dict(update_seconds=2*steps*rate,save_seconds=saves*save_allowance,
            reload_seconds=segments*reload_allowance,diagnostic_seconds=diagnostic_allowance,
            total_seconds=2*steps*rate+saves*save_allowance+segments*reload_allowance+diagnostic_allowance)
    remaining=float(r['training_deadline_unix'])-time.time()
    selected=next((s for s in (6000,4000) if estimates[str(s)]['total_seconds']<=remaining),None)
    value=dict(status='LOCKED' if selected else 'BUDGET_BLOCKED',updates_per_arm=selected,utc=utc(),
        training_deadline_unix=r['training_deadline_unix'],remaining_seconds_at_lock=remaining,
        estimates=estimates,conservative_seconds_per_update=rate,save_allowance_per_checkpoint=save_allowance,
        reload_allowance_per_segment=reload_allowance,diagnostic_allowance_seconds=diagnostic_allowance,
        preflight_sha256=sha(OUT/'preflight/report.json'),manifest_sha256=sha(gate['train_path']),
        data_freeze_sha256=sha(OUT/'data/data_freeze.json'),shared_initialization_sha256=read(OUT/'initialization/common_step0000.json')['sha256'],
        decision_basis='Actual preflight inclusive of OT wait; explicit conservative save/reload/128F allowance; no generated quality',
        downgrade=selected==4000,no_best_selection=True)
    write(p,value)
    if selected is None:raise Stop('BUDGET_BLOCKED','Both 6000 and 4000 paired arms exceed remaining training window')
    return value


def prepare(gate):
    clock_check(True)
    # Inputs and actual Gaussian arrays freeze before any formal training.
    dispatch('evaluation_prepare','evaluation.py',['--stage','prepare','--panel-manifest',str(gate['panel_path']),'--large-label',gate['large']])
    init=dispatch('initialization','training.py',['--stage','init'],gpu=True,training=True,child_owns_gpu=True,estimate_mib=2600.)
    dispatch('fixtures','training.py',['--stage','fixtures'],training=True)
    dispatch('preflight','training.py',['--stage','preflight'],gpu=True,training=True,child_owns_gpu=True,estimate_mib=2600.)
    return lock_budget(gate,init)


def select_endpoint(gate, lock):
    target=lock['updates_per_arm']; labels=['USER_SMALL',gate['large']]
    def exists(step):
        return all((OUT/'inference'/f'{a}_generator_step{step}.pt').is_file() and (OUT/'inference'/f'{a}_generator_step{step}.json').is_file() for a in labels)
    if exists(target):return target
    if time.time()>=registration()['training_deadline_unix'] and target==6000 and exists(4000):
        p=OUT/'evaluation/time_budget_fallback.json'
        if not p.exists():
            write(p,dict(status='TIME_BUDGET_FALLBACK',selected_step=4000,planned_step=6000,utc=utc(),
                reason='Registered training cutoff reached; both preregistered fixed4000 exports exist; no quality selection',
                consumed_steps={a:read(OUT/'training'/a/'status.json',{}).get('global_step') for a in labels}))
        return 4000
    raise Stop('TRAINING_PARTIAL','No common registered endpoint; do not evaluate unequal or invented endpoints')


def train_segments(gate,lock):
    arms=['USER_SMALL',gate['large']]; target=lock['updates_per_arm']; pre=read(OUT/'preflight/report.json')
    observed=pre.get('resources',{}).get('process_peak_mib')
    if observed is None:observed=pre.get('resources',{}).get('total_device_used_peak_mib')
    estimate=max(2600.,1.2*float(observed)) if observed is not None else 2600.
    for endpoint in range(2000,target+1,2000):
        for arm in arms:
            state=read(OUT/'training'/arm/'status.json',{}); step=int(state.get('global_step',0))
            if state and state.get('status') not in ('SEGMENT_COMPLETE','INTERRUPTED_SAFE','TIME_CUTOFF','COMPLETE'):
                raise Stop('RECOVERY_REQUIRED','Unsafe prior training state for '+arm)
            if step>=endpoint:continue
            if time.time()>=registration()['training_deadline_unix']:return select_endpoint(gate,lock)
            attempt='segment_'+str(endpoint)
            dispatch(arm+'_'+attempt,'training.py',['--stage','train','--arm',arm,'--stop-step',str(endpoint),'--attempt',attempt],
                gpu=True,training=True,child_owns_gpu=True,estimate_mib=estimate)
            after=read(OUT/'training'/arm/'status.json')
            if after.get('status')=='INTERRUPTED_SAFE':raise Stop('INTERRUPTED_SAFE','Owned training stopped safely; inspect receipt before explicit recovery')
            if int(after.get('global_step',0))<endpoint:return select_endpoint(gate,lock)
    return select_endpoint(gate,lock)


def reviewed_record():
    p=OUT/'evaluation/gallery/output_review_receipt.json'
    if not p.exists():raise Stop('WAITING_ACTUAL_VISUAL_REVIEW','Render complete; actual AI_PROXY/human observation must be supplied before numeric aggregate')
    r=read(p)
    if r.get('status') not in ('COMPLETE','PARTIAL') or r.get('review_kind') not in ('AI_PROXY','HUMAN'):
        raise Stop('WAITING_ACTUAL_VISUAL_REVIEW','Review must honestly declare COMPLETE/PARTIAL and reviewer type')
    f=checked(r['review_file'])
    if not f.is_file() or sha(f)!=r.get('review_file_sha256'):
        raise Stop('IDENTITY_BLOCKED','Actual visual review file identity differs')
    with f.open(encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h))
    manifest=read(OUT/'evaluation/registration.json');expected={(q['request_id'],a,s) for q in manifest['requests'] for a in manifest['labels'] for s in ('RAW','POST')}
    keys=[(x['request_id'],x['label'],x['stage']) for x in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise Stop('REVIEW_BLOCKED','Review file must retain all registered rows including explicitly unreviewed ones')
    viewed=[x for x in rows if x.get('review_status') in ('REVIEWED','REVIEWED_AI_PROXY','REVIEWED_HUMAN')]
    if not viewed or len(viewed)!=r.get('actually_reviewed') or any(not x.get('notes') or not x.get('reviewed_image_sha256') for x in viewed):
        raise Stop('REVIEW_BLOCKED','Actual image-backed review observations/count are missing')
    if r['status']=='COMPLETE' and len(viewed)!=len(rows):raise Stop('REVIEW_BLOCKED','Cannot call partial image review complete')
    if 'numeric_ranking_seen_before_review' not in r:raise Stop('REVIEW_BLOCKED','Declare whether numerical rankings were seen; do not infer blinding')
    return r


def finish():
    data_gate();clock_check();review=reviewed_record()
    dispatch('aggregate','scoring.py',['--stage','aggregate'])
    value=dict(status='SCORES_COMPLETE_REVIEW_'+review['status'],utc=utc(),
        actual_visual_review=review,report_status='CONTROLLER_DOES_NOT_WRITE_SCIENTIFIC_CONCLUSION',
        next='Use preserved per-parent results and actual observations to write REPORT; no next experiment/default change/commit/push')
    write(OUT/'controller/status.json',value,replace=True)
    return value


def run(d0,preflight_only=False):
    gate=data_gate();clock_check()
    if not Path(d0).is_file() or not HIST_WEIGHT.is_file():
        raise Stop('ASSET_BLOCKED','Existing official D0 and frozen historical USER_MIX must be available; no downloads/substitution')
    if os.name=='nt':raise Stop('ENVIRONMENT_BLOCKED','Use registered WSL Python for scientific execution')
    with file_lock('controller/owner.lock'):
        lock=read(OUT/'training_budget_lock.json',{})
        if not lock:lock=prepare(gate)
        else:lock=lock_budget(gate,read(OUT/'controller/dispatch/initialization/receipt.json'))
        if preflight_only:return dict(status='PREFLIGHT_PASS_BUDGET_LOCKED',updates_per_arm=lock['updates_per_arm'])
        # A locked evaluation registration is a prerequisite, not first frozen
        # after observing model outputs or selecting trained checkpoints.
        if not (OUT/'evaluation/registration.json').exists():raise Stop('IDENTITY_BLOCKED','Missing pretraining frozen evaluation inputs')
        endpoint=train_segments(gate,lock);clock_check()
        for label in ('USER_SMALL',gate['large'],HIST):
            path=HIST_WEIGHT if label==HIST else OUT/'inference'/f'{label}_generator_step{endpoint}.pt'
            dispatch('bind_'+label,'evaluation.py',['--stage','bind','--label',label,'--checkpoint',str(path)])
            dispatch('sample_'+label,'evaluation.py',['--stage','sample','--label',label,'--controller-owned'],gpu=True,estimate_mib=2600.)
        dispatch('postprocess','evaluation.py',['--stage','postprocess','--checkpoint',str(Path(d0)),'--controller-owned'],gpu=True,estimate_mib=2600.)
        dispatch('score','scoring.py',['--stage','score'])
        dispatch('render','rendering.py',[])
        return finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('status','verify','preflight','run','start','aggregate'),default='status')
    p.add_argument('--d0-checkpoint',type=Path,default=DEFAULT_D0)
    a=p.parse_args()
    try:
        if a.stage=='status':value=status()
        elif a.stage=='verify':value=verify()
        elif a.stage=='aggregate':value=finish()
        else:value=run(a.d0_checkpoint,preflight_only=a.stage=='preflight')
        print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    except Stop as exc:
        value=dict(status=exc.status,reason=exc.reason,utc=utc(),no_automatic_retry=True)
        if a.stage not in ('status','verify'):
            write(OUT/'controller/status.json',value,replace=True);event('CONTROLLER_STOP',**value)
        print(json.dumps(value,ensure_ascii=False,indent=2))
        return 0 if exc.status=='WAITING_ACTUAL_VISUAL_REVIEW' else 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
