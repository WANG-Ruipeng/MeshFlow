"""Explicit recovery of a pre-launch resource gate, not a model replay.

Uses existing full training measurements and keeps the original scientific
entrypoints, data, optimizer, attempt, budgets and safety headroom unchanged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.user_data_scaling_32_67_v1 import resume_all as R

ENGINEERING=R.OUT/'engineering/resource_dispatch_recovery'
ARM='USER_EXPANDED_67'
ATTEMPT='segment_6000'
DISPATCH=ARM+'_'+ATTEMPT+'_resource_recovered'
ESTIMATE_MIB=5120.0


def run():
    R.data_gate();R.clock_check()
    with R.file_lock('controller/owner.lock'):
        small=R.read(R.OUT/'training/USER_SMALL/status.json')
        expanded=R.read(R.OUT/'training'/ARM/'status.json')
        if small['status']!='COMPLETE' or small['global_step']!=6000:
            raise R.Stop('RECOVERY_REQUIRED','SMALL final6000 must be complete')
        if expanded['status']=='COMPLETE' and expanded['global_step']==6000:
            pass  # Continue original controller below; no training replay.
        else:
            if expanded['status']!='SEGMENT_COMPLETE' or expanded['global_step']!=4000:
                raise R.Stop('RECOVERY_REQUIRED','Only the exact safe EXPANDED4000 state is authorized here')
            counts=expanded['calls']
            wanted={'optimizer_update':4000,'train_forward':32000,'train_backward':32000}
            if counts['successful']!=wanted or counts['attempted']!=wanted or counts['failed'] or counts['pending_call_ids']:
                raise R.Stop('RECOVERY_REQUIRED','EXPANDED4000 scientific ledger is not closed')
            old=R.OUT/'controller/dispatch'/f'{ARM}_{ATTEMPT}'
            if not old.is_dir() or list(old.iterdir()):
                raise R.Stop('RECOVERY_REQUIRED','Original blocked dispatch must remain the preserved empty prelaunch directory')
            if (R.OUT/'training'/ARM/'attempts'/ATTEMPT).exists():
                raise R.Stop('RECOVERY_REQUIRED','Training attempt already exists; never replay it silently')
            basis=ENGINEERING/'memory_basis.json'
            if not basis.is_file():raise R.Stop('RECOVERY_REQUIRED','Missing independently reviewed measurement basis')
            if R.read(R.OUT/'training_budget_lock.json')['updates_per_arm']!=6000:
                raise R.Stop('IDENTITY_BLOCKED','The original paired6000 budget must remain locked')
            R.clock_check(training=True)
            # Gate runs before the new dispatch directory is created. Existing
            # dispatch() immediately remeasures again before launching the child.
            current=R.resources('resource_recovery_precheck',ESTIMATE_MIB)
            decision=ENGINEERING/'decision.json'
            if not decision.exists():
                R.write(decision,dict(status='AUTHORIZED_MEASUREMENT_BASED_RESOURCE_RECOVERY',utc=R.utc(),
                    basis_path=str(basis),basis_sha256=R.sha(basis),
                    old_dispatch_directory=str(old),old_dispatch_model_calls=0,
                    unchanged_scientific_state=expanded['checkpoint'],
                    data_freeze_sha256=R.sha(R.OUT/'data/data_freeze.json'),
                    budget_sha256=R.sha(R.OUT/'training_budget_lock.json'),
                    training_source_sha256=R.sha(R.CODE/'training.py'),
                    original_controller_sha256=R.sha(R.CODE/'resume_all.py'),
                    new_dispatch=DISPATCH,training_attempt=ATTEMPT,
                    incremental_estimate_MiB=ESTIMATE_MIB,
                    formula='ceil_to_1GiB(1.2 * max(6401-2497, 3204+1024)) = 5120 MiB',
                    process_peak_status='NA_WDDM; device delta is a proxy; 1024 MiB context/measurement allowance is not an observed context peak',
                    headroom_unchanged='max(2048 MiB, 15% total); freshly checked before dispatch',
                    initial_recovery_resources=current,
                    reason='Old fallback used 1.2 times whole-device peak including background memory already present in current used VRAM',
                    new_model_probe_calls=0,model_retries=0,unrelated_processes_stopped=0,
                    changes_to_batch_precision_optimizer_data_updates=0))
                R.event('EXPLICIT_RESOURCE_RECOVERY',decision_path=str(decision),decision_sha256=R.sha(decision))
            else:
                d=R.read(decision)
                if d['basis_sha256']!=R.sha(basis) or d['training_source_sha256']!=R.sha(R.CODE/'training.py') or d['data_freeze_sha256']!=R.sha(R.OUT/'data/data_freeze.json'):
                    raise R.Stop('IDENTITY_BLOCKED','Recovery evidence or frozen implementation changed')
            R.dispatch(DISPATCH,'training.py',['--stage','train','--arm',ARM,'--stop-step','6000','--attempt',ATTEMPT],
                gpu=True,training=True,child_owns_gpu=True,estimate_mib=ESTIMATE_MIB)
    # Original controller skips both completed arms, then performs its same
    # binding, RAW, D0/GEO_C, source winding, scoring, rendering and review gate.
    return R.run(R.DEFAULT_D0)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    if not a.execute:
        print(json.dumps({'status':'READ_ONLY','next':'Use --execute for the explicitly documented resource recovery; no model imported.'}));raise SystemExit(0)
    try:
        print(json.dumps(run(),ensure_ascii=False,indent=2))
    except R.Stop as exc:
        value=dict(status=exc.status,reason=exc.reason,utc=R.utc(),no_automatic_retry=True)
        R.write(R.OUT/'controller/status.json',value,replace=True);R.event('RESOURCE_RECOVERY_CONTROLLER_STOP',**value)
        print(json.dumps(value,ensure_ascii=False,indent=2))
        raise SystemExit(0 if exc.status=='WAITING_ACTUAL_VISUAL_REVIEW' else 2)
