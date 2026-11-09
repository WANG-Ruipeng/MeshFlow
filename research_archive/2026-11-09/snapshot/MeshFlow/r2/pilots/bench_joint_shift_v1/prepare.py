"""Freeze the identical RELATION pool/stream plus a separate translation stream.

Reuse existing data acceptance receipts; only the new affine packing is tested.
This entry point performs no learned-model forward, GPU job, registration or
training dispatch. It never rewrites source cases or old experiment artifacts.
"""
from __future__ import annotations
import collections
import copy
import json
import time
import traceback
from .common import ROOT, OUT, read, resolve, sha, resources
from . import data as D
from . import cpu_checks


def main():
    if (OUT/'cpu_gate.json').exists():
        gate=read(OUT/'cpu_gate.json')
        if gate['status']!='PASS': raise ValueError('Prior gate was not PASS')
        for p,h in gate['protected'].items():
            if sha(p)!=h: raise ValueError('Protected input changed: '+p)
        receipt=read(gate['translation_checks']['path'])
        for p,h in receipt['code_hashes'].items():
            if sha(p)!=h: raise ValueError('Checked code changed: '+p)
        return dict(status='PASS',mode='REUSED_HASH_CHECK_ONLY',new_OT_calls=0)
    started=time.perf_counter()
    D.write_new(OUT/'data/cpu_prepare_resources.json',resources())
    reference=D.REFERENCE
    old_manifest_path=reference/'data/manifest.json'
    old_schedule_path=reference/'data/schedule.json'
    prior_gate_path=reference/'cpu_gate.json'
    prior_gate=read(prior_gate_path)
    if prior_gate['status']!='PASS' or prior_gate['reused_passed_OT_conditions']!=91:
        raise ValueError('RELATION accepted source data receipt required')
    for path in (old_manifest_path,old_schedule_path):
        expected=next((h for p,h in prior_gate['protected'].items() if resolve(p)==path),None)
        if expected is None or sha(path)!=expected:
            raise ValueError('Frozen RELATION manifest/schedule differs: '+str(path))
    manifest=read(old_manifest_path)
    train,dev=manifest['train'],manifest['dev']
    if (len(train),len({r['parent_id'] for r in train}),len(dev),len({r['parent_id'] for r in dev}))!=(91,26,39,8):
        raise ValueError('Expected unchanged26/91+8/39 pool')
    protected={str(p):sha(p) for p in (old_manifest_path,old_schedule_path,prior_gate_path)}
    receipts=copy.deepcopy(prior_gate['validation_receipts'])
    # Verify compact sealed receipts and effective cases; do not rehash the huge
    # original archive or repeat all old production-OT acceptance operations.
    for receipt in receipts:
        if sha(receipt['path'])!=receipt['sha256']:
            raise ValueError('Sealed source acceptance receipt changed')
        protected[receipt['path']]=receipt['sha256']
    for row in train+dev:
        if sha(row['case_path'])!=row['case_sha256']:
            raise ValueError('Effective source case changed: '+row['condition_id'])
        protected[row['case_path']]=row['case_sha256']
    schedule=D.make_schedule(train)
    stream_check=D.validate_schedule(schedule['steps'])
    parents=collections.Counter(r['parent_id'] for e in schedule['steps'] for r in e['samples'])
    conditions=collections.Counter(r['condition_id'] for e in schedule['steps'] for r in e['samples'])
    parts=collections.Counter(r['part_type'] for e in schedule['steps'] for r in e['samples'])
    if sum(parents.values())!=12000 or max(parents.values())-min(parents.values())>2:
        raise ValueError('Original parent exposure changed')
    # Manifest rows remain exactly the old source rows. Metadata names the new
    # online augmentation without pretending to create new source conditions.
    prepared_manifest=copy.deepcopy(manifest)
    prepared_manifest.update(schema='BENCH_JOINT_SHIFT_DATA_V1',
        same_training_rows_as=str(old_manifest_path),same_training_rows_sha256=sha(old_manifest_path),
        online_augmentation='Same XYZ scale, plus joint model-space C/Y translation on ANISOTROPIC half only',
        new_independent_parents_this_experiment=0,new_conditions_this_experiment=0)
    inventory=copy.deepcopy(read(reference/'data/inventory.json'))
    inventory.update(new_independent_parents_this_experiment=0,new_conditions_this_experiment=0,
        reference_experiment='R2_BENCH_RELATION_TRAINING_V1',
        augmentation_only=True,identity_instances=6000,translated_instances=6000,
        translation_fixed_L=D.L,translation_per_axis_fraction=D.TRANSLATION_AXIS_FRACTION,
        translation_per_axis_model_bound=D.TRANSLATION_AXIS_MAX,
        amplitude_basis='New conservative engineering choice; not literature optimum or previously validated magnitude')
    for name,obj in (
        ('manifest.json',prepared_manifest),('schedule.json',schedule),
        ('schedule_identity_check.json',stream_check),('inventory.json',inventory),
        ('evaluation_membership.json',read(reference/'data/evaluation_membership.json')),
        ('exposure_plan.json',dict(parent=dict(parents),condition=dict(conditions),part=dict(parts),
            minimum_parent_exposure=min(parents.values()),maximum_parent_exposure=max(parents.values())))):
        D.write_new(OUT/'data'/name,obj)
    try:
        checked=cpu_checks.run(prepared_manifest,schedule)
    except BaseException as exc:
        D.write_new(OUT/'data'/f'cpu_prepare_failure_{time.time_ns()}.json',dict(
            status='FAILED',error=repr(exc),traceback=traceback.format_exc(),neural_forwards=0,
            no_training_started=True,elapsed_seconds=time.perf_counter()-started))
        raise
    checks_path=OUT/'data/translation_cpu_checks.json'
    D.write_new(checks_path,checked)
    for p in sorted((OUT/'data').glob('*.json')):
        protected[str(p)]=sha(p)
    # New files are frozen by registration as well; gate permits hash-only reuse.
    for p in (ROOT/'r2/pilots/bench_joint_shift_v1/data.py',
              ROOT/'r2/pilots/bench_joint_shift_v1/prepare.py',
              ROOT/'r2/pilots/bench_joint_shift_v1/cpu_checks.py'):
        protected[str(p)]=sha(p)
    gate=dict(status='PASS',neural_forwards=0,GPU_calls=0,new_OT_calls=checked['new_OT_calls'],
        inherited_passed_OT_conditions=91,validation_receipts=receipts,
        inherited_source_gate=dict(path=str(prior_gate_path),sha256=sha(prior_gate_path),
            scope='Data/source provenance acceptance reused; no redundant archive hashing or old per-condition OT'),
        translation_checks=dict(path=str(checks_path),sha256=sha(checks_path)),
        protected=protected,data_counts=dict(train=91,train_parents=26,dev=39,dev_parents=8),
        planned_parent_exposure=[min(parents.values()),max(parents.values())],
        exact_reference_stream=stream_check,
        training_adapter='Original SEAM/FM/Euclidean OT/HYBRID; only joint C/Y translation added to original second scaled sample; Gaussian/epsilon unchanged',
        elapsed_seconds=time.perf_counter()-started)
    D.write_new(OUT/'cpu_gate.json',gate)
    return {k:v for k,v in gate.items() if k not in ('protected','validation_receipts')}


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False))
