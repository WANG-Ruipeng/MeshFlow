"""Two additional exposure cycles with newly registered random streams."""
import collections
import copy
import hashlib
import json
from . import common as C
from r2.pilots.bench_high_joint_coverage_v1.stream import validate_manifest

ARMS=C.ARMS
DEFAULT_SEED=2026102707
SCHEMA='BENCH_FOUR_ROUTES_M_PLUS3000_STREAM_V1'


def make():
    original=C.read(C.ORIGINAL_SCHEDULE)
    coeff=C.read(C.ORIGINAL_COEFFICIENTS)
    steps=[];alloc=[]
    for cycle in range(2):
        for source,old_gain in zip(original['steps'],coeff['steps']):
            local=cycle*1500+source['stage_step'];entry=copy.deepcopy(source)
            entry.update(step=local,stage_step=local)
            for micro,row in enumerate(entry['samples']):
                for field in ('noise_seed','epsilon_seed','time_seed','permutation_seed'):
                    key=f'{SCHEMA}|{DEFAULT_SEED}|{local}|{micro}|{field}'
                    row[field]=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')
            steps.append(entry)
            a=copy.deepcopy(old_gain);a['stage_step']=local;alloc.append(a)
    exposure=collections.Counter(r['parent_id'] for e in steps for r in e['samples'])
    schedule=dict(schema=SCHEMA,arm=ARMS[0],seed=DEFAULT_SEED,updates=3000,batch=8,
        steps=steps,parent_exposure=dict(sorted(exposure.items())),
        policy='Two repeats of registered parent/augmentation order; fresh per-slot noise/epsilon/time/permutation; no RNG from worker ordering',
        original_schedule_sha256=C.sha(C.ORIGINAL_SCHEDULE))
    allocations=dict(schema='BENCH_FOUR_ROUTES_M_PLUS3000_COEFFICIENTS_V1',updates=3000,steps=alloc,
        original_coefficients_sha256=C.sha(C.ORIGINAL_COEFFICIENTS),loss_normalization=C.LOSS_NORMALIZATION)
    return schedule,allocations


def validate(schedule,train,updates=3000,arm=ARMS[0]):
    if updates!=3000 or arm!=ARMS[0] or schedule!=make()[0]:raise ValueError('Registered continuation stream differs')
    byid={r['condition_id']:r for r in train};counts=collections.Counter();perparent=collections.Counter()
    for i,e in enumerate(schedule['steps'],1):
        if e['stage_step']!=i or len(e['samples'])!=8:raise ValueError('Stream budget/order')
        if len({r['parent_id'] for r in e['samples']})!=4:raise ValueError('Four paired parents required')
        for r in e['samples']:
            src=byid[r['condition_id']]
            if src['parent_id']!=r['parent_id'] or src['replay_group']!=r['replay_group']:raise ValueError('Source identity')
            counts[r['replay_group']]+=1;perparent[r['parent_id']]+=1
    if dict(counts)!={'OLD':18000,'FOCUS':3000,'HIGH':3000}:raise ValueError('Exposure totals')
    high=[perparent[r['parent_id']] for r in train if r['replay_group']=='HIGH']
    if high!=[500]*6:raise ValueError('Exactly500 additional instances per HIGH parent')
    return dict(status='PASS',updates=3000,instances=24000,group_exposure=dict(counts),
                parent_exposure=dict(perparent),HIGH_cumulative_each=750)


def validate_frozen_narrow(schedule,train):
    return validate(schedule,train)


def checked_coefficients(path,schedule,manifest):
    validate(schedule,manifest['train'])
    result=C.read(path)
    if result!=make()[1]:raise ValueError('Count-only M coefficient policy changed')
    byid={r['condition_id']:r for r in manifest['train']}
    for e,a in zip(schedule['steps'],result['steps']):
        counts=[9*(int(byid[r['condition_id']]['N'])-int(byid[r['condition_id']]['K'])) for r in e['samples']]
        if counts!=a['micro_free_scalar_counts']:raise ValueError('Actual scalar counts differ')
    return result['steps']


def prepare():
    m=C.read(C.MANIFEST);validate_manifest(m)
    schedule,coeffs=make();proof=validate(schedule,m['train'])
    C.write_new(C.SCHEDULE,schedule);C.write_new(C.COEFFICIENTS,coeffs)
    checked_coefficients(C.COEFFICIENTS,schedule,m)
    proof.update(model_calls=0,OT_calls=0,new_training_random_streams=True,
        source_schedule_sha256=C.sha(C.ORIGINAL_SCHEDULE),source_coefficients_sha256=C.sha(C.ORIGINAL_COEFFICIENTS),
        schedule_sha256=C.sha(C.SCHEDULE),coefficient_schedule_sha256=C.sha(C.COEFFICIENTS))
    C.write_new(C.OUT/'data/stream_gate.json',proof)
    return proof


if __name__=='__main__':print(json.dumps(prepare()))
