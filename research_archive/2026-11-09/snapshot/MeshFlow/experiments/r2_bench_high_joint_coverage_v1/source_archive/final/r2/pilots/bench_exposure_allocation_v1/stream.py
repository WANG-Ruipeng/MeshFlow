"""Frozen exposure allocation; no model import or process-global RNG changes."""
from __future__ import annotations
import collections
import hashlib
import numpy as np

SCHEMA = 'BENCH_EXPOSURE_ALLOCATION_STREAM_V1'
ARMS = ('A_50_50', 'B_75_25')
DEFAULT_SEED = 2026102501
QUOTAS = {'A_50_50': (2, 2), 'B_75_25': (3, 1)}


def groups_for(train):
    groups = {g: {} for g in ('OLD', 'NEW')}
    seen = set()
    for i, r in enumerate(train):
        if r['condition_id'] in seen or r['replay_group'] not in groups:
            raise ValueError('Distinct condition IDs and OLD/NEW identities required')
        seen.add(r['condition_id'])
        g, p = r['replay_group'], r['parent_id']
        if p in groups[g]:
            raise ValueError('Exactly one frozen C per parent; no new within-parent sampler')
        groups[g][p] = i
    if len(groups['OLD']) != 22 or len(groups['NEW']) != 2 or set(groups['OLD']) & set(groups['NEW']):
        raise ValueError('Exactly the disjoint OLD22 and priority2 pool is required')
    return groups


def make_schedule(train, arm, updates=1500, seed=DEFAULT_SEED):
    if arm not in ARMS or type(updates) is not int or updates <= 0 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Finite budget, registered arm and uint32 seed required')
    groups = groups_for(train)
    parents = {g: sorted(rows) for g, rows in groups.items()}
    rng = {g: np.random.RandomState(seed ^ x) for g, x in (('OLD', 0x0D), ('NEW', 0x0E))}
    aug = np.random.RandomState(seed ^ 0xABCDEF)
    queue = {g: [] for g in groups}
    exposures = collections.Counter()
    steps = []
    for step in range(1, updates + 1):
        chosen = []
        for g, quota in zip(('OLD', 'NEW'), QUOTAS[arm]):
            local = []
            while len(local) < quota:
                if not queue[g]:
                    queue[g].extend(parents[g][int(i)] for i in rng[g].permutation(len(parents[g])))
                pos = next((i for i, p in enumerate(queue[g]) if p not in local), None)
                if pos is None:
                    queue[g].extend(parents[g][int(i)] for i in rng[g].permutation(len(parents[g])))
                    continue
                local.append(queue[g].pop(pos))
            chosen.extend((g, p) for p in local)
        samples = []
        for slot, (g, p) in enumerate(chosen):
            index = groups[g][p]
            source = train[index]
            for variant in (0, 1):
                # Deliberately excludes arm and parent: paired slot-level random
                # identities are equal while the registered exposure changes.
                key = f'{SCHEMA}|{seed}|{step}|{slot}|{variant}'
                digest = hashlib.sha256(key.encode()).digest()
                seeds = [int.from_bytes(digest[i:i+4], 'little') for i in (0, 4, 8, 12)]
                alpha = 1. if variant == 0 else float(aug.uniform(.9, 1.1))
                samples.append(dict(index=index, condition_index=index, condition_id=source['condition_id'],
                    parent_id=p, replay_group=g, part_type=source.get('part_type', 'SEAT_SURFACE'),
                    sample_index=len(samples), alpha=alpha, scale_xyz=[alpha, 1., 1.],
                    augmentation='CLEAN' if variant == 0 else 'X_WIDTH', noise_seed=seeds[0],
                    epsilon_seed=seeds[1], time_seed=seeds[2], permutation_seed=seeds[3]))
                exposures[source['condition_id']] += 1
        steps.append(dict(step=step, stage='NATIVE', stage_step=step, samples=samples))
    return dict(schema=SCHEMA, arm=arm, seed=seed, updates=updates, batch=8,
        old_parents_per_update=QUOTAS[arm][0], new_parents_per_update=QUOTAS[arm][1],
        condition_exposure=dict(sorted(exposures.items())), steps=steps,
        training_budget_authorized_by_this_file=False,
        policy='Original separate shuffled parent-cycle policy; fixed OLD/NEW quota; one C and CLEAN+X_WIDTH per parent; shared slot RNG identities, no compensation loss.')


def validate(schedule, train, updates=1500, arm=None):
    arm = schedule.get('arm') if arm is None else arm
    groups_for(train)
    if arm not in ARMS or schedule.get('arm') != arm or schedule.get('schema') != SCHEMA or schedule.get('updates') != updates:
        raise ValueError('Frozen stream identity/budget differs')
    if schedule != make_schedule(train, arm, updates, schedule['seed']):
        raise ValueError('Schedule differs from deterministic frozen parent/augmentation/RNG policy')
    parents = collections.Counter()
    for e in schedule['steps']:
        batch = e['samples']
        if len(batch) != 8 or len({r['parent_id'] for r in batch}) != 4:
            raise ValueError('Four distinct parents / batch8 required')
        parents.update(r['parent_id'] for r in batch)
    old = [parents[r['parent_id']] for r in train if r['replay_group'] == 'OLD']
    priority = [parents[r['parent_id']] for r in train if r['replay_group'] == 'NEW']
    if max(old) - min(old) > 2 or max(priority) - min(priority) > 2:
        raise ValueError('Uniform within-group parent exposure differs')
    if updates == 1500:
        expected = (6000, 6000, {272, 274}, [3000, 3000]) if arm == ARMS[0] else (9000, 3000, {408, 410}, [1500, 1500])
        if (sum(old), sum(priority), set(old), sorted(priority)) != expected:
            raise ValueError('Predeclared exposure totals differ')
    return dict(status='PASS', arm=arm, updates=updates, instances=8*updates,
        old_instances=sum(old), priority_instances=sum(priority), parent_exposure=dict(sorted(parents.items())))
