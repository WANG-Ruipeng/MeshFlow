"""Deterministic OLD2/NEW2 parent replay schedule; no model or Gaussian draws."""
from __future__ import annotations
import argparse, collections, hashlib, json
import numpy as np
from .common import OUT, read, sha, write_new

SCHEMA = 'BENCH_SEAT_COVERAGE_PARENT_REPLAY_STREAM_V1'
DEFAULT_SEED = 2026102307


def groups_for(train):
    groups = {name: collections.defaultdict(list) for name in ('OLD', 'NEW')}
    ids = set()
    for i, row in enumerate(train):
        if row['condition_id'] in ids:
            raise ValueError('Repeated condition identity')
        ids.add(row['condition_id'])
        group = row['replay_group']
        if group not in groups:
            raise ValueError('Each training condition needs OLD or NEW replay_group')
        groups[group][row['parent_id']].append(i)
    if set(groups['OLD']) & set(groups['NEW']):
        raise ValueError('A parent cannot occur in both replay groups')
    if min(map(len, groups.values())) < 2:
        raise ValueError('At least two distinct parents in each replay group')
    for parents in groups.values():
        for p, indices in parents.items():
            parents[p] = sorted(indices, key=lambda i: train[i]['condition_id'])
    return groups


def make_schedule(train, updates, seed=DEFAULT_SEED):
    if type(updates) is not int or updates <= 0:
        raise ValueError('Positive finite update budget required')
    groups = groups_for(train)
    parents = {g: sorted(rows) for g, rows in groups.items()}
    rng = {g: np.random.RandomState(seed ^ extra) for g, extra in (('OLD', 0x0D), ('NEW', 0x0E))}
    aug = np.random.RandomState(seed ^ 0xABCDEF)
    queue = {g: [] for g in groups}
    cursor = collections.Counter()
    exposure = collections.Counter()
    steps = []
    for step in range(1, updates + 1):
        chosen = []
        for g in ('OLD', 'NEW'):
            local = []
            while len(local) < 2:
                if not queue[g]:
                    queue[g].extend(parents[g][int(i)] for i in rng[g].permutation(len(parents[g])))
                pos = next((i for i, p in enumerate(queue[g]) if p not in local), None)
                if pos is None:
                    queue[g].extend(parents[g][int(i)] for i in rng[g].permutation(len(parents[g])))
                    continue
                local.append(queue[g].pop(pos))
            chosen.extend((g, p) for p in local)
        samples = []
        for j, (g, p) in enumerate(chosen):
            options = groups[g][p]
            index = options[cursor[(g, p)] % len(options)]
            cursor[(g, p)] += 1
            source = train[index]
            for variant in (0, 1):
                key = f'{SCHEMA}|{seed}|{step}|{j}|{variant}'
                digest = hashlib.sha256(key.encode()).digest()
                seeds = [int.from_bytes(digest[i:i+4], 'little') for i in (0, 4, 8, 12)]
                alpha = 1. if variant == 0 else float(aug.uniform(.9, 1.1))
                samples.append(dict(index=index, condition_index=index, condition_id=source['condition_id'],
                    parent_id=p, replay_group=g, part_type=source.get('part_type', 'SEAT_SURFACE'),
                    sample_index=len(samples), alpha=alpha, scale_xyz=[alpha, 1., 1.],
                    augmentation='CLEAN' if variant == 0 else 'X_WIDTH',
                    noise_seed=seeds[0], epsilon_seed=seeds[1], time_seed=seeds[2], permutation_seed=seeds[3]))
                exposure[source['condition_id']] += 1
        steps.append(dict(step=step, stage='NATIVE', stage_step=step, samples=samples))
    return dict(schema=SCHEMA, seed=seed, updates=updates, batch=8, old_parents_per_update=2,
        new_parents_per_update=2, training_budget_authorized_by_this_file=False,
        condition_exposure=dict(sorted(exposure.items())), steps=steps,
        policy='Separate shuffled parent cycles OLD2 NEW2; per-parent condition round-robin; identity plus x-width U(.9,1.1); no model/part-label input changes.')


def validate(schedule, train, updates):
    groups = groups_for(train)
    if schedule.get('schema') != SCHEMA or schedule.get('updates') != updates:
        raise ValueError('Schedule schema/budget differs')
    if [r['step'] for r in schedule['steps']] != list(range(1, updates + 1)):
        raise ValueError('Schedule cursor is incomplete')
    lookup = {r['condition_id']: r for r in train}
    seen = set()
    for e in schedule['steps']:
        if e.get('stage') != 'NATIVE' or e.get('stage_step') != e['step']:
            raise ValueError('Only a finite Native stage is registered')
        batch = e['samples']
        if len(batch) != 8 or len({r['parent_id'] for r in batch}) != 4:
            raise ValueError('Exactly four distinct parents / eight microbatches required')
        if collections.Counter(r['replay_group'] for r in batch) != {'OLD': 4, 'NEW': 4}:
            raise ValueError('OLD2/NEW2 parent quota changed')
        for r in batch:
            source = lookup[r['condition_id']]
            if (r['parent_id'], r['replay_group']) != (source['parent_id'], source['replay_group']):
                raise ValueError('Schedule changes parent/group identity')
            alpha = float(r['alpha'])
            if not np.isfinite(alpha) or not .9 <= alpha <= 1.1 or r['scale_xyz'] != [alpha, 1., 1.]:
                raise ValueError('Width augmentation contract changed')
            if any(type(r[key]) is not int or not 0 <= r[key] < 2**32 for key in ('noise_seed', 'epsilon_seed', 'time_seed', 'permutation_seed')):
                raise ValueError('Explicit valid scalar RNG seeds required')
            seen.add(r['condition_id'])
        for p in {r['parent_id'] for r in batch}:
            pair = [r for r in batch if r['parent_id'] == p]
            if len(pair) != 2 or len({r['condition_id'] for r in pair}) != 1 or sorted(r['augmentation'] for r in pair) != ['CLEAN', 'X_WIDTH']:
                raise ValueError('Each parent requires one C with paired identity/width')
            if next(r for r in pair if r['augmentation'] == 'CLEAN')['alpha'] != 1.:
                raise ValueError('Clean member must have identity augmentation')
    if seen != set(lookup):
        raise ValueError('Registered pool has conditions never exposed within budget')
    rebuilt = make_schedule(train, updates, schedule['seed'])
    if rebuilt != schedule:
        raise ValueError('Frozen schedule is not reproducible from registered seed/pool')
    return dict(status='PASS', updates=updates, exposures=updates*8,
        old_parents=len(groups['OLD']), new_parents=len(groups['NEW']), conditions=len(train))


def prepare(updates=1500):
    manifest = read(OUT/'data/manifest.json')
    budget_path = OUT/'budget.json'
    if budget_path.exists():
        budget = read(budget_path)
        endpoint = budget.get('final_global_step',budget.get('endpoint_global_step'))
        if budget['updates'] != updates or budget.get('source_global_step') != 18000 or endpoint != 18000+updates:
            raise ValueError('Existing budget differs; never silently override')
    else:
        budget = dict(updates=updates, source_global_step=18000, final_global_step=18000+updates,
                      full_steps=[0, updates], checkpoint_every=250, no_automatic_extension=True)
    result = make_schedule(manifest['train'], updates)
    validate(result, manifest['train'], updates)
    if not budget_path.exists():write_new(budget_path, budget)
    write_new(OUT/'data/schedule.json', result)
    return dict(status='PREPARED_NOT_AUTHORIZED', budget=budget,
                manifest_sha256=sha(OUT/'data/manifest.json'), schedule_sha256=sha(OUT/'data/schedule.json'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates', type=int, default=1500)
    print(json.dumps(prepare(p.parse_args().updates), indent=2))
