"""Frozen two-arm Native stream; no model import or global RNG mutation."""
from __future__ import annotations
import collections
import hashlib
import numpy as np

SCHEMA = "BENCH_NATURAL_N_RECIPE_STREAM_V1"
ARMS = ("CONTROL_OLD", "NATURAL_HIGH")
DEFAULT_SEED = 2026102601
UPDATES = 1500


def groups_for(train):
    groups = {g: {} for g in ("OLD", "FOCUS", "HIGH")}
    seen = set()
    for i, row in enumerate(train):
        group = row["replay_group"]
        if row["condition_id"] in seen or group not in groups:
            raise ValueError("Distinct conditions and explicit OLD/FOCUS/HIGH groups required")
        seen.add(row["condition_id"])
        parent = row["parent_id"]
        if any(parent in members for members in groups.values()):
            raise ValueError("Exactly one frozen C per parent; no within-parent/LOD sampler")
        n, k = int(row["N"]), int(row["K"])
        if not 128 <= n <= (799 if group == "HIGH" else 256) or not 2 <= k < n:
            raise ValueError("Natural-N completion range differs")
        groups[group][parent] = i
    if tuple(len(groups[g]) for g in ("OLD","FOCUS","HIGH")) != (22,2,2):
        raise ValueError("Exactly OLD22, original FOCUS2 and independent HIGH2 required")
    if not all(any(p.startswith(prefix) for p in groups["FOCUS"]) for prefix in ("a8e987", "27eb79")):
        raise ValueError("Original a8/27 FOCUS identities required")
    if not all(any(p.startswith(prefix) for p in groups["HIGH"]) for prefix in ("28b9", "abf250")):
        raise ValueError("Pre-registered 28b9/abf250 HIGH identities required")
    if sorted(int(train[i]["N"]) for i in groups["HIGH"].values()) != [308,384]:
        raise ValueError("The two full high-N targets must remain N308/N384")
    return groups


def make_schedule(train, arm, updates=UPDATES, seed=DEFAULT_SEED):
    if arm not in ARMS or type(updates) is not int or updates != UPDATES:
        raise ValueError("Exactly the authorized 1500-update paired experiment is required")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("uint32 stream seed required")
    groups = groups_for(train)
    parents = {g: sorted(rows) for g, rows in groups.items()}
    rng = {g: np.random.RandomState(seed ^ salt)
           for g,salt in (("OLD",0x0D),("FOCUS",0x0E),("HIGH",0x0F))}
    aug = np.random.RandomState(seed ^ 0xABCDEF)
    queue = {g: [] for g in groups}
    exposures = collections.Counter()
    steps = []

    def take(group, quota):
        chosen = []
        while len(chosen) < quota:
            if not queue[group]:
                queue[group].extend(parents[group][int(i)] for i in rng[group].permutation(len(parents[group])))
            pos = next((i for i,p in enumerate(queue[group]) if p not in chosen), None)
            if pos is None:
                queue[group].extend(parents[group][int(i)] for i in rng[group].permutation(len(parents[group])))
                continue
            chosen.append(queue[group].pop(pos))
        return [(group,p) for p in chosen]

    for step in range(1,updates+1):
        fourth = "HIGH" if arm == "NATURAL_HIGH" and step % 2 == 0 else "FOCUS"
        chosen = take("OLD",3) + take(fourth,1)
        samples = []
        for slot,(group,parent) in enumerate(chosen):
            index = groups[group][parent]
            source = train[index]
            for variant in (0,1):
                # Same slot/step/variant identities in both arms, independent
                # of chosen parent and worker order, as in exposure allocation.
                key = f"{SCHEMA}|{seed}|{step}|{slot}|{variant}"
                digest = hashlib.sha256(key.encode()).digest()
                seeds = [int.from_bytes(digest[i:i+4],"little") for i in (0,4,8,12)]
                alpha = 1. if variant == 0 else float(aug.uniform(.9,1.1))
                samples.append(dict(index=index,condition_index=index,condition_id=source["condition_id"],
                    parent_id=parent,replay_group=group,part_type=source.get("part_type","SEAT_SURFACE"),
                    sample_index=len(samples),alpha=alpha,scale_xyz=[alpha,1.,1.],
                    augmentation="CLEAN" if variant == 0 else "X_WIDTH",noise_seed=seeds[0],
                    epsilon_seed=seeds[1],time_seed=seeds[2],permutation_seed=seeds[3]))
                exposures[source["condition_id"]] += 1
        steps.append(dict(step=step,stage="NATIVE",stage_step=step,samples=samples))
    return dict(schema=SCHEMA,arm=arm,seed=seed,updates=updates,batch=8,
        old_parents_per_update=3,fourth_slot_policy="FOCUS" if arm == "CONTROL_OLD" else "odd_FOCUS_even_HIGH",
        condition_exposure=dict(sorted(exposures.items())),steps=steps,
        training_budget_authorized_by_this_file=False,
        policy="Original separate shuffled parent cycles; OLD3 every update; fourth slot FOCUS or alternating FOCUS/HIGH; two CLEAN/X_WIDTH instances at unchanged natural N; paired slot RNG; unchanged whole-batch free-coordinate FM.")


def make(train, updates, arm, seed=DEFAULT_SEED):
    return make_schedule(train,arm,updates,seed)


def validate(schedule, train, updates=UPDATES, arm=None):
    arm = schedule.get("arm") if arm is None else arm
    groups = groups_for(train)
    if arm not in ARMS or schedule.get("arm") != arm or schedule.get("schema") != SCHEMA or schedule.get("updates") != updates:
        raise ValueError("Frozen stream identity/budget differs")
    if schedule != make_schedule(train,arm,updates,schedule["seed"]):
        raise ValueError("Stream differs from the registered deterministic policy")
    counts = collections.Counter()
    for entry in schedule["steps"]:
        batch = entry["samples"]
        if len(batch) != 8 or len({r["parent_id"] for r in batch}) != 4:
            raise ValueError("Exactly four distinct parents and eight instances per update")
        if any(batch[j]["parent_id"] != batch[j+1]["parent_id"] or
               (batch[j]["augmentation"],batch[j+1]["augmentation"]) != ("CLEAN","X_WIDTH")
               for j in (0,2,4,6)):
            raise ValueError("Paired clean/width instances changed")
        counts.update(r["parent_id"] for r in batch)
    per_group = {g:[counts[p] for p in groups[g]] for g in groups}
    if any(max(v)-min(v)>2 for v in per_group.values()):
        raise ValueError("Within-group parent exposure is not balanced")
    expected = {"OLD":9000,"FOCUS":3000,"HIGH":0} if arm == "CONTROL_OLD" else {"OLD":9000,"FOCUS":1500,"HIGH":1500}
    if {g:sum(v) for g,v in per_group.items()} != expected:
        raise ValueError("Authorized group exposure totals differ")
    return dict(status="PASS",arm=arm,updates=updates,instances=8*updates,
        old_instances=sum(per_group["OLD"]),focus_instances=sum(per_group["FOCUS"]),
        high_instances=sum(per_group["HIGH"]),parent_exposure=dict(sorted(counts.items())),
        all_registered_parent_exposure={p:counts[p] for p in sorted(p for v in groups.values() for p in v)})


def validate_pair(control, natural, train):
    """Check the intended paired stochastic stream without generating arrays."""
    result = {a:validate(s,train,UPDATES,a) for a,s in zip(ARMS,(control,natural))}
    random_fields = ("alpha","scale_xyz","augmentation","noise_seed","epsilon_seed","time_seed","permutation_seed")
    for left,right in zip(control["steps"],natural["steps"]):
        for index,(a,b) in enumerate(zip(left["samples"],right["samples"])):
            if any(a[k] != b[k] for k in random_fields):
                raise ValueError("Paired slot noise/time/augmentation identities differ")
            if index < 6 and a != b:
                raise ValueError("The OLD3 samples must be identical in both arms")
    return dict(status="PASS",paired_updates=UPDATES,paired_instances=UPDATES*8,arms=result)

