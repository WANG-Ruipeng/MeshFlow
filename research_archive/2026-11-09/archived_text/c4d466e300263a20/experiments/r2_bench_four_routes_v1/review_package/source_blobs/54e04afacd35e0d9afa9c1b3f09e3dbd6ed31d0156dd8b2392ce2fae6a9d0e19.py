"""Frozen wider HIGH-pool Native stream; no model import or global RNG mutation."""
from __future__ import annotations
import collections
import hashlib
import numpy as np

SCHEMA = "BENCH_HIGH_JOINT_COVERAGE_STREAM_V1"
RNG_SCHEMA = "BENCH_NATURAL_N_RECIPE_STREAM_V1"
ARMS = ("WIDE_HIGH",)
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
        if group == "HIGH" and n <= 256:
            raise ValueError("HIGH parents require unchanged natural N257..799")
        groups[group][parent] = i
    if len(groups["OLD"]) != 22 or len(groups["FOCUS"]) != 2 or len(groups["HIGH"]) != 6:
        raise ValueError("Exactly OLD22, original FOCUS2 and the accepted HIGH6 pool required")
    if not all(any(p.startswith(prefix) for p in groups["FOCUS"]) for prefix in ("a8e987", "27eb79")):
        raise ValueError("Original a8/27 FOCUS identities required")
    if not all(any(p.startswith(prefix) for p in groups["HIGH"]) for prefix in ("28b9", "abf250")):
        raise ValueError("Pre-registered 28b9/abf250 HIGH identities required")
    if sorted(int(train[i]["N"]) for p,i in groups["HIGH"].items() if p.startswith(("28b9", "abf250"))) != [308,384]:
        raise ValueError("The two full high-N targets must remain N308/N384")
    return groups


def make_schedule(train, arm, updates=UPDATES, seed=DEFAULT_SEED):
    if arm not in ARMS or type(updates) is not int or updates != UPDATES:
        raise ValueError("Exactly the authorized WIDE_HIGH 1500-update experiment is required")
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
        fourth = "HIGH" if step % 2 == 0 else "FOCUS"
        chosen = take("OLD",3) + take(fourth,1)
        samples = []
        for slot,(group,parent) in enumerate(chosen):
            index = groups[group][parent]
            source = train[index]
            for variant in (0,1):
                # Same slot/step/variant identities in both arms, independent
                # of chosen parent and worker order, as in exposure allocation.
                key = f"{RNG_SCHEMA}|{seed}|{step}|{slot}|{variant}"
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
        old_parents_per_update=3,fourth_slot_policy="odd_FOCUS_even_HIGH",
        condition_exposure=dict(sorted(exposures.items())),steps=steps,
        training_budget_authorized_by_this_file=False,
        policy="Original separate shuffled parent cycles; OLD3 every update; fourth slot odd FOCUS/even HIGH; two CLEAN/X_WIDTH instances at unchanged natural N; paired slot RNG; unchanged whole-batch free-coordinate FM.")


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
    expected = {"OLD":9000,"FOCUS":1500,"HIGH":1500}
    if {g:sum(v) for g,v in per_group.items()} != expected:
        raise ValueError("Authorized group exposure totals differ")
    return dict(status="PASS",arm=arm,updates=updates,instances=8*updates,
        old_instances=sum(per_group["OLD"]),focus_instances=sum(per_group["FOCUS"]),
        high_instances=sum(per_group["HIGH"]),parent_exposure=dict(sorted(counts.items())),
        all_registered_parent_exposure={p:counts[p] for p in sorted(p for v in groups.values() for p in v)})


def validate_manifest(manifest):
    """Only a real, reviewed wider pool may be frozen or dispatched."""
    if manifest.get("status") != "READY_NATIVE_DATA":
        raise ValueError("Actual accepted Native data must be ready")
    groups = groups_for(manifest["train"])
    contract = manifest["high_pool_contract"]
    actual = sorted(groups["HIGH"])
    prior = sorted(p for p in actual if p.startswith(("28b9", "abf250")))
    if (contract.get("status"), contract.get("sampling"), contract.get("natural_N"),
            contract.get("lower_N"), contract.get("upper_N")) != (
            "FROZEN", "balanced_shuffled_parent_cycles", True, 257, 799):
        raise ValueError("Actual HIGH pool admission contract differs")
    if contract.get("parent_ids") != actual or contract.get("prior_parent_ids") != prior:
        raise ValueError("Frozen HIGH parent identities differ")
    if contract.get("new_parent_ids") != sorted(set(actual)-set(prior)):
        raise ValueError("New HIGH parent membership differs")
    heldout = {r["parent_id"] for key in ("dev", "high_holdout", "holdout") for r in manifest.get(key, [])}
    if heldout & {p for v in groups.values() for p in v}:
        raise ValueError("Held-out parent entered the training pool")
    return dict(status="PASS",high_parents=len(actual),new_high_parents=len(actual)-2,
                high_parent_ids=actual,old_parents=22,focus_parents=2)


def validate_narrow(wide, train, narrow, narrow_train):
    from r2.pilots.bench_natural_n_recipe_v1 import stream as prior
    validate(wide, train, UPDATES, "WIDE_HIGH")
    prior.validate(narrow, narrow_train, UPDATES, "NATURAL_HIGH")
    if wide["seed"] != narrow["seed"]:
        raise ValueError("Narrow and wide stream seeds differ")
    random_fields = ("alpha", "scale_xyz", "augmentation", "noise_seed", "epsilon_seed", "time_seed", "permutation_seed")
    paired_old = paired_focus = paired_high = 0
    for left, right in zip(narrow["steps"], wide["steps"]):
        if {k:v for k,v in left.items() if k != "samples"} != {k:v for k,v in right.items() if k != "samples"}:
            raise ValueError("Update/stage ordering changed")
        for old, new in zip(left["samples"], right["samples"]):
            if any(old[k] != new[k] for k in random_fields):
                raise ValueError("Slot augmentation/noise/time/permutation changed")
            if old["replay_group"] != new["replay_group"]:
                raise ValueError("HIGH slot position or OLD/FOCUS exposure changed")
            if old["replay_group"] != "HIGH" and old != new:
                raise ValueError("Original OLD/FOCUS stream row changed")
            if old["replay_group"] == "OLD": paired_old += 1
            elif old["replay_group"] == "FOCUS": paired_focus += 1
            else: paired_high += 1
    if (paired_old,paired_focus,paired_high) != (9000,1500,1500):
        raise ValueError("Exact paired exposure counts differ")
    return dict(status="PASS",paired_updates=UPDATES,exact_old_rows=paired_old,
                exact_focus_rows=paired_focus,paired_high_random_rows=paired_high,
                rng_namespace=RNG_SCHEMA,only_changed_factor="HIGH parent-pool membership and resulting balanced selection")


def validate_frozen_narrow(wide, train):
    from .common import NARROW_ROOT, read
    return validate_narrow(wide, train, read(NARROW_ROOT/"data/schedule_NATURAL_HIGH.json"),
                           read(NARROW_ROOT/"data/manifest.json")["train"])


