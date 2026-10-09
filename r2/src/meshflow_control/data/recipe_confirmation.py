"""Independent input-stream confirmation for an explicitly selected recipe pair.

This module does not authorize execution. It preserves the existing heterogeneous
20/40 condition pools and all original OT/FM arithmetic. Only the training input
random stream is repeated independently; official/Geo initialization is unchanged.
No historical paired-input table or historical OT cache is consumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import time

import numpy as np

from . import recipe_factorial as recipes
from . import schedule40
from . import stream as base
from .dataset import TrainingDataset
from .io import array_hash, atomic_json, digest, freeze_json, json_hash, read_json
from ..training.coupling import CONTRACT

PLAN_SCHEMA = "meshflow_recipe_confirmation_plan_v1"
NAMESPACE_PREFIX = "CHAIR_CONDITION_RECIPE_FACTORIAL_V1/confirmation/"
NAMESPACE = NAMESPACE_PREFIX + "input_repeat1"
TOTAL_UPDATES = 6000
PAIR_SCHEMA = "meshflow_recipe_confirmation_paired_inputs_v1"
DOMAINS = ("parent_schedule", "task40", "alpha", "alpha_slot_shuffle",
           "gaussian", "epsilon", "permutation", "time")


def validate_namespace(namespace):
    if not isinstance(namespace, str) or not namespace.startswith(NAMESPACE_PREFIX):
        raise ValueError("A distinct confirmation input namespace is required")
    suffix = namespace[len(NAMESPACE_PREFIX):]
    if not suffix or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_/-]*", suffix):
        raise ValueError("Invalid confirmation input namespace suffix")
    return namespace


def domain_seed(namespace, domain, *identity):
    validate_namespace(namespace)
    return recipes.domain_seed(namespace, "confirmation_base/" + domain, *identity)


def source_identity():
    root = Path(__file__).parents[1]
    paths = (Path(__file__), Path(recipes.__file__), Path(schedule40.__file__),
             Path(base.__file__), Path(__file__).with_name("dataset.py"),
             Path(__file__).with_name("io.py"), root / "training/coupling.py")
    return {str(p.relative_to(root)): digest(p) for p in paths}


def selected_schedules(selected_pair, hybrid_schedules=None):
    if not isinstance(selected_pair, (list, tuple)) or len(selected_pair) != 2:
        raise ValueError("An explicit candidate and matched control are required")
    pair = [recipes.canonical_recipe(r) for r in selected_pair]
    if len(set(pair)) != 2:
        raise ValueError("Confirmation requires two different registered recipes")
    supplied = {} if hybrid_schedules is None else dict(hybrid_schedules)
    if set(supplied) - set(pair):
        raise ValueError("HYBRID schedule supplied for an unselected recipe")
    resolved = {}
    for recipe in pair:
        fixed = "late" if recipe in ("R3_40_H_LATE", "R4_MIX_H_LATE") else "all"
        if recipe.startswith("C") and recipe not in supplied:
            raise ValueError("Course HYBRID schedule must be explicit")
        policy = supplied.get(recipe, fixed)
        recipes.lambda_for_step(recipe, 1, policy)
        resolved[recipe] = policy
    return pair, resolved


def make_base_plan(manifest, namespace=NAMESPACE, total_updates=TOTAL_UPDATES):
    """Same 26-parent schedule algorithm, with every random domain rederived."""
    validate_namespace(namespace)
    if type(total_updates) is not int or not 6 <= total_updates <= TOTAL_UPDATES or total_updates % 6:
        raise ValueError("Confirmation updates must be a positive multiple of six up to6000")
    tasks = schedule40.eligible_tasks(manifest)
    parents = sorted(tasks)
    rng = {name: np.random.RandomState(domain_seed(namespace, name))
           for name in DOMAINS[:4]}
    records = []
    block = []
    for batch in range(total_updates):
        if batch % 13 == 0:
            permutation = rng["parent_schedule"].permutation(26)
            block = [(parents[int(permutation[i % 26])],
                      ((i % 26) % 2 + i // 26) % 2 == 0) for i in range(104)]
        entries = block[(batch % 13) * 8:(batch % 13 + 1) * 8]
        entries = [entries[int(i)] for i in rng["alpha_slot_shuffle"].permutation(8)]
        for slot, (uid, clean) in enumerate(entries):
            index = 8 * batch + slot
            options = tasks[uid]
            task = options[int(rng["task40"].randint(len(options)))]
            alpha = np.float32(1) if clean else np.float32(rng["alpha"].uniform(.9, 1.1))
            if not clean and alpha == np.float32(1):
                raise ValueError("An augmented draw rounded to alpha1; no redraw permitted")
            records.append(dict(sample_index=index, origin_sample_index=index,
                effective_batch_index=batch, slot_index=slot, step=batch + 1,
                conditional_total_step=batch + 1, uid=uid, task_id=task["task_id"],
                N=int(task["N"]), K=int(task["K"]), ratio=.4, alpha=float(alpha),
                gaussian_seed=domain_seed(namespace, "gaussian", index),
                epsilon_seed=domain_seed(namespace, "epsilon", index),
                permutation_seed=domain_seed(namespace, "permutation", index),
                t_seed=domain_seed(namespace, "time", index)))
        if len({r["uid"] for r in records[-8:]}) != 8 or sum(r["alpha"] == 1 for r in records[-8:]) != 4:
            raise ValueError("Parent/augmentation batch contract differs")
    return dict(namespace=namespace, effective_batch_count=total_updates,
                eligible_parents=parents, eligible_task_count=203,
                eligible_tasks={uid: [r["task_id"] for r in rows] for uid, rows in tasks.items()},
                base_domain_seeds={name: domain_seed(namespace, name) for name in DOMAINS},
                records=records)


def make_plan(manifest, selected_pair, hybrid_schedules=None, *, namespace=NAMESPACE,
              total_updates=TOTAL_UPDATES):
    pair, policies = selected_schedules(selected_pair, hybrid_schedules)
    mother = make_base_plan(manifest, namespace, total_updates)
    old = mother["records"]
    options = recipes.eligible20(manifest, mother["eligible_parents"])
    mixed, pure20, pure40 = [], [], []
    balance = {uid: 0 for uid in mother["eligible_parents"]}
    for start in range(0, len(old), 16):
        source = old[start:start + 16]
        mask = recipes.choose_ratio_mask(source, balance, namespace, start // 16)
        ids_by_ratio = {.2: [], .4: []}
        for offset, original in enumerate(source):
            index = start + offset
            record = dict(original)
            if mask[offset]:
                candidates = options[record["uid"]]
                rng = np.random.RandomState(domain_seed(namespace, "task20", index, record["uid"]))
                task = candidates[int(rng.randint(len(candidates)))]
                if task["N"] != record["N"]:
                    raise ValueError("The20% pool changes total parent face count")
                record.update(task_id=task["task_id"], K=int(task["K"]), ratio=.2)
            mixed.append(record)
            ids_by_ratio[record["ratio"]].append(index)
        for ratio, target in ((.2, pure20), (.4, pure40)):
            ids = ids_by_ratio[ratio]
            chosen = [mixed[index] for index in ids]
            if len(ids) != 8 or len({r["uid"] for r in chosen}) != 8 or sum(r["alpha"] == 1 for r in chosen) != 4:
                raise ValueError("Adjacent-batch pure-ratio decomposition failed")
            target.append(ids)
    segment = total_updates // 3
    first = [i for batch in pure20[:segment] for i in batch]
    second = [i for batch in pure40[:segment] for i in batch]
    tail = list(range(16 * segment, len(old)))
    orders = {"C20_40_MIX": first + second + tail, "C40_20_MIX": second + first + tail}
    for order in orders.values():
        if sorted(order) != list(range(len(old))) or order[16 * segment:] != tail:
            raise ValueError("Curriculum changed the MIX multiset or final suffix")
    return dict(schema=PLAN_SCHEMA, namespace=namespace, selected_pair=pair,
        hybrid_schedules=policies, total_updates=total_updates,
        effective_batch_count=total_updates, batch_size=8, microbatch_size=1,
        sample_count=8 * total_updates, independent_training_input_stream=True,
        independent_model_initialization=False, Geo_seed=1010,
        no_historical_pair_or_OT_reuse=True,
        eligible_parents=mother["eligible_parents"], eligible40_task_count=203,
        eligible20_task_count=208, eligible40_tasks=mother["eligible_tasks"],
        eligible20_tasks={uid: [r["task_id"] for r in rows] for uid, rows in options.items()},
        base_domain_seeds=mother["base_domain_seeds"],
        base_records=old, mixed_records=mixed, curriculum_orders=orders,
        curriculum_segment_updates=segment, parent20_minus40_balance=balance,
        coupling_contract=CONTRACT, condition_pool_policy="EXISTING_HETEROGENEOUS_POOLS_UNCHANGED",
        strict_same_candidate_selection_policy="NOT_SATISFIED",
        interpretation="Independent input-stream confirmation of complete recipes; not pure ratio causality or a new initialization seed",
        FM_normalization="Unchanged effective-batch valid free scalar count",
        record_rng_contract="Origin-slot seeds are shared between selected recipes; different K changes actual array dimensions")


def records_for_recipe(plan, recipe):
    recipe = recipes.canonical_recipe(recipe)
    if recipe not in plan["selected_pair"]:
        raise ValueError("Recipe outside the selected confirmation pair")
    if recipe in ("R1_40_H_ALL", "R3_40_H_LATE"):
        source = plan["base_records"]
        order = list(range(len(source)))
    else:
        source = plan["mixed_records"]
        order = plan["curriculum_orders"].get(recipe, list(range(len(source))))
    return [dict(source[origin], origin_sample_index=origin,
                 step=position // 8 + 1, optimizer_step=position // 8 + 1,
                 conditional_total_step=position // 8 + 1,
                 effective_batch_index=position // 8, slot_index=position % 8)
            for position, origin in enumerate(order)]


def cache_record_identity(record):
    # Step/order and lambda never affect free-only OT. All actual random/input
    # fields remain in the comparison; a course shares maps by immutable origin.
    ignored = {"step", "optimizer_step", "conditional_total_step",
               "effective_batch_index", "slot_index"}
    return {key: value for key, value in record.items() if key not in ignored}


def registered_cache_records(plan):
    union = {}
    for recipe in plan["selected_pair"]:
        for record in records_for_recipe(plan, recipe):
            key = (record["origin_sample_index"], record["task_id"])
            if key in union and cache_record_identity(union[key]) != cache_record_identity(record):
                raise ValueError("Same OT slot/task has conflicting registered inputs")
            union[key] = record
    return dict(sorted(union.items()))


def prepare(data_root, stream_root, recipes, hybrid_schedules=None, *, namespace=NAMESPACE,
            total_updates=TOTAL_UPDATES):
    """Freeze inputs only; callers must separately authorize any real execution."""
    if total_updates != TOTAL_UPDATES and "/cpu_fixture/" not in namespace:
        raise ValueError("Real confirmation requires exactly6000 updates")
    data = TrainingDataset(data_root)
    plan = make_plan(data.manifest, recipes, hybrid_schedules,
                     namespace=namespace, total_updates=total_updates)
    plan.update(data_manifest_sha256=data.manifest_sha256, implementation_sources=source_identity())
    path = Path(stream_root).resolve() / "confirmation_plan.json"
    plan_sha = freeze_json(path, plan)
    return dict(status="PREPARED_NO_OT", plan_path=str(path), plan_sha256=plan_sha,
                namespace=namespace, selected_pair=plan["selected_pair"],
                registered_unique_OT_inputs=len(registered_cache_records(plan)),
                historical_OT_reused=0, model_calls=0, optimizer_updates=0)


class ConfirmationStream(base.HybridStream):
    def __init__(self, data_root, stream_root, recipe, hybrid_schedule=None, require_cache=True):
        self.out = Path(stream_root).resolve()
        self.plan_path = self.out / "confirmation_plan.json"
        self.plan = read_json(self.plan_path)
        self.plan_sha256 = digest(self.plan_path)
        if self.plan.get("schema") != PLAN_SCHEMA:
            raise ValueError("Wrong confirmation plan schema")
        self.namespace = validate_namespace(self.plan["namespace"])
        self.data = TrainingDataset(data_root)
        expected = make_plan(self.data.manifest, self.plan["selected_pair"], self.plan["hybrid_schedules"],
                             namespace=self.namespace, total_updates=self.plan["total_updates"])
        expected.update(data_manifest_sha256=self.data.manifest_sha256, implementation_sources=source_identity())
        if self.plan != expected:
            raise ValueError("Frozen independent confirmation plan does not reproduce")
        self.recipe = recipes.canonical_recipe(recipe)
        self.arm = self.recipe
        self.records = records_for_recipe(self.plan, self.recipe)
        self.hybrid_schedule = self.plan["hybrid_schedules"][self.recipe]
        if hybrid_schedule is not None and hybrid_schedule != self.hybrid_schedule:
            raise ValueError("HYBRID schedule differs from selected pair registration")
        self.total_updates = self.plan["total_updates"]
        self.batch_index = 0
        self.require_cache = bool(require_cache)
        self.cache_dir = self.out / "confirmation_ot_cache"
        self.paired_path = self.out / "paired_inputs" / (self.recipe + ".json")
        self.implementation_sha256 = json_hash(source_identity())
        self.counts = {"actual_OT_attempts": 0, "actual_OT_returns": 0, "OT_cache_hits": 0}
        self._by_origin = {r["origin_sample_index"]: r for r in self.records}
        self._union = registered_cache_records(self.plan)

    def state_dict(self):
        return dict(schema="meshflow_recipe_confirmation_stream_state_v1",
                    namespace=self.namespace, recipe=self.recipe, hybrid_schedule=self.hybrid_schedule,
                    total_updates=self.total_updates, batch_index=self.batch_index,
                    consumed_samples=8 * self.batch_index, plan_sha256=self.plan_sha256,
                    data_manifest_sha256=self.data.manifest_sha256,
                    implementation_sha256=self.implementation_sha256)

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError("Confirmation stream state keys differ")
        for key, value in expected.items():
            if key not in ("batch_index", "consumed_samples") and state[key] != value:
                raise ValueError("Confirmation restore identity differs: " + key)
        cursor = state["batch_index"]
        if type(cursor) is not int or not 0 <= cursor <= self.total_updates or state["consumed_samples"] != 8 * cursor:
            raise ValueError("Invalid confirmation stream cursor")
        self.batch_index = cursor

    def _mapping(self, pre, noise, source_ids, corners, known_ids, record):
        key = (record["origin_sample_index"], record["task_id"])
        if key not in self._union or cache_record_identity(record) != cache_record_identity(self._union[key]):
            raise ValueError("OT input is outside the selected pair's frozen union")
        if self.require_cache:
            identity = dict(contract=CONTRACT, implementation_sha256=self.implementation_sha256,
                pre_target=array_hash(pre), raw_noise=array_hash(noise), source_ids=array_hash(source_ids),
                corners=array_hash(corners), known_ids=array_hash(known_ids), alpha=record["alpha"],
                task_id=record["task_id"], sample_index=record["sample_index"],
                plan_sha256=self.plan_sha256, data_manifest_sha256=self.data.manifest_sha256)
            path = self.cache_dir / (str(record["sample_index"]).zfill(4) + "_" + json_hash(identity) + ".npz")
            if not path.is_file():
                raise RuntimeError("Confirmation OT must be prepared; historical cache substitution is forbidden")
        return super()._mapping(pre, noise, source_ids, corners, known_ids, record)

    def sample(self, record):
        if record != self._by_origin.get(record.get("origin_sample_index")):
            raise ValueError("Sample differs from its selected recipe record")
        sample = base.HybridStream.sample(self, record)
        lam = recipes.lambda_for_step(self.recipe, record["step"], self.hybrid_schedule)
        result = recipes.set_coupling(sample, lam)
        result.update(origin_sample_index=record["origin_sample_index"],
                      optimizer_step=record["step"], step=record["step"],
                      condition_policy="40" if self.recipe in ("R1_40_H_ALL", "R3_40_H_LATE") else "MIX_MULTISET",
                      cache_origin="INDEPENDENT_CONFIRMATION_SHARED")
        return result

    def samples(self, step):
        if type(step) is not int or not 1 <= step <= self.total_updates:
            raise ValueError("Step outside confirmation stream")
        return [self.sample(row) for row in self.records[8 * (step - 1):8 * step]]

    def next_effective_batch(self):
        samples = self.samples(self.batch_index + 1)
        self.batch_index += 1
        self.last_audit = dict(batch_index=self.batch_index,
            shared_batch_sha256=json_hash([s["shared_input_sha256"] for s in samples]),
            input_batch_sha256=json_hash([s["input_bytehash"] for s in samples]),
            label_batch_sha256=json_hash([s["label_bytehash"] for s in samples]),
            effective_free_coordinates=sum(s["free_coordinate_count"] for s in samples),
            ratio20_samples=sum(s["ratio"] == .2 for s in samples),
            ratio40_samples=sum(s["ratio"] == .4 for s in samples),
            lambda_value=samples[0]["lambda_value"],
            origin_sample_indices=[s["origin_sample_index"] for s in samples])
        return samples


def precompute(data_root, stream_root, workers=4):
    """Prepare only the selected pair's union; thread order never defines RNG."""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("CPU OT worker count must be1..8")
    out = Path(stream_root).resolve()
    plan = read_json(out / "confirmation_plan.json")
    stream = ConfirmationStream(data_root, out, plan["selected_pair"][0], require_cache=False)
    union = stream._union
    for path in stream.cache_dir.glob("*.attempt.json"):
        if not path.with_suffix("").with_suffix(".npz").is_file():
            raise RuntimeError("Unresolved prior confirmation OT attempt; no silent retry")
    required_lambdas = {key: set() for key in union}
    records = {label: records_for_recipe(plan, label) for label in plan["selected_pair"]}
    for label, rows in records.items():
        for row in rows:
            key = (row["origin_sample_index"], row["task_id"])
            required_lambdas[key].add(recipes.lambda_for_step(label, row["step"], plan["hybrid_schedules"][label]))
    started = time.perf_counter()
    attempt_id = str(time.time_ns())
    before_attempts = {p.name for p in stream.cache_dir.glob("*.attempt.json")}
    before_returns = {p.name for p in stream.cache_dir.glob("*.return.json")}
    complete, cache_hits, new_returns = {}, 0, 0
    def one(item):
        key, row = item
        sample = base.HybridStream.sample(stream, row)
        sample["origin_sample_index"] = row["origin_sample_index"]
        by_lambda = {lam: recipes.paired_record(recipes.set_coupling(sample, lam))
                     for lam in sorted(required_lambdas[key])}
        return key, by_lambda, sample["ot_cache_status"]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, values, status in pool.map(one, union.items()):
                complete[key] = values
                cache_hits += status == "HIT"
                new_returns += status == "MISS"
                if len(complete) % 256 == 0 or len(complete) == len(union):
                    atomic_json(out / "precompute_progress.json", dict(status="RUNNING",
                        completed_unique_inputs=len(complete), registered_unique_inputs=len(union),
                        actual_new_OT_returns=new_returns, cache_hits=cache_hits,
                        wall_seconds=time.perf_counter() - started))
        for label, rows in records.items():
            items = []
            for row in rows:
                lam = recipes.lambda_for_step(label, row["step"], plan["hybrid_schedules"][label])
                pair = dict(complete[row["origin_sample_index"], row["task_id"]][lam])
                pair.update(step=row["step"], uid=row["uid"], N=row["N"], K=row["K"],
                            ratio=row["ratio"], alpha=row["alpha"])
                items.append(pair)
            if sorted(r["sample_index"] for r in items) != list(range(plan["sample_count"])):
                raise ValueError("Confirmation paired table is incomplete")
            freeze_json(out / "paired_inputs" / (label + ".json"), dict(schema=PAIR_SCHEMA,
                recipe=label, namespace=stream.namespace, selected_pair=plan["selected_pair"],
                hybrid_schedule=plan["hybrid_schedules"][label], plan_sha256=stream.plan_sha256,
                data_manifest_sha256=stream.data.manifest_sha256, implementation_sha256=stream.implementation_sha256,
                historical_paired_inputs_reused=False, records=items))
    except BaseException as exc:
        failure = dict(status="BLOCKED", error=repr(exc), namespace=stream.namespace,
            plan_sha256=stream.plan_sha256, completed_aggregated_inputs=len(complete),
            new_attempt_files=len({p.name for p in stream.cache_dir.glob("*.attempt.json")} - before_attempts),
            new_return_files=len({p.name for p in stream.cache_dir.glob("*.return.json")} - before_returns),
            wall_seconds=time.perf_counter() - started, model_calls=0)
        freeze_json(out / "precompute_attempts" / (attempt_id + ".json"), failure)
        atomic_json(out / "precompute_progress.json", failure)
        raise
    actual_attempts = len({p.name for p in stream.cache_dir.glob("*.attempt.json")} - before_attempts)
    actual_returns = len({p.name for p in stream.cache_dir.glob("*.return.json")} - before_returns)
    if actual_attempts != new_returns or actual_returns != new_returns:
        raise ValueError("Completed-return aggregation differs from immutable OT attempt/return files")
    receipt = dict(status="PASS", namespace=stream.namespace, selected_pair=plan["selected_pair"],
        plan_sha256=stream.plan_sha256, registered_unique_OT_inputs=len(union),
        completed_unique_OT_inputs=len(complete), actual_OT_attempts=actual_attempts,
        actual_OT_returns=actual_returns, OT_cache_hits=cache_hits, historical_OT_reused=0,
        paired_input_rows_per_recipe={label: len(rows) for label, rows in records.items()},
        workers=workers, executor_type="ThreadPoolExecutor",
        accounting_authority="Main-thread ordered returns checked against immutable per-map receipts",
        wall_seconds=time.perf_counter() - started, model_calls=0, optimizer_updates=0)
    freeze_json(out / "precompute_attempts" / (attempt_id + ".json"), receipt)
    atomic_json(out / "precompute_receipt.json", receipt)
    atomic_json(out / "precompute_progress.json", dict(receipt, status="COMPLETE"))
    return receipt


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate-hybrid", choices=("all", "late"))
    parser.add_argument("--control-hybrid", choices=("all", "late"))
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--precompute", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    pair = [recipes.canonical_recipe(args.candidate), recipes.canonical_recipe(args.control)]
    schedules = {label: policy for label, policy in zip(pair, [args.candidate_hybrid, args.control_hybrid]) if policy}
    print(json.dumps(prepare(args.data, args.out, pair, schedules, namespace=args.namespace)), flush=True)
    if args.precompute:
        print(json.dumps(precompute(args.data, args.out, args.workers)), flush=True)


if __name__ == "__main__":
    main()