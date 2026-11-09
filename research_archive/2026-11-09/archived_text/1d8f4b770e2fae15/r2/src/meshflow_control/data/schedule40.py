"""Frozen 40%-condition HYBRID inputs for the early/late alignment schedule pilot.

This module changes registration and scheduling only. The inherited sample method
retains the audited free-only OT, six-corner matching, x2 target scaling and
sqrt(.75) Z + .5 epsilon mathematics. It performs no model forward.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from . import stream as base
from .dataset import TrainingDataset
from .io import array_hash, atomic_json, digest, freeze_json, json_hash, read_json
from ..training.coupling import CONTRACT

NAMESPACE = "CHAIR_EARLY_ALIGNMENT_SCHEDULE_40_V1/train/v1"
SEED = int.from_bytes(hashlib.sha256(NAMESPACE.encode()).digest()[:4], "little")
TOTAL_UPDATES = 6000
BATCH_SIZE = 8


def domain_seed(domain, *identity):
    payload = json.dumps([NAMESPACE, SEED, domain, *identity], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def source_identity():
    paths = (Path(__file__), Path(base.__file__),
             Path(__file__).with_name("dataset.py"),
             Path(__file__).parents[1] / "training/coupling.py")
    return {p.name: digest(p) for p in paths}


def eligible_tasks(manifest):
    """Use existing READY 40% tasks; never construct substitute conditions."""
    train = set(manifest["splits"]["train"])
    tasks = defaultdict(list)
    for row in manifest["tasks"]:
        if row["role"] != "train" or row["status"] != "READY":
            continue
        if float(row["ratio"]) != .4:
            continue
        if row["uid"] not in train or row.get("split", "train") != "train":
            raise ValueError("Eligible task outside training split")
        n, k = int(row["N"]), int(row["K"])
        if not 128 <= n <= 256 or k != round(.4 * n):
            raise ValueError("40% face count contract differs")
        tasks[row["uid"]].append(row)
    if len(tasks) != 26 or sum(map(len, tasks.values())) != 203:
        raise ValueError("Expected exactly 26 eligible parents and 203 existing 40% tasks")
    return {uid: sorted(rows, key=lambda r: r["task_id"])
            for uid, rows in sorted(tasks.items())}


def make_plan(manifest, total_updates=TOTAL_UPDATES):
    if type(total_updates) is not int or not 1 <= total_updates <= TOTAL_UPDATES:
        raise ValueError("Update count must be an integer in 1..6000")
    tasks = eligible_tasks(manifest)
    parents = sorted(tasks)
    rng = {d: np.random.RandomState(domain_seed(d)) for d in
           ("parent_schedule", "task_schedule", "alpha", "alpha_slot_shuffle")}
    records = []
    # Each 13-batch block has 104 samples: four occurrences per parent, exactly
    # two clean and two augmented. Every eight contiguous positions are distinct
    # because the same 26-permutation is repeated within a block.
    block = []
    for batch in range(total_updates):
        if batch % 13 == 0:
            permutation = rng["parent_schedule"].permutation(26)
            block = [(parents[int(permutation[i % 26])],
                      ((i % 26) % 2 + i // 26) % 2 == 0)
                     for i in range(104)]
        entries = block[(batch % 13)*8:(batch % 13+1)*8]
        entries = [entries[int(i)] for i in rng["alpha_slot_shuffle"].permutation(8)]
        for slot, (uid, clean) in enumerate(entries):
            index = batch*8 + slot
            options = tasks[uid]
            task = options[int(rng["task_schedule"].randint(len(options)))]
            alpha = np.float32(1) if clean else np.float32(rng["alpha"].uniform(.9, 1.1))
            if not clean and alpha == np.float32(1):
                # Do not draw again or silently increase the teacher population.
                raise ValueError("Registered augmented draw rounded exactly to alpha1")
            records.append(dict(sample_index=index, effective_batch_index=batch,
                slot_index=slot, step=batch+1, conditional_total_step=batch+1,
                uid=uid, task_id=task["task_id"], N=int(task["N"]), K=int(task["K"]),
                ratio=.4, alpha=float(alpha),
                gaussian_seed=domain_seed("gaussian", index),
                epsilon_seed=domain_seed("epsilon", index),
                permutation_seed=domain_seed("permutation", index),
                t_seed=domain_seed("time", index)))
        tail = records[-8:]
        if len({r["uid"] for r in tail}) != 8 or sum(r["alpha"] == 1 for r in tail) != 4:
            raise ValueError("Eight distinct parents and four clean examples required")
    windows = {}
    for name, lo, hi in (("all", 1, total_updates),
                         ("early_B", 1, min(1000, total_updates)),
                         ("late_B", 5001, min(6000, total_updates))):
        chosen = [r for r in records if lo <= r["step"] <= hi]
        windows[name] = {
            "step_first": lo, "step_last": hi, "samples": len(chosen),
            "all_by_parent": dict(Counter(r["uid"] for r in chosen)),
            "clean_by_parent": dict(Counter(r["uid"] for r in chosen if r["alpha"] == 1)),
            "augmented_by_parent": dict(Counter(r["uid"] for r in chosen if r["alpha"] != 1))}
    return dict(schema="meshflow_schedule40_train_plan_v1",
        namespace=NAMESPACE, seed=SEED, effective_batch_count=total_updates,
        batch_size=8, microbatch_size=1, sample_count=8*total_updates,
        RNG_mode="stateless_per_sample_frozen_seeds", coupling_lambda=.25,
        parent_distribution="Independent shuffled 26-parent cycle repeated four times per 13 batches; 4 appearances, 2 clean, 2 augmented per full block",
        context_distribution="Uniform among each parent's existing READY 40% conditions",
        alpha_distribution="Four exact alpha1 and four FP32 Uniform[.9,1.1] per batch",
        eligible_parents=parents, eligible_parent_count=26, eligible_task_count=203,
        excluded_training_parents=sorted(set(manifest["splits"]["train"])-set(parents)),
        eligible_tasks={uid: [r["task_id"] for r in rows] for uid, rows in tasks.items()},
        object_counts=dict(Counter(r["uid"] for r in records)), exposure_windows=windows,
        records=records)


def prepare_training_plan(data_root, stream_root, total_updates=TOTAL_UPDATES):
    data = TrainingDataset(data_root)
    out = Path(stream_root).resolve()
    plan = dict(make_plan(data.manifest, total_updates),
                data_manifest_sha256=data.manifest_sha256,
                implementation_sources=source_identity())
    sha = freeze_json(out / "train_plan.json", plan)
    return dict(path=str(out/"train_plan.json"), sha256=sha, seed=SEED,
                namespace=NAMESPACE, samples=8*total_updates,
                updates_per_trajectory=total_updates,
                data_manifest_sha256=data.manifest_sha256,
                eligible_parents=26, eligible_tasks=203)


class Schedule40Stream(base.HybridStream):
    """Same scientific inputs for all arms; cursors can be restored at a fork."""
    def __init__(self, data_root, stream_root, arm="FM_ONLY", namespace=None,
                 require_cache=True):
        self.data = TrainingDataset(data_root)
        self.out = Path(stream_root).resolve()
        self.arm = namespace if namespace is not None else arm
        self.plan_path = self.out / "train_plan.json"
        self.plan = read_json(self.plan_path)
        self.plan_sha256 = digest(self.plan_path)
        updates = self.plan["effective_batch_count"]
        expected = dict(make_plan(self.data.manifest, updates),
                        data_manifest_sha256=self.data.manifest_sha256,
                        implementation_sources=source_identity())
        if self.plan != expected:
            raise ValueError("Frozen schedule40 plan does not reproduce")
        self.total_updates = updates
        self.records = self.plan["records"]
        self.batch_index = 0
        self.cache_dir = self.out / "ot_cache"
        self.require_cache = bool(require_cache)
        self.counts = {"actual_OT_attempts": 0, "actual_OT_returns": 0, "OT_cache_hits": 0}
        self.implementation_sha256 = json_hash(source_identity())

    def state_dict(self):
        # The arm is deliberately absent: a complete COMMON5000 state must be
        # clonable into FM_ONLY and LATE without changing the input identity.
        return dict(schema="meshflow_schedule40_stream_state_v1",
            batch_index=self.batch_index, sample_index=8*self.batch_index,
            namespace=NAMESPACE, seed=SEED, total_updates=self.total_updates,
            plan_sha256=self.plan_sha256, data_manifest_sha256=self.data.manifest_sha256,
            coupling_contract=CONTRACT, lambda_value=.25,
            implementation_sha256=self.implementation_sha256)

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError("Stream state keys differ")
        for key in expected:
            if key not in ("batch_index", "sample_index") and state[key] != expected[key]:
                raise ValueError("Stream restore identity differs: " + key)
        cursor = state["batch_index"]
        if type(cursor) is not int or not 0 <= cursor <= self.total_updates or state["sample_index"] != 8*cursor:
            raise ValueError("Invalid schedule40 stream cursor")
        self.batch_index = cursor

    def _mapping(self, pre, noise, source_ids, corners, known_ids, record):
        if self.require_cache:
            identity = dict(contract=CONTRACT, implementation_sha256=self.implementation_sha256,
                pre_target=array_hash(pre), raw_noise=array_hash(noise), source_ids=array_hash(source_ids),
                corners=array_hash(corners), known_ids=array_hash(known_ids), alpha=record["alpha"],
                task_id=record["task_id"], sample_index=record["sample_index"],
                plan_sha256=self.plan_sha256, data_manifest_sha256=self.data.manifest_sha256)
            path = self.cache_dir / (str(record["sample_index"]).zfill(4)+"_"+json_hash(identity)+".npz")
            if not path.exists():
                raise RuntimeError("CPU OT must be prepared before training: "+str(path))
        return super()._mapping(pre, noise, source_ids, corners, known_ids, record)

    def samples(self, step):
        if type(step) is not int or not 1 <= step <= self.total_updates:
            raise ValueError("Step outside frozen plan")
        return [self.sample(r) for r in self.records[8*(step-1):8*step]]

    def next_effective_batch(self):
        samples = self.samples(self.batch_index + 1)
        self.batch_index += 1
        self.last_audit = dict(batch_index=self.batch_index,
            shared_batch_sha256=json_hash([s["shared_input_sha256"] for s in samples]),
            input_batch_sha256=json_hash([s["input_bytehash"] for s in samples]),
            label_batch_sha256=json_hash([s["label_bytehash"] for s in samples]),
            effective_free_coordinates=sum(s["free_coordinate_count"] for s in samples))
        return samples


def precompute(data_root, stream_root, workers=4):
    """Prepare only registered OT maps; completed maps resume without re-solving."""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("CPU OT workers must be in 1..8")
    out = Path(stream_root)
    stream = Schedule40Stream(data_root, out, require_cache=False)
    for path in stream.cache_dir.glob("*.attempt.json"):
        if not path.with_suffix("").with_suffix(".npz").exists():
            raise RuntimeError("Unresolved prior OT attempt; no silent retry: "+str(path))
    started = time.perf_counter()
    fields = ("sample_index", "task_id", "shared_input_sha256", "input_bytehash",
              "label_bytehash", "ot_cache_key", "ot_cache_status", "free_coordinate_count")
    def compute(record):
        sample = stream.sample(record)
        return {key: sample[key] for key in fields}
    count = len(stream.records)
    items = []
    completed_new = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for item in pool.map(compute, stream.records):
            items.append(item)
            completed_new += item["ot_cache_status"] == "MISS"
            if len(items) % 256 == 0 or len(items) == count:
                atomic_json(out/"precompute_progress.json",
                    dict(status="RUNNING" if len(items) < count else "MAPS_COMPLETE",
                         completed=len(items), registered=count, workers=workers,
                         actual_OT_returns_this_run=completed_new,
                         cache_hits_this_run=len(items)-completed_new,
                         elapsed_seconds=time.perf_counter()-started,
                         plan_sha256=stream.plan_sha256))
    if [r["sample_index"] for r in items] != list(range(count)):
        raise ValueError("Incomplete registered OT preparation")
    new = sum(r["ot_cache_status"] == "MISS" for r in items)
    freeze_json(out/"paired_inputs.json",
        dict(schema="meshflow_schedule40_paired_inputs_v1",
             plan_sha256=stream.plan_sha256, data_manifest_sha256=stream.data.manifest_sha256,
             records=[{k: v for k, v in r.items() if k != "ot_cache_status"} for r in items]))
    receipt = dict(status="PASS", namespace=NAMESPACE, seed=SEED, workers=workers,
        samples=count, actual_OT_attempts_this_run=new, actual_OT_returns_this_run=new,
        cache_hits_this_run=count-new, wall_seconds=time.perf_counter()-started,
        model_forwards=0, model_backwards=0, optimizer_updates=0, generation_requests=0,
        plan_sha256=stream.plan_sha256, data_manifest_sha256=stream.data.manifest_sha256,
        implementation_sha256=stream.implementation_sha256)
    receipt_root = out/"precompute_attempts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_root/(str(time.time_ns())+".json")
    freeze_json(receipt_path, receipt)
    atomic_json(out/"precompute_receipt.json", dict(receipt, attempt_receipt=str(receipt_path)))
    return receipt


def prepare_stream(data_root, out_root, total_updates=TOTAL_UPDATES, workers=4):
    prepare_training_plan(data_root, out_root, total_updates)
    return precompute(data_root, out_root, workers)


def prepare_teacher_registration(data_root, stream_root, old_teacher_root, out_root):
    """Reuse exact old features with new training IDs; never load/run the teacher.

    The old global training camera is deliberately retained. Re-fitting a camera
    to the eligible 26-parent subset would produce a different teacher target.
    """
    from ..teachers import siglip_targets as teacher
    from ..teachers.siglip_render import array_hash as teacher_hash, RENDER_CONFIG, json_hash as teacher_json_hash

    data = TrainingDataset(data_root)
    stream = Schedule40Stream(data_root, stream_root)
    old_root = Path(old_teacher_root).resolve()
    out = Path(out_root).resolve()
    old_path = old_root/"teacher_manifest.json"
    old = read_json(old_path)
    folder = old_root/"teacher"
    render_path = folder/"render_manifest.json"
    encoding_path = folder/"encoding_plan.json"
    assets_path = folder/"assets/assets_verified.json"
    render = read_json(render_path)
    encoding = read_json(encoding_path)
    assets = read_json(assets_path)
    if old["status"] != "COMPLETE" or render["status"] != "COMPLETE" or assets["status"] != "PASS":
        raise ValueError("Original teacher evidence incomplete")
    for key, value in (("model_id", teacher.MODEL_ID), ("revision", teacher.REVISION),
                       ("weight_sha256", teacher.WEIGHT_SHA256)):
        if old[key] != value:
            raise ValueError("Teacher identity differs: "+key)
    if old["render_manifest_sha256"] != digest(render_path) or old["encoding_plan_sha256"] != digest(encoding_path):
        raise ValueError("Teacher render/encoding manifest changed")
    if encoding["assets_sha256"] != digest(assets_path) or render["identity"]["assets_sha256"] != digest(assets_path):
        raise ValueError("Teacher asset registration changed")
    current_sources = teacher.source_identity()
    if encoding["sources"] != current_sources:
        raise ValueError("Teacher encoding implementation changed")
    render_source_audit = dict(render_sources=render["sources"], encoding_sources=current_sources,
                              rendering_functions_unchanged=True)
    if render["sources"] != current_sources:
        # The original run added resource instrumentation to encode_targets
        # after rendering. Its separately archived rendering code must match
        # the old manifest. Outside encode_targets the only accepted change
        # is the reviewed completed-render-manifest reuse guard below.
        import ast
        archived = folder/"render_source_snapshot"
        for name, sha in render["sources"].items():
            if digest(archived/name) != sha:
                raise ValueError("Archived teacher render source changed: "+name)
            if name != "siglip_targets.py" and sha != current_sources[name]:
                raise ValueError("Teacher renderer/assets source changed")
        def excluding_encode(path):
            text = Path(path).read_text(encoding="utf-8-sig")
            reuse_guard = '''    if (folder/"render_manifest.json").exists():
        previous=read(folder/"render_manifest.json")
        if previous["identity"]!=manifest["identity"] or previous["rows"]!=manifest["rows"]:
            raise ValueError("Frozen training render geometry/config/receipt differs")
        manifest=previous
    else:freeze(folder/"render_manifest.json",manifest)'''
            text = text.replace(reuse_guard, '    freeze(folder/"render_manifest.json",manifest)')
            tree = ast.parse(text)
            tree.body = [node for node in tree.body
                         if not isinstance(node, ast.FunctionDef) or node.name != "encode_targets"]
            return ast.dump(tree, include_attributes=False)
        if excluding_encode(archived/"siglip_targets.py") != excluding_encode(teacher.__file__):
            raise ValueError("Teacher render/normalization logic changed")
        render_source_audit.update(archive=str(archived),
            exception="Only encode_targets instrumentation/resume and exact reviewed render-manifest reuse guard differ; archived render/current encode hashes match original manifests")
    if render["identity"]["inputs"] != old["inputs"] or old["inputs"]["data_manifest_sha256"] != data.manifest_sha256:
        raise ValueError("Teacher training data identity differs")
    for row in assets["files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Original teacher asset changed: "+row["name"])
    config = render["identity"]["render_config"]
    if config["renderer"] != RENDER_CONFIG or config["renderer_source_sha256"] != digest(Path(teacher.__file__).with_name("siglip_render.py")):
        raise ValueError("Teacher renderer definition changed")
    if teacher_json_hash(config) != render["identity"]["render_config_sha256"]:
        raise ValueError("Teacher camera/render configuration changed")
    if digest(folder/"assets/preprocessor_config.json") != old["processor_sha256"]:
        raise ValueError("Teacher processor changed")
    render_rows = {r["geometry_sha256"]: r for r in render["rows"]}
    feature_rows = {r["geometry_sha256"]: r for r in old["rows"]}
    selected = [r for r in stream.records if r["alpha"] == 1.]
    task_ids = sorted({r["task_id"] for r in selected})
    tasks = {}
    hashes = set()
    for task_id in task_ids:
        task = data.tasks[task_id]
        case = data.task(task_id, 1.)
        full = np.ascontiguousarray(case["full_target"].reshape(-1, 3, 3))
        gh = teacher_hash(full)
        if gh not in render_rows or gh not in feature_rows:
            raise ValueError("No exact existing teacher geometry: "+task_id)
        rr = render_rows[gh]
        if digest(rr["geometry_path"]) != rr["geometry_file_sha256"]:
            raise ValueError("Original teacher geometry file changed")
        with np.load(rr["geometry_path"], allow_pickle=False) as z:
            if teacher_hash(z["full_target"]) != gh or not np.array_equal(z["full_target"], full):
                raise ValueError("Actual clean full target does not match teacher geometry")
        tasks[task_id] = dict(geometry_sha256=gh, uid=task["uid"], N=int(task["N"]), K=int(task["K"]))
        hashes.add(gh)
    features = []
    for gh in sorted(hashes):
        rr, fr = render_rows[gh], feature_rows[gh]
        if len(rr["views"]) != 4:
            raise ValueError("Teacher must retain four views")
        for view in rr["views"]:
            if digest(view["path"]) != view["png_sha256"]:
                raise ValueError("Original teacher image changed")
            if view["identity"]["geometry_sha256"] != gh or view["identity"]["render_config_sha256"] != render["identity"]["render_config_sha256"]:
                raise ValueError("Teacher image geometry/camera registration differs")
        if digest(fr["path"]) != fr["sha256"]:
            raise ValueError("Original teacher feature file changed")
        with np.load(fr["path"], allow_pickle=False) as z:
            normalized, target = teacher.normalize_multiview(z["pooler"])
            if not np.array_equal(normalized, z["normalized_views"]) or not np.array_equal(target, z["target"]):
                raise ValueError("Original teacher feature normalization differs")
            if teacher_hash(target) != fr["target_array_sha256"]:
                raise ValueError("Original teacher target array changed")
        features.append(dict(fr, task_ids=sorted(tid for tid, row in tasks.items() if row["geometry_sha256"] == gh),
            training_frequency=sum(tasks[r["task_id"]]["geometry_sha256"] == gh for r in selected)))
    inputs = dict(data_root=str(data.root), data_manifest_path=str(data.manifest_path),
        data_manifest_sha256=data.manifest_sha256, train_plan_path=str(stream.plan_path),
        train_plan_sha256=stream.plan_sha256, alignment_samples=len(selected),
        unique_meshes=len(hashes), unique_tasks=len(tasks), unique_parents=26,
        target_geometry="Exact original alpha1 full_target, C plus clean free verified by source mapping",
        inference_inputs_unchanged=True, tasks=tasks,
        alpha1_samples={str(r["sample_index"]): dict(task_id=r["task_id"], geometry_sha256=tasks[r["task_id"]]["geometry_sha256"]) for r in selected})
    final = dict(schema="meshflow_schedule40_reused_teacher_v1", status="COMPLETE",
        model_id=teacher.MODEL_ID, revision=teacher.REVISION, weight_sha256=teacher.WEIGHT_SHA256,
        processor_sha256=old["processor_sha256"], inputs=inputs, rows=features,
        unique_meshes=len(hashes), images=4*len(hashes),
        model_forwards=0, teacher_forwards=0, newly_rendered_images=0,
        source_teacher_manifest=str(old_path), source_teacher_manifest_sha256=digest(old_path),
        source_render_manifest=str(render_path), source_render_manifest_sha256=digest(render_path),
        source_encoding_plan_sha256=digest(encoding_path), source_assets_sha256=digest(assets_path),
        render_config_sha256=render["identity"]["render_config_sha256"],
        camera_policy="Reuse unchanged original32 training-only global camera; no new26-subset fitting",
        registration_sources=source_identity(), teacher_sources=teacher.source_identity(),
        render_source_audit=render_source_audit,
        supervision="Only alpha1 samples; original FP32 4-view normalized SigLIP targets",
        model_not_loaded_for_training=True)
    freeze_json(out/"teacher_manifest.json", final)
    Schedule40TargetStore(out)
    return dict(status="PASS", path=str(out/"teacher_manifest.json"),
                sha256=digest(out/"teacher_manifest.json"), parents=26,
                clean_samples=len(selected), unique_tasks=len(tasks), teacher_forwards=0)


class Schedule40TargetStore:
    def __init__(self, out):
        from ..teachers.siglip_targets import TargetStore as OriginalStore
        self._store = OriginalStore(out)
        self.path = self._store.path
        self.sha256 = self._store.sha256
        self.manifest = self._store.manifest
        if self.manifest.get("schema") != "meshflow_schedule40_reused_teacher_v1":
            raise ValueError("Wrong teacher registration namespace")

    def for_sample(self, sample):
        return self._store.for_sample(sample)


TargetStore = Schedule40TargetStore


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True, help="Frozen stream directory")
    parser.add_argument("--precompute", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--teacher-source")
    parser.add_argument("--teacher-out")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_training_plan(args.data, args.out)), flush=True)
    if args.teacher_source:
        if not args.teacher_out:
            parser.error("--teacher-source requires --teacher-out")
        print(json.dumps(prepare_teacher_registration(args.data, args.out, args.teacher_source, args.teacher_out)), flush=True)
    if args.precompute:
        print(json.dumps(precompute(args.data, args.out, args.workers)), flush=True)


if __name__ == "__main__":
    main()
