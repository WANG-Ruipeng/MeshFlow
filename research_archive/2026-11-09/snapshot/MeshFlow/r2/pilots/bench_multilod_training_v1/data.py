"""Frozen paired parent-first Native schedules and count-only M coefficients.

No environment, model or random-state mutation happens on import. Cases are
cached by case_id: the physical condition_id alone is not a geometry identity.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import numpy as np

from . import common as C
from r2.data_tools.bench_multilod_v1.sampler import tag_exposure_pairs, adapt_schedule

SOURCE_SCHEDULE = C.OLD / "data/schedule_M_PLUS3000.json"
SOURCE_COEFFICIENTS = C.OLD / "data/coefficient_schedule.json"
GROUPS = ("OLD", "FOCUS", "HIGH")
RANDOM_FIELDS = ("noise_seed", "epsilon_seed", "time_seed", "permutation_seed")
SCHEMA = "BENCH_MULTILOD_PAIRED_NATIVE1500_STREAM_V1"
MANIFEST_PATH = C.OUT / "data/manifest.json"


def require(ok, text):
    if not ok:
        raise ValueError(text)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def schedule_path(arm):
    require(arm in C.ARMS, "Unknown arm")
    return C.OUT / f"data/schedule_{arm}.json"


def coefficients_path(arm):
    require(arm in C.ARMS, "Unknown arm")
    return C.OUT / f"data/coefficients_{arm}.json"


def load_case(row):
    require(C.sha(row["case_path"]) == row["case_sha256"], "Case hash changed: " + row["case_id"])
    with np.load(C.resolve(row["case_path"]), allow_pickle=False) as z:
        case = {k: z[k].copy() for k in z.files}
    n, k = int(case["N"]), int(case["K"])
    require((n, k) == (row["N"], row["K"]), "Selected N/K differs from actual case")
    require(128 <= n <= 799 and 2 <= k < n, "Native N/K contract")
    require(case["full_target"].shape == (n, 3, 3) and case["full_target"].dtype == np.float32,
            "Actual full target dtype/shape differs")
    require(np.isfinite(case["full_target"]).all(), "Nonfinite target")
    require(np.array_equal(case["pre_ot_full"]*np.float32(2), case["full_target"]), "Pre-OT half scale differs")
    known, free = case["known_ids"], case["free_ids"]
    require(len(known) == k and len(free) == n-k and np.array_equal(np.sort(np.r_[known, free]), np.arange(n)),
            "C/Y source face IDs do not partition target")
    require(np.array_equal(case["C"], case["full_target"][known]) and np.array_equal(case["Y"], case["full_target"][free]),
            "C/Y does not refer to selected complete version")
    require(np.array_equal(case["model_vertices"][case["source_face_vertex_ids"]], case["full_target"]),
            "Source canonical face identity differs")
    require(str(case["target_version_id"].item()) == row["target_version_id"], "Version identity differs")
    return case


def allocation(entry, original):
    """Preserve original group-MSE coefficients with actual arm scalar counts."""
    groups = [r["replay_group"] for r in entry["samples"]]
    require(groups == original["micro_groups"], "Original group sequence differs")
    counts = [9*(int(r["N"])-int(r["K"])) for r in entry["samples"]]
    require(len(counts) == 8 and min(counts) > 0, "Eight nonempty Native samples required")
    old_counts, old_gains = original["micro_free_scalar_counts"], original["micro_gains"]
    old_d = int(original["D_W"])
    require(sum(old_counts) == old_d, "Historical coefficient denominator differs")
    reference = {g: math.fsum(c*q for c, q, x in zip(old_counts, old_gains, groups) if x == g)/old_d
                 for g in GROUPS}
    actual = {g: sum(c for c, x in zip(counts, groups) if x == g) for g in GROUPS}
    ref_counts = original["reference_micro_free_scalar_counts"]
    ref_mass = {g: sum(c for c, x in zip(ref_counts, groups) if x == g) for g in GROUPS}
    denominator = sum(counts)
    gains = [reference[g]*denominator/actual[g] for g in groups]
    effective = {g: math.fsum(c*q for c, q, x in zip(counts, gains, groups) if x == g)/denominator for g in GROUPS}
    require(abs(math.fsum(reference.values())-1.) < 1e-12 and
            all(abs(reference[g]-effective[g]) < 1e-12 for g in GROUPS), "Group-MSE weight changed")
    return dict(stage_step=entry["stage_step"], branch=original["branch"],
                D_W=denominator, D_B=int(original["D_B"]),
                micro_groups=groups, micro_free_scalar_counts=counts, micro_gains=gains,
                actual_group_scalar_counts=actual, ref_group_scalar_counts=ref_mass,
                reference_group_mse_weights=reference, effective_group_mse_weights=effective,
                group_mse_weights_sum=math.fsum(effective.values()),
                reference_micro_free_scalar_counts=list(ref_counts),
                original_micro_free_scalar_counts=list(old_counts), original_micro_gains=list(old_gains),
                original_D_W=old_d)


def make():
    pool, old, original_coeff = C.read(C.MANIFEST), C.read(SOURCE_SCHEDULE), C.read(SOURCE_COEFFICIENTS)
    require(len(pool["parents"]) == 30 and pool["total_versions"] == 39, "Frozen 30-parent / 39-version pool required")
    require(len(old["steps"]) == len(original_coeff["steps"]) == 3000, "Expected completed EXPOSURE stream")
    source_hashes = {str(p): C.sha(p) for p in (C.MANIFEST, SOURCE_SCHEDULE, SOURCE_COEFFICIENTS)}
    train = []
    for p in pool["parents"]:
        for v in p["variants"]:
            row = {k: copy.deepcopy(x) for k, x in p.items() if k != "variants"}
            row.update(copy.deepcopy(v), variant=v["kind"], physical_condition_id=p["condition_id"],
                       case_id=p["condition_id"]+"::"+v["variant_id"], part_type="SEAT_SURFACE")
            train.append(row)
    manifest = dict(schema="BENCH_MULTILOD_TRAINING_CASES_V1", parents=copy.deepcopy(pool["parents"]), train=train,
                    physical_parents=30, physical_conditions=30, case_versions=39,
                    source_manifest=str(C.MANIFEST), source_manifest_sha256=C.sha(C.MANIFEST),
                    sampling="Frozen parent schedule first; versions never enter the parent sampler as extra rows",
                    source_owner_metadata="Supervision/provenance only; Native all-free runtime", source_hashes=source_hashes)
    template = copy.deepcopy(old)
    template["steps"] = template["steps"][:C.UPDATES]
    template["updates"] = C.UPDATES
    same_seed_slots = 0
    old_seed_overlap = collections.Counter()
    old_seed_sets = {f: {r[f] for e in old["steps"] for r in e["samples"]} for f in RANDOM_FIELDS}
    for entry in template["steps"]:
        local = entry["stage_step"]
        entry.update(stage="NATIVE", local_step=local, curriculum_step=7500+local,
                     global_step=C.BASE_STEP+local, native_total_step=C.NATIVE_BASE+local, seam_total_step=0)
        for micro, row in enumerate(entry["samples"]):
            for field in RANDOM_FIELDS:
                key = f"{C.EXPERIMENT}|{entry['stage_step']}|{micro}|{field}"
                new = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
                same_seed_slots += int(new == row[field])
                old_seed_overlap[field] += int(new in old_seed_sets[field])
                row[field] = new
    require(same_seed_slots == 0 and not any(old_seed_overlap.values()), "Registered new seeds collide with historical same-field seed identities")
    tagged = tag_exposure_pairs(template)
    schedules, coefficients = {}, {}
    for arm in C.ARMS:
        available = copy.deepcopy(pool)
        if arm == "PLAIN1500":
            for parent in available["parents"]:
                parent["variants"] = [v for v in parent["variants"] if v["kind"] == "NATURAL"]
        selected, proof = adapt_schedule(tagged, available)
        for entry in selected["steps"]:
            for row in entry["samples"]:
                row["variant"] = row["variant_kind"]
        selected.update(schema=SCHEMA, arm=arm, experiment=C.EXPERIMENT, updates=C.UPDATES,
                        source_schedule_file_sha256=C.sha(SOURCE_SCHEDULE), source_hashes=source_hashes,
                        seed_policy="uint32 little-endian SHA256(experiment|stage_step|micro|field)[:4]",
                        same_field_seed_overlap_with_old= dict(old_seed_overlap),
                        trainer_integration_status="VERSION_AWARE_CASE_CACHE_AND_REGISTERED_COEFFICIENTS_REQUIRED",
                        coefficient_status="RECOMPUTED_ACTUAL_COUNTS_FIXED_REFERENCE_GROUP_MSE")
        if arm == "PLAIN1500":
            selected["variant_selection_policy"] = "ALWAYS_NATURAL_WHOLE_CLEAN_WIDTH_PAIR_V1"
            for e in selected["steps"]:
                for row in e["samples"]:
                    row["requested_variant_kind"] = "NATURAL"
        allocations = [allocation(e, orig) for e, orig in zip(selected["steps"], original_coeff["steps"])]
        if arm == "PLAIN1500":
            require(all(a["micro_free_scalar_counts"] == a["original_micro_free_scalar_counts"] and
                        max(abs(x-y) for x,y in zip(a["micro_gains"], a["original_micro_gains"])) < 1e-12
                        for a in allocations), "PLAIN does not reproduce EXPOSURE M objective")
        schedules[arm] = selected
        coefficients[arm] = dict(schema="BENCH_MULTILOD_REFERENCE_GROUP_MSE_ACTUAL_COUNTS_V1", arm=arm,
            updates=C.UPDATES, loss_normalization=C.LOSS_NORMALIZATION, steps=allocations,
            source_coefficients_sha256=C.sha(SOURCE_COEFFICIENTS),
            formula="a_g=sum(original_count_i*original_gain_i)/original_D_W; gain_i=a_g*actual_D_W/actual_m_g; loss=sum_i(gain_i*SSE_i)/actual_D_W",
            semantics="Original narrow-reference group MSE coefficients retained. Within-group scalar weighting uses actual N; equal parent exposure does not assert equal parent gradient weight.")
    validate_pair(schedules, manifest)
    return manifest, schedules, coefficients


def validate_pair(schedules, manifest):
    plain, multi = (schedules[a] for a in C.ARMS)
    by_case = {r["case_id"]: r for r in manifest["train"]}
    fields = ("parent_id", "condition_id", "replay_group", "augmentation", "alpha", "scale_xyz", "logical_instance_id", *RANDOM_FIELDS)
    for ep, em in zip(plain["steps"], multi["steps"]):
        require(ep["stage_step"] == em["stage_step"], "Paired step ordering changed")
        for rp, rm in zip(ep["samples"], em["samples"]):
            require(all(rp[k] == rm[k] for k in fields), "Paired scientific stream differs")
            require(rp["variant_kind"] == "NATURAL", "PLAIN selected a LOW version")
            for row in (rp, rm):
                target = by_case[row["case_id"]]
                require(all(row[k] == target[k] for k in ("N", "K", "case_path", "case_sha256", "target_version_id")), "Case cache identity does not match selected row")
    require(plain["parent_exposure"] == multi["parent_exposure"] and plain["group_exposure"] == multi["group_exposure"], "Arm exposure differs")
    return dict(status="PASS", paired_steps=C.UPDATES, paired_instances=C.UPDATES*8,
                group_exposure=plain["group_exposure"], parent_exposure=plain["parent_exposure"])


def validate_schedule(schedule, manifest, arm):
    expected_manifest, schedules, _ = make()
    require(manifest == expected_manifest and schedule == schedules[arm], "Frozen schedule/manifest differs")
    return dict(status="PASS", arm=arm, updates=C.UPDATES, instances=C.UPDATES*8)


def checked_coefficients(path, schedule, manifest):
    expected_manifest, schedules, coefficients = make()
    arm = schedule["arm"]
    actual = C.read(path)
    require(manifest == expected_manifest and schedule == schedules[arm] and actual == coefficients[arm],
            "Frozen actual-count coefficient identity differs")
    return actual["steps"]


def prepare():
    manifest, schedules, coefficients = make()
    actual_cases = []
    for row in manifest["train"]:
        a = load_case(row)
        actual_cases.append(dict(case_id=row["case_id"], case_sha256=row["case_sha256"], N=int(a["N"]), K=int(a["K"])))
    C.write_new(MANIFEST_PATH, manifest)
    for arm in C.ARMS:
        C.write_new(schedule_path(arm), schedules[arm])
        C.write_new(coefficients_path(arm), coefficients[arm])
    paired = validate_pair(schedules, manifest)
    kinds = {arm: dict(collections.Counter(r["variant_kind"] for e in schedules[arm]["steps"] for r in e["samples"])) for arm in C.ARMS}
    result = dict(status="PASS", schema="BENCH_MULTILOD_TRAINING_STREAM_GATE_V1", paired=paired,
                  actual_cases=actual_cases, case_versions=len(actual_cases), model_forwards=0, optimizer_updates=0, actual_CPU_OT_calls=0,
                  variant_exposure=kinds, loss_normalization=C.LOSS_NORMALIZATION,
                  source_hashes=manifest["source_hashes"],
                  manifest_sha256=C.sha(MANIFEST_PATH),
                  schedule_sha256={a:C.sha(schedule_path(a)) for a in C.ARMS},
                  coefficients_sha256={a:C.sha(coefficients_path(a)) for a in C.ARMS},
                  seed_fields_paired=True, historical_same_field_seed_reuse=0,
                  group_MSE_coefficients_equal_between_arms=True, PLAIN_original_gain_max_error=max(
                      abs(x-y) for a in coefficients["PLAIN1500"]["steps"] for x,y in zip(a["micro_gains"], a["original_micro_gains"])))
    C.write_new(C.OUT/"data/stream_gate.json", result)
    return result
