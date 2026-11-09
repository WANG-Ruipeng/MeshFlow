"""Reproducible CPU audit of LOD selection on the completed EXPOSURE stream.

This only replays metadata selection: no training, array augmentation, noise,
OT, model forward, or scoring. The selected 3000-step schedule stays in RAM.
Only its canonical hash and compact aggregate evidence are written to disk.
"""
from __future__ import annotations

import collections
import hashlib
import json

from . import common as C
from .sampler import adapt_schedule, tag_exposure_pairs, RANDOM_FIELDS

SOURCE = C.ROOT / "experiments/r2_bench_four_routes_v1/exposure/data/schedule_M_PLUS3000.json"
MANIFEST = C.OUT / "data/manifest.json"
DESTINATION = C.OUT / "data/sampler_contract_audit.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def audit():
    source_hash, manifest_hash = C.sha(SOURCE), C.sha(MANIFEST)
    source, manifest = C.read(SOURCE), C.read(MANIFEST)
    require(source["updates"] == len(source["steps"]) == 3000, "Expected the completed 3000-step EXPOSURE stream")
    require(len(manifest["parents"]) == 30 and manifest["total_versions"] == 39,
            "Expected the frozen 30-parent / 39-version data pool")

    # File verification is independent of metadata selection; originals remain read-only.
    verified, case_lookup = [], {}
    for parent in manifest["parents"]:
        natural = next(v for v in parent["variants"] if v["kind"] == "NATURAL")
        natural_case = C.arrays(natural["case_path"])
        for variant in parent["variants"]:
            path = C.resolve(variant["case_path"])
            actual_sha = C.sha(path)
            require(actual_sha == variant["case_sha256"], "Case hash differs: " + str(path))
            case = natural_case if variant["kind"] == "NATURAL" else C.arrays(path)
            n, k = int(case["N"]), int(case["K"])
            require((n, k) == (variant["N"], variant["K"]), "Case N/K differs: " + str(path))
            require(len(case["full_target"]) == n and len(case["C"]) == k and len(case["Y"]) == n-k,
                    "Array lengths differ from N/K: " + str(path))
            require(str(case["target_version_id"].item()) == variant["target_version_id"],
                    "Actual target version differs: " + str(path))
            require(case["C"].dtype == natural_case["C"].dtype and case["C"].shape == natural_case["C"].shape
                    and case["C"].tobytes() == natural_case["C"].tobytes(),
                    "Physical C bytes changed within parent: " + parent["parent_id"])
            case_id = parent["condition_id"] + "::" + variant["variant_id"]
            require(case_id not in case_lookup, "Duplicate versioned case identity")
            case_lookup[case_id] = variant
            verified.append(dict(parent_id=parent["parent_id"], condition_id=parent["condition_id"],
                                 replay_group=parent["replay_group"], case_id=case_id,
                                 variant_kind=variant["kind"], case_path=variant["case_path"],
                                 case_sha256=actual_sha, N=n, K=k,
                                 target_version_id=variant["target_version_id"],
                                 free_scalar_count=9*(n-k), physical_C_bytes_unchanged=True))
    require(len(verified) == 39, "Not all 39 actual cases verified")
    tagged = tag_exposure_pairs(source)
    selected, proof = adapt_schedule(tagged, manifest)
    source_rows = [row for step in source["steps"] for row in step["samples"]]
    selected_rows = [row for step in selected["steps"] for row in step["samples"]]
    require(len(source_rows) == len(selected_rows) == 24000, "Source exposure count changed")
    source_parents = collections.Counter(r["parent_id"] for r in source_rows)
    source_groups = collections.Counter(r["replay_group"] for r in source_rows)
    require(source_parents == collections.Counter(r["parent_id"] for r in selected_rows), "Parent exposure changed")
    require(source_groups == collections.Counter(r["replay_group"] for r in selected_rows), "Group exposure changed")
    require(set(source_parents) == {p["parent_id"] for p in manifest["parents"]}, "Source/manifest parent domains differ")
    kinds = collections.Counter()
    group_kinds = collections.defaultdict(collections.Counter)
    parent_kinds = collections.defaultdict(collections.Counter)
    seeds_unchanged = {key: 0 for key in RANDOM_FIELDS}
    for old, new in zip(source_rows, selected_rows):
        # Every historical row field, not only the seed subset, must survive.
        require(all(new.get(key) == value for key, value in old.items()), "Original source row metadata changed")
        for key in RANDOM_FIELDS:
            require(old[key] == new[key], "Original scientific seed changed: " + key)
            seeds_unchanged[key] += 1
        v = case_lookup[new["case_id"]]
        require(all(new[key] == v[key] for key in ("variant_id", "case_path", "case_sha256", "target_version_id", "N", "K")),
                "Selected case fields do not resolve to verified variant")
        kinds[new["variant_kind"]] += 1
        group_kinds[new["replay_group"]][new["variant_kind"]] += 1
        parent_kinds[new["parent_id"]][new["variant_kind"]] += 1

    parent_summary = []
    for parent in manifest["parents"]:
        pid = parent["parent_id"]
        counts = parent_kinds[pid]
        parent_summary.append(dict(parent_id=pid, replay_group=parent["replay_group"],
                                   available_variants=[v["kind"] for v in parent["variants"]],
                                   original_instances=source_parents[pid], selected_instances=sum(counts.values()),
                                   natural_instances=counts["NATURAL"], low_instances=counts["LOW"],
                                   low_instance_fraction=counts["LOW"]/source_parents[pid]))
    group_summary = [dict(replay_group=g, original_instances=source_groups[g],
                          natural_instances=group_kinds[g]["NATURAL"], low_instances=group_kinds[g]["LOW"],
                          low_instance_fraction=group_kinds[g]["LOW"]/source_groups[g])
                     for g in ("OLD", "FOCUS", "HIGH")]
    for row in verified:
        row["selected_instances"] = proof["variant_exposure"].get(row["case_id"], 0)
    require(C.sha(SOURCE) == source_hash and C.sha(MANIFEST) == manifest_hash, "Inputs changed during audit")
    result = dict(
        schema="BENCH_FIXED_C_MULTILOD_REAL_EXPOSURE_AUDIT_V1", status="PASS_METADATA_SELECTION_ONLY",
        source_schedule=str(SOURCE), source_schedule_file_sha256=source_hash,
        variants_manifest=str(MANIFEST), variants_manifest_file_sha256=manifest_hash,
        source_schedule_canonical_sha256=canonical_hash(source),
        tagged_schedule_canonical_sha256=canonical_hash(tagged),
        selected_schedule_canonical_sha256=canonical_hash(selected),
        code_sha256={name: C.sha(C.ROOT/"r2/data_tools/bench_multilod_v1"/name)
                     for name in ("audit_sampler.py", "sampler.py")},
        historical_updates_replayed_as_metadata=3000, original_instances=24000, selected_instances=24000,
        logical_clean_width_pairs=proof["logical_instances"], parents=30, verified_actual_cases=39,
        natural_versions=sum(v["variant_kind"] == "NATURAL" for v in verified),
        low_versions=sum(v["variant_kind"] == "LOW" for v in verified),
        selected_natural_instances=kinds["NATURAL"], selected_low_instances=kinds["LOW"],
        selected_low_instance_fraction=kinds["LOW"]/24000,
        variant_policy=proof["policy"], whole_clean_width_pairs_same_variant=True,
        parent_exposure_exactly_preserved=True, group_exposure_exactly_preserved=True,
        all_original_row_fields_preserved=True, scientific_seed_unchanged_rows=seeds_unchanged,
        parent_exposure=parent_summary, group_exposure=group_summary, variant_exposure=verified,
        schedule_materialized_to_disk=False, per_pair_ledger_materialized_to_disk=False,
        coefficient_status=selected["coefficient_status"], original_coefficients_reusable=False,
        trainer_integration_status="NOT_IMPLEMENTED", training_authorized=False,
        model_calls=0, GPU_calls=0, optimizer_updates=0, noise_arrays_drawn=0, OT_calls=0,
        integration=proof["integration"],
        scope="Existing completed stream replayed only to prove CPU version selection. This does not launch, authorize or schedule new 3000-step training.")
    C.write(DESTINATION, result)
    return result


def main():
    result = audit()
    print(json.dumps({k: result[k] for k in (
        "status", "historical_updates_replayed_as_metadata", "parents", "verified_actual_cases",
        "selected_instances", "selected_natural_instances", "selected_low_instances",
        "selected_low_instance_fraction", "group_exposure", "trainer_integration_status", "optimizer_updates")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
