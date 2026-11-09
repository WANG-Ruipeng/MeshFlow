"""CPU-only parent-first selection of already-admitted fixed-C mesh versions.

This is a schedule adapter, NOT an integrated or authorized training entrypoint.
The caller supplies a frozen parent schedule and an accepted parent/variants
manifest.  Each logical occurrence must contain one CLEAN/X_WIDTH pair and have
an explicit logical_instance_id.  Both rows select the same LOD.  Parent/group
order, exposure count and scientific random seeds remain unchanged.

Trainer integration still needs a version-keyed case cache, fresh N-dependent
FM coefficient accounting and a separate registered training branch. Never feed
the returned rows into the historical single-condition cache or reuse old
coefficient_schedule.json after changing N. See integration_notes() below.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json

SCHEMA = "BENCH_FIXED_C_PARENT_FIRST_LOD_SCHEDULE_V1"
POLICY = "ODD_NATURAL_EVEN_LOW_WHOLE_CLEAN_WIDTH_PAIR_V1"
PAIR_KEY = "logical_instance_id"
RANDOM_FIELDS = ("noise_seed", "epsilon_seed", "time_seed", "permutation_seed")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def integration_notes():
    return dict(
        status="CPU_SELECTION_ADAPTER_ONLY_NOT_TRAINER_INTEGRATED",
        cache="Load accepted cases by case_id (physical condition plus variant), not condition_id alone.",
        sampler="Retain the old parent/group schedule; choose one variant per explicit logical CLEAN/X_WIDTH pair. Do not flatten variants into a parent sampler.",
        normalization="Selected N changes 9*(N-K), whole-batch denominator and M/HIGH gains. Register and recompute the chosen count-based coefficient policy; do not reuse historical coefficients or silently switch to instance/parent-MSE weighting.",
        equality="Parent exposure preserved means sampling count. Existing OLD/FOCUS/HIGH quotas are retained; this is not uniform sampling over all parents and does not certify equal loss/gradient weight.",
        randomness="Keep registered seed identities, alpha, time and parent order. Different N requires the correct-size Gaussian/epsilon and new OT; equal seed does not certify identical complete noise arrays across different shapes.",
        recovery="Create a separately versioned training branch with explicit source checkpoint, matching optimizer/RNG policy, immutable manifest/schedule hashes and budget. Do not bypass old resume checks.",
        validation="Before training, CPU-check case identity, same C, actual N/K, free-mask scalar count, OT/corner transport and parent/variant exposure. Model/resource checks follow only within a separately authorized training budget.",
        evaluation="Freeze inference C/N/noise independently. LOW training versions do not authorize changing evaluation requests or scoring source-geometry reconstruction as the open-completion objective.",
    )


def tag_exposure_pairs(schedule):
    """Explicitly adapt the known eight-row EXPOSURE pair schema.

    No generic heuristic is used. Unknown schemas must provide pair IDs directly.
    """
    accepted = {"BENCH_FOUR_ROUTES_M_PLUS3000_STREAM_V1", "BENCH_HIGH_JOINT_COVERAGE_STREAM_V1"}
    _require(schedule.get("schema") in accepted, "Unknown schedule schema: provide explicit logical_instance_id")
    result = copy.deepcopy(schedule)
    seen = set()
    for entry in result["steps"]:
        step = entry.get("stage_step")
        _require(isinstance(step, int) and step not in seen, "Unique integer stage_step required")
        seen.add(step)
        rows = entry["samples"]
        _require(len(rows) == 8, "Known EXPOSURE schema requires eight rows")
        _require(len({r["parent_id"] for r in rows}) == 4, "Known EXPOSURE schema requires four distinct parents")
        for start in (0, 2, 4, 6):
            pair = rows[start:start+2]
            _require([r.get("sample_index") for r in pair] == [start, start+1], "Explicit sample_index pair contract differs")
            _require([r.get("augmentation") for r in pair] == ["CLEAN", "X_WIDTH"], "CLEAN/X_WIDTH ordering differs")
            for name in ("parent_id", "condition_id", "replay_group"):
                _require(pair[0][name] == pair[1][name], "Pair identities differ: " + name)
            for r in pair:
                _require(PAIR_KEY not in r, "Existing logical pair IDs must not be overwritten")
                r[PAIR_KEY] = f"stage_step={step}/parent_position={start//2}"
    result["explicit_pair_identity_contract"] = "EXPOSURE_EIGHT_ROWS_FOUR_DISTINCT_PARENTS_ADJACENT_CLEAN_WIDTH_V1"
    return result


def _parents(manifest):
    _require(isinstance(manifest, dict) and isinstance(manifest.get("parents"), list), "Expected manifest.parents list")
    out = {}
    for parent in manifest["parents"]:
        pid, cid, group = (parent[k] for k in ("parent_id", "condition_id", "replay_group"))
        _require(pid not in out, "Duplicate parent in parent-first manifest")
        by_kind = {}
        for v in parent["variants"]:
            for name in ("variant_id", "kind", "case_path", "case_sha256", "target_version_id", "N", "K"):
                _require(name in v, "Missing variant field: " + name)
            kind = v["kind"]
            _require(kind in ("NATURAL", "LOW") and kind not in by_kind, "Exactly one NATURAL and at most one LOW per parent")
            _require(isinstance(v["N"], int) and isinstance(v["K"], int) and 128 <= v["N"] <= 799 and 2 <= v["K"] < v["N"], "Variant N/K invalid")
            digest = v["case_sha256"]
            _require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdefABCDEF" for c in digest), "Variant must reference an immutable case SHA256")
            _require(bool(v["variant_id"]) and bool(v["target_version_id"]), "Version identity required")
            by_kind[kind] = v
        _require("NATURAL" in by_kind, "Original NATURAL must remain available for every parent")
        if "LOW" in by_kind:
            natural, low = by_kind["NATURAL"], by_kind["LOW"]
            _require(low["N"] < natural["N"] and low["K"] == natural["K"], "LOW must reduce N and preserve K")
            _require(low["variant_id"] != natural["variant_id"] and low["target_version_id"] != natural["target_version_id"], "Distinct LOD geometry/version identities required")
        out[pid] = dict(condition_id=cid, replay_group=group, variants=by_kind)
    return out


def adapt_schedule(parent_schedule, manifest):
    """Return (new_schedule, selection_ledger), without reading cases or RNG draws.

    The physical condition_id remains stable. case_id is the version-aware key
    which a future trainer MUST use to load a case. Old index/condition_index
    remain parent-schedule provenance and are explicitly not case-cache indices.
    """
    parents = _parents(manifest)
    result = copy.deepcopy(parent_schedule)
    _require(result.get("schema") != SCHEMA, "Do not adapt a version-selected schedule twice")
    source_hash, manifest_hash = _hash(parent_schedule), _hash(manifest)
    seen_pairs, occurrence = set(), collections.Counter()
    parent_exposure, group_exposure, variant_exposure = collections.Counter(), collections.Counter(), collections.Counter()
    ledger = []
    for entry in result["steps"]:
        grouped = collections.OrderedDict()
        for index, row in enumerate(entry["samples"]):
            _require(PAIR_KEY in row and isinstance(row[PAIR_KEY], str) and row[PAIR_KEY], "Explicit logical_instance_id required; do not infer pairs from row parity")
            grouped.setdefault(row[PAIR_KEY], []).append((index, row))
        for pair_id, members in grouped.items():
            _require(pair_id not in seen_pairs, "Logical instance ID repeated across steps")
            seen_pairs.add(pair_id)
            pair = [r for _, r in members]
            _require(len(pair) == 2 and [r.get("augmentation") for r in pair] == ["CLEAN", "X_WIDTH"], "Logical instance must contain one ordered CLEAN/X_WIDTH pair")
            pid = pair[0]["parent_id"]
            _require(pid in parents, "Scheduled parent absent from accepted manifest")
            p = parents[pid]
            for r in pair:
                for key, expected in (("parent_id", pid), ("condition_id", p["condition_id"]), ("replay_group", p["replay_group"])):
                    _require(r.get(key) == expected, "Parent/condition/group selection changed: " + key)
                _require(all(key in r for key in RANDOM_FIELDS), "Original scientific RNG identities missing")
                _require("case_id" not in r and "variant_id" not in r, "Rows already have selected geometry")
            occurrence[pid] += 1
            requested = "LOW" if occurrence[pid] % 2 == 0 else "NATURAL"
            selected = requested if requested in p["variants"] else "NATURAL"
            v = p["variants"][selected]
            case_id = p["condition_id"] + "::" + v["variant_id"]
            for _, r in members:
                r.update(physical_condition_id=p["condition_id"], case_id=case_id,
                         variant_id=v["variant_id"], variant_kind=selected,
                         case_path=v["case_path"], case_sha256=v["case_sha256"],
                         target_version_id=v["target_version_id"], N=v["N"], K=v["K"],
                         parent_occurrence=int(occurrence[pid]), requested_variant_kind=requested)
                parent_exposure[pid] += 1; group_exposure[r["replay_group"]] += 1
                variant_exposure[case_id] += 1
            ledger.append(dict(logical_instance_id=pair_id, stage_step=entry.get("stage_step", entry.get("step")),
                parent_id=pid, physical_condition_id=p["condition_id"], replay_group=p["replay_group"],
                parent_occurrence=int(occurrence[pid]), requested_kind=requested, selected_kind=selected,
                natural_only_fallback=bool(requested != selected), case_id=case_id,
                variant_id=v["variant_id"], N=v["N"], K=v["K"], instances=2))
    original_rows = [r for e in parent_schedule["steps"] for r in e["samples"]]
    selected_rows = [r for e in result["steps"] for r in e["samples"]]
    _require(len(original_rows) == len(selected_rows), "Adapter changed total exposure")
    for before, after in zip(original_rows, selected_rows):
        for key in ("parent_id", "condition_id", "replay_group", "augmentation", "alpha", "scale_xyz", *RANDOM_FIELDS):
            _require(before.get(key) == after.get(key), "Scientific stream changed: " + key)
    result.update(schema=SCHEMA, source_schema=parent_schedule.get("schema"),
        source_parent_schedule_sha256=source_hash, variants_manifest_sha256=manifest_hash,
        variant_selection_policy=POLICY, variant_selection_random_draws=0,
        parent_exposure=dict(sorted(parent_exposure.items())), group_exposure=dict(group_exposure),
        variant_exposure=dict(sorted(variant_exposure.items())),
        coefficient_status="INVALIDATED_BY_POTENTIALLY_CHANGED_N_RECOMPUTE_BEFORE_TRAINING",
        trainer_integration_status="NOT_IMPLEMENTED", training_authorized=False)
    proof = dict(status="PASS_PARENT_SELECTION_ONLY", schema=SCHEMA, policy=POLICY,
        source_parent_schedule_sha256=source_hash, variants_manifest_sha256=manifest_hash,
        updates=len(result["steps"]), instances=len(selected_rows), logical_instances=len(ledger),
        parent_exposure=dict(sorted(parent_exposure.items())), group_exposure=dict(group_exposure),
        variant_exposure=dict(sorted(variant_exposure.items())), selections=ledger,
        scientific_seed_fields_unchanged=True, geometry_case_resolution_required=True,
        original_coefficients_reusable=False, extra_parent_exposure=0, random_draws=0,
        model_calls=0, OT_calls=0, integration=integration_notes())
    return result, proof
