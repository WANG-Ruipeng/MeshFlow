#!/usr/bin/env python3
"""Prepare authorized curriculum waves; never authorize or dispatch GPU work."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "tools"))
from meshflow_control.evaluation.io import digest, read_json, write_json
from evaluate_recipe_factorial import root, phase_registration, register_phase, bind_checkpoint, completed_rows

VERSION = "recipe_curriculum_preparation_v1"
COURSES = ("C20_40_MIX", "C40_20_MIX")
BASELINE = "R2_MIX_H_ALL"
OFFICIAL = Path("/mnt/e/MeshFlow/native_t1/runs/control_mvp48_v1/checkpoints/official_chair_last.pt")
OFFICIAL_SHA = "bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d"
INITIAL_SHA = "792ad8d6249ed4ccd1b5c5a81dfba0e527fc61911be96144729f8a2fc7f4811a"

def checked(path, expected):
    if digest(path) != expected:
        raise ValueError("Frozen identity changed: " + str(path))
    return expected

def authorization_contract(auth, out, namespace):
    expected = dict(schema="chair_condition_recipe_factorial_training_authorization_v1",
        experiment="CHAIR_CONDITION_RECIPE_FACTORIAL_V1", namespace=namespace,
        out=str(Path(out).resolve()), updates_per_recipe=6000, effective_batch=8, microbatch=1)
    for key, value in expected.items():
        if auth.get(key) != value:
            raise ValueError("Curriculum authorization differs: " + key)
    if len(auth.get("authorized_recipes", [])) != 2 or set(auth["authorized_recipes"]) != set(COURSES):
        raise ValueError("Exactly the two curriculum recipes must be explicitly authorized")
    if auth.get("hybrid_schedules") != {r: "all" for r in COURSES}:
        raise ValueError("This comparison authorizes both curricula with H_ALL only")
    return auth

def wave(out, authorization, kind, source_hashes):
    """Pure job construction. R2 never enters a training or sampling job."""
    out = Path(out); jobs = []
    for label in COURSES:
        folder = out / ("training" if kind == "training" else "phases/course/sampling") / label
        job = dict(name=label, phase=kind, status_root=str(folder), progress_path=str(folder / "progress.json"),
            completed_path=str(folder / "completed.json"), source_hashes=source_hashes)
        if kind == "training":
            job.update(latest_checkpoint=str(folder / "latest.pt"), latest_receipt=str(folder / "latest.json"),
                required_completion_files=[str(out / "inference" / (label + "_generator_step6000.pt"))],
                command=[sys.executable, "-B", str(REPO / "tools/train_recipe_factorial.py"),
                    "--recipe", label, "--hybrid-schedule", "all", "--official", str(OFFICIAL),
                    "--data-manifest", str(out.parents[1] / "data/train_manifest.json"),
                    "--stream-root", str(out / "streams"), "--out", str(out), "--registration", str(authorization)])
        elif kind == "sampling":
            job.update(attempt_option="--attempt", command=[sys.executable, "-B", str(REPO / "tools/evaluate_recipe_factorial.py"),
                "--stage", "sample", "--out", str(out), "--phase", "course", "--label", label, "--controller-lock-owned"])
        else:
            raise ValueError("Unknown wave kind")
        jobs.append(job)
    return dict(schema="recipe_factorial_dispatch_wave_v1", wave="curriculum_" + kind, max_workers=2, jobs=jobs)

def prerequisites(out, authorization):
    out = root(out); authorization = Path(authorization).resolve()
    plan_path = out / "streams/recipe_plan.json"; plan = read_json(plan_path)
    auth = authorization_contract(read_json(authorization), out, plan["namespace"])
    frozen = read_json(out / "registrations/source_hashes.json")
    if len(frozen) != 97:
        raise ValueError("Expected the original 97-file freeze")
    for name, sha in frozen.items():
        checked(REPO / name, sha)
        if auth.get("source_hashes", {}).get(name) != sha:
            raise ValueError("Authorization must retain all original source hashes: " + name)
    from meshflow_control.training.recipe_factorial import check_authorization
    for label in COURSES:
        check_authorization(authorization, label, "all", plan["namespace"], out)
    checked(OFFICIAL, OFFICIAL_SHA)
    pre = read_json(out / "streams/precompute_receipt.json")
    if pre.get("status") != "PASS" or pre.get("slots") != 48000 or pre.get("plan_sha256") != digest(plan_path):
        raise ValueError("Existing CPU paired-input precompute is incomplete; no automatic rerun")
    pairs = read_json(out / "registrations/paired_hashes.json")
    files = [authorization, plan_path, out / "streams/precompute_receipt.json", out / "registrations/source_hashes.json",
        out / "registrations/paired_hashes.json", out / "evaluation/requests.json", out / "phases/main/gallery/cameras.json",
        out / "PROTOCOL_CLARIFICATIONS.md", out / "audit/condition_construction_audit.json", Path(__file__).resolve()]
    for label in COURSES:
        relative = "streams/paired_inputs/" + label + "__H_ALL.json"
        checked(out / relative, pairs[relative]); files.append(out / relative)
    from score_recipe_factorial_conflict import load_registration, load_queries, query_registration_path
    query = load_registration(out, out / "evaluation/requests.json")
    if len(query["rows"]) != 16:
        raise ValueError("Expected exactly 16 frozen condition query banks")
    for row in query["rows"]:
        load_queries(row); files.append(Path(row["query_path"]))
    files.append(query_registration_path(out, out / "evaluation/requests.json"))
    main = read_json(out / "phases/main/registration.json")
    if BASELINE not in main["labels"] or main["registered_per_model"] != 64:
        raise ValueError("Existing R2 baseline registration is unavailable")
    return out, auth, {str(p): digest(p) for p in files}

def prepare_training(out, authorization):
    out, auth, identities = prerequisites(out, authorization)
    registration = dict(version=VERSION, authorization_path=str(Path(authorization).resolve()),
        authorization_sha256=digest(authorization), source_and_input_hashes=identities,
        courses=list(COURSES), baseline=BASELINE, updates_per_course=6000, additional_outputs=128,
        comparison_outputs=192, baseline_new_rollouts=0, baseline_new_scores=0,
        condition_pool_caveat="Historical condition-pool recipe comparison; not pure ratio causality",
        curriculum_caveat="Curricula reorder the matched R2 input multiset and change effective-batch grouping")
    write_json(out / "registrations/curriculum_preparation.json", registration, freeze=True)
    path = out / "registrations/curriculum_training_jobs.json"
    write_json(path, wave(out, Path(authorization).resolve(), "training", identities), freeze=True)
    return dict(status="PREPARED_NOT_DISPATCHED", wave=str(path), source_hashes=identities,
        command=[sys.executable, "-B", str(REPO / "tools/dispatch_recipe_factorial.py"), "--out", str(out), "--jobs", str(path)])

def verify_preparation(out, authorization):
    out, _, identities = prerequisites(out, authorization)
    saved = read_json(out / "registrations/curriculum_preparation.json")
    if saved["source_and_input_hashes"] != identities or saved["authorization_sha256"] != digest(authorization):
        raise ValueError("Curriculum preparation changed on resume")
    return out, saved

def completion_contract(value, label, expected_identities):
    if value.get("status") != "COMPLETE" or value.get("step") != 6000 or value.get("recipe") != label or value.get("hybrid_schedule") != "all":
        raise ValueError("Curriculum training is not complete at the authorized endpoint")
    if value.get("lineage", {}).get("generator_initial_sha256") != INITIAL_SHA or value["lineage"].get("fresh_AdamW") is not True:
        raise ValueError("Curriculum initialization/fresh optimizer differs")
    for key, sha in expected_identities.items():
        if value.get("identities", {}).get(key) != sha:
            raise ValueError("Curriculum training identity differs: " + key)
    if value.get("physical_optimizer_updates_this_attempt") != value.get("successful_calls", {}).get("optimizer_update"):
        raise ValueError("Physical update ledger disagrees")
    if value.get("stream_cursor", 6000) != 6000:
        raise ValueError("Final training stream cursor differs")

def verify_training(out, authorization):
    out, saved = verify_preparation(out, authorization); results = []
    from meshflow_control.training.recipe_factorial import protocol, source_identity
    from meshflow_control.data.io import json_hash
    namespace = read_json(out / "streams/recipe_plan.json")["namespace"]
    for label in COURSES:
        folder = out / "training" / label; value = read_json(folder / "completed.json")
        expected = dict(authorization_sha256=digest(authorization), plan_sha256=digest(out / "streams/recipe_plan.json"),
            paired_inputs_sha256=digest(out / "streams/paired_inputs" / (label + "__H_ALL.json")),
            data_manifest_sha256=digest(out.parents[1] / "data/train_manifest.json"))
        completion_contract(value, label, expected)
        if value["identities"]["training_sources"] != source_identity():
            raise ValueError("Training implementation sources changed")
        expected_protocol = json_hash(protocol(label, "all", namespace))
        for key, path in (("checkpoint", out / "checkpoints" / (label + "_step6000.pt")),
                          ("inference_export", out / "inference" / (label + "_generator_step6000.pt"))):
            record = value[key]
            if Path(record["path"]).resolve() != path.resolve() or record.get("global_step") != 6000 or record.get("recipe") != label or record.get("protocol_sha256") != expected_protocol:
                raise ValueError("Endpoint export/checkpoint contract differs")
            checked(path, record["sha256"])
        attempts = []
        for p in sorted((folder / "attempts").glob("*/final_status.json")):
            row = read_json(p)
            attempts.append(dict(path=str(p), sha256=digest(p), status=row.get("status"), attempt_id=row.get("attempt_id"),
                physical_optimizer_updates=row.get("physical_optimizer_updates_this_attempt", 0), successful_calls=row.get("successful_calls", {})))
        if not attempts or sum(a["status"] == "COMPLETE" for a in attempts) != 1:
            raise ValueError("Complete attempt ledger required")
        results.append(dict(label=label, completed_sha256=digest(folder / "completed.json"), attempts=attempts,
            physical_updates_all_attempts=sum(a["physical_optimizer_updates"] for a in attempts), unique_updates=6000))
    result = dict(status="COMPLETE", rows=results, unique_updates=12000, authorization_sha256=saved["authorization_sha256"])
    write_json(out / "audit/curriculum_training_verification.json", result, freeze=True)
    return result

def prepare_sampling(out, authorization):
    verify_training(out, authorization); out, saved = verify_preparation(out, authorization)
    models = [dict(label=r, loader="recipe_factorial", checkpoint_path=str(out / "inference" / (r + "_generator_step6000.pt"))) for r in COURSES]
    register_phase(out, "course", models, out / "evaluation/requests.json", "course")
    for label in COURSES:
        bind_checkpoint(out, "course", label)
    path = out / "registrations/curriculum_sampling_jobs.json"
    write_json(path, wave(out, authorization, "sampling", saved["source_and_input_hashes"]), freeze=True)
    return dict(status="PREPARED_NOT_DISPATCHED", wave=str(path), registered_outputs=128,
        command=[sys.executable, "-B", str(REPO / "tools/dispatch_recipe_factorial.py"), "--out", str(out), "--jobs", str(path)])

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("prepare-training", "verify-training", "prepare-sampling", "collect", "score"))
    p.add_argument("--out", required=True); p.add_argument("--authorization", required=True); p.add_argument("--workers", type=int, default=2)
    a = p.parse_args(argv)
    if a.stage == "prepare-training": result = prepare_training(a.out, a.authorization)
    elif a.stage == "verify-training": result = verify_training(a.out, a.authorization)
    elif a.stage == "prepare-sampling": result = prepare_sampling(a.out, a.authorization)
    else:
        out, _ = verify_preparation(a.out, a.authorization)
        if a.stage == "collect": result = dict(status="COMPLETE", outputs=len(completed_rows(out, "course")))
        else:
            from score_recipe_factorial_conflict import score
            receipt = out / "phases/course/evaluation/scoring_receipt.json"
            if receipt.exists() and read_json(receipt).get("status") == "COMPLETE":
                completed_rows(out, "course"); result = dict(read_json(receipt), reused_completed_scores=True)
            else: result = score(out, "course", a.workers)
    print(json.dumps(result), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
