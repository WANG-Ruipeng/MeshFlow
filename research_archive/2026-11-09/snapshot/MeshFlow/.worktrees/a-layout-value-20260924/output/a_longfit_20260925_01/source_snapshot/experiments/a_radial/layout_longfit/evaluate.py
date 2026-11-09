"""Fixed 240k final surface evaluation with immutable fitdiag20k references."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from ..exporter import RadialSupport
from ..legacy.networks import DomainVelocity
from ..io_utils import array_hash, file_hash, save_npz, write_json
from ..layout_student.deployment import deploy, named_quality, qualification
from ..layout_student.integrity import mesh_identity, STORED_HASH_SCHEMA
from ..layout_student.train import model_sha256
from ..layout_student.evaluate import metric_cache
from ..layout_value.geometry import audit_mesh

MODEL_NAMES = ('fixed', 'teacher_reference', 'parent_joint_10', 'parent_joint_11',
               'joint_10', 'joint_11', 'single_energy_10', 'single_prior_10')
POLICIES = {name: (("offline_teacher",) if name == "teacher_reference"
                  else ("raw",) if name == "fixed"
                  else ("raw", "quality_fallback")) for name in MODEL_NAMES}
SEED_BASES = (792501, 892501)
POINTS = 8192


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def source_single_jobs(protocol):
    nested=protocol['evaluation'].get('source_single_jobs')
    top=protocol.get('source_single_jobs')
    require(nested is not None or top is not None,'Frozen single-job metadata missing')
    if nested is not None and top is not None:
        require(nested==top,'Duplicated single-job metadata disagrees')
    return nested if nested is not None else top


def registered_uids(protocol, name):
    selected = protocol['evaluation']['model_uids'][name]
    require(len(selected) == len(set(selected)) and set(selected) <= set(protocol['uids']),
            'Unregistered or duplicate parent in final evaluation')
    if name.startswith('single_'):
        require(len(selected)==1 and selected[0] in protocol['train_uids']
                and selected==protocol['evaluation']['single_reference_uids'][name]
                and selected==source_single_jobs(protocol)[name]['active_training_uids'],
                'Frozen single reference may evaluate only its own training parent')
        require(name not in protocol['jobs'],'Frozen single reference must not become a new training job')
    else:
        require(set(selected)==set(protocol['uids']),'Joint/baseline evaluation omits parents')
    return selected


def expected_slots(protocol):
    return {(uid, model, policy) for model in MODEL_NAMES
            for uid in registered_uids(protocol, model) for policy in POLICIES[model]}


def aggregate(run):
    run = Path(run)
    protocol = read_json(run / "RUN_PROTOCOL.json")
    expected_attempts = {(uid, model) for model in MODEL_NAMES for uid in registered_uids(protocol, model)}
    attempts = [read_json(path) for path in sorted((run / "attempts").glob("*/*/result.json"))]
    ids = [(a["uid"], a["model"]) for a in attempts]
    require(len(ids) == len(set(ids)) and set(ids) <= expected_attempts, "Duplicate or unregistered attempt")
    rows = []
    errors = []
    for attempt in attempts:
        if attempt["status"] == "success":
            selected = attempt["rows"]
            require({r["policy"] for r in selected} == set(POLICIES[attempt["model"]]),
                    "Successful attempt lacks a predeclared policy")
            require(len(selected) == len(POLICIES[attempt["model"]]), "Duplicate policy in attempt")
            require(all(r["uid"] == attempt["uid"] and r["model"] == attempt["model"]
                        and r["status"] == "success" for r in selected), "Attempt row identity mismatch")
            rows.extend(selected)
        else:
            require(attempt.get("rows", []) == [], "Failed attempt contains successful result rows")
            errors.append(attempt)
    keys = [(r["uid"], r["model"], r["policy"]) for r in rows]
    require(len(keys) == len(set(keys)), "Duplicate result row")
    expected = expected_slots(protocol)
    failed_slots = [(a["uid"], a["model"], policy) for a in errors for policy in POLICIES[a["model"]]]
    value = {
        "rows": rows, "attempt_count": len(attempts), "errors": errors,
        "expected_attempt_count": len(expected_attempts), "expected_row_count": len(expected),
        "failed_row_slots": failed_slots,
        "unattempted": sorted(expected_attempts - set(ids)),
        "missing_success_rows": sorted(expected - set(keys)),
    }
    write_json(run / "results.json", value, replace=(run / "results.json").exists())
    return value


def load_network(run, name, protocol):
    run=Path(run).resolve()
    require(name in MODEL_NAMES,'Unregistered model')
    if name in ('fixed','teacher_reference'):
        return None,{'kind':name,'checkpoint':None,'deployment_weight':False}
    evaluation=protocol['evaluation'];path=(run/evaluation['model_checkpoints'][name]).resolve()
    require(path.is_relative_to(run),'Checkpoint path escapes run')
    seed=evaluation['model_seeds'][name];before=file_hash(path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    require(checkpoint['is_predeclared_final'] is True,'Only fixed final weights deployable')
    source_reference=name.startswith(('parent_joint_','single_'))
    cfg=checkpoint['config']
    if source_reference:
        require(before==evaluation['source_checkpoint_sha256'][name],'Frozen source checkpoint changed')
        require(checkpoint['update']==evaluation['source_model_updates']==20000,'Wrong source20k budget')
        require(checkpoint['protocol_sha256']==protocol['hashes']['source_protocol_sha256'],
                'Frozen reference source protocol differs')
        require(cfg['job_id']==evaluation['source_model_jobs'][name],'Wrong source job identity')
        expected_active=(source_single_jobs(protocol)[name]['active_training_uids']
                         if name.startswith('single_') else protocol['train_uids'])
        require(cfg['active_training_uids']==expected_active,'Frozen source active parents differ')
    else:
        require(name in protocol['jobs'] and cfg['job_id']==name,'Final checkpoint belongs to another job')
        require(checkpoint['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json'),'Training protocol differs')
        require(checkpoint['update']==evaluation['formal_final_update']==240000,'Only fixed 240000 final is evaluated')
        require(cfg['active_training_uids']==protocol['jobs'][name]['active_training_uids']==protocol['train_uids'],
                'Final training parents differ')
    require(cfg['seed']==seed and cfg['arm']=='c0','Checkpoint seed/arm mismatch')
    observed=model_sha256(checkpoint['model'])
    require(observed==checkpoint['model_sha256'],'Model tensor hash mismatch')
    if source_reference:
        require(observed==evaluation['source_model_sha256'][name],'Frozen source model changed')
    require(file_hash(path)==before,'Checkpoint changed while loading')
    model=DomainVelocity().cpu();model.load_state_dict(checkpoint['model'],strict=True)
    require(sum(p.numel() for p in model.parameters())==29907,'Architecture changed')
    model.eval().requires_grad_(False)
    return model,{'kind':name,'checkpoint':str(path),'file_sha256':before,'model_sha256':observed,
        'training_seed':seed,'updates':checkpoint['update'],'parameter_count':29907,
        'architecture':'unchanged legacy.networks.DomainVelocity',
        'weight_selection':'new joint fixed240000 final; source joint/single references fixed prior20000 final',
        'single_object_memorization_diagnostic':name.startswith('single_'),
        'frozen_source_reference':source_reference,'parent_reference':name.startswith('parent_joint_'),
        'single_reference_was_not_retrained_in_this_run':name.startswith('single_')}


def save_mesh_with_identity(path, mesh, template_faces):
    save_npz(path, **mesh)
    identity = mesh_identity(path, template_faces)
    identity["stored_array_hash_schema"] = STORED_HASH_SCHEMA
    with np.load(path, allow_pickle=False) as stored:
        for key in mesh:
            require(np.array_equal(mesh[key], stored[key]), "Saved NPZ differs from produced mesh")
    return identity


def evaluate_run(run, models=None, selected_uids=None):
    run = Path(run).resolve()
    protocol = read_json(run / "RUN_PROTOCOL.json")
    uids = sorted(protocol["uids"])
    require(len(uids) == len(set(uids)) == 16, "Expected exactly 16 registered parents")
    selected = uids if selected_uids is None else list(selected_uids)
    models = list(MODEL_NAMES if models is None else models)
    require(len(selected) == len(set(selected)) and set(selected) <= set(uids), "Unregistered/duplicate selected UID")
    require(len(models) == len(set(models)) and set(models) <= set(MODEL_NAMES), "Unregistered/duplicate model")
    manifest_path = run / "DATA_MANIFEST.json"
    require(file_hash(manifest_path) == protocol["hashes"]["data_manifest_sha256"], "Data manifest changed")
    manifest = read_json(manifest_path)["files"]
    evaluation = protocol["evaluation"]
    require(tuple(evaluation["models"]) == MODEL_NAMES, "Evaluation model contract changed")
    require((evaluation["seed_base"], evaluation["second_seed_base"]) == SEED_BASES,
            "Evaluation seeds differ from frozen protocol")
    require(evaluation["points_each_direction"] == POINTS and evaluation["hot_repeats"] == 3,
            "Evaluation sampling/timing contract changed")
    for relative, expected_hash in protocol["fixed_source_hashes"].items():
        require(file_hash(Path(__file__).parents[1] / relative) == expected_hash,
                "Frozen deployment source changed: " + relative)
    require(len(expected_slots(protocol)) == 164, "Final diagnostic denominator changed")
    support = RadialSupport()
    module_root = Path(__file__).parents[1]
    sources = {str(path.relative_to(module_root)): file_hash(path) for path in module_root.rglob("*.py")}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    write_json(run / "eval_invocations" / f"{stamp}.json", {
        "models": models, "uids": selected, "source_hashes": sources,
        "protocol_sha256": file_hash(run / "RUN_PROTOCOL.json"),
        "evaluation_seed_bases": list(SEED_BASES), "points_each_direction": POINTS,
        "deployment": "unchanged layout_student.deployment.deploy CPU32",
    })
    for name in models:
        started = time.perf_counter()
        load_error = None
        identity = {"kind": name}
        try:
            flow, identity = load_network(run, name, protocol)
        except Exception as exc:
            flow = None
            load_error = {"error": repr(exc), "traceback": traceback.format_exc(), "stage": "checkpoint_load"}
        load_seconds = time.perf_counter() - started
        for uid in selected:
            if uid not in registered_uids(protocol, name):
                continue
            dest = run / "attempts" / uid / name
            if (dest / "result.json").exists():
                previous = read_json(dest / "result.json")
                require(previous["uid"] == uid and previous["model"] == name, "Existing attempt identity mismatch")
                emit({"event": "skip_recorded_attempt", "uid": uid, "model": name, "status": previous["status"]})
                continue
            if dest.exists():
                write_json(dest / "result.json", {"uid": uid, "model": name, "status": "error",
                    "error": "Interrupted attempt directory retained; no automatic retry", "rows": [],
                    "stage": "previous_interruption"})
                aggregate(run)
                continue
            dest.mkdir(parents=True)
            write_json(dest / "START.json", {"uid": uid, "model": name, "identity": identity})
            emit({"event": "start", "uid": uid, "model": name})
            try:
                if load_error is not None:
                    write_json(dest / "result.json", {"uid": uid, "model": name, "status": "error",
                               "rows": [], **load_error})
                    aggregate(run)
                    emit({"event": "error", "uid": uid, "model": name, **load_error})
                    continue
                relative_data = f"data/{uid}.npz"
                data_path = run / relative_data
                require(file_hash(data_path) == manifest[relative_data], "Prepared object changed")
                with np.load(data_path, allow_pickle=False) as z:
                    y, base, faces = z["logr"], z["base_u"], z["faces"]
                    reference_v, reference_f = z["ref_vertices"], z["ref_faces"]
                    proxy = z["proxy_points"]
                    teacher_u = z["teacher_u"] if name == "teacher_reference" else None
                split = "train" if uid in protocol["train_uids"] else "internal_check"
                seeds = [seed + uids.index(uid) for seed in SEED_BASES]
                if name == "teacher_reference":
                    fixed = deploy(None, y, support, base, faces, fixed_only=True)
                    radii = support.query(y, teacher_u)
                    mesh = {"vertices": teacher_u * radii[:, None], "faces": faces, "u": teacher_u, "r": radii}
                    quality = named_quality(mesh["vertices"], faces)
                    audit = audit_mesh(mesh["vertices"], faces, exact=True)
                    result = {"raw_mesh": mesh, "raw_quality": quality, "raw_audit": audit,
                              "floor": fixed["floor"],
                              "timing": {"offline_teacher_reference": True, "generation_seconds": None}}
                    hot = []
                else:
                    result = deploy(flow, y, support, base, faces, fixed_only=name == "fixed")
                    hot = []
                    for _ in range(3):
                        repeat = deploy(flow, y, support, base, faces, fixed_only=name == "fixed")
                        for prefix in ("raw", "delivered"):
                            for key in ("vertices", "faces", "u", "r"):
                                require(np.array_equal(repeat[prefix + "_mesh"][key],
                                                       result[prefix + "_mesh"][key]), "Repeated deployment changed")
                        require(repeat["delivery"] == result["delivery"], "Repeated fallback decision changed")
                        hot.append(repeat["timing"])
                    if name != "fixed":
                        require(result["timing"]["proxy_points_hash"] == array_hash(proxy),
                                "Deployment proxy differs from original input contract")
                result_rows = []
                for policy in POLICIES[name]:
                    prefix = "delivered" if policy == "quality_fallback" else "raw"
                    mesh = result[prefix + "_mesh"]
                    quality, audit = result[prefix + "_quality"], result[prefix + "_audit"]
                    path = dest / f"{policy}.npz"
                    hashes = save_mesh_with_identity(path, mesh, faces)
                    metrics = metric_cache(run, uid, mesh["vertices"], faces, reference_v, reference_f, seeds, POINTS)
                    require(all(item["metrics"]["seed"] == seed and
                                item["metrics"]["points_per_direction"] == POINTS
                                for item, seed in zip(metrics, seeds)), "Cached metric identity mismatch")
                    timing = result["timing"]
                    if policy == "offline_teacher":
                        seconds = hot_seconds = None
                    else:
                        key = ("quality_fallback_total_seconds" if policy == "quality_fallback"
                               else "raw_with_audit_seconds")
                        seconds = timing[key]
                        hot_seconds = float(np.median([record[key] for record in hot]))
                    result_rows.append({
                        "uid": uid, "split": split, "model": name, "policy": policy, "status": "success",
                        "offline_teacher": policy == "offline_teacher",
                        "mesh_file": str(path.relative_to(run)), **hashes,
                        "mainfield_hash": array_hash(y), "data_file_sha256": file_hash(data_path),
                        "identity": identity, "quality": quality, "audit": audit, "quality_floor": result["floor"],
                        **qualification(quality, audit, result["floor"]),
                        "metrics": [item["metrics"] for item in metrics],
                        "metric_cache_evaluation_seconds": [item["evaluation_seconds"] for item in metrics],
                        "fallback_to_fixed": bool(policy == "quality_fallback" and result["delivery"]["fallback_to_fixed"]),
                        "fallback_reasons": result.get("delivery", {}).get("fallback_reasons", [])
                            if policy == "quality_fallback" else [],
                        "generation_seconds": timing["generation_seconds"],
                        "delivery_with_audit_seconds": seconds,
                        "hot_delivery_with_audit_seconds_median_3": hot_seconds,
                        "model_load_seconds": load_seconds, "timing": timing, "hot_timing_repeats": hot,
                        "cost_scope": "raw: export+self audit; fallback policy includes Fixed floor setup+audit+raw+audit+selection; no AE/FM or RMS cost",
                        "independent_evaluation_not_used_for_deployment": True,
                    })
                write_json(dest / "result.json", {"uid": uid, "model": name, "status": "success", "rows": result_rows})
                aggregate(run)
                emit({"event": "complete", "uid": uid, "model": name,
                      "raw_qualified": result_rows[0]["qualified"],
                      "raw_rms": result_rows[0]["metrics"][0]["rms_bbox_percent"]})
            except Exception as exc:
                error = {"uid": uid, "model": name, "status": "error", "error": repr(exc),
                         "traceback": traceback.format_exc(), "rows": []}
                write_json(dest / "result.json", error)
                aggregate(run)
                emit(error)
    return aggregate(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--uids", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    evaluate_run(args.run, args.models, args.uids)


if __name__ == "__main__":
    main()


