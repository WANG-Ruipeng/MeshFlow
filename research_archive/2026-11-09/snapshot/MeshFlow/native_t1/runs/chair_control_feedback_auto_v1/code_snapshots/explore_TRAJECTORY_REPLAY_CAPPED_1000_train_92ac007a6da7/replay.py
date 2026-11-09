"""Frozen CPU feedback selection and supervised late-state replay.

Hard CPU scores select records; gradients flow through a new velocity MSE,
never through the scores. GT is supervision only. No model is loaded here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import OUT, read_json, digest, atomic_json
from .assets import arrays, bits, freeze_json, freeze_npz, resolve
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.coupling import (
    array_hash, nested_ot_map, to_noise_slots, coupling_path, validate_map,
)

VERSION = "chair_control_feedback_late_replay_v1"
SEEDS = (20261008, 20261009)
QUEUES = ("S", "I", "E")
SCORE_FIELDS = {"S": "SeamCov", "I": "ConflictCov_C_upper", "E": "ExposedRetain_global_lower"}
_CACHE = {}


def epsilon_seed(request_id, training_seed):
    if training_seed not in SEEDS:
        raise ValueError("Unregistered training random stream")
    text = json.dumps([VERSION, training_seed, "teacher_epsilon", request_id], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def specification():
    return dict(version=VERSION, source_pool="START/TRAIN_FEEDBACK",
        candidate_count=2, parent_count=8, lambda_aux=.1,
        queue_order=list(QUEUES), queue_index="(step-1)%3",
        parent_index="((step-1)//3)%8 in sorted UID order",
        time_index="((step-1)//24)%2", times=[.6, .8], path_indices=[30, 40],
        selection="Per parent: minimum S lower bound, maximum I upper bound, minimum E global lower bound; exact ties use smaller candidate index",
        score_fields=SCORE_FIELDS, unavailable_score_policy="BLOCKED; never substitute0/1 or remove difficult records",
        trajectory_target="nested_ot_map(GT_free,final_free), both final model coordinates; to_noise_slots(GT_free); u=(Ymatch-X_saved)/(1-t)",
        teacher_target="nested_ot_map(original_pre_ot_free,saved_FP32_Gaussian_free), then target*2 and inverse mapping; mix Gaussian with independent FP32 epsilon at lambda.25; no second OT",
        teacher_noise_note="Extra teacher control uses the exact frozen inference FP32 Z. The unchanged primary eight-sample training stream retains its original pre-match FP64 Gaussian arithmetic.",
        gradient="auxiliary masked mean((model(X_saved,t,C)-u_detached)^2) over9*(N-K) free coordinates; coefficient.1 applied by trainer",
        collation="Use collate_aux: old collate followed by exact saved-Xt override, avoiding recomputation/rounding of the actual trajectory state",
        target_correctness="Targets are immutable true training GT faces; generated endpoints only establish free face/corner correspondence and supply visited states",
        inference_inputs=["xt", "t", "N", "known_mask", "valid_mask"],
        limitations="Composite state/correspondence intervention; old START replay may become off-policy as training progresses. No claim that the frozen hard S/I/E scores themselves are differentiable.")


def _paths(out):
    out = resolve(out)
    return out, out / "feedback/pool_v1", out / "manifests/TRAIN_FEEDBACK.json", out / "evaluation/START/TRAIN_FEEDBACK/per_output.json"


def _validate_frozen(manifest, folder):
    for path, expected in manifest["source_hashes"].items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError("Frozen replay source changed: " + path)
    for record in manifest["records"]:
        path = folder / record["payload"]
        if digest(path) != record["payload_sha256"]:
            raise ValueError("Frozen replay payload changed: " + str(path))
    if manifest["replay_source_sha256"] != digest(__file__):
        raise ValueError("Frozen replay implementation differs")
    return manifest


def prepare_feedback(out=OUT):
    """Freeze all16 scored training trajectories and both registered controls.

    Missing raw/scores return BLOCKED, without freezing a partial active pool.
    Completed pools are revalidated and reused; their OT maps never recomputed.
    """
    out, folder, manifest_path, score_path = _paths(out)
    pool_path = folder / "manifest.json"
    if pool_path.exists():
        return _validate_frozen(read_json(pool_path), folder)
    missing = [str(p) for p in (manifest_path, score_path) if not p.is_file()]
    if missing:
        status = dict(status="BLOCKED", version=VERSION, missing=missing, model_calls=0)
        atomic_json(out / "feedback/preparation_status.json", status)
        return status
    requests = read_json(manifest_path)["rows"]
    scores = read_json(score_path)
    if len(requests) != 16 or len(scores) != 16:
        raise ValueError("Require all16 registered TRAIN_FEEDBACK results")
    score_map = {row["request_id"]: row for row in scores}
    if len(score_map) != 16 or set(score_map) != {r["request_id"] for r in requests}:
        raise ValueError("Feedback scores do not match the fixed request set")
    by_parent = defaultdict(list)
    source_hashes = {str(manifest_path): digest(manifest_path), str(score_path): digest(score_path)}
    for request in requests:
        rid = request["request_id"]
        score = score_map[rid]
        if request["split"] != "TRAIN_FEEDBACK" or request["status"] != "READY" or score["split"] != "TRAIN_FEEDBACK":
            raise ValueError("Only registered ready training feedback may enter learning")
        if score["status"] != "PASS":
            raise ValueError("Hard/scoring failure in feedback: " + rid)
        for key in ("uid", "condition_id", "candidate"):
            if score[key] != request[key]:
                raise ValueError("Feedback row identity differs: " + rid)
        for field in SCORE_FIELDS.values():
            value = score.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("Feedback NA/undefined score cannot be ranked: " + rid + "/" + field)
        by_parent[request["uid"]].append(request)
        raw = out / "runtime/START" / rid / "raw.npz"
        if not raw.is_file(): missing.append(str(raw))
    if missing:
        status = dict(status="BLOCKED", version=VERSION, missing=missing, model_calls=0)
        atomic_json(out / "feedback/preparation_status.json", status)
        return status
    if len(by_parent) != 8 or any(sorted(r["candidate"] for r in rows) != [0, 1] for rows in by_parent.values()):
        raise ValueError("Require exactly8 training parents and candidates0/1")
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for request in sorted(requests, key=lambda r: (r["uid"], r["candidate"])):
        rid, n, k = request["request_id"], int(request["N"]), int(request["K"])
        raw_path = out / "runtime/START" / rid / "raw.npz"
        bp, cp = resolve(request["bank_npz"]), resolve(request["condition_npz"])
        if digest(bp) != request["bank_sha256"] or digest(cp) != request["condition_sha256"]:
            raise ValueError("Feedback registered inputs changed")
        if score_map[rid]["raw_sha256"] != digest(raw_path):
            raise ValueError("Feedback score did not evaluate these raw bytes")
        source_hashes.update({str(p): digest(p) for p in (raw_path, bp, cp)})
        bank, condition, raw = arrays(bp), arrays(cp), arrays(raw_path)
        C, gt, pre = condition["C"], condition["GT_free"], condition["pre_ot_free"]
        output, path = raw["output"], raw["path"]
        if output.shape != (n, 9) or path.shape != (51, n, 9) or bank["z"].shape != (1, n, 9):
            raise ValueError("Unexpected raw trajectory/inference-bank shape")
        if path.dtype != np.float32 or output.dtype != np.float32 or not np.isfinite(path).all() or not np.isfinite(output).all():
            raise ValueError("Feedback path must be finite original FP32")
        if not bits(output, path[-1]) or not all(bits(path[i, :k].reshape(k, 3, 3), C) for i in range(51)):
            raise ValueError("Feedback output/path/C contract failed")
        if not bits(bank["C"], C) or not bits((pre * np.float32(2)).reshape(-1, 3, 3), gt):
            raise ValueError("Feedback C or original preOT*2 GT identity failed")
        if not np.array_equal(bank["known_mask"], np.arange(n) < k) or not bank["valid_mask"].all():
            raise ValueError("Feedback bank must preserve registered known-first slots")
        z = bank["z"][0, k:].reshape(n-k, 3, 3)
        if not bits(path[0, k:].reshape(n-k, 3, 3), z):
            raise ValueError("Saved path did not start from the registered Gaussian")
        trajectory_face, trajectory_corner = nested_ot_map(gt, output[k:].reshape(n-k, 3, 3))
        trajectory_target = to_noise_slots(gt, trajectory_face, trajectory_corner)
        teacher_face, teacher_corner = nested_ot_map(pre, z)
        teacher_target = to_noise_slots((pre * np.float32(2)).astype(np.float32), teacher_face, teacher_corner)
        payload = dict(C=C, known_mask=bank["known_mask"], valid_mask=bank["valid_mask"],
            N=np.asarray(n, np.int64), K=np.asarray(k, np.int64), z=bank["z"],
            xt30=path[30], xt40=path[40], trajectory_target=trajectory_target,
            trajectory_face_map=trajectory_face, trajectory_corner_map=trajectory_corner,
            teacher_target=teacher_target, teacher_face_map=teacher_face, teacher_corner_map=teacher_corner)
        eps_records = {}
        for training_seed in SEEDS:
            seed = epsilon_seed(rid, training_seed)
            epsilon = np.random.RandomState(seed).randn(n-k, 3, 3).astype(np.float32)
            mixed, _, _ = coupling_path(z, teacher_target, epsilon, np.asarray(.6, np.float32), .25)
            payload["teacher_epsilon_"+str(training_seed)] = epsilon
            payload["teacher_x0_"+str(training_seed)] = mixed
            eps_records[str(training_seed)] = dict(seed=seed, epsilon_sha256=array_hash(epsilon), mixed_sha256=array_hash(mixed))
        payload_path = folder / (rid + ".npz")
        freeze_npz(payload_path, payload)
        records.append(dict(request_id=rid, uid=request["uid"], candidate=request["candidate"],
            condition_id=request["condition_id"], task_id=request["source_task_id"], N=n, K=k,
            payload=payload_path.name, payload_sha256=digest(payload_path),
            array_sha256={key: array_hash(a) for key, a in payload.items()},
            raw_path=str(raw_path), raw_sha256=digest(raw_path), bank_sha256=digest(bp),
            condition_sha256=digest(cp), C_all51_bitwise=True, original_GT_preserved=True,
            epsilons=eps_records, selection_values={q: score_map[rid][field] for q, field in SCORE_FIELDS.items()},
            full_feedback_score=score_map[rid], actual_CPU_OT_maps=2))
    queues = {}
    for queue in QUEUES:
        queues[queue] = {}
        for uid, candidates in sorted(by_parent.items()):
            ordered = sorted(candidates, key=lambda r: ((-1 if queue=="I" else 1) * score_map[r["request_id"]][SCORE_FIELDS[queue]], r["candidate"]))
            queues[queue][uid] = dict(selected_request_id=ordered[0]["request_id"],
                metric=SCORE_FIELDS[queue], candidates=[dict(request_id=r["request_id"], candidate=r["candidate"], value=score_map[r["request_id"]][SCORE_FIELDS[queue]]) for r in ordered])
    manifest = dict(status="PASS", version=VERSION, specification=specification(),
        replay_source_sha256=digest(__file__), source_hashes=source_hashes,
        sorted_parents=sorted(by_parent), queues=queues, records=records,
        source_model="START", real_feedback_rollouts=16, actual_CPU_OT_maps=32,
        model_calls=0, optimizer_updates=0, feedback_scores_combined_into_scalar=False,
        GT_used_only_for_supervision=True, DEV_and_CONFIRM_scores_read=False)
    freeze_json(pool_path, manifest)
    atomic_json(out / "feedback/preparation_status.json", dict(status="PASS", manifest=str(pool_path), manifest_sha256=digest(pool_path)))
    return manifest


def _load_pool(out):
    out, folder, _, _ = _paths(out)
    pool_path = folder / "manifest.json"
    key = (str(out), digest(pool_path))
    if key not in _CACHE:
        manifest = _validate_frozen(read_json(pool_path), folder)
        _CACHE[key] = (manifest, {r["request_id"]: r for r in manifest["records"]}, {})
    return _CACHE[key], folder


def aux_sample(step, arm, device=None, *, out=OUT, training_seed=20261008):
    """Return a detached NumPy sample, or its ready model batch if device set.

    This does not move the primary stream cursor and performs no model call.
    Use collate_aux rather than bare historical collate for returned samples.
    """
    if arm == "FM_CONTROL": return None
    if arm not in ("REPLAY_FM_CONTROL", "TRAJECTORY_REPLAY") or training_seed not in SEEDS:
        raise ValueError("Unregistered auxiliary arm/random stream")
    if type(step) is not int or not 1 <= step <= 1000:
        raise ValueError("Auxiliary step must be1..1000")
    (manifest, records, cache), folder = _load_pool(out)
    queue = QUEUES[(step-1)%3]
    uid = manifest["sorted_parents"][((step-1)//3)%8]
    rid = manifest["queues"][queue][uid]["selected_request_id"]
    record = records[rid]
    if rid not in cache:
        cache[rid] = arrays(folder / record["payload"])
    p = cache[rid]
    t_index = ((step-1)//24)%2
    t = np.asarray([.6, .8][t_index], np.float32)
    state_index = [30, 40][t_index]
    n, k = record["N"], record["K"]
    context = np.zeros((n, 9), np.float32); context[:k] = p["C"].reshape(k, 9)
    x0 = np.zeros((n, 9), np.float32); x1 = np.zeros_like(x0)
    x0[:k] = context[:k]; x1[:k] = context[:k]
    if arm == "TRAJECTORY_REPLAY":
        xt = p["xt"+str(state_index)].copy()
        x0[:] = xt
        x1[k:] = p["trajectory_target"].reshape(n-k, 9)
        u = np.zeros_like(xt)
        u[k:] = (x1[k:] - xt[k:]) / (np.float32(1)-t)
        map_kind = "trajectory"
    else:
        x0[k:] = p["teacher_x0_"+str(training_seed)].reshape(n-k, 9)
        x1[k:] = p["teacher_target"].reshape(n-k, 9)
        xt = (np.float32(1)-t)*x0+t*x1
        xt[:k] = context[:k]
        u = x1-x0
        map_kind = "teacher"
    if not bits(xt[:k], context[:k]) or np.any(u[:k]) or not np.isfinite(u).all():
        raise ValueError("Auxiliary C/free/finite invariant failed")
    audit = dict(version=VERSION, pool_manifest_sha256=digest(folder/"manifest.json"),
        step=step, arm=arm, training_seed=training_seed, queue=queue, uid=uid,
        request_id=rid, candidate=record["candidate"], t=float(t), state_index=state_index,
        payload_sha256=record["payload_sha256"], selection=manifest["queues"][queue][uid],
        correspondence=map_kind, target_sha256=array_hash(p[map_kind+"_target"]),
        xt_sha256=array_hash(xt), u_sha256=array_hash(u), C_sha256=array_hash(p["C"]),
        epsilon=record["epsilons"][str(training_seed)] if map_kind=="teacher" else None,
        denominator=9*(n-k), lambda_aux=.1, requires_exact_xt_override=True)
    sample = dict(x0=x0, x1=x1, u=u, xt=xt, context=context,
        t=t, y=np.asarray(n, np.int64), known_mask=p["known_mask"].copy(), valid_mask=p["valid_mask"].copy(),
        object_id=uid, task_id=record["task_id"], patch_id=record["task_id"], K=k,
        alpha=1., sample_index=step-1, pilot_step=step, feedback_step=step,
        conditional_total_step=5000+step, free_coordinate_count=9*(n-k),
        shared_input_sha256=array_hash(xt), aux_audit=audit)
    return collate_aux(sample, device) if device is not None else sample


def collate_aux(sample, device="cpu"):
    """Reuse shape/mask handling, but feed precisely the saved auxiliary Xt."""
    import torch
    from experiments.mf_h1_local0.control_mvp48.train_stream import collate
    batch = collate([sample], device)
    batch["xt"] = torch.from_numpy(np.ascontiguousarray(sample["xt"])[None].copy()).to(device)
    batch["u"] = torch.from_numpy(np.ascontiguousarray(sample["u"])[None].copy()).to(device)
    batch["aux_audit"] = sample["aux_audit"]
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    value = prepare_feedback(args.out)
    print(json.dumps({k: value[k] for k in ("status", "version", "actual_CPU_OT_maps", "missing") if k in value}, indent=2))
