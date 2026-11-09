"""Fixed three-model complete-seat Euler50 evaluation: 72 rollouts, 12 head calls.

COMMON and NATIVE have no seam anchors or head calls. SEAM uses the fixed
OLD_S15000 head once per C; no source-owner labels enter inference.
"""
from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse, copy, json, time, traceback
from pathlib import Path
import numpy as np
from .eval_runtime import (
    OUT, P, LABELS, BASELINE_LABEL, HEAD_ROOT, arrays, label_spec, requests,
    new_requests, resolve, read, sha, verify, raw_completion, verify_raw,
    source_assets, reuse_spec, copy_reference,
)

HEAD_BUDGETS = dict(zip(LABELS, (0, 0, 12)))


def freeze_label(label):
    spec = label_spec(label)
    path = resolve(spec["generator_path"])
    receipt = read(path.with_suffix(".json"))
    if sha(path) != receipt["sha256"]:
        raise ValueError("Generator export hash differs")
    inputs = [path, path.with_suffix(".json"), OUT / "registration.json", OUT / "eval_manifest.json"]
    if label == LABELS[2]:
        inputs.extend([HEAD_ROOT / "old_s15000_head_only.pt", HEAD_ROOT / "HEAD_PROTOCOL.json"])
    from r2.pilots.bench_control_transfer_v1 import sample as S
    from r2.pilots.bench_parts_convergence_v1 import data as D
    from r2.pilots.seam_slot_c200_v1 import generate as G
    from meshflow_control.seam import model_adapter as A, occupancy as H
    code = [Path(__file__), Path(__file__).with_name("eval_runtime.py"), Path(__file__).with_name("panel.py"),
            Path(S.__file__), Path(D.__file__), Path(G.__file__), Path(A.__file__), Path(H.__file__)]
    rows = new_requests(label)
    unique = {q["condition_id"]: q for q in rows}
    count = sum(len(D.edge_features(source_assets(q)[0]["C_vertices"],
                    source_assets(q)[0]["C_faces_original"], q["N"])[0]) > 0 for q in unique.values())
    count = count if spec["mode"] == "SEAM" else 0
    if count != HEAD_BUDGETS[label]:
        raise ValueError("Open-condition head budget differs from registration")
    result = dict(status="FROZEN", label=label, spec=spec, outputs=24,
        new_rollouts=len(rows), reused_rollouts=24-len(rows), solver="Euler50",
        rollout_arm=spec["arm"], maximum_head_forwards=count,
        head_precision="BF16 CUDA forward / FP32 parameters", head_threshold_logit=0.0,
        closed_C_head_forwards=0, source_labels_read=False,
        maximum_generator_forwards=50*len(rows), new_noise=0,
        protected={str(p): sha(p) for p in inputs}, code={str(p): sha(p) for p in code})
    result['representation']=spec['mode']; result['historical_reuse']=False
    P.immutable(OUT / "evaluation/inference_registrations" / (label + "__RAW.json"), result)
    return result


def load_head(label, payload):
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.seam.occupancy import OccupancyHead
    if label == LABELS[2]:
        protocol = read(HEAD_ROOT / "HEAD_PROTOCOL.json")
        path = HEAD_ROOT / "old_s15000_head_only.pt"
        if sha(path) != protocol["head_only_sha256"]:
            raise ValueError("Old head-only snapshot changed")
        state = torch.load(path, map_location="cpu", weights_only=True)["head_state"]
        # The CPU audit module clears CUDA_VISIBLE_DEVICES on import. Use the
        # independently verified native digest; never import that audit in GPU code.
        if state_hash(state) != '21229acce25e3e070ae3adfa580798e1534e704cc6512e006de1654d2fefb9c4':
            raise ValueError("Old head tensor identity differs")
        origin = dict(kind="FROZEN_OLD_S15000_HEAD", path=str(path), sha256=sha(path))
    else:
        return None, dict(kind="NONE_NATIVE_ALL_FREE", parameters=0, source_labels_read=False)
    head = OccupancyHead(24).eval().requires_grad_(False)
    head.load_state_dict(state, strict=True)
    if sum(p.numel() for p in head.parameters()) != 833:
        raise ValueError("Unexpected head architecture")
    identity = dict(**origin, state_sha256=state_hash(state), parameters=833,
                    threshold_logit=0.0, input="C_ONLY_24_FEATURES", source_labels_read=False)
    return head.to("cuda"), identity


def save_layout(q, label, head, head_identity):
    import torch
    from r2.pilots.bench_parts_convergence_v1.data import edge_features, inference_layout
    from r2.pilots.bench_control_transfer_v1 import train as T
    core, source = source_assets(q)
    dest = OUT / "evaluation" / label / "layouts" / q["condition_id"]
    edges, features = edge_features(core["C_vertices"], core["C_faces_original"], core["total_N"])
    if (dest / "receipt.json").exists():
        r = read(dest / "receipt.json")
        verify(r["files"])
        if r["head"] != head_identity or r["input_sha256"] != q["inference_npz_sha256"]:
            raise ValueError("Saved layout head/input identity differs")
        a = arrays(dest / "layout.npz")
        lay = inference_layout(**core, occupied=a["occupied"])
        for key in ("anchor_ids", "edge_records", "seam_slots", "bulk_slots", "known_mask"):
            if not np.array_equal(lay[key], a[key]):
                raise ValueError("Saved predicted layout reconstruction differs")
        return lay, a["occupancy_logits"], source, dest, 0
    if dest.exists():
        raise RuntimeError("Unresolved layout attempt, no silent replay: " + str(dest))
    dest.mkdir(parents=True)
    T.write(dest / "attempt.json", dict(condition_id=q["condition_id"], label=label,
        head=head_identity, occupancy_forward_budget=int(head is not None and len(edges) > 0), source_labels_read=False))
    calls = 0
    if head is not None and len(edges):
        T.event(dest / "calls.jsonl", "HEAD_FORWARD_STARTED", budget=1)
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(torch.from_numpy(features).cuda(),
                          torch.ones(len(edges), dtype=torch.bool, device="cuda")).float().cpu().numpy()
        torch.cuda.synchronize()
        T.event(dest / "calls.jsonl", "HEAD_FORWARD_RETURNED", budget=1)
        calls = 1
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
            raise RuntimeError("Unexpected occupancy inference RNG change")
    else:
        logits = np.full(len(edges), -1.0, np.float32) if head is None else np.empty(0, np.float32)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite occupancy logits")
    lay = inference_layout(**core, occupied=logits >= 0)
    P.array_file(dest / "layout.npz", {**lay, "occupancy_logits": logits})
    P.array_file(dest / "source_condition.npz", source)
    P.immutable(dest / "receipt.json", dict(status="COMPLETE", condition_id=q["condition_id"],
        label=label, occupancy_forward_calls=calls, source_labels_read=False,
        input_sha256=q["inference_npz_sha256"], head=head_identity,
        active=int(lay["occupied"].sum()), boundary_edges=len(lay["occupied"]),
        threshold_logit=0.0 if head is not None else None, closed_C_skipped=len(edges) == 0,
        occupancy_logits_applicability="PREDICTED" if head is not None else "NA_NATIVE_NEGATIVE_PLACEHOLDER_NO_HEAD",
        files={str(p): sha(p) for p in dest.iterdir() if p.is_file()}))
    return lay, logits, source, dest, calls


def run(label, estimated_peak_mib=2800):
    spec, qs = label_spec(label), requests(label)
    out = OUT / "evaluation" / label
    if raw_completion(label).exists():
        verify_raw(label)
        return read(raw_completion(label))
    import torch
    from meshflow_control import runtime
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.models.native import build_model
    from meshflow_control.geo_io import write_indexed_ply
    from r2.pilots.bench_control_transfer_v1 import train as T, sample as S
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays, dispatch_guard
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    reg = freeze_label(label)
    gen, gen_hash = resolve(spec["generator_path"]), sha(spec["generator_path"])
    pending = []
    for q in qs:
        if reuse_spec(q, label):
            copy_reference(q, "raw")
            continue
        dest = out / q["request_id"] / "raw"
        if (dest / "receipt.json").exists():
            old = read(dest / "receipt.json")
            verify(old["files"])
            if old["generator"]["sha256"] != gen_hash or old["noise_sha256"] != q["noise_sha256"]:
                raise ValueError("Completed RAW belongs to another generator/noise")
        elif dest.exists():
            raise RuntimeError("Unresolved RAW attempt, no silent replay: " + str(dest))
        else:
            pending.append(q)
    attempt = out / ("sampling_attempt_" + str(time.time_ns()))
    attempt.mkdir(parents=True)
    snapshot = T.resources()
    T.write(attempt / "resources_before.json", snapshot)
    dispatch_guard(snapshot, estimated_peak_mib)
    T.write(attempt / "dispatch.json", dict(workers=1, controller_owns_concurrency=True,
        estimated_process_peak_mib=estimated_peak_mib,
        headroom="max(2GiB,10%total) plus 1.2x observed comparable process peak",
        serial_reason="One model/head owns this label's deterministic per-C layout cache; independent labels may dispatch separately when measured resources permit."))
    model = head = None
    completed = head_calls = 0
    tick = time.perf_counter()
    observer = GPUObserver("full_seat_curriculum_registered_requests", interval=1.)
    observer.__enter__()
    try:
        p = torch.load(gen, map_location="cpu", weights_only=True, mmap=True)
        if (p.get("schema") != "bench_full_seat_curriculum_generator_v1" or p["arm"] != spec["arm"] or int(p["local_step"]) != spec["local_step"]
                or int(p["global_step"]) != spec["global_step"] or p.get("EMA", False)):
            raise ValueError("Wrong frozen live generator endpoint")
        if state_hash(p["generator_state"]) != p["generator_state_sha256"]:
            raise ValueError("Generator tensor hash differs")
        runtime.configure_stable_runtime()
        model = build_model(copy.deepcopy(p["model_config"]), readout_mode="none", trainable=False)
        model.load_state_dict(p["generator_state"], strict=True)
        model.to("cuda").set_trainable(False).eval()
        head, head_identity = load_head(label, p)
        identity = dict(path=str(gen), sha256=gen_hash, state_sha256=p["generator_state_sha256"],
                        local_step=p["local_step"], global_step=p["global_step"], EMA=False)
        T.write(attempt / "model_identity.json", dict(generator=identity, head=head_identity))
        T.write(attempt / "resources_loaded.json", T.resources())
        layouts = {}
        for q in pending:
            cid = q["condition_id"]
            if cid not in layouts:
                lay, logits, source, lp, calls = save_layout(q, label, head, head_identity)
                head_calls += calls
                layouts[cid] = lay, logits, source, lp
            lay, logits, source, lp = layouts[cid]
            dest = out / q["request_id"] / "raw"
            dest.mkdir(parents=True)
            T.write(dest / "attempt.json", dict(label=label, request=q, generator=identity, forward_budget=50))
            raw, trajectory, audit = S.rollout(model, lay, arrays(q["noise_npz"])["z"], spec["mode"], dest / "calls.jsonl")
            mesh = raw_arrays(raw, lay, logits, source)
            P.array_file(dest / "raw.npz", mesh)
            P.array_file(dest / "trajectory.npz", dict(state=trajectory))
            P.array_file(dest / "source_condition.npz", source)
            write_indexed_ply(dest / "raw.ply", mesh["vertices"], mesh["faces"])
            T.write(dest / "audit.json", audit)
            T.write(dest / "receipt.json", dict(status="COMPLETE", execution="NEW", label=label,
                arm=label, request_id=q["request_id"], condition_id=cid, generator=identity,
                checkpoint_sha256=gen_hash, generator_forwards=50, new_generator_forwards=50,
                D0_forwards=0, new_D0_forwards=0, raw_sha256=sha(dest / "raw.npz"),
                noise_sha256=q["noise_sha256"], source_labels_read=False,
                head_forwards=0, head_accounting="Counted once in unique-C layout receipt", layout_receipt=str(lp / "receipt.json"), representation=spec["mode"],
                audit=audit, files={str(f): sha(f) for f in dest.iterdir() if f.is_file()}))
            completed += 1
            if completed <= 2 or completed % 8 == 0:
                T.write(attempt / f"resources_output{completed}.json", T.resources())
                T.write(attempt / f"observed_peak_output{completed}.json", dict(
                    summary=observer.summary(), samples=observer.samples,
                    scoped_rollout_allocator=audit, extra_model_calls=0))
            print(json.dumps(dict(stage="RAW", label=label, completed=completed, pending=len(pending),
                                 request_id=q["request_id"], seconds=audit["seconds"])), flush=True)
        if state_hash(model.state_dict()) != p["generator_state_sha256"]:
            raise ValueError("Frozen generator mutated")
        if head is not None and state_hash(head.state_dict()) != head_identity["state_sha256"]:
            raise ValueError("Frozen occupancy head mutated")
        all_layouts = {q["condition_id"] for q in new_requests(label)}
        total_head = sum(read(out / "layouts" / cid / "receipt.json")["occupancy_forward_calls"] for cid in all_layouts)
        if total_head != HEAD_BUDGETS[label]:
            raise ValueError("Total registered head calls differ")
        verify(reg["protected"])
        verify(reg["code"])
        result = dict(status="COMPLETE", label=label, subset=spec["subset"], outputs=24,
            new_rollouts_this_attempt=completed, new_forwards_this_attempt=50*completed,
            new_head_forwards_this_attempt=head_calls, total_new_rollouts=len(new_requests(label)),
            total_new_generator_forwards=50*len(new_requests(label)), total_new_head_forwards=total_head,
            reused_rollouts=24-len(new_requests(label)), checkpoint_sha256=gen_hash,
            seconds=time.perf_counter()-tick,
            files={str(out/q["request_id"]/"raw/receipt.json"): sha(out/q["request_id"]/"raw/receipt.json") for q in qs})
        P.immutable(raw_completion(label), result)
        return result
    except BaseException as exc:
        T.write(attempt / "failure.json", dict(error=repr(exc), traceback=traceback.format_exc(),
            completed_rollouts=completed, returned_head_calls=head_calls, no_automatic_retry=True))
        raise
    finally:
        observer.__exit__(None, None, None)
        T.write(attempt / "resource_observation.json", dict(summary=observer.summary(), samples=observer.samples))
        T.write(attempt / "resources_finished.json", T.resources())
        del model, head
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", choices=LABELS, required=True)
    p.add_argument("--estimated-peak-mib", type=float, default=2800)
    a = p.parse_args()
    print(json.dumps(run(a.label, a.estimated_peak_mib)), flush=True)
