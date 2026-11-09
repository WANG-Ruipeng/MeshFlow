"""Frozen SHIFT state[25] -> RELATION Euler suffix; two disjoint request writers.

Each shard owns 24 registered requests and exactly 600 new generator forwards.
Shared layouts and the final RAW barrier are prepared/collected by the controller.
No model is imported or invoked by --check-inputs.
"""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import copy
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .eval_plan import (OUT, P, LABELS, arrays, cached_layout, label_spec,
                        requests, resolve, read, sha, verify)

CANDIDATE = "SHIFT_RELATION_T050"
ABSOLUTE_STEPS = tuple(range(25, 50))
DT = 1 / 50


def event(path, name, **fields):
    """A completed forward is counted even if a later finite/export check fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(event=name, unix=time.time(), **fields), allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def shard_requests(qs, shard):
    if shard not in (0, 1) or len(qs) != 48:
        raise ValueError("Exactly two fixed 24-request shards are supported")
    ids = [q["request_id"] for q in qs]
    if len(set(ids)) != 48 or any(Path(r).name != r or r in (".", "..") for r in ids):
        raise ValueError("Unsafe or duplicate frozen request identity")
    return qs[24 * shard:24 * (shard + 1)]


def registered(label, *, execute):
    if label != CANDIDATE or tuple(LABELS) != (CANDIDATE,):
        raise ValueError("Only the single registered crossover candidate may sample")
    spec = label_spec(label)
    reg = read(OUT / "registration.json")
    if execute and reg.get("status") != "FROZEN":
        raise ValueError("Final scientific registration must be frozen before GPU dispatch")
    if execute:
        gate = read(OUT / "blind_review/gate.json")
        if (gate.get("status") != "PASS"
                or gate.get("protocol_sha256") != sha(OUT / "PROTOCOL.md")):
            raise ValueError("Anonymous review gate has not passed for the frozen protocol")
    for key in ("source_hashes", "protected"):
        if not isinstance(reg.get(key), dict) or not reg[key]:
            raise ValueError("Missing scientific protection ledger: " + key)
        verify(reg[key])
    if spec.get("source_prefix_label") != "JOINT_SHIFT_S1500":
        raise ValueError("Wrong prefix model")
    return spec, reg


def load_input(q, label=CANDIDATE):
    """Read authenticated real states; never infer a midpoint from endpoints."""
    trajectory_path = resolve(q["prefix_trajectory_path"])
    receipt_path = resolve(q["prefix_receipt_path"])
    for path, key in ((trajectory_path, "prefix_trajectory_sha256"),
                      (receipt_path, "prefix_receipt_sha256")):
        if sha(path) != q[key]:
            raise ValueError("Frozen prefix evidence changed: " + str(path))
    receipt = read(receipt_path)
    if (receipt["status"] != "COMPLETE" or receipt["label"] != "JOINT_SHIFT_S1500"
            or receipt["request_id"] != q["request_id"] or receipt["generator_forwards"] != 50
            or receipt["noise_sha256"] != q["noise_sha256"]):
        raise ValueError("Prefix receipt identity differs")
    expected = {resolve(p).resolve(): h for p, h in receipt["files"].items()}
    if expected.get(trajectory_path.resolve()) != q["prefix_trajectory_sha256"]:
        raise ValueError("Prefix trajectory is not authenticated by its original receipt")

    layout, logits, source = cached_layout(q)
    folder = OUT / "evaluation" / label / "layouts" / q["condition_id"]
    saved_receipt = read(folder / "receipt.json")
    if saved_receipt["status"] != "COMPLETE":
        raise ValueError("Shared layout must be prepared completely before workers start")
    verify(saved_receipt["files"])
    saved = arrays(folder / "layout.npz")
    for key, value in {**layout, "occupancy_logits": logits}.items():
        a, b = np.asarray(value), np.asarray(saved[key])
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise ValueError("Prepared layout differs from production layout: " + key)
    saved_source = arrays(folder / "source_condition.npz")
    if set(source) != set(saved_source) or any(
            np.asarray(source[k]).dtype != np.asarray(saved_source[k]).dtype
            or np.asarray(source[k]).shape != np.asarray(saved_source[k]).shape
            or np.asarray(source[k]).tobytes() != np.asarray(saved_source[k]).tobytes()
            for k in source):
        raise ValueError("Prepared source-C export metadata differs")

    tr = arrays(trajectory_path)
    if set(tr) != {"state"}:
        raise ValueError("Expected production trajectory key state, not interpolated endpoints")
    states = tr["state"]
    n, k = int(q["N"]), int(q["K"])
    if states.shape != (51, n, 3, 3) or states.dtype != np.float32 or not np.isfinite(states).all():
        raise ValueError("Invalid original FP32 trajectory")
    fixed = layout["anchor_ids"] >= 0
    anchors = layout["C_vertices"][layout["anchor_ids"][fixed]]
    if not all(np.array_equal(state[fixed], anchors) for state in states):
        raise ValueError("An original C/seam anchor moved")
    noise = arrays(q["noise_npz"])["z"]
    initial = np.zeros((n, 3, 3), np.float32)
    initial[k:] = noise
    initial[fixed] = anchors
    if not np.array_equal(initial, states[0]):
        raise ValueError("Prefix initial state differs from registered noise and anchors")
    if "prefix_raw_path" in q:
        raw_path = resolve(q["prefix_raw_path"])
        if sha(raw_path) != q["prefix_raw_sha256"]:
            raise ValueError("Prefix RAW changed")
        if not np.array_equal(arrays(raw_path)["raw"], states[-1]):
            raise ValueError("Original trajectory endpoint differs from RAW")
    return states[25].copy(), layout, logits, saved_source, folder


def suffix(model, layout, initial, ledger):
    """Identical torch Euler arithmetic to the production sampler, steps 25..49."""
    import torch
    from meshflow_control.models.native import native_execution
    from meshflow_control.seam.model_adapter import seam_forward
    n = int(layout["N"])
    fixed = layout["anchor_ids"] >= 0
    state = torch.from_numpy(initial.reshape(1, n, 9)).cuda()
    anchor = torch.from_numpy(fixed[None]).cuda()
    free = ~anchor
    condition = torch.from_numpy(initial[None]).cuda()
    known = torch.from_numpy(layout["known_mask"][None]).cuda()
    valid = torch.ones_like(known)
    y = torch.tensor([n], dtype=torch.int64, device="cuda")
    trajectory = [initial.copy()]
    velocities = []
    rng_cpu = torch.get_rng_state().clone()
    rng_cuda = torch.cuda.get_rng_state().clone()
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    with torch.no_grad(), native_execution(model, coordinates=state):
        for step in ABSOLUTE_STEPS:
            t = torch.full((1,), step / 50, dtype=torch.float32, device="cuda")
            event(ledger, "FORWARD_STARTED", absolute_step=step, t=step / 50)
            v = seam_forward(model, state, t, y, valid, known, anchor)
            torch.cuda.synchronize()
            event(ledger, "FORWARD_COMPLETE", absolute_step=step, t=step / 50)
            if v.dtype != torch.bfloat16 or not bool(torch.isfinite(v).all()):
                raise FloatingPointError("Invalid suffix BF16 velocity")
            cur = state.reshape(1, n, 3, 3)
            nxt = cur.clone()
            nxt[free] = cur[free] + v.float().reshape_as(cur)[free] * (1 / 50)
            nxt[anchor] = condition[anchor]
            if not bool(torch.isfinite(nxt).all()):
                raise FloatingPointError("Nonfinite suffix state")
            state = nxt.reshape(1, n, 9)
            trajectory.append(nxt[0].cpu().numpy().copy())
            vv = v.float().reshape(1, n, 3, 3)[0].cpu().numpy().copy()
            vv[fixed] = 0
            velocities.append(vv)
    if not torch.equal(rng_cpu, torch.get_rng_state()) or not torch.equal(rng_cuda, torch.cuda.get_rng_state()):
        raise RuntimeError("Unexpected suffix sampling RNG use")
    traj = np.stack(trajectory)
    if len(traj) != 26 or any(not np.array_equal(s[fixed], initial[fixed]) for s in traj):
        raise ValueError("Suffix anchor preservation or step count failed")
    return traj[-1], dict(state=traj, velocities=np.stack(velocities),
                         absolute_steps=np.arange(25, 51, dtype=np.int64),
                         times=np.arange(25, 51, dtype=np.float64) / 50), dict(
        generator_forwards=25, cached_prefix_forwards=25, start_t=.5, end_t=1.,
        anchored_all26_exact=True, RNG_unchanged=True, seconds=time.perf_counter() - tick,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_scope="Real 25-step suffix after model loading; allocator is not process VRAM")


def ledger_counts(qs, out):
    counts = dict(started=0, completed=0)
    for q in qs:
        path = out / q["request_id"] / "raw/calls.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                name = json.loads(line)["event"]
                counts["started"] += int(name == "FORWARD_STARTED")
                counts["completed"] += int(name == "FORWARD_COMPLETE")
    return counts


def check_inputs(label=CANDIDATE):
    spec, reg = registered(label, execute=False)
    qs = requests(label)
    parts = [shard_requests(qs, s) for s in (0, 1)]
    if {q["request_id"] for q in parts[0]} & {q["request_id"] for q in parts[1]}:
        raise AssertionError("Shard overlap")
    checks = []
    for q in qs:
        x, lay, _, _, _ = load_input(q, label)
        checks.append(dict(request_id=q["request_id"], shape=list(x.shape),
                           N=int(lay["N"]), K=int(lay["K"]),
                           real_state_index=25, finite=True, fixed_anchors_exact=True))
    return dict(status="PASS", requests=48, shards=[24, 24], suffix_steps=list(ABSOLUTE_STEPS),
                authorized_future_forwards=1200, model_calls=0, D0_calls=0,
                OT_calls=0, new_noise=0, checks=checks)


def run(label, shard, estimated_peak_mib=2800):
    spec, reg = registered(label, execute=True)
    qs = shard_requests(requests(label), shard)
    out = OUT / "evaluation" / label
    attempt = out / ("sampling_shard_" + str(shard))
    done = attempt / "complete.json"
    if done.exists():
        old = read(done)
        if old["request_ids"] != [q["request_id"] for q in qs] or old["new_forwards"] != 600:
            raise ValueError("Completed shard differs from frozen assignment")
        verify(old["files"])
        for path in old["files"]:
            verify(read(path)["files"])
        return old
    if attempt.exists() or any((out / q["request_id"] / "raw").exists() for q in qs):
        raise RuntimeError("Unresolved shard attempt; no implicit retry or replay")
    # Validate the owned CPU inputs before importing torch or allocating CUDA.
    for q in qs:
        load_input(q, label)
    from r2.pilots.bench_control_transfer_v1 import train as T
    from r2.pilots.seam_slot_c200_v1.generate import dispatch_guard
    snapshot = T.resources()
    dispatch_guard(snapshot, estimated_peak_mib)
    attempt.mkdir(parents=True)
    P.immutable(attempt / "resources_before.json", snapshot)
    P.immutable(attempt / "attempt.json", dict(label=label, shard=shard,
        request_ids=[q["request_id"] for q in qs], new_forward_budget=600,
        cached_prefix_forwards=600, no_replay=True, registration_sha256=sha(OUT / "registration.json")))
    P.immutable(attempt / "dispatch.json", dict(workers_in_this_process=1,
        controller_owns_concurrency=True, estimated_process_peak_mib=estimated_peak_mib,
        headroom="max(2GiB,10%total) plus 1.2x process estimate",
        shared_layouts_read_only=True, independent_output_writer=True))
    model = None
    observer = None
    count = 0
    tick = time.perf_counter()
    try:
        import torch
        from meshflow_control import runtime
        from meshflow_control.checkpoints import state_hash
        from meshflow_control.models.native import build_model
        from meshflow_control.geo_io import write_indexed_ply
        from r2.pilots.seam_slot_c200_v1.generate import raw_arrays
        from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
        observer = GPUObserver("crossover_real_suffix_shard_" + str(shard), interval=1.)
        observer.__enter__()
        gen = resolve(spec["generator_path"])
        gen_receipt = read(gen.with_suffix(".json"))
        gen_hash = sha(gen)
        if gen_hash != gen_receipt["sha256"]:
            raise ValueError("RELATION export hash differs")
        p = torch.load(gen, map_location="cpu", weights_only=True, mmap=True)
        if (p["arm"] != spec["source_arm"] or int(p["local_step"]) != 1500
                or int(p["global_step"]) != 13500 or p.get("EMA", False)):
            raise ValueError("Wrong suffix model identity")
        if state_hash(p["generator_state"]) != p["generator_state_sha256"]:
            raise ValueError("Suffix generator tensor hash differs")
        runtime.configure_stable_runtime()
        torch.set_num_threads(1)
        model = build_model(copy.deepcopy(p["model_config"]), readout_mode="none", trainable=False)
        model.load_state_dict(p["generator_state"], strict=True)
        model.to("cuda").set_trainable(False).eval()
        identity = dict(path=str(gen), sha256=gen_hash,
            state_sha256=p["generator_state_sha256"], source_arm=p["arm"],
            local_step=p["local_step"], global_step=p["global_step"], EMA=False)
        if state_hash(model.state_dict()) != identity["state_sha256"]:
            raise ValueError("Loaded generator identity differs")
        del p
        P.immutable(attempt / "model_identity.json", identity)
        P.immutable(attempt / "resources_loaded.json", T.resources())
        for q in qs:
            initial, lay, logits, source, lp = load_input(q, label)
            dest = out / q["request_id"] / "raw"
            dest.mkdir(parents=True, exist_ok=False)
            P.immutable(dest / "attempt.json", dict(label=label, shard=shard,
                request=q, generator=identity, new_forward_budget=25, cached_prefix_steps=25))
            raw, trajectory, audit = suffix(model, lay, initial, dest / "calls.jsonl")
            mesh = raw_arrays(raw, lay, logits, source)
            P.array_file(dest / "raw.npz", mesh)
            P.array_file(dest / "trajectory.npz", trajectory)
            P.array_file(dest / "source_condition.npz", source)
            write_indexed_ply(dest / "raw.ply", mesh["vertices"], mesh["faces"])
            P.immutable(dest / "audit.json", audit)
            P.immutable(dest / "receipt.json", dict(status="COMPLETE", label=label, arm=label,
                request_id=q["request_id"], condition_id=q["condition_id"], generator=identity,
                checkpoint_sha256=gen_hash, generator_forwards=25, cached_prefix_forwards=25,
                total_effective_generator_steps=50, D0_forwards=0, head_forwards=0,
                prefix_model="JOINT_SHIFT_S1500", suffix_model="RELATION_V1_S1500",
                prefix_trajectory_path=q["prefix_trajectory_path"],
                prefix_trajectory_sha256=q["prefix_trajectory_sha256"],
                prefix_receipt_sha256=q["prefix_receipt_sha256"], prefix_state_index=25,
                raw_sha256=sha(dest / "raw.npz"), noise_sha256=q["noise_sha256"],
                source_labels_read=False, layout_receipt=str(lp / "receipt.json"), audit=audit,
                files={str(f): sha(f) for f in dest.iterdir() if f.is_file()}))
            count += 1
            if count <= 2 or count % 8 == 0:
                P.immutable(attempt / f"resources_output{count}.json", T.resources())
                P.immutable(attempt / f"observed_peak_output{count}.json", dict(
                    summary=observer.summary(), samples=observer.samples,
                    scoped_suffix_allocator=audit, extra_model_calls=0))
            print(json.dumps(dict(stage="RAW_SUFFIX", label=label, shard=shard,
                completed=count, total=24, request_id=q["request_id"], seconds=audit["seconds"])), flush=True)
        if state_hash(model.state_dict()) != identity["state_sha256"]:
            raise ValueError("Frozen suffix generator mutated")
        verify(reg["protected"])
        verify(reg["source_hashes"])
        counts = ledger_counts(qs, out)
        if counts != dict(started=600, completed=600):
            raise ValueError("Shard call ledger is not exactly 600 completed forwards")
        result = dict(status="COMPLETE", label=label, shard=shard, outputs=24,
            request_ids=[q["request_id"] for q in qs], new_rollouts=24, new_forwards=600,
            cached_prefix_forwards=600, new_head_forwards=0, new_D0_forwards=0,
            checkpoint_sha256=gen_hash, seconds=time.perf_counter() - tick, call_ledger=counts,
            files={str(out / q["request_id"] / "raw/receipt.json"):
                   sha(out / q["request_id"] / "raw/receipt.json") for q in qs})
        P.immutable(done, result)
        return result
    except BaseException as exc:
        P.immutable(attempt / "failure.json", dict(status="BLOCKED", error=repr(exc),
            traceback=traceback.format_exc(), completed_outputs=count,
            call_ledger=ledger_counts(qs, out), no_automatic_retry=True))
        raise
    finally:
        if observer is not None:
            observer.__exit__(None, None, None)
            P.immutable(attempt / "resource_observation.json", dict(
                summary=observer.summary(), samples=observer.samples))
        P.immutable(attempt / "resources_finished.json", T.resources())
        if model is not None:
            del model
            import torch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=(CANDIDATE,), default=CANDIDATE)
    parser.add_argument("--shard", type=int, choices=(0, 1))
    parser.add_argument("--estimated-peak-mib", type=float, default=2800)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    if args.check_inputs:
        value = check_inputs(args.label)
    elif args.shard is not None:
        value = run(args.label, args.shard, args.estimated_peak_mib)
    else:
        parser.error("--shard or --check-inputs required")
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
