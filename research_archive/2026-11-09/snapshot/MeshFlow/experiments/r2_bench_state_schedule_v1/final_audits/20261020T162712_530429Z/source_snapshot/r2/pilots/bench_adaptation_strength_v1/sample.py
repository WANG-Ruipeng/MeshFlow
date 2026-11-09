"""Registered eight-label Euler50 sampling; no layout-head calls or new noise."""
from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse, copy, json, time, traceback
from pathlib import Path
from .eval_plan import (
    OUT, P, LABELS, arrays, cached_layout, label_spec, requests, resolve, read,
    sha, verify, raw_completion, verify_raw,
)


def freeze_label(label):
    spec = label_spec(label)
    path = resolve(spec["generator_path"])
    receipt = read(path.with_suffix(".json"))
    if sha(path) != receipt["sha256"]:
        raise ValueError("Generator export hash differs")
    inputs = [path, path.with_suffix(".json"), OUT / "registration.json", OUT / "eval_manifest.json"]
    from r2.pilots.bench_control_transfer_v1 import sample as S
    from r2.pilots.seam_slot_c200_v1 import generate as G
    from meshflow_control.seam import model_adapter as A
    code = [Path(__file__), Path(__file__).with_name("eval_plan.py"), Path(S.__file__), Path(G.__file__), Path(A.__file__)]
    result = dict(status="FROZEN", label=label, spec=spec, outputs=len(requests(label)),
                  solver="Euler50", head_forwards=0, source_labels_read=False,
                  generator_forwards=50 * len(requests(label)), new_noise=0,
                  protected={str(p): sha(p) for p in inputs},
                  code={str(p): sha(p) for p in code})
    P.immutable(OUT / "evaluation/inference_registrations" / (label + ".json"), result)
    return result


def save_layout(q, label):
    lay, logits, source = cached_layout(q)
    dest = OUT / "evaluation" / label / "layouts" / q["condition_id"]
    if (dest / "receipt.json").exists():
        r = read(dest / "receipt.json")
        verify(r["files"])
        if r["old_layout_sha256"] != q["old_layout_sha256"]:
            raise ValueError("Saved layout cache identity changed")
        return lay, logits, source, dest
    if dest.exists():
        raise RuntimeError("Partial layout attempt retained: " + str(dest))
    dest.mkdir(parents=True)
    P.array_file(dest / "layout.npz", {**lay, "occupancy_logits": logits})
    P.array_file(dest / "source_condition.npz", source)
    P.immutable(dest / "receipt.json", dict(
        status="COMPLETE", condition_id=q["condition_id"], label=label,
        occupancy_forward_calls=0, source_labels_read=False,
        head=dict(kind="FROZEN_OLD_S15000_LAYOUT", model_calls=0),
        old_layout_sha256=q["old_layout_sha256"],
        active=int(lay["occupied"].sum()), boundary_edges=len(lay["occupied"]),
        files={str(p): sha(p) for p in dest.iterdir() if p.is_file()}))
    return lay, logits, source, dest


def run(label, estimated_peak_mib=2800):
    spec = label_spec(label)
    qs = requests(label)
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
    reg = freeze_label(label)
    gen = resolve(spec["generator_path"])
    gen_hash = sha(gen)
    pending = []
    for q in qs:
        dest = out / q["request_id"] / "raw"
        if (dest / "receipt.json").exists():
            old = read(dest / "receipt.json")
            verify(old["files"])
            if old["generator"]["sha256"] != gen_hash or old["noise_sha256"] != q["noise_sha256"]:
                raise ValueError("Completed RAW belongs to another generator or noise")
        elif dest.exists():
            raise RuntimeError("Unresolved sampling attempt; no silent retry: " + str(dest))
        else:
            pending.append(q)
    attempt = out / ("sampling_attempt_" + str(time.time_ns()))
    attempt.mkdir(parents=True)
    snapshot = T.resources()
    T.write(attempt / "resources_before.json", snapshot)
    dispatch_guard(snapshot, estimated_peak_mib)
    T.write(attempt / "dispatch.json", dict(workers=1, controller_owns_concurrency=True,
            estimated_process_peak_mib=estimated_peak_mib,
            headroom="max(2GiB,10%total) plus 1.2x comparable measured peak"))
    model = None
    completed = 0
    tick = time.perf_counter()
    try:
        p = torch.load(gen, map_location="cpu", weights_only=True, mmap=True)
        if int(p["local_step"]) != int(spec["local_step"]) or int(p["global_step"]) != int(spec["global_step"]):
            raise ValueError("Wrong checkpoint step")
        if p.get("EMA", False):
            raise ValueError("Expected frozen live generator export")
        if state_hash(p["generator_state"]) != p["generator_state_sha256"]:
            raise ValueError("Generator tensor hash differs")
        runtime.configure_stable_runtime()
        model = build_model(copy.deepcopy(p["model_config"]), readout_mode="none", trainable=False)
        model.load_state_dict(p["generator_state"], strict=True)
        model.to("cuda").set_trainable(False).eval()
        identity = dict(path=str(gen), sha256=gen_hash, state_sha256=p["generator_state_sha256"],
                        local_step=p["local_step"], global_step=p["global_step"], EMA=False)
        T.write(attempt / "model_identity.json", identity)
        T.write(attempt / "resources_loaded.json", T.resources())
        layouts = {}
        for q in pending:
            cid = q["condition_id"]
            if cid not in layouts:
                layouts[cid] = save_layout(q, label)
            lay, logits, source, lp = layouts[cid]
            dest = out / q["request_id"] / "raw"
            dest.mkdir(parents=True)
            T.write(dest / "attempt.json", dict(label=label, request=q, generator=identity, forward_budget=50))
            raw, trajectory, audit = S.rollout(model, lay, arrays(q["noise_npz"])["z"], "SEAM", dest / "calls.jsonl")
            mesh = raw_arrays(raw, lay, logits, source)
            P.array_file(dest / "raw.npz", mesh)
            P.array_file(dest / "trajectory.npz", dict(state=trajectory))
            P.array_file(dest / "source_condition.npz", source)
            write_indexed_ply(dest / "raw.ply", mesh["vertices"], mesh["faces"])
            T.write(dest / "audit.json", audit)
            T.write(dest / "receipt.json", dict(
                status="COMPLETE", label=label, arm=label, request_id=q["request_id"], condition_id=cid,
                generator=identity, checkpoint_sha256=gen_hash, generator_forwards=50, D0_forwards=0,
                raw_sha256=sha(dest / "raw.npz"), noise_sha256=q["noise_sha256"],
                source_labels_read=False, head_forwards=0, layout_receipt=str(lp / "receipt.json"),
                audit=audit, files={str(f): sha(f) for f in dest.iterdir() if f.is_file()}))
            completed += 1
            if completed <= 2 or completed % 8 == 0:
                T.write(attempt / f"resources_output{completed}.json", T.resources())
            print(json.dumps(dict(stage="RAW", label=label, completed=completed, pending=len(pending),
                                 request_id=q["request_id"], seconds=audit["seconds"])), flush=True)
        if state_hash(model.state_dict()) != p["generator_state_sha256"]:
            raise ValueError("Frozen generator mutated")
        verify(reg["protected"])
        verify(reg["code"])
        result = dict(status="COMPLETE", label=label, subset=spec["subset"], outputs=len(qs),
                      new_rollouts=completed, new_forwards=50 * completed, new_head_forwards=0,
                      checkpoint_sha256=gen_hash, seconds=time.perf_counter() - tick,
                      files={str(out / q["request_id"] / "raw/receipt.json"):
                             sha(out / q["request_id"] / "raw/receipt.json") for q in qs})
        P.immutable(raw_completion(label), result)
        return result
    except BaseException as exc:
        T.write(attempt / "failure.json", dict(error=repr(exc), traceback=traceback.format_exc(),
                completed_rollouts=completed, no_automatic_retry=True))
        raise
    finally:
        T.write(attempt / "resources_finished.json", T.resources())
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=LABELS, required=True)
    parser.add_argument("--estimated-peak-mib", type=float, default=2800)
    a = parser.parse_args()
    print(json.dumps(run(a.label, a.estimated_peak_mib)), flush=True)
