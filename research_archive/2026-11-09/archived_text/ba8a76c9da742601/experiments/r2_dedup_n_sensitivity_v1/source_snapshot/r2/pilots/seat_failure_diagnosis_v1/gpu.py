"""Frozen EXP67 inference and single D0 for the seat diagnosis.

Only generate/post may load a model. Worker partitions own disjoint cells;
reuse and collect are CPU-only, single-writer operations. A reserved model
call without its completed cell is never replayed automatically.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from . import common as C

LABEL = "EXP67_FROZEN"
CHECKPOINT_SHA = "2e2a40bcacda7e5ee04a5754018f71986ef5d378345058075ba412bec808f925"


def old():
    return C.module("_seat_diagnosis_original_evaluation", C.ROOT / "r2/pilots/user_condition_finetune_v1/evaluation.py")


def event(name, worker=None, workers=None, **fields):
    suffix = "controller" if worker is None else f"worker_{worker}_of_{workers}"
    path = C.checked(C.OUT / "execution" / (suffix + "_events.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(event=name, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def registration():
    r = C.load_registration()
    if r.get("status") != "FROZEN" or r["checkpoint"]["sha256"] != CHECKPOINT_SHA:
        raise ValueError("Unfrozen registration or wrong EXP67 endpoint")
    requests = r["requests"]
    if len({q["request_id"] for q in requests}) != len(requests):
        raise ValueError("Duplicate request identity")
    if any(q.get("action") not in ("NEW", "CACHE") for q in requests):
        raise ValueError("Each request needs an explicit NEW/CACHE action")
    count = sum(q["action"] == "NEW" for q in requests)
    b = r["budget"]
    # The registration owns the final count; no diagnostic forward is added.
    if b.get("rollouts", b.get("new_rollouts")) != count:
        raise ValueError("Registered rollout budget differs from NEW cells")
    if b.get("generator_forwards_max", b.get("generator_sampling_forwards")) != 50 * count:
        raise ValueError("Registered generator forward budget differs from Euler50")
    if b.get("D0_forwards_max") != count:
        raise ValueError("Registered D0 budget differs from NEW cells")
    if b.get("training_updates", 0) != 0 or b.get("backwards", 0) != 0:
        raise ValueError("This is a zero-training diagnostic")
    return r


def record_path(q, stage):
    return C.OUT / "evaluation" / stage / LABEL / q["request_id"] / "receipt.json"


def source(q, condition):
    if C.sha(q["condition_source"]) != q["condition_source_sha256"]:
        raise ValueError("Frozen source metadata changed")
    C.package()
    from meshflow_control.source_winding import validate_source
    arrays = C.arrays(q["condition_source"])
    validate_source(arrays, condition)
    return arrays


def verify_ledger(path):
    events = [json.loads(line) for line in C.resolve(path).read_text(encoding="utf-8-sig").splitlines()]
    if len(events) != 103 or events[0].get("event") != "CELL_REGISTERED":
        raise ValueError("Incomplete or extra Euler call ledger")
    if events[1].get("event") != "RESERVE" or events[1].get("kind") != "rollout_attempt":
        raise ValueError("Missing rollout reservation")
    for step in range(50):
        a, b = events[2 + 2 * step:4 + 2 * step]
        if (a.get("event"), a.get("kind"), a.get("ordinal"), a.get("context", {}).get("euler_step")) != ("RESERVE", "sampling_forward", step + 1, step):
            raise ValueError("Sampling reservation mismatch")
        if (b.get("event"), b.get("kind")) != ("RETURN", "sampling_forward"):
            raise ValueError("Reserved sampling call has no return; no replay")
    if events[-1].get("event") != "ROLLOUT_COMPLETE" or events[-1].get("nfe") != 50:
        raise ValueError("Missing rollout completion")
    return dict(sampling_reserved=50, sampling_returned=50, unresolved=0, sha256=C.sha(path))


def read_raw(q, binding):
    rec = C.read(record_path(q, "raw"))
    if rec["request_id"] != q["request_id"] or rec["input_sha256"] != q["input_sha256"]:
        raise ValueError("RAW receipt request/input identity mismatch")
    arrays, _ = old().verify_output(q, rec, binding["sha256"])
    source(q, arrays["C"])
    if q["action"] == "NEW":
        if rec.get("new_rollouts") != 1:
            raise ValueError("NEW cell has no rollout")
        verify_ledger(record_path(q, "raw").parent / "calls.jsonl")
        if C.sha(rec["velocities_path"]) != rec["velocities_sha256"]:
            raise ValueError("Saved velocities changed")
        v = C.arrays(rec["velocities_path"])["velocity"]
        n, k = q["N"], q["K"]
        path = arrays["trajectory"].reshape(51, n, 9)
        if v.shape != (50, n, 9) or v.dtype != np.float32 or not np.isfinite(v).all():
            raise ValueError("Invalid saved BF16 velocities in FP32 storage")
        if np.any(v.view(np.uint32) & 0xffff):
            raise ValueError("Velocity values are not lossless BF16")
        if not all(C.same(path[s + 1, k:], path[s, k:] + v[s, k:] * np.float32(1 / 50)) for s in range(50)):
            raise ValueError("Euler recurrence mismatch")
    else:
        if rec.get("new_rollouts") != 0 or C.sha(q["cache_raw_receipt"]) != q["cache_raw_receipt_sha256"]:
            raise ValueError("CACHE receipt changed")
    return rec


def read_post(q, raw):
    rec = C.read(record_path(q, "post"))
    if rec.get("status") != "COMPLETE" or rec["raw_sha256"] != raw["raw_sha256"]:
        raise ValueError("POST completion/RAW identity mismatch")
    if rec["source_sha256"] != q["condition_source_sha256"]:
        raise ValueError("POST source metadata identity mismatch")
    for path, digest in rec["files"].items():
        if C.sha(path) != digest:
            raise ValueError("POST artifact changed: " + str(path))
    if C.sha(rec["post_path"]) != rec["post_sha256"]:
        raise ValueError("POST geometry changed")
    if q["action"] == "CACHE" and C.sha(q["cache_post_receipt"]) != q["cache_post_receipt_sha256"]:
        raise ValueError("Historical POST receipt changed")
    return rec


def all_raw(r):
    return [read_raw(q, r["checkpoint"]) for q in r["requests"]]


def reuse():
    """Bind verified old artifacts without changing/copying their geometry."""
    check_code()
    r = registration()
    reused = []
    for q in r["requests"]:
        if q["action"] != "CACHE":
            continue
        for stage in ("raw", "post"):
            if C.sha(q[f"cache_{stage}_receipt"]) != q[f"cache_{stage}_receipt_sha256"]:
                raise ValueError("Historical receipt hash mismatch")
        prior = C.read(q["cache_raw_receipt"])
        arrays, facts = old().verify_output(q, prior, r["checkpoint"]["sha256"])
        new_source = source(q, arrays["C"])
        original_source = source(prior, arrays["C"])
        # Original winding/connectivity identities are part of POST cache validity.
        for key in ("C", "C_original", "C_source_corner_permutations", "C_source_vertex_ids", "C_original_source_vertex_ids"):
            if (key in new_source) != (key in original_source) or (key in new_source and not C.same(new_source[key], original_source[key])):
                raise ValueError("Source mapping changed; cannot reuse historical POST: " + key)
        rawrec = dict(prior)
        rawrec.update(q)
        rawrec.update(status="COMPLETE", label=LABEL, new_rollouts=0, new_sampling_F=0,
                      generation_status="VERIFIED_HISTORICAL_RAW_REUSE", contracts=facts,
                      historical_receipt=q["cache_raw_receipt"], historical_receipt_sha256=q["cache_raw_receipt_sha256"])
        C.write(record_path(q, "raw"), rawrec)
        historical = C.read(q["cache_post_receipt"])
        if historical["raw_sha256"] != prior["raw_sha256"] or historical["source_sha256"] != prior["condition_source_sha256"]:
            raise ValueError("Historical POST provenance mismatch")
        for path, digest in historical["files"].items():
            if C.sha(path) != digest:
                raise ValueError("Historical POST artifact changed")
        proposal = C.read(historical["proposal_receipt"])
        if proposal["checkpoint_sha256"] != r["d0_checkpoint"]["sha256"] or proposal["model_forwards"] != 1:
            raise ValueError("Historical POST is not the registered single official D0")
        postrec = dict(historical)
        postrec.update(q)
        postrec.update(status="COMPLETE", label=LABEL, source_sha256=q["condition_source_sha256"],
                       new_D0_F=0, GEO_calls=0, post_status="VERIFIED_HISTORICAL_POST_REUSE",
                       historical_receipt=q["cache_post_receipt"], historical_receipt_sha256=q["cache_post_receipt_sha256"])
        C.write(record_path(q, "post"), postrec)
        read_raw(q, r["checkpoint"])
        read_post(q, rawrec)
        reused.append(q["request_id"])
    value = dict(status="COMPLETE", reused=len(reused), request_ids=reused, model_forwards=0, geometry_calls=0)
    C.write(C.OUT / "execution/cache_reuse.json", value)
    return value


def code_identity():
    paths = [Path(__file__), C.CODE / "common.py", C.CODE / "prepare.py",
             C.ROOT / "r2/pilots/user_condition_finetune_v1/evaluation.py",
             C.ROOT / "r2/pilots/user_data_scaling_32_67_v1/training.py",
             C.ROOT / "r2/pilots/user_data_scaling_32_67_v1/training_data.py"]
    paths += list((C.ROOT / "r2/src/meshflow_control").rglob("*.py"))
    return {str(path): C.sha(path) for path in sorted(set(paths))}


def freeze_code():
    registration()
    value = dict(registration_sha256=C.sha(C.OUT / "registration.json"), files=code_identity())
    C.write(C.OUT / "execution/gpu_code_freeze.json", value)
    return dict(status="FROZEN", files=len(value["files"]))


def check_code():
    frozen = C.read(C.OUT / "execution/gpu_code_freeze.json")
    if frozen["registration_sha256"] != C.sha(C.OUT / "registration.json") or frozen["files"] != code_identity():
        raise ValueError("GPU implementation/registration changed after freeze")


def partition(r, worker, workers):
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError("Invalid worker partition")
    return [q for i, q in enumerate(q for q in r["requests"] if q["action"] == "NEW") if i % workers == worker]


@contextmanager
def owner(stage, worker, workers):
    import fcntl
    path = C.checked(C.OUT / "execution" / f"{stage}_worker_{worker}_of_{workers}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This worker partition already has an owner")
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def worker_path(stage, worker, workers, name):
    return C.OUT / "execution" / f"{stage}_worker_{worker}_of_{workers}_{name}.json"


def snap(tag, worker, workers, need_mib=0):
    value = C.resources()
    C.write(C.OUT / "resources" / f"{tag}_worker_{worker}_of_{workers}_{time.time_ns()}.json", value)
    if value["gpu"]["returncode"] != 0:
        raise RuntimeError("Resource observation failed")
    data = value["gpu"]["stdout"].strip().splitlines()[0].split(",")
    total, free = float(data[1]), float(data[3])
    if free - max(2048., .1 * total) < need_mib:
        raise RuntimeError("Insufficient VRAM headroom; no model call attempted")
    return value


def generate(worker, workers):
    check_code()
    r = registration()
    requests = partition(r, worker, workers)
    pending = []
    for q in requests:
        if record_path(q, "raw").exists():
            read_raw(q, r["checkpoint"])
        elif record_path(q, "raw").parent.exists():
            raise RuntimeError("Incomplete RAW cell; no silent replay: " + q["request_id"])
        else:
            old().read_input(q)
            pending.append(q)
    if not pending:
        return dict(status="COMPLETE", worker=worker, workers=workers, new_rollouts=0, assigned=len(requests))
    C.package()
    from meshflow_control import runtime
    from meshflow_control.sampling import sample
    import torch
    runtime.configure_stable_runtime()
    torch.set_num_threads(2)
    snap("generator_before_load", worker, workers, 2600.)
    loader = C.module("_seat_diagnosis_generator_loader", C.ROOT / "r2/pilots/user_data_scaling_32_67_v1/training.py")
    model, identity = loader.load_generator(C.resolve(r["checkpoint"]["path"]), device="cuda", expected_sha256=r["checkpoint"]["sha256"])
    if identity["arm"] != "USER_EXPANDED_67" or identity["global_step"] != 6000:
        raise ValueError("Wrong generator endpoint")
    C.write(worker_path("generator", worker, workers, "identity"), identity)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    snap("generator_resident", worker, workers)
    count = 0
    tick = time.perf_counter()

    class Budget(old().CellBudget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.velocities = []

        def call(self, kind, fn, context=None):
            value = super().call(kind, fn, context)
            if kind == "sampling_forward":
                self.velocities.append(value.detach().float().cpu().numpy()[0])
            return value

    try:
        for q in pending:
            arrays = old().read_input(q)
            source(q, arrays["C"])
            dest = C.checked(record_path(q, "raw").parent)
            dest.mkdir(parents=True, exist_ok=False)
            context = dict(label=LABEL, request_id=q["request_id"], input_sha256=q["input_sha256"], checkpoint_sha256=r["checkpoint"]["sha256"])
            budget = Budget(dest / "calls.jsonl", context)
            _, trajectory, audit = sample(model, arrays["C"], int(q["N"]), seed=q["noise_seed"], z=arrays["z"], budget=budget, out=dest, context=context)
            velocities = np.stack(budget.velocities)
            C.frozen_npz(dest / "velocities.npz", dict(velocity=velocities, source_dtype=np.array("torch.bfloat16")))
            rec = dict(q, **context, status="COMPLETE", new_rollouts=1, generation_status="NEW_EULER50",
                       velocities_path=str(dest / "velocities.npz"), velocities_sha256=C.sha(dest / "velocities.npz"), worker=worker, workers=workers)
            for field, name in (("raw", "raw.npz"), ("ply", "raw.ply"), ("result", "run.json")):
                rec[field + "_path"] = str(dest / name)
                rec[field + "_sha256"] = C.sha(dest / name)
            _, rec["contracts"] = old().verify_output(q, rec, r["checkpoint"]["sha256"])
            C.write(dest / "receipt.json", rec)
            read_raw(q, r["checkpoint"])
            count += 1
            if count == 1 or count % 8 == 0:
                snap("generator_after_rollout_" + str(count), worker, workers)
            C.write(worker_path("generator", worker, workers, "progress"), dict(stage="RAW", completed=sum(record_path(v, "raw").exists() for v in requests), registered=len(requests), seconds=time.perf_counter()-tick, last_sampling=audit), replace=True)
            print(json.dumps(dict(stage="RAW", worker=worker, completed=count, request_id=q["request_id"], seconds=audit["seconds"])), flush=True)
        state_hash = loader.state_hash(model.state_dict())
        if state_hash != identity["generator_state_sha256"]:
            raise ValueError("Frozen generator changed")
        snap("generator_finish", worker, workers)
        result = dict(status="COMPLETE", worker=worker, workers=workers, assigned=len(requests), new_rollouts=count, new_sampling_F=50*count,
                      seconds=time.perf_counter()-tick, generator_state_sha256=state_hash, training=0, backward=0)
        C.write(worker_path("generator", worker, workers, "complete"), result)
        return result
    except BaseException as exc:
        event("GENERATOR_STOPPED", worker, workers, error=repr(exc), traceback=traceback.format_exc(), returned_rollouts=count)
        raise
    finally:
        del model
        torch.cuda.empty_cache()


def postprocess(worker, workers):
    check_code()
    r = registration()
    raw_by_id = {row["request_id"]: row for row in all_raw(r)}
    # This barrier checks ALL logical RAW, including cache, before any D0 load.
    requests = partition(r, worker, workers)
    C.package()
    from meshflow_control import runtime, denoise, denoise_model, geo_c, geo_io, source_winding
    import torch
    runtime.configure_stable_runtime()
    torch.set_num_threads(2)
    model = None
    calls = 0
    tick = time.perf_counter()
    rows = []
    try:
        for q in requests:
            rawrec = raw_by_id[q["request_id"]]
            dest = C.checked(record_path(q, "post").parent)
            if record_path(q, "post").exists():
                rows.append(read_post(q, rawrec))
                continue
            raw = C.arrays(rawrec["raw_path"])["raw"].reshape(-1, 3, 3)
            k = q["K"]
            src = source(q, raw[:k])
            proposal_path, proposal_receipt = dest / "proposal.npz", dest / "proposal_receipt.json"
            if (dest / "geometry_started.json").exists():
                raise RuntimeError("Incomplete geometry; recorded engineering recovery required")
            if proposal_receipt.exists():
                saved = C.read(proposal_receipt)
                if saved["checkpoint_sha256"] != r["d0_checkpoint"]["sha256"]:
                    raise ValueError("Cached D0 identity changed")
                proposal = denoise.cached_proposal(proposal_path, proposal_receipt, raw, rawrec["raw_sha256"])
                newcall = 0
            elif dest.exists():
                raise RuntimeError("Unresolved D0 attempt; no silent retry")
            else:
                if model is None:
                    snap("D0_before_load", worker, workers, 3000.)
                    binding = r["d0_checkpoint"]
                    model, identity = denoise_model.load_denoiser(C.resolve(binding["path"]), device="cuda", expected_sha256=binding["sha256"])
                    C.write(worker_path("D0", worker, workers, "identity"), identity)
                    snap("D0_resident", worker, workers)
                dest.mkdir(parents=True, exist_ok=False)
                C.write(dest / "D0_started.json", dict(status="RESERVED", raw_sha256=rawrec["raw_sha256"], checkpoint_sha256=r["d0_checkpoint"]["sha256"]))
                event("D0_FORWARD_RESERVED", worker, workers, request_id=q["request_id"])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                proposal = denoise_model.predict_proposal(model, raw)
                torch.cuda.synchronize()
                calls += 1
                newcall = 1
                C.frozen_npz(proposal_path, dict(official_unconstrained_proposal=proposal))
                C.write(proposal_receipt, dict(status="PASS", raw_sha256=rawrec["raw_sha256"], checkpoint_sha256=r["d0_checkpoint"]["sha256"], output_sha256=C.sha(proposal_path), model_forwards=1,
                        seconds=time.perf_counter()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                event("D0_FORWARD_RETURNED", worker, workers, request_id=q["request_id"])
                if calls == 1:
                    snap("D0_first_actual_forward", worker, workers)
            denoise.validate_proposal(proposal, raw)
            C.write(dest / "geometry_started.json", dict(status="STARTED", new_D0_F=newcall))
            den = proposal.copy()
            den[:k] = raw[:k]
            C.frozen_npz(dest / "pre_geo.npz", dict(output=den, C=raw[:k], known_faces=np.asarray(k)))
            geometry = geo_c.process(den, k)
            canonical = geo_io._validate_result(den, k, geometry)
            C.frozen_npz(dest / "post_canonical.npz", canonical)
            result = source_winding.export_source_winding(canonical, src)
            post = result["oriented_arrays"]
            C.frozen_npz(dest / "post.npz", post)
            geo_io.write_indexed_ply(dest / "post.ply", post["vertices"], post["faces"])
            vertices, faces = geo_io.read_indexed_ply(dest / "post.ply")
            if not C.same(vertices, post["vertices"]) or not C.same(faces, post["faces"]):
                raise ValueError("POST PLY readback mismatch")
            C.write(dest / "geometry.json", {key: value for key, value in geometry.items() if not isinstance(value, np.ndarray)})
            C.write(dest / "source_restoration.json", result["restoration_summary"])
            C.write(dest / "orientation.json", {key: value for key, value in result["orientation_result"].items() if not isinstance(value, np.ndarray)})
            rec = {key: value for key, value in rawrec.items() if key not in ("files", "contracts")}
            rec.update(status="COMPLETE", source_sha256=q["condition_source_sha256"], proposal_path=str(proposal_path), proposal_receipt=str(proposal_receipt),
                       post_path=str(dest / "post.npz"), post_sha256=C.sha(dest / "post.npz"), canonical_path=str(dest / "post_canonical.npz"), pre_geo_path=str(dest / "pre_geo.npz"),
                       source_restoration=result["restoration_summary"], orientation=result["orientation_result"]["summary"], new_D0_F=newcall, GEO_calls=1,
                       files={str(path): C.sha(path) for path in dest.iterdir() if path.is_file()})
            C.write(record_path(q, "post"), rec)
            rows.append(read_post(q, rawrec))
            C.write(worker_path("D0", worker, workers, "progress"), dict(completed=len(rows), registered=len(requests), new_D0_F=calls, seconds=time.perf_counter()-tick), replace=True)
            print(json.dumps(dict(stage="POST", worker=worker, completed=len(rows), request_id=q["request_id"])), flush=True)
        snap("D0_finish", worker, workers)
        value = dict(status="COMPLETE", worker=worker, workers=workers, POST=len(rows), new_D0_F=calls, seconds=time.perf_counter()-tick)
        complete = worker_path("D0", worker, workers, "complete")
        if not complete.exists():
            C.write(complete, value)
        return value
    except BaseException as exc:
        event("POST_STOPPED", worker, workers, error=repr(exc), traceback=traceback.format_exc(), completed=len(rows), D0_returned=calls)
        raise
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()


def collect():
    check_code()
    r = registration()
    rows = all_raw(r)
    C.write(C.OUT / "evaluation/raw_index.json", dict(status="COMPLETE", rows=rows))
    post_complete = all(record_path(q, "post").exists() for q in r["requests"])
    if post_complete:
        post = [read_post(q, raw) for q, raw in zip(r["requests"], rows)]
        C.write(C.OUT / "evaluation/post_index.json", dict(status="COMPLETE", rows=post))
    return dict(status="COMPLETE", RAW=len(rows), POST=len(rows) if post_complete else "PENDING", model_forwards=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze-code", "reuse", "generate", "post", "collect", "check"), required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "freeze-code":
        value = freeze_code()
    elif args.stage == "reuse":
        value = reuse()
    elif args.stage == "collect":
        value = collect()
    elif args.stage == "check":
        check_code()
        r = registration()
        for q in r["requests"]:
            arrays = old().read_input(q)
            source(q, arrays["C"])
        value = dict(status="PASS", requests=len(r["requests"]), model_forwards=0)
    else:
        with owner(args.stage, args.worker, args.workers):
            value = generate(args.worker, args.workers) if args.stage == "generate" else postprocess(args.worker, args.workers)
    print(json.dumps(dict(stage=args.stage, result=value), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
