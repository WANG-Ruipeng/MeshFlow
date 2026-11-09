"""Bounded final evaluation: 84 new RAW, 84 D0 calls, then anonymous review.

This coordinator never imports torch. GPU computation belongs to explicit child
workers. Three sampling methods may overlap only after measured admission;
all final RAW precede one resident official D0 batch. Original BASE24 is cached.
"""
from __future__ import annotations

from . import common as W
from . import data as D
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

VERSION = "terminal_reward_nft_final_run_v1"
LABELS = ("NFT", "SELF_FM", "BASE")
COUNTS = {"BASE": 12, "NFT": 36, "SELF_FM": 36}
CONTROL = D.OUT / "evaluation/final_controller"
FINAL_D0 = D.OUT / "evaluation/denoise_all_worker"
MAX_SAMPLING_WORKERS = 3


def _event(event_name, **fields):
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / "events.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps(D.clean(dict(event=event_name, utc=W.utc(), **fields)), allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def source_verify():
    lock = D.OUT / "execution_source_lock.json"
    if not lock.exists():
        raise RuntimeError("Frozen execution source lock required")
    for path, expected in D.read(lock)["hashes"].items():
        if D.sha(path) != expected:
            raise RuntimeError("Execution source changed: " + path)
    final_lock = CONTROL / "contract.json"
    if final_lock.exists():
        for path, expected in D.read(final_lock)["code_sha256"].items():
            if D.sha(path) != expected:
                raise RuntimeError("Final-stage source changed: " + path)


def checkpoint_gate():
    """Accept canonical completed transactions and explicit recovery evidence."""
    accepted, missing = {}, []
    for arm in ("NFT", "SELF_FM"):
        checkpoint = D.OUT / f"checkpoints/{arm}/round15.pt"
        receipt_path = checkpoint.with_suffix(".json")
        if not checkpoint.exists() or not receipt_path.exists():
            missing.append(arm + ": final checkpoint/receipt missing")
            continue
        receipt = D.read(receipt_path)
        digest = D.sha(checkpoint)
        if (receipt.get("step"), receipt.get("round_index"), receipt.get("sha256")) != (256, 15, digest):
            raise RuntimeError("Final checkpoint hash/clock mismatch: " + arm)
        original = D.OUT / f"training/round_15/train_{arm}"
        candidates = [original / "completion.json", original / "completion_recovery.json", original / "complete_recovery.json",
                      original.with_name(original.name + "_recovery01") / "completion.json"]
        proof = None
        for path in candidates:
            if not path.exists():
                continue
            value = D.read(path)
            if value.get("status") != "COMPLETE":
                continue
            identity = value.get("identity", value)
            identity_arm = identity.get("arm", value.get("arm"))
            identity_round = identity.get("round_index", value.get("round_index"))
            if identity_arm != arm or identity_round != 15:
                raise RuntimeError("Completion proof identity differs: " + str(path))
            if path.name in ("completion_recovery.json", "complete_recovery.json"):
                if value.get("step") != 256 or value.get("checkpoint_sha256") != digest:
                    raise RuntimeError("Recovery proof must bind final checkpoint hash and step")
            elif identity.get("stop_step") != 256:
                raise RuntimeError("Phase completion does not certify stop_step256")
            metric_path = path.parent / "training_metrics.json"
            if not metric_path.exists() and path.name in ("completion_recovery.json", "complete_recovery.json"):
                metric_path = original / "training_metrics.json"
            metrics = D.read(metric_path)
            if (metrics.get("status"), metrics.get("arm"), metrics.get("round_index"),
                    metrics.get("completed_updates"), metrics.get("stop_step")) != ("COMPLETE", arm, 15, 16, 256):
                raise RuntimeError("Final training metrics do not certify completed round")
            expected_metrics = value.get("metrics_sha", value.get("metrics_sha256"))
            if expected_metrics is not None and D.sha(metric_path) != expected_metrics:
                raise RuntimeError("Recovery metrics hash differs")
            if value.get("original_failure_path"):
                expected_failure = value.get("original_failure_sha", value.get("original_failure_sha256"))
                if not expected_failure or D.sha(value["original_failure_path"]) != expected_failure:
                    raise RuntimeError("Recovery original failure record changed")
            proof = dict(path=str(path), sha256=D.sha(path), metrics_path=str(metric_path),
                         metrics_sha256=D.sha(metric_path), recovered=path != candidates[0])
            break
        if proof is None:
            missing.append(arm + ": final complete Phase/recovery proof missing")
        else:
            accepted[arm] = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
                                 receipt_path=str(receipt_path), receipt_sha256=D.sha(receipt_path), proof=proof)
    return dict(status="PASS" if not missing else "WAITING_TRAINING", accepted=accepted, missing=missing)


def require_training():
    gate = checkpoint_gate()
    if gate["status"] != "PASS":
        raise RuntimeError("Both arms must finish before any final GPU launch: " + "; ".join(gate["missing"]))
    return gate


def freeze_contract(gate):
    code = sorted((D.ROOT / "r2/pilots/terminal_reward_nft_v1").glob("*.py"))
    contract = dict(version=VERSION, checkpoint_gate=gate,
        manifest_sha256=D.sha(D.OUT / "inputs/manifest.json"),
        code_sha256={str(p): D.sha(p) for p in code},
        budgets=dict(new_RAW=84, new_RAW_by_label=COUNTS, sampling_forwards=4200,
                     single_D0_calls=84, original_BASE_RAW_POST_reuse=24,
                     logical_RAW=108, logical_POST=108, final_training_updates=0),
        dispatch=dict(max_sampling_processes=3, first_real_rollout_before_next_worker=True,
            incremental_slots="floor(max(0,free-headroom)/(1.2*observed_sampling_process_peak))",
            headroom="max(2048MiB,10%total)", utilization_wait_at=85,
            separate_D0_memory_estimate=True, D0_resident_instances=1,
            CPU_scoring_workers=2, no_resource_probe_jobs=True),
        finish="WAITING_ACTUAL_ANONYMOUS_REVIEW; no labels assigned by coordinator")
    D.write(CONTROL / "contract.json", contract)
    return contract


def gpu_values(snapshot):
    result = snapshot["gpu"]
    if result["returncode"] != 0:
        raise RuntimeError("nvidia-smi resource query failed")
    rows = list(csv.reader(io.StringIO(result["stdout"])))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("Expected one measured GPU")
    name, total, used, free, util = rows[0]
    return dict(name=name.strip(), total=float(total), used=float(used), free=float(free), util=float(util))


def mem_available_mib(snapshot):
    for line in (snapshot.get("meminfo") or "").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024
    raise RuntimeError("Available system RAM could not be measured")


def snapshot():
    value = W.resource_snapshot()
    value["available_CPU_cores"] = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    value["load_average"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    p = subprocess.run(["ps", "-eo", "pid,comm,pcpu,pmem", "--sort=-pcpu"],
                       capture_output=True, text=True, timeout=10)
    value["CPU_process_load"] = dict(returncode=p.returncode, stdout="\n".join(p.stdout.splitlines()[:17]))
    return value


def process_rss_mib(pid):
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return 0.0
    for line in path.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024
    return 0.0


def process_gpu_mib(snap, pid):
    values = []
    for row in csv.reader(io.StringIO(snap["processes"]["stdout"])):
        if len(row) >= 3 and row[0].strip() == str(pid):
            try:
                values.append(float(row[-1].strip()))
            except ValueError:
                pass  # WDDM can report N/A; retain the device-delta bound.
    return max(values, default=0.0)


def admission(snap, *, process_peak_mib, ram_peak_mib, active):
    g = gpu_values(snap)
    headroom = max(2048.0, 0.1 * g["total"])
    slots = (math.floor(max(0.0, g["free"] - headroom) / (1.2 * process_peak_mib))
             if process_peak_mib > 0 else int(g["free"] > headroom))
    ram_free = mem_available_mib(snap)
    ram_need = max(1024.0, 1.2 * ram_peak_mib) + 1024.0
    cores = int(snap["available_CPU_cores"])
    load = snap.get("load_average")
    reasons = []
    if slots < 1:
        reasons.append("VRAM_HEADROOM_OR_SLOT_LIMIT")
    if g["util"] >= 85:
        reasons.append("GPU_COMPUTE_AT_LEAST_85_PERCENT")
    if ram_free < ram_need:
        reasons.append("SYSTEM_RAM_HEADROOM")
    if cores < 2 * (active + 1) or (load is not None and load[0] >= max(1, 0.9 * cores)):
        reasons.append("CPU_THREADS_OR_LOAD")
    return dict(admitted=not reasons, reasons=reasons, incremental_slots=slots,
                headroom_MiB=headroom, process_peak_estimate_MiB=process_peak_mib,
                estimate_margin=1.2, available_RAM_MiB=ram_free, required_RAM_MiB=ram_need,
                active_owned_processes=active, gpu=g)


def prior_measurement(kind):
    """Use only existing real sampling or D0 evidence, separately by phase."""
    if kind == "sampling":
        phase = D.OUT / "training/round_00/sampling_worker/resources.json"
        receipts = list((D.OUT / "training/round_00/raw").glob("*/receipt.json"))
        peak_key = "sampling"
    elif kind == "D0":
        phase = D.OUT / "training/round_00/denoise_worker/resources.json"
        receipts = list((D.OUT / "training/round_00/proposal").glob("*/receipt.json"))
        peak_key = None
    else:
        raise ValueError(kind)
    peak = 0.0
    if phase.exists():
        evidence = D.read(phase)
        baseline = gpu_values(evidence["initial"])["used"]
        observations = [evidence["initial"], *evidence.get("samples", []), evidence["final"]]
        peak = max((max(0.0, gpu_values(s)["used"] - baseline) for s in observations if "gpu" in s), default=0.0)
    reserved = 0.0
    for path in receipts:
        r = D.read(path)
        r = r[peak_key] if peak_key else r
        reserved = max(reserved, r.get("cuda_peak_reserved_bytes", 0) / 2**20)
    # Device deltas can be confounded and WDDM can hide process bytes. Include
    # allocator-reserved memory plus 512 MiB context margin, never allocated-only.
    if reserved:
        peak = max(peak, reserved + 512.0)
    return dict(kind=kind, estimate_MiB=peak, allocator_reserved_MiB=reserved,
                process_exact=False, source_resources=str(phase), real_receipts=len(receipts),
                scope="existing real same-model phase; conservative device delta/reserved+context bound")


def launch(name, module, args, snap, decision):
    source_verify()
    root = D.OUT / "dispatch" / name
    if root.exists():
        raise RuntimeError("Unresolved/prior dispatch cannot be silently replayed: " + name)
    D.write(root / "admission.json", dict(resources=snap, decision=decision, command=[module, *args]))
    log = (root / "stdout.log").open("x", encoding="utf8")
    cmd = [sys.executable, "-u", "-B", "-m", module, *args]
    try:
        proc = subprocess.Popen(cmd, cwd=D.ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    except BaseException:
        log.close()
        raise
    job = dict(name=name, proc=proc, log=log, path=root, initial=snap,
               started=time.perf_counter(), gpu_peak=0.0, ram_peak=0.0, label=None)
    try:
        _event("LAUNCH", name=name, pid=proc.pid, command=cmd, decision=decision)
        print("LAUNCH", name, "PID", proc.pid, flush=True)
    except BaseException:
        # The owned process already started: logging errors must not terminate
        # its parent WSL invocation while it is still computing.
        while proc.poll() is None:
            time.sleep(1)
        finish(job)
        raise
    return job


def finish(job):
    code = job["proc"].poll()
    if code is None:
        return None
    if not job.get("closed"):
        job["log"].close()
        D.write(job["path"] / "exit.json", dict(returncode=code,
            seconds=time.perf_counter() - job["started"], resources=snapshot(),
            observed_GPU_bound_MiB=job["gpu_peak"], observed_process_RSS_MiB=job["ram_peak"]))
        job["closed"] = True
        _event("EXIT", name=job["name"], returncode=code)
        print("EXIT", job["name"], code, flush=True)
    return code


def validate_raw(label, *, require_phase=True):
    from .denoise_worker import prepare_jobs
    manifest = D.read(D.OUT / "inputs/manifest.json")
    path = D.OUT / "evaluation" / label / "raw_index.json"
    if not path.exists():
        return False
    if require_phase:
        completion = path.parent / "sampling_worker/completion.json"
        if not completion.exists() or D.read(completion).get("status") != "COMPLETE":
            raise RuntimeError("Final RAW index lacks completed sampling Phase: " + label)
    _, _, jobs, _ = prepare_jobs(manifest, evaluation=label)
    index = D.read(path)
    if len(jobs) != COUNTS[label] or index["actual_new_rollouts"] != COUNTS[label] or index["actual_sampling_forwards"] != 50 * COUNTS[label]:
        raise RuntimeError("Final RAW budget mismatch: " + label)
    return True


def sample_all():
    pending = [label for label in LABELS if not validate_raw(label)]
    evidence = prior_measurement("sampling")
    _event("SAMPLING_PRIOR_EVIDENCE", evidence=evidence)
    peak, ram_peak = evidence["estimate_MiB"], 0.0
    jobs, latest, failure = [], None, None
    tick = 0
    try:
        while pending or jobs:
            snap = snapshot()
            g = gpu_values(snap)
            for job in list(jobs):
                job["gpu_peak"] = max(job["gpu_peak"], process_gpu_mib(snap, job["proc"].pid),
                                      max(0.0, g["used"] - gpu_values(job["initial"])["used"]))
                job["ram_peak"] = max(job["ram_peak"], process_rss_mib(job["proc"].pid))
                peak, ram_peak = max(peak, job["gpu_peak"]), max(ram_peak, job["ram_peak"])
                if job["proc"].poll() is not None:
                    code = finish(job)
                    jobs.remove(job)
                    if code:
                        failure = "Sampling worker failed; no further dispatch: " + job["name"]
                    elif not validate_raw(job["label"]):
                        raise RuntimeError("Successful sampling process lacks complete RAW index: " + job["label"])
            ready = latest is None
            if latest is not None:
                raw = D.OUT / "evaluation" / latest["label"] / "raw"
                completed = list(raw.glob("*/receipt.json"))
                ready = bool(completed)
                if completed:
                    r = D.read(completed[0])["sampling"]
                    peak = max(peak, r["cuda_peak_reserved_bytes"] / 2**20 + 512.0)
            decision = admission(snap, process_peak_mib=peak, ram_peak_mib=ram_peak, active=len(jobs))
            if tick % 5 == 0:
                _event("RESOURCE_OBSERVATION", resources=snap, admission=decision,
                       first_real_rollout_observed=ready, active=[j["name"] for j in jobs])
            if failure:
                pending.clear()  # Already running authorized workers finish; nothing is killed.
            elif pending and len(jobs) < MAX_SAMPLING_WORKERS and ready and decision["admitted"]:
                label = pending.pop(0)
                latest = launch("final_sample_" + label, "r2.pilots.terminal_reward_nft_v1.sample_worker",
                                ["--evaluation", label], snap, decision)
                latest["label"] = label
                jobs.append(latest)
            if jobs or pending:
                time.sleep(1)
            tick += 1
    finally:
        # If admission, validation or logging fails, let every already launched
        # authorized child finish before this owning WSL process exits.
        while any(job["proc"].poll() is None for job in jobs):
            time.sleep(1)
        for job in jobs:
            finish(job)
    if failure:
        raise RuntimeError(failure)
    if not all(validate_raw(label) for label in LABELS):
        raise RuntimeError("All 84 final RAW must complete before D0")


def all_d0_jobs():
    from .denoise_worker import prepare_jobs
    manifest = D.read(D.OUT / "inputs/manifest.json")
    jobs = []
    for label in LABELS:
        if not validate_raw(label):
            raise RuntimeError("All final RAW required before resident D0 batch")
        jobs.extend(prepare_jobs(manifest, evaluation=label)[2])
    if len(jobs) != 84:
        raise RuntimeError("Exactly 84 new final D0 jobs required")
    return jobs


def validate_d0():
    path = D.OUT / "evaluation/final_proposal_index.json"
    if not path.exists():
        return False
    value = D.read(path)
    phase = FINAL_D0 / "completion.json"
    if value.get("status") != "COMPLETE" or not phase.exists() or D.read(phase).get("status") != "COMPLETE":
        raise RuntimeError("Incomplete final D0 transaction")
    if value["jobs"] != 84 or value["new_D0_forwards"] != 84 or value["resident_models_loaded"] != 1:
        raise RuntimeError("Final D0 budgets differ")
    from .denoise_worker import OFFICIAL_D0_SHA256
    expected = {(j["label"], j["request_id"]): j for j in all_d0_jobs()}
    receipts = value["receipts"]
    if len(receipts) != 84 or {(r["label"], r["request_id"]) for r in receipts} != set(expected):
        raise RuntimeError("Final D0 receipts differ from the 84 registered identities")
    for r in receipts:
        job = expected[r["label"], r["request_id"]]
        if (r.get("status") != "PROPOSAL_COMPLETE" or r.get("model_forwards") != 1
                or D.resolve(r["proposal_path"]) != D.resolve(job["proposal_dir"]) / "proposal.npz"
                or D.resolve(r["proposal_receipt"]) != D.resolve(job["proposal_dir"]) / "receipt.json"
                or D.resolve(r["raw_path"]) != D.resolve(job["raw_path"])):
            raise RuntimeError("Final D0 proposal ownership differs")
        proof = D.read(r["proposal_receipt"])
        if (proof.get("status") != "PASS" or proof["model_forwards"] != 1
                or proof["checkpoint_sha256"] != OFFICIAL_D0_SHA256
                or proof["request_id"] != r["request_id"] or proof["raw_sha256"] != job["raw_sha256"]):
            raise RuntimeError("Final proposal is not a single official D0 for this RAW")
        if D.sha(r["proposal_path"]) != r["proposal_sha256"] or proof["output_sha256"] != r["proposal_sha256"]:
            raise RuntimeError("Final proposal hash changed")
    return True


def denoise_worker():
    require_training()
    source_verify()
    jobs = all_d0_jobs()
    from .denoise_worker import retained_d0_binding
    binding = retained_d0_binding()
    if any(D.resolve(j["proposal_dir"]).exists() for j in jobs):
        raise RuntimeError("Prior proposal activity requires explicit accounting recovery, not another D0 batch")
    with W.Phase(FINAL_D0, dict(stage="FINAL_RESIDENT_D0", jobs=84, checkpoint=binding)) as phase:
        D.write(FINAL_D0 / "resources_initial.json", phase.initial)
        from .evaluation import postprocess_batch
        result = postprocess_batch(jobs, binding["path"], FINAL_D0 / "batch",
            resource_snapshot_path=FINAL_D0 / "resources_initial.json", proposal_only=True)
        if (result["jobs"], result["new_D0_forwards"], result["resident_models_loaded"]) != (84, 84, 1):
            raise RuntimeError("Final batch must perform exactly 84 D0 calls using one model")
        D.write(D.OUT / "evaluation/final_proposal_index.json", result)
        for label in LABELS:
            receipts = [r for r in result["receipts"] if r["label"] == label]
            if len(receipts) != COUNTS[label]:
                raise RuntimeError("D0 per-label count differs")
            D.write(D.OUT / "evaluation" / label / "proposal_index.json",
                    dict(status="COMPLETE", jobs=len(receipts), receipts=receipts, label=label,
                         new_D0_forwards=len(receipts), proposal_only=True,
                         resident_batch_receipt=str(D.OUT / "evaluation/final_proposal_index.json")))


def wait_single(name, module, args, *, gpu_kind=None):
    evidence = prior_measurement(gpu_kind) if gpu_kind else None
    tick = 0
    while True:
        snap = snapshot()
        decision = (admission(snap, process_peak_mib=evidence["estimate_MiB"], ram_peak_mib=0, active=0)
                    if gpu_kind else dict(admitted=(mem_available_mib(snap) >= 2048
                        and snap["available_CPU_cores"] >= 2
                        and (snap.get("load_average") is None
                             or snap["load_average"][0] < .9 * snap["available_CPU_cores"])),
                        reason="two CPU workers with available RAM/cores/load; model-free stage"))
        if decision["admitted"]:
            break
        if tick % 5 == 0:
            _event("WAIT_RESOURCE", name=name, resources=snap, decision=decision, prior=evidence)
        tick += 1
        time.sleep(1)
    job = launch(name, module, args, snap, decision)
    tick = 0
    try:
        while job["proc"].poll() is None:
            snap = snapshot()
            g = gpu_values(snap)
            job["gpu_peak"] = max(job["gpu_peak"], process_gpu_mib(snap, job["proc"].pid),
                                  max(0.0, g["used"] - gpu_values(job["initial"])["used"]) if gpu_kind else 0.0)
            job["ram_peak"] = max(job["ram_peak"], process_rss_mib(job["proc"].pid))
            if tick % 5 == 0:
                _event("RESOURCE_OBSERVATION", name=name, resources=snap, GPU_kind=gpu_kind,
                       observed_GPU_bound_MiB=job["gpu_peak"], observed_process_RSS_MiB=job["ram_peak"])
            tick += 1
            time.sleep(1)
    finally:
        while job["proc"].poll() is None:
            time.sleep(1)
        finish(job)
    if job["proc"].returncode:
        raise RuntimeError("Worker failed; no automatic retry: " + name)


def run():
    gate = require_training()
    source_verify()
    freeze_contract(gate)
    sample_all()
    if not validate_d0():
        wait_single("final_D0_all84", "r2.pilots.terminal_reward_nft_v1.final_run",
                    ["--stage", "denoise-worker"], gpu_kind="D0")
    if not validate_d0():
        raise RuntimeError("Final D0 did not complete")
    cpu_done = D.OUT / "evaluation/cpu/post_score_render_complete.json"
    if not cpu_done.exists():
        wait_single("final_CPU_all", "r2.pilots.terminal_reward_nft_v1.final_cpu",
                    ["--stage", "all", "--workers", "2"])
    cpu = D.read(cpu_done)
    if cpu.get("status") != "COMPLETE" or cpu.get("requests") != 108 or cpu.get("required_new_POST") != 84:
        raise RuntimeError("Final CPU count/completion mismatch")
    for path, expected in cpu["files"].items():
        if D.sha(path) != expected:
            raise RuntimeError("Final CPU artifact changed: " + path)
    gallery = D.OUT / "evaluation/blind_gallery/manifest.json"
    if not gallery.exists():
        raise RuntimeError("Anonymous gallery not completed")
    result = dict(status="WAITING_ACTUAL_ANONYMOUS_REVIEW", version=VERSION,
        new_RAW=84, new_sampling_forwards=4200, new_D0_forwards=84,
        BASE_cached_RAW_POST=24, logical_RAW=108, logical_POST=108,
        CPU_completion_path=str(cpu_done), CPU_completion_sha256=D.sha(cpu_done),
        gallery_manifest=str(gallery), gallery_sha256=D.sha(gallery),
        review_labels_created=0, review_scope="36 requests x anonymous local A/B/C aliases; actual review required",
        private_mapping_must_not_be_sent_to_reviewer=True,
        scientific_training_updates_in_final_stage=0)
    D.write(D.OUT / "evaluation/review_gate.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "status", "denoise-worker"), default="all")
    args = parser.parse_args()
    if args.stage == "status":
        print(json.dumps(checkpoint_gate(), sort_keys=True))
        return
    if args.stage == "denoise-worker":
        denoise_worker()
        return
    # One coordinator owns admission; this does not lock or stop user processes.
    import fcntl
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / "owner.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except BaseException as exc:
            _event("FINAL_STAGE_STOPPED", error=repr(exc), automatic_retry=False)
            raise
        finally:
            fcntl.flock(owner, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
