"""Read-only scientific accounting; writes only final delivery artifacts.

No torch/numpy, checkpoint deserialization, model, OT, geometry or scoring calls.
Run only after workers have terminated. Completed ledgers are counted once;
aggregate/cached references are reconciliation evidence, never additional work.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
D_LABELS = ("G0", "R_B_raw250", "R_J_raw250", "R_B_EMA250", "R_J_EMA250",
            "R_B_EMA500", "R_J_EMA500")
EXPECTED = dict(model_forward_attempts=29098, model_forward_returns=29098,
                full_rollout_attempts=229, complete_rollouts=229,
                suffix_attempts=78, suffix_returns=78,
                training_backward_attempts=8000, training_backward_returns=8000,
                velocity_leaf_VJP_attempts=1200, velocity_leaf_VJP_returns=1200,
                parameter_VJP_attempts=24, parameter_VJP_returns=24,
                OT_attempts=8020, OT_returns=8020,
                optimizer_step_attempts=1000, optimizer_updates=1000,
                EMA_update_attempts=1000, EMA_updates=1000, postprocessing_repairs=0)
ACTIVE = {"PREPARING", "RUNNING", "TRAINING", "LOADING", "SCORING", "STARTED"}
BLOCKED = {"BLOCKED", "STOPPED", "REPLAY_MISMATCH", "FAILED", "FAIL"}
SAMPLE_KEYS = ("model_forward_attempts", "model_forward_returns", "full_rollout_attempts",
               "complete_rollouts", "suffix_attempts", "suffix_returns", "postprocessing_repairs")
SAMPLING_TIMES = ("seconds", "model_forward_host_wall_seconds", "model_forward_cuda_event_seconds",
                  "explicit_cuda_synchronize_seconds", "state_CPU_copy_seconds")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def _resolve(path, out):
    p = Path(path)
    if p.is_absolute():
        return p
    a = out / p
    return a if a.exists() else ROOT / p


def _git(*args, cwd=ROOT, allowed=(0,)):
    p = subprocess.run(["git", "--no-pager", *args], cwd=cwd, capture_output=True)
    if p.returncode not in allowed:
        raise RuntimeError("git read/delivery operation failed: " + p.stderr.decode("utf-8", errors="replace"))
    return p.stdout


class Ledger:
    def __init__(self, out):
        self.out = Path(out)
        self.execution = []
        self.checks = []
        self.missing = []
        self.records = {}
        self.timings = {}

    def check(self, name, condition, actual=None, expected=None):
        self.checks.append(dict(name=name, passed=bool(condition), actual=actual, expected=expected))

    def read(self, relative, required=True):
        path = self.out / relative
        if not path.is_file():
            if required:
                self.missing.append(relative)
            return None
        d = _read(path)
        if d.get("status") in ACTIVE:
            raise RuntimeError(f"Refusing to finalize an active worker: {relative}: {d['status']}")
        self.records[relative] = dict(_record(path), status=d.get("status"))
        return d

    def add(self, key, record, d, mapping):
        counts = d.get("counts", {})
        values = {name: 0 for name in EXPECTED}
        for destination, origin in mapping.items():
            value = counts.get(origin, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Invalid counter {key}/{origin}: {value!r}")
            values[destination] = value
        self.execution.append(dict(unit=key, source=record, status=d.get("status"), **values))
        for attempt, returned in (("model_forward_attempts", "model_forward_returns"),
                                  ("full_rollout_attempts", "complete_rollouts"),
                                  ("suffix_attempts", "suffix_returns"),
                                  ("training_backward_attempts", "training_backward_returns"),
                                  ("velocity_leaf_VJP_attempts", "velocity_leaf_VJP_returns"),
                                  ("parameter_VJP_attempts", "parameter_VJP_returns"),
                                  ("OT_attempts", "OT_returns"),
                                  ("optimizer_step_attempts", "optimizer_updates"),
                                  ("EMA_update_attempts", "EMA_updates")):
            self.check(key + "/" + returned + "_not_above_attempts", values[returned] <= values[attempt],
                       [values[attempt], values[returned]])
        return values

    def sample(self, key, relative, d, expected_f, expected_full, expected_suffix, expected_rows):
        values = self.add(key, relative, d, {k: k for k in SAMPLE_KEYS})
        self.check(key + "/sampler_no_backward_update_or_geometry", all(d.get("counts", {}).get(k, 0) == 0
                   for k in ("model_backward_calls", "optimizer_updates", "geometry_gradient_calls")))
        rows = d.get("rows", []) + d.get("same_model_replays", [])
        keys = [(x.get("model"), x.get("source_model"), x.get("continuation_model"), x.get("parent"),
                 x.get("patch"), x.get("seed"), x.get("precision"), x.get("nfe"), x.get("start_step", 0)) for x in rows]
        self.check(key + "/unique_output_requests", len(keys) == len(set(keys)), len(keys), len(set(keys)))
        if d.get("status") == "PASS":
            self.check(key + "/registered_count", (values["model_forward_returns"], values["complete_rollouts"],
                       values["suffix_returns"], len(rows)) == (expected_f, expected_full, expected_suffix, expected_rows),
                       [values["model_forward_returns"], values["complete_rollouts"], values["suffix_returns"], len(rows)],
                       [expected_f, expected_full, expected_suffix, expected_rows])
            self.check(key + "/frozen_state", bool(d.get("state_before")) and d.get("state_before") == d.get("state_after"))
        for i, row in enumerate(rows):
            a = row.get("audit", {})
            self.check(f"{key}/row{i}/C_all_saved_states_recorded", a.get("known_all_states_bitwise") is True)
            self.check(f"{key}/row{i}/no_repair", a.get("postprocessing") is None and not a.get("geometry_gradient_calls", 0))
            self.check(f"{key}/row{i}/no_model_update", not a.get("model_backward_calls", 0) and not a.get("optimizer_updates", 0))
            if row.get("same_model_replay"):
                self.check(f"{key}/row{i}/same_suffix_exact", row.get("same_model_full_suffix_bitwise") is True)
        times = {k: sum(float(row.get("audit", {}).get(k, 0.)) for row in rows) for k in SAMPLING_TIMES}
        times["archive_write_seconds"] = sum(float(row.get("archive_write_seconds", 0.)) for row in rows)
        times.update(worker_wall_seconds=d.get("wall_seconds", d.get("timing", {}).get("wall_seconds")),
                     load_seconds=d.get("load_seconds", d.get("timing", {}).get("load_seconds")),
                     sample_records=len(rows))
        self.timings[key] = times
        return values


def collect(out, allow_incomplete=False):
    """Read final metadata only. No artifacts are written by this helper."""
    a = Ledger(out)
    protocol = a.read("registration/protocol.json")
    preflight = a.read("registration/preflight.json")
    if protocol is None or preflight is None:
        raise RuntimeError("Missing immutable registration/preflight")
    a.check("preflight_pass", preflight.get("status") == "PASS")
    identities = {x["label"]: x for x in protocol["checkpoint_identities"]}
    checkpoints = {}
    for label in ("B", "J"):
        rel = f"A/{label}/results.json"
        d = a.read(rel)
        if d:
            a.sample("A_" + label, rel, d, 6850 if label == "B" else 6800,
                     85 if label == "B" else 84, 0, 84)
            a.check("A_" + label + "/registered_model", d.get("state_before") == identities[label]["state_sha256"])
            if label == "B":
                proof = d.get("production_replay", {})
                a.check("production_51_state_replay", proof.get("status") == "PASS" and proof.get("all_51_states_bitwise") is True)
                a.timings["A_B_production_replay_subset"] = dict(sampling=proof.get("sampling", {}),
                    seconds=proof.get("seconds"), archive_write_seconds=proof.get("archive", {}).get("archive_write_seconds"),
                    note="Counts already included in A_B; production sampler did not separately measure CUDA events/host/sync.")
        for mode, filename in (("gate", "gate_results.json"), ("cross", "results.json")):
            rel = f"B/{label}/{filename}"
            d = a.read(rel)
            if d:
                a.sample(f"B_{label}_{mode}", rel, d, 150 if mode == "gate" else 1800, 0,
                         3 if mode == "gate" else 36, 3 if mode == "gate" else 36)
                a.check(f"B_{label}_{mode}/registered_model", d.get("state_before") == identities[label]["state_sha256"])

    bank = a.read("stage_c/teacher_bank.json")
    if bank:
        a.add("C_teacher_bank_once", "stage_c/teacher_bank.json", bank, dict(OT_attempts="OT_calls", OT_returns="OT_returns"))
        a.check("C_private_bank", bank.get("formal_training_stream_accessed") is False)
        a.timings["C_teacher_bank_once"] = dict(CPU_OT_and_bank_seconds=bank.get("seconds"))
    for label in ("B", "J"):
        combined = {}
        for suffix in ("bf16", "fp32", "parameter_probe"):
            rel = f"stage_c/teacher/{label}_{suffix}.json"
            d = a.read(rel)
            if not d:
                continue
            param = suffix == "parameter_probe"
            prefix = "parameter" if param else "teacher"
            mapping = dict(model_forward_attempts=prefix + "_forward_attempts", model_forward_returns=prefix + "_forward_returns")
            mapping.update(dict(parameter_VJP_attempts="parameter_autograd_attempts", parameter_VJP_returns="parameter_autograd_returns")
                           if param else dict(velocity_leaf_VJP_attempts="velocity_leaf_autograd_attempts", velocity_leaf_VJP_returns="velocity_leaf_autograd_returns"))
            values = a.add(f"C_{label}_{suffix}", rel, d, mapping)
            if d.get("status") == "PASS":
                a.check(f"C_{label}_{suffix}/count", values["model_forward_returns"] == (4 if param else 100)
                        and values["parameter_VJP_returns" if param else "velocity_leaf_VJP_returns"] == (12 if param else 300))
            for k, v in d.get("counts", {}).items():
                combined[k] = combined.get(k, 0) + v
            a.timings[f"C_{label}_{suffix}"] = dict(d.get("timing", {}))
            if param:
                a.timings[f"C_{label}_{suffix}"]["scope"] = "Combined network/loss/parameter-VJP; finer separation NOT_MEASURED."
                a.check(f"C_{label}/probe_grad_cleanup", d.get("restored_frozen_and_grad_cleared") is True)
        process = a.read(f"stage_c/{label}_process.json")
        if process:
            a.check("C_" + label + "/aggregate_not_recounted", combined == process.get("counts"), combined, process.get("counts"))
            a.check("C_" + label + "/frozen", process.get("frozen_tensor_identity") is True and
                    process.get("final_state_sha256") == identities[label]["state_sha256"])
            a.timings["C_" + label + "_process_overlap"] = dict(worker_wall_seconds=process.get("wall_seconds"),
                    note="Encloses teacher/probe sub-timings; not additive to them.")

    replays = {}
    for arm, label in (("R_B", "B"), ("R_J", "J")):
        rel = f"stage_d/{arm}/replay.json"
        d = a.read(rel)
        if not d:
            continue
        replays[arm] = d
        mapping = dict(model_forward_attempts="forward_attempts", model_forward_returns="forward_returns",
                       training_backward_attempts="backward_attempts", training_backward_returns="backward_returns",
                       optimizer_step_attempts="optimizer_step_attempts", optimizer_updates="optimizer_updates",
                       EMA_update_attempts="EMA_update_attempts", EMA_updates="EMA_updates", OT_attempts="OT_calls", OT_returns="OT_returns")
        v = a.add("D_train_" + arm, rel, d, mapping)
        a.timings["D_train_" + arm] = d.get("timing", {})
        exact = (d.get("status") == "PASS_EXACT_REPLAY" and d.get("completed_updates") == 500 and
                 d.get("raw500_bitwise_reference_equal") is True and d.get("EMA_evaluation_allowed") is True and
                 d.get("raw500_state_sha256") == identities[label]["state_sha256"])
        if d.get("status") == "PASS_EXACT_REPLAY":
            a.check(arm + "/exact_replay_not_merely_completed", exact)
            a.check(arm + "/all_500_updates_and_microbatches", all(v[k] == n for k, n in
                (("model_forward_attempts", 4000), ("model_forward_returns", 4000), ("training_backward_attempts", 4000),
                 ("training_backward_returns", 4000), ("OT_attempts", 4000), ("OT_returns", 4000),
                 ("optimizer_step_attempts", 500), ("optimizer_updates", 500), ("EMA_update_attempts", 500), ("EMA_updates", 500))))
            a.check(arm + "/common_G0", d.get("initial_identity", {}).get("state_sha256") == identities["G0"]["state_sha256"])
            a.check(arm + "/no_partial_update", d.get("batch_in_progress") in (False, None) and d.get("optimizer_step_in_progress") is False)
        else:
            a.check(arm + "/nonexact_cannot_allow_EMA", d.get("EMA_evaluation_allowed") is not True)
        checkpoints[arm] = dict(status=d.get("status"), exact_replay=exact,
            raw500_state_sha256=d.get("raw500_state_sha256"), reference=identities[label]["state_sha256"],
            first_observed_training_scalar_mismatch=d.get("first_observed_training_scalar_mismatch"),
            completed_updates=d.get("completed_updates"), checkpoints=d.get("checkpoints", {}))
    if len(replays) == 2 and all(x.get("status") == "PASS_EXACT_REPLAY" for x in replays.values()):
        a.check("D_identical_restored_stream", bool(replays["R_B"].get("initial_stream_sha256")) and
                replays["R_B"].get("initial_stream_sha256") == replays["R_J"].get("initial_stream_sha256"))
        a.check("D_identical_all_500_stream_audits", len(replays["R_B"].get("stream_audit", [])) == 500 and
                replays["R_B"].get("stream_audit") == replays["R_J"].get("stream_audit"))

    for label in D_LABELS:
        rel = f"D_evaluation/{label}/results.json"
        d = a.read(rel)
        if not d:
            continue
        full = 20 if label.endswith("500") else 4
        vals = a.sample("D_sample_" + label, rel, d, 50 * full, full, 0, full)
        arm = None if label == "G0" else label[:3]
        if arm:
            a.check(label + "/evaluation_requires_exact_replay", checkpoints.get(arm, {}).get("exact_replay") is True)
        trel = f"D_evaluation/{label}/stage_d/teacher/{label}_bf16.json"
        teacher = a.read(trel)
        if teacher:
            tv = a.add("D_teacher_" + label, trel, teacher, dict(model_forward_attempts="teacher_forward_attempts", model_forward_returns="teacher_forward_returns"))
            a.check(label + "/teacher_parent_counts_are_same_record", teacher.get("counts") == d.get("teacher_counts"))
            a.check(label + "/teacher_has_no_VJP", not teacher.get("counts", {}).get("velocity_leaf_autograd_attempts", 0)
                    and not teacher.get("counts", {}).get("parameter_autograd_attempts", 0))
            if teacher.get("status") == "PASS":
                a.check(label + "/teacher20", tv["model_forward_attempts"] == 20 and tv["model_forward_returns"] == 20)
            for k in ("model_forward_attempts", "model_forward_returns"):
                a.check(label + "/total_is_disjoint_sum_" + k, d.get("total_counts", {}).get(k) == vals[k] + tv[k])
            a.timings["D_teacher_" + label] = teacher.get("timing", {})
        a.timings["D_worker_" + label + "_overlap"] = dict(d.get("timing", {}),
                note="Worker encloses sampling, archive writes and teacher; these sub-timings overlap this wall measure.")

    totals = {key: sum(row[key] for row in a.execution) for key in EXPECTED}
    nonpass = {rel: v["status"] for rel, v in a.records.items()
               if (rel.startswith(("A/", "B/", "stage_c/", "stage_d/", "D_evaluation/")))
               and v["status"] not in ("PASS", "PASS_EXACT_REPLAY")}
    if a.missing and not allow_incomplete and not any(d.get("status") in BLOCKED for d in replays.values()):
        raise RuntimeError("Missing final workers; refuse premature finalization (or explicitly --allow-incomplete): " + ", ".join(a.missing))
    complete = not a.missing and not nonpass
    for key, cap in EXPECTED.items():
        a.check("budget/" + key, totals[key] <= (230 if key in ("full_rollout_attempts", "complete_rollouts") else cap), totals[key], cap)
        if complete:
            a.check("complete_expected/" + key, totals[key] == cap, totals[key], cap)
    a.check("no_repairs", totals["postprocessing_repairs"] == 0)
    return a, protocol, preflight, totals, complete, nonpass, checkpoints


def protected_assets(a, protocol):
    assets = []
    for item in protocol["checkpoint_identities"]:
        assets.append(("checkpoint_" + item["label"], item["path"], item.get("checkpoint_sha256", item["sha256"])))
    for key, item in protocol["inputs"].items():
        assets.append(("data_" + key, item["path"], item["sha256"]))
    a.check("protected_asset_count", len(assets) == 5, len(assets), 5)
    records = []
    for name, path, expected in assets:
        path = _resolve(path, a.out)
        rec = dict(name=name, path=str(path), expected_sha256=expected, exists=path.is_file())
        if path.is_file():
            rec.update(bytes=path.stat().st_size, actual_sha256=sha256(path))
        rec["unchanged"] = rec.get("actual_sha256") == expected
        a.check("protected/" + name, rec["unchanged"])
        records.append(rec)
    return records


def git_identity(a, protocol):
    head = _git("rev-parse", "HEAD").decode().strip()
    # Retain pre-existing user changes. Only maintained production Python is a hard guard.
    status = _git("status", "--porcelain").decode("utf-8", errors="replace")
    tracked = _git("ls-files", "-z", "--", "native_t1").decode().split("\0")
    production = [p for p in tracked if p.endswith(".py") and not any(x in Path(p).parts for x in ("tests", "runs", "maintenance"))]
    differences = []
    for path in production:
        current = ROOT / path
        # Git's working-tree conversion is used, respecting existing EOL attributes.
        for staged, opts in ((False, []), (True, ["--cached"])):
            diff = _git("diff", "--no-ext-diff", *opts, "--", path)
            if diff:
                differences.append(dict(path=path, staged=staged, diff_sha256=hashlib.sha256(diff).hexdigest()))
        if not current.is_file():
            differences.append(dict(path=path, deleted=True))
    a.check("git_HEAD_unchanged", head == protocol["HEAD"], head, protocol["HEAD"])
    a.check("tracked_native_t1_production_no_diff", not differences, differences)
    return dict(HEAD=head, registered_HEAD=protocol["HEAD"], current_status=status,
                registered_status=protocol.get("git_status"), production_python_files_checked=len(production),
                production_differences=differences,
                scope="Tracked native_t1 production Python; pre-existing unrelated/test changes are retained and listed, not reset.")


def score_timing_and_references(a, complete):
    result = []
    expected_counts = dict(A_B=84, A_J=84, B_B=36, B_J=36,
                          **{"D_" + label: (20 if label.endswith("500") else 4) for label in D_LABELS})
    seen = set()
    for path in sorted((a.out / "scores").glob("*/results.json")):
        d = _read(path)
        if d.get("status") in ACTIVE:
            raise RuntimeError("CPU scoring is still active: " + str(path))
        a.check("CPU_score/" + path.parent.name + "/no_network_or_repair", d.get("model_forwards", 0) == 0 and d.get("repairs", 0) == 0)
        if path.parent.name in expected_counts:
            seen.add(path.parent.name)
            if complete:
                a.check("CPU_score/" + path.parent.name + "/complete", d.get("status") == "PASS" and
                        d.get("completed") == expected_counts[path.parent.name] and len(d.get("rows", [])) == expected_counts[path.parent.name])
            # These are references to counted generations, never additional network work.
            source = (("A/" + path.parent.name[2:] + "/results.json") if path.parent.name.startswith("A_") else
                      ("B/" + path.parent.name[2:] + "/results.json") if path.parent.name.startswith("B_") else
                      ("D_evaluation/" + path.parent.name[2:] + "/results.json"))
            source_path = a.out / source
            if source_path.is_file():
                origin = _read(source_path).get("rows", [])
                identities = lambda rows: [(x.get("path"), x.get("sha256"), x.get("model"), x.get("parent"),
                                            x.get("patch"), x.get("seed"), x.get("precision"), x.get("nfe"),
                                            x.get("source_model"), x.get("continuation_model"), x.get("start_step", 0)) for x in rows]
                scored = identities(d.get("rows", []))
                a.check("CPU_score/" + path.parent.name + "/same_generated_rows", scored == identities(origin)[:len(scored)])
        result.append(dict(**_record(path), phase=d.get("phase"), status=d.get("status"),
                           completed=d.get("completed"), worker_wall_seconds=d.get("seconds"),
                           sum_endpoint_seconds=sum(float(x.get("seconds", 0.)) for x in d.get("rows", [])),
                           note="Endpoint times are a subset of this worker's wall time; concurrent workers cannot be summed as elapsed wall."))
    if complete:
        a.check("CPU_scores/all_registered_workers_present", seen == set(expected_counts), sorted(seen), sorted(expected_counts))
    return result


def source_delivery(out):
    """Actual git --no-index patches, verified by applying in a disposable empty tree."""
    files = sorted(SOURCE.rglob("*.py"))
    chunks, entries = [], []
    snapshot = out / "source_snapshot"
    for source in files:
        relative = source.relative_to(ROOT).as_posix()
        # New isolated sources are delivered against /dev/null, not a guessed historical base.
        patch = _git("diff", "--no-index", "--binary", "--", "/dev/null", relative, allowed=(1,))
        chunks.append(patch)
        data = source.read_bytes()
        destination = snapshot / (relative + ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        entries.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                            snapshot=str(destination.relative_to(out))))
    patch_data = b"".join(chunks)
    patch_path = out / "code.diff"
    patch_path.write_bytes(patch_data)
    with tempfile.TemporaryDirectory(prefix="jedge5_delivery_diff_") as temporary:
        _git("apply", "--check", str(patch_path.resolve()), cwd=temporary)
        _git("apply", str(patch_path.resolve()), cwd=temporary)
        for entry in entries:
            restored = Path(temporary) / entry["path"]
            if not restored.is_file() or sha256(restored) != entry["sha256"]:
                raise RuntimeError("Delivered diff cannot restore source exactly: " + entry["path"])
    test_records = []
    record_paths = set(SOURCE.glob("*record*.json")) | set(out.glob("*test*record*.json"))
    for path in sorted(record_paths):
        data = path.read_bytes()
        if len(data) > 2 * 1024 * 1024:
            raise RuntimeError("Unexpected large test record; refusing a bulk asset copy: " + str(path))
        parsed = json.loads(data.decode("utf-8-sig"))
        destination = out / "test_records" / (("source_" if path.parent == SOURCE else "run_") + path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        test_records.append(dict(source=str(path), copy=str(destination.relative_to(out)),
                                 bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), status=parsed.get("status"),
                                 recorded_counts=parsed.get("counts"), tests_run=parsed.get("tests_run"),
                                 checks=parsed.get("checks"), serialization_note=parsed.get("serialization_note")))
    manifest = dict(schema="jedge5_actual_delivery_sources_v1", sources=entries, code_diff=_record(patch_path),
                    patch_reconstruction=dict(status="PASS", files=len(entries), all_source_bytes_identical=True),
                    test_records=test_records,
                    note="Actual final isolated experiment sources. Original test records retain their original source hashes, including documented pre-execution amendments. Snapshot names end in .py.txt to keep historical copies non-importable. No weights, old packages or datasets copied.")
    manifest_path = out / "code_manifest.json"
    _write(manifest_path, manifest)
    return dict(manifest=_record(manifest_path), code_diff=_record(patch_path),
                source_count=len(entries), test_record_count=len(test_records), patch_reconstruction=manifest["patch_reconstruction"])


def finalize(out, allow_incomplete=False):
    out = Path(out).resolve()
    if not out.is_dir():
        raise FileNotFoundError(out)
    if (out / "final_audit.json").exists():
        raise FileExistsError("Final audit already exists; do not silently overwrite it")
    start = time.perf_counter()
    a, protocol, preflight, totals, complete, nonpass, checkpoints = collect(out, allow_incomplete)
    score_records = score_timing_and_references(a, complete)
    for relative in ("analysis/A/results.json", "analysis/B/results.json", "stage_c/analysis.json", "analysis/D.json"):
        path = out / relative
        if complete:
            a.check("analysis/" + relative + "/complete", path.is_file() and _read(path).get("status") == "PASS")
        if path.is_file():
            a.records[relative] = dict(_record(path), status=_read(path).get("status"))
    protected = protected_assets(a, protocol)
    git = git_identity(a, protocol)
    # Reused raw3000 A/C and cached aggregate records never enter this accounting.
    exact_d = len(checkpoints) == 2 and all(v["exact_replay"] for v in checkpoints.values())
    if complete:
        a.check("complete_requires_D_raw_exact", exact_d)
    static_path = out / "registration/static_context_audit.json"
    static = _read(static_path) if static_path.is_file() else {}
    timing = dict(schema="jedge5_timing_v1", units=a.timings, CPU_scoring=score_records,
                  preflight_seconds=preflight.get("seconds"), static_context_audit_seconds=static.get("seconds"),
                  global_experiment_elapsed_wall_seconds="NOT_AVAILABLE_IN_CANONICAL_WORKER_LEDGERS",
                  notes=["Per-worker wall durations are not elapsed experiment time when workers overlap.",
                         "CUDA event, host enqueue, explicit synchronization and worker wall durations overlap; do not sum them.",
                         "D model_update_seconds encloses parts of EMA/synchronization; original nested timing labels are preserved.",
                         "C parameter probe records a combined network/loss/VJP duration; no unmeasured split is invented.",
                         "CPU scoring timings are read from saved scores; no scores or geometry are recomputed."])
    _write(out / "timing.json", timing)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["unit", "source", "status", *EXPECTED])
    writer.writeheader()
    writer.writerows(a.execution)
    (out / "execution.csv").write_text(buffer.getvalue(), encoding="utf-8")
    delivery = source_delivery(out)
    failures = [v for v in a.checks if not v["passed"]]
    status = "FAIL" if failures else ("PASS" if complete and exact_d else "BLOCKED_INCOMPLETE")
    report = dict(schema="jedge5_final_audit_v1", status=status,
                  utc=datetime.now(timezone.utc).isoformat(), all_registered_work_complete=complete,
                  missing_records=a.missing, nonpass_records=nonpass,
                  counts=totals, expected_all_pass_counts=EXPECTED, full_generation_ceiling=230,
                  accounting_note="A169 includes one production replay; B78 are suffixes, not full generations. C400 teacher+8 parameter forwards. D8000 training+3000 sampling+140 teacher. Raw3000 and aggregate references are not recounted.",
                  formal_full_generations=totals["complete_rollouts"] - (1 if any(x["unit"] == "A_B" and x["complete_rollouts"] == 85 for x in a.execution) else 0),
                  full_including_production_replay=totals["complete_rollouts"], D=checkpoints,
                  checks=a.checks, failed_checks=failures, protected_assets=protected, git=git,
                  records=a.records, scoring_references=score_records, delivery=delivery,
                  execution_csv=_record(out / "execution.csv"), timing=_record(out / "timing.json"),
                  audit_execution=dict(seconds=time.perf_counter() - start, model_constructions=0,
                     model_forwards=0, model_backwards=0, leaf_VJPs=0, parameter_VJPs=0,
                     OT_calls=0, optimizer_updates=0, geometry_calls=0, scoring_calls=0,
                     checkpoint_deserializations=0, weight_access="Streaming file SHA only; no tensor/model load"),
                  notes=["Existing view_image permission failure and subsequent base64 PNG display are presentation operations, not scientific calls.",
                         "Global-gate CPU fixture execution succeeded; the first record serializer raised NameError. The existing result was serialized without rerunning the fixtures; its original note is preserved.",
                         "Recorded C preservation and replay identities are checked from saved audits; this finalizer does not regenerate or rescore arrays.",
                         "PASS denotes completed bookkeeping/protection checks, not efficacy of JEdge5 or proof that 200-step output is ground truth."])
    _write(out / "final_audit.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicitly record terminal blocked/missing stages; never grants PASS or permits active workers")
    args = parser.parse_args(argv)
    report = finalize(args.out, args.allow_incomplete)
    print(json.dumps(dict(status=report["status"], counts=report["counts"], failed_checks=len(report["failed_checks"]))))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
