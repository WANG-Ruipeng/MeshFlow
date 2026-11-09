"""Final registered curriculum audit using text/JSON only; no torch or weights."""
import collections
import datetime
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("transition_audit", HERE / "audit_native6000_seam25.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
OUT, ROOT = A.OUT, A.ROOT


def run():
    reg = A.read(OUT / "training_registration.json")
    regsha = A.sha(OUT / "training_registration.json")
    A.require(regsha == A.sha(OUT / "registration.json"), "Registration aliases differ")
    A.require(reg["stage_updates"] == {"NATIVE": 6000, "SEAM": 3000}, "Stage update budget differs")
    A.require((reg["source_global_step"], reg["final_global_step"], reg["generator_training_forwards"],
               reg["generator_backwards"], reg["optimizer_updates"], reg["training_head_forwards"],
               reg["training_head_updates"], reg["probe_forwards"]) == (12000, 21000, 72000, 72000, 9000, 0, 0, 0),
              "Global/call budget differs")
    schedule_path = OUT / "data/schedule.json"
    A.require(A.sha(schedule_path) == reg["input_schedule_sha256"], "Schedule identity changed")
    schedule = A.read(schedule_path)["steps"]
    A.require(len(schedule) == 9000, "Schedule length differs")
    stages, sources, boundaries = {}, {}, {}
    files = [OUT / "registration.json", OUT / "training_registration.json", Path(__file__), HERE / "audit_native6000_seam25.py"]
    for stage, count in (("NATIVE", 6000), ("SEAM", 3000)):
        folder = OUT / "training" / stage
        status = A.read(folder / "status.json")
        workflow_path = OUT / "workflow" / stage.lower() / "status.json"
        workflow = A.read(workflow_path)
        A.require(status["status"] == workflow["status"] == "COMPLETE" and workflow["exit_code"] == 0,
                  "Stage workflow or training is not COMPLETE/exit0")
        A.cursor(status, stage, count)
        A.require(status["registration_sha256"] == regsha, "Training registration differs")
        attempts = list((folder / "attempts").iterdir())
        A.require(len(attempts) == 1 and attempts[0].name == status["attempt"], "Unexpected attempt/restart")
        adir = attempts[0]
        A.require(not (adir / "failure.json").exists(), "Stage has failure receipt")
        A.require(A.read(adir / "final_status.json") == status, "Final attempt not sealed or differs")
        A.require(status["calls"]["pending_call_ids"] == [] and not status["calls"]["failed"], "Failed/pending final calls")
        A.require(status["head_step"] == status["head_calls"] == status["probe_calls"] == 0, "Unexpected head/probe activity")
        limits = dict(train_forward=8 * count, train_backward=8 * count, optimizer_update=count,
                      head_forward=0, head_backward=0, head_optimizer_update=0, probe_forward=0)
        A.require(status["calls"]["limits"] == limits, "Runtime call budget differs")
        for k in ("attempted", "successful"):
            A.require(status["calls"][k] == {name: value for name, value in limits.items() if value}, "Runtime counts differ")
        # The reusable audit checks every step/sample and role denominator. This
        # additional whole-file pass rules out hidden suffixes beyond an endpoint.
        for name, wanted in (("calls", 34 * count), ("exposure", 8 * count), ("training_log", count)):
            actual = 0
            for row in A.records(folder / (name + ".jsonl")):
                cur = row["context"] if name == "calls" else row
                A.require(1 <= cur["stage_step"] <= count, "Out-of-budget scientific ledger suffix")
                actual += 1
            A.require(actual == wanted, "Exact ledger row count differs: " + name)
        detail = A.audit_stage(stage, count, schedule)
        detail.update(final_status="COMPLETE", observed_attempt_elapsed_seconds=status["elapsed_seconds"],
                      workflow=workflow, observed_device_resources=status["resources"],
                      initialization=A.read(adir / "initialization.json"))
        stages[stage] = detail
        for step in reg["stage_full_steps"][stage]:
            receipt_path = folder / ("checkpoint_step" + str(step) + ".json")
            receipt = A.read(receipt_path)
            offset = 0 if stage == "NATIVE" else 6000
            A.require(receipt["stage"] == stage and receipt["stage_step"] == receipt["local_step"] == step
                      and receipt["curriculum_step"] == offset + step and receipt["global_step"] == 12000 + offset + step,
                      "Immutable milestone cursor differs")
            A.require(receipt["head_state_sha256"] == reg["head_state_sha256"], "Immutable milestone head differs")
            boundaries[stage + "_" + str(step)] = receipt
            files.append(receipt_path)
        end = boundaries[stage + "_" + str(count)]
        export_path = folder / ("generator_step" + str(count) + ".json")
        exported = A.read(export_path)
        A.require(status["checkpoint"] == end and status["inference_export"] == exported, "Status/endpoint export mismatch")
        for k in ("generator_state_sha256", "head_state_sha256", "global_step", "curriculum_step", "stage_step", "stage"):
            A.require(end[k] == exported[k], "Export/end state differs: " + k)
        sources[stage] = dict(checkpoint=end, inference_export=exported)
        files.extend([folder / "status.json", workflow_path, adir / "final_status.json",
                      adir / "initialization.json", adir / "resources.json", export_path])
    n0, ne, s0 = (boundaries[k] for k in ("NATIVE_0", "NATIVE_6000", "SEAM_0"))
    for key, registered in (("generator_state_sha256", "source_generator_sha256"),
                            ("optimizer_state_sha256", "source_optimizer_sha256"),
                            ("rng_sha256", "source_rng_sha256"), ("head_state_sha256", "head_state_sha256")):
        A.require(n0[key] == reg[registered], "COMMON to Native start identity differs")
    A.require(all(ne[k] == s0[k] for k in A.HASH_KEYS), "Native to SEAM four-state inheritance differs")
    for stage in stages.values():
        init = stage["initialization"]
        A.require(init["inherited_optimizer"] and init["inherited_rng"] and init["fresh_AdamW"] is False,
                  "COMMON AdamW/RNG initialization policy differs")
        A.require(init["head_frozen"] and init["head_calls"] == 0 and init["head_state_sha256"] == reg["head_state_sha256"],
                  "Initialization head activity/identity differs")
    transition = stages["SEAM"]["initialization"]["stage_transition"]
    A.require(transition["optimizer_reset"] is False and transition["head_frozen"] is True
              and transition["same_manifest_sha256"] == reg["manifest_sha256"]
              and transition["source_global_step"] == 18000 and transition["source_sha256"] == ne["sha256"],
              "Stage transition reset/source/pool differs")
    A.require(all(transition[k] == ne[k] for k in A.HASH_KEYS[:3]), "Transition initialization state identity differs")
    parent = collections.Counter()
    for stage in stages.values():
        parent.update(stage["exposure"]["parent_counts"])
    A.require(sum(parent.values()) == 72000 and len(parent) == 22, "Total exposure/parent count differs")
    total_calls = collections.Counter()
    for stage in stages.values():
        total_calls.update(stage["calls"]["successful"])
    A.require(total_calls == {"train_forward": 72000, "train_backward": 72000, "optimizer_update": 9000}, "Final calls differ")
    overlaps = []
    for task in ("raw-common", "raw-native"):
        path = OUT / "workflow" / task / "status.json"
        if path.is_file():
            sample = A.read(path)
            files.append(path)
            for stage, detail in stages.items():
                training = detail["workflow"]
                seconds = max(0., min(training["finished"], sample.get("finished", training["finished"]))
                              - max(training["started"], sample["started"]))
                if seconds:
                    overlaps.append(dict(training_stage=stage, independent_task=task,
                                         overlap_seconds=seconds, sampling_workflow=sample))
    result = dict(schema="BENCH_FULL_SEAT_CURRICULUM_FINAL_TRAINING_AUDIT_V1", status="PASS",
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  global_steps=dict(source=12000, native=18000, seam=21000),
                  actual_calls=dict(total_calls), training_head_calls=0, training_head_updates=0, probe_calls=0,
                  failures=0, pending_calls=0, stage_attempts={k: v["calls"]["attempt_ids"] for k, v in stages.items()},
                  scientific_retries=0, total_exposures=72000, parents=22, conditions=22,
                  actual_parent_exposure=dict(sorted(parent.items())), stages=stages,
                  four_state_transition="PASS", final_saved_identities=sources, milestone_receipts=boundaries,
                  registered_independent_sampling_overlap=overlaps,
                  evidence_hashes={str(p.relative_to(ROOT)): A.sha(p) for p in files},
                  own_activity=dict(model_forwards=0, GPU_tasks=0, scoring_calls=0, tensor_loads=0, large_weight_rehashes=0,
                                    source_archive=False, intervention=False),
                  boundaries=["Saved JSON state hashes independently compared; tensor/storage hashes were not independently recomputed here.",
                              "Allocator peak is not complete process VRAM; device peak includes concurrent/background processes.",
                              "Concurrent stage timings are observed wall time, not isolated compute benchmarks.",
                              "Native and SEAM have different free-coordinate roles; loss is source-supervised diagnosis, not generated quality.",
                              "Training completion does not certify controller READY_REVIEW, evaluation completion, or actual visual review."])
    path = HERE / "final_training_9000.json"
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("schema", "status", "actual_calls", "total_exposures", "failures", "scientific_retries", "four_state_transition")}))


if __name__ == "__main__":
    run()
