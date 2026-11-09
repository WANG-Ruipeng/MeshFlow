"""Read-only compact report facts from the completed, audited training run."""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

ROOT = Path("/mnt/e/MeshFlow")
OUT = ROOT / "experiments/r2_bench_clean_multipart_seam_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    summary = OUT / "training_summary/20261020T035233_197593Z"
    status = read(OUT / "training/status.json")
    assert status["status"] == "COMPLETE"
    with (summary / "fixed_probe_summary.csv").open(encoding="utf-8-sig") as f:
        probes = list(csv.DictReader(f))
    with (summary / "historical_fixed_probe_summary.csv").open(encoding="utf-8-sig") as f:
        history = list(csv.DictReader(f))
    with (summary / "exposure_by_condition.csv").open(encoding="utf-8-sig") as f:
        exposure = list(csv.DictReader(f))
    log = [json.loads(s) for s in (OUT / "training/training_log.jsonl").read_text().splitlines()]
    calls = [json.loads(s) for s in (OUT / "training/calls.jsonl").read_text().splitlines()]
    parent_counts = collections.Counter()
    for row in exposure:
        parent_counts[row["parent_id"]] += int(row["exposures"])
    done_seconds = collections.defaultdict(float)
    for row in calls:
        if row["event"] == "CALL_DONE":
            done_seconds[row["kind"]] += row["seconds"]
    fields = ("all_free_MSE", "seam_tip_MSE", "bulk_MSE")

    def metrics(row):
        return {
            "split": row["split"],
            **{k: float(row[k]) for k in fields},
            **{k: int(row[k]) for k in
               ("examples", "all_free_scalar_count", "seam_tip_scalar_count", "bulk_scalar_count")},
        }

    payload = {
        "schema": "BENCH_CLEAN_COMPACT_TRAINING_REPORT_V1",
        "source_summary": str(summary),
        "status": "COMPLETE_AUDITED_PASS",
        "new_final": [
            {"local_step": 1500, "global_step": 13500, **metrics(r)}
            for r in probes if r["part"] == "ALL" and r["local_step"] == "1500"
        ],
        "new_fixed_curve": [
            {"local_step": int(r["local_step"]), "global_step": 12000 + int(r["local_step"]), **metrics(r)}
            for r in probes if r["part"] == "ALL"
        ],
        "historical_same_probe": [
            {"model": r["model"], "global_step": int(r["global_step"]),
             "stage_updates": int(r["stage_updates"]), "comparison": r["comparison"], **metrics(r)}
            for r in history if r["part"] == "ALL"
        ],
        "exposure": {
            "parents": len(parent_counts), "conditions": len(exposure),
            "instances": sum(parent_counts.values()),
            "per_parent_min": min(parent_counts.values()),
            "per_parent_max": max(parent_counts.values()),
            "per_condition_min": min(int(r["exposures"]) for r in exposure),
            "per_condition_max": max(int(r["exposures"]) for r in exposure),
            "clean": sum(int(r["clean"]) for r in exposure),
            "anisotropic": sum(int(r["augmented"]) for r in exposure),
        },
        "accounting": read(summary / "accounting.json"),
        "timing": {
            "training_process_elapsed_seconds": status["elapsed_seconds"],
            "scope": "Formal trainer start to COMPLETE; includes setup, four fixed probes, updates and checkpoint/export I/O.",
            "sum_formal_batch_update_seconds": sum(r["update_seconds"] for r in log),
            "sum_input_preparation_seconds": sum(r["input_seconds"] for r in log),
            "sum_instrumented_call_seconds_by_kind": dict(done_seconds),
            "timing_limits": [
                "Instrumented call sums and input times overlap batch wall scope; do not add them to elapsed.",
                "No isolated-compute benchmark claim: unrelated graphics used the same GPU.",
            ],
        },
        "memory": {
            **status["resources"],
            "max_formal_batch_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in log),
            "max_formal_batch_peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in log),
            "pytorch_scope": "Reset/sample within each real batch including generator/head backward and both AdamW updates; not full process VRAM.",
        },
        "interpretation_limits": [
            "Old SEAM15000 and SEAM21000 are unequal-budget recipe references, not isolated data/head ablations.",
            "Fixed probes use source-target interpolation and source owner layouts; they do not establish rollout quality.",
            "DEV ALL/tip errors rebound from step1000 to1500; the preselected endpoint remains1500.",
            "This summary runs zero model constructions, forwards, backwards or GPU calls.",
        ],
    }
    target = summary / "compact_training_report.json"
    assert not target.exists()
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "path": str(target)}))


if __name__ == "__main__":
    main()
