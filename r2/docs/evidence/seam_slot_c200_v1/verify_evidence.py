"""Read-only checksum and scalar check of the shipped evidence; stdlib only."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent


def rows(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize(table, label, stage="POST"):
    subset = [r for r in table if r["label"] == label and r["stage"] == stage]
    assert len(subset) == 16, (label, stage, len(subset))
    assert len({r["request_id"] for r in subset}) == 16
    return {
        "label": label,
        "stage": stage,
        "n": len(subset),
        "FF_percent": 100 * mean(float(r["FF_illegal_affected_free_fraction"]) for r in subset),
        "I_percent": 100 * mean(float(r["ConflictCov_C"]) for r in subset),
        "legal_CF_edges": sum(int(r["C_F_valid_edge_count"]) for r in subset),
        "requests_with_legal_CF": sum(int(r["C_F_has_valid_edge"]) for r in subset),
    }


def main():
    manifest = rows("source_hashes.csv")
    assert len(manifest) == 45
    for item in manifest:
        path = (ROOT / item["evidence_path"]).resolve()
        path.relative_to(ROOT)
        data = path.read_bytes()
        assert len(data) == int(item["bytes"]), str(path)
        assert hashlib.sha256(data).hexdigest() == item["sha256"], str(path)
    original = rows("c200/evaluation/per_output.csv")
    layouts = rows("occupancy_rollout/analysis/per_output.csv")
    capacity = rows("geometry_guard/capacity/per_output.csv")
    assert (len(original), len(layouts), len(capacity)) == (64, 192, 32)
    for path, expected in [
        ("c200/evaluation/visual_review_all64.csv", 64),
        ("occupancy_rollout/gallery/actual_visual_review.csv", 144),
        ("geometry_guard/gallery/actual_visual_review.csv", 16),
    ]:
        assert len(rows(path)) == expected, path
    results = [summarize(original, label) for label in ("FULL4_C200", "SEAM_SLOT_C200")]
    for label in ("PREDICTED", "ALL_ON", "SOURCE_ORACLE"):
        for stage in ("POST", "CAPACITY_POST"):
            result = summarize(layouts, label, stage)
            subset = [r for r in layouts if r["label"] == label and r["stage"] == stage]
            fractions = [float(r["free_only_reached_bulk_area_fraction"]) for r in subset]
            result.update(bulk_reach_mean_percent=100 * mean(fractions),
                          bulk_reach_nonzero=sum(x > 0 for x in fractions),
                          bulk_reach_ge30=sum(x >= 0.3 for x in fractions))
            results.append(result)
    # The cached PREDICTED arm is overlapping evidence, not a new independent run.
    old = {r["request_id"]: r for r in original
           if r["label"] == "SEAM_SLOT_C200" and r["stage"] == "POST"}
    for current in layouts:
        if current["label"] == "PREDICTED" and current["stage"] == "POST":
            previous = old[current["request_id"]]
            for key in ("FF_illegal_affected_free_fraction", "ConflictCov_C",
                        "C_F_valid_edge_count", "free_only_reached_bulk_area_fraction"):
                assert float(previous[key]) == float(current[key]), (current["request_id"], key)
    print(json.dumps({"status": "PASS", "hashed_files": len(manifest),
                      "copied_bytes": sum(int(r["bytes"]) for r in manifest),
                      "new_model_or_mesh_scoring_calls": 0, "observed": results}, indent=2))


if __name__ == "__main__":
    main()
