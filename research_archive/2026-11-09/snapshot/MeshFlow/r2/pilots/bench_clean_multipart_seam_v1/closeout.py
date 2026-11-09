"""Validate the completed bounded experiment and write its delivery receipt."""
from __future__ import annotations
import argparse
import json
from .common import OUT, read, sha, resolve, atomic_write


def run():
    execution = read(OUT / "independent_execution_audit.json")
    visual = read(OUT / "visual_review/completion.json")
    source = read(OUT / "source_hashes.json")
    training = read(OUT / "training/status.json")
    assert execution["status"] == "PASS"
    assert visual["status"] == "ACTUAL_AI_PROXY_REVIEW_COMPLETE"
    assert (visual["requests"], visual["image_sheets"], visual["cells"]) == (48, 96, 576)
    assert source["status"] == "COMPLETE"
    assert training["status"] == "COMPLETE" and training["local_step"] == 1500
    for r in source["source_files"]:
        assert sha(resolve(r["path"])) == sha(resolve(r["archive"])) == r["sha256"]
    assert sha(resolve(source["diff_path"])) == source["diff_sha256"]
    weights = {}
    for key in ("checkpoint", "inference_export", "occupancy_export"):
        r = training[key]
        assert sha(resolve(r["path"])) == r["sha256"]
        weights[key] = r
    documents = ("REPORT.md", "DELIVERY_INDEX.md", "independent_execution_audit.json",
                 "visual_review/completion.json", "visual_review/per_cell.csv",
                 "comparison/completion.json", "layout_diagnostic/completion.json",
                 "source_hashes.json", "code.diff")
    for name in ("REPORT.md", "DELIVERY_INDEX.md"):
        text = (OUT / name).read_text(encoding="utf-8")
        assert text.strip() and "\ufffd" not in text, "Invalid UTF-8 report"
    result = dict(
        status="COMPLETE", experiment="R2_BENCH_CLEAN_MULTIPART_SEAM_V1",
        training_updates=1500, global_step=13500, new_RAW=96, new_single_D0=96,
        historical_CPU_outputs=144, primary_RAW_BASE_FINAL_metric_cells=720,
        original_metric_fields=179, actual_AI_PROXY_cells=576, formal_human_ratings=0,
        weights=weights, files={name: sha(OUT / name) for name in documents},
        training_replays=0, extra_partial_CPU_BASE_scores=2,
        stopped_at_registered_endpoint=True, best_checkpoint_selection=False,
        default_changed=False, cleanup=False, commit=False, push=False,
        model_calls_by_closeout=0,
        caveat="Completion certifies delivered work and accounting, not a positive scientific result.")
    atomic_write(OUT / "completion.json", result)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))
