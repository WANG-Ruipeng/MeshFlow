"""Seal completed, audited local evidence; never invoke a model or a scorer."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_adaptation_strength_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.audit.resolve()
    audit_path.relative_to(OUT / "closeout_audits_v2")
    audit = read(audit_path)
    if audit["status"] != "PASS" or not audit["final_requested"]:
        raise ValueError("A completed final V2 audit is required")
    visual = OUT / "visual_review/completion.json"
    analysis = OUT / "result_analysis/completion.json"
    values = OUT / "diagnostics/results_v2_independent_audit.json"
    v, a, n = read(visual), read(analysis), read(values)
    if (v["status"] != "ACTUAL_AI_PROXY_REVIEW_COMPLETE"
            or v["reviewed_cells"] != 672 or v["missing_cells"] != 0
            or a["status"] != "COMPLETE" or n["status"] != "PASS"):
        raise ValueError("Actual review, analysis and independent values must be complete")
    report = OUT / "REPORT.md"
    text = report.read_text(encoding="utf-8-sig")
    if "状态：COMPLETE" not in text:
        raise ValueError("Root must finish the scientific report before sealing")
    files = [
        report, OUT / "RUNBOOK.md", audit_path, visual, analysis, values,
        OUT / "results_v2/completion.json", OUT / "registration.json",
        OUT / "training_registration.json", OUT / "eval_manifest.json",
        OUT / "diagnostics/execution_summary.json",
        OUT / "training_summary/latest.json", Path(__file__).resolve(),
    ]
    hashes = {str(p): sha(p) for p in files}
    destination = OUT / "completion.json"
    if destination.exists():
        old = read(destination)
        if old["files"] != hashes:
            raise ValueError("Refuse to overwrite an existing seal with changed evidence")
        print(json.dumps(old, ensure_ascii=False, indent=2))
        return
    result = dict(
        schema="BENCH_ADAPTATION_ROOT_CLOSEOUT_V1",
        status="COMPLETE", utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        experiment="R2_BENCH_ADAPTATION_STRENGTH_V1",
        conclusion="No overall promotion established in this fixed-budget three-arm comparison",
        updates=4500, updates_per_arm=1500, RAW_outputs=224, D0_forwards=224,
        scored_cells=672, gallery_sheets=144, actual_AI_PROXY_review_cells=672,
        formal_human_ratings=0, blind=False,
        authoritative_results="results_v2", final_audit=str(audit_path),
        files=hashes, automatic_extension=False, default_changed=False,
        Chair_changed=False, cleanup_actions=0, commits=0, pushes=0,
        this_seal_model_calls=0, this_seal_scoring_calls=0,
    )
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
