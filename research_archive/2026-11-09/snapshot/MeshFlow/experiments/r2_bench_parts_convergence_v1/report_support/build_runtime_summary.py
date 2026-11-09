"""Read completed child receipts only; no model or scoring work."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_support" / "runtime"
OUT.mkdir(exist_ok=True)

def load(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def interval_stats(intervals):
    events = sorted([(a, 1) for a, b in intervals] + [(b, -1) for a, b in intervals])
    active = peak = 0
    covered = 0.0
    previous = None
    for t, change in events:
        if previous is not None and active:
            covered += t - previous
        active += change
        peak = max(peak, active)
        previous = t
    return covered, peak

rows = []
for p in sorted((ROOT / "dispatch" / "jobs").glob("*/completion.json")):
    d = load(p)
    key = d["key"]
    if key.startswith("train_"):
        phase = "training"
    elif key.startswith("raw_"):
        phase = "Euler_sampling"
    elif key.startswith("post_"):
        phase = "D0_and_geometry"
    elif key.startswith("score_"):
        phase = "CPU_scoring"
    elif key.startswith("gallery_"):
        phase = "CPU_gallery"
    else:
        phase = "CPU_other"
    rows.append(dict(phase=phase, **{k: d.get(k) for k in (
        "key", "returncode", "wall_seconds", "started", "finished_unix",
        "peak_system_delta_mib", "peak_reserved_mib")},
        source=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))

with (OUT / "completed_children.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
groups = defaultdict(list)
for row in rows:
    groups[row["phase"]].append(row)
summary = {}
for phase, group in sorted(groups.items()):
    successful = [r for r in group if r["returncode"] == 0]
    intervals = [(r["started"], r["finished_unix"]) for r in successful]
    covered, concurrency = interval_stats(intervals)
    summary[phase] = dict(completed_children=len(group), successful_children=len(successful),
        sum_successful_child_wall_seconds=sum(r["wall_seconds"] for r in successful),
        successful_intervals_union_seconds=covered, max_successful_child_overlap=concurrency,
        max_reported_allocator_reserved_mib=max((r["peak_reserved_mib"] or 0 for r in successful), default=0),
        max_reported_system_delta_mib=max((r["peak_system_delta_mib"] or 0 for r in successful), default=0))
result = dict(status="COMPLETE_RECEIPTED_CHILD_SCOPE", phases=summary,
    scope="Completed dispatch/jobs/*/completion.json; interrupted attempts without a completion receipt are not silently assigned wall time. Recovery computation is separately accounted. Union seconds measure child-process interval coverage, not GPU kernel time. Overlap is child concurrency, not proof of kernel overlap. System VRAM delta includes concurrent/user allocations; allocator reserved is not whole-process VRAM.",
    model_calls=0, rescores=0, source_count=len(rows))
(OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
