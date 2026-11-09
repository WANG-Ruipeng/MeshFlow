"""Read saved scalar tables only; no models, geometry scoring or weight reads."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "experiments/r2_bench_adaptation_strength_v1"
HERE = Path(__file__).resolve().parent
SOURCES = {
    "historical": ROOT / "experiments/r2_bench_clean_multipart_seam_v1/comparison/per_output.csv",
    "current": OUT / "results_v2/per_output.csv",
}
METRICS = [
    "FF_illegal_affected_free_fraction", "CF_illegal_affected_free_fraction",
    "ConflictCov_C", "free_minimum_angle_lt5_fraction",
    "FF_illegal_pair_count", "CF_illegal_pair_count",
    "actual_free_faces", "deleted_free_faces", "free_total_surface_area",
    "nonmanifold_edges", "C_F_valid_edge_count", "C_F_has_valid_edge",
    "C_F_valid_edge_total_length", "hinge_diag_isolated_1face_edge_components",
    "legal_seam_bulk_edge_count", "free_only_reached_bulk_area_fraction",
    "bulk_reach_ge_0_30_descriptive", "free_q_gt10_fraction",
    "free_exact_zero_area_fraction", "A_VMR__L_tau_0.005",
]
NEW = ("A_S1500", "B_S1500", "C_S1500")
OLD = ("NATIVE_S21000", "SEAM_S15000", "SEAM_S21000", "OLD_S15000_LAYOUT", "NEW_HEAD")
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(text):
    if text in ("", None):
        return None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(text)
    return value


def parent_equal(rows, metric):
    parents = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = number(row.get(metric))
        if value is not None:
            parents[row["parent_id"]][row["condition_id"]].append(value)
    values = [statistics.mean(statistics.mean(v) for v in conditions.values())
              for conditions in parents.values()]
    return statistics.mean(values) if values else None


def main():
    old_root = SOURCES["historical"].parents[1]
    old_manifest = old_root / "eval_manifest.json"
    new_manifest = OUT / "eval_manifest.json"
    load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    assert load(old_manifest) == load(new_manifest)
    old_scoring = old_root / "evaluation/score_registration.json"
    new_scoring = OUT / "evaluation/score_registration.json"
    osrc, nsrc = load(old_scoring)["source_hashes"], load(new_scoring)["source_hashes"]
    common = sorted(set(osrc) & set(nsrc))
    assert len(common) == 23 and all(osrc[key] == nsrc[key] for key in common)
    comparability = dict(
        identical_complete_eval_manifest=True,
        metadata_files={str(p): sha(p) for p in (old_manifest, new_manifest, old_scoring, new_scoring)},
        common_registered_scorer_sources={key: osrc[key] for key in common},
        basis="Identical saved request identities and shared scoring source hashes; no new geometry scoring.",
    )
    tables = {}
    for name, path in SOURCES.items():
        with path.open(encoding="utf-8-sig", newline="") as f:
            tables[name] = list(csv.DictReader(f))
    historical = tables["historical"]
    current = [r for r in tables["current"] if r["label"] in NEW]
    allrows = historical + current
    index = {(r["label"], r["stage"], r["request_id"]): r for r in allrows}
    assert len(index) == len(allrows) == 8 * 48 * 3
    reference = {r["request_id"]: r for r in current if r["label"] == NEW[0] and r["stage"] == STAGES[0]}
    identities = ("condition_id", "parent_id", "noise_index", "N", "K", "split", "closed_C", "part_type")
    for label in OLD + NEW:
        for stage in STAGES:
            rows = [r for r in allrows if r["label"] == label and r["stage"] == stage]
            assert {r["request_id"] for r in rows} == set(reference)
            for row in rows:
                assert all(row[k] == reference[row["request_id"]][k] for k in identities)
                assert row["scoring_definition_version"] == "hinge_v2_original_quality_bridge_role_fixed_parent_v1"
    summaries = []
    for label in OLD + NEW:
        for stage in STAGES:
            for split in ("TRAIN", "DEV", "ALL"):
                rows = [r for r in allrows if r["label"] == label and r["stage"] == stage
                        and (split == "ALL" or r["split"] == split)]
                summary = dict(label=label, stage=stage, split=split,
                               outputs=len(rows), parents=len({r["parent_id"] for r in rows}),
                               conditions=len({r["condition_id"] for r in rows}))
                for metric in METRICS:
                    vals = [number(r.get(metric)) for r in rows]
                    valid = [v for v in vals if v is not None]
                    summary[metric] = dict(parent_equal=parent_equal(rows, metric),
                                           valid_outputs=len(valid), NA_outputs=len(vals)-len(valid),
                                           sum=sum(valid) if valid else None)
                summaries.append(summary)
    pairs = []
    for new in NEW:
        for old in OLD:
            for stage in STAGES:
                for split in ("TRAIN", "DEV", "ALL"):
                    requests = [rid for rid, r in reference.items() if split == "ALL" or r["split"] == split]
                    item = dict(new=new, old=old, stage=stage, split=split, requests=len(requests), metrics={})
                    for metric in METRICS:
                        changes = []
                        pseudo = []
                        per_parent = defaultdict(list)
                        for rid in requests:
                            n, o = index[new, stage, rid], index[old, stage, rid]
                            nv, ov = number(n.get(metric)), number(o.get(metric))
                            if nv is None or ov is None:
                                continue
                            delta = nv - ov
                            changes.append(delta)
                            pseudo.append(dict(parent_id=n["parent_id"], condition_id=n["condition_id"], delta=delta))
                            per_parent[n["parent_id"]].append(delta)
                        item["metrics"][metric] = dict(
                            paired_outputs=len(changes),
                            NA_outputs=len(requests)-len(changes),
                            delta_parent_equal=parent_equal(pseudo, "delta"),
                            median_output_delta=statistics.median(changes) if changes else None,
                            lower_outputs=sum(d < -1e-12 for d in changes),
                            equal_outputs=sum(abs(d) <= 1e-12 for d in changes),
                            higher_outputs=sum(d > 1e-12 for d in changes),
                            per_parent_delta={p: parent_equal([r for r in pseudo if r["parent_id"] == p], "delta")
                                              for p in sorted(per_parent)},
                        )
                    pairs.append(item)
    result = dict(status="PASS", source_tables={str(p): sha(p) for p in SOURCES.values()},
                  script_sha256=sha(Path(__file__)), scalar_rows=1152, models=8,
                  matching_requests=48, matching_identity_fields=identities,
                  stages=STAGES, aggregation="noise mean -> condition mean -> parent equal; metric-specific NA retained",
                  comparability=comparability, model_calls=0, scoring_calls=0,
                  summaries=summaries, paired_comparisons=pairs)
    (HERE / "historical_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 同一固定48请求的历史与本轮结果复算", "", "比例列为百分比；其余为父等权均值。Native 的 seam→bulk 不适用。", ""]
    fields = METRICS[:4] + ["C_F_valid_edge_count", "deleted_free_faces", "actual_free_faces",
                           "nonmanifold_edges", "hinge_diag_isolated_1face_edge_components", "free_total_surface_area"]
    for split in ("DEV", "TRAIN"):
        for stage in STAGES:
            lines += [f"## {split} / {stage}", "", "| 模型 | FF% | CF% | I% | 小角% | CF合法边 | 删除free | 有效free | 非流形边 | 单面边分量 | free面积 | bulk≥30% |",
                      "|---|" + "---:|"*11]
            for row in summaries:
                if row["split"] != split or row["stage"] != stage:
                    continue
                values = []
                for field in fields:
                    value = row[field]["parent_equal"]
                    values.append("NA" if value is None else f"{value*(100 if field in METRICS[:4] else 1):.4f}")
                reach = row["bulk_reach_ge_0_30_descriptive"]
                count = "NA" if not reach["valid_outputs"] else f"{int(reach['sum'])}/{reach['valid_outputs']}"
                lines.append("| " + " | ".join([row["label"]] + values + [count]) + " |")
            lines += [""]
    (HERE / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(status="PASS", scalar_rows=1152, models=8, matched_requests=48,
                          summaries=len(summaries), paired_comparisons=len(pairs), model_calls=0,
                          scoring_calls=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
