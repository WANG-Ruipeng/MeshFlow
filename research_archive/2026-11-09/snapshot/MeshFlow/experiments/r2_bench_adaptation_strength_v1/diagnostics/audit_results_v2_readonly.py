"""Independent standard-library comparison; no scorer/model code imported."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
OUT = BASE / "diagnostics/results_v2_independent_audit.json"


def resolve(value):
    value = str(value).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if value.startswith(prefix):
            return REPO / value[len(prefix):]
    return Path(value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def table(name):
    with (BASE / "results_v2" / (name + ".csv")).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


errors, comparisons, near_comparisons = [], 0, 0


def check(ok, kind, **facts):
    if not ok:
        errors.append(dict(kind=kind, **facts))


def equal_text(text, expected, *, approximate=False):
    global comparisons, near_comparisons
    comparisons += 1
    if expected is None:
        return text == ""
    if isinstance(expected, bool):
        return text == str(expected)
    if isinstance(expected, (int, float)):
        try:
            got = float(text)
        except (ValueError, TypeError):
            return False
        if approximate:
            near_comparisons += 1
            return math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)
        return got == expected
    if isinstance(expected, (dict, list)):
        try:
            return json.loads(text) == expected
        except (ValueError, TypeError):
            return False
    return text == expected


scope = read(BASE / "registration.json")
manifest = read(BASE / "eval_manifest.json")
completion = read(BASE / "results_v2/completion.json")
headers, csvrows = table("per_output")
lookup = {(r["label"], r["request_id"], r["stage"]): r for r in csvrows}
check(len(csvrows) == len(lookup) == 672, "unique_672_output_rows", rows=len(csvrows), unique=len(lookup))
process = set(scope["process_request_ids"])
stages = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
planned = [(label, q, stage)
           for label, spec in scope["evaluation_labels"].items()
           for q in manifest["requests"] if spec["subset"] == "full" or q["request_id"] in process
           for stage in stages]
check(set(lookup) == {(label, q["request_id"], stage) for label, q, stage in planned},
      "exact_registered_output_identities")
source_hashes, receipt_hashes, direct_rows = {}, {}, []
field_counts, none_values, boundary_different_from_C = collections.Counter(), 0, 0
for label, q, stage in planned:
    key = (label, q["request_id"], stage)
    folder = (BASE / "evaluation/scores" / label / q["request_id"] / "RAW" if stage == "RAW"
              else BASE / "cpu" / label / q["request_id"] / stage / "scores")
    path = folder / "row.json"
    source, receipt = read(path), read(folder / "complete.json")
    source_hashes[str(path)] = digest(path)
    receipt_hashes[str(folder / "complete.json")] = digest(folder / "complete.json")
    authentic = {str(resolve(p)): h for p, h in receipt["files"].items()}
    check(receipt["status"] == "PASS" and authentic.get(str(path)) == source_hashes[str(path)],
          "source_row_receipt_authentication", identity=key)
    row = lookup.get(key, {})
    field_counts[len(source)] += 1
    for field, expected in source.items():
        if field in ("label", "stage"):
            continue  # Explicit joint-table model/stage identity decoration.
        none_values += expected is None
        check(field in row and equal_text(row.get(field), expected), "source_field_value",
              identity=key, field=field, expected=expected, actual=row.get(field))
    metadata = dict(label=label, stage=stage, part_type=q["part_type"], closed_C=q["closed_C"],
                    C_boundary_edges=q["boundary_edges"],
                    seam_applicability="NA_CLOSED_C_NO_SEAM_INTERFACE" if q["closed_C"] else "APPLICABLE")
    for field, expected in metadata.items():
        check(equal_text(row.get(field), expected), "metadata_value",
              identity=key, field=field, expected=expected, actual=row.get(field))
    check("boundary_edges" in source and equal_text(row.get("boundary_edges"), source["boundary_edges"]),
          "whole_mesh_boundary_edges_preserved", identity=key)
    boundary_different_from_C += source["boundary_edges"] != q["boundary_edges"]
    direct_rows.append(dict(source, **metadata))

registered_source_hashes = {str(resolve(p)): h for p, h in completion["source_rows"].items()}
registered_receipt_hashes = {str(resolve(p)): h for p, h in completion["source_receipts"].items()}
check(registered_source_hashes == source_hashes, "collector_all_source_hashes")
check(registered_receipt_hashes == receipt_hashes, "collector_all_source_receipt_hashes")
metrics = sorted({k for row in direct_rows for k, v in row.items()
                  if isinstance(v, (float, int)) and not isinstance(v, bool)}
                 - {"N", "K", "noise_index", "C_boundary_edges"})
check(metrics == completion["metric_fields"], "all_numeric_metric_fields",
      independent_count=len(metrics), registered_count=len(completion["metric_fields"]))
check("boundary_edges" in metrics and "C_boundary_edges" not in metrics,
      "whole_boundary_metric_vs_C_metadata")


def independent_groups(rows, keys):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for identity, rr in grouped.items():
        result = dict(zip(keys, identity), rows=len(rr))
        for field in metrics:
            values = [r[field] for r in rr if isinstance(r.get(field), (int, float))
                      and not isinstance(r.get(field), bool) and math.isfinite(r[field])]
            result[field] = statistics.mean(values) if values else None
            result[field + "__valid_n"] = len(values)
        out.append(result)
    return out


aggregate_counts = {}
current = direct_rows
for name, keys in (
    ("per_condition", ("label", "stage", "split", "parent_id", "condition_id")),
    ("per_parent", ("label", "stage", "split", "parent_id")),
    ("parent_macro_summary", ("label", "stage", "split")),
):
    current = independent_groups(current, keys)
    _, saved = table(name)
    byid = {tuple(r[k] for k in keys): r for r in saved}
    check(len(saved) == len(byid) == len(current), "aggregate_unique_size", table=name)
    for row in current:
        identity = tuple(row[k] for k in keys)
        observed = byid.get(identity, {})
        for field, expected in row.items():
            check(equal_text(observed.get(field), expected, approximate=field in metrics),
                  "aggregate_value", table=name, identity=identity, field=field,
                  expected=expected, actual=observed.get(field))
    aggregate_counts[name] = len(current)

_, pairrows = table("paired_final")
pairmap = {(r["reference"], r["candidate"], r["request_id"], r["stage"]): r for r in pairrows}
check(len(pairrows) == len(pairmap) == 432, "paired_432_unique_rows")
direct = {(r["label"], r["request_id"], r["stage"]): r for r in direct_rows}
contrasts = (("A_S1500", "B_S1500", "PRIMARY_LR_B_MINUS_A"),
             ("B_S1500", "C_S1500", "PRIMARY_INITIALIZATION_C_MINUS_B"),
             ("A_S1500", "C_S1500", "ADDITIONAL_C_MINUS_A"))
for a, b, contrast in contrasts:
    for q in manifest["requests"]:
        for stage in stages:
            key = (a, b, q["request_id"], stage)
            observed = pairmap.get(key, {})
            check(observed.get("contrast") == contrast, "pair_contrast", identity=key)
            for field in metrics:
                av = direct[a, q["request_id"], stage].get(field)
                bv = direct[b, q["request_id"], stage].get(field)
                delta = float(bv - av) if av is not None and bv is not None else None
                for suffix, expected in (("__reference", av), ("__candidate", bv), ("__delta", delta)):
                    check(equal_text(observed.get(field + suffix), expected),
                          "pair_field_value", identity=key, field=field + suffix,
                          expected=expected, actual=observed.get(field + suffix))

table_hashes = {}
for name in ("per_output", "per_condition", "per_parent", "parent_macro_summary", "paired_final"):
    path = BASE / "results_v2" / (name + ".csv")
    table_hashes[str(path)] = digest(path)
check({str(resolve(p)): h for p, h in completion["files"].items()} == table_hashes,
      "all_collected_table_hashes")
check("torch" not in sys.modules and "numpy" not in sys.modules, "standard_library_only")
report = dict(
    schema="BENCH_ADAPTATION_RESULTS_V2_INDEPENDENT_VALUE_AUDIT_V1",
    status="PASS" if not errors else "FAIL", source_rows=672, per_output_rows=len(csvrows),
    source_field_count_distribution=dict(field_counts), source_None_values_checked=none_values,
    original_source_fields_compared="Every source field except intentionally decorated label/stage; all score values exact CSV round-trip equality",
    additional_metadata_checked=True, whole_mesh_boundary_edges_matches_source=True if not errors else None,
    C_boundary_edges_matches_manifest=True if not errors else None,
    source_whole_boundary_differs_from_input_C_rows=boundary_different_from_C,
    numeric_metric_fields=len(metrics), aggregate_rows_checked=aggregate_counts,
    paired_rows_checked=len(pairrows), comparison_count=comparisons,
    aggregate_numeric_tolerance=dict(relative=1e-12, absolute=1e-12, comparisons=near_comparisons),
    errors=errors, model_loads=0, model_calls=0, scorer_imports=0, scoring_calls=0,
    geometry_changes=0, source_files_modified=0,
    independent_implementation="This script imports only Python standard library and does not call collect_v2 or original scorers",
    audit_script_sha256=digest(__file__), registration_sha256=digest(BASE / "registration.json"),
    eval_manifest_sha256=digest(BASE / "eval_manifest.json"),
    v2_completion_sha256=digest(BASE / "results_v2/completion.json"),
    table_hashes=table_hashes, source_row_hashes=source_hashes, source_receipt_hashes=receipt_hashes,
)
with OUT.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
print(json.dumps({k: v for k, v in report.items()
                  if k not in ("source_row_hashes", "source_receipt_hashes", "table_hashes", "errors")},
                 ensure_ascii=False, indent=2))
if errors:
    print(json.dumps(errors[:20], ensure_ascii=False, indent=2))
    raise SystemExit(1)
