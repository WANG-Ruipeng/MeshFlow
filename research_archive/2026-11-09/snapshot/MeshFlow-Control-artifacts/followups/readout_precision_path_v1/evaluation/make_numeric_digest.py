"""Pure CPU interpretation of saved four-cell CSVs. No geometric scoring."""
from pathlib import Path
import csv, hashlib, json, math
from collections import Counter

OUT = Path("/mnt/e/MeshFlow-Control-artifacts/followups/readout_precision_path_v1")
KEYS = (
    "SeamCov", "ConflictCov_C", "ExposedRetain", "D_sym2048",
    "GT_to_F_MSE2048_L0", "F_to_GT_MSE2048_L0", "fixed_query_MSE",
    "fixed_query_p95_L", "coverage_fraction", "CF_proper_affected_free_fraction",
    "FF_illegal_affected_free_fraction", "free_minimum_angle_lt5_fraction",
    "free_near_degenerate_fraction", "free_exact_zero_area_fraction",
    "EXACT_COORD_face_components", "EXACT_COORD_boundary_edges",
    "EXACT_COORD_nonmanifold_edges",
)
VALUES = ("BF16_ON", "BF16_OFF", "FP32_ON", "FP32_OFF")
CONTRASTS = ("branch_BF", "branch_FP", "precision_ON", "precision_OFF", "interaction")

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def number(v):
    if v in ("", None):
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None
def table(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
def percent_change(a, b):
    return None if b == 0 else 100 * (a-b) / b

rows = table(OUT / "rollout_metrics.csv")
paired = table(OUT / "paired_rollout_comparisons.csv")
assert len(rows) == 32 and len({(r["label"], r["request_id"]) for r in rows}) == 32
assert all(r["status"] == "PASS" for r in rows)
errors = []
equations_checked = 0
for r in paired:
    vals = [number(r[k]) for k in VALUES]
    if all(v is not None for v in vals):
        a,b,c,d = vals
        expected = (a-b,c-d,c-a,d-b,(c-d)-(a-b))
        for key,v in zip(CONTRASTS,expected):
            if not math.isclose(number(r[key]),v,rel_tol=1e-12,abs_tol=1e-14):
                errors.append({"model":r["model_label"],"request":r["request_id"],"metric":r["metric"],"key":key})
            equations_checked += 1
        for key in CONTRASTS:
            lo,hi = number(r[key+"_lower"]),number(r[key+"_upper"])
            if lo is not None and hi is not None and lo > hi + 1e-14:
                errors.append({"metric":r["metric"],"key":key,"reason":"reversed_interval"})
    elif any(number(r[k]) is not None for k in CONTRASTS):
        errors.append({"metric":r["metric"],"reason":"NA_not_propagated"})
assert not errors, errors
groups = []
lookup = {}
for model in ("B1000", "C1000"):
    for request in sorted({r["request_id"] for r in paired}):
        subset = [r for r in paired if r["model_label"]==model and r["request_id"]==request]
        metrics = {}
        for r in subset:
            if r["metric"] not in KEYS:
                continue
            item = {k:number(r[k]) for k in VALUES + CONTRASTS}
            item["status"] = r["status"]
            item["cell_statuses"] = {k:r[k+"_status"] for k in VALUES}
            item["contrast_intervals"] = {k:[number(r[k+"_lower"]),number(r[k+"_upper"])] for k in CONTRASTS}
            metrics[r["metric"]] = item
            lookup[(model,request,r["metric"])] = item
        groups.append({"model_label":model,"request_id":request,"panel":subset[0]["panel"],"metrics":metrics})
p1f = next(x for x in {r["request_id"] for r in rows} if "f596" in x)
bds = lookup[("B1000",p1f,"D_sym2048")]
bgt = lookup[("B1000",p1f,"GT_to_F_MSE2048_L0")]
bfg = lookup[("B1000",p1f,"F_to_GT_MSE2048_L0")]
bs = lookup[("B1000",p1f,"SeamCov")]
status = {}
for panel in ("P1","P2"):
    rr = [r for r in rows if r["panel"]==panel]
    status[panel] = {
        "outputs":len(rr),
        "SeamCov_status":dict(Counter(r.get("SeamCov_status","") for r in rr)),
        "ConflictCov_C_status":dict(Counter(r.get("ConflictCov_C_status","") for r in rr)),
        "ExposedRetain_status":dict(Counter(r.get("ExposedRetain_status","") for r in rr)),
    }
contracts = {}
for k in ("C_all51_bitwise","C_output_bitwise","NPZ_C_bitwise","PLY_C_bitwise","N_preserved","finite","actual_Gaussian_bitwise"):
    contracts[k] = dict(Counter(r.get(k,"") for r in rows))
summary = {
    "status":"PASS",
    "scope":"Saved scalar tables only; no rescore, model call, new threshold, ranking, or visual review.",
    "outputs":32,"old_cache":8,"new_scores":24,
    "cell_value_order":VALUES,
    "contrast_formulas":{
        "branch_BF":"BF16_ON - BF16_OFF",
        "branch_FP":"FP32_ON - FP32_OFF",
        "precision_ON":"FP32_ON - BF16_ON",
        "precision_OFF":"FP32_OFF - BF16_OFF",
        "interaction":"branch_FP - branch_BF",
    },
    "paired_rows":len(paired),
    "independent_scalar_equations_checked":equations_checked,
    "scalar_equation_errors":errors,
    "original_metric_statuses":status,
    "contracts_from_saved_scores":contracts,
    "P1_f596_B_FP_ON_vs_BF_ON":{
        "S_delta_pp":100*bs["precision_ON"],
        "D_sym_relative_change_pct":percent_change(bds["FP32_ON"],bds["BF16_ON"]),
        "GT_to_F_MSE_relative_change_pct":percent_change(bgt["FP32_ON"],bgt["BF16_ON"]),
        "F_to_GT_MSE_relative_change_pct":percent_change(bfg["FP32_ON"],bfg["BF16_ON"]),
        "coverage_delta_pp":100*lookup[("B1000",p1f,"coverage_fraction")]["precision_ON"],
        "interpretation":"D_sym decrease is driven by F-to-GT, while GT-to-F increases. This is not uniform source reconstruction improvement.",
    },
    "groups":groups,
    "limits":[
        "Only two P1 source parents and two hand-crafted P2 conditions, one frozen Gaussian each.",
        "OFF is a complete rollout of the same jointly trained checkpoint; it is not arm A or a retrained no-readout model.",
        "Changing precision affects the entire forward path, not only the residual addition.",
        "Paired four-cell effects are final-output trajectory contrasts, not same-state local derivatives.",
        "P1 S/E and source distances are reference-completion diagnostics, not unique-target validity or usability gates.",
        "P2 S/E/source distances remain not applicable; no cross-panel pooling.",
        "PARTIAL original control statuses remain recorded even when saved numerical bounds coincide.",
        "Indexed topology is exact-coordinate diagnostic only; no welding/repair/watertightness guarantee.",
        "AI visual review is separate; human usability is NOT_RUN.",
    ],
    "source_hashes":{
        str(OUT / "rollout_metrics.csv"):sha(OUT / "rollout_metrics.csv"),
        str(OUT / "paired_rollout_comparisons.csv"):sha(OUT / "paired_rollout_comparisons.csv"),
        str(Path(__file__)):sha(Path(__file__)),
    },
}
target = OUT / "evaluation" / "four_cell_numeric_digest.json"
target.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"path":str(target),"sha256":sha(target),"equations_checked":equations_checked,"contracts":contracts,"metric_statuses":status,"f596_B":summary["P1_f596_B_FP_ON_vs_BF_ON"]},ensure_ascii=False))
