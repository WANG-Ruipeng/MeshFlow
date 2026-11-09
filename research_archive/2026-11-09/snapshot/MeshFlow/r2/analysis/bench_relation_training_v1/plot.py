"""CPU-only plots of the completed, fixed-budget RELATION training ledger.

Reads authenticated offline analysis tables, not model weights or raw meshes.
No pilot imports, neural-network calls, new scores, smoothing, checkpoint
selection, convergence judgement, or cross-training-pool loss comparison.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / "experiments/r2_bench_relation_training_v1/analysis_relation_v1"
SCHEMA = "BENCH_RELATION_TRAINING_CURVES_V1"
MODEL = "RELATION_V1_S1500"
SPECS = (
    ("all_free", "All valid free velocity", "all_free_velocity_sse", "all_free_scalar_count", "all_free_velocity_MSE", "#2166ac"),
    ("bulk", "Bulk velocity", "bulk_velocity_sse", "bulk_scalar_count", "bulk_velocity_MSE", "#17826c"),
    ("seam_tip", "Seam-tip velocity", "seam_tip_velocity_sse", "seam_tip_scalar_count", "seam_tip_velocity_MSE", "#a34b0a"),
    ("one_step_endpoint", "One-step endpoint (all valid free)", "one_step_endpoint_sse", "one_step_endpoint_scalar_count", "one_step_endpoint_MSE", "#8055a0"),
)


def resolve(value):
    s = str(value).replace("\\", "/")
    if os.name == "nt" and s.startswith("/mnt/"):
        return Path(s[5].upper() + ":/" + s[7:])
    if os.name != "nt" and len(s) > 2 and s[1] == ":":
        return Path("/mnt/" + s[0].lower() + s[2:])
    return Path(s)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError("Refusing to overwrite different plot artifact: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def finite(value, field):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite original analysis value: " + field)
    return result


def authenticated_inputs(analysis):
    paths = [analysis / name for name in ("completion.json", "training_windows.csv", "training_summary.json")]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return {"status": "WAITING_FOR_COMPLETED_ANALYSIS", "missing": missing}
    complete, summary = read(paths[0]), read(paths[2])
    if complete.get("status") != "COMPLETE" or complete.get("schema") != "BENCH_RELATION_OFFLINE_ANALYSIS_V1":
        raise ValueError("Expected completed versioned relation-analysis receipt")
    authenticated = {resolve(path).resolve(): digest for path, digest in complete["files"].items()}
    for path in paths[1:]:
        if authenticated.get(path.resolve()) != sha(path):
            raise ValueError("Analysis-table hash mismatch: " + str(path))
    if summary.get("updates") != 1500:
        raise ValueError("Expected the fixed 1500-update experiment")
    with paths[1].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 15:
        raise ValueError("Expected fifteen original 100-update windows")
    series, coverage = [], {}
    for index, row in enumerate(rows):
        first, last, updates = (int(row[name]) for name in ("first_step", "last_step", "updates"))
        if (first, last, updates) != (100 * index + 1, 100 * (index + 1), 100):
            raise ValueError("Window boundaries differ from registered 100-update bins")
        for key, title, numerator, denominator, metric, _ in SPECS:
            # Endpoint aliases retain the original aggregation; never infer it
            # from mean velocity MSE or a made-up average diffusion time.
            fields = (numerator, denominator, metric)
            if key == "one_step_endpoint" and not all(row.get(name, "") != "" for name in fields):
                aliases = ("endpoint_sse", "free_coordinate_denominator", "endpoint_MSE")
                if all(row.get(name, "") != "" for name in aliases):
                    fields = aliases
            available = all(row.get(name, "") != "" for name in fields)
            item = dict(model=MODEL, metric=key, title=title, first_step=first,
                        last_step=last, updates=updates, bin_midpoint=(first + last) / 2,
                        status="PASS" if available else "NA_NOT_IN_COMPLETED_ANALYSIS",
                        numerator_field=fields[0], scalar_denominator_field=fields[1], mse_field=fields[2],
                        numerator=None, scalar_count=None, mse=None)
            if available:
                sse = finite(row[fields[0]], fields[0])
                count_value = finite(row[fields[1]], fields[1])
                count = int(count_value)
                mse = finite(row[fields[2]], fields[2])
                if count_value != count or count < 0 or sse < 0:
                    raise ValueError("Invalid scalar count or SSE: " + key)
                if not count:
                    raise ValueError("Recorded MSE has zero scalar denominator: " + key)
                if not math.isclose(mse, sse / count, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("Stored MSE is not original SSE/scalar_count: " + key)
                item.update(numerator=sse, scalar_count=count, mse=mse)
            elif key != "one_step_endpoint":
                raise ValueError("Required velocity curve is missing: " + key)
            series.append(item)
    for key, *_ in SPECS:
        points = [item for item in series if item["metric"] == key]
        valid = [item for item in points if item["status"] == "PASS"]
        coverage[key] = dict(valid_bins=len(valid), missing_bins=len(points) - len(valid),
                             scalar_count=sum(item["scalar_count"] for item in valid),
                             scalar_count_min=min((item["scalar_count"] for item in valid), default=None),
                             scalar_count_max=max((item["scalar_count"] for item in valid), default=None))
    return dict(status="READY", summary=summary, series=series, coverage=coverage,
                input_hashes={str(path): sha(path) for path in paths})


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def render(analysis=DEFAULT, check_inputs=False):
    analysis = resolve(analysis).resolve()
    data = authenticated_inputs(analysis)
    if data["status"] != "READY":
        return data
    if check_inputs:
        return dict(status="READY", model=MODEL, bins=15, updates=1500,
                    metric_coverage=data["coverage"], model_calls=0, scoring_calls=0)
    dest = analysis / "training_curve_plots_v1"
    receipt = dest / "completion.json"
    if receipt.exists():
        old = read(receipt)
        if old["source_sha256"] != sha(__file__) or old["input_hashes"] != data["input_hashes"]:
            raise ValueError("Existing curve receipt has a different source or input identity")
        for path, digest in old["files"].items():
            if sha(resolve(path)) != digest:
                raise ValueError("Completed curve artifact changed: " + path)
        return old

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    for ax, spec in zip(axes.flat, SPECS):
        key, title, *_fields, color = spec
        points = [row for row in data["series"] if row["metric"] == key]
        valid = [row for row in points if row["status"] == "PASS"]
        if valid:
            ax.plot([row["bin_midpoint"] for row in points],
                    [row["mse"] if row["status"] == "PASS" else math.nan for row in points],
                    color=color, marker="o", linewidth=1.8, markersize=3.4)
            info = data["coverage"][key]
            counts = (f"Scalar denominator / bin: {info['scalar_count_min']:,}"
                      f" to {info['scalar_count_max']:,}\n"
                      f"Total scalar observations: {info['scalar_count']:,}; valid bins {len(valid)}/15")
            ax.text(0.98, 0.97, counts, ha="right", va="top", transform=ax.transAxes,
                    fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=.88))
            ax.set_ylim(bottom=0)
        else:
            ax.text(.5, .5, "NA: endpoint SSE / scalar count\nnot present in completed analysis", ha="center",
                    va="center", transform=ax.transAxes, color="#555555")
        ax.set(title=title, xlabel="Local optimizer updates (100-update bin midpoint)",
               ylabel="SSE / valid scalar count", xlim=(0, 1500))
        ax.set_xticks([0, 300, 600, 900, 1200, 1500])
        ax.grid(alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Bench RELATION V1: fixed 1,500-update training curve", y=.975, fontsize=16)
    note = (
        "Each point = sum of logged SSE / sum of its valid scalar denominator across 100 updates; no smoothing.\n"
        "One-step endpoint uses the logged training extrapolation; it is not a free-generation rollout metric.\n"
        "This arm only. Different condition pools are not a same-task loss comparison. Fixed budget completed; no convergence claim."
    )
    fig.text(.06, .035, note, fontsize=9, color="#444444", va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=.075, right=.98, top=.90, bottom=.19, hspace=.43, wspace=.23)
    outputs = {}
    try:
        for extension in ("png", "pdf"):
            buffer = io.BytesIO()
            metadata = {"Software": SCHEMA} if extension == "png" else {
                "Creator": SCHEMA, "Title": "Bench RELATION V1 fixed-budget training curves",
                "CreationDate": None, "ModDate": None,
            }
            fig.savefig(buffer, format=extension, dpi=180, metadata=metadata)
            path = dest / ("training_curves." + extension)
            write_new(path, buffer.getvalue())
            outputs[str(path)] = sha(path)
    finally:
        plt.close(fig)
    table = dest / "plotted_scalar_bins.csv"
    write_new(table, csv_bytes(data["series"]))
    outputs[str(table)] = sha(table)
    notes = dest / "definitions.json"
    write_new(notes, json_bytes(dict(
        model=MODEL, fixed_budget_updates=1500, window_updates=100,
        aggregation="Each curve is original sum(SSE)/sum(valid scalar count), not an unweighted mean of per-update MSE.",
        units="Original logged squared-coordinate error units. No normalization or percentage conversion.",
        endpoint="Original logged one-step extrapolation Y_hat=X_t+(1-t)*v. This uses training interpolation states, not autonomous rollout.",
        scalar_denominators="Retained per bin in plotted_scalar_bins.csv; role denominators are not assumed equal.",
        objective="The per-update objective may use a different aggregation; it is not relabelled as this pooled MSE.",
        missing="Missing endpoint entries remain NA; never reconstructed from mean velocity MSE.",
        source_metric_note=data["summary"].get("metric_note"), metric_coverage=data["coverage"],
        cross_pool_comparison=False, convergence_claim=False, checkpoint_selection=False,
        model_calls=0, scoring_calls=0,
    )))
    outputs[str(notes)] = sha(notes)
    for path, digest in data["input_hashes"].items():
        if sha(resolve(path)) != digest:
            raise ValueError("Read-only source changed while rendering: " + path)
    result = dict(schema=SCHEMA, status="COMPLETE", model=MODEL, updates=1500, bins=15,
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  input_hashes=data["input_hashes"], source_sha256=sha(__file__), files=outputs,
                  metric_coverage=data["coverage"], model_calls=0, scoring_calls=0,
                  original_inputs_modified=False, convergence_claim=False)
    write_new(receipt, json_bytes(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", default=str(DEFAULT))
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    result = render(args.analysis, args.check_inputs)
    print(json.dumps({key: value for key, value in result.items() if key not in {"files", "input_hashes"}},
                     ensure_ascii=False, indent=2, allow_nan=False))
