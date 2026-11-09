"""CPU-only loss tables, learning curves and progress from saved statistics.

No forward or metric rescoring. Training sufficient statistics are summed as
SSE/count; moving averages of batch objectives are explicitly labelled as such.
Fixed-probe and changing training examples never share one convergence series.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from .convergence import summarize_training_records, DEFAULT_FIELDS
from .controller import load_probe_history, read


VERSION = "bench_parts_loss_curves_cpu_v1"
ARMS = ("COMMON", "PLAIN", "SEAM")


def read_jsonl(path):
    return read_jsonl_snapshot(path)[0]


def read_jsonl_snapshot(path):
    """Hash exactly the bytes parsed, even while the append-only writer runs."""
    if not Path(path).exists():
        return [], None
    raw = Path(path).read_bytes()
    rows = []
    for index, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Incomplete/malformed JSONL line {index}: {path}; retry read after writer completes, never edit source")
            rows.append(row)
    return rows, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                  "bytes_parsed": len(raw), "scope": "Exact parsed append-only log snapshot; later appends are allowed"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, columns=None):
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def _metric_pairs(row):
    for name, alternatives in DEFAULT_FIELDS.items():
        for sse_key, count_key in alternatives:
            if sse_key in row and count_key in row:
                sse, count = row[sse_key], int(row[count_key])
                if count < 0 or sse is None or not math.isfinite(float(sse)) or float(sse) < 0:
                    raise ValueError("Invalid training sufficient statistics")
                if not count and float(sse) != 0:
                    raise ValueError("Nonzero SSE with no scalar coordinates")
                yield name, float(sse), count
                break


def collect(root):
    root = Path(root)
    table, updates, histories, progress, inputs = [], {}, {}, {}, []
    exposure_groups = []
    for arm in ARMS:
        folder = root/"training"/arm
        path = folder/"training_log.jsonl"
        records, receipt = read_jsonl_snapshot(path)
        if receipt:
            inputs.append(receipt)
        # One scientific update may have auxiliary events; only metric-bearing
        # rows are plotted. Duplicate update rows are an error, not averaged.
        rows = [r for r in records if "step" in r and "FM" in r]
        steps = [r["step"] for r in rows]
        if len(steps) != len(set(steps)) or steps != sorted(steps):
            raise ValueError("Training update metric cursors are duplicate/out of order: "+arm)
        updates[arm] = rows
        for row in rows:
            base = dict(kind="TRAIN_UPDATE", arm=arm, step=row["step"], split="TRAIN", part="ALL",
                        time_bin="ALL", source=str(path), aggregation="batch objective")
            table.append(dict(base, metric="FM", sse=None, scalar_count=row.get("free_coordinate_denominator"),
                              value=row["FM"], status="REPORTED_BATCH_OBJECTIVE"))
            for metric, sse, count in _metric_pairs(row):
                table.append(dict(base, metric=metric, sse=sse, scalar_count=count, value=sse/count if count else None,
                                  aggregation="sum(SSE)/sum(valid scalars)", status="DEFINED" if count else "NA_NO_COORDINATES"))
            for metric in ("occupancy_BCE", "generator_gradient_norm", "update_seconds", "input_seconds"):
                if metric in row:
                    table.append(dict(base, metric=metric, sse=None, scalar_count=None, value=row[metric],
                                      aggregation="reported update diagnostic", status="DEFINED" if row[metric] is not None else "NA_NOT_APPLICABLE"))
        histories[arm] = load_probe_history(root, arm)
        for probe in histories[arm]:
            for source in probe["source_paths"]:
                inputs.append({"path": source, "sha256": sha(source)})
            for split in ("train", "dev"):
                domain = probe.get(split, {})
                entries = [(metric, "ALL", domain.get(metric)) for metric in ("ALL_free", "bulk", "seam_tip")]
                entries += [("ALL_free", part, pair) for part, pair in domain.get("parts", {}).items()]
                for metric, part, pair in entries:
                    count = 0 if pair is None else int(pair["count"])
                    sse = None if pair is None else pair["sse"]
                    table.append(dict(kind="FIXED_PROBE", arm=arm, step=probe["step"], split=split.upper(), part=part,
                                      time_bin="ALL", metric=metric, sse=sse, scalar_count=count,
                                      value=sse/count if count else None, status="DEFINED" if count else "NA_NO_COORDINATES",
                                      source=probe["source_paths"][0], aggregation="sum(SSE)/sum(valid scalars)"))
            if probe.get("rows"):
                grouped = summarize_training_records(probe["rows"])
                for group in grouped["rows"]:
                    for metric in ("ALL_free", "bulk", "seam_tip"):
                        stat = group[metric]
                        table.append(dict(kind="FIXED_PROBE_PART_TIME", arm=arm, step=probe["step"], split=group["split"],
                                          part=group["part"], time_bin=group["time_bin"], metric=metric,
                                          sse=stat["sse"], scalar_count=stat["count"], value=stat["mse"], status=stat["status"],
                                          source=probe["source_paths"][0], aggregation="sum(SSE)/sum(valid scalars)"))
        exposure = folder/"exposure.jsonl"
        if exposure.exists():
            exposure_records, exposure_receipt = read_jsonl_snapshot(exposure)
            inputs.append(exposure_receipt)
            # Preserve windows so a whole-run average cannot conceal changing
            # exposure/time distribution. Windows are global 1000-update bins.
            completed_steps = set(steps)
            enriched = [dict(r, update_window=f"{((int(r['step'])-1)//1000)*1000+1}-{((int(r['step'])-1)//1000+1)*1000}")
                        for r in exposure_records if r.get("step") in completed_steps]
            aggregated = summarize_training_records(enriched, group_keys=("update_window", "split", "part", "time_bin"))
            for row in aggregated["rows"]:
                for metric in ("ALL_free", "bulk", "seam_tip"):
                    stat = row[metric]
                    exposure_groups.append(dict(arm=arm, update_window=row["update_window"], split=row["split"],
                                                part=row["part"], time_bin=row["time_bin"], metric=metric,
                                                sse=stat["sse"], scalar_count=stat["count"], mse=stat["mse"],
                                                defined_rows=stat["defined_rows"], NA_rows=stat["NA_rows"]))
        latest = rows[-100:]
        summaries = summarize_training_records([dict(r, split="TRAIN", part="ALL", time_bin="ALL") for r in latest]) if latest else None
        progress[arm] = dict(updates_logged=len(rows), last_global_step=steps[-1] if steps else None,
                             fixed_probe_steps=[p["step"] for p in histories[arm]],
                             last100_batch_FM_mean=math.fsum(r["FM"] for r in latest)/len(latest) if latest else None,
                             last100_FM_interpretation="mean of reported batch objectives; not a fixed-probe convergence statistic",
                             last100_scalar_weighted=summaries,
                             mean_update_seconds=math.fsum(r.get("update_seconds", 0) for r in latest)/len(latest) if latest else None)
    return dict(table=table, updates=updates, histories=histories, progress=progress,
                exposure_groups=exposure_groups, input_hashes=inputs)


def rolling_weighted(rows, metric, width=100):
    """Training windows: objective mean for FM, sufficient-statistic MSE otherwise."""
    output = []
    for end in range(len(rows)):
        window = rows[max(0, end-width+1):end+1]
        if metric in ("FM", "occupancy_BCE"):
            values = [r[metric] for r in window if r.get(metric) is not None]
            y = math.fsum(values)/len(values) if values else None
        else:
            pairs = [pair for r in window for pair in _metric_pairs(r) if pair[0] == metric]
            denominator = sum(p[2] for p in pairs)
            y = math.fsum(p[1] for p in pairs)/denominator if denominator else None
        output.append((rows[end]["step"], y))
    return output


def render(root, out, window=100):
    """Write a new snapshot directory; reject silent replacement of old figures."""
    root, out = Path(root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new snapshot output directory: "+str(out))
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); data = collect(root)
    write_csv(out/"loss.csv", data["table"],
              ["kind", "arm", "step", "split", "part", "time_bin", "metric", "value", "sse", "scalar_count", "status", "aggregation", "source"])
    write_csv(out/"training_by_part_time.csv", data["exposure_groups"],
              ["arm", "update_window", "split", "part", "time_bin", "metric", "sse", "scalar_count", "mse", "defined_rows", "NA_rows"])
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = {"COMMON": "#777777", "PLAIN": "#2463a1", "SEAM": "#bf5319"}
    last_visible_step = max([1]+[r["step"] for rows in data["updates"].values() for r in rows]
                            +[p["step"] for rows in data["histories"].values() for p in rows])
    top = (("FM", "Training batch FM (window mean)"),
           ("bulk", "Training bulk MSE (sum SSE / scalars)"),
           ("seam_tip", "Training seam-tip MSE (sum SSE / scalars)"))
    for ax, (metric, title) in zip(axes[0], top):
        for arm in ARMS:
            points = [(x, y) for x, y in rolling_weighted(data["updates"][arm], metric, window) if y is not None]
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], color=colors[arm], label=arm)
        ax.set_title(title); ax.set_xlabel("Global optimizer update"); ax.grid(alpha=.2)
        ax.set_xlim(0, last_visible_step)
        if ax.lines:
            ax.legend()
        else:
            ax.text(.5, .5, "NA: no applicable coordinates yet", ha="center", va="center", transform=ax.transAxes)
    for ax, metric in zip(axes[1], ("ALL_free", "bulk", "seam_tip")):
        for arm in ARMS:
            for split, style in (("TRAIN", "-"), ("DEV", "--")):
                rows = [r for r in data["table"] if r["kind"] == "FIXED_PROBE" and r["arm"] == arm and
                        r["metric"] == metric and r["split"] == split and r["part"] == "ALL" and r["value"] is not None]
                if rows:
                    ax.plot([r["step"] for r in rows], [r["value"] for r in rows], style, marker="o" if split == "TRAIN" else "^",
                            color=colors[arm], label=arm+" "+split)
        ax.set_title("Fixed probe "+metric+" (SSE/scalars)")
        ax.set_xlabel("Global optimizer update"); ax.grid(alpha=.2)
        ax.set_xlim(0, last_visible_step)
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, "NA: no applicable probe stratum yet", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(f"Bench parts: changing training stream vs fixed probes | training window={window}\n"
                 "Loss plateau is not a generated-quality or mathematical-convergence certificate", fontsize=12)
    fig.savefig(out/"loss.png", dpi=160); plt.close(fig)
    summary = dict(schema=VERSION, root=str(root), window_updates=window, progress=data["progress"],
                   input_hashes=data["input_hashes"], CSV_rows=len(data["table"]),
                   model_calls=0, GPU_calls=0, wall_seconds=time.perf_counter()-start,
                   outputs={p.name: sha(p) for p in (out/"loss.csv", out/"training_by_part_time.csv", out/"loss.png")})
    (out/"progress.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/r2_bench_parts_convergence_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    result = render(args.root, args.out, args.window)
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_hashes",)}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
