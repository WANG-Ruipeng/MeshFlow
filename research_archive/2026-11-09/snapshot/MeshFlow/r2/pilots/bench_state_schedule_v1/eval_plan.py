"""Frozen 48-request panel; two new final labels and a read-only historical A."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .common import ROOT, OUT, read, resolve, sha
from r2.pilots.bench_parts_convergence_v1 import panel as P

LABELS = ("PERTURBED_STATE_S1500", "ASYNC_LEAD_SEAM_S1500")
FINAL_LABELS = LABELS
BASELINE_LABEL = "A_S1500"
BASELINE_ROOT = ROOT / "experiments/r2_bench_adaptation_strength_v1"
ALL_LABELS = (BASELINE_LABEL, *LABELS)
OLD = ROOT / "experiments/r2_bench_parts_convergence_v1"
ARM_BY_LABEL = {
    "PERTURBED_STATE_S1500": "PERTURBED_STATE",
    "ASYNC_LEAD_SEAM_S1500": "ASYNC_LEAD_SEAM",
}


def verify(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError("Protected asset changed: " + str(path))


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def plan():
    value = read(OUT / "registration.json")
    if set(value["evaluation_labels"]) != set(LABELS):
        raise ValueError("Exactly two newly generated endpoint labels required")
    return value


def label_spec(label):
    if label not in LABELS:
        raise ValueError("Historical A is read-only; new endpoint label required")
    spec = plan()["evaluation_labels"][label]
    if (spec["arm"], spec["subset"], int(spec["local_step"]), int(spec["global_step"])) != (
            ARM_BY_LABEL[label], "full", 1500, 13500):
        raise ValueError("Unexpected frozen final checkpoint or panel")
    expected = OUT / "training" / spec["arm"] / "generator_step1500.pt"
    if resolve(spec["generator_path"]).resolve() != expected.resolve():
        raise ValueError("Generator export must belong to the registered new arm")
    return spec


def root_for(label):
    if label == BASELINE_LABEL:
        return BASELINE_ROOT
    if label in LABELS:
        return OUT
    raise ValueError("Unknown model label")


def requests(label=None):
    if label is not None and label not in ALL_LABELS:
        raise ValueError("Unknown requested label")
    manifest = read(OUT / "eval_manifest.json")
    baseline = read(BASELINE_ROOT / "eval_manifest.json")
    if manifest != baseline:
        raise ValueError("Panel must exactly reproduce the historical A manifest")
    verify(manifest.get("protected", {}))
    rows = manifest["requests"]
    if len(rows) != 48 or len({q["request_id"] for q in rows}) != 48:
        raise ValueError("48 original unique requests required")
    return rows


def source_assets(q):
    for key in ("inference_npz", "metadata_npz", "source_npz", "noise_npz"):
        digest = q.get(key + "_sha256")
        if digest is None and key == "noise_npz":
            digest = q["noise_sha256"]
        if digest is None or sha(q[key]) != digest:
            raise ValueError("Input identity differs: " + key)
    a = arrays(q["inference_npz"])
    core = {k: a[k] for k in ("C_vertices", "C_faces_original", "C_faces_model")}
    core["total_N"] = int(a["N"])
    if core["total_N"] != q["N"] or len(core["C_faces_model"]) != q["K"]:
        raise ValueError("Input face count changed")
    return core, arrays(q["source_npz"])


def cached_layout(q):
    """Only cached C-only predictions enter generation; source is export metadata."""
    from r2.pilots.bench_parts_convergence_v1.data import inference_layout
    from meshflow_control.source_winding import validate_source
    core, source = source_assets(q)
    if sha(q["old_layout_npz"]) != q["old_layout_sha256"]:
        raise ValueError("Cached OLD_S15000 layout changed")
    old = arrays(q["old_layout_npz"])
    layout = inference_layout(**core, occupied=old["occupied"])
    for key in ("anchor_ids", "edge_records", "seam_slots", "bulk_slots", "known_mask"):
        if not np.array_equal(layout[key], old[key]):
            raise ValueError("Layout reconstruction differs: " + key)
    validate_source(source, core["C_vertices"][core["C_faces_model"]])
    noise = arrays(q["noise_npz"])["z"]
    if noise.shape != (q["N"] - q["K"], 3, 3) or noise.dtype != np.float32:
        raise ValueError("Frozen Gaussian shape or precision differs")
    return layout, old["occupancy_logits"], source


def raw_completion(label):
    label_spec(label)
    return OUT / "evaluation" / label / "raw_complete_full.json"


def verify_raw(label):
    qs = requests(label)
    done = read(raw_completion(label))
    if done["status"] != "COMPLETE" or done["outputs"] != 48:
        raise ValueError("Incomplete new RAW label: " + label)
    verify(done["files"])
    receipts = []
    for q in qs:
        path = OUT / "evaluation" / label / q["request_id"] / "raw/receipt.json"
        r = read(path)
        if (r["request_id"] != q["request_id"] or r["label"] != label
                or r["generator_forwards"] != 50 or r["D0_forwards"] != 0
                or r["raw_sha256"] != sha(path.parent / "raw.npz")
                or r["noise_sha256"] != q["noise_sha256"]):
            raise ValueError("Invalid RAW receipt: " + str(path))
        verify(r["files"])
        receipts.append(path)
    return qs, receipts


def all_raw_barrier():
    receipts = []
    for label in LABELS:
        _, rr = verify_raw(label)
        receipts.extend(rr)
    if len(receipts) != 96:
        raise ValueError("All 96 new RAW outputs must complete before any new D0")
    return receipts


def cpu_check():
    import ast
    names = ("eval_plan.py", "sample.py", "post_d0.py", "evaluation.py", "collect.py", "gallery.py")
    for name in names:
        ast.parse(Path(__file__).with_name(name).read_text(encoding="utf-8"))
    rows = []
    for q in requests():
        layout, _, _ = cached_layout(q)
        rows.append(dict(request_id=q["request_id"], N=q["N"], K=q["K"],
                         active=int(layout["occupied"].sum()), source_identity=True,
                         noise_identity=True, cached_layout_identity=True))
    for label in LABELS:
        label_spec(label)
    value = dict(status="PASS", new_outputs=96, reused_outputs=48, checks=rows,
                 parsed=list(names), model_calls=0, new_noise=0, new_scores=0)
    P.immutable(OUT / "evaluation_cpu_checks.json", value)
    return value


if __name__ == "__main__":
    print(cpu_check())
