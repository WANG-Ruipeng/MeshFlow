"""Read-only evaluation contract for the registered 224-output Bench study."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .common import ROOT, OUT, read, resolve, sha
from r2.pilots.bench_parts_convergence_v1 import panel as P

LABELS = (
    "COMMON_S0", "OFFICIAL_S0", "A_S500", "B_S500", "C_S500",
    "A_S1500", "B_S1500", "C_S1500",
)
FINAL_LABELS = ("A_S1500", "B_S1500", "C_S1500")
PROCESS_LABELS = LABELS[:5]
OLD = ROOT / "experiments/r2_bench_parts_convergence_v1"


def verify(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError("Protected asset changed: " + str(path))


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def plan():
    value = read(OUT / "registration.json")
    labels = value["evaluation_labels"]
    if set(labels) != set(LABELS):
        raise ValueError("The registered eight model labels are required")
    return value


def label_spec(label):
    if label not in LABELS:
        raise ValueError("Unregistered evaluation label")
    spec = plan()["evaluation_labels"][label]
    expected = "full" if label in FINAL_LABELS else "process"
    if spec["subset"] != expected:
        raise ValueError("Wrong registered panel size")
    return spec


def requests(label=None):
    manifest = read(OUT / "eval_manifest.json")
    verify(manifest.get("protected", {}))
    rows = manifest["requests"]
    if len(rows) != 48 or len({q["request_id"] for q in rows}) != 48:
        raise ValueError("Original 48 unique requests required")
    if label is not None and label_spec(label)["subset"] == "process":
        ids = plan()["process_request_ids"]
        if len(ids) != 16 or len(set(ids)) != 16:
            raise ValueError("Exactly 16 registered process requests required")
        by = {q["request_id"]: q for q in rows}
        rows = [by[rid] for rid in ids]
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
    """No head is imported; source metadata is used only for C export."""
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
    return OUT / "evaluation" / label / ("raw_complete_" + label_spec(label)["subset"] + ".json")


def verify_raw(label):
    qs = requests(label)
    done = read(raw_completion(label))
    if done["status"] != "COMPLETE" or done["outputs"] != len(qs):
        raise ValueError("Incomplete RAW label: " + label)
    verify(done["files"])
    receipts = []
    for q in qs:
        path = OUT / "evaluation" / label / q["request_id"] / "raw/receipt.json"
        r = read(path)
        if (r["request_id"] != q["request_id"] or r["label"] != label
                or r["generator_forwards"] != 50 or r["D0_forwards"] != 0
                or r["raw_sha256"] != sha(path.parent / "raw.npz")):
            raise ValueError("Invalid RAW receipt: " + str(path))
        verify(r["files"])
        receipts.append(path)
    return qs, receipts


def all_raw_barrier():
    receipts = []
    for label in LABELS:
        _, rr = verify_raw(label)
        receipts.extend(rr)
    if len(receipts) != 224:
        raise ValueError("All 224 RAW must complete before any D0")
    return receipts


def cpu_check():
    import ast
    filenames = ("eval_plan.py", "sample.py", "post_d0.py", "evaluation.py", "gallery.py")
    for name in filenames:
        ast.parse(Path(__file__).with_name(name).read_text(encoding="utf-8"))
    rows = []
    for q in requests():
        lay, _, _ = cached_layout(q)
        rows.append(dict(request_id=q["request_id"], N=q["N"], K=q["K"],
                         boundary_edges=len(lay["occupied"]), closed_C=bool(q["closed_C"]),
                         anchor_identity=True, source_identity=True, noise_identity=True))
    sizes = {label: len(requests(label)) for label in LABELS}
    assert sum(sizes.values()) == 224
    result = dict(status="PASS", sizes=sizes, checks=rows, parsed=list(filenames),
                  head_forwards=0, generator_forwards=0, D0_forwards=0, new_noise=0)
    P.immutable(OUT / "evaluation_cpu_checks.json", result)
    return result


if __name__ == "__main__":
    print(cpu_check())
