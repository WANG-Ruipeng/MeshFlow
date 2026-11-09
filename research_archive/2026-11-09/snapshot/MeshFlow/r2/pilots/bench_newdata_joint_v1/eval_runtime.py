"""Immutable runtime contracts for the 28-request new-data joint experiment.

Inference reads C, requested N, saved noise and the frozen head only. Source
metadata is used only to restore the input's original winding at export.
"""
from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
from .common import ROOT, OUT, read, resolve, sha
from r2.pilots.bench_parts_convergence_v1 import panel as P

BASELINE_LABEL = "RELATION_V1_S1500"
NEW_LABEL = "NEWDATA_JOINT_S1500"
LABELS = (BASELINE_LABEL, NEW_LABEL)
FINAL_LABELS = LABELS
ALL_LABELS = LABELS
OLD_ROOT = ROOT / "experiments/r2_bench_relation_training_v1"
HEAD_ROOT = ROOT / "experiments/r2_bench_three_family_v1/production_contract"
ARM_BY_LABEL = {BASELINE_LABEL: "RELATION_V1_SEAM", NEW_LABEL: "NEW_DATA_JOINT"}


def verify(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError("Protected asset changed: " + str(path))


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def label_spec(label):
    if label not in LABELS:
        raise ValueError("Unknown registered evaluation label")
    plan = read(OUT / "registration.json")
    if set(plan["evaluation_labels"]) != set(LABELS):
        raise ValueError("Exactly the frozen reference and new endpoint required")
    spec = plan["evaluation_labels"][label]
    if (spec["arm"], spec["subset"], int(spec["local_step"]), int(spec["global_step"])) != (
            ARM_BY_LABEL[label], "full", 1500, 13500):
        raise ValueError("Unexpected frozen endpoint")
    root = OLD_ROOT if label == BASELINE_LABEL else OUT
    expected = root / "training" / spec["arm"] / "generator_step1500.pt"
    if resolve(spec["generator_path"]).resolve() != expected.resolve():
        raise ValueError("Generator is outside the registered endpoint")
    return spec


def root_for(label):
    if label not in LABELS:
        raise ValueError("Unknown label")
    # Reused reference products are copied with explicit zero-new-call receipts.
    return OUT


def requests(label=None):
    if label is not None and label not in LABELS:
        raise ValueError("Unknown evaluation label")
    manifest = read(OUT / "eval_manifest.json")
    verify(manifest.get("protected", {}))
    rows = manifest["requests"]
    if len(rows) != 28 or len({q["request_id"] for q in rows}) != 28:
        raise ValueError("Exactly 28 unique frozen requests required")
    conditions = {q["condition_id"] for q in rows}
    if len(conditions) != 14 or any(sum(q["condition_id"] == c for q in rows) != 2 for c in conditions):
        raise ValueError("Exactly 14 conditions with two frozen noises required")
    for q in rows:
        if Path(q["request_id"]).name != q["request_id"]:
            raise ValueError("Unsafe request ID")
    return rows


def reuse_spec(q, label):
    if label != BASELINE_LABEL or not q.get("baseline_reuse"):
        return None
    cached = q["baseline_cached"]
    protected = read(OUT / "eval_manifest.json")["protected"]
    # Textual host-path normalization only. Path.resolve() would stat every
    # ancestor of all 768 protected files through WSL/NTFS for each request.
    # sha()/verify() still check bytes; the lookup does not need symlink I/O.
    normalized = {}
    for path, digest in protected.items():
        key = resolve(path).as_posix()
        if key in normalized and normalized[key] != digest:
            raise ValueError("Conflicting frozen identity for normalized path")
        normalized[key] = digest
    def frozen_hash(path):
        key = resolve(path).as_posix()
        if key not in normalized:
            raise ValueError("Historical receipt must be frozen in manifest: " + str(path))
        return normalized[key]
    return dict(root=cached["root"], request_id=q["request_id"],
        raw_receipt_sha256=frozen_hash(cached["raw_receipt"]),
        post_receipt_sha256=frozen_hash(cached["post_receipt"]))


def new_requests(label):
    return [q for q in requests(label) if not reuse_spec(q, label)]


def source_assets(q):
    for key in ("inference_npz", "metadata_npz", "source_npz", "noise_npz"):
        digest = q.get(key + "_sha256")
        if digest is None and key == "noise_npz":
            digest = q["noise_sha256"]
        if digest is None or sha(q[key]) != digest:
            raise ValueError("Frozen input identity differs: " + key)
    a = arrays(q["inference_npz"])
    core = {k: a[k] for k in ("C_vertices", "C_faces_original", "C_faces_model")}
    core["total_N"] = int(a["N"])
    if core["total_N"] != q["N"] or len(core["C_faces_model"]) != q["K"]:
        raise ValueError("Input N/K differs")
    noise = arrays(q["noise_npz"])["z"]
    if noise.shape != (q["N"] - q["K"], 3, 3) or noise.dtype != np.float32:
        raise ValueError("Frozen standard Gaussian shape or precision differs")
    from meshflow_control.source_winding import validate_source
    source = arrays(q["source_npz"])
    validate_source(source, core["C_vertices"][core["C_faces_model"]])
    return core, source


def raw_completion(label):
    label_spec(label)
    return OUT / "evaluation" / label / "raw_complete_full.json"


def verify_raw(label):
    qs = requests(label)
    done = read(raw_completion(label))
    if done["status"] != "COMPLETE" or done["outputs"] != 28:
        raise ValueError("Incomplete RAW wave")
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
        if bool(reuse_spec(q, label)) != (r.get("execution") == "REUSED"):
            raise ValueError("Reference reuse provenance differs")
        verify(r["files"])
        receipts.append(path)
    return qs, receipts


def all_raw_barrier():
    receipts = []
    for label in LABELS:
        _, rr = verify_raw(label)
        receipts.extend(rr)
    if len(receipts) != 56:
        raise ValueError("All 56 paired RAW products must exist before any new D0")
    return receipts


def copy_reference(q, kind):
    """Copy preserved small artifacts; never replay a historical model call."""
    spec = reuse_spec(q, BASELINE_LABEL)
    if not spec:
        raise ValueError("This request has no registered reference reuse")
    origin = resolve(spec["root"]) / "evaluation" / BASELINE_LABEL / spec["request_id"] / kind
    if resolve(spec["root"]).resolve() != OLD_ROOT.resolve():
        raise ValueError("Reference origin is not the frozen RELATION experiment")
    receipt_path = origin / "receipt.json"
    expected = spec[kind + "_receipt_sha256"]
    if sha(receipt_path) != expected:
        raise ValueError("Historical receipt changed")
    old = read(receipt_path)
    verify(old["files"])
    if (old["request_id"] != q["request_id"] or old["label"] != BASELINE_LABEL
            or old[kind + "_sha256"] != sha(origin / (kind + ".npz"))):
        raise ValueError("Historical request/mesh identity differs")
    if kind == "raw" and old["noise_sha256"] != q["noise_sha256"]:
        raise ValueError("Historical Gaussian identity differs")
    dest = OUT / "evaluation" / BASELINE_LABEL / q["request_id"] / kind
    if (dest / "receipt.json").exists():
        current = read(dest / "receipt.json")
        if current.get("origin_receipt_sha256") != expected or current.get("execution") != "REUSED":
            raise ValueError("Reused producer provenance differs")
        verify(current["files"])
        return current
    if dest.exists():
        raise RuntimeError("Partial reference copy retained: " + str(dest))
    dest.mkdir(parents=True)
    # Keep the original large trajectory in its protected history directory.
    for path in origin.iterdir():
        if path.is_file() and path.name not in ("receipt.json", "trajectory.npz"):
            shutil.copy2(path, dest / path.name)
    P.immutable(dest / "origin_receipt.json", old)
    receipt = dict(old)
    receipt.update(execution="REUSED", origin_receipt=str(receipt_path),
        origin_receipt_sha256=expected, new_generator_forwards=0, new_D0_forwards=0,
        new_GEO_calls=0, new_head_forwards=0,
        files={str(p): sha(p) for p in dest.iterdir() if p.is_file()})
    P.immutable(dest / "receipt.json", receipt)
    return receipt
