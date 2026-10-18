"""Portable CPU replay of the frozen HINGE V2 postprocessing stages.

Requires the actual HINGE_OPP015 indexed mesh, its completion receipt and pair
cache, and the same original PRE_GEO/RAW arrays. Plain PLY/OBJ and an arbitrary
triangle soup are insufficient: original corner identities and seam roles are
part of this experiment's contract. No source Y, model or new query is used.

This adapter sets only the archived legacy-math directory. It does not replace
or edit a scientific function. Run in a fresh Python process (prefer -B -I).
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ARCHIVE = Path(__file__).resolve().parent
VERSION = "hinge_v2_explicit_arrays_portable_replay_v1"
MODES = ("midpoint", "alpha", "hole", "alpha-hole")
REFERENCE_LENGTH = 9.208138264587333  # sqrt(3)*2/.3762, same fixed model coordinates.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source_manifest():
    """Verify the byte-preserved source closure before importing it."""
    manifest_path = ARCHIVE / "source_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Missing source_manifest.json; finish archive construction first")
    manifest = read(manifest_path)
    records = manifest["source_files"]
    checked = {}
    for row in records:
        relative = row["archive_relative_path"]
        path = (ARCHIVE / relative).resolve()
        if not path.is_relative_to(ARCHIVE / "source"):
            raise ValueError("Scientific source outside archived source/: " + relative)
        if not path.is_file() or sha(path) != row["sha256"]:
            raise ValueError("Missing or changed archived source: " + relative)
        checked[str(path)] = row["sha256"]
    if not checked:
        raise ValueError("Empty source manifest")
    return manifest_path, checked


def archived_modules(allowed):
    modules = {}
    for name, module in sys.modules.items():
        if not (name == "r2" or name.startswith("r2.")
                or name.startswith("_partial_binding_")
                or name.startswith("_r2_user_pilot_original_evaluation")):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if str(path) not in allowed or sha(path) != allowed[str(path)]:
            raise RuntimeError("Scientific import escaped verified archive: " + str(path))
        modules[name] = dict(path=path.relative_to(ARCHIVE).as_posix(), sha256=allowed[str(path)])
    if "torch" in sys.modules:
        raise RuntimeError("CPU-only replay unexpectedly imported torch")
    return modules


def validate_baseline(base, pre, raw, provenance, paths, np):
    """Validate identities without inventing a new mesh or rescanning its pairs."""
    required = ("vertices", "faces", "output", "C", "free", "input_faces", "known_faces",
                "source_face_ids", "source_corner_ids", "raw_slot_ids", "corner_vertex_ids",
                "raw_seam_slot_mask", "output_anchor_ids", "original_POST_vertex_to_hinge_vertex")
    missing = [key for key in required if key not in base]
    if missing:
        raise ValueError("Missing indexed HINGE identities: " + ", ".join(missing))
    if provenance.get("status") != "COMPLETE" or provenance.get("row", {}).get("stage") != "HINGE_OPP015":
        raise ValueError("Completion receipt must identify the frozen HINGE_OPP015 baseline")
    old_summary = provenance.get("summary", {})
    if (old_summary.get("require_opposite_winding") is not True
            or old_summary.get("delta_fraction") != .015
            or abs(old_summary.get("delta_model", -1) - .015*REFERENCE_LENGTH) > 1e-12):
        raise ValueError("Baseline must use opposite-winding HINGE at the fixed model scale and 1.5%L cap")
    for key, basename in (("post", "post.npz"), ("pairs", "pairs.json")):
        hashes = [digest for name, digest in provenance.get("files", {}).items()
                  if name.replace("\\", "/").rsplit("/", 1)[-1] == basename]
        if len(hashes) != 1 or hashes[0] != sha(paths[key]):
            raise ValueError("Baseline receipt does not authenticate supplied " + key)
    n, k = int(base["input_faces"]), int(base["known_faces"])
    if (n, k) != (provenance["row"].get("N"), provenance["row"].get("K")):
        raise ValueError("Baseline receipt and arrays disagree on original N/K")
    v, f = base["vertices"], base["faces"]
    m = len(f)
    if not (0 < k < m <= n):
        raise ValueError("Invalid C/free/original-N cardinalities")
    if v.ndim != 2 or v.shape[1] != 3 or f.shape != (m, 3) or f.dtype.kind not in "iu":
        raise ValueError("Expected indexed [V,3] vertices and integer [F,3] faces")
    if v.dtype != np.float32 or pre.dtype != np.float32 or raw.dtype != np.float32:
        raise ValueError("Frozen geometry requires FP32 coordinates; no implicit conversion")
    if not np.isfinite(v).all() or not np.isfinite(pre).all() or not np.isfinite(raw).all():
        raise ValueError("Nonfinite geometry")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("Face vertex ID out of bounds")
    if pre.shape != (n, 3, 3) or raw.shape != (n, 3, 3):
        raise ValueError("PRE_GEO and RAW must preserve the original N x 3 x 3 slots")
    sf, sc, cv = base["source_face_ids"], base["source_corner_ids"], base["corner_vertex_ids"]
    if sf.shape != (m,) or sc.shape != (m, 3) or cv.shape != (n, 3):
        raise ValueError("Source-face/source-corner/GEO corner mapping shape mismatch")
    if any(a.dtype.kind not in "iu" for a in (sf, sc, cv)):
        raise ValueError("Identity arrays must contain integer IDs")
    if len(np.unique(sf)) != m or sf.min() < 0 or sf.max() >= n:
        raise ValueError("Original surviving RAW face IDs must be unique and in bounds")
    if not np.array_equal(sf[:k], np.arange(k)) or not np.array_equal(base["raw_slot_ids"], sf):
        raise ValueError("C prefix or RAW slot identity mismatch")
    if not np.all(np.sort(sc, axis=1) == np.arange(3)):
        raise ValueError("Source corners must preserve each original triangle's permutation")
    if cv.min() < -1 or cv.max() >= len(v):
        raise ValueError("GEO corner-to-vertex mapping outside original vertex domain")
    if not np.array_equal(cv[sf[:, None], sc], f):
        raise ValueError("Actual faces disagree with original corner-to-vertex identities")
    if not np.array_equal(v[f], base["output"]) or not np.array_equal(base["C"], base["output"][:k]) or not np.array_equal(base["free"], base["output"][k:]):
        raise ValueError("Indexed mesh and triangle payload disagree")
    if base["raw_seam_slot_mask"].shape != (n,) or base["output_anchor_ids"].shape != (m, 3):
        raise ValueError("Seam role/anchor identities are required")
    anchors = base["output_anchor_ids"] >= 0
    if not set(map(int, f[anchors])) <= set(map(int, f[:k].ravel())):
        raise ValueError("A seam base anchor is not an actual indexed C vertex")
    pre_original_order = pre[sf[:, None], sc]
    if not np.array_equal(pre_original_order[:k], base["output"][:k]):
        raise ValueError("PRE_GEO C coordinates/permutation do not match fixed source C")
    # Unreferenced representative IDs intentionally survive old HINGE merges.
    # The original GEO member mapping, not a new radius clustering, is retained.
    return dict(original_N=n, C_faces=k, baseline_faces=m,
                baseline_vertices=len(v), unused_preserved_vertex_ids=len(v)-len(np.unique(f)),
                request_id=provenance["row"].get("request_id"),
                reference_length_L_model=REFERENCE_LENGTH,
                scale_contract="Fixed canonical model coordinates sqrt(3)*2/.3762; arbitrary mesh rescaling unsupported",
                input_contract="HINGE_OPP015 frozen indexed endpoint; original PRE_GEO/RAW identity domain")


def run(args):
    if getattr(args, "mode", "alpha-hole") not in MODES:
        raise ValueError("Unknown frozen mode")
    if float(getattr(args, "delta", .015)) != .015:
        raise ValueError("This archive supports only frozen delta=0.015")
    paths = {key: Path(getattr(args, key)).expanduser().resolve()
             for key in ("post", "pre_geo", "raw", "pairs", "baseline_receipt")}
    for key in ("reference", "reference_pairs"):
        if getattr(args, key, None):
            paths[key] = Path(getattr(args, key)).expanduser().resolve()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return dict(status="BLOCKED_MISSING_FROZEN_ASSET", missing=missing, model_calls=0, geometry_calls=0)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        return dict(status="BLOCKED_OUTPUT_ALREADY_EXISTS", output=str(output), model_calls=0, geometry_calls=0)
    source_path, allowed = source_manifest()
    source_root = ARCHIVE / "source/repository"
    legacy = ARCHIVE / "source/legacy/src/meshflow_control/evaluation"
    # Import isolation is part of the API contract; never silently reuse a
    # checkout module already imported by the caller.
    if "torch" in sys.modules or any(name == "r2" or name.startswith("r2.") for name in sys.modules):
        raise RuntimeError("Use a fresh Python process; preloaded torch/r2 is not supported")
    sys.path.insert(0, str(source_root))
    import numpy as np
    from r2.tools.hinge_v2_cpu_v1 import alpha, holes
    if abs(alpha.C.L - REFERENCE_LENGTH) > 1e-12:
        raise RuntimeError("Archived reference length differs from the frozen coordinate contract")
    Q = alpha.Q
    adapter = Q._module("_partial_binding_original_math_adapter",
                        Q.ROOT / "r2/pilots/user_condition_finetune_v1/evaluation.py")
    adapter.LEGACY = legacy  # Filesystem configuration only, never a math override.
    archived_modules(allowed)

    def arrays(path):
        with np.load(path, allow_pickle=False) as value:
            return {key: value[key].copy() for key in value.files}

    inputs = {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()}
    base = arrays(paths["post"])
    pre, raw = arrays(paths["pre_geo"])["output"], arrays(paths["raw"])["output"]
    pairs = read(paths["pairs"])
    contract = validate_baseline(base, pre, raw, read(paths["baseline_receipt"]), paths, np)
    for kind in ("CF", "FF"):
        if not isinstance(pairs.get(kind, {}).get("contacts"), list):
            raise ValueError("Frozen CF/FF pair cache is required")
    state, current_pairs = base, pairs
    records, summaries = {}, {}
    started = time.perf_counter()
    if args.mode in ("midpoint", "alpha", "alpha-hole"):
        state, rec, summary, _, _, current_pairs = alpha.decode(
            state, pre, raw, current_pairs, .015, alphas=args.mode != "midpoint")
        records["alpha"] = rec
        summaries["alpha"] = summary
    if args.mode in ("hole", "alpha-hole"):
        state, current_pairs, _, rec, summary = holes.decode(state, current_pairs)
        records["hole"] = rec
        summaries["hole"] = summary
    elapsed = time.perf_counter() - started
    m, k = len(base["faces"]), int(base["known_faces"])
    if len(state["faces"]) < m or not np.array_equal(state["faces"][:k], base["faces"][:k]):
        raise AssertionError("Original faces deleted or C topology changed")
    if state["output"][:k].tobytes() != base["output"][:k].tobytes():
        raise AssertionError("C coordinates changed")
    anchors = base["output_anchor_ids"] >= 0
    if state["output"][:m][anchors].tobytes() != base["output"][anchors].tobytes():
        raise AssertionError("Original seam anchors changed")
    modules = archived_modules(allowed)
    for key, path in paths.items():
        if sha(path) != inputs[key]["sha256"]:
            raise RuntimeError("Read-only input changed during replay: " + str(path))
    checks = []
    if "reference" in paths:
        expected = arrays(paths["reference"])
        for key in sorted(set(expected) | set(state)):
            good = (key in expected and key in state and expected[key].shape == state[key].shape
                    and expected[key].dtype == state[key].dtype and expected[key].tobytes() == state[key].tobytes())
            checks.append(dict(array=key, byte_equal=bool(good)))
    pair_equal = current_pairs == read(paths["reference_pairs"]) if "reference_pairs" in paths else None
    passed = all(row["byte_equal"] for row in checks) and pair_equal is not False
    receipt = dict(status="PASS" if passed else "FAIL_REFERENCE_MISMATCH", version=VERSION,
                   mode=args.mode, delta_fraction=.015, original_first=True,
                   reference_length_L_model=REFERENCE_LENGTH,
                   delta_model=.015*REFERENCE_LENGTH,
                   displacement_origin="Cumulative original PRE_GEO, including the preceding HINGE displacement",
                   source_manifest_sha256=sha(source_path), verified_source_files=len(allowed),
                   inputs=inputs, input_contract=contract, scientific_modules=modules,
                   reference_array_comparison=checks, reference_pairs_equal=pair_equal,
                   summaries=summaries, elapsed_seconds=elapsed, model_calls=0, GPU_calls=0,
                   new_D0=0, new_noise=0, new_queries=0, original_input_hashes_unchanged=True,
                   C_and_original_seam_anchors_byte_equal=True,
                   scope="One explicit frozen-array postprocess; no aggregate rescoring or image adjudication")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "post.npz").open("xb") as handle:
        np.savez_compressed(handle, **state)
    for filename, value in (("pairs.json", current_pairs), ("candidate_records.json", records), ("summary.json", summaries)):
        (output / filename).write_text(json.dumps(alpha.C.clean(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    receipt["outputs"] = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(output.iterdir())}
    (output / "receipt.json").write_text(json.dumps(alpha.C.clean(receipt), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("post", "pre-geo", "raw", "pairs", "baseline-receipt", "output"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--mode", choices=MODES, default="alpha-hole")
    parser.add_argument("--delta", type=float, choices=(.015,), default=.015)
    parser.add_argument("--reference", help="Optional saved endpoint NPZ: exact key/dtype/shape/byte checks")
    parser.add_argument("--reference-pairs", help="Optional saved endpoint pair JSON for full equality")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("scientific_modules", "reference_array_comparison", "inputs", "summaries")}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
