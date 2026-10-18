"""Portable, CPU-only HINGE decode on explicitly supplied frozen arrays.

Only the archived adapter's LEGACY directory setting is supplied explicitly.
No function is replaced, rebound or edited. No model or source Y is required.
The full historical multi-request runner retains its local recovery contract.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ARCHIVE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    paths = {key: Path(getattr(args, key)).expanduser().resolve()
             for key in ("post", "pre_geo", "raw", "pairs")}
    if args.reference:
        paths["reference"] = Path(args.reference).expanduser().resolve()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return dict(status="BLOCKED_MISSING_FROZEN_ASSET", missing=missing,
                    model_calls=0, geometry_calls=0)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        return dict(status="BLOCKED_OUTPUT_ALREADY_EXISTS", output=str(output),
                    model_calls=0, geometry_calls=0)
    source_root = ARCHIVE / "source/repository"
    legacy = ARCHIVE / "source/legacy/src/meshflow_control/evaluation"
    if not (source_root / "r2/tools/hinge_opposite_v1/engine.py").is_file() or not legacy.is_dir():
        return dict(status="BLOCKED_ARCHIVED_SOURCE_MISSING", model_calls=0, geometry_calls=0)
    sys.path.insert(0, str(source_root))
    import numpy as np
    from r2.tools.hinge_opposite_v1 import engine
    Q = engine.Q
    # This is an explicit filesystem-root configuration, not a math override.
    adapter = Q._module("_partial_binding_original_math_adapter",
                        Q.ROOT / "r2/pilots/user_condition_finetune_v1/evaluation.py")
    adapter.LEGACY = legacy
    if "torch" in sys.modules:
        raise RuntimeError("CPU-only replay cannot run with torch already imported")

    def arrays(path):
        with np.load(path, allow_pickle=False) as value:
            return {key: value[key].copy() for key in value.files}

    base = arrays(paths["post"])
    pre = arrays(paths["pre_geo"])["output"]
    raw = arrays(paths["raw"])["output"]
    pairs = json.loads(paths["pairs"].read_text(encoding="utf-8-sig"))
    n = int(base["input_faces"])
    k = int(base["known_faces"])
    if pre.shape != (n, 3, 3) or raw.shape != (n, 3, 3):
        raise ValueError("PRE_GEO and RAW must preserve the original N x 3 x 3 slots")
    if base["corner_vertex_ids"].shape != (n, 3):
        raise ValueError("Original GEO corner-to-vertex identities are required")
    if not np.array_equal(base["vertices"][base["faces"]], base["output"]):
        raise ValueError("Actual indexed mesh and triangle payload disagree")
    if not np.array_equal(base["source_face_ids"][:k], np.arange(k)):
        raise ValueError("C identity/prefix mismatch")
    started = time.perf_counter()
    result, records, summary, _, _, _ = engine.decode(
        base, pre, raw, pairs, args.delta, require_opposite_winding=True)
    seconds = time.perf_counter() - started
    modules = {}
    for name, module in sys.modules.items():
        if not (name.startswith("r2") or name.startswith("_partial_binding_")
                or name.startswith("_r2_user_pilot_original_evaluation")):
            continue
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if not path.is_relative_to(ARCHIVE / "source"):
                raise RuntimeError("Scientific code escaped archive: " + str(path))
            modules[name] = dict(path=path.relative_to(ARCHIVE).as_posix(), sha256=sha(path))
    if "torch" in sys.modules:
        raise RuntimeError("CPU replay unexpectedly imported torch")
    checks = []
    if args.reference:
        expected = arrays(paths["reference"])
        for key in sorted(set(expected) | set(result)):
            good = (key in expected and key in result
                    and expected[key].shape == result[key].shape
                    and expected[key].dtype == result[key].dtype
                    and expected[key].tobytes() == result[key].tobytes())
            checks.append(dict(array=key, byte_equal=bool(good)))
    good = all(row["byte_equal"] for row in checks)
    receipt = dict(status="PASS" if good else "FAIL_REFERENCE_ARRAY_MISMATCH",
                   engine_version=engine.VERSION, require_opposite_winding=True,
                   delta_fraction=args.delta, source_ROOT=str(source_root), legacy_ROOT=str(legacy),
                   inputs={key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()},
                   scientific_modules=modules, original_array_comparison=checks,
                   summary=summary, elapsed_seconds=seconds, model_calls=0,
                   GPU_calls=0, new_noise=0, new_queries=0,
                   scope="One explicit array decode; no whole-output rescoring or image adjudication")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "post.npz").open("xb") as handle:
        np.savez_compressed(handle, **result)
    receipt["output_sha256"] = sha(output / "post.npz")
    (output / "receipt.json").write_text(
        json.dumps(engine.C.clean(receipt), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "candidate_records.json").write_text(
        json.dumps(engine.C.clean(records), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("post", "pre-geo", "raw", "pairs", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--delta", type=float, choices=(.015, .025), required=True)
    parser.add_argument("--reference", help="Optional existing reference NPZ for exact per-array comparison")
    args = parser.parse_args()
    receipt = run(args)
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in ("scientific_modules", "original_array_comparison", "inputs", "summary")}, indent=2))
    raise SystemExit(0 if receipt["status"] == "PASS" else 2)
