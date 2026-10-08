"""NumPy-only file interface for the fixed C-preserving GEO_C cleanup.

This operates on an existing RAW triangle soup. It does not generate geometry
with a model, load weights, score outputs, normalize coordinates or tune radius.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from .postprocessing import FIXED_CONFIG, RADIUS, process

SCHEMA = "native_t1_c_preserving_postprocess_cli_v1"
_PLY_FACE = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
_DELETE_REASONS = {"exact_repeated_vertex", "exact_collinear", "duplicate_C", "duplicate_free"}


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _array_record(value):
    a = np.ascontiguousarray(value)
    return dict(dtype=str(a.dtype), shape=list(a.shape),
                data_sha256=hashlib.sha256(a.tobytes()).hexdigest())


def _atomic_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                    prefix=".postprocess.", suffix=".partial",
                                    dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_raw(path, known_faces, input_key="output"):
    """Read one finite FP32 RAW array; no dtype or coordinate conversion."""
    path = Path(path)
    if path.suffix.lower() not in (".npy", ".npz"):
        raise ValueError("--input must be an NPY or NPZ file")
    if type(known_faces) is not int:
        raise ValueError("--known-faces must be an explicit integer")
    if not isinstance(input_key, str) or not input_key:
        raise ValueError("--input-key must be nonempty")
    source_hash = file_sha256(path)
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        try:
            if input_key not in value.files:
                raise ValueError(f"NPZ key {input_key!r} is absent; pass --input-key explicitly")
            raw = value[input_key].copy()
        finally:
            value.close()
        selected_key = input_key
    else:
        raw = value
        selected_key = None
    if not isinstance(raw, np.ndarray) or raw.dtype != np.float32:
        raise TypeError("RAW must already be FP32; implicit dtype conversion is forbidden")
    if not ((raw.ndim == 2 and raw.shape[1:] == (9,))
            or (raw.ndim == 3 and raw.shape[1:] == (3, 3))):
        raise ValueError("RAW must have shape [N,9] or [N,3,3]")
    if not np.isfinite(raw).all():
        raise ValueError("RAW coordinates must be finite")
    if not 0 < known_faces < len(raw):
        raise ValueError("Require 0 < --known-faces K < input N")
    if file_sha256(path) != source_hash:
        raise RuntimeError("Input file changed while being read")
    source_shape = list(raw.shape)
    triangles = np.ascontiguousarray(raw.reshape(-1, 3, 3)).copy()
    triangles.flags.writeable = False
    return triangles, dict(path=str(path.resolve()), key=selected_key,
                           file_sha256=source_hash, original_shape=source_shape,
                           dtype="float32", stage="RAW",
                           array_hash_definition="SHA256 of C-contiguous array bytes; dtype/shape stored separately",
                           triangles=_array_record(triangles))


def _validate_indexed(vertices, faces):
    if not isinstance(vertices, np.ndarray) or vertices.dtype != np.float32:
        raise TypeError("PLY vertices must already be FP32")
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError("PLY requires finite vertices[V,3]")
    if not isinstance(faces, np.ndarray) or faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("PLY requires indexed triangles[F,3]")
    if faces.dtype.kind not in "iu":
        raise TypeError("PLY face indices must be integers")
    if len(vertices) > np.iinfo(np.int32).max:
        raise ValueError("PLY vertex count exceeds signed int32 index capacity")
    if faces.size and (np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("PLY face index is out of bounds")


def write_indexed_ply(path, vertices, faces):
    """Exclusive binary little-endian PLY write, preserving FP32 signed zeros."""
    _validate_indexed(vertices, faces)
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment native_t1 GEO_C indexed geometry; no coordinate conversion\n"
              f"element vertex {len(vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\nend_header\n").encode("ascii")
    records = np.empty(len(faces), dtype=_PLY_FACE)
    records["count"] = 3
    records["indices"] = faces.astype("<i4", copy=False)
    with Path(path).open("xb") as stream:
        stream.write(header)
        stream.write(np.ascontiguousarray(vertices, dtype="<f4").tobytes())
        stream.write(records.tobytes())
        stream.flush()
        os.fsync(stream.fileno())


def read_indexed_ply(path):
    """Strict reader for this CLI's indexed binary format; no mesh repair."""
    with Path(path).open("rb") as stream:
        lines = []
        for _ in range(32):
            line = stream.readline(1024)
            if not line or not line.endswith(b"\n"):
                raise ValueError("Truncated or oversized PLY header")
            text = line.decode("ascii").rstrip("\r\n")
            if not text.startswith("comment "):
                lines.append(text)
            if text == "end_header":
                break
        else:
            raise ValueError("Missing PLY end_header")
        if len(lines) != 9 or lines[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("Unsupported PLY format")
        if not lines[2].startswith("element vertex ") or not lines[6].startswith("element face "):
            raise ValueError("Unsupported PLY elements")
        if lines[3:6] != ["property float x", "property float y", "property float z"]:
            raise ValueError("Unsupported PLY vertex layout")
        if lines[7:] != ["property list uchar int vertex_indices", "end_header"]:
            raise ValueError("Unsupported PLY face layout")
        V, F = int(lines[2].split()[-1]), int(lines[6].split()[-1])
        if V < 0 or F < 0 or V > np.iinfo(np.int32).max:
            raise ValueError("Invalid PLY element count")
        data = stream.read()
    expected = V * 12 + F * _PLY_FACE.itemsize
    if len(data) != expected:
        raise ValueError("Truncated PLY payload or unexpected trailing bytes")
    vertices = np.frombuffer(data, dtype="<f4", count=V * 3).reshape(V, 3).copy()
    records = np.frombuffer(data, dtype=_PLY_FACE, count=F, offset=V * 12)
    if np.any(records["count"] != 3):
        raise ValueError("Only triangle faces are supported")
    faces = records["indices"].astype(np.int64)
    _validate_indexed(vertices, faces)
    return vertices, faces


def _validate_result(raw, K, result):
    """Check exported contracts using returned arrays, not a second process()."""
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    if any(a.dtype.hasobject for a in arrays.values()):
        raise TypeError("Object arrays are forbidden")
    output, vertices, faces = (arrays[k] for k in ("output", "vertices", "faces"))
    _validate_indexed(vertices, faces)
    if output.dtype != np.float32 or output.ndim != 3 or output.shape[1:] != (3, 3):
        raise ValueError("Processor output must be FP32 [N_after,3,3]")
    if len(output) < K or not _same(output[:K], raw[:K]):
        raise AssertionError("C coordinate bits/order/winding/count changed")
    if not _same(vertices[faces], output):
        raise AssertionError("Indexed triangles differ from output")
    ids, keep = arrays["source_face_ids"], arrays["source_face_kept"]
    if keep.dtype != np.bool_ or keep.shape != (len(raw),) or not keep[:K].all():
        raise AssertionError("Known faces were removed or keep map is invalid")
    if ids.dtype != np.int64 or not _same(ids, np.flatnonzero(keep)):
        raise AssertionError("Retained source-face order/mapping differs")
    if not _same(arrays["source_corner_ids"], np.tile(np.arange(3), (len(output), 1))):
        raise AssertionError("Source corner order/winding mapping differs")
    targets = arrays["corner_target_input_ids"]
    if targets.shape != (len(raw), 3) or targets.dtype != np.int64:
        raise AssertionError("Invalid corner target shape/dtype")
    if np.any(targets < 0) or np.any(targets >= len(raw) * 3):
        raise AssertionError("Corner target outside original input")
    corrected = raw.reshape(-1, 3)[targets]
    if not _same(corrected[ids], output):
        raise AssertionError("Original corner targets do not reconstruct output")
    source_vertices = arrays["vertex_source_input_corner_ids"]
    if source_vertices.dtype != np.int64 or source_vertices.shape != (len(vertices),):
        raise AssertionError("Invalid indexed-vertex source map")
    if np.any(source_vertices < 0) or np.any(source_vertices >= len(raw) * 3):
        raise AssertionError("Indexed vertex source outside original input")
    if not _same(raw.reshape(-1, 3)[source_vertices], vertices):
        raise AssertionError("Indexed-vertex source map does not reconstruct coordinates")
    mapped = arrays["corner_vertex_ids"]
    if mapped.dtype != np.int64 or mapped.shape != (len(raw), 3):
        raise AssertionError("Invalid original-corner to output-vertex map")
    if np.any(mapped < -1) or np.any(mapped >= len(vertices)) or not _same(mapped[ids], faces):
        raise AssertionError("Corner/output indexed map is invalid")
    present = mapped >= 0
    if not _same(vertices[mapped[present]], corrected[present]):
        raise AssertionError("Present deleted/kept corner mapping changes coordinate bits")
    vector = corrected.astype(np.float64) - raw.astype(np.float64)
    distance = np.sqrt(np.einsum("nci,nci->nc", vector, vector))
    if not _same(arrays["corner_displacement_vector"], vector):
        raise AssertionError("Displacement vectors disagree with original corner targets")
    if not _same(arrays["corner_displacement"], distance) or np.any(distance > RADIUS):
        raise AssertionError("Displacement distances disagree or exceed fixed radius")
    records = result["face_records"]
    if len(records) != len(raw):
        raise AssertionError("Missing face deletion records")
    for i, record in enumerate(records):
        if record["source_face_id"] != i or record["known"] != (i < K) or record["kept"] != bool(keep[i]):
            raise AssertionError("Face record/source map mismatch")
        if (keep[i] and record["reason"] != "kept") or (not keep[i] and record["reason"] not in _DELETE_REASONS):
            raise AssertionError("Missing or unsupported deletion reason")
    arrays.update(C=raw[:K].copy(), known_faces=np.asarray(K, dtype=np.int64),
                  input_faces=np.asarray(len(raw), dtype=np.int64),
                  source_face_reason=np.asarray([r["reason"] for r in records], dtype="U32"),
                  duplicate_of_source_face=np.asarray([r["duplicate_of_source_face"] for r in records], dtype=np.int64))
    return arrays


def run(input_path, known_faces, out, input_key="output"):
    """Write one new output directory; leave STOPPED evidence on stage failure."""
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Output directory must be new: {out}")
    raw, source = load_raw(input_path, known_faces, input_key)
    input_array_before = raw.tobytes()
    started = time.perf_counter()
    report = dict(schema=SCHEMA, status="RUNNING", input=source, output_directory=str(out.resolve()),
                  stages=dict(input="RAW", output="GEO_C"),
                  config=dict(FIXED_CONFIG), known_faces=known_faces, input_faces=len(raw),
                  C_original_index_graph="NOT_PROVIDED",
                  C_identity_contract="First K triangle coordinates/order/multiplicity/winding and FP32 bit patterns, including signed zero",
                  model_calls=0, input_modified=False, files={})
    out.mkdir(parents=True, exist_ok=False)
    try:
        _atomic_json(out / "postprocess.json", report)
        result = process(raw, known_faces, radius=RADIUS)
        arrays = _validate_result(raw, known_faces, result)
        if raw.tobytes() != input_array_before:
            raise AssertionError("Processor mutated RAW input")
        with (out / "post.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(out / "post.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not _same(saved[k], v) for k, v in arrays.items()):
                raise AssertionError("NPZ roundtrip changed an array or source map")
        write_indexed_ply(out / "post.ply", arrays["vertices"], arrays["faces"])
        pv, pf = read_indexed_ply(out / "post.ply")
        if not _same(pv, arrays["vertices"]) or not _same(pf, arrays["faces"]):
            raise AssertionError("PLY indexed array roundtrip differs")
        if not _same(pv[pf], arrays["output"]) or not _same(pv[pf[:known_faces]], raw[:known_faces]):
            raise AssertionError("PLY triangle/C bit patterns or face order/winding changed")
        if file_sha256(input_path) != source["file_sha256"]:
            raise AssertionError("Original RAW file changed")
        report.update(status="PASS", stats=result["stats"], face_records=result["face_records"],
                      output_faces=len(arrays["output"]), arrays={k: _array_record(v) for k, v in arrays.items()},
                      contracts=dict(input_file_unchanged=True, input_array_unchanged=True,
                          C_bits_order_winding_multiplicity_unchanged=True, NPZ_all_arrays_bitwise=True,
                          PLY_vertices_and_indices_bitwise=True, indexed_triangles_bitwise=True,
                          all_corner_maps_verified=True, all_displacements_within_fixed_radius=True,
                          known_faces_never_deleted=True, C_original_index_graph="NOT_PROVIDED"),
                      seconds=time.perf_counter() - started)
        report["files"] = {p.name: dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=file_sha256(p))
                           for p in (out / "post.npz", out / "post.ply")}
        _atomic_json(out / "postprocess.json", report)
        return report
    except BaseException as error:
        report.update(status="STOPPED", error=dict(type=type(error).__name__, message=str(error)),
                      seconds=time.perf_counter() - started)
        report["files"] = {p.name: dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=file_sha256(p))
                           for p in (out / "post.npz", out / "post.ply") if p.is_file()}
        _atomic_json(out / "postprocess.json", report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU-only fixed GEO_C postprocess of existing FP32 RAW; no model or coordinate normalization")
    parser.add_argument("--input", required=True, type=Path, help="Read-only NPY or NPZ RAW triangle soup")
    parser.add_argument("--input-key", default="output", help="NPZ array key (default: output; use raw for raw-key files); ignored for NPY")
    parser.add_argument("--known-faces", required=True, type=int, help="Explicit K: first K input faces are immutable C; 0<K<N")
    parser.add_argument("--out", required=True, type=Path, help="New directory; existing paths are never overwritten")
    args = parser.parse_args(argv)
    try:
        report = run(args.input, args.known_faces, args.out, args.input_key)
    except Exception as error:
        print(f"Postprocess stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(dict(status=report["status"],out=report["output_directory"],
                          input_faces=report["input_faces"],output_faces=report["output_faces"],model_calls=0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
