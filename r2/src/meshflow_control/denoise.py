"""Portable single-step D0 -> optional RBF -> GEO_C -> source-winding export.

File and cached-proposal paths are NumPy-only. Torch is imported only for an
explicit checkpoint invocation. Original inputs and intermediate geometry stay
separate; incomplete model attempts are never silently replayed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from . import geo_c, geo_io

SCHEMA = "meshflow_r2_denoise_source_export_v1"
OFFICIAL_SHA = "6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    geo_io._atomic_json(Path(path), clean(value))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def save_npz(path, arrays):
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise TypeError("Object arrays are not exportable")
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    loaded = load_npz(path)
    if loaded.keys() != arrays.keys() or not all(geo_io._same(loaded[k], np.asarray(v)) for k, v in arrays.items()):
        raise AssertionError("NPZ roundtrip changed array bytes")


def event(out, name, **fields):
    with (out / "ledger.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(clean(dict(event=name, time_unix=time.time(), **fields)), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def implementation_hashes():
    root = Path(__file__).resolve().parent
    names = ["denoise.py", "denoise_model.py", "source_winding.py", "winding_solver.py", "geo_c.py", "geo_io.py", "rbf_coordination.py"]
    paths = [root/n for n in names] + list((root/"denoiser_models").glob("*.py"))
    return {p.relative_to(root).as_posix():geo_io.file_sha256(p) for p in sorted(paths)}


def new_output(out):
    out = Path(out).absolute()
    if out.exists() or any(x.is_symlink() for x in (out, *out.parents)):
        raise FileExistsError("Use a new output directory without symlink components")
    if out.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("Outputs cannot be placed inside the installed source package")
    out.mkdir(parents=True, exist_ok=False)
    return out


def save_mesh(out, name, arrays):
    save_npz(out / (name + ".npz"), arrays)
    geo_io.write_indexed_ply(out / (name + ".ply"), arrays["vertices"], arrays["faces"])
    v, f = geo_io.read_indexed_ply(out / (name + ".ply"))
    if not geo_io._same(v, arrays["vertices"]) or not geo_io._same(f, arrays["faces"]):
        raise AssertionError("PLY roundtrip changed indexed geometry")


def source_for(condition, canonical_C):
    from .source_winding import validate_source
    if condition is None:
        raise ValueError("Source-winding export requires --condition with saved source metadata; use prepare-condition before sampling")
    before = geo_io.file_sha256(condition)
    source = load_npz(condition)
    lineage = validate_source(source, canonical_C)
    if geo_io.file_sha256(condition) != before:
        raise ValueError("Source condition changed while loading")
    lineage["source_file_sha256"] = before
    return source, lineage


def finish_orientation(post, source, out):
    from .source_winding import export_source_winding
    result = export_source_winding(post, source)
    save_mesh(out, "post_restored_C", result["restored_arrays"])
    save_mesh(out, "post", result["oriented_arrays"])
    write_json(out / "source_restoration.json", result["restoration_summary"])
    orientation = result["orientation_result"]
    write_json(out / "orientation.json", dict(summary=orientation["summary"],
               components=orientation["component_rows"], edges=orientation["edge_rows"]))
    return result


def file_hashes(out):
    return {p.name: geo_io.file_sha256(p) for p in sorted(out.iterdir())
            if p.is_file() and p.name not in {"denoise.json", "export.json", "failure.json"}}


def verify_hashes(out, hashes):
    for name, expected in hashes.items():
        if geo_io.file_sha256(out / name) != expected:
            raise ValueError("Saved output changed: " + name)


def prepare_condition(*, input, out, input_key="C", vertex_ids_key=None,
                      permutation_key=None, face_ids_key=None):
    """Explicitly snapshot user C *before* canonicalization; no parent or GT."""
    from .source_winding import capture_source
    p = Path(input)
    before = geo_io.file_sha256(p)
    arrays = load_npz(p) if p.suffix.lower() == ".npz" else {input_key: np.load(p, allow_pickle=False)}
    selected = {"vertices", "faces"} if input_key == "indexed" else {input_key}
    selected.update(k for k in (vertex_ids_key, permutation_key, face_ids_key) if k)
    if set(arrays) - selected:
        raise ValueError("Unselected source attributes/arrays are not silently dropped: " + ", ".join(sorted(set(arrays)-selected)))
    if input_key == "indexed":
        vertices, faces = arrays["vertices"], arrays["faces"]
        geo_io._validate_indexed(vertices, faces)
        original = vertices[faces]
        ids = faces
    else:
        original = arrays[input_key]
        if original.ndim == 2 and original.shape[1:] == (9,):
            original = original.reshape(-1, 3, 3)
        ids = arrays[vertex_ids_key] if vertex_ids_key else None
    source = capture_source(original, source_vertex_ids=ids,
                            corner_permutation=arrays[permutation_key] if permutation_key else None,
                            source_face_ids=arrays[face_ids_key] if face_ids_key else None)
    out = Path(out)
    if out.suffix.lower() != ".npz":
        raise ValueError("Prepared source condition output must use .npz")
    if out.exists() or any(x.is_symlink() for x in (out.absolute(), *out.absolute().parents)):
        raise FileExistsError("Condition output must be a new file")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, source)
    if geo_io.file_sha256(p) != before:
        raise ValueError("Source input changed during capture")
    return dict(status="COMPLETE", schema=SCHEMA, path=str(out.resolve()),
                sha256=geo_io.file_sha256(out), source_sha256=before,
                model_forwards=0, scope="Explicit source capture, not retroactive recovery of unknown order")


def orient(*, input, condition, out):
    """Orient a saved canonical GEO_C NPZ without another D0 or GEO invocation."""
    input_hash = geo_io.file_sha256(input)
    post = load_npz(input)
    if geo_io.file_sha256(input) != input_hash:
        raise ValueError("POST changed while loading")
    K = int(post["known_faces"])
    source, lineage = source_for(condition, post["output"][:K])
    identity = dict(schema=SCHEMA, input_sha256=input_hash,
                    condition_sha256=lineage["source_file_sha256"], implementation_hashes=implementation_hashes())
    out = new_output(out)
    write_json(out / "registration.json", identity)
    try:
        event(out, "SOURCE_EXPORT_STARTED")
        result = finish_orientation(post, source, out)
        if geo_io.file_sha256(input) != identity["input_sha256"] or geo_io.file_sha256(condition) != identity["condition_sha256"]:
            raise ValueError("Source/POST changed during export")
        event(out, "SOURCE_EXPORT_COMPLETE")
        receipt = dict(status="COMPLETE", identity=identity, files=file_hashes(out),
                       orientation=result["orientation_result"]["summary"],
                       restoration=result["restoration_summary"], model_forwards=0, GEO_calls=0,
                       coordinates_unchanged=True, unordered_faces_unchanged=True)
        write_json(out / "export.json", receipt)
        return receipt
    except BaseException as error:
        write_json(out / "failure.json", dict(status="STOPPED", error=repr(error), model_forwards=0))
        raise


def resources():
    def call(args):
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    return dict(gpu=call(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]),
                processes=call(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]),
                cpu_logical=os.cpu_count(), loadavg=list(os.getloadavg()) if hasattr(os,"getloadavg") else None,
                memory=Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else "NA")


def actual_proposal(raw, checkpoint, out):
    # Validate resource availability before model loading, with a measured D0
    # inference allowance; one owned model, never parallel model-loading spikes.
    snapshot = resources()
    write_json(out / "resources_before_model.json", snapshot)
    if snapshot["gpu"]["returncode"] != 0:
        raise RuntimeError("nvidia-smi is required before GPU dispatch")
    first = snapshot["gpu"]["stdout"].strip().splitlines()[0].split(",")
    total, free = float(first[1]), float(first[3])
    if free - max(2048., .1 * total) < 2600.:
        raise RuntimeError("Insufficient measured D0 inference headroom; no model attempt dispatched")
    from .denoise_model import load_denoiser, predict_proposal
    model, model_identity = load_denoiser(checkpoint)
    write_json(out / "model_identity.json", model_identity)
    write_json(out / "resources_model_resident.json", resources())
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    event(out, "D0_FORWARD_STARTED", budget=1)
    proposal = predict_proposal(model, raw)
    torch.cuda.synchronize()
    event(out, "D0_FORWARD_RETURNED", seconds=time.perf_counter()-tick)
    import torch
    write_json(out / "resources_after_forward.json", dict(snapshot=resources(),
               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               scope="One FP32 resident D0 model with BF16 forward; reset immediately before predict_proposal; no optimizer"))
    return proposal


def cached_proposal(path, receipt_path, raw, raw_sha):
    if receipt_path is None:
        raise ValueError("--proposal requires --proposal-receipt with raw/model/output identity")
    receipt_hash = geo_io.file_sha256(receipt_path)
    proposal_hash = geo_io.file_sha256(path)
    receipt = read_json(receipt_path)
    if (receipt.get("raw_sha256") != raw_sha or receipt.get("checkpoint_sha256") != OFFICIAL_SHA
            or receipt.get("output_sha256") != proposal_hash or receipt.get("status") != "PASS"):
        raise ValueError("Cached official proposal identity mismatch")
    proposal = load_npz(path)["official_unconstrained_proposal"]
    validate_proposal(proposal, raw)
    if geo_io.file_sha256(receipt_path) != receipt_hash or geo_io.file_sha256(path) != proposal_hash:
        raise ValueError("Cached proposal/receipt changed while loading")
    return proposal


def validate_proposal(proposal, raw):
    if proposal.dtype != np.float32 or proposal.shape != raw.shape or not np.isfinite(proposal).all():
        raise ValueError("Full unconstrained proposal must be finite FP32 with the exact RAW shape")


def run(*, input, known_faces, out, condition=None, input_key="raw", checkpoint=None,
        proposal=None, proposal_receipt=None, coordination="none", reference_length=None,
        winding="source", resume=False):
    if (checkpoint is None) == (proposal is None):
        raise ValueError("Select exactly one --checkpoint or --proposal")
    if coordination not in {"none", "rbf"} or winding not in {"source", "canonical"}:
        raise ValueError("Unknown coordination/export contract")
    raw, raw_record = geo_io.load_raw(input, known_faces, input_key)
    K = known_faces
    source = lineage = None
    if condition is not None or winding == "source" or coordination == "rbf":
        source, lineage = source_for(condition, raw[:K])
    if coordination == "rbf":
        if not lineage["has_real_source_vertex_ids"]:
            raise ValueError("RBF requires provided source vertex IDs, not synthetic corner placeholders")
        if reference_length is None or not np.isfinite(reference_length) or reference_length <= 0:
            raise ValueError("RBF requires a fixed positive --reference-length, never a generated bbox")
    elif reference_length is not None:
        raise ValueError("--reference-length applies only to explicit RBF coordination")
    if checkpoint is not None and geo_io.file_sha256(checkpoint) != OFFICIAL_SHA:
        raise ValueError("Expected the pinned official Chair D0 checkpoint")
    cache_hash = geo_io.file_sha256(proposal) if proposal else None
    cache_receipt_hash = geo_io.file_sha256(proposal_receipt) if proposal_receipt else None
    incoming = cached_proposal(proposal, proposal_receipt, raw, raw_record["file_sha256"]) if proposal else None
    if proposal and (geo_io.file_sha256(proposal) != cache_hash or geo_io.file_sha256(proposal_receipt) != cache_receipt_hash):
        raise ValueError("Cache identity changed during preflight")
    identity = dict(schema=SCHEMA, raw_sha256=raw_record["file_sha256"], input_key=input_key,
                    known_faces=K, condition_sha256=lineage["source_file_sha256"] if condition else None,
                    checkpoint_sha256=OFFICIAL_SHA, proposal_sha256=cache_hash,
                    proposal_receipt_sha256=cache_receipt_hash,
                    coordination=coordination, reference_length=reference_length, winding=winding,
                    implementation_hashes=implementation_hashes())
    out = Path(out)
    if any(x.is_symlink() for x in (out.absolute(), *out.absolute().parents)):
        raise ValueError("Output symlink components are forbidden")
    if out.exists():
        if not resume:
            raise FileExistsError("Use a new output directory or explicit --resume for completed stages")
        if read_json(out / "registration.json") != identity:
            raise ValueError("Resume identity differs")
        if (out / "denoise.json").exists():
            result = read_json(out / "denoise.json")
            if result["status"] != "COMPLETE" or result.get("identity") != identity:
                raise ValueError("Invalid completion receipt")
            verify_hashes(out, result["files"])
            return dict(result, reused=True, model_forwards_this_invocation=0)
        if not (out / "proposal_receipt.json").exists():
            raise RuntimeError("Incomplete D0 attempt; no automatic model replay")
        # Resume only after a complete D0 stage, before geometry starts.
        if (out / "geometry_started.json").exists():
            raise RuntimeError("Incomplete geometry/export attempt; no implicit replay or overwrite")
        incoming = cached_proposal(out / "proposal.npz", out / "proposal_receipt.json", raw, identity["raw_sha256"])
    else:
        out = new_output(out)
        write_json(out / "registration.json", identity)
    calls = 0
    try:
        if not (out / "proposal_receipt.json").exists():
            if incoming is None:
                incoming = actual_proposal(raw, checkpoint, out)
                calls = 1
            validate_proposal(incoming, raw)
            save_npz(out / "proposal.npz", {"official_unconstrained_proposal": incoming})
            write_json(out / "proposal_receipt.json", dict(status="PASS", raw_sha256=identity["raw_sha256"],
                       checkpoint_sha256=OFFICIAL_SHA, output_sha256=geo_io.file_sha256(out / "proposal.npz"),
                       model_forwards=calls, source="CACHE" if proposal else "OFFICIAL_D0"))
        write_json(out / "geometry_started.json", dict(status="STARTED", time_unix=time.time()))
        # RBF must consume the full proposal BEFORE the C prefix is written back.
        if coordination == "rbf":
            from .rbf_coordination import coordinate_restore
            denoised, diagnostic, detail = coordinate_restore(raw[:K], incoming, source["C_source_vertex_ids"], L=reference_length)
            write_json(out / "coordination.json", diagnostic)
            save_npz(out / "coordination_arrays.npz", detail)
        else:
            denoised = incoming.copy()
            denoised[:K] = raw[:K]
        save_npz(out / "denoised.npz", dict(output=denoised, C=raw[:K].copy(), known_faces=np.asarray(K)))
        geometry = geo_c.process(denoised, K)
        post = geo_io._validate_result(denoised, K, geometry)
        save_mesh(out, "post_canonical", post)
        write_json(out / "geometry.json", {k:v for k,v in geometry.items() if not isinstance(v,np.ndarray)})
        oriented = finish_orientation(post, source, out) if winding == "source" else None
        if oriented is None:
            save_mesh(out, "post", post)
        if geo_io.file_sha256(input) != identity["raw_sha256"] or (condition and geo_io.file_sha256(condition) != identity["condition_sha256"]):
            raise ValueError("Input changed during denoise/export")
        if proposal and (geo_io.file_sha256(proposal) != cache_hash or geo_io.file_sha256(proposal_receipt) != cache_receipt_hash):
            raise ValueError("Cached input changed during execution")
        event(out, "COMPLETE", new_model_forwards=calls, GEO_calls=1, orientation_solves=int(oriented is not None))
        result = dict(status="COMPLETE", schema=SCHEMA, identity=identity, files=file_hashes(out),
                      model_forwards_this_invocation=calls, model_forwards_in_output=read_json(out / "proposal_receipt.json")["model_forwards"], generator_forwards=0, training_updates=0,
                      new_noise=0, full_quality_scores=0, GEO_calls=1,
                      source_export=oriented["restoration_summary"] if oriented else {"status":"LEGACY_CANONICAL"},
                      orientation=oriented["orientation_result"]["summary"] if oriented else None,
                      quality_claim="Execution and geometry contracts only; no automatic quality scoring")
        write_json(out / "denoise.json", result)
        return result
    except BaseException as error:
        write_json(out / "failure.json", dict(status="STOPPED", identity=identity, error=repr(error),
                   model_forward_attempts_in_output=sum(json.loads(line)["event"] == "D0_FORWARD_STARTED" for line in (out / "ledger.jsonl").read_text().splitlines()) if (out / "ledger.jsonl").exists() else 0, automatic_retry=False))
        raise
