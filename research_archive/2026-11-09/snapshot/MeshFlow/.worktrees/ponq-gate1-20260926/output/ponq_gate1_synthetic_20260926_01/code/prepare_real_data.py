"""Select four raw local meshes before any fitting or extraction.

Freeze the candidate order and eligibility rules first; never repair a mesh,
resample a candidate order, or choose objects based on reconstruction quality.
Only this run's new output directory is written. Old source assets are read-only.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
from common import dump, load_mesh, mesh_audit, np, save_mesh, sha
from exact_audit import exact_mesh


def ordered(rows):
    return sorted(rows, key=lambda row: (hashlib.sha256(row["uid"].encode("utf-8")).hexdigest(), row["uid"]))


def compact_identity(row, category):
    result = dict(uid=row["uid"], category=category,
                  ordering_sha256=hashlib.sha256(row["uid"].encode("utf-8")).hexdigest(),
                  source_split=row["source_split"])
    if category == "loudspeaker":
        result.update(relative_path=row["relative_path"], source_sha256=row["sha256"],
                      previous_status="closed16_development_parent")
    else:
        result.update(archive_member=row["archive_member"], source_sha256=row["source_sha256"],
                      source_bytes=row["source_bytes"], previous_status=row["status"],
                      previous_missing_rays=row.get("missing_rays"),
                      previous_multiple_rays=row.get("multiple_rays"))
    return result


def check_arrays(v, f):
    reasons = []
    if v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 4:
        reasons.append("invalid_vertex_array_shape")
    if f.ndim != 2 or f.shape[1:] != (3,) or len(f) < 4:
        reasons.append("invalid_triangle_array_shape")
    if reasons:
        return reasons
    if not np.issubdtype(v.dtype, np.number) or not np.isfinite(v).all():
        reasons.append("nonfinite_or_nonnumeric_vertices")
    if not np.issubdtype(f.dtype, np.integer):
        reasons.append("face_indices_not_integer_dtype")
    elif f.min() < 0 or f.max() >= len(v):
        reasons.append("out_of_range_face_indices")
    if not reasons:
        if np.any(f[:, 0] == f[:, 1]) or np.any(f[:, 1] == f[:, 2]) or np.any(f[:, 2] == f[:, 0]):
            reasons.append("repeated_vertex_in_triangle")
        if len(np.unique(f)) != len(v):
            reasons.append("unused_vertices_are_not_a_pure_surface_mesh")
        diag = float(np.linalg.norm(np.ptp(v.astype(np.float64), axis=0)))
        if not np.isfinite(diag) or diag <= 0:
            reasons.append("invalid_bbox_diagonal")
    return reasons


def geometric_audit(v, f):
    topology = mesh_audit(v, f)
    exact = exact_mesh(v, f)
    reasons = []
    if not topology["closed_oriented_2manifold"]:
        reasons.append("not_closed_consistently_oriented_2manifold")
    if topology["duplicate_faces"]:
        reasons.append("duplicate_faces")
    if topology["signed_volume"] <= 0:
        reasons.append("nonpositive_signed_volume_no_orientation_repair")
    if exact["exact_degenerate_faces"]:
        reasons.append("exact_degenerate_faces")
    if exact["exact_intersection_pairs"]:
        reasons.append("exact_illegal_triangle_intersections")
    return dict(topology=topology, exact=exact, rejection_reasons=reasons,
                passes=not reasons)


def audit_candidate(raw_bytes, identity, dest, source_location):
    dest.mkdir()
    raw_path = dest / "original.npz"
    raw_path.write_bytes(raw_bytes)
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != identity["source_sha256"]:
        # A provenance mismatch is an execution error, not a substitute candidate.
        raise RuntimeError(f"Source hash mismatch: {identity['uid']}")
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as archive:
        v0 = archive["vertices"].copy()
        f0 = archive["faces"].copy()
    audit = dict(identity=identity, source_location=source_location,
                 source_sha256=actual_sha, original_file=str(raw_path.resolve()),
                 source_vertices_dtype=str(v0.dtype), source_faces_dtype=str(f0.dtype),
                 raw_vertices_shape=list(v0.shape), raw_faces_shape=list(f0.shape),
                 repairs_performed=[], normalization_applied_count=0)
    reasons = check_arrays(v0, f0)
    if reasons:
        audit.update(status="REJECTED_RAW_ARRAYS", rejection_reasons=reasons)
        dump(dest / "AUDIT.json", audit)
        return audit, None
    # Exact value-preserving float64/int64 promotion, with original bytes retained.
    v = v0.astype(np.float64)
    f = f0.astype(np.int64)
    raw_audit = geometric_audit(v, f)
    audit["raw_audit"] = raw_audit
    if not raw_audit["passes"]:
        audit.update(status="REJECTED_RAW_GEOMETRY", rejection_reasons=raw_audit["rejection_reasons"])
        dump(dest / "AUDIT.json", audit)
        return audit, None
    centre = (v.min(0) + v.max(0)) * 0.5
    diag = float(np.linalg.norm(np.ptp(v, axis=0)))
    scale = 2.0 / diag
    normalized = (v - centre) * scale
    normalization = dict(kind="bbox_center_and_diagonal_2", centre=centre,
                         original_bbox_diagonal=diag, scale=scale,
                         operation="(raw_vertices - centre) * (2 / original_bbox_diagonal)",
                         application_count=1, already_normalized=True,
                         output_bbox_diagonal=float(np.linalg.norm(np.ptp(normalized, axis=0))))
    reference = dest / "reference.obj"
    save_mesh(reference, normalized, f)
    rv, rf = load_mesh(reference)
    if not np.array_equal(rv, normalized) or not np.array_equal(rf, f):
        raise RuntimeError(f"OBJ roundtrip changed arrays: {identity['uid']}")
    np.savez_compressed(dest / "raw_arrays.npz", vertices=v0, faces=f0)
    normalized_audit = geometric_audit(rv, rf)
    audit.update(normalization=normalization, normalization_applied_count=1,
                 normalized_audit=normalized_audit, reference_sha256=sha(reference))
    raw_betti = [raw_audit["topology"][f"beta{i}"] for i in range(3)]
    normalized_betti = [normalized_audit["topology"][f"beta{i}"] for i in range(3)]
    normalized_reasons = list(normalized_audit["rejection_reasons"])
    if raw_betti != normalized_betti:
        normalized_reasons.append("normalization_changed_betti")
    if normalized_reasons:
        audit.update(status="REJECTED_NORMALIZED_GEOMETRY", rejection_reasons=normalized_reasons)
        dump(dest / "AUDIT.json", audit)
        return audit, None
    audit.update(status="ELIGIBLE", rejection_reasons=[])
    dump(dest / "AUDIT.json", audit)
    spec = dict(type="local_mesh", path=str(reference.resolve()), expected_betti=raw_betti,
                source_uid=identity["uid"], source_sha256=actual_sha,
                source_location=source_location, source_split=identity["source_split"],
                category=identity["category"], previous_status=identity["previous_status"],
                normalization=normalization, already_normalized=True,
                reference_sha256=sha(reference), audit_path=str((dest / "AUDIT.json").resolve()),
                original_path=str(raw_path.resolve()))
    return audit, spec


def prepare(args):
    root = Path(args.out).resolve()
    if root.exists():
        raise RuntimeError(f"Refusing existing output directory: {root}")
    parents_path = Path(args.speaker_manifest).resolve()
    rows_path = Path(args.bottle_rows).resolve()
    old_protocol_path = Path(args.bottle_protocol).resolve()
    archive_path = Path(args.archive).resolve()
    parents = json.loads(parents_path.read_text())
    old_protocol = json.loads(old_protocol_path.read_text())
    previous_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    speakers = ordered(parents["objects"])
    bottles = ordered([row for row in previous_rows if row["status"] == "ray_reject"])
    if len(speakers) != 16 or len(bottles) != 61 or len(previous_rows) != 225:
        raise RuntimeError("Prior pool size differs from the authorized selection scope")
    if any(row["source_split"] != "train" for row in speakers + bottles):
        raise RuntimeError("Unexpected non-train source split")
    archive_sha = sha(archive_path)
    if archive_sha != old_protocol["source_archive_sha256"] or archive_sha != parents["source_archive_sha256"]:
        raise RuntimeError("Archive hash does not match the prior provenance")
    ordered_speakers = [compact_identity(row, "loudspeaker") for row in speakers]
    ordered_bottles = [compact_identity(row, "bottle") for row in bottles]
    protocol = dict(
        schema="PONQ_GATE1_REAL_DATA_SELECTION_V1",
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="Two raw loudspeaker parents and two freshly audited previously ray-rejected bottles; no reconstruction-based selection",
        output_dir=str(root), source_archive=str(archive_path), source_archive_sha256=archive_sha,
        parent_manifest=dict(path=str(parents_path), sha256=sha(parents_path)),
        prior_bottle_rows=dict(path=str(rows_path), sha256=sha(rows_path)),
        prior_bottle_protocol=dict(path=str(old_protocol_path), sha256=sha(old_protocol_path)),
        code_hashes={name: sha(Path(__file__).parent / name) for name in ["prepare_real_data.py", "common.py", "exact_audit.py"]},
        ordering="Ascending (sha256(uid.encode('utf-8')).hexdigest(), uid), independently within each category",
        selection="Audit in frozen order; select first two eligible per category; preserve every attempted failure; no substitute pool if fewer than two qualify",
        eligibility=["Finite N x 3 vertices; integer M x 3 in-range triangle indices; no repeated-index triangles or unused vertices",
                     "Closed consistently oriented two-manifold; no duplicate faces; positive total signed volume without changing face winding",
                     "Zero exact dyadic/rational illegal triangle intersections and zero exact degenerate faces",
                     "Only bbox centering and uniform positive scaling to diagonal 2; audit both original and exported normalized geometry",
                     "No connected-component count, genus, radial-hit, face-count, reconstruction quality or timing selection rule"],
        exclusions_removed=["single-hit radial condition", "genus zero"],
        limitation="This is not a new qualification rate for all 225 bottles. The preferred 61-object pool previously passed a genus-zero screen, so this selection cannot establish nonzero-genus bottle coverage.",
        repairs_allowed=False, reconstruction_or_training_performed=False,
        slot_indices={"loudspeaker_local_1": 4, "loudspeaker_local_2": 5,
                      "bottle_local_1": 6, "bottle_local_2": 7},
        candidate_pools=dict(loudspeaker=ordered_speakers, bottle=ordered_bottles))
    root.mkdir(parents=True)
    dump(root / "FROZEN_SELECTION_PROTOCOL.json", protocol)
    frozen_sha = sha(root / "FROZEN_SELECTION_PROTOCOL.json")
    dump(root / "FROZEN_SELECTION_PROTOCOL_SHA256.json", dict(sha256=frozen_sha))
    (root / "candidates").mkdir()
    records = []
    objects = {}
    manifest = dict(schema="PONQ_GATE1_REAL_DATA_SELECTION_V1", protocol_sha256=frozen_sha,
                    objects=objects, attempts=records,
                    normalization_contract="References are already bbox-centered with diagonal 2; consumers must not normalize them again.")
    with tarfile.open(archive_path, "r:gz") as archive:
        for category, candidates in [("loudspeaker", ordered_speakers), ("bottle", ordered_bottles)]:
            selected = 0
            for index, identity in enumerate(candidates):
                if selected == 2:
                    break
                if category == "loudspeaker":
                    source = parents_path.parent / identity["relative_path"]
                    raw_bytes = source.read_bytes()
                    source_location = str(source)
                else:
                    member = archive.getmember(identity["archive_member"])
                    if not member.isfile():
                        raise RuntimeError("Candidate archive member is not a file")
                    raw_bytes = archive.extractfile(member).read()
                    source_location = str(archive_path) + "::" + member.name
                candidate_dir = root / "candidates" / (category + "__" + identity["uid"])
                start = time.perf_counter()
                audit, spec = audit_candidate(raw_bytes, identity, candidate_dir, source_location)
                record = dict(category=category, uid=identity["uid"], ordered_index=index,
                              status=audit["status"], rejection_reasons=audit["rejection_reasons"],
                              elapsed_seconds=time.perf_counter()-start,
                              audit_path=str((candidate_dir / "AUDIT.json").resolve()))
                if spec is not None:
                    selected += 1
                    slot = category + "_local_" + str(selected)
                    spec["object_index"] = protocol["slot_indices"][slot]
                    spec["selection_ordered_index"] = index
                    objects[slot] = spec
                    record["selected_slot"] = slot
                records.append(record)
                dump(root / "SELECTION_PARTIAL.json", manifest)
                print(json.dumps(record), flush=True)
            if selected != 2:
                raise RuntimeError(f"Only {selected} eligible {category} objects; no substitute pool allowed")
    if sha(root / "FROZEN_SELECTION_PROTOCOL.json") != frozen_sha:
        raise RuntimeError("Frozen protocol changed during selection")
    manifest.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    status="FOUR_REAL_OBJECTS_SELECTED_NO_EXTRACTION", attempted_count=len(records),
                    rejected_count=sum(row["status"] != "ELIGIBLE" for row in records),
                    selected_count=len(objects))
    dump(root / "SELECTION.json", manifest)
    lines = ["# Real-object Gate 1 data selection", "",
             "Selected before any PoNQ fitting or extraction. Source files were read-only; no mesh repair was performed.", "",
             "|Slot|UID|Raw faces|Betti|Prior status|", "|---|---|---:|---|---|"]
    for slot, spec in objects.items():
        a = json.loads(Path(spec["audit_path"]).read_text())
        lines.append(f"|{slot}|{spec['source_uid']}|{a['raw_audit']['topology']['faces']}|{spec['expected_betti']}|{spec['previous_status']}|")
    lines.extend(["", f"Audited {len(records)} candidates; rejected {manifest['rejected_count']}. All attempted failures are retained in SELECTION.json and candidates/.",
                  "", protocol["limitation"], "", manifest["normalization_contract"],
                  "", "The speaker raw NPZ identity is bound to the earlier assets/parents.json. This does not newly certify the raw archive's historical preprocessing or make these unseen test subjects.", ""])
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    dump(root / "FILE_MANIFEST.json", dict(files=[dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                                                for p in sorted(root.rglob("*")) if p.is_file()]))
    print("SELECTION_COMPLETE", root / "SELECTION.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-manifest", required=True)
    parser.add_argument("--bottle-rows", required=True)
    parser.add_argument("--bottle-protocol", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
