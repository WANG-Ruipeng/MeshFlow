"""Read-only source-contract audit and immutable anchor-denoiser data whitelist.

Only existing active entries and existing source-quality classifications are used.
No model imports, geometry rescoring, RNG, noise, repair, or source mutation.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path("/mnt/e/MeshFlow")
SOURCE = Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1")
OUT = ROOT / "experiments/r2_anchor_aware_denoiser_v1/data_audit"
VERSION = "anchor_aware_denoiser_source_contract_v1"
LEGAL = frozenset(("shared_edge_contact", "shared_vertex_contact",
    "coplanar_boundary_or_tolerance_contact", "noncoplanar_edge_or_tangent_contact",
    "point_or_tangent_contact"))
ILLEGAL = frozenset(("proper_transverse_penetration", "coplanar_positive_area_overlap"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_target(path, source=None):
    """Reject before opening; resolve containment and refuse every symlink parent."""
    p = Path(path).absolute()
    for q in (p, *p.parents):
        if q.is_symlink():
            raise PermissionError("Symlink in output path: " + str(q))
    base = OUT.resolve()
    resolved = p.resolve()
    if resolved == base or not resolved.is_relative_to(base):
        raise PermissionError("Write outside owned data_audit: " + str(p))
    if source is not None and resolved == Path(source).resolve():
        raise PermissionError("Source and target identical")
    return resolved


def atomic(path, data):
    p = safe_target(path)
    if p.exists():
        if p.read_bytes() != data:
            raise FileExistsError("Refusing to change existing audit artifact: " + str(p))
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = safe_target(p.with_name(p.name + ".atomic.tmp"))
    if tmp.exists():
        raise FileExistsError("Inspect interrupted temporary output first: " + str(tmp))
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def write_json(name, value):
    atomic(OUT / name, (json.dumps(value, sort_keys=True, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n").encode("utf-8"))


def write_csv(name, rows):
    if not rows:
        raise ValueError("Expected nonempty audit table " + name)
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=list(rows[0]))
    w.writeheader()
    for row in rows:
        w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            for k, v in row.items()})
    atomic(OUT / name, s.getvalue().encode("utf-8"))


def identical(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def classify(label):
    if label in ILLEGAL:
        return "illegal"
    if label in LEGAL:
        return "legal"
    if label == "disjoint":
        return "disjoint"
    return "unknown"


def validate(row, parent, condition):
    m = parent["full_target"].reshape(-1, 3, 3)
    ci, yi = condition["source_face_ids"], condition["free_source_ids"]
    assert m.dtype == np.float32 and np.isfinite(m).all()
    assert len(m) == row["N"] == int(condition["N"])
    assert len(ci) == row["K"] == int(condition["K"])
    assert 0 < len(ci) < len(m)
    assert ci.tolist() == row["source_face_ids"]
    assert yi.tolist() == row["free_source_ids"]
    assert sorted(np.concatenate((ci, yi)).tolist()) == list(range(len(m)))
    assert len(set(ci).intersection(yi)) == 0
    assert identical(condition["C"], m[ci]) and identical(condition["Y"], m[yi])
    assert identical(condition["C_pre_ot"] * np.float32(2), condition["C"])
    assert identical(condition["Y_pre_ot"] * np.float32(2), condition["Y"])
    vids = parent["source_face_vertex_ids"]
    assert identical(parent["model_vertices"][vids], m)
    original = parent["original_source_faces"][parent["retained_to_original_face"]]
    permutation = parent["source_corner_permutations"]
    assert np.array_equal(np.sort(permutation, axis=1), np.tile(np.arange(3), (len(m), 1)))
    assert np.array_equal(original[np.arange(len(m))[:, None], permutation], vids)
    for label, ids in (("C", ci), ("Y", yi)):
        assert np.array_equal(condition[label + "_source_vertex_ids"], vids[ids])
        assert np.array_equal(condition[label + "_source_corner_permutations"], permutation[ids])
    assert np.array_equal(condition["original_source_face_ids"], parent["retained_to_original_face"][ci])
    assert np.array_equal(condition["free_original_source_face_ids"], parent["retained_to_original_face"][yi])
    # Every repeated source vertex ID maps to the identical saved FP32 coordinate.
    coords = defaultdict(list)
    for fi, face in enumerate(vids):
        for corner, vid in enumerate(face):
            coords[int(vid)].append(m[fi, corner].tobytes())
    assert all(len(set(v)) == 1 for v in coords.values())
    return m, vids, original


def interface_audit(row, parent, condition, qparent):
    m, vids, original = validate(row, parent, condition)
    ci = {int(f): i for i, f in enumerate(condition["source_face_ids"])}
    yi = {int(f): i for i, f in enumerate(condition["free_source_ids"])}
    edges = defaultdict(list)
    for fi, face in enumerate(vids):
        for j in range(3):
            u, v = int(face[j]), int(face[(j + 1) % 3])
            edges[tuple(sorted((u, v)))].append((fi, j, (j + 1) % 3, u, v))
    contacts = {tuple(sorted(p["pair"])): p for p in qparent["pair_contacts"]}
    badfaces = set(qparent["basic_geometry"]["exact_zero_area_source_face_ids"])
    badfaces.update(qparent["basic_geometry"]["nonfinite_source_face_ids"])
    duplicate = {tuple(sorted((a, b))) for group in qparent["basic_geometry"]["exact_duplicate_groups"]
        for a in group for b in group if a != b}
    records, all_open_vertices, supported_y_corners, supported_c_vertices = [], set(), set(), set()
    for edge, incidents in sorted(edges.items()):
        known = [r for r in incidents if r[0] in ci]
        free = [r for r in incidents if r[0] in yi]
        if not known:
            continue
        if len(known) == 1:
            all_open_vertices.update(edge)
        if len(known) == 2 and not free:
            continue  # ordinary internal C edge is not a candidate boundary
        point0, point1 = parent["model_vertices"][list(edge)].astype(np.float64)
        length = float(np.linalg.norm(point1 - point0))
        classification = "NONMANIFOLD_OR_AMBIGUOUS"
        pair_label, opposite_original, opposite_canonical = None, None, None
        if len(incidents) == 1 and len(known) == 1:
            classification = "NATURAL_OPEN_SOURCE_EDGE"
        elif len(incidents) == 2 and len(known) == len(free) == 1:
            c, f = known[0], free[0]
            pair = tuple(sorted((c[0], f[0])))
            pair_label = contacts.get(pair, {}).get("classification", "MISSING_EXISTING_CONTACT_EVIDENCE")
            def direction(face):
                for j in range(3):
                    if tuple(sorted((int(face[j]), int(face[(j+1) % 3])))) == edge:
                        return (int(face[j]), int(face[(j+1) % 3]))
                raise AssertionError("Source original edge missing")
            opposite_original = direction(original[c[0]]) == direction(original[f[0]])[::-1]
            opposite_canonical = (c[3], c[4]) == (f[4], f[3])
            if length <= 0 or c[0] in badfaces or f[0] in badfaces or pair in duplicate:
                classification = "DEGENERATE_OR_DUPLICATE_INTERFACE"
            elif classify(pair_label) == "illegal":
                classification = "SOURCE_SHARED_EDGE_ILLEGAL_GEOMETRY"
            elif classify(pair_label) != "legal":
                classification = "SOURCE_SHARED_EDGE_UNKNOWN_GEOMETRY"
            else:
                # Winding is recorded separately because the canonical training representation
                # intentionally permutes corners; this is coordinate support, not oriented closure.
                classification = "SOURCE_SUPPORTED_INTERFACE"
                supported_c_vertices.update(edge)
                for corner, vid in enumerate(vids[f[0]]):
                    if int(vid) in edge:
                        supported_y_corners.add((yi[f[0]], corner))
        incident_rows = []
        for fi, j, k, u, v in incidents:
            is_c = fi in ci
            incident_rows.append(dict(source_face_id=fi,
                original_source_face_id=int(parent["retained_to_original_face"][fi]),
                role="C" if is_c else "Y", local_face_index=ci[fi] if is_c else yi[fi],
                local_corner_indices=[j, k], source_vertex_ids=[u, v],
                joint_C_then_Y_face_index=ci[fi] if is_c else len(ci)+yi[fi]))
        records.append(dict(edge_source_vertex_ids=list(edge), classification=classification,
            length_model_units=length, C_incidence=len(known), Y_incidence=len(free),
            total_incidence=len(incidents), original_winding_opposite=opposite_original,
            canonical_winding_opposite=opposite_canonical, geometric_pair_classification=pair_label,
            incidents=incident_rows))
    corners = [list(x) for x in sorted(supported_y_corners)]
    return dict(schema=VERSION, condition_id=row["condition_id"], uid=row["uid"], split=row["split"],
        source_condition_npz=row["npz"], source_condition_sha256=row["sha256"],
        role="AUDIT_AND_EVALUATION_STRATIFICATION_ONLY_NOT_MODEL_INPUT_OR_REQUIRED_INTERFACE",
        source_coordinate_correspondence="BITWISE_VERIFIED", face_partition="EXACT_COMPLEMENT_BITWISE",
        winding_policy="Source and canonical winding recorded separately; support is two-face nondegenerate nonduplicate legal geometric source incidence, not oriented watertightness.",
        all_C_source_open_boundary_vertex_ids=sorted(all_open_vertices),
        supported_C_source_vertex_ids=sorted(supported_c_vertices),
        supported_Y_local_corner_indices=corners,
        supported_Y_flat_corner_indices=[3*f+c for f,c in sorted(supported_y_corners)],
        supported_joint_C_then_Y_flat_corner_indices=[3*(len(ci)+f)+c for f,c in sorted(supported_y_corners)],
        classification_counts=dict(sorted(Counter(r["classification"] for r in records).items())),
        edge_records=records)


def main():
    # Do not create OUT until actual old-source/weight/outside path checks reject.
    negative = []
    for candidate in (SOURCE / "active_conditions.json", SOURCE / "dataset_handoff_active.json",
            ROOT / "experiments/r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_0030000.pt",
            OUT / "../unsafe.json"):
        try:
            safe_target(candidate, source=candidate)
        except PermissionError:
            negative.append(dict(path=str(candidate), status="REJECTED_BEFORE_OPEN"))
        else:
            raise AssertionError("Write guard accepted protected/outside path")
    protected = {}
    def protect(p, expected=None):
        p = Path(p)
        actual = sha(p)
        if expected is not None and expected != actual:
            raise ValueError("Input hash mismatch: " + str(p))
        protected[str(p)] = actual
        return p
    active_path = protect(SOURCE / "active_conditions.json")
    handoff_path = protect(SOURCE / "dataset_handoff_active.json")
    protect(SOURCE / "README_HANDOFF_ACTIVE.md")
    protect(Path(__file__).resolve())
    active, handoff = read(active_path), read(handoff_path)
    assert handoff["prepared_data"]["sha256"] == sha(active_path)
    assert len(active["rows"]) == 123
    assert len([r for r in active["rows"] if r["split"] == "train"]) == 117
    parent_cache, audits, exclusions, eligible, dev, interfaces = {}, [], [], [], [], []
    for row in sorted(active["rows"], key=lambda r: (r["split"], r["condition_id"])):
        uid, cid = row["uid"], row["condition_id"]
        assert cid not in active["held_condition_ids"]
        p_path = protect(row["parent_npz"], row["parent_sha256"])
        c_path = protect(row["npz"], row["sha256"])
        qp_path = protect(SOURCE / "quality/parents" / (uid + ".json"))
        qc_path = protect(SOURCE / "quality/conditions" / (cid + ".json"))
        if uid not in parent_cache:
            with np.load(p_path, allow_pickle=False) as z:
                parent_cache[uid] = {k: z[k].copy() for k in z.files}
        parent = parent_cache[uid]
        with np.load(c_path, allow_pickle=False) as z:
            condition = {k: z[k].copy() for k in z.files}
        qp, qc = read(qp_path), read(qc_path)
        assert qp["input_npz_sha256"] == row["parent_sha256"]
        assert qc["parent_quality_sha256"] == sha(qp_path)
        assert qc["source_face_ids"] == row["source_face_ids"]
        assert qc["K"] == row["K"] and qc["N"] == row["N"]
        # Independently repartition saved existing pair records; no new pair tests.
        counts = {g: Counter() for g in ("CC", "CF", "FF")}
        cs = set(row["source_face_ids"])
        for pair in qp["pair_contacts"]:
            i, j = pair["pair"]
            group = "CC" if i in cs and j in cs else "FF" if i not in cs and j not in cs else "CF"
            counts[group][classify(pair["classification"])] += 1
        reasons = []
        for group in ("CC", "CF", "FF"):
            assert counts[group]["illegal"] == qc["partitions"][group]["confirmed_illegal_pair_count"]
            assert counts[group]["unknown"] == qc["partitions"][group]["unknown_pair_count"]
            if group != "FF":
                for label in ("illegal", "unknown"):
                    if counts[group][label]:
                        reasons.append(group.replace("CF", "CY") + "_" + label.upper())
        interface = interface_audit(row, parent, condition, qp)
        interface_path = OUT / "interfaces" / (cid + ".json")
        write_json("interfaces/" + cid + ".json", interface)
        support = interface["classification_counts"].get("SOURCE_SUPPORTED_INTERFACE", 0)
        interfaces.append(interface)
        result = {k: row[k] for k in ("condition_id", "uid", "split", "region_type", "tier", "N", "K", "npz", "sha256", "parent_npz", "parent_sha256", "source_face_ids", "free_source_ids")}
        result.update(source_interface_json=str(interface_path), source_interface_sha256=sha(interface_path),
            source_supported_edge_count=support, quality_condition_json=str(qc_path),
            quality_condition_sha256=sha(qc_path), Y_illegal_pair_count=counts["FF"]["illegal"],
            Y_unknown_pair_count=counts["FF"]["unknown"], source_input_verified=True)
        decision = "KEEP_TRAIN" if row["split"] == "train" and not reasons else "EXCLUDE_TRAIN" if row["split"] == "train" else "KEEP_DEV_EVAL_ONLY"
        audit = dict(condition_id=cid, uid=uid, split=row["split"], region_type=row["region_type"],
            tier=row["tier"], N=row["N"], K=row["K"], decision=decision, reasons=reasons,
            CC_illegal=counts["CC"]["illegal"], CC_unknown=counts["CC"]["unknown"],
            CY_illegal=counts["CF"]["illegal"], CY_unknown=counts["CF"]["unknown"],
            Y_illegal=counts["FF"]["illegal"], Y_unknown=counts["FF"]["unknown"],
            source_supported_edge_count=support, source_supported_Y_corner_count=len(interface["supported_Y_local_corner_indices"]),
            natural_open_edge_count=interface["classification_counts"].get("NATURAL_OPEN_SOURCE_EDGE", 0),
            nonmanifold_or_ambiguous_edge_count=interface["classification_counts"].get("NONMANIFOLD_OR_AMBIGUOUS", 0),
            coordinate_partition_and_vertex_mapping="BITWISE_VERIFIED", source_interface_json=str(interface_path))
        audits.append(audit)
        if decision == "KEEP_TRAIN":
            eligible.append(result)
        elif decision == "EXCLUDE_TRAIN":
            exclusions.append(dict(**audit, existing_problem_pairs={g: qc["partitions"][g]["problem_pairs"] for g in ("CC", "CF")}))
        else:
            dev.append(result)
    parents = sorted({r["uid"] for r in eligible})
    supported = sum(r["source_supported_edge_count"] for r in eligible)
    status = "PASS" if len(parents) >= 8 and supported > 0 else "DATA_CONTRACT_BLOCKED"
    summary = dict(schema=VERSION, status=status, active_train_conditions=117, active_train_parents=25,
        eligible_train_conditions=len(eligible), eligible_train_parents=len(parents), excluded_train_conditions=len(exclusions),
        dev_conditions=len(dev), eligible_source_supported_edges=supported,
        eligible_conditions_with_source_supported_interface=sum(r["source_supported_edge_count"] > 0 for r in eligible),
        dev_source_supported_edges=sum(r["source_supported_edge_count"] for r in dev),
        eligible_conditions_with_Y_illegal=sum(r["Y_illegal_pair_count"] > 0 for r in eligible),
        eligible_conditions_with_Y_unknown=sum(r["Y_unknown_pair_count"] > 0 for r in eligible),
        training_parent_ids=parents, exclusion_rule="Existing confirmed illegal OR UNKNOWN in CC/CY only",
        Y_only_exclusion=False, generated_output_based_selection=False, full_M_clean_claim=False,
        required_user_interface_claim=False, semantic_labels_are_model_inputs=False,
        original_all_C_open_boundary_metric_unchanged=True,
        source_interface_definition="Two source-ID incident faces, one C one Y, exact coordinate correspondence, nonzero edge and face area, no exact duplicate and existing legal geometric pair classification; orientation is separately disclosed.",
        actual_counts=dict(model_loads=0, forwards=0, backwards=0, optimizer_updates=0, OT=0,
            Gaussian_arrays=0, new_geometry_pair_tests=0, source_writes=0),
        negative_write_tests=negative)
    whitelist = dict(schema=VERSION + "_training_whitelist", status=status, rows=eligible,
        parent_ids=parents, source_manifest=str(active_path), source_manifest_sha256=sha(active_path),
        shared_by_both_training_arms=True, parent_first_uniform_then_condition_uniform=True,
        input_arrays="Read existing NPZ C/Y; concatenate C,Y; source IDs/interfaces/region text NOT model inputs.")
    write_json("training_whitelist.json", whitelist)
    write_json("dev_conditions.json", dict(schema=VERSION + "_dev", rows=dev, training_eligible=False,
        note="All six active DEV retained; source defects disclosed, never used for training selection or loss."))
    write_json("exclusions.json", dict(schema=VERSION + "_excluded", rows=exclusions))
    write_csv("condition_audit.csv", audits)
    write_csv("exclusions.csv", [a for a in audits if a["decision"] == "EXCLUDE_TRAIN"])
    parent_rows = []
    for uid in sorted({a["uid"] for a in audits if a["split"] == "train"}):
        r = [a for a in audits if a["uid"] == uid]
        parent_rows.append(dict(uid=uid, active_conditions=len(r), eligible_conditions=sum(a["decision"] == "KEEP_TRAIN" for a in r),
            excluded_conditions=sum(a["decision"] == "EXCLUDE_TRAIN" for a in r),
            eligible_Y_illegal_conditions=sum(a["Y_illegal"] > 0 and a["decision"] == "KEEP_TRAIN" for a in r),
            eligible_Y_unknown_conditions=sum(a["Y_unknown"] > 0 and a["decision"] == "KEEP_TRAIN" for a in r)))
    write_csv("parents.csv", parent_rows)
    edge_rows = []
    for item in interfaces:
        for edge in item["edge_records"]:
            edge_rows.append(dict(condition_id=item["condition_id"], uid=item["uid"], split=item["split"], **edge))
    write_csv("SOURCE_SUPPORTED_INTERFACE.csv", edge_rows)
    write_json("summary.json", summary)
    for p, expected in protected.items():
        if sha(p) != expected:
            raise RuntimeError("PROTECTED_INPUT_CHANGED: " + p)
    write_json("protected_inputs.json", dict(before=protected, after=protected, all_unchanged=True))
    artifacts = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob("*") if p.is_file() and p.name != "completion.json"}
    write_json("completion.json", dict(schema=VERSION, status=status, artifacts=artifacts,
        protected_input_count=len(protected), source_code_sha256=sha(Path(__file__)),
        resume_policy="Verify all hashes and reuse immutable CPU artifacts; do not alter sources or regenerate conditions."))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
