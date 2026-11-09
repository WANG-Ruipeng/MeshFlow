"""Independent CPU arithmetic checks for partial edge binding outputs.

No decoder, model, renderer or quality scorer is imported.  Geometry remains in
the original input-corner domain until the existing output source maps are used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _require(condition, description):
    if not condition:
        raise AssertionError(description)


def _triangle_geometry(tri):
    """Direct independent arithmetic in a common source-corner order."""
    tri = np.asarray(tri, np.float64)
    edges = np.roll(tri, -1, axis=0) - tri
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    double_area = float(np.linalg.norm(normal))
    angles = []
    for corner in range(3):
        a, b = tri[(corner + 1) % 3] - tri[corner], tri[(corner + 2) % 3] - tri[corner]
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        angles.append(float(np.degrees(np.arccos(np.clip(np.dot(a, b) / den, -1, 1)))) if den else 0.)
    return dict(area=double_area / 2, minimum_angle=min(angles),
                q=float((edges * edges).sum() / (2 * np.sqrt(3) * double_area)) if double_area else float('inf'),
                normal=normal / double_area if double_area else normal)


def verify_arrays(baseline, final, pre_geo, K, mode, delta_model,
                  assignments, *, reference_length=None):
    """Check indexed assignment and all original cluster member displacement.

    ``assignments`` maps flattened *original pre_geo* corner IDs to retained
    baseline C vertex IDs.  Include removed-face cluster members in B.  Source
    corner IDs in final output must point into that same original input domain.
    Baseline and final arrays are dictionaries loaded with allow_pickle=False.
    """
    K = int(K)
    pre = np.asarray(pre_geo, np.float32)
    _require(pre.ndim == 3 and pre.shape[1:] == (3, 3), "pre_geo shape")
    bvertices = np.asarray(baseline["vertices"])
    bfaces = np.asarray(baseline["faces"])
    vertices = np.asarray(final["vertices"])
    faces = np.asarray(final["faces"])
    _require(same_bits(vertices, bvertices), "baseline representatives modified")
    _require(faces.ndim == 2 and faces.shape[1] == 3, "output face shape")
    _require(np.issubdtype(faces.dtype, np.integer), "non-integer indexed faces")
    _require(np.isfinite(vertices).all(), "nonfinite vertices")
    _require(not faces.size or (faces.min() >= 0 and faces.max() < len(vertices)), "face index range")
    _require(same_bits(faces[:K], bfaces[:K]), "source C index/order changed")
    _require(same_bits(vertices[faces[:K]], bvertices[bfaces[:K]]), "source C coordinate bytes changed")
    _require(same_bits(vertices[faces], final["output"]), "indexed output roundtrip")
    if "C" in final:
        _require(same_bits(vertices[faces[:K]], final["C"]), "C output view")
    if "free" in final:
        _require(same_bits(vertices[faces[K:]], final["free"]), "free output view")
    bs = np.asarray(baseline["source_face_ids"], np.int64)
    fs = np.asarray(final["source_face_ids"], np.int64)
    sc = np.asarray(final["source_corner_ids"], np.int64)
    _require(fs.shape == (len(faces),) and len(set(fs.tolist())) == len(fs), "final face identity")
    _require(sc.shape == faces.shape and np.array_equal(np.sort(sc, axis=1), np.tile(np.arange(3), (len(faces), 1))), "final corner permutations")
    _require(np.array_equal(fs[:K], bs[:K]), "source C face IDs changed")
    _require(set(fs.tolist()) <= set(bs.tolist()), "deleted baseline face resurrected")
    _require(np.array_equal(fs, bs[np.isin(bs, fs)]), "baseline source face order changed")
    original_ids = np.asarray(baseline.get("input_corner_vertex_ids", baseline["corner_vertex_ids"]), np.int64)
    _require(original_ids.shape == pre.shape[:2], "original cluster map shape")
    original_ids = original_ids.reshape(-1)
    anchor_ids = set(int(x) for x in bfaces[:K].reshape(-1))
    assign = {int(k): int(v) for k, v in assignments.items()}
    target = original_ids.copy()
    maximum = 0.0
    checked_clusters = set()
    for corner, destination in sorted(assign.items()):
        _require(3 * K <= corner < 3 * len(pre), "bound corner is not free")
        _require(destination in anchor_ids, "assignment is not a real C indexed vertex")
        old = int(original_ids[corner])
        _require(old >= 0, "assignment references absent original cluster")
        _require(old not in anchor_ids or old == destination, "existing C anchor changed identity")
        displacement = float(np.linalg.norm(pre.reshape(-1, 3)[corner].astype(np.float64) - bvertices[destination].astype(np.float64)))
        _require(displacement <= float(delta_model) + 1e-12, "pre_geo displacement exceeds hard budget")
        maximum = max(maximum, displacement)
        target[corner] = destination
        if mode == "B" and old not in checked_clusters:
            members = np.flatnonzero(original_ids == old)
            members = members[members >= 3 * K]
            _require(all(assign.get(int(member)) == destination for member in members), "B did not assign every original GEO_C cluster member")
            checked_clusters.add(old)
    expected_ids = target.reshape(-1, 3)[fs[:, None], sc]
    _require(np.array_equal(faces, expected_ids), "final indices differ from declared corner assignments")
    affected_sources = sorted(set(corner // 3 for corner in assign) & set(bs[K:].tolist()))
    maximum_physical_normal_rotation = 0.
    for source in affected_sources:
        original = _triangle_geometry(bvertices[original_ids[3*source:3*source+3]])
        now_ids = target[3*source:3*source+3]
        _require(len(set(now_ids.tolist())) == 3, "affected face has duplicate vertex IDs")
        current = _triangle_geometry(bvertices[now_ids])
        if reference_length is not None:
            _require(current['area'] > 1e-12 * reference_length**2, "affected area below locked floor")
        _require(not (current['minimum_angle'] < 5 and current['minimum_angle'] < original['minimum_angle'] - 1e-9), "new/worse small angle")
        _require(not (current['q'] > 10 and current['q'] > original['q'] + 1e-9), "new/worse q above10")
        cosine = float(np.dot(original['normal'], current['normal']))
        _require(cosine >= .5 - 1e-10, "physical normal rotation exceeds60degrees")
        maximum_physical_normal_rotation = max(maximum_physical_normal_rotation, float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    # Deleted-face provenance is checked independently of final orientation.
    expected_source = []
    seen = set()
    deletion = []
    baseline_tri = bvertices[bfaces]
    for row, source in enumerate(bs):
        canonical_ids = target[3 * source:3 * source + 3]
        tri = bvertices[canonical_ids]
        numeric = [tuple(float(c) for c in xyz) for xyz in tri]
        key = tuple(sorted(numeric))
        degenerate = len(set(numeric)) < 3 or bool(np.all(np.cross(tri[1].astype(float) - tri[0], tri[2].astype(float) - tri[0]) == 0))
        reason = None if row < K else "exact_degenerate" if degenerate else "duplicate" if key in seen else None
        if reason is None:
            expected_source.append(int(source))
            seen.add(key)
        else:
            area = .5 * float(np.linalg.norm(np.cross(baseline_tri[row, 1].astype(float) - baseline_tri[row, 0], baseline_tri[row, 2].astype(float) - baseline_tri[row, 0])))
            deletion.append(dict(source_face_id=int(source), reason=reason, baseline_area=area))
    _require(np.array_equal(fs, expected_source), "final exact cleanup does not match declared assignments")
    return dict(status="PASS", mode=mode, source_C_faces=K, output_faces=len(faces),
                assigned_input_corners=len(assign), checked_B_clusters=len(checked_clusters),
                max_displacement_model=maximum,
                independently_checked_affected_faces=len(affected_sources),
                maximum_physical_normal_rotation_degrees=maximum_physical_normal_rotation,
                max_displacement_percent_L=None if reference_length is None else 100 * maximum / reference_length,
                removed_baseline_faces=len(deletion), removed_baseline_area=sum(x["baseline_area"] for x in deletion),
                deleted_faces=deletion, model_calls=0, quality_calls=0)


def verify_protected(records):
    """Verify frozen source file hashes; accepts {path: hash} or record rows."""
    if isinstance(records, dict):
        pairs = records.items()
    else:
        pairs = ((row["path"], row["sha256"]) for row in records)
    count = 0
    for filename, expected in pairs:
        _require(sha256(filename) == expected, "protected original changed: " + str(filename))
        count += 1
    return count


def verify_candidate_records(asset, records, summary, final):
    """Verify the public candidate funnel, capacities, ranking and cap witnesses."""
    from collections import Counter
    pre = np.asarray(asset['pre_geo'], np.float64).reshape(-1, 3)
    base = asset['canonical']; vertices = np.asarray(base['vertices'], np.float64)
    original_ids = base['corner_vertex_ids'].reshape(-1)
    sf = base['source_face_ids']; K = int(asset['row']['K'])
    counts = Counter(r['decision'] for r in records)
    _require(dict(counts) == summary['rejections'], 'candidate funnel does not partition candidates')
    _require(len(records) == summary['candidate_count'], 'candidate total mismatch')
    _require(counts['ACCEPTED'] == summary['accepted'], 'accepted total mismatch')
    previous = None; used_C = set(); used_free = set(); declared = {}
    for rank, row in enumerate(records):
        _require(row['rank'] == rank, 'candidate ranks not consecutive')
        key = (row['max_displacement'], row['source_face_id'], *row['local_edge'], *row['C_edge'], *row['target_C_vertex_ids'])
        _require(previous is None or key >= previous, 'candidate ranking not deterministic displacement order')
        previous = key
        fi = int(row['face_row']); a, b = map(int, row['local_edge'])
        _require(fi >= K and row['source_face_id'] == int(sf[fi]), 'candidate source face identity')
        targets = list(map(int, row['target_C_vertex_ids']))
        _require(sorted(targets) == sorted(row['C_edge']), 'candidate endpoint targets not claimed C edge')
        endpoints = [3*int(sf[fi])+a, 3*int(sf[fi])+b]
        actual_max = 0.
        for corner, old_vid, members, destination in zip(endpoints, row['free_vertex_ids'], row['assignment_input_corner_ids'], targets):
            _require(original_ids[corner] == old_vid, 'candidate baseline corner identity')
            expected = [corner] if row['mode'] == 'A' else np.flatnonzero((original_ids == old_vid) & (np.arange(len(original_ids)) >= 3*K)).tolist()
            _require(members == expected, 'candidate has wrong true cluster membership')
            displacement = float(np.linalg.norm(pre[members]-vertices[destination], axis=1).max())
            actual_max = max(actual_max, displacement)
            if row['decision'] == 'ACCEPTED':
                for member in members:
                    _require(member not in declared or declared[member] == destination, 'accepted corner anchor conflict')
                    declared[member] = destination
        _require(abs(actual_max-row['max_displacement']) <= 1e-11, 'candidate displacement witness mismatch')
        endpoint_max = max(float(np.linalg.norm(pre[corner]-vertices[destination])) for corner, destination in zip(endpoints, targets))
        _require(abs(endpoint_max-row['endpoint_max_displacement']) <= 1e-11, 'candidate endpoint cap witness mismatch')
        _require(endpoint_max <= summary['delta_model'] + 1e-12, 'candidate entered outside endpoint budget')
        if row['decision'] == 'ACCEPTED':
            _require(actual_max <= summary['delta_model'] + 1e-12, 'accepted candidate over member cap')
            ce = tuple(sorted(row['C_edge'])); fe = (fi, a, b)
            _require(ce not in used_C and fe not in used_free, 'accepted edge capacity duplicated')
            used_C.add(ce); used_free.add(fe)
    actual = dict(zip(map(int, final['assignment_input_corner_ids']), map(int, final['assignment_target_C_vertex_ids'])))
    _require(actual == declared, 'output assignment list differs from accepted candidate records')
    return dict(status='PASS', candidate_records=len(records), accepted=counts['ACCEPTED'], reasons=dict(counts), assigned_corners=len(declared))


def _accounting_check(before, after, paired, before_pairs, after_pairs):
    """Independent set arithmetic; original pair classification is not rerun."""
    summary = paired['summary']
    def edge_key(row):
        return tuple(tuple(map(int, part)) for part in row['source_edge_key'])
    for kind in ('CF', 'FF'):
        old = {edge_key(row):row for row in before['edges'] if row['kind']==kind and row['valid_edge']}
        new = {edge_key(row):row for row in after['edges'] if row['kind']==kind and row['valid_edge']}
        retained, lost, gained = set(old)&set(new), set(old)-set(new), set(new)-set(old)
        for key, value in ((kind+'_baseline_legal_edges',len(old)),(kind+'_retained_legal_edges',len(retained)),(kind+'_lost_legal_edges',len(lost)),(kind+'_new_legal_edges',len(gained))):
            _require(summary[key] == value, 'edge fate accounting mismatch: '+key)
        for key, value in ((kind+'_baseline_legal_length',sum(r['length_model'] for r in old.values())),(kind+'_retained_baseline_length',sum(old[k]['length_model'] for k in retained)),(kind+'_retained_current_length',sum(new[k]['length_model'] for k in retained)),(kind+'_lost_baseline_length',sum(old[k]['length_model'] for k in lost)),(kind+'_new_current_length',sum(new[k]['length_model'] for k in gained))):
            _require(abs(summary[key]-value)<1e-9, 'edge length accounting mismatch: '+key)
        def illegal(pairs, ids, K):
            result=set()
            for row in pairs[kind]['contacts']:
                if row['classification'] not in ('proper_transverse_penetration','coplanar_positive_area_overlap'):
                    continue
                i,j=map(int,row['pair'])
                result.add((int(ids[i]),int(ids[K+j])) if kind=='CF' else tuple(sorted((int(ids[K+i]),int(ids[K+j])))))
            return result
        oi=illegal(before_pairs,before['source_face_ids'],before['K'])
        ni=illegal(after_pairs,after['source_face_ids'],after['K'])
        for key,value in ((kind+'_illegal_pairs_persisted',len(oi&ni)),(kind+'_illegal_pairs_removed',len(oi-ni)),(kind+'_new_illegal_pairs',len(ni-oi))):
            _require(summary[key]==value,'illegal facing-pair accounting mismatch: '+key)
        if kind=='CF': _require(not (ni-oi),'final mesh has newly illegal CF pair')
    return dict(status='PASS', CF_baseline_legal_edges=summary['CF_baseline_legal_edges'], CF_retained=summary['CF_retained_legal_edges'], CF_lost=summary['CF_lost_legal_edges'], FF_lost=summary['FF_lost_legal_edges'])


def verify_all(out=None, manifest=None):
    """Audit exactly the 144 saved cells; zero decoder/scoring/model calls."""
    from . import assets as A
    out=A.OUT if out is None else A.resolve(out)
    manifest=A.read(out/'input_manifest.json') if manifest is None else manifest
    rows=manifest['rows'] if isinstance(manifest,dict) else manifest
    labels=[f'{m}_{d:03d}' for m in ('A','B') for d in (10,20,35)]
    expected={(label,row['short_id']) for label in labels for row in rows}
    actual={(p.parent.parent.name,p.parent.name) for p in (out/'cells').glob('*/*/complete.json')}
    _require(len(rows)==24 and len(expected)==144 and actual==expected,'incomplete/extra/duplicate scientific cells')
    reg=A.read(out/'registration.json')
    code_hashes=dict(reg['source_sha256']); amendments=[]
    amendment_path=out/'engineering_amendments.json'
    if amendment_path.exists():
        amendments=A.read(amendment_path)['items']
        for change in amendments:
            _require(code_hashes.get(change['path'])==change['old_sha256'],'broken engineering amendment hash chain')
            _require(bool(change.get('reason')),'undocumented engineering amendment')
            code_hashes[change['path']]=change['new_sha256']
    A.verify(reg['protected_files']); A.verify(code_hashes)
    cells=[]
    for row in rows:
        asset=A.load_asset(row,verify_hashes=True)
        before=A.read(out/'baseline'/row['short_id']/'connectivity.json')
        before_pairs=A.read(A.resolve(row['score_path']).parent/'pairs.json')
        for label in labels:
            dest=out/'cells'/label/row['short_id']
            done=A.read(dest/'complete.json'); A.verify(done['files'])
            _require(done['status']=='COMPLETE' and done['model_calls']==0 and done['CPU_only'] is True,'cell completion does not assert zero model calls')
            final=A.arrays(dest/'post.npz'); records=A.read(dest/'candidate_records.json'); summary=A.read(dest/'binding.json')
            assignments=dict(zip(map(int,final['assignment_input_corner_ids']),map(int,final['assignment_target_C_vertex_ids'])))
            array_check=verify_arrays(asset['post'],final,asset['pre_geo'],row['K'],summary['mode'],summary['delta_model'],assignments,reference_length=reg['L'])
            candidate_check=verify_candidate_records(asset,records,summary,final)
            after=A.read(dest/'connectivity.json'); paired=A.read(dest/'paired_ledger.json'); score=A.read(dest/'score.json')
            accounting=_accounting_check(before,after,paired,before_pairs,score['pairs'])
            _require(array_check['removed_baseline_faces']==paired['summary']['deleted_source_faces'],'deleted-face accounting count mismatch')
            _require(abs(array_check['removed_baseline_area']-paired['summary']['deleted_source_area'])<1e-9,'deleted-face accounting area mismatch')
            cells.append(dict(label=label,request_id=row['request_id'],short_id=row['short_id'],arrays=array_check,candidates=candidate_check,accounting=accounting))
    A.verify(reg['protected_files'])
    result=dict(status='PASS',verified_cells=len(cells),verified_requests=len(rows),protected_files=len(reg['protected_files']),
                input_manifest_sha256=A.sha(out/'input_manifest.json'),verifier_sha256=sha256(__file__),
                engineering_amendments=len(amendments), engineering_amendment_sha256=A.sha(amendment_path) if amendments else None,
                final_scientific_code_sha256=code_hashes,
                model_calls=0,decoder_calls=0,quality_calls=0,scope='Independent array/shape/membership/accounting arithmetic on all144 cached cells; original geometric predicate results reused, not independent intersection implementation',cells=cells)
    target=out/'independent_verification.json'
    serialized=json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    if target.exists() and target.read_text(encoding='utf-8')!=serialized:
        raise ValueError('Refusing to overwrite different independent verification')
    if not target.exists():target.write_text(serialized,encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out');args=parser.parse_args()
    result=verify_all(args.out)
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
