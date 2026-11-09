"""Frozen geometry-only bottle-train screening; no model, optimizer or repair."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, collections, datetime, hashlib, io, json, pickle, platform, shutil, tarfile, time, traceback, zipfile
from pathlib import Path
import numpy as np
from numpy.core import multiarray as numpy_multiarray
from ..exporter import RadialSupport
from ..io_utils import array_hash, file_hash, new_dir, save_npz, write_json
from ..legacy.geometry_task import fibonacci
from ..legacy.exact_accel import audit
from ..mesh_ops import topology, ray_hits

ARCHIVE_SHA = '93a7d0e3ab3f3caeb3ddb5e368c8e1f6384e638e2e5296a58d64c86c08b5330e'
PREFIX = 'shapenet-02876657/'
PARENT_SALT = 'a-condition16-parent-v1:'
DONOR_SALT = 'a-condition16-donor-v1:'
EXPECTED_TRAIN = 225


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ordered(uids, salt=PARENT_SALT):
    return sorted(uids, key=lambda u: (hashlib.sha256((salt + u).encode()).hexdigest(), u))


def donor_map(uids):
    uu = ordered(uids, DONOR_SALT)
    require(len(uu) >= 2 and len(uu) == len(set(uu)), 'Donor pool must have unique >=2 parents')
    return {uid: uu[(i + 1) % len(uu)] for i, uid in enumerate(uu)}


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('numpy', 'ndarray'):
            return np.ndarray
        if (module, name) == ('numpy', 'dtype'):
            return np.dtype
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            return getattr(numpy_multiarray, name)
        raise pickle.UnpicklingError((module, name))


def split_records(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        f = io.BytesIO(z.read('npz_list.npy'))
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        require(dtype.hasobject, 'Expected original object-array split metadata')
        value = Restricted(f).load()
        require(value.shape == shape and value.ndim == 1, 'Bad split metadata shape')
        records = value.tolist()
        require(all(isinstance(x, dict) and isinstance(x.get('uid'), str) for x in records), 'Bad original split UID record')
        return records


def mesh_arrays(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        rawv, rawf = np.array(z['vertices']), np.array(z['faces'])
        count = int(z['faces_num'])
    require(rawv.ndim == 2 and rawv.shape[1] == 3 and rawf.ndim == 2 and rawf.shape[1] == 3, 'Expected triangle arrays')
    require(np.issubdtype(rawf.dtype, np.integer), 'Non-integer face indices')
    v, f = rawv.astype(np.float64), rawf.astype(np.int64)
    require(len(f) == count and len(f) > 0, 'Source face metadata mismatch')
    require(np.isfinite(v).all() and f.min() >= 0 and f.max() < len(v), 'Nonfinite or invalid source indices')
    return v, f


def source_face_allowed(count):
    # User explicitly allows source F>512; only the later output template is512.
    return 20 < count < 800


def canonical_triangles_hash(v, f):
    """Exact float64 triangle-coordinate multiset, invariant to index/order/orientation."""
    tri = np.array(np.asarray(v, np.float64)[np.asarray(f, np.int64)], copy=True)
    tri[tri == 0] = 0.0  # IEEE signed zero has identical geometry.
    ordered_tri = []
    for t in tri:
        index = np.lexsort((t[:, 2], t[:, 1], t[:, 0]))
        ordered_tri.append(t[index].ravel())
    a = np.asarray(ordered_tri, dtype='<f8')
    order = np.lexsort(tuple(a[:, j] for j in range(8, -1, -1)))
    return array_hash(np.ascontiguousarray(a[order]))


def dependencies(repo):
    root = repo / 'experiments/a_radial'
    names = ['time_quality_followup/bottle_screen.py', 'mesh_ops.py', 'exporter.py', 'io_utils.py',
             'prepare_data.py', 'legacy/geometry_task.py', 'legacy/exact_accel.py',
             'legacy/exact_triangles.py', 'legacy/audit_fast.py', 'assets/support.npz']
    return {str(root / n): file_hash(root / n) for n in names}


def freeze(repo, source, out):
    repo, source = Path(repo).resolve(), Path(source).resolve()
    require(file_hash(source) == ARCHIVE_SHA, 'Archive source hash mismatch')
    old_path = repo / 'experiments/a_radial/assets/parents.json'
    old = read(old_path)
    old_uids = {x['uid'] for x in old['objects']}
    old_shas = {x['sha256'] for x in old['objects']}
    require(len(old_uids) == 16, 'Old development manifest changed')
    with tarfile.open(source, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)), 'Duplicate archive member names')
        train_blob = tar.extractfile(PREFIX + 'split/train.npz').read()
        records = split_records(train_blob)
        uids = [x['uid'] for x in records]
        require(len(uids) == EXPECTED_TRAIN and len(set(uids)) == EXPECTED_TRAIN, 'Expected exactly225 unique original train entries')
        require(not old_uids.intersection(uids), 'Bottle UID overlaps old16 development pool')
        require(all('/' not in u and '\\' not in u and u not in ('.', '..') for u in uids), 'Unsafe UID')
        inputs = {}
        for uid in sorted(uids):
            member = PREFIX + 'objaverse_occ_v5_ids/' + uid + '.npz'
            blob = tar.extractfile(member).read()
            sha = hashlib.sha256(blob).hexdigest()
            require(sha not in old_shas, 'Bottle source bytes overlap old16 development pool')
            inputs[uid] = {'archive_member': member, 'source_sha256': sha, 'source_bytes': len(blob)}
    ttq = repo / 'output/a_time_quality16_20260925_01/RUN_PROTOCOL.json'
    tp = read(ttq)
    old_fields = {uid: tp['data'][uid]['mainfield_hash'] for uid in tp['uids']}
    require(set(old_fields) == old_uids, 'Old field registry UID set changed')
    for uid in tp['uids']:
        require(file_hash(tp['data'][uid]['path']) == tp['data'][uid]['sha256'], 'Old field file changed')
    olddata = Path(tp['data'][sorted(tp['uids'])[0]]['path'])
    require(file_hash(olddata) == tp['data'][sorted(tp['uids'])[0]]['sha256'], 'Old template source changed')
    with np.load(olddata, allow_pickle=False) as z:
        base_u, faces = np.array(z['base_u'], dtype=np.float64), np.array(z['faces'], dtype=np.int32)
    require(base_u.shape == (258, 3) and faces.shape == (512, 3), 'Old final template mismatch')
    support = RadialSupport()
    dirs = np.vstack([support.u, fibonacci(8194)])
    require(dirs.shape == (10244, 3), 'Ray direction contract mismatch')
    out = new_dir(out).resolve()
    save_npz(out / 'TEMPLATE.npz', base_u=base_u, faces=faces, support_u=support.u,
             support_faces=support.faces.astype(np.int32), screening_directions=dirs)
    sources = dependencies(repo)
    sources[str(old_path)] = file_hash(old_path)
    sources[str(ttq)] = file_hash(ttq)
    for uid in tp['uids']:
        sources[tp['data'][uid]['path']] = tp['data'][uid]['sha256']
    sources[str(olddata)] = file_hash(olddata)
    study = repo / 'output/a_bottle_study_20260925_01/STUDY_PROTOCOL_v2.json'
    require(file_hash(study) == '46ad79b31d7a86ee6cafa4bfee87b3c96504b0220879c1be63c96b641cab497c', 'Root study registration changed')
    sources[str(study)] = file_hash(study)
    handoff = Path('/mnt/c/Users/warpwang/Downloads/A_MESHFLOW_CODEX_HANDOFF_20260924/A_MESHFLOW_CODEX_HANDOFF_20260924/tools/recover16.py')
    sources[str(handoff)] = file_hash(handoff)
    p = {'schema': 'A-BOTTLE-GEOMETRY-SCREEN-v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_archive': str(source), 'source_archive_sha256': ARCHIVE_SHA, 'source_synset': '02876657',
         'source_category': 'bottle', 'source_split': 'original_train_only', 'source_prefix': PREFIX,
         'train_count': EXPECTED_TRAIN, 'train_uids': sorted(uids), 'processing_uids': ordered(uids),
         'train_split_bytes_sha256': hashlib.sha256(train_blob).hexdigest(), 'inputs': inputs,
         'old_development_uids': sorted(old_uids), 'old_development_mainfield_hashes': old_fields, 'uid_overlap_with_old16': [], 'source_byte_hash_overlap_with_old16': [],
         'screening': {'source_face_count': '20<F<800', 'source_F_le512_required': False,
              'topology': 'unchanged mesh_ops.topology.closed_oriented_g0',
              'origin': 'source bbox centre', 'coordinate_normalization': 'divide by source bbox diagonal',
              'ray_count': 10244, 'ray_directions': 'original support2050 + fibonacci8194',
              'ray_acceptance': 'each direction has exactly one distinct positive hit using unchanged ray_hits tolerances',
              'exact': 'unchanged legacy.exact_accel.audit on original raw vertices/faces; embedded_endpoint_pass',
              'repairs_subdivision_simplification': False, 'model_optimizer_MC_calls': 0},
         'duplicate_rule': {'ordering': PARENT_SALT, 'keep': 'first eligible in parent hash order',
              'equivalence': ['identical raw NPZ SHA256', 'exact float64 unordered raw triangle coordinate multiset',
                              'exact float64 unordered bbox-normalized triangle coordinate multiset',
                              'bitwise identical normalized float32 support2050 logr'],
              'approximate_similarity_filter': False, 'all_duplicate_evidence_preserved': True,
              'mainfield_duplicate_semantics': 'identical representation input, not a claim of same underlying parent asset',
              'old_pool_exact_mainfield_alias': 'exclude and preserve old representative UID; no approximate matching'},
         'selection': {'required': 16, 'parent_hash_salt': PARENT_SALT, 'rule': 'first16 unique eligible after all225 screened',
              'insufficient_policy': 'record insufficient; no change of category/rules/thresholds and no substitute',
              'donor_hash_salt': DONOR_SALT, 'donor_rule': 'cyclic next of hash-sorted selected16',
              'post_selection_failure_replacement': False,
              'runtime_error_policy': 'any runtime/dependency/auditor exception blocks selection until separately investigated; errors are not geometry rejects'},
         'field_construction': 'logr=log(r0/sqrt(mean(r0*r0))).astype(float32), r0 first2050 unchanged normalized ray radii',
         'reference_semantics': 'same frozen2050node4096face radial field, not raw asset surface; raw source exact audit retained',
         'data_keys': ['logr', 'base_u', 'faces', 'ref_vertices', 'ref_faces'],
         'data_shapes': {'logr': [2050], 'base_u': [258, 3], 'faces': [512, 3], 'ref_vertices': [2050, 3], 'ref_faces': [4096, 3]},
         'template_path': str(out / 'TEMPLATE.npz'), 'template_sha256': file_hash(out / 'TEMPLATE.npz'),
         'template_hash': array_hash(base_u, faces), 'support_hash': array_hash(support.u, support.faces),
         'sources_sha256': sources, 'source_code_sha256': file_hash(__file__),
         'scope': 'new bottle category geometry screen; no claim all upstream pretraining provenance independently audited',
         'cost_scope': 'source screening and construction of frozen fields are offline preprocessing, excluded from later TTQ case timing',
         'root_study_protocol': str(study), 'root_study_protocol_sha256': file_hash(study),
         'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'cpu'}}
    write_json(out / 'SCREEN_PROTOCOL.json', p)
    return out, p


def run(repo, source, out):
    root, p = freeze(repo, source, out)
    protocol_sha = file_hash(root / 'SCREEN_PROTOCOL.json')
    print(json.dumps({'event': 'PROTOCOL_FROZEN', 'run': str(root), 'protocol_sha256': protocol_sha,
                      'script_sha256': p['source_code_sha256'], 'count': p['train_count']}), flush=True)
    with np.load(root / 'TEMPLATE.npz', allow_pickle=False) as z:
        template = {k: np.array(z[k]) for k in z.files}
    support = RadialSupport()
    dirs, base_u, faces = template['screening_directions'], template['base_u'], template['faces']
    rows, eligible = [], []
    seen = {('exact_mainfield', digest): 'historical:' + uid for uid, digest in p['old_development_mainfield_hashes'].items()}
    started = time.perf_counter()
    with tarfile.open(p['source_archive'], 'r:gz') as tar, (root / 'SCREEN_ROWS.jsonl').open('x', encoding='utf-8') as ledger:
        for index, uid in enumerate(p['processing_uids']):
            tick = time.perf_counter()
            row = {'uid': uid, 'index': index, **p['inputs'][uid], 'source_split': 'train', 'status': 'error'}
            print(json.dumps({'event': 'BEGIN', 'index': index + 1, 'uid': uid, 'eligible_so_far': len(eligible)}), flush=True)
            try:
                blob = tar.extractfile(row['archive_member']).read()
                require(hashlib.sha256(blob).hexdigest() == row['source_sha256'], 'Source changed after protocol')
                v, f = mesh_arrays(blob)
                row.update(vertices=len(v), faces=len(f), source_mesh_hash=array_hash(v, f),
                           source_canonical_triangles_hash=canonical_triangles_hash(v, f))
                row['source_face_filter_pass'] = source_face_allowed(len(f))
                row['topology'] = topology(v, f)
                row['status'] = 'topology_reject'
                if row['topology']['closed_oriented_g0']:
                    centre = (v.min(0) + v.max(0)) / 2
                    diag = float(np.linalg.norm(np.ptp(v, axis=0)))
                    require(np.isfinite(diag) and diag > 0, 'Invalid bbox diagonal')
                    normalized = (v - centre) / diag
                    row.update(bbox_centre=centre.tolist(), bbox_diagonal=diag,
                               normalized_canonical_triangles_hash=canonical_triangles_hash(normalized, f))
                    counts, radii = ray_hits(normalized, f, dirs)
                    row.update(missing_rays=int((counts == 0).sum()), multiple_rays=int((counts > 1).sum()),
                               ray_hit_count_hash=array_hash(counts), status='ray_reject')
                    if np.all(counts == 1):
                        row['exact'] = audit(v, f)
                        row['status'] = 'exact_reject'
                        if row['exact']['embedded_endpoint_pass']:
                            row['status'] = 'source_face_count_reject'
                            if row['source_face_filter_pass']:
                                r0 = radii[:2050]
                                require(np.isfinite(r0).all() and (r0 > 0).all(), 'Invalid radial values')
                                rs = float(np.sqrt(np.mean(r0 * r0)))
                                logr = np.log(r0 / rs).astype(np.float32)
                                rv, rf = support.mesh(logr)
                                row.update(radial_rms_bbox=rs, model_to_source_scale=diag * rs,
                                           mainfield_hash=array_hash(logr), reference_hash=array_hash(rv, rf),
                                           base_template_hash=array_hash(base_u, faces))
                                keys = [('source_bytes', row['source_sha256']), ('source_exact_triangles', row['source_canonical_triangles_hash']),
                                        ('normalized_exact_triangles', row['normalized_canonical_triangles_hash']),
                                        ('exact_mainfield', row['mainfield_hash'])]
                                matches = [{'kind': kind, 'sha256': digest, 'kept_uid': seen[(kind, digest)]}
                                           for kind, digest in keys if (kind, digest) in seen]
                                row['duplicate_matches'] = matches
                                old_overlap = any(x['kept_uid'].startswith('historical:') for x in matches)
                                row['status'] = 'old_representation_overlap' if old_overlap else ('exact_duplicate' if matches else 'eligible')
                                if not matches:
                                    for key in keys:
                                        seen[key] = uid
                                    path = root / 'candidates' / (uid + '.npz')
                                    save_npz(path, logr=logr, base_u=base_u, faces=faces,
                                             ref_vertices=rv, ref_faces=rf)
                                    row.update(candidate_data_path=str(path), candidate_data_sha256=file_hash(path))
                                    eligible.append(uid)
            except Exception as exc:
                row.update(status='error', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            row['screening_seconds'] = time.perf_counter() - tick
            ledger.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + '\n')
            ledger.flush()
            os.fsync(ledger.fileno())
            rows.append(row)
            print(json.dumps({'event': 'END', 'index': index + 1, 'uid': uid, 'status': row['status'],
                              'eligible_so_far': len(eligible), 'seconds': row['screening_seconds']}), flush=True)
    require(len(rows) == EXPECTED_TRAIN and {r['uid'] for r in rows} == set(p['train_uids']), 'Incomplete screening ledger')
    require(eligible == ordered(eligible), 'Eligible order drift')
    require(file_hash(root / 'SCREEN_PROTOCOL.json') == protocol_sha, 'Frozen protocol changed')
    require(file_hash(p['source_archive']) == ARCHIVE_SHA, 'Archive changed')
    for path, digest in p['sources_sha256'].items():
        require(file_hash(path) == digest, 'Frozen source changed: ' + path)
    runtime_errors = sum(r['status'] == 'error' for r in rows)
    selected = eligible[:16] if len(eligible) >= 16 and runtime_errors == 0 else []
    final_status = 'SCREEN_RUNTIME_ERROR' if runtime_errors else ('SELECTED16' if selected else 'INSUFFICIENT_ELIGIBLE')
    selected_rows = {}
    by_uid = {r['uid']: r for r in rows}
    with tarfile.open(p['source_archive'], 'r:gz') as tar:
        for uid in selected:
            row = by_uid[uid]
            path = root / 'data' / (uid + '.npz')
            path.parent.mkdir(exist_ok=True)
            with path.open('xb') as dst:
                dst.write(Path(row['candidate_data_path']).read_bytes())
            raw_path = root / 'raw' / (uid + '.npz')
            raw_path.parent.mkdir(exist_ok=True)
            with raw_path.open('xb') as dst:
                dst.write(tar.extractfile(row['archive_member']).read())
            require(file_hash(path) == row['candidate_data_sha256'] and file_hash(raw_path) == row['source_sha256'], 'Selected copy mismatch')
            selected_rows[uid] = {'path': str(path), 'sha256': file_hash(path), 'source_path': str(raw_path),
                                 'source_sha256': row['source_sha256'], 'archive_member': row['archive_member'],
                                 'source_faces': row['faces'], 'source_mesh_hash': row['source_mesh_hash'],
                                 'mainfield_hash': row['mainfield_hash'], 'reference_hash': row['reference_hash'],
                                 'split': 'new_bottle_train_geometry_screened_no_model_use'}
    selection = {'schema': 'A-BOTTLE-SELECTION-v1', 'protocol_path': str(root / 'SCREEN_PROTOCOL.json'),
                 'protocol_sha256': protocol_sha, 'source_archive_sha256': ARCHIVE_SHA,
                 'status': final_status, 'screened_count': len(rows),
                 'train_uids': p['train_uids'], 'eligible_uids_hash_order': eligible, 'eligible_count': len(eligible),
                 'selected16': selected, 'lex_sorted_uids': sorted(selected), 'donor_mapping': donor_map(selected) if selected else {},
                 'donor_uids_hash_order': ordered(selected, DONOR_SALT), 'data': selected_rows,
                 'counts': dict(collections.Counter(r['status'] for r in rows)),
                 'screen_rows_sha256': file_hash(root / 'SCREEN_ROWS.jsonl'),
                 'all225_completed_before_selection': True, 'model_optimizer_MC_calls': 0,
                 'post_selection_substitution_allowed': False}
    write_json(root / 'SELECTION.json', selection)
    manifest = {str(x.relative_to(root)): file_hash(x) for x in sorted(root.rglob('*')) if x.is_file()}
    write_json(root / 'FILE_MANIFEST.json', {'files': manifest})
    complete = {'status': selection['status'], 'screen_protocol_sha256': protocol_sha,
                'script_sha256': p['source_code_sha256'], 'selection_sha256': file_hash(root / 'SELECTION.json'),
                'file_manifest_sha256': file_hash(root / 'FILE_MANIFEST.json'), 'screened_count': len(rows),
                'eligible_count': len(eligible), 'selected_count': len(selected), 'counts': selection['counts'],
                'elapsed_seconds': time.perf_counter() - started, 'source_and_protocol_unchanged': True,
                'runtime_errors_preserved': sum(r['status'] == 'error' for r in rows), 'model_optimizer_MC_calls': 0}
    write_json(root / 'COMPLETE.json', complete)
    print(json.dumps({'event': 'COMPLETE', **complete}), flush=True)
    return complete


def self_test():
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    f = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [0, 2, 3]])
    perm = np.array([2, 0, 3, 1])
    inv = np.argsort(perm)
    require(canonical_triangles_hash(v, f) == canonical_triangles_hash(v[perm], inv[f][::-1, ::-1]), 'Duplicate permutation test')
    changed = v.copy(); changed[1, 0] = np.nextafter(1., 2.)
    require(canonical_triangles_hash(v, f) != canonical_triangles_hash(changed, f), 'No approximate duplicate tolerance')
    require(source_face_allowed(600) and not source_face_allowed(20) and not source_face_allowed(800), 'SourceF contract test')
    uu = [str(i) for i in range(16)]
    require(ordered(uu) == ordered(uu[::-1]), 'Selection input order invariance')
    donors = donor_map(uu)
    require(set(donors.values()) == set(uu) and all(k != v for k, v in donors.items()), 'Donor permutation test')
    try:
        Restricted(io.BytesIO(b"cos\nsystem\n.")).load()
    except pickle.UnpicklingError:
        pass
    else:
        raise AssertionError('Unsafe split pickle accepted')
    buf = io.BytesIO()
    np.savez_compressed(buf, npz_list=np.array([{'uid': 'metadata_roundtrip', 'faces_num': 600}], dtype=object))
    require(split_records(buf.getvalue()) == [{'uid': 'metadata_roundtrip', 'faces_num': 600}], 'Restricted original split roundtrip')
    print(json.dumps({'self_tests': 7, 'pass': True, 'models_or_geometry_screened': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--source', default='/mnt/c/Users/warpwang/Downloads/shapenet-rebuttal2.tar.gz')
    parser.add_argument('--out')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        require(args.out is not None, '--out required')
        run(args.repo, args.source, args.out)


if __name__ == '__main__':
    main()