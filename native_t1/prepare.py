"""Prepare the fixed two-parent Native T1 inputs from external official data.

The recipe contains identities and face indices, never coordinates. Official
preprocessing runs once per source with an instance-local noise callback; this
captures the original pre-OT scale without drawing noise or running OT. Known
faces and their ordered complement retain all 112 genuine source faces.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import numpy as np

from .data import FACE_COUNT, SUPPORTED_K, PatchCase, array_hash, file_sha256, load_archive

RECIPE_PATH = Path(__file__).parent / 'recipes' / 'chair_n112.json'
PERMUTATIONS = np.asarray(list(itertools.permutations(range(3))), dtype=np.int64)


def _source_path(data_root, uid):
    root = Path(data_root).expanduser().resolve()
    candidates = [root / f'{uid}.npz', root / 'objaverse_occ_v5_ids' / f'{uid}.npz']
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f'Expected exactly one {uid}.npz under {root} or '
                                f'{root / "objaverse_occ_v5_ids"}; found {len(found)}. '
                                'Point --data-root at the extracted official dataset or its NPZ directory.')
    return found[0]


def _read_source(path, parent):
    if file_sha256(path) != parent['source_sha256']:
        raise ValueError(f'Official source SHA256 mismatch: {path}')
    with np.load(path, allow_pickle=False) as saved:
        raw = {key: saved[key].copy() for key in ('vertices', 'faces', 'faces_num', 'uid')}
    vertices, faces = raw['vertices'], raw['faces']
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.shape != (FACE_COUNT, 3) or not np.issubdtype(faces.dtype, np.integer)
            or int(raw['faces_num']) != FACE_COUNT or np.any((faces < 0) | (faces >= len(vertices)))
            or str(raw['uid'].item()) != parent['uid'] or np.ptp(vertices, axis=0).max() <= 0):
        raise ValueError(f'Invalid real N112 source: {path}; no padding, truncation, or repair is allowed.')
    return raw


def _canonical_source(raw):
    # Lazy import: importing this module does not load any source or model.
    from datasets.mesh_dataset import ObjaverseDataset, sort_triangle_soup

    dataset = ObjaverseDataset.__new__(ObjaverseDataset)
    dataset.__dict__.update(training=True, use_custom_prior=False, do_dataset_normalize=True,
        max_face_length=800, use_rot_aug=False, use_scale_aug=False, noise_sort='ot',
        use_permut_aug=False, std=.3762, data=[raw])
    captured_pre, captured_vertices = [], []

    def capture_noise(coords):
        captured_pre.append(coords.copy())
        return coords, np.zeros_like(coords)

    def capture_vertices(vertices, faces, num_tokens=None):
        captured_vertices.append(vertices.copy())
        return ObjaverseDataset.sort_vertices_and_faces(dataset, vertices, faces, num_tokens)

    dataset.sample_noise = capture_noise
    dataset.sort_vertices_and_faces = capture_vertices
    sample = dataset[0]
    if len(captured_pre) != 1 or len(captured_vertices) != 1:
        raise RuntimeError('Official preprocessing did not capture exactly one original coordinate array.')
    pre = sort_triangle_soup(captured_pre[0]).reshape(FACE_COUNT, 3, 3)
    target = sort_triangle_soup(np.asarray(sample['coords'])).astype(np.float32).reshape(FACE_COUNT, 9)
    vertices = (captured_vertices[0] / dataset.std * 2).astype(np.float32)
    if not np.array_equal((pre * 2).astype(np.float32).reshape(FACE_COUNT, 9), target):
        raise RuntimeError('Official pre-OT/final x2 coordinate contract changed.')
    triangles = vertices[raw['faces']]
    canonical_to_raw, canonical_ids = [], []
    for triangle in target.reshape(FACE_COUNT, 3, 3):
        matches = (triangles[:, PERMUTATIONS, :] == triangle[None, None]).all(axis=(2, 3))
        source = np.flatnonzero(matches.any(axis=1))
        if len(source) != 1:
            raise ValueError('Source-to-canonical face mapping must be exact and unique.')
        face = int(source[0]); permutation = PERMUTATIONS[np.flatnonzero(matches[face])[0]]
        canonical_to_raw.append(face)
        canonical_ids.append(raw['faces'][face, permutation])
    ids = np.asarray(canonical_ids, dtype=np.int64)
    if len(set(canonical_to_raw)) != FACE_COUNT or not np.array_equal(vertices[ids], target.reshape(-1, 3, 3)):
        raise RuntimeError('Source vertex identity mapping is not a coordinate-preserving bijection.')
    return target, pre, vertices, ids, np.asarray(canonical_to_raw, dtype=np.int64)


def _patch_order(target, source_vertex_ids):
    """Original input-only connected-patch rule; no semantic labels or predictions."""
    edges = defaultdict(list)
    for index, face in enumerate(source_vertex_ids):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges[tuple(sorted((int(face[a]), int(face[b]))))].append(index)
    adjacency = [set() for _ in source_vertex_ids]
    for members in edges.values():
        for index in members:
            adjacency[index].update(other for other in members if other != index)
    triangles = np.asarray(target, dtype=np.float64).reshape(FACE_COUNT, 3, 3)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=-1) / 2
    if (areas <= 0).any():
        raise ValueError('A degenerate source face cannot be used as a patch input.')
    normals, centers = cross / (2 * areas[:, None]), triangles.mean(axis=1)
    lo, hi = triangles.reshape(-1, 3).min(0), triangles.reshape(-1, 3).max(0)
    height = (centers[:, 1] - lo[1]) / (hi[1] - lo[1])
    seeds = np.flatnonzero((height >= .2) & (height <= .75) & (abs(normals[:, 1]) >= .9))
    if not len(seeds):
        raise ValueError('Source has no eligible geometric patch seed.')
    seed = min(seeds, key=lambda i: (-areas[i], int(i)))
    level, xz_center = float(centers[seed, 1]), (lo + hi)[[0, 2]] / 2

    def score(index):
        return (float(abs(triangles[index, :, 1] - level).max()),
                float(np.linalg.norm(centers[index, [0, 2]] - xz_center)),
                float(-areas[index]), int(index))

    selected, frontier = [int(seed)], set(adjacency[seed])
    while len(selected) < 12:
        if not frontier:
            raise ValueError('Source has no edge-connected twelve-face patch.')
        face = min(frontier, key=score); frontier.remove(face); selected.append(face)
        frontier |= adjacency[face] - set(selected)
    return selected


def build_inputs(data_root):
    """Return training/condition arrays and provenance; no files, OT, or model calls."""
    recipe = json.loads(RECIPE_PATH.read_text(encoding='utf-8'))
    arrays, parents = {}, []
    for parent in recipe['parents']:
        index = parent['parent_index']; path = _source_path(data_root, parent['uid'])
        raw = _read_source(path, parent)
        target, pre, vertices, vertex_ids, face_ids = _canonical_source(raw)
        if (array_hash(target) != parent['target_array_sha256']
                or array_hash(pre) != parent['pre_ot_array_sha256']):
            raise RuntimeError('Official normalized coordinates differ from the pinned Native T1 recipe.')
        selected = _patch_order(target, vertex_ids)
        if selected != parent['nested_canonical_face_ids']:
            raise RuntimeError('Input-only connected patch selection differs from the fixed recipe.')
        arrays.update({f'parent{index}_target': target, f'parent{index}_pre_ot': pre,
            f'parent{index}_source_vertices': vertices, f'parent{index}_source_face_vertex_ids': vertex_ids,
            f'parent{index}_canonical_to_raw_face': face_ids})
        bbox = float(np.linalg.norm(np.ptp(target.reshape(-1, 3), axis=0)))
        for K in SUPPORTED_K:
            known = np.asarray(selected[:K], dtype=np.int64)
            free = np.asarray([i for i in range(FACE_COUNT) if i not in known], dtype=np.int64)
            case = PatchCase(index, K, target[known].reshape(K, 3, 3), target, target[free],
                             known, free, bbox, pre[free])
            for name in ('constraints', 'free_target', 'source_face_ids', 'free_source_ids', 'pre_ot_free'):
                arrays[f'parent{index}_K{K}_{name}'] = getattr(case, name)
        parents.append(dict(parent_index=index, uid=parent['uid'], source_path=str(path),
            source_sha256=parent['source_sha256'], source_vertex_count=len(raw['vertices']),
            source_face_count=len(raw['faces']), nested_canonical_face_ids=selected,
            target_sha256=array_hash(target), pre_ot_sha256=array_hash(pre), exact_source_mapping=True))
    audit = dict(schema='native_t1_prepared_inputs_v1', recipe_sha256=file_sha256(RECIPE_PATH),
        parents=parents, face_count=FACE_COUNT, training_K=[4, 8, 12], supported_K=list(SUPPORTED_K),
        case_count=8, OT_calls=0, model_forwards=0, optimizer_updates=0,
        sampling_noises_included=False, coordinates_repaired=False,
        arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_hash(value))
                for key, value in arrays.items()})
    return arrays, audit


def prepare(data_root, output):
    """Write one new NPZ, refusing any existing output. Returns its provenance."""
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != '.npz':
        raise ValueError('Output must be an explicit .npz file path.')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    arrays, audit = build_inputs(data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another process winning the race.
    with output.open('xb') as stream:
        np.savez_compressed(stream, **arrays, prepare_metadata_json=np.asarray(json.dumps(audit, sort_keys=True)))
    archive = load_archive(output)
    if len(archive.cases) != 8:
        raise RuntimeError('Prepared archive failed the Native T1 loader contract.')
    audit['output'] = dict(path=str(output), sha256=archive.sha256, bytes=output.stat().st_size)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path,
                        help='Extracted official ShapeNet directory, or its objaverse_occ_v5_ids directory.')
    parser.add_argument('--output', required=True, type=Path, help='New output .npz; existing files are rejected.')
    args = parser.parse_args(argv)
    audit = prepare(args.data_root, args.output)
    print(json.dumps(dict(output=audit['output'], parents=audit['parents'], case_count=audit['case_count'],
                         training_K=audit['training_K'], OT_calls=0, model_forwards=0), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
