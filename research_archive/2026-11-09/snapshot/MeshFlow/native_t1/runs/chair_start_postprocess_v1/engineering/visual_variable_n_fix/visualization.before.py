"""CPU display of frozen RAW / GEO_C / GT meshes, with no geometry processing.

Public API: render(case, outputdir, *, panel_size=280, cameras=None), and
run(runroot). A complete run uses one common full-field camera per parent over
all four noises and all three columns. A direct render uses a case-wide camera
unless the caller supplies the parent camera record. No model/scorer imports.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import re
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LP_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'native_t1/runs/chair_start_postprocess_v1'
RENDERER = ROOT / 'native_t1/runs/chair_two_triangle_control_v1/scripts/raw_renderer.py'
VERSION = 'chair_start_postprocess_gallery_v1'
LABELS = ('RAW', 'GEO_C', 'GT')
VIEWS = (('front_3q', 45., 22.), ('back_3q', 225., 22.),
         ('front', 0., 8.), ('side', 90., 8.), ('wire', 45., 22.))
_module = None


def resolve(value, root=None):
    text = str(value).replace('\\', '/')
    if os.name != 'nt' and len(text) > 2 and text[1:3] == ':/':
        text = '/mnt/' + text[0].lower() + text[2:]
    elif os.name == 'nt' and text.startswith('/mnt/') and len(text) > 7:
        text = text[5].upper() + ':' + text[6:]
    path = Path(text)
    return path if path.is_absolute() else Path(root or ROOT) / path


def digest(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((str(value.dtype), value.shape)).encode() + value.tobytes()).hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def _canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = _canonical(value)
    if path.exists():
        if _read(path) != value:
            raise ValueError('Frozen display metadata differs: ' + str(path))
        return
    with path.open('x', encoding='utf-8') as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _renderer():
    global _module
    if _module is None:
        # Software GL only. Importing this file does not create a GL context.
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
        os.environ['LP_NUM_THREADS'] = '2'
        spec = importlib.util.spec_from_file_location('chair_postprocess_display_renderer', RENDERER)
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
    return _module


def close_renderer():
    if _module is not None:
        _module.close_renderer()


def _font(size, bold=False):
    paths = ['/usr/share/fonts/truetype/dejavu/DejaVuSans' + ('-Bold' if bold else '') + '.ttf',
             'C:/Windows/Fonts/' + ('arialbd.ttf' if bold else 'arial.ttf')]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _triangles(value, label):
    value = np.asarray(value)
    if value.ndim == 2 and value.shape[1:] == (9,):
        value = value.reshape(-1, 3, 3)
    if value.ndim != 3 or value.shape[1:] != (3, 3) or value.dtype not in (np.float32, np.float64):
        raise ValueError(label + ': expected original float triangles [N,3,3] or [N,9]')
    if not np.isfinite(value).all():
        raise ValueError(label + ': nonfinite geometry cannot be hidden in the gallery')
    return value


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _arrays(path):
    if path.suffix.lower() == '.npy':
        return {'array': np.load(path, allow_pickle=False)}
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _pick(payload, keys, label):
    for key in keys:
        if key in payload:
            return payload[key]
    raise ValueError(label + ': missing keys ' + ', '.join(keys))


def load_case(case):
    """Read only unified frozen arrays; do not call an export/repair function."""
    root = resolve(case.get('runroot', OUT))
    cid = str(case['id'])
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', cid) or cid in ('.', '..'):
        raise ValueError('Case id must be a safe filename component')
    paths = dict(C=resolve(case['C_path'], root), RAW=resolve(case['raw_path'], root),
                 GEO_C=resolve(case.get('geo_path', 'outputs/GEO_C/' + cid + '.npz'), root),
                 GT=resolve(case.get('GT_path', case.get('queries_path')), root))
    if case.get('GT_path') is None and case.get('queries_path') is None:
        raise ValueError('A frozen GT path is required')
    sources = {str(path): digest(path) for path in paths.values()}
    c = _arrays(paths['C'])
    C = _triangles(_pick(c, ('C', 'context', 'array'), 'C'), 'C')
    k = int(case['K'])
    if k != len(C) or k <= 0:
        raise ValueError('Known face count differs')
    meshes = {}
    for label in ('RAW', 'GEO_C'):
        payload = _arrays(paths[label])
        mesh = _triangles(_pick(payload, ('output', 'raw', 'array'), label), label)
        if len(mesh) <= k or not _same(mesh[:k], C):
            raise ValueError(label + ': known C bytes/order differ from frozen condition')
        if 'vertices' in payload or 'faces' in payload:
            if not {'vertices', 'faces'}.issubset(payload):
                raise ValueError(label + ': incomplete indexed mesh export')
            vertices, faces = payload['vertices'], payload['faces']
            if faces.dtype.kind not in 'iu' or faces.shape != (len(mesh), 3):
                raise ValueError(label + ': invalid face index array')
            if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError(label + ': face index outside vertex array')
            if not _same(vertices[faces], mesh):
                raise ValueError(label + ': indexed mesh coordinates/order/winding differ from output')
        meshes[label] = mesh
    if meshes['RAW'].shape != meshes['GEO_C'].shape:
        raise ValueError('Displacement display requires unchanged face/corner correspondence')
    gt = _arrays(paths['GT'])
    if 'C' in gt and not _same(_triangles(gt['C'], 'GT C'), C):
        raise ValueError('GT condition differs from frozen C')
    if 'GT_free' in gt or 'GTfree' in gt:
        free = _triangles(_pick(gt, ('GT_free', 'GTfree'), 'GT'), 'GT free')
    else:
        full = _triangles(_pick(gt, ('GT_full', 'GTfull', 'GT', 'array'), 'GT'), 'GT full')
        # Remove exactly one matching occurrence of each C face, preserving all
        # remaining GT faces and winding. No tolerance or coordinate rewriting.
        face_slots = defaultdict(list)
        for index, face in enumerate(full):
            face_slots[(str(face.dtype), face.tobytes())].append(index)
        remove = set()
        for face in C:
            slots = face_slots[(str(face.dtype), face.tobytes())]
            if not slots:
                raise ValueError('Full GT does not contain byte-identical C faces')
            remove.add(slots.pop(0))
        free = full[[index for index in range(len(full)) if index not in remove]]
    meshes['GT'] = np.concatenate((C, free), axis=0)
    return dict(case=case, C=C, meshes=meshes, K=k, sources=sources,
                geometry_sha256={name: array_hash(mesh) for name, mesh in meshes.items()})


def prepare_cameras(cases):
    """Return JSON-safe parent cameras fitted to every registered full mesh."""
    grouped = defaultdict(list)
    source_files = {}
    ids = []
    for case in cases:
        data = load_case(case)
        ids.append(str(case['id']))
        grouped[str(case['parent'])].extend(data['meshes'].values())
        source_files.update(data['sources'])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Cases must be nonempty and unique')
    r = _renderer()
    cameras = {parent: {name: r.frame_for(values, az, el, 1.) for name, az, el in VIEWS}
               for parent, values in grouped.items()}
    return _canonical(dict(version=VERSION, status='FINAL_COMPLETE', cameras=cameras,
        scope='All registered noises and all RAW/GEO_C/GT geometry per parent; no outlier exclusion',
        case_ids=ids, source_files=source_files, views=VIEWS, renderer_sha256=digest(RENDERER)))


def _render_panel(C, free, path, camera, size, wire):
    r = _renderer()
    original = r._actor
    if wire:
        def wire_actor(triangles, color, edges=False):
            actor, receipt = original(triangles, color, edges=False)
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(1.)
            receipt.update(representation='wireframe', line_width=1.)
            return actor, receipt
        r._actor = wire_actor
    try:
        result = r.render_panel(C, free, path, camera, size=(size, size), mode='output')
    finally:
        r._actor = original
    if not any('llvmpipe' in line.lower() or 'softpipe' in line.lower()
               for line in result.get('gl_capabilities', [])):
        raise RuntimeError('The display context did not confirm software rendering')
    result['wireframe_display_only'] = bool(wire)
    result['style'] = dict(result['style'], representation='wireframe' if wire else 'surface')
    return result


def _layout(path, title, subtitle, columns, rows, row_names, size):
    left, top, gap, bottom = 112, 111, 7, 48
    page = Image.new('RGB', (left + len(columns) * (size + gap) + 8,
                             top + len(rows) * (size + gap) + bottom), 'white')
    draw = ImageDraw.Draw(page)
    draw.text((12, 10), title, font=_font(19, True), fill='#202833')
    draw.text((12, 39), subtitle, font=_font(13), fill='#465464')
    for index, column in enumerate(columns):
        draw.text((left + index * (size + gap) + 4, 79), column,
                  font=_font(17, True), fill='#202833')
    boxes = []
    for i, (row, name) in enumerate(zip(rows, row_names)):
        y = top + i * (size + gap)
        draw.text((8, y + size // 2 - 10), name, font=_font(13, True), fill='#465464')
        for j, source in enumerate(row):
            with Image.open(source) as image:
                image = image.convert('RGB')
                if image.size != (size, size):
                    raise ValueError('Layout does not crop or resize full-field panels')
                x = left + j * (size + gap)
                page.paste(image, (x, y))
                if page.crop((x, y, x + size, y + size)).tobytes() != image.tobytes():
                    raise AssertionError('Original panel pixels changed in layout')
            boxes.append(dict(row=i, column=j, path=str(source), sha256=digest(source),
                              xyxy=[x, y, x + size, y + size]))
    draw.text((12, page.height - 33), 'Blue: exact C. Orange: free (also GT). Full field; no smoothing or geometry edits.',
              font=_font(12), fill='#465464')
    page.save(path, compress_level=9, dpi=(150, 150))
    webp = path.with_suffix('.webp')
    page.save(webp, lossless=True, quality=100, method=6)
    with Image.open(webp) as check:
        if check.convert('RGB').tobytes() != page.tobytes():
            raise AssertionError('WebP delivery image is not pixel lossless')
    return dict(path=str(path), sha256=digest(path), webp_path=str(webp),
                webp_sha256=digest(webp), size=list(page.size), panel_boxes=boxes,
                full_panel_pixels_unchanged=True, webp_lossless_verified=True)


def _project(points, camera, size):
    relative = np.asarray(points, np.float64) - np.asarray(camera['focal_point'])
    x = .5 + relative @ np.asarray(camera['right']) / (2 * camera['parallel_scale'] * camera['aspect'])
    y = .5 - relative @ np.asarray(camera['view_up']) / (2 * camera['parallel_scale'])
    return np.column_stack((x, y)) * size


def _c_boundary(C):
    edges = {}
    counts = defaultdict(int)
    for face in C:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = face[i], face[j]
            # Exact coordinate keys, with +0/-0 treated as one vertex.
            key = tuple(sorted((tuple(map(float, a)), tuple(map(float, b)))))
            counts[key] += 1
            edges[key] = np.stack((a, b))
    return [edges[key] for key in sorted(edges) if counts[key] == 1]


def _overlay(background, path, C, raw, corrected, camera, size, arrows=False):
    with Image.open(background) as source:
        base = source.convert('RGBA')
    layer = Image.new('RGBA', base.size)
    draw = ImageDraw.Draw(layer)
    boundary = _c_boundary(C)
    for edge in boundary:
        a, b = _project(edge, camera, size)
        draw.line([tuple(a), tuple(b)], fill=(10, 125, 83, 220), width=2)
    shown, subpixel = 0, 0
    if arrows:
        before, after = raw[len(C):].reshape(-1, 3), corrected[len(C):].reshape(-1, 3)
        pairs = {}
        for a, b in zip(before, after):
            if not np.array_equal(a, b):
                pairs[(a.tobytes(), b.tobytes())] = (a, b)
        for a, b in pairs.values():
            p, q = _project(np.stack((a, b)), camera, size)
            distance = float(np.linalg.norm(q - p))
            if distance < .75:
                subpixel += 1
                continue
            draw.line([tuple(p), tuple(q)], fill=(192, 25, 132, 170), width=1)
            direction = (q - p) / distance
            normal = np.array([-direction[1], direction[0]])
            head = min(3., distance * .35)
            draw.polygon([tuple(q), tuple(q - head * direction + .45 * head * normal),
                          tuple(q - head * direction - .45 * head * normal)], fill=(192, 25, 132, 210))
            shown += 1
    Image.alpha_composite(base, layer).convert('RGB').save(path, compress_level=9)
    return dict(path=str(path), sha256=digest(path), background=str(background),
                background_sha256=digest(background), camera=camera,
                C_exact_boundary_edges=len(boundary), xray_overlay=True,
                displacement_arrows=shown, projected_subpixel_vectors_not_drawn=subpixel,
                displacement_scale=1., note='Green = C boundary, not a seam score. Magenta = actual RAW-to-GEO_C displacement; no amplification.')


def render(case, outputdir, *, panel_size=280, cameras=None):
    """Write one 3-column x 5-view page and two-view boundary/displacement page."""
    if not 240 <= int(panel_size) <= 600 or int(panel_size) != panel_size:
        raise ValueError('Panel size must be an integer in [240,600]')
    size = int(panel_size)
    tick = time.perf_counter()
    dest = resolve(outputdir)
    dest.mkdir(parents=True, exist_ok=True)
    data = load_case(case)
    C, meshes, k = data['C'], data['meshes'], data['K']
    if cameras is None:
        cameras = prepare_cameras([case])
    if str(case['id']) not in cameras['case_ids']:
        raise ValueError('Case is absent from the frozen camera union')
    if any(cameras['source_files'].get(path) != sha for path, sha in data['sources'].items()):
        raise ValueError('Current mesh differs from the frozen camera source')
    parent = str(case['parent'])
    camera = cameras['cameras'][parent]
    source_files = dict(data['sources'], **{str(Path(__file__)): digest(__file__), str(RENDERER): digest(RENDERER)})
    identity = _canonical(dict(version=VERSION, case_id=case['id'], parent=parent,
        noise=case['noise'], panel=case['panel'], size=size, labels=LABELS, views=VIEWS,
        camera_manifest=cameras, geometry_sha256=data['geometry_sha256'], source_files=source_files))
    manifest_path = dest / 'manifest.json'
    if manifest_path.exists():
        saved = _read(manifest_path)
        if saved.get('identity') != identity:
            raise ValueError('Existing gallery identity changed; do not overwrite')
        for path, sha in saved['artifact_hashes'].items():
            if digest(path) != sha:
                raise ValueError('Existing gallery artifact changed: ' + path)
        return saved
    panels = dest / 'panels'
    panels.mkdir(exist_ok=True)
    receipts, rows, artifacts = [], [], {}
    byview = {}
    for view, _, _ in VIEWS:
        row = []
        for label in LABELS:
            path = panels / (label + '__' + view + '.png')
            metadata = path.with_suffix('.json')
            panel_identity = _canonical(dict(C=array_hash(C), free=array_hash(meshes[label][k:]),
                camera=camera[view], size=size, wire=view == 'wire', source_sha256=identity['source_files']))
            if metadata.exists():
                receipt = _read(metadata)
                if receipt['display_identity'] != panel_identity or digest(path) != receipt['png_sha256']:
                    raise ValueError('Cached mesh panel identity differs')
            else:
                if path.exists():
                    raise RuntimeError('Unreceipted image exists; preserve it for explicit engineering recovery')
                receipt = _render_panel(C, meshes[label][k:], path, camera[view], size, view == 'wire')
                receipt['display_identity'] = panel_identity
                _write_new(metadata, receipt)
            artifacts[str(path)] = digest(path)
            artifacts[str(metadata)] = digest(metadata)
            receipts.append(dict(label=label, view=view, path=str(metadata), sha256=digest(metadata), render=receipt))
            row.append(path)
        rows.append(row)
        byview[view] = row
    title = str(case['panel']) + ' | ' + parent[:12] + ' | noise ' + str(case['noise'])
    main = _layout(dest / 'comparison.png', title, 'RAW / GEO_C / GT: common parent cameras across all registered noises.',
                   LABELS, rows, [v[0] for v in VIEWS], size)
    overlay_rows, overlay_receipts = [], []
    for view in ('front_3q', 'side'):
        raw_bg, geo_bg = byview[view][:2]
        overlay_row = []
        for name, background, arrows in (('RAW_boundary', raw_bg, False), ('GEO_C_boundary', geo_bg, False),
                                         ('displacement', raw_bg, True)):
            path = panels / (name + '__' + view + '.png')
            receipt = _overlay(background, path, C, meshes['RAW'], meshes['GEO_C'], camera[view], size, arrows)
            overlay_receipts.append(receipt)
            overlay_row.append(path)
            artifacts[str(path)] = digest(path)
        overlay_rows.append(overlay_row)
    detail = _layout(dest / 'boundary_displacement.png', title + ' | diagnostic display',
        'Green: exact C boundary (not score). Magenta: actual displacement, x-ray, scale 1.',
        ('RAW + C boundary', 'GEO_C + C boundary', 'Displacement on RAW'), overlay_rows,
        ('front_3q', 'side'), size)
    delta = meshes['GEO_C'].astype(np.float64) - meshes['RAW'].astype(np.float64)
    norms = np.linalg.norm(delta, axis=-1)
    for page in (main, detail):
        artifacts[page['path']] = page['sha256']
        artifacts[page['webp_path']] = page['webp_sha256']
    for path, sha in source_files.items():
        if digest(path) != sha:
            raise ValueError('Source bytes changed during rendering')
    result = dict(version=VERSION, status='COMPLETE', identity=identity,
        id=case['id'], parent=parent, noise=case['noise'], panel=case['panel'],
        comparison=main, boundary_displacement=detail, render_receipts=receipts,
        overlay_receipts=overlay_receipts, artifact_hashes=artifacts,
        changed_free_corners=int(np.count_nonzero(norms[k:])),
        max_free_corner_displacement=float(norms[k:].max()),
        known_C_unchanged=_same(meshes['RAW'][:k], meshes['GEO_C'][:k]),
        all_vertices_inside=all(r['render']['all_vertices_inside'] for r in receipts),
        input_arrays_unchanged=all(r['render']['input_arrays_unchanged'] for r in receipts),
        source_files=source_files, camera_status=cameras['status'],
        model_calls=0, scorer_calls=0, geometry_repair_calls=0,
        AI_VISUAL_REVIEW='PENDING', human_usability='NOT_RUN',
        local_cropped_mesh_views=False, initial_wall_seconds=time.perf_counter() - tick,
        software_GL=True, GL_workers=1, CPU_threads=2)
    _write_new(manifest_path, result)
    return _canonical(result)


def run(runroot=OUT, *, panel_size=280):
    """Render exactly the registered 32 cases once; cache verified completed pages."""
    out = resolve(runroot)
    document = _read(out / 'cases.json')
    cases = document if isinstance(document, list) else document['cases']
    if len(cases) != 32 or len({q['id'] for q in cases}) != 32:
        raise ValueError('Expected the complete fixed 32-case gallery')
    cases = [dict(q, runroot=str(out)) for q in cases]
    grouped = defaultdict(list)
    for case in cases:
        grouped[str(case['parent'])].append((case['panel'], int(case['noise'])))
    if len(grouped) != 8 or any(sorted(v) != [('FRESH', 2), ('FRESH', 3), ('OLD', 0), ('OLD', 1)] for v in grouped.values()):
        raise ValueError('Expected eight parents with OLD 0/1 and FRESH 2/3')
    tick = time.perf_counter()
    cameras = prepare_cameras(cases)
    directory = out / 'visuals'
    directory.mkdir(parents=True, exist_ok=True)
    camera_path = directory / 'cameras.json'
    _write_new(camera_path, cameras)
    records = []
    try:
        for case in cases:
            records.append(render(case, directory / case['id'], panel_size=panel_size, cameras=cameras))
    finally:
        close_renderer()
    contacts = []
    for panel in ('OLD', 'FRESH'):
        panel_records = [r for r in records if r['panel'] == panel]
        size, gap, left, top, row_height = 128, 5, 126, 65, 150
        image = Image.new('RGB', (left + 3 * (size + gap) + 12, top + len(panel_records) * row_height + 12), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((10, 8), panel + ' | all 16 cases; front 3/4 overview', font=_font(18, True), fill='#223')
        for j, label in enumerate(LABELS):
            draw.text((left + j * (size + gap), 38), label, font=_font(15, True), fill='#223')
        for i, record in enumerate(panel_records):
            y = top + i * row_height
            draw.text((8, y + 42), record['parent'][:12] + '\nn' + str(record['noise']), font=_font(12), fill='#465464')
            for j, box in enumerate(record['comparison']['panel_boxes'][:3]):
                with Image.open(box['path']) as small:
                    image.paste(small.convert('RGB').resize((size, size), Image.Resampling.LANCZOS),
                                (left + j * (size + gap), y))
        path = directory / ('contact_' + panel + '.webp')
        image.save(path, lossless=True, quality=100, method=6)
        contacts.append(dict(panel=panel, path=str(path), sha256=digest(path),
                             thumbnail_resampling='Lanczos; full-resolution pages retain original pixels'))
    body = ['<!doctype html><meta charset="utf-8"><title>START postprocessing comparison</title>',
            '<style>body{font:16px sans-serif;margin:24px;color:#223}img{max-width:100%;height:auto}a{color:#167}</style>',
            '<h1>RAW / GEO_C / GT</h1><p>32 frozen cases. Blue: exact known C. Orange: free, also in GT. ',
            'Four views and wireframe; one full-field camera per parent across all methods and noises. ',
            'No geometry smoothing or filtering. AI review pending; human usability not tested.</p>']
    for contact in contacts:
        body.append('<a href="' + Path(contact['path']).name + '">' + contact['panel'] + ' contact sheet</a><br>')
    for record in records:
        cid = record['id']
        body.extend(['<h2>' + html.escape(cid) + '</h2>',
            '<a href="' + cid + '/comparison.png">Original PNG</a> | ',
            '<a href="' + cid + '/boundary_displacement.webp">Boundary and actual displacement</a>',
            '<img loading="lazy" src="' + cid + '/comparison.webp">'])
    (directory / 'index.html').write_text('\n'.join(body), encoding='utf-8')
    if any(digest(path) != sha for path, sha in cameras['source_files'].items()):
        raise ValueError('A camera source changed during the completed gallery run')
    result = dict(version=VERSION, status='COMPLETE', cases=32, parents=8,
        camera_manifest=str(camera_path), camera_sha256=digest(camera_path), camera_status='FINAL_COMPLETE',
        pages=[dict(id=r['id'], parent=r['parent'], noise=r['noise'], panel=r['panel'],
                    path=r['comparison']['path'], sha256=r['comparison']['sha256'],
                    webp_path=r['comparison']['webp_path'], webp_sha256=r['comparison']['webp_sha256'],
                    detail_path=r['boundary_displacement']['webp_path'],
                    detail_sha256=r['boundary_displacement']['webp_sha256']) for r in records],
        contacts=contacts, index=str(directory / 'index.html'),
        all_vertices_inside=all(r['all_vertices_inside'] for r in records),
        all_geometry_unchanged=all(r['input_arrays_unchanged'] for r in records),
        initial_render_wall_seconds=sum(r['initial_wall_seconds'] for r in records),
        current_invocation_wall_seconds=time.perf_counter() - tick,
        model_calls=0, scorer_calls=0, geometry_repair_calls=0,
        delivery_recommendation='Package 32 lossless comparison.webp pages, two contacts and index; omit panel cache. Add diagnostic pages if byte budget permits.',
        delivery_comparison_bytes=sum(Path(r['comparison']['webp_path']).stat().st_size for r in records))
    # Current invocation time is intentionally separate from immutable first-render timings.
    (directory / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runroot', default=str(OUT))
    parser.add_argument('--panel-size', type=int, default=280)
    args = parser.parse_args()
    value = run(args.runroot, panel_size=args.panel_size)
    print(json.dumps({k: value[k] for k in ('status', 'cases', 'index', 'delivery_comparison_bytes')}))
