"""Depth-correct CPU review pages for Bench user-data v2 proposals.

This is a renderer, not a semantic or geometry validator. It reads parent
original/target meshes and exact target face IDs, writes only its gallery,
and leaves every page NOT_REVIEWED until a separate actual AI_PROXY review.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import collections
import hashlib
import html
import io
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from r2.pilots.regular_patch_control_v1.depth_gallery import basis, raster

ROOT = Path(__file__).resolve().parents[3]
GALLERY = ROOT / 'experiments/r2_bench_user_data_v2/gallery'
RASTER_SOURCE = ROOT / 'r2/pilots/regular_patch_control_v1/depth_gallery.py'
VERSION = 'BENCH_USER_V2_CPU_SOURCE_REVIEW_V1'
VIEWS = [('oblique_A', -60., 22.), ('oblique_B', 120., 22.)]
CELL = 450
FONT_CACHE = {}


def resolve(value):
    value = str(value).replace('\\', '/')
    if os.name != 'nt' and re.match(r'^[A-Za-z]:/', value):
        value = '/mnt/' + value[0].lower() + value[2:]
    elif os.name == 'nt' and re.match(r'^/mnt/[a-z]/', value):
        value = value[5].upper() + ':/' + value[7:]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def immutable_bytes(path, data):
    path = Path(path)
    if not path.resolve().is_relative_to(GALLERY.resolve()):
        raise ValueError('Renderer write outside owned gallery: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError('Changed output refused; use a new output name: ' + str(path))
    else:
        with path.open('xb') as f:
            f.write(data)


def write_json(path, value):
    immutable_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8'))


def font(size):
    if size not in FONT_CACHE:
        candidates = [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/mnt/c/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyh.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf',
        ]
        path = next((p for p in candidates if Path(p).is_file()), None)
        FONT_CACHE[size] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    return FONT_CACHE[size]


def text_lines(draw, xy, text, width, size=16, max_lines=3, color='#333333'):
    """Wrap by measured glyph width, including CJK, without changing source text."""
    x, y = xy
    ft = font(size)
    lines, current = [], ''
    for char in str(text):
        if char == '\n' or (current and draw.textlength(current + char, font=ft) > width):
            lines.append(current)
            current = '' if char == '\n' else char
        else:
            current += char
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-3] + '...'
    for i, line in enumerate(lines):
        draw.text((x, y + i * (size + 5)), line, font=ft, fill=color)


def checked_mesh(path, expected_hash):
    path = resolve(path)
    if not expected_hash or sha(path) != expected_hash:
        raise ValueError('Mesh hash missing/mismatch: ' + str(path))
    with np.load(path, allow_pickle=False) as z:
        vertices, faces = z['vertices'].copy(), z['faces'].copy()
    if vertices.dtype != np.float32 or faces.dtype != np.int64:
        raise ValueError('Expected model-coordinate float32 vertices and int64 original-winding faces: ' + str(path))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Invalid mesh dimensions: ' + str(path))
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError('Empty/nonfinite mesh: ' + str(path))
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid mesh face indices: ' + str(path))
    return vertices[faces], dict(path=str(path), sha256=expected_hash,
                                vertices=len(vertices), faces=len(faces))


def shared_bounds(meshes, cameras):
    points = np.concatenate([m.reshape(-1, 3) for m in meshes if m is not None])
    center = (points.min(0).astype(float) + points.max(0)) * .5
    span = max(float(np.ptp((points - center) @ b, axis=0)[:2].max()) for b in cameras)
    return center, max(span, 1e-8)


def gizmo(image, camera):
    d = ImageDraw.Draw(image)
    start = np.array([45., image.height - 47.])
    for j, (name, color) in enumerate(zip('XYZ', ('#ba3434', '#278444', '#3355bb'))):
        end = start + np.array([camera[j, 0], -camera[j, 1]]) * 31
        d.line([tuple(start), tuple(end)], fill=color, width=2)
        d.text(tuple(end + [2, -9]), name, font=font(12), fill=color)


def cell(triangles, k, camera, center, span, size=CELL, missing=None):
    if triangles is None or len(triangles) == 0:
        image = Image.new('RGB', (size, size), '#f4f5f7')
        d = ImageDraw.Draw(image)
        text_lines(d, (25, size // 3), missing or 'EMPTY MESH / REVIEW REQUIRED', size - 50, 22, 6, '#a02424')
        return image, dict(visible_C_pixels=0, visible_Y_pixels=0, empty=True)
    image, owner = raster(triangles, int(k), camera, center, span, size=size)
    gizmo(image, camera)
    return image, dict(visible_C_pixels=int(((owner >= 0) & (owner < k)).sum()),
                       visible_Y_pixels=int((owner >= k).sum()), empty=False)


def base_receipt(parent, context, page_id, page_kind):
    return dict(page_id=page_id, page_kind=page_kind, parent_id=parent['parent_id'],
                split=parent['split'], parent_status=parent.get('status', 'UNSPECIFIED'),
                coordinate_contract=parent.get('coordinate_contract'),
                processing_note=parent.get('processing_note'),
                review_status='NOT_REVIEWED', semantic_decision=None,
                center=context['center'].tolist(), span=float(context['span']),
                common_frame='UNION_OF_ORIGINAL_AND_TARGET_IN_STORED_MODEL_COORDINATES',
                source_meshes=context['sources'], version=VERSION,
                views=[dict(name=n, azimuth=a, elevation=e) for n, a, e in VIEWS],
                source_code_sha256=sha(__file__), raster_source_sha256=sha(RASTER_SOURCE),
                model_calls=0, metric_calls=0, geometry_mutations=0,
                limitation='CPU pixel raster can miss subpixel gaps; exact depth ties favor C. No topology or semantic certification.')


def save_page(path, page, receipt):
    buffer = io.BytesIO()
    page.save(buffer, format='PNG')
    immutable_bytes(path, buffer.getvalue())
    receipt = dict(receipt, image_path=str(path), image_sha256=sha(path), pixels=list(page.size))
    write_json(path.with_suffix('.json'), receipt)
    return receipt


def existing_page(path, expected):
    """Resume by verifying completed immutable pages, without rasterizing again."""
    record = path.with_suffix('.json')
    if not path.exists() and not record.exists():
        return None
    if not path.exists() or not record.exists():
        raise FileExistsError('Incomplete prior page retained for explicit recovery: ' + str(path))
    receipt = json.loads(record.read_text(encoding='utf-8'))
    if sha(path) != receipt.get('image_sha256'):
        raise ValueError('Existing image hash mismatch: ' + str(path))
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError('Existing page contract changed: ' + str(path) + ' field=' + key)
    return receipt


def parent_page(parent, context, dest):
    expected = base_receipt(parent, context, parent['parent_id'], 'PARENT_ORIGINAL_TARGET')
    found = existing_page(dest / 'parents' / (parent['parent_id'] + '.png'), expected)
    if found is not None:
        return found
    size, header, foot = 900, 158, 72
    page = Image.new('RGB', (2 * size, header + 2 * (size + 34) + foot), 'white')
    d = ImageDraw.Draw(page)
    d.text((16, 10), parent['parent_id'] + ' | ' + parent['split'] + ' | ORIGINAL / V2', font=font(24), fill='black')
    text_lines(d, (16, 46), 'Status: ' + parent.get('status', 'UNSPECIFIED') + ' | NOT_REVIEWED; target is a proposal until separately accepted.', 1768, 18, 2)
    text_lines(d, (16, 93), parent.get('processing_note', ''), 1768, 17, 2)
    visibility = []
    for view, camera in enumerate(context['cameras']):
        y = header + view * (size + 34)
        for col, (name, tri) in enumerate([('Original M', context['original']), ('Target M v2 proposal', context['target'])]):
            count = len(tri) if tri is not None else 'NA'
            d.text((col * size + 16, y), f'{name} | faces={count} | {VIEWS[view][0]}', font=font(20), fill='black')
            im, vis = cell(tri, 0, camera, context['center'], context['span'], size,
                           'BLOCKED / NO TARGET PROPOSAL\nOriginal source shown in adjacent column.')
            page.paste(im, (col * size, y + 34))
            visibility.append(dict(column=col, view=view, **vis))
    text_lines(d, (16, page.height - foot + 8),
               'Identical camera, center and scale for original and target. Stored coordinates are never independently normalized. Y-up; original winding.', 1768, 16, 2)
    receipt = base_receipt(parent, context, parent['parent_id'], 'PARENT_ORIGINAL_TARGET')
    receipt['visibility'] = visibility
    return save_page(dest / 'parents' / (parent['parent_id'] + '.png'), page, receipt)


def condition_data(row, context):
    if row.get('case_path'):
        path = resolve(row['case_path'])
        if not row.get('case_sha256') or sha(path) != row['case_sha256']:
            raise ValueError('Condition case hash missing/mismatch: ' + str(path))
    ids = np.asarray(row['target_face_ids'], np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Condition face IDs not a unique 1D list: ' + row['condition_id'])
    target = context['target']
    if target is None:
        return dict(C=None, Y=None, CY=None, ids=ids, free=None, zoom=None,
                    warning='BLOCKED: parent has no target proposal; C cannot be rendered.')
    if len(ids) and (ids.min() < 0 or ids.max() >= len(target)):
        raise ValueError('Target face ID out of range: ' + row['condition_id'])
    free = np.setdiff1d(np.arange(len(target), dtype=np.int64), ids)
    c, y = target[ids], target[free]
    zoom = shared_bounds([c], context['cameras']) if len(c) else None
    return dict(C=c, Y=y, CY=np.concatenate([c, y]), ids=ids, free=free, zoom=zoom,
                warning='EMPTY C / INVALID PROPOSAL' if not len(c) else None)


def condition_receipt(row, parent, context, data, kind):
    receipt = base_receipt(parent, context, row['condition_id'], kind)
    receipt.update(condition_id=row['condition_id'], template=row['template'],
                   extent_description=row['extent_description'],
                   source_condition_review_status=row.get('review_status'),
                   target_face_ids=data['ids'].tolist(),
                   free_target_face_ids=data['free'].tolist() if data['free'] is not None else None,
                   K=len(data['ids']), N=len(context['target']) if context['target'] is not None else None,
                   case_path=row.get('case_path'), case_sha256=row.get('case_sha256'),
                   warning=data['warning'])
    return receipt


def condition_page(row, parent, context, data, dest):
    expected = condition_receipt(row, parent, context, data, 'CONDITION_TWO_VIEWS')
    found = existing_page(dest / 'conditions' / (row['condition_id'] + '.png'), expected)
    if found is not None:
        return found
    header, foot = 180, 86
    page = Image.new('RGB', (CELL * 4, header + CELL * 2 + foot), 'white')
    d = ImageDraw.Draw(page)
    text_lines(d, (14, 10), row['condition_id'], 1772, 23, 2, 'black')
    text_lines(d, (14, 66), row['template'] + ' | K=' + str(len(data['ids'])) + ' | NOT_REVIEWED | parent: ' + parent.get('status', 'UNSPECIFIED'), 1772, 18, 2)
    text_lines(d, (14, 112), row['extent_description'], 1772, 17, 2)
    panels = [('C only / whole scale', data['C'], len(data['ids']), False),
              ('C blue + Y gray / whole scale', data['CY'], len(data['ids']), False),
              ('Y only / whole scale', data['Y'], 0, False),
              ('C only / explicit AUTO ZOOM', data['C'], len(data['ids']), True)]
    visibility = []
    for col, (title, tri, k, zoom) in enumerate(panels):
        d.text((col * CELL + 12, header - 28), title, font=font(16), fill='black')
        for view, camera in enumerate(context['cameras']):
            center, span = data['zoom'] if zoom and data['zoom'] else (context['center'], context['span'])
            im, vis = cell(tri, k, camera, center, span, missing=data['warning'])
            page.paste(im, (col * CELL, header + view * CELL))
            visibility.append(dict(column=col, view=view, auto_zoom=zoom,
                                   center=center.tolist(), span=float(span), **vis))
    text_lines(d, (14, page.height - foot + 8),
               'First three columns share original/target union scale; last column is explicitly zoomed. Two opposite views, Y-up. Blue is actual C; gray is exact target-face complement.', 1772, 16, 2)
    text_lines(d, (14, page.height - 33), 'A coherent patch or K>=2 is not automatically a plausible user input. Actual AI_PROXY review is recorded separately.', 1772, 16, 1)
    receipt = condition_receipt(row, parent, context, data, 'CONDITION_TWO_VIEWS')
    receipt['visibility'] = visibility
    return save_page(dest / 'conditions' / (row['condition_id'] + '.png'), page, receipt)


def overview_pages(parent, context, condition_rows, data_by_id, dest):
    records = []
    # Four conditions per sheet keeps the rendered text/cells legible when viewed.
    for start in range(0, len(condition_rows), 4):
        batch = condition_rows[start:start + 4]
        number = start // 4 + 1
        pid = parent['parent_id'] + f'__p{number:02d}'
        expected = base_receipt(parent, context, pid, 'PARENT_CONDITION_OVERVIEW')
        expected.update(condition_count=len(batch), sheet_number=number,
                        condition_ids=[r['condition_id'] for r in batch])
        found = existing_page(dest / 'overview' / (pid + '.png'), expected)
        if found is not None:
            records.append(found)
            continue
        header, rowheight, foot = 126, CELL + 100, 72
        page = Image.new('RGB', (CELL * 4, header + len(batch) * rowheight + foot), 'white')
        d = ImageDraw.Draw(page)
        d.text((14, 10), parent['parent_id'] + f' | condition overview {number} | NOT_REVIEWED', font=font(23), fill='black')
        text_lines(d, (14, 48), 'Parent status: ' + parent.get('status', 'UNSPECIFIED') + ' | Blue C / gray target complement; source-mesh face identities, no imagined surfaces.', 1772, 17, 2)
        titles = ['C+Y / view A / whole scale', 'C+Y / view B / whole scale',
                  'C only / view A / same scale', 'C only / view B / same scale']
        for col, title in enumerate(titles):
            d.text((col * CELL + 12, header - 28), title, font=font(16), fill='black')
        children = []
        for ri, row in enumerate(batch):
            data = data_by_id[row['condition_id']]
            y = header + ri * rowheight
            suffix = row['condition_id'].removeprefix(parent['parent_id'] + '__')
            text_lines(d, (14, y), suffix + ' | ' + row['template'] + ' | K=' + str(len(data['ids'])), 1772, 20, 2, 'black')
            text_lines(d, (14, y + 48), row['extent_description'], 1772, 16, 2)
            visibility = []
            for col in range(4):
                camera = context['cameras'][col % 2]
                tri = data['CY'] if col < 2 else data['C']
                im, vis = cell(tri, len(data['ids']), camera, context['center'], context['span'], missing=data['warning'])
                page.paste(im, (col * CELL, y + 100))
                visibility.append(dict(column=col, view=col % 2, auto_zoom=False, **vis))
            entry = condition_receipt(row, parent, context, data, 'OVERVIEW_CONDITION_ROW')
            entry.update(row_index=ri, visibility=visibility)
            children.append(entry)
        text_lines(d, (14, page.height - foot + 7),
                   'All four columns use the same whole-parent scale. C-only detail zoom is available on each condition page. Semantic acceptance requires actual review.', 1772, 16, 2)
        pid = parent['parent_id'] + f'__p{number:02d}'
        receipt = base_receipt(parent, context, pid, 'PARENT_CONDITION_OVERVIEW')
        receipt.update(conditions=children, condition_count=len(children), sheet_number=number,
                       condition_ids=[r['condition_id'] for r in batch])
        records.append(save_page(dest / 'overview' / (pid + '.png'), page, receipt))
    return records


def run(manifest, mode='all', threads=2, output_name='review_v1', only_parent=None, only_condition=None, prioritize_parent=None):
    if threads not in (1, 2):
        raise ValueError('CPU thread limit must be 1 or 2')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', output_name):
        raise ValueError('Output name must be one simple directory name')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
        thread_context = threadpool_limits(limits=threads)
        thread_policy = 'threadpoolctl and pre-import environment limits'
    except ImportError:
        from contextlib import nullcontext
        thread_context = nullcontext()
        thread_policy = 'pre-import OMP/OPENBLAS/MKL/NUMEXPR environment limit 2; optional threadpoolctl unavailable'
        if threads != 2:
            raise RuntimeError('threads=1 needs threadpoolctl; use the authorized fixed limit2 or install separately')
    source = resolve(manifest)
    document = json.loads(source.read_text(encoding='utf-8-sig'))
    parents, rows = document['parents'], document['conditions']
    pmap = {p['parent_id']: p for p in parents}
    if len(pmap) != len(parents):
        raise ValueError('Duplicate parent identity')
    if len({r['condition_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate condition identity')
    if any(r['parent_id'] not in pmap for r in rows):
        raise ValueError('Unknown parent in condition manifest')
    ps, cs = set(only_parent or []), set(only_condition or [])
    if ps - set(pmap):
        raise ValueError('Unknown requested parent')
    if cs - {r['condition_id'] for r in rows}:
        raise ValueError('Unknown requested condition')
    rows = [r for r in rows if (not ps or r['parent_id'] in ps) and (not cs or r['condition_id'] in cs)]
    if cs and cs != {r['condition_id'] for r in rows}:
        raise ValueError('Condition and parent filters conflict')
    if cs:
        ps = {r['parent_id'] for r in rows}
    priority = list(prioritize_parent or [])
    for prefix in priority:
        if sum(p.startswith(prefix) for p in pmap) != 1:
            raise ValueError('Priority prefix must identify exactly one parent: ' + prefix)
    def parent_order(p):
        return (next((i for i, prefix in enumerate(priority) if p['parent_id'].startswith(prefix)), len(priority)), p['parent_id'])
    parents = sorted([p for p in parents if not ps or p['parent_id'] in ps], key=parent_order)
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['parent_id']].append(row)
    dest = GALLERY / output_name
    cameras = [basis(a, e) for _, a, e in VIEWS]
    records = []
    started = time.monotonic()
    with thread_context:
        for parent in parents:
            original, origref = checked_mesh(parent['original_mesh_path'], parent['original_mesh_sha256'])
            target, targetref = None, None
            if parent.get('target_mesh_path'):
                target, targetref = checked_mesh(parent['target_mesh_path'], parent['target_mesh_sha256'])
            center, span = shared_bounds([original, target], cameras)
            context = dict(original=original, target=target, center=center, span=span, cameras=cameras,
                           sources=dict(original=origref, target=targetref))
            if mode in ('parents', 'all'):
                records.append(parent_page(parent, context, dest))
            if mode in ('conditions', 'overview', 'all'):
                selected = sorted(grouped[parent['parent_id']], key=lambda r:r['condition_id'])
                data_by_id = {r['condition_id']:condition_data(r, context) for r in selected}
                if mode in ('conditions', 'all'):
                    for row in selected:
                        records.append(condition_page(row, parent, context, data_by_id[row['condition_id']], dest))
                if mode in ('overview', 'all'):
                    records.extend(overview_pages(parent, context, selected, data_by_id, dest))
            print(json.dumps(dict(parent_rendered=parent['parent_id'], pages_so_far=len(records)), ensure_ascii=False), flush=True)
    receipt = dict(version=VERSION, mode=mode, manifest_path=str(source), manifest_sha256=sha(source),
                   pages=records, page_count=len(records), review_status='NOT_REVIEWED',
                   source_code_sha256=sha(__file__), raster_source_sha256=sha(RASTER_SOURCE),
                   views=[dict(name=n, azimuth=a, elevation=e) for n, a, e in VIEWS],
                   worker_count=1, cpu_thread_limit=threads, thread_policy=thread_policy, model_calls=0, metric_calls=0,
                   geometry_mutations=0, automatic_semantic_decisions=0,
                   filters=dict(only_parent=sorted(ps), only_condition=sorted(cs)))
    # A filtered render uses a deterministic distinct index, preserving other subsets.
    suffix = '' if not ps and not cs else '_' + hashlib.sha256(json.dumps(receipt['filters'], sort_keys=True).encode()).hexdigest()[:12]
    write_json(dest / ('manifest_' + mode + suffix + '.json'), receipt)
    blocks = ['<!doctype html><meta charset="utf-8"><title>Bench v2 actual source inputs</title>',
              '<style>body{font:16px system-ui;margin:24px;background:#f3f5f8}article{background:white;margin:24px 0;padding:16px}img{max-width:100%;height:auto}small{color:#555}</style>',
              '<h1>Bench v2 proposals — actual source/target meshes</h1>',
              '<p>NOT_REVIEWED. Blue is supplied C, gray is its target complement. CPU depth buffer. No automatic semantic acceptance.</p>']
    for rec in records:
        relative = Path(rec['image_path']).relative_to(dest).as_posix()
        blocks.append('<article><h2>' + html.escape(rec['page_kind'] + ' | ' + rec['page_id']) + '</h2>'
                      + '<small>' + html.escape(rec['parent_status']) + '</small><p><a href="' + html.escape(relative)
                      + '"><img loading="lazy" src="' + html.escape(relative) + '"></a></p></article>')
    immutable_bytes(dest / ('index_' + mode + suffix + '.html'), '\n'.join(blocks).encode('utf-8'))
    print(json.dumps(dict(status='RENDERED_NOT_REVIEWED', pages=len(records), wall_seconds=time.monotonic()-started,
                         output=str(dest), model_calls=0, metric_calls=0, cpu_thread_limit=threads)), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--mode', choices=('parents', 'conditions', 'overview', 'all'), default='all')
    parser.add_argument('--threads', type=int, choices=(1, 2), default=2)
    parser.add_argument('--output-name', default='review_v1')
    parser.add_argument('--only-parent', action='append')
    parser.add_argument('--only-condition', action='append')
    parser.add_argument('--prioritize-parent', action='append', help='Unique parent ID/prefix rendered first; no selection change')
    args = parser.parse_args()
    run(args.manifest, args.mode, args.threads, args.output_name, args.only_parent, args.only_condition, args.prioritize_parent)


if __name__ == '__main__':
    main()
