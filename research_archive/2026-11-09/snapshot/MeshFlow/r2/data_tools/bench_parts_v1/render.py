"""CPU, depth-correct source-identity review figures for Bench part conditions.

Reads immutable source meshes and candidate face IDs. It neither assigns part
labels nor runs metrics/models. All semantic labels remain proposals until an
actual recorded review. Final pages show the exact set difference Y=M\\C.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
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
# The historical raster module sets OPENBLAS=2 at import, after NumPy import.
# Restore the single-thread environment and use threadpoolctl below as well.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_parts_v1'
VERSION = 'BENCH_SOURCE_PARTS_DEPTH_REVIEW_V1'
VIEWS = [('oblique A', -60., 22.), ('oblique B', 120., 22.)]
SIZE = 380


def resolve(path):
    value = str(path).replace('\\', '/')
    if os.name != 'nt' and re.match(r'^[A-Za-z]:/', value):
        value = '/mnt/' + value[0].lower() + value[2:]
    elif os.name == 'nt' and re.match(r'^/mnt/[a-z]/', value):
        value = value[5].upper() + ':/' + value[7:]
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def write_bytes(path, value):
    path = resolve(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Renderer output outside its experiment: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != value:
        raise FileExistsError('Refusing to overwrite a changed figure: ' + str(path))
    if not path.exists():
        path.write_bytes(value)


def write_json(path, obj):
    write_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_source(row):
    path = resolve(row['source_case_path'])
    if row.get('source_case_sha256') and sha(path) != row['source_case_sha256']:
        raise ValueError('Source identity changed: ' + str(path))
    with np.load(path, allow_pickle=False) as z:
        vertices = z['model_vertices'].copy()
        faces = z['source_faces'].copy()
    ids = np.asarray(row.get('source_face_ids', row.get('region_source_face_ids')), dtype=np.int64)
    if len(ids) < 2 or len(set(ids.tolist())) != len(ids):
        raise ValueError('C requires at least two unique source face IDs')
    if ids.min() < 0 or ids.max() >= len(faces):
        raise ValueError('Source face ID out of range')
    free = np.setdiff1d(np.arange(len(faces)), ids)
    M = vertices[faces]
    return M, M[ids], M[free], ids, free


def bounds(tri, cameras):
    pts = tri.reshape(-1, 3).astype(float)
    center = (pts.min(0) + pts.max(0)) / 2
    span = max(float(np.ptp((pts - center) @ B, axis=0)[:2].max()) for B in cameras)
    return center, max(span, 1e-8)


def axes_gizmo(image, B):
    """Explicit global X/Y/Z, source Y-up; no relabeling by guessed semantics."""
    d = ImageDraw.Draw(image)
    origin = np.array([48., image.height - 48.])
    for j, (name, color) in enumerate(zip('XYZ', ('#b93030', '#298d38', '#3554ae'))):
        endpoint = origin + np.array([B[j, 0], -B[j, 1]]) * 33
        d.line([tuple(origin), tuple(endpoint)], fill=color, width=2)
        d.text(tuple(endpoint + [2, -7]), name, font=font(11), fill=color)
    return image


def image_cell(tri, k, B, center, span, size=SIZE):
    image, owner = raster(tri, k, B, center, span, size=size)
    axes_gizmo(image, B)
    return image, dict(visible_C_pixels=int(((owner >= 0) & (owner < k)).sum()),
                       visible_Y_pixels=int((owner >= k).sum()))


def save_page(path, page):
    buffer = io.BytesIO()
    page.save(buffer, format='PNG')
    write_bytes(path, buffer.getvalue())
    return dict(path=str(path), sha256=sha(path), pixels=list(page.size))


def candidate_sheet(parent, rows, dest):
    cameras = [basis(a, e) for _, a, e in VIEWS]
    first, _, _, _, _ = load_source(rows[0])
    center, span = bounds(first, cameras)
    header, rowheight, footer = 72, SIZE + 55, 65
    page = Image.new('RGB', (SIZE * 4, header + rowheight * len(rows) + footer), 'white')
    d = ImageDraw.Draw(page)
    d.text((12, 8), parent + ' | ' + rows[0]['group'] + ' | proposed parts: REVIEW REQUIRED', font=font(19), fill='black')
    d.text((12, 34), 'Blue=C, gray=remaining source M. Y-up. No transparency or through-surface overlays.', font=font(15), fill='#333333')
    visibility, records = [], []
    for ri, row in enumerate(rows):
        M, C, Y, ids, free = load_source(row)
        if not np.array_equal(M, first):
            raise ValueError('Candidate parent source geometry mismatch')
        ccenter, cspan = bounds(C, cameras)
        y = header + ri * rowheight
        label = row['candidate_id'].split('__')[-1]
        d.text((12, y), label + ' | ' + row['proposed_part_type'] + ' | K=' + str(len(C)) + '/' + str(len(M)), font=font(16), fill='black')
        inputs = [(np.concatenate([C, Y]), len(C), cameras[0], center, span, 'M + blue C / view A / whole scale'),
                  (np.concatenate([C, Y]), len(C), cameras[1], center, span, 'M + blue C / view B / whole scale'),
                  (C, len(C), cameras[0], ccenter, cspan, 'C only / view A / AUTO ZOOM'),
                  (C, len(C), cameras[1], ccenter, cspan, 'C only / view B / AUTO ZOOM')]
        for col, (tri, k, B, bc, bs, title) in enumerate(inputs):
            im, vis = image_cell(tri, k, B, bc, bs)
            page.paste(im, (col * SIZE, y + 47))
            d.text((col * SIZE + 10, y + 25), title, font=font(12), fill='#444444')
            visibility.append(dict(candidate_id=row['candidate_id'], column=col, **vis))
        records.append(dict(candidate_id=row['candidate_id'], parent_id=parent, source_face_ids=ids.tolist(),
                            proposed_part_type=row['proposed_part_type'], row_index=ri, K=len(C), N=len(M),
                            source_case_path=row['source_case_path'], source_sha256=sha(row['source_case_path']),
                            C_zoom_center=ccenter.tolist(), C_zoom_span=cspan))
    y = header + rowheight * len(rows) + 5
    d.text((12, y), 'Whole views share parent scale; C-only views zoom to selected C. Exact depth ties favor C; hidden surfaces remain hidden.', font=font(13), fill='#444444')
    d.text((12, y + 23), 'No auto semantic labels. A coherent patch is not automatically a complete seat/back/leg. Source artifacts remain unchanged.', font=font(13), fill='#444444')
    receipt = save_page(dest / (parent + '.png'), page)
    receipt.update(parent_id=parent, group=rows[0]['group'], rows=records, visibility=visibility,
                   whole_center=center.tolist(), whole_span=span, review_status='NOT_REVIEWED')
    return receipt


def final_page(row, dest):
    M, C, Y, ids, free = load_source(row)
    cameras = [basis(a, e) for _, a, e in VIEWS]
    center, span = bounds(M, cameras)
    size = 340
    page = Image.new('RGB', (size * 5, size * 2 + 145), 'white')
    d = ImageDraw.Draw(page)
    cid = row.get('condition_id', row.get('candidate_id'))
    d.text((12, 8), cid, font=font(18), fill='black')
    label = row.get('part_type', row.get('proposed_part_type', 'REVIEWED_REGION'))
    d.text((12, 33), f'{label} | K={len(C)} / N={len(M)} | original source faces, C + Y = M; no repair/retriangulation', font=font(14), fill='#333333')
    panels = [('M: full source', M, 0), ('C: input only', C, len(C)), ('Y: source complement', Y, 0),
              ('C + Y: blue input', np.concatenate([C, Y]), len(C)), ('C only: zoom detail', C, len(C))]
    ccenter, cspan = bounds(C, cameras)
    visibility = []
    for col, (title, tri, k) in enumerate(panels):
        d.text((col * size + 8, 60), title, font=font(14), fill='black')
        for ri, B in enumerate(cameras):
            bc, bs = (ccenter, cspan) if col == 4 else (center, span)
            im, vis = image_cell(tri, k, B, bc, bs, size=size)
            page.paste(im, (col * size, 83 + ri * size))
            visibility.append(dict(column=col, view=ri, **vis))
    d.text((12, size * 2 + 93), 'Two opposite depth-correct views. Columns 1-4 use identical whole-parent scale; last column explicitly zooms C. Global Y-up.', font=font(13), fill='#333333')
    d.text((12, size * 2 + 115), 'C and Y are source-face-ID complements, not a unique reconstruction objective; visual roles are recorded separately.', font=font(13), fill='#333333')
    receipt = save_page(dest / (cid + '.png'), page)
    receipt.update(condition_id=cid, parent_id=row['parent_id'], group=row['group'], source_face_ids=ids.tolist(),
                   free_source_face_ids=free.tolist(), K=len(C), N=len(M), source_case_path=row['source_case_path'],
                   source_sha256=sha(row['source_case_path']), visibility=visibility,
                   whole_center=center.tolist(), whole_span=span, C_zoom_center=ccenter.tolist(), C_zoom_span=cspan,
                   part_type=label, extent_description=row.get('extent_description', ''),
                   status=row.get('status'), input_obj_path=row.get('input_obj_path'),
                   source_edge_components=row.get('source_edge_components'),
                   seam_applicability=row.get('seam_applicability'))
    return receipt


def run(manifest, mode='candidates', limit=None, output_name=None):
    from contextlib import nullcontext
    try:
        from threadpoolctl import threadpool_limits
        thread_context = threadpool_limits(limits=1)
    except ImportError:
        # NumPy is imported only after all three thread environment limits are
        # set. Do not install a new package merely for this optional guard.
        thread_context = nullcontext()
    src = resolve(manifest)
    payload = json.loads(src.read_text(encoding='utf-8-sig'))
    rows = payload.get('rows', payload.get('conditions'))
    if rows is None:
        raise ValueError('Expected manifest rows or conditions')
    if mode == 'final' and any(r.get('status') != 'ACTIVE_REVIEWED_SOURCE_CONDITION' for r in rows):
        raise ValueError('Final gallery accepts only ACTIVE reviewed source conditions')
    name = output_name or (mode if limit is None else mode + '_demo')
    if not re.fullmatch(r'[a-zA-Z0-9_]+', name):
        raise ValueError('Output gallery name must be a simple directory name')
    dest = OUT / 'gallery' / name
    started = time.monotonic()
    with thread_context:
        if mode == 'candidates':
            byparent = collections.defaultdict(list)
            for row in rows:
                byparent[row['parent_id']].append(row)
            jobs = sorted(byparent.items(), key=lambda x: (x[1][0]['group'] != 'DEV', x[0]))
            if limit is not None:
                jobs = jobs[:limit]
            records = []
            for i, (parent, rs) in enumerate(jobs):
                records.append(candidate_sheet(parent, rs, dest))
                print(json.dumps(dict(rendered=i + 1, total=len(jobs), parent_id=parent)), flush=True)
        else:
            jobs = rows if limit is None else rows[:limit]
            records = []
            for i, row in enumerate(jobs):
                records.append(final_page(row, dest))
                print(json.dumps(dict(rendered=i + 1, total=len(jobs), condition_id=records[-1]['condition_id'])), flush=True)
    receipt = dict(version=VERSION, mode=mode, manifest_path=str(src), manifest_sha256=sha(src),
                   pages=records, page_count=len(records), source_code_sha256=sha(__file__),
                   raster_source_sha256=sha(ROOT / 'r2/pilots/regular_patch_control_v1/depth_gallery.py'),
                   views=[dict(name=n, azimuth=a, elevation=e) for n, a, e in VIEWS],
                   model_calls=0, geometry_mutations=0, metric_calls=0, semantic_labels_automatically_assigned=0,
                   worker_count=1, thread_limit=1)
    write_json(dest / 'manifest.json', receipt)
    blocks = ['<meta charset="utf-8"><title>Bench source part inputs</title><h1>Bench source part inputs: ' + html.escape(mode) + '</h1>',
              '<p>Blue C; gray source complement. CPU depth-buffer views, Y-up. Semantic review is separate from candidate proposals.</p>']
    for r in records:
        name = Path(r['path']).name
        ident = r.get('condition_id', r['parent_id'])
        blocks.append('<h2>' + html.escape(ident + ' | ' + r['group']) + '</h2><a href="' + name + '"><img loading="lazy" style="max-width:100%" src="' + name + '"></a>')
    write_bytes(dest / 'index.html', '\n'.join(blocks).encode('utf-8'))
    if mode == 'final':
        parts = sorted({r['part_type'] for r in records})
        filterpage = ['<!doctype html><meta charset="utf-8"><title>Bench reviewed part conditions</title>',
            '<style>body{font:16px system-ui;margin:24px;background:#f4f7fb;color:#142333}header{position:sticky;top:0;background:#fff;padding:16px;border:1px solid #ccd6e0;z-index:3}select,input{margin:8px;padding:8px}article{background:white;margin:24px 0;padding:16px;border:1px solid #d9e1e9}img{max-width:100%;height:auto}small{color:#526171}.hidden{display:none}</style>',
            '<header><h1>Bench: reviewed multi-part inputs</h1><p>Only ACTIVE data entries are shown. Blue = supplied C; gray = source complement. These are source input constructions, not newly generated models.</p>',
            '<label>Split <select id="split"><option value="">All</option><option>TRAIN</option><option>DEV</option></select></label>',
            '<label>Part <select id="part"><option value="">All</option>' + ''.join('<option>'+html.escape(p)+'</option>' for p in parts) + '</select></label>',
            '<label>Parent / condition <input id="search" placeholder="ID fragment"></label><b id="count"></b>',
            '<p><small>Five columns: M | C | Y | C+Y | C zoom. Two opposite views. Names describe selected surfaces/parts, not automatically complete solid objects. Some valid inputs have multiple separated components.</small></p></header>']
        for r in records:
            relative = os.path.relpath(resolve(r['path']), OUT / 'gallery').replace('\\','/')
            objlink = os.path.relpath(resolve(r['input_obj_path']), OUT / 'gallery').replace('\\','/') if r.get('input_obj_path') else None
            filterpage += ['<article data-split="'+html.escape(r['group'])+'" data-part="'+html.escape(r['part_type'])+'" data-id="'+html.escape(r['condition_id'])+'">',
                '<h2>'+html.escape(r['condition_id'])+'</h2><p><b>'+html.escape(r['part_type']+' | '+r['group'])+'</b> | K='+str(r['K'])+' / N='+str(r['N'])+' | source components='+str(r['source_edge_components'])+'</p>',
                '<p>'+html.escape(r['extent_description'])+'</p><small>Current SEAM interface: '+html.escape(str(r['seam_applicability']))+'</small>',
                ('<p><a href="'+html.escape(objlink)+'">Download the actual C-only OBJ</a></p>' if objlink else ''),
                '<a href="'+html.escape(relative)+'"><img loading="lazy" src="'+html.escape(relative)+'"></a></article>']
        filterpage.append('<script>const controls=["split","part","search"].map(x=>document.getElementById(x));function filter(){const [s,p,q]=controls.map(x=>x.value.toLowerCase());let n=0;document.querySelectorAll("article").forEach(a=>{const show=(!s||a.dataset.split.toLowerCase()===s)&&(!p||a.dataset.part.toLowerCase()===p)&&(!q||a.dataset.id.toLowerCase().includes(q));a.classList.toggle("hidden",!show);n+=show;});document.getElementById("count").textContent=n+" conditions shown";}controls.forEach(x=>x.addEventListener("input",filter));filter();</script>')
        write_bytes(OUT / 'gallery/index.html', '\n'.join(filterpage).encode('utf-8'))
    print(json.dumps(dict(status='COMPLETE', pages=len(records), wall_seconds=time.monotonic() - started,
                          output=str(dest))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default=str(OUT / 'candidates/manifest.json'))
    p.add_argument('--mode', choices=('candidates', 'final'), default='candidates')
    p.add_argument('--limit', type=int)
    p.add_argument('--output-name')
    a = p.parse_args()
    run(a.manifest, a.mode, a.limit, a.output_name)
