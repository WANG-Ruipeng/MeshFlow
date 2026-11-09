"""Twelve fixed five-column geometry pages; metadata always awaits actual viewing.

No score reads, automatic ratings, model calls, reprocessing, or frame fitting.
The root dispatches render-one jobs after the CPU producers are complete.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import multiprocessing
import os
import resource
import subprocess
import time
from pathlib import Path

from . import analysis_v2 as A
import numpy as np

OUT = A.OUT
DEST = OUT / 'gallery'
VERSION = 'local_shape_five_column_gallery_v1'
MODELS = ('RAW', 'D0_ALL', 'ASSIST11', 'LOCKED_RAW', 'LOCAL_SHAPE')


def render_one(request_index, stage, jobs=None):
    if stage not in A.STAGES or not 0 <= request_index < 6:
        raise ValueError('Only six registered requests times two stages')
    context = A.load_context(jobs)
    ref = context['reg']['references'][request_index]
    q, rid = ref['request'], ref['request_id']
    key = 'base' if stage == 'BASE_POST' else 'final'
    cells = [dict(model='RAW', runtime_model='RAW', stage='RAW', path=ref['paths']['raw'])]
    for label in MODELS[1:]:
        cells.append(dict(model=label, runtime_model='M_D0' if label == 'D0_ALL' else label,
                          stage=stage, path=context['artifacts'][label, rid][key]))
    page_id = f'R{request_index:02d}_{stage}'
    definition = dict(version=VERSION, page_id=page_id, request_index=request_index, request=q,
                      stage=stage, cells=cells, registration_sha256=context['registration_sha256'],
                      views=[dict(azimuth=-60, elevation=23), dict(azimuth=120, elevation=23)],
                      sections='Original front/side position index1; no frame refitting',
                      renderer_sha256=A.sha(__file__), helper_sha256=A.sha(A.__file__))
    digest = A.canonical_hash(definition)
    image, receipt_path = DEST / (page_id + '.png'), DEST / (page_id + '.json')
    if receipt_path.exists():
        receipt = A.read(receipt_path)
        if receipt['definition_sha256'] != digest or A.sha(image) != receipt['image_sha256']:
            raise ValueError('Existing render identity differs')
        A.verify(receipt['source_meshes'])
        return receipt
    if image.exists():
        raise RuntimeError('Unsealed prior image retained; no silent rerender')
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    A.assert_cpu()
    D, H = G.D, G.H
    plt = H._mpl()
    start = time.perf_counter()
    DEST.mkdir(parents=True, exist_ok=True)
    k, length = int(q['K']), float(context['reg']['L'])
    inp = A.arrays(q['inference_npz'])
    original = inp['C_vertices'][inp['C_faces_original']]
    bounds, depth = q['gallery_bounds'], q['gallery_depth_bounds']
    sources = {str(A.resolve(q['inference_npz'])): A.sha(q['inference_npz'])}
    display = []
    fig = plt.figure(figsize=(26.5, 13.5), dpi=110, facecolor='white')
    grid = fig.add_gridspec(4, 5, left=.02, right=.99, bottom=.045, top=.92, wspace=.15, hspace=.23)
    try:
        for col, cell in enumerate(cells):
            tri, role = G.roles(A.arrays(cell['path']), k)
            sources[str(A.resolve(cell['path']))] = A.sha(cell['path'])
            geometry = dict(C=tri[:k], F=tri[k:], C_original=original, tri=tri)
            facts = dict(model=cell['model'], runtime_model=cell['runtime_model'], stage=cell['stage'],
                         role_counts={r: int((role == r).sum()) for r in ('C', 'SEAM', 'BULK', 'BRIDGE')}, views=[], sections=[])
            for row, azimuth in enumerate((-60, 120)):
                raster, scope = G.role_raster(tri, role, k, D.basis(azimuth, 23), np.asarray(depth['center']), depth['span'])
                ax = fig.add_subplot(grid[row, col])
                ax.imshow(raster); ax.set_axis_off()
                ax.set_title(f'{cell["model"]} {cell["stage"]} | {azimuth}', fontsize=9)
                if scope['projected_vertices_outside_inherited_frame']:
                    ax.text(.01, .01, 'Inherited-frame clipping', transform=ax.transAxes, fontsize=7, color='#b34b27')
                facts['views'].append(dict(azimuth=azimuth, elevation=23, **scope))
            for row, view in ((2, 'front'), (3, 'side')):
                ax = fig.add_subplot(grid[row, col])
                H.draw_section(ax, geometry, np.flatnonzero(role[k:] == 'SEAM'), bounds, length, view, 1)
                from matplotlib.collections import LineCollection
                axis, project = (2, (0, 1)) if view == 'front' else (0, (2, 1))
                position = bounds['front_positions' if view == 'front' else 'side_positions'][1]
                segments, ids = H.finite_sections(geometry['F'], axis, position, 1e-12 * length)
                bridge_mask = role[k:][ids] == 'BRIDGE'
                if bridge_mask.any():
                    ax.add_collection(LineCollection(segments[bridge_mask][:, :, project], colors='#41a46c', linewidths=1.6, zorder=3))
                ax.set_title(f'{cell["model"]} | {view}={position:.5f}', fontsize=8)
                facts['sections'].append(dict(view=view, position=position, inherited_index=1))
            display.append(facts)
        fig.suptitle(f'{rid} | {page_id} | N={q["N"]} K={k}\nOriginal RAW / original D0 chain / prior ASSIST11 / fixed assembly RAW / local shape fit; blue C, gray free, green bridge', fontsize=10)
        fig.text(.5, .012, 'Nonblind AI_PROXY review is separate and pending. Fixed sections do not certify cracks. No unique-source style ranking.', ha='center', fontsize=8)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=110)
        with image.open('xb') as f:
            f.write(buffer.getvalue())
    finally:
        plt.close(fig)
    A.assert_cpu(); A.verify(sources)
    result = dict(status='PENDING_ACTUAL_VIEW', version=VERSION, utc=A.utc(), page_id=page_id,
                  request_index=request_index, request_id=rid, parent_id=q['parent_id'], noise_index=q['noise_index'],
                  stage=stage, definition_sha256=digest, registration_sha256=context['registration_sha256'],
                  image_path=str(image), image_sha256=A.sha(image), cells=cells, source_meshes=sources, display=display,
                  renderer_sha256=A.sha(__file__), seconds=time.perf_counter()-start,
                  peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  peak_scope='Process lifetime CPU RSS high-water mark; includes imports and hash checking',
                  actually_viewed=False, model_calls=0, scoring_calls=0, ratings_generated=0)
    A.write(receipt_path, result)
    return result


def collect(jobs=None):
    """Seal twelve existing render receipts without rendering or claiming review."""
    context = A.load_context(jobs)
    if (DEST / 'COMPLETE.json').exists():
        done = A.read(DEST / 'COMPLETE.json')
        if done['registration_sha256'] != context['registration_sha256']:
            raise ValueError('Gallery registration changed')
        A.verify(done['files'])
        return done
    pages, files = [], {}
    for index, ref in enumerate(context['reg']['references']):
        for stage in A.STAGES:
            page_id = f'R{index:02d}_{stage}'
            path = DEST / (page_id + '.json')
            receipt = A.read(path)
            if receipt['request_id'] != ref['request_id'] or receipt['stage'] != stage or receipt['registration_sha256'] != context['registration_sha256']:
                raise ValueError('Render receipt identity differs')
            if [c['model'] for c in receipt['cells']] != list(MODELS):
                raise ValueError('Frozen five-column order differs')
            A.verify(receipt['source_meshes'])
            image = A.resolve(receipt['image_path'])
            if A.sha(image) != receipt['image_sha256']:
                raise ValueError('Rendered image changed')
            files[str(path)], files[str(image)] = A.sha(path), receipt['image_sha256']
            pages.append(dict(page_id=page_id, request_id=ref['request_id'], stage=stage,
                              receipt_path=str(path), image_path=str(image), image_sha256=receipt['image_sha256']))
    html = '<!doctype html><meta charset="utf-8"><title>Local shape fixed comparison</title><h1>12 pages; actual review pending</h1>'
    for page in pages:
        html += '<p>' + page['page_id'] + '</p><a href="' + Path(page['image_path']).name + '"><img style="width:100%" src="' + Path(page['image_path']).name + '"></a>'
    with (DEST / 'index.html').open('x', encoding='utf-8') as f:
        f.write(html)
    files[str(DEST / 'index.html')] = A.sha(DEST / 'index.html')
    result = dict(status='COMPLETE', actual_review_status='PENDING_ACTUAL_VIEW', version=VERSION, utc=A.utc(),
                  registration_sha256=context['registration_sha256'], pages=pages, rendered_pages=12, displayed_cells=60,
                  unique_geometry_observations=54, repeated_RAW_cells=6, actual_viewed_pages=0, visual_labels_generated=0,
                  unique_definition='6 original RAW + 4 processed models x 6 requests x 2 stages; RAW repeats across stages',
                  model_calls=0, scoring_calls=0, files=files)
    A.write(DEST / 'COMPLETE.json', result)
    return result


def resources():
    import psutil
    memory = psutil.virtual_memory()
    result = dict(utc=A.utc(), cpu_cores=os.cpu_count(), cpu_percent=psutil.cpu_percent(interval=.1),
                  ram_available_bytes=memory.available, ram_total_bytes=memory.total)
    for key, cmd in [('gpu', ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader']),
                     ('gpu_processes', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])]:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        result[key] = dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    return result


def run():
    """Root-invoked bounded render dispatch: one measured page, then at most two."""
    context = A.load_context()
    if (DEST / 'COMPLETE.json').exists():
        return collect()
    plan_path = DEST / 'dispatch.json'
    if plan_path.exists():
        raise RuntimeError('A prior render dispatch exists; verify/reuse pages with render-one, no automatic replay')
    tasks = [(index, stage) for index in range(6) for stage in A.STAGES]
    existing, pending = [], []
    for index, stage in tasks:
        receipt_path = DEST / f'R{index:02d}_{stage}.json'
        if receipt_path.exists():
            existing.append(render_one(index, stage))
        else:
            pending.append((index, stage))
    if not pending:
        return collect()
    before = resources()
    headroom = max(2 * 1024**3, .1 * before['ram_total_bytes'])
    if before['ram_available_bytes'] <= headroom:
        raise RuntimeError('Insufficient RAM margin; root must wait before rendering')
    DEST.mkdir(parents=True, exist_ok=True)
    A.write(plan_path, dict(status='STARTED', utc=A.utc(), registration_sha256=context['registration_sha256'],
                           before=before, maximum_workers=2, threads_per_worker=1,
                           first_authorized_page=list(pending[0]), pending_pages=[list(t) for t in pending],
                           reason='First authorized page provides a representative CPU renderer process peak; no probe render.'))
    started = time.perf_counter()
    # Spawn gives the first measurement an isolated process lifetime and releases it.
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        first = pool.submit(render_one, *pending.pop(0)).result()
    measured = first['peak_RSS_KiB'] * 1024
    completed = [first]
    scheduling = []
    # Use bounded waves so each extra launch follows a fresh resource measurement.
    while pending:
        snapshot = resources()
        margin = max(2 * 1024**3, .1 * snapshot['ram_total_bytes'])
        ram_slots = int(max(0, snapshot['ram_available_bytes'] - margin) // (1.2 * measured))
        cpu_slots = max(0, int(snapshot['cpu_cores'] * (1 - snapshot['cpu_percent'] / 100)))
        workers = min(2, len(pending), ram_slots, cpu_slots)
        if workers == 0:
            A.write(DEST / 'dispatch_wait.json', dict(status='WAIT_RESOURCE_MARGIN', utc=A.utc(), snapshot=snapshot,
                    observed_peak_RSS_bytes=measured, remaining_pages=[list(t) for t in pending]))
            raise RuntimeError('No measured CPU/RAM slot; completed pages retained for explicit continuation')
        wave = [pending.pop(0) for _ in range(workers)]
        scheduling.append(dict(resources=snapshot, workers=workers, pages=[list(t) for t in wave],
                               observed_peak_RSS_bytes=measured, headroom_bytes=margin))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            values = list(pool.map(_render_task, wave))
        completed.extend(values)
        measured = max(measured, *(r['peak_RSS_KiB'] * 1024 for r in values))
    A.write(DEST / 'dispatch_complete.json', dict(status='COMPLETE', utc=A.utc(), seconds=time.perf_counter()-started,
            rendered_pages=len(completed), existing_pages=len(existing), first_page_peak_RSS_bytes=first['peak_RSS_KiB']*1024,
            maximum_observed_worker_RSS_bytes=measured, waves=scheduling, after=resources(),
            timing_scope='Concurrent render/check/write wall time; not an isolated speed benchmark', model_calls=0, scoring_calls=0))
    return collect()


def _render_task(task):
    return render_one(*task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'render-one', 'collect'))
    parser.add_argument('--request-index', type=int)
    parser.add_argument('--stage', choices=A.STAGES)
    parser.add_argument('--jobs')
    args = parser.parse_args()
    if args.action == 'render-one':
        if args.request_index is None or args.stage is None:
            parser.error('render-one requires --request-index and --stage')
        result = render_one(args.request_index, args.stage, args.jobs)
    elif args.action == 'collect':
        result = collect(args.jobs)
    else:
        if args.jobs:
            parser.error('run uses the fixed local jobs.json')
        result = run()
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'cells', 'display', 'source_meshes', 'pages')}, indent=2))
