"""Anonymous two-column CPU re-render of the existing 32 DEV final pairs.

Identity and provenance stay under private_identity. reviewer contains only
opaque PNG names and the protocol. No model/scorer/noise/geometry operation.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse, concurrent.futures, io, json, multiprocessing, random, resource, time
from pathlib import Path
import numpy as np
from r2.pilots.chair_multipart_seam_v1 import gallery as G

C, D, H = G.C, G.D, G.H
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'experiments/r2_bench_joint_shift_v1/gallery'
OUT = ROOT / 'experiments/r2_bench_shift_relation_crossover_v1/blind_review'
PRIVATE, PUBLIC = OUT / 'private_identity', OUT / 'reviewer'
SEED = 2026102207
MODELS = ('RELATION_V1_S1500', 'JOINT_SHIFT_S1500')
VERSION = 'existing_dev32_final_anonymous_pair_v1'
PROTOCOL = '''# Anonymous final-output review

Review every page_001.png through page_032.png. Each page compares anonymous
columns A and B for the same condition and fixed noise. Left/right assignment
is independently balanced and shuffled across pages; A is not one fixed model.
Do not infer a model from page order or use prior rankings, scores or labels.
Only this reviewer directory may be consulted. Do not open private_identity,
the preparation source, original galleries, reports, meshes or numerical tables.

Open each complete PNG and inspect both whole views and both fixed-position
finite sections for each column. Blue is supplied C, orange seam, gray bulk,
green added bridge. Colors describe stored roles, not correctness. The views,
scale and section planes are inherited unchanged; no output-specific framing.
The frame-clipping annotation, when present, is a display limitation, not a score.

Record page_id, column A/B, actually_viewed=true, the following five independent
axes, and a short image-grounded note:

- overall_shape: REASONABLE / INCOMPLETE / UNCERTAIN
- C_use: PLAUSIBLE / REPLACED_OR_IGNORED / UNCERTAIN
- support_retention: PRESENT / MISSING_OR_DAMAGED / UNCERTAIN
- visible_fragment_or_gap: YES / NO_OBVIOUS / UNCERTAIN
- layering_or_fold: YES / NO_OBVIOUS / UNCERTAIN

A reasonable whole shape may still ignore C. Do not require resemblance to a
unique source completion or a specific furniture style. Use UNCERTAIN for
occlusion, ambiguous input purpose, ordinary thickness versus extra layers,
or finite-section limitations. Section endpoints alone do not establish cracks,
isolated faces, intersections, or shared-index topology. NO_OBVIOUS is not a
certificate of defect absence. Judge each column separately, not merely a winner.

No labels may be prefilled, copied from previous review, or inferred from scores.
All 32 pages and 64 columns must actually be viewed, including contrary cases.
Record reviewer_kind=AI_PROXY for AI review, blind=true only for hidden identities,
and prior_exposure=true if the reviewer recognizes previously seen assets.
This is not an independent human study; anonymity cannot remove prior exposure.
The package preparer must not serve as reviewer. Write review records outside
this sealed reviewer input directory; do not edit its PNGs or protocol.
'''


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        if path.exists():
            if path.read_text(encoding='utf-8') != value:
                raise ValueError('Existing frozen text differs: ' + str(path))
        else:
            with path.open('x', encoding='utf-8', newline='\n') as f:
                f.write(value)
    else:
        C.write(path, value)


def verify(hashes):
    for path, expected in hashes.items():
        if C.sha(path) != expected:
            raise ValueError('Protected source changed: ' + path)


def prepare():
    completion_path, manifest_path = SOURCE / 'completion.json', SOURCE / 'manifest.json'
    completion, manifest = C.read(completion_path), C.read(manifest_path)
    if len(completion['images']) != 144 or len(manifest['pages']) != 144:
        raise ValueError('Expected complete 144-page source gallery')
    original = {p['page_id']: p for p in completion['images']}
    pages = sorted((p for p in manifest['pages'] if p['split'] == 'DEV' and p['sheet'] == 'CLEAN_FINAL_POST'), key=lambda p: p['page_id'])
    if len(pages) != 32 or len({p['request_id'] for p in pages}) != 32 or len({p['parent_id'] for p in pages}) != 8:
        raise ValueError('Expected all 32 DEV requests from eight parents')
    protected = {str(p): C.sha(p) for p in (completion_path, manifest_path, SOURCE / 'definition.json', Path(__file__), Path(G.__file__), Path(D.__file__), Path(H.__file__), Path(C.__file__))}
    rng = random.Random(SEED)
    rng.shuffle(pages)
    swaps = [False] * 16 + [True] * 16
    rng.shuffle(swaps)
    jobs = []
    for index, (page, swap) in enumerate(zip(pages, swaps), 1):
        prior = original[page['page_id']]
        if C.sha(prior['path']) != prior['image_sha256']:
            raise ValueError('Original gallery PNG hash mismatch')
        protected[prior['path']] = prior['image_sha256']
        protected[page['inference_npz']] = C.sha(page['inference_npz'])
        cells = {c['model']: c for c in page['cells']}
        prior_cells = {c['model']: c for c in prior['cells']}
        order = MODELS[::-1] if swap else MODELS
        chosen = []
        for model in order:
            cell = cells[model]
            if cell != prior_cells[model] or cell['stage'] != 'CLEAN_FINAL_POST':
                raise ValueError('Manifest/completion final-cell identity differs')
            protected.update(G.authenticate(cell))
            chosen.append(cell)
        jobs.append(dict(opaque_id=f'page_{index:03d}', source_page=page, cells=chosen,
                         source_png=prior['path'], source_png_sha256=prior['image_sha256']))
    value = dict(version=VERSION, seed=SEED, stage='CLEAN_FINAL_POST', requests=32,
                 cells=64, models=list(MODELS), left_counts={m:16 for m in MODELS}, jobs=jobs,
                 protected=protected, model_calls=0, scoring_calls=0, new_noise=0,
                 geometry_edits=0, review_status='PENDING_ACTUAL_ANONYMOUS_REVIEW')
    put(PRIVATE / 'identity_key.json', value)
    put(PUBLIC / 'review_protocol.md', PROTOCOL)
    return value


def render_one(job):
    opaque, page = job['opaque_id'], job['source_page']
    png, receipt = PUBLIC / (opaque + '.png'), PRIVATE / (opaque + '.json')
    if receipt.exists():
        prior = C.read(receipt)
        if prior['identity_key_sha256'] != C.sha(PRIVATE / 'identity_key.json') or C.sha(png) != prior['image_sha256']:
            raise ValueError('Existing anonymous output differs')
        return prior
    started = time.perf_counter()
    plt = H._mpl()
    fig = plt.figure(figsize=(10.6, 13.5), dpi=110, facecolor='white')
    grid = fig.add_gridspec(4, 2, left=.025, right=.99, bottom=.055, top=.91, wspace=.15, hspace=.23)
    bundle = C.arrays(page['inference_npz'])
    original = bundle['C_vertices'][bundle['C_faces_original']]
    K, bounds, depth = page['K'], page['bounds'], page['depth_bounds']
    display = []
    for col, cell in enumerate(job['cells']):
        alias = 'AB'[col]
        tri, role = G.roles(C.arrays(cell['path']), K)
        g = dict(C=tri[:K], F=tri[K:], C_original=original, tri=tri)
        facts = dict(column=alias, model=cell['model'], views=[])
        for row, azimuth in enumerate((-60, 120)):
            raster, scope = G.role_raster(tri, role, K, D.basis(azimuth, 23), np.array(depth['center']), depth['span'])
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(raster); ax.set_axis_off()
            ax.set_title(alias + ' | View ' + str(row + 1), fontsize=10)
            facts['views'].append(dict(azimuth=azimuth, **scope))
            if scope['projected_vertices_outside_inherited_frame']:
                ax.text(.01, .01, 'Inherited-frame clipping', transform=ax.transAxes, fontsize=7, color='#b34b27')
        for row, view in ((2, 'front'), (3, 'side')):
            ax = fig.add_subplot(grid[row, col])
            H.draw_section(ax, g, np.flatnonzero(role[K:] == 'SEAM'), bounds, C.L, view, 1)
            from matplotlib.collections import LineCollection
            axis, project = (2, (0, 1)) if view == 'front' else (0, (2, 1))
            position = bounds['front_positions' if view == 'front' else 'side_positions'][1]
            segments, ids = H.finite_sections(g['F'], axis, position, 1e-12 * C.L)
            bridge = role[K:][ids] == 'BRIDGE'
            if bridge.any():
                ax.add_collection(LineCollection(segments[bridge][:, :, project], colors='#41a46c', linewidths=1.6, zorder=3))
            ax.set_title(alias + ' | Fixed ' + view + ' section', fontsize=9)
        display.append(facts)
    fig.suptitle(opaque + '\nBlue C / orange SEAM / gray BULK / green BRIDGE\nSame views, scale and fixed section planes', fontsize=11)
    fig.text(.5, .018, 'Anonymous review. Section endpoints alone do not certify cracks or topology.', ha='center', fontsize=8)
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=110, metadata={'Software':'CPU anonymous fixed-view renderer'})
        with png.open('xb') as f:
            f.write(buffer.getvalue())
    finally:
        plt.close(fig)
    value = dict(opaque_id=opaque, image_path=str(png), image_sha256=C.sha(png),
                 identity_key_sha256=C.sha(PRIVATE / 'identity_key.json'), display=display,
                 seconds=time.perf_counter()-started, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 status='PENDING_ACTUAL_REVIEW', model_calls=0, scoring_calls=0)
    put(receipt, value)
    return value


def run(max_workers=2):
    started = time.perf_counter()
    plan = prepare()
    done = PRIVATE / 'completion.json'
    if done.exists():
        result = C.read(done)
        verify(plan['protected'])
        verify(result['public_hashes'])
        return result
    before = C.resources()
    put(PRIVATE / 'resources_before.json', before)
    first = render_one(plan['jobs'][0])
    current = C.resources()
    available = int(next(line.split()[1] for line in current['memory'].splitlines() if line.startswith('MemAvailable:'))) * 1024
    peak = first['peak_RSS_KiB'] * 1024
    cpu_slots = max(1, int(current['cpu_logical'] - current['loadavg'][0] - 2))
    ram_slots = max(1, int(max(0, available-(4<<30)) / max(1, peak*1.2)))
    count = max(1, min(int(max_workers), 2, len(plan['jobs']), cpu_slots, ram_slots))
    put(PRIVATE / 'dispatch.json', dict(workers=count, resources=current, first_peak_RSS_bytes=peak,
                                       available_RAM_bytes=available, CPU_slots=cpu_slots, model_calls=0,
                                       reason='CPU raster only; two-worker cap with observed RSS and available resources'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context('spawn')) as pool:
        images = [first, *list(pool.map(render_one, plan['jobs'][1:]))]
    expected = {'review_protocol.md'} | {j['opaque_id']+'.png' for j in plan['jobs']}
    if {p.name for p in PUBLIC.iterdir()} != expected:
        raise ValueError('Reviewer directory contains unexpected files')
    verify(plan['protected'])
    public_hashes = {str(p):C.sha(p) for p in sorted(PUBLIC.iterdir())}
    result = dict(status='ANONYMOUS_RENDER_COMPLETE_PENDING_REVIEW', pages=32, cells=64,
                  workers=count, wall_seconds=time.perf_counter()-started, public_hashes=public_hashes,
                  identity_key_sha256=C.sha(PRIVATE/'identity_key.json'), images=images,
                  model_calls=0, GPU_compute_calls=0, scoring_calls=0, new_noise=0, geometry_edits=0,
                  source_hashes_unchanged=True, preparer_reviewed_labels=False)
    put(done, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    result = run(args.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in ('public_hashes','images')}, ensure_ascii=False))
