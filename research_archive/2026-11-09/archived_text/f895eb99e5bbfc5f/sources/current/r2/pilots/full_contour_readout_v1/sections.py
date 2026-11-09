"""CPU display supplement: fixed finite C-central cuts of saved FULL meshes.

This module adds figures only. It does not change the scientific registration,
frozen evaluator or depth renderer, generate queries, compute metrics, alter
geometry, or fill visual judgments. Historical rendering helpers stay read-only.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import collections
import html
import json
from pathlib import Path
import sys
import time
import numpy as np
from . import common as C
from . import evaluation as E
from r2.pilots.regular_patch_control_v1 import render as R

VERSION = 'full_contour_readout_fixed_finite_sections_v1'
ROWS = (('RAW', 'front'), ('RAW', 'side'), ('POST', 'front'), ('POST', 'side'))
LABELS = E.LABELS


def definition(labels):
    return dict(version=VERSION, purpose='DISPLAY_SUPPLEMENT_NOT_NEW_EVALUATION',
        labels=list(labels), columns='Fixed checkpoint labels in the specified order',
        rows=[dict(stage=s, view=v) for s, v in ROWS],
        coordinate_contract='Saved model/world coordinates; X/Y front and Z/Y side. No recentering, scaling, rotation, smoothing or geometry mutation.',
        cut_contract='C-only axis-aligned bbox midpoint: FRONT Z=(C_z_min+C_z_max)/2; SIDE X=(C_x_min+C_x_max)/2. Existing helper index=1 of fractions .25/.50/.75; only .50 is rendered.',
        intersection_contract='Existing FP64 finite triangle-plane intersection; eps=1e-12*L. Coplanar triangles contribute finite edges. No projected silhouette or infinite face extrapolation.',
        local_margins_L=dict(X=.075, Y=.10, Z=.075), reference_length=C.L,
        scale_contract='Identical C-derived bounds for every checkpoint, RAW/POST and all four noises within a condition; equal coordinate aspect. Partial-endpoint selection cannot change the C-only local bounds.',
        colors=dict(C='#1686c0', free='#747b83', source_winding_normal='#7c428f'),
        normal_contract='Original source C winding, not outward/interior certification; source metadata validation only, no orientation solver.',
        inherited_definition=R.VERSION, candidate_coloring=False, source_GT_display=False,
        new_queries=0, new_metrics=0, new_meshes=0, model_calls=0, actual_reviews=0,
        limitations='Two predetermined central cuts can miss off-plane contacts and surfaces. Coplanar edges may overprint. Neither line count nor visible contact certifies layers, adoption or indexed topology. Blue is drawn above gray; exact overlap needs existing independent diagnostics.',
        figure_dpi=150, row_count=4, nonblind=True)


def load(labels):
    if 'torch' in sys.modules:
        raise RuntimeError('Finite section display requires a CPU-only fresh process')
    r = E.registration()
    qs = {q['request_id']: q for q in r['requests']}
    refs, cells, sources = {}, {}, {}
    C.package()
    from meshflow_control.source_winding import validate_source
    for q in r['requests']:
        E.verify({q['input_path']: q['input_sha256'], q['condition_source']: q['condition_source_sha256']})
        inp = C.arrays(q['input_path']); src = C.arrays(q['condition_source'])
        canonical = inp['C'].reshape(int(q['K']), 3, 3)
        sources[q['request_id']] = (canonical, validate_source(src, canonical)['C_original'])
        refs[q['input_path']] = q['input_sha256']
        refs[q['condition_source']] = q['condition_source_sha256']
    for label in labels:
        for stage in ('RAW', 'POST'):
            path = C.OUT / 'evaluation/indices' / label / (stage.lower() + '_index.json')
            saved = C.read(path)
            if saved['status'] != 'COMPLETE' or len(saved['rows']) != 24 or {x['request_id'] for x in saved['rows']} != set(qs):
                raise ValueError('Complete fixed 24-request endpoint required; missing requests are not dropped')
            refs[str(path)] = C.sha(path)
            for rec in saved['rows']:
                q = qs[rec['request_id']]
                if rec['label'] != label or rec['input_sha256'] != q['input_sha256'] or rec['checkpoint_sha256'] != E.binding(label)['sha256']:
                    raise ValueError('Saved geometry identity differs')
                p, digest = rec[stage.lower() + '_path'], rec[stage.lower() + '_sha256']
                E.verify({p: digest}); refs[p] = digest
                arrays = C.arrays(p)
                tri = arrays['raw' if stage == 'RAW' else 'output'].reshape(-1, 3, 3)
                k = int(q['K']); canonical, original = sources[q['request_id']]
                if (stage == 'RAW' and len(tri) != int(q['N'])) or len(tri) < k or not np.isfinite(tri).all():
                    raise ValueError('Malformed saved geometry')
                if not C.same(tri[:k], canonical if stage == 'RAW' else original):
                    raise ValueError('Saved C does not satisfy its original stage contract')
                cells[q['request_id'], label, stage] = dict(C=tri[:k], F=tri[k:], C_original=original,
                    path=p, sha256=digest, C_array_sha256=C.array_sha(tri[:k]), F_array_sha256=C.array_sha(tri[k:]))
    if len(cells) != 48 * len(labels):
        raise ValueError('Incomplete geometry barrier')
    return r, cells, refs


def figure(q, labels, cells, bounds):
    H = R.helper(); plt = H._mpl()
    fig = plt.figure(figsize=(max(9., 4.5 * len(labels)), 13.2), dpi=150, facecolor='white')
    grid = fig.add_gridspec(4, len(labels), left=.055, right=.986, bottom=.091, top=.925, wspace=.24, hspace=.43)
    for column, label in enumerate(labels):
        for row, (stage, view) in enumerate(ROWS):
            ax = fig.add_subplot(grid[row, column])
            H.draw_section(ax, cells[q['request_id'], label, stage], [], bounds, C.L, view, 1)
            old_title = ax.get_title()
            ax.set_title((label + ' | ' if row == 0 else '') + stage + ' | ' + old_title, fontsize=9)
    fig.suptitle(q['parent_id'][:8] + ' | SEAT_FULL | n' + str(q['candidate']) +
                 ' | K=' + str(q['K']) + ', N=' + str(q['N']) + '\nFixed central finite sections; NONBLIND display supplement', fontsize=13, y=.978)
    fig.text(.5, .052, 'BLUE preserved C | GRAY free | PURPLE source winding normals (not certified outward)', ha='center', fontsize=8)
    fig.text(.5, .034, 'Finite intersections only. Shared C-local coordinate limits across checkpoints, stages and noises.', ha='center', fontsize=8)
    fig.text(.5, .017, 'Two cuts may miss off-plane geometry; visible contact and line count do not certify connectivity or layers. No judgments assigned.', ha='center', fontsize=8)
    return fig


def run(labels=LABELS):
    labels = tuple(labels)
    if not labels or len(set(labels)) != len(labels) or any(x not in LABELS for x in labels):
        raise ValueError('Choose distinct fixed learning-curve endpoint labels only')
    tick = time.perf_counter(); r, cells, refs = load(labels); H = R.helper()
    groups = collections.defaultdict(list)
    for q in r['requests']:
        groups[q['condition_id']].append(q)
    if len(groups) != 6 or any(len(qs) != 4 for qs in groups.values()):
        raise ValueError('Six conditions with four original noises each are required')
    bounds = {cid: H.bounds_for_condition([cells[q['request_id'], label, stage]
        for q in qs for label in labels for stage in ('RAW', 'POST')], None, C.L) for cid, qs in groups.items()}
    name = 'sections_gallery' if labels == LABELS else 'sections_gallery_' + '_'.join(labels)
    dest = C.OUT / 'evaluation' / name
    source_paths = (Path(__file__), Path(C.__file__), Path(E.__file__), Path(R.__file__), Path(H.__file__),
                    C.ROOT / 'r2/src/meshflow_control/source_winding.py')
    hashes = {str(p): C.sha(p) for p in source_paths}
    contract = definition(labels); registration_sha = C.sha(C.OUT / 'registration.json')
    C.write(dest / 'definition.json', contract)
    C.write(dest / 'physical_bounds_and_cuts.json', bounds)
    pages = []
    for q in r['requests']:
        path = dest / 'images' / (q['parent_id'][:8] + '_n' + str(q['candidate']) + '.png')
        geometry = [dict(label=label, stage=stage, **{key: cells[q['request_id'], label, stage][key]
            for key in ('path', 'sha256', 'C_array_sha256', 'F_array_sha256')})
            for label in labels for stage in ('RAW', 'POST')]
        identity = dict(version=VERSION, registration_sha256=registration_sha, request_id=q['request_id'],
            definition=contract, source_hashes=hashes, bounds=bounds[q['condition_id']], geometry=geometry)
        receipt = H.save_page(C, dest, path, identity, lambda q=q: figure(q, labels, cells, bounds[q['condition_id']]))
        pages.append(dict(request_id=q['request_id'], parent_id=q['parent_id'], candidate=q['candidate'],
            path=str(path), sha256=receipt['image_sha256'], pixels=receipt['pixels'], geometry=geometry))
        print(json.dumps(dict(stage='FINITE_SECTIONS', rendered=len(pages), registered=24)), flush=True)
    page = '<!doctype html><meta charset="utf-8"><title>FULL fixed central sections</title><style>body{font:16px sans-serif;margin:24px}img{width:100%;max-width:1800px}section{margin-bottom:32px}</style><h1>FULL learning curve: finite central sections</h1><p>Display supplement only; two C-derived central cuts. Blue C, gray free. No new metrics or visual labels.</p>'
    page += ''.join('<section><h2>' + html.escape(x['parent_id'][:8] + ' / n' + str(x['candidate'])) + '</h2><a href="images/' + Path(x['path']).name + '"><img loading="lazy" src="images/' + Path(x['path']).name + '"></a></section>' for x in pages)
    C._write_bytes(dest / 'index.html', page.encode('utf-8'))
    E.verify(refs); E.verify(hashes)
    out = dict(status='COMPLETE_RENDERING_NOT_REVIEW', version=VERSION, labels=list(labels), pages=24,
        geometry_cells=len(cells), registration_sha256=registration_sha, definition=contract, source_hashes=hashes,
        input_hashes=refs, physical_bounds_and_cuts=bounds, images=pages,
        files={str(dest / name): C.sha(dest / name) for name in ('definition.json', 'physical_bounds_and_cuts.json', 'index.html')},
        model_calls=0, new_geometry=0, new_queries=0, new_metrics=0, actual_reviews=0)
    C.write(dest / 'manifest.json', out)
    C.event('LEARNING_CURVE_FINITE_SECTIONS_COMPLETE', labels=labels, pages=24, geometry_cells=len(cells), seconds=time.perf_counter()-tick, model_calls=0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', nargs='+', choices=LABELS, default=list(LABELS))
    args = parser.parse_args(); result = run(args.labels)
    print(json.dumps(dict(status=result['status'], pages=result['pages'], geometry_cells=result['geometry_cells'], labels=result['labels'])), flush=True)


if __name__ == '__main__':
    main()
