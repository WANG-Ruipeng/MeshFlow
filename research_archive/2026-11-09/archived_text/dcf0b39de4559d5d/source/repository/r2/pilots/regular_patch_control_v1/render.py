"""CPU-only, nonblind regular-patch comparison: 24 pages and 96 blank reviews.

All four panels use frozen existing geometry. No model invocation, score,
noise draw, mesh repair, candidate detector, or automatic visual judgment.
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
import time
import numpy as np
from . import common as C
from . import inference as I

GALLERY = C.OUT / 'evaluation/gallery'
VERSION = 'r2_regular_patch_four_column_finite_sections_v1'
COLUMNS = (('SMALL_C200', 'RAW'), ('CREF_C200', 'RAW'), ('SMALL_C200', 'POST'), ('CREF_C200', 'POST'))
FIELDS = ('C_adoption', 'C_floating', 'C_replaced_by_other_surface', 'C_wrong_layer',
          'whole_body_complete', 'support_missing', 'fragments_or_folded_layers', 'visible_join')
META = ('request_id', 'condition_id', 'parent_id', 'panel', 'split', 'condition_type', 'N', 'K', 'candidate', 'action')
DEFINITION = dict(version=VERSION, nonblind=True, pages=24, review_rows=96,
    column_order=[dict(label=a, stage=s) for a, s in COLUMNS],
    rows=['Whole mesh', 'C-local opposite camera', 'Front XY finite section', 'Side ZY finite section'],
    whole_camera=dict(projection='ORTHOGRAPHIC', elevation=22, azimuth=-60),
    local_camera=dict(projection='ORTHOGRAPHIC', elevation=23, azimuth=120),
    geometry_colors=dict(C='#1686c0', free='#afb6be', source_C_normals='#7c428f'),
    cuts='FRONT at midpoint of C Z bbox; SIDE at midpoint of C X bbox; positions depend only on C.',
    sections='Actual FP64 finite triangle-plane intersections; coplanar triangles contribute edges; no projected outlines.',
    scale='One whole-mesh union bound per condition across all models, stages, and noises; C-only local bounds identical within each condition.',
    local_margins_L=dict(X=.075, Y=.10, Z=.075), reference_length=C.L,
    unchanged=True, smoothing=False, candidate_coloring=False, unique_GT_ranking=False,
    review='NONBLIND AI_PROXY only after actual viewing; original geometry/hash saved. Render never fills judgment fields.',
    limitations='Two central sections can miss off-plane contacts or layers. Purple winding normals do not certify outward. Line count is not layer count. Visible contact is not indexed topology.',
    pixels=[2720, 2320], dpi=160)


def helper():
    return C.module('_regular_patch_finite_section_renderer', C.ROOT / 'r2/tools/cref_layer_audit_v1/render.py')


def definitions():
    paths = [Path(__file__), Path(C.__file__), Path(I.__file__),
             C.ROOT / 'r2/tools/cref_layer_audit_v1/render.py']
    return {str(p): C.sha(p) for p in paths}


def load_all(r):
    """Full validated barrier before rendering; no partial gallery selection."""
    source_hashes, cells = {}, {}
    C.package()
    from meshflow_control.source_winding import validate_source
    for q in r['requests']:
        inp = C.old().read_input(q)
        src = C.arrays(q['condition_source'])
        source = validate_source(src, inp['C'])
        source_hashes[q['input_path']] = q['input_sha256']
        source_hashes[q['condition_source']] = q['condition_source_sha256']
        for arm in I.LABELS:
            raw, post = I.read_raw(q, arm), I.read_post(q, arm)
            for stage, rec, field in (('RAW', raw, 'raw'), ('POST', post, 'post')):
                p = rec[field + '_path']; h = rec[field + '_sha256']
                source_hashes[p] = h
                a = C.arrays(p)
                tri = np.asarray(a['raw'] if stage == 'RAW' else a['output']).reshape(-1, 3, 3)
                k = int(q['K'])
                expected = inp['C'] if stage == 'RAW' else source['C_original']
                if not C.same(tri[:k], expected):
                    raise ValueError('Figure C contract failed')
                cells[(q['request_id'], arm, stage)] = dict(C=tri[:k], F=tri[k:],
                    C_original=source['C_original'], path=p, sha256=h,
                    C_array_sha256=C.array_sha(tri[:k]), F_array_sha256=C.array_sha(tri[k:]))
    if len(cells) != 96:
        raise ValueError('All 96 model/stage geometries required')
    I.verify(source_hashes)
    return cells, source_hashes


def figure(q, cells, bounds):
    H = helper(); plt = H._mpl()
    fig = plt.figure(figsize=(17, 14.5), dpi=160, facecolor='white')
    gs = fig.add_gridspec(4, 4, height_ratios=[1.4, 1.12, .80, .80],
                         left=.035, right=.987, top=.94, bottom=.075, wspace=.22, hspace=.24)
    for col, (arm, stage) in enumerate(COLUMNS):
        g = cells[q['request_id'], arm, stage]
        ax = fig.add_subplot(gs[0, col], projection='3d')
        H.draw_mesh(ax, g, [], bounds, C.L, close=False)
        ax.set_title(arm + ' | ' + stage + '\nWhole mesh; shared physical scale', fontsize=11, pad=5)
        ax = fig.add_subplot(gs[1, col], projection='3d')
        H.draw_mesh(ax, g, [], bounds, C.L, close=True)
        ax.set_title('C-local opposite view', fontsize=9, pad=1)
        H.draw_section(fig.add_subplot(gs[2, col]), g, [], bounds, C.L, 'front', 1)
        H.draw_section(fig.add_subplot(gs[3, col]), g, [], bounds, C.L, 'side', 1)
    fig.suptitle(q['parent_id'][:8] + ' | ' + q['condition_type'] +
                 f' | noise {q["candidate"]} | K={q["K"]}, N={q["N"]} | NONBLIND AI_PROXY', fontsize=15, y=.985)
    fig.text(.5, .044, 'BLUE: preserved C    GRAY: generated free    PURPLE: source winding normals, not outward certification', ha='center', fontsize=10)
    fig.text(.5, .026, 'Exact finite sections; no projected outlines. Two cuts may miss off-plane contacts/layers. Visible contact does not certify topology.', ha='center', fontsize=9)
    fig.text(.5, .010, 'Same C and fixed noise within each paired request. RAW and POST stay separate; source parent is not a unique required style.', ha='center', fontsize=9)
    return fig


def render():
    tick = time.perf_counter(); I.check_code(); r = I.registration()
    cells, source_hashes = load_all(r)
    H = helper(); groups = collections.defaultdict(list)
    for q in r['requests']:
        groups[q['condition_id']].append(q)
    if len(groups) != 6 or any(len(qs) != 4 for qs in groups.values()):
        raise ValueError('Expected six complete patches with four noises each')
    bounds = {cid: H.bounds_for_condition([cells[q['request_id'], arm, stage]
              for q in qs for arm, stage in COLUMNS], None, C.L) for cid, qs in groups.items()}
    definition_hashes = definitions()
    C.write(GALLERY / 'definition.json', DEFINITION)
    C.write(GALLERY / 'physical_bounds_and_cuts.json', bounds)
    C.write(GALLERY / 'source_hashes.json', source_hashes)
    images, reviews = [], []
    registration_hash = C.sha(C.OUT / 'registration.json')
    for q in r['requests']:
        path = C.checked(GALLERY / 'images' / (q['request_id'] + '.png'))
        geometry = [dict(label=a, stage=s, path=cells[q['request_id'], a, s]['path'],
                         sha256=cells[q['request_id'], a, s]['sha256'],
                         C_array_sha256=cells[q['request_id'], a, s]['C_array_sha256'],
                         F_array_sha256=cells[q['request_id'], a, s]['F_array_sha256']) for a, s in COLUMNS]
        identity = dict(version=VERSION, registration_sha256=registration_hash,
                        request_id=q['request_id'], definition=DEFINITION, definition_hashes=definition_hashes,
                        bounds=bounds[q['condition_id']], geometry=geometry)
        result = H.save_page(C, GALLERY, path, identity,
                             lambda q=q: figure(q, cells, bounds[q['condition_id']]))
        base = {key:q.get(key, '') for key in META}
        images.append(dict(base, image_path=str(path), image_sha256=result['image_sha256'],
                           pixels=result['pixels'], columns=DEFINITION['column_order']))
        for column, (arm, stage) in enumerate(COLUMNS, 1):
            row = dict(base, label=arm, stage=stage, column=column,
                       image_path=str(path), image_sha256=result['image_sha256'],
                       review_status='NOT_REVIEWED', reviewer_kind='AI_PROXY', reviewer='',
                       blinded=False, prior_familiarity='', reviewed_image_sha256='', notes='', limitations='')
            row.update({field:'' for field in FIELDS})
            reviews.append(row)
        print(json.dumps(dict(stage='RENDER', completed=len(images), requests=24, request_id=q['request_id'])), flush=True)
    if len(images) != 24 or len(reviews) != 96:
        raise ValueError('Incomplete figure/template count')
    C.csvwrite(GALLERY / 'images.csv', images)
    C.csvwrite(GALLERY / 'AI_PROXY_review_template.csv', reviews)
    cards = '\n'.join('<section><h2>' + html.escape(x['parent_id'][:8] + ' / noise ' + str(x['candidate'])) +
                      '</h2><a href="images/' + html.escape(Path(x['image_path']).name) + '"><img loading="lazy" src="images/' +
                      html.escape(Path(x['image_path']).name) + '"></a></section>' for x in images)
    page = '<!doctype html><meta charset="utf-8"><title>Regular patch paired meshes</title><style>body{font:16px sans-serif;margin:24px}img{width:100%;max-width:1500px}section{margin-bottom:32px}</style><h1>Regular patches: paired RAW / POST</h1><p>NONBLIND AI_PROXY. Blue C / gray free. Exact finite cuts; no source-GT style ranking. Judgments require actual viewing.</p>' + cards
    C._write_bytes(GALLERY / 'index.html', page.encode('utf-8'))
    I.verify(source_hashes)
    paths = [GALLERY / name for name in ('definition.json', 'physical_bounds_and_cuts.json', 'source_hashes.json', 'images.csv', 'AI_PROXY_review_template.csv', 'index.html')]
    out = dict(status='COMPLETE_RENDERING_NOT_REVIEW', version=VERSION, pages=24, review_rows=96,
               actual_reviews=0, model_calls=0, geometry_score_calls=0, new_geometry=0,
               definition_hashes=definition_hashes, files={str(p):C.sha(p) for p in paths},
               images={x['image_path']:x['image_sha256'] for x in images})
    C.write(GALLERY / 'manifest.json', out)
    C.event('GALLERY_COMPLETE', pages=24, review_rows=96, seconds=time.perf_counter()-tick, model_calls=0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('render',), default='render')
    parser.parse_args()
    result = render()
    print(json.dumps({k:v for k,v in result.items() if k not in ('files', 'images', 'definition_hashes')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
