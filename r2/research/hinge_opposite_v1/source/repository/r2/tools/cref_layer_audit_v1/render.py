"""Offline, nonblind layer-audit figures from frozen existing geometries only.

No torch, learned model, quality scorer, noise draw, smoothing, welding or mesh
mutation is used.  Six finite-plane cuts are fixed by C before seeing free.
Orange faces are descriptive proximity candidates, never automatic layer labels.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import collections
import csv
import hashlib
import html
import json
import time
from pathlib import Path
import numpy as np

VERSION = 'cref_layer_fixed_six_finite_sections_v1'
FRACTIONS = (.25, .50, .75)
MODEL_ORDER = ('SMALL_C200', 'CREF_C200')
REVIEW_FIELDS = ('sidewall_support', 'remaining_replacement_surface',
    'C_participates_in_seat', 'whole_body_complete', 'visible_local_connection',
    'evidence_sufficient')
LAYER_LABELS = ('NORMAL_THICKNESS_SUPPORTED', 'EXTRA_LAYER_SUPPORTED', 'MIXED',
    'UNCERTAIN', 'NO_NEAR_PARALLEL')
BLUE = '#1686c0'
GRAY = '#afb6be'
ORANGE = '#e88724'
NORMAL = '#7c428f'
DEFINITION = dict(version=VERSION,nonblind=True,reviewer_kind='AI_PROXY',
    pages_per_request=2,views_per_request='4 states x (whole + three front cuts + close opposite view + three side cuts)',
    row_order='SMALL RAW, CREF RAW, SMALL POST, CREF POST',
    cuts='Front XY at C Z-bbox 25/50/75%; side ZY at C X-bbox 25/50/75%. Positions depend only on C.',
    section_geometry='FP64 finite triangle-plane intersection; coplanar triangles contribute edges. No projected C outlines are drawn.',
    whole_bounds='Union of all registered cells and available optional reference of each C; identical physical scale across methods/noises/stages.',
    local_bounds='C bbox plus fixed .075L X/Z and .10L Y margins; shared by all six cuts and local view.',
    colors=dict(C=BLUE,free=GRAY,near_parallel_proximity_candidates=ORANGE,source_C_normals=NORMAL),
    orange='All F face IDs supplied by fixed original-query proximity diagnostic; not a prediction of redundant layers or illegality.',
    normals='Source C face winding, at up to three spatially fixed face IDs; positive is not certified outward/interior.',
    geometry_unchanged=True,smoothing=False,GT_style_ranking=False,
    section_limit='Six predetermined slices can miss off-plane contacts/layers. Coplanar edges can overprint; count neither lines nor connected components as layers.',
    figure_judgments='Renderer never fills review judgments. Optional source completion is local structural reference only, not a unique generation target.')


def triangles(value, name):
    a = np.asarray(value)
    if a.ndim == 2 and a.shape[1:] == (9,):
        a = a.reshape(-1, 3, 3)
    if a.ndim != 3 or a.shape[1:] != (3, 3) or a.dtype.kind != 'f' or not np.isfinite(a).all():
        raise ValueError(name + ': expected finite triangle array')
    return a


def finite_sections(tri, axis, position, eps=1e-11):
    """All finite intersections, retaining source face IDs (no projected edges)."""
    segments, owners = [], []
    for index, face in enumerate(np.asarray(tri, dtype=np.float64)):
        delta = face[:, axis] - float(position)
        if np.all(np.abs(delta) <= eps):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                if np.linalg.norm(face[a] - face[b]) > eps:
                    segments.append(face[[a, b]])
                    owners.append(index)
            continue
        points = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if abs(delta[a]) <= eps:
                points.append(face[a])
            if delta[a] * delta[b] < 0:
                points.append(face[a] + (face[b] - face[a]) * (-delta[a] / (delta[b] - delta[a])))
        unique = []
        for point in points:
            if not any(np.linalg.norm(point - prior) <= eps for prior in unique):
                unique.append(point)
        if len(unique) >= 2:
            pair = max(((a, b) for a in range(len(unique)) for b in range(a + 1, len(unique))),
                       key=lambda ab: np.linalg.norm(unique[ab[0]] - unique[ab[1]]))
            segment = np.asarray([unique[pair[0]], unique[pair[1]]])
            if np.linalg.norm(segment[0] - segment[1]) > eps:
                segments.append(segment)
                owners.append(index)
    return np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3), np.asarray(owners, dtype=np.int64)


def source_normals(original_C):
    tri = np.asarray(original_C, dtype=np.float64)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, length[:, None], out=np.zeros_like(cross), where=length[:, None] > 0)
    return tri.mean(1), normals


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def bounds_for_condition(geometries, reference, L):
    C0 = triangles(geometries[0]['C'], 'C')
    # Face/corner winding may differ by stage; coordinate multiset must agree.
    cv = C0.reshape(-1, 3)
    canonical = cv[np.lexsort((cv[:, 2], cv[:, 1], cv[:, 0]))]
    for g in geometries:
        cc = triangles(g['C'], 'C')
        vertices = cc.reshape(-1, 3)
        sorted_vertices = vertices[np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))]
        if cc.shape != C0.shape or not np.array_equal(sorted_vertices, canonical):
            raise ValueError('C coordinates differ within registered condition')
    xyz = [triangles(g[k], k).reshape(-1, 3).astype(float) for g in geometries for k in ('C', 'F')]
    if reference is not None:
        xyz.extend(triangles(reference[k], k).reshape(-1, 3).astype(float) for k in ('C', 'F'))
    xyz = np.concatenate(xyz)
    low, high = xyz.min(0), xyz.max(0)
    cc = C0.reshape(-1, 3).astype(float)
    clo, chi = cc.min(0), cc.max(0)
    margins = np.asarray([.075, .10, .075]) * L
    return dict(whole_center=((low + high) / 2).tolist(),
        whole_span=float(max((high - low).max(), .05 * L) * 1.10),
        local_low=(clo - margins).tolist(), local_high=(chi + margins).tolist(),
        C_low=clo.tolist(), C_high=chi.tolist(),
        front_positions=(clo[2] + np.asarray(FRACTIONS) * (chi[2] - clo[2])).tolist(),
        side_positions=(clo[0] + np.asarray(FRACTIONS) * (chi[0] - clo[0])).tolist())


def draw_mesh(ax, g, candidates, bounds, L, *, close=False):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    C0, free = triangles(g['C'], 'C'), triangles(g['F'], 'F')
    candidate = np.zeros(len(free), bool)
    candidate[np.asarray(candidates, np.int64)] = True
    groups = ((free[~candidate], GRAY, '#525b64', .30),
              (free[candidate], ORANGE, '#905018', .38), (C0, BLUE, '#09527e', .52))
    for tri, color, edge, width in groups:
        if len(tri):
            artist = Poly3DCollection(tri[..., (0, 2, 1)], facecolors=color,
                edgecolors=edge, linewidths=width, alpha=1.0, zsort='average')
            ax.add_collection3d(artist)
    centers, normals = source_normals(g['C_original'])
    chosen = np.unique(np.linspace(0, len(centers) - 1, min(3, len(centers))).astype(int))
    if len(chosen):
        pp = centers[chosen][:, (0, 2, 1)]
        nn = normals[chosen][:, (0, 2, 1)]
        ax.quiver(pp[:, 0], pp[:, 1], pp[:, 2], nn[:, 0], nn[:, 1], nn[:, 2],
            length=.035 * L, color=NORMAL, linewidth=.85, arrow_length_ratio=.25, normalize=False)
    if close:
        lo, hi = np.asarray(bounds['local_low']), np.asarray(bounds['local_high'])
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2]); ax.set_zlim(lo[1], hi[1])
        ax.set_box_aspect((hi - lo)[[0, 2, 1]])
        ax.view_init(elev=23, azim=120)
    else:
        center = np.asarray(bounds['whole_center'])[[0, 2, 1]]
        radius = bounds['whole_span'] / 2
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-60)
    ax.set_proj_type('ortho')
    ax.set_axis_off()


def draw_section(ax, g, candidates, bounds, L, view, index):
    from matplotlib.collections import LineCollection
    axis, project = (2, (0, 1)) if view == 'front' else (0, (2, 1))
    pos = bounds['front_positions' if view == 'front' else 'side_positions'][index]
    cc, ci = finite_sections(g['C'], axis, pos, 1e-12 * L)
    ff, fi = finite_sections(g['F'], axis, pos, 1e-12 * L)
    selected = np.isin(fi, np.asarray(candidates, np.int64))
    if len(ff):
        ax.add_collection(LineCollection(ff[~selected][:, :, project], colors='#747b83', linewidths=.9, zorder=2))
        # Orange below thin blue preserves a halo at exactly coincident cuts.
        ax.add_collection(LineCollection(ff[selected][:, :, project], colors=ORANGE, linewidths=2.4, alpha=.85, zorder=3))
    if len(cc):
        ax.add_collection(LineCollection(cc[:, :, project], colors=BLUE, linewidths=1.45, zorder=4))
        _, normals = source_normals(g['C_original'])
        chosen = np.unique(np.linspace(0, len(cc)-1, min(2, len(cc))).astype(int))
        for j in chosen:
            anchor = cc[j].mean(0)[list(project)]
            vec = normals[ci[j]][list(project)] * (.027 * L)
            if np.linalg.norm(vec) > 1e-9:
                ax.annotate('', xy=anchor + vec, xytext=anchor,
                    arrowprops=dict(arrowstyle='-|>', color=NORMAL, lw=.8, mutation_scale=8), zorder=5)
    lo, hi = np.asarray(bounds['local_low']), np.asarray(bounds['local_high'])
    ax.set_xlim(lo[project[0]], hi[project[0]]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.15, linewidth=.45); ax.tick_params(labelsize=7, width=.5, length=2)
    ax.set_xlabel(('X' if view == 'front' else 'Z') + ' (model)', fontsize=8)
    ax.set_ylabel('Y (model)', fontsize=8)
    ax.set_title(f'{view.upper()} {int(FRACTIONS[index]*100)}% | '+('Z' if view=='front' else 'X')+f'={pos:.5f}', fontsize=9)
    if not len(cc):
        ax.text(.02, .94, 'No finite C intersection', transform=ax.transAxes, fontsize=7, color=BLUE)


def request_figure(request, ordered, geometry, candidate_map, bounds, L, page):
    plt = _mpl(); fig = plt.figure(figsize=(18, 13.3), dpi=150, facecolor='white')
    view = 'front' if page == 'FRONT' else 'side'
    for row, cell in enumerate(ordered):
        cid = cell['cell_id']; g = geometry[cid]; ids = candidate_map[cid]['indices']
        label = cell.get('label', cell.get('model', ''))
        ax = fig.add_subplot(4, 4, row*4+1, projection='3d')
        draw_mesh(ax, g, ids, bounds, L, close=page=='SIDE')
        ax.set_title(f'{label} | {cell["stage"]}\n'+('C-local close view (opposite camera)' if page=='SIDE' else 'WHOLE, common scale'), fontsize=10)
        for j in range(3):
            ax = fig.add_subplot(4, 4, row*4+j+2)
            draw_section(ax, g, ids, bounds, L, view, j)
    parent = request.get('parent_id', request['request_id'])[:8]
    noise = request.get('candidate', request.get('noise', '?'))
    fig.suptitle(f'{parent} | noise {noise} | {page} layer audit | NONBLIND AI_PROXY', fontsize=15, y=.985)
    fig.text(.5, .055, 'BLUE: fixed C   GRAY: free   ORANGE: near-parallel proximity candidates, NOT an extra-layer/illegal verdict', ha='center', fontsize=10)
    fig.text(.5, .036, 'PURPLE arrows: source C winding normals, NOT outward certification. Cuts are finite intersections; no projected outlines.', ha='center', fontsize=9)
    fig.text(.5, .017, 'Same C-derived cut positions and local scale. Six slices can miss off-plane walls/contact. Do not infer layer count from line count.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.025, right=.99, top=.94, bottom=.09, wspace=.22, hspace=.34)
    return fig


def reference_figure(cell, g, candidate, bounds, L):
    plt = _mpl(); fig = plt.figure(figsize=(18, 7.8), dpi=150, facecolor='white')
    for row, view in enumerate(('front', 'side')):
        ax = fig.add_subplot(2, 4, row*4+1, projection='3d')
        draw_mesh(ax, g, candidate['indices'], bounds, L, close=bool(row))
        ax.set_title('SOURCE COMPLEMENT reference only\n'+('close opposite view' if row else 'whole common scale'), fontsize=10)
        for j in range(3):
            draw_section(fig.add_subplot(2, 4, row*4+j+2), g, candidate['indices'], bounds, L, view, j)
    fig.suptitle(cell.get('parent_id', cell['condition_id'])[:8]+' | source C + existing source free | NOT a required target style', fontsize=15, y=.98)
    fig.text(.5, .055, 'Local thickness/sidewall reference only. Source geometry is not automatically clean or a unique valid completion.', ha='center', fontsize=10)
    fig.text(.5, .025, 'BLUE C; GRAY free; ORANGE proximity candidates if available; PURPLE source normals (not outward). Exact finite sections.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.025,right=.99,top=.90,bottom=.12,wspace=.22,hspace=.32)
    return fig


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def _common():
    from . import common
    return common


def candidate_information(C, cell, g, *, optional=False):
    path = C.OUT/'diagnostics'/cell['cell_id']/'summary.json'
    if 'near_parallel_free_indices' in g:
        indices, source = g['near_parallel_free_indices'], {'origin':'load_geometry'}
    elif path.exists():
        info = C.read(path)
        if 'near_parallel_free_indices' not in info:
            raise ValueError('Candidate summary lacks explicit F-local IDs: '+str(path))
        indices, source = info['near_parallel_free_indices'], {'path':str(path),'sha256':C.sha(path)}
    elif optional:
        return dict(indices=[],status='NA_NO_REFERENCE_PROXIMITY_DIAGNOSTIC',source={})
    else:
        raise FileNotFoundError('Wait for fixed candidate diagnostic: '+str(path))
    ids = np.asarray(indices, np.int64)
    if ids.ndim != 1 or len(ids) != len(np.unique(ids)) or (len(ids) and (ids.min()<0 or ids.max()>=len(g['F']))):
        raise ValueError('Invalid F-local candidate IDs')
    return dict(indices=ids.tolist(), status='AVAILABLE', source=source)


def save_page(C, gallery, path, identity, draw):
    receipt = gallery/'receipts'/(path.stem+'.json')
    if receipt.exists():
        prior = C.read(receipt)
        if prior['identity'] != identity or C.sha(path) != prior['image_sha256']:
            raise ValueError('Frozen figure/identity changed: '+str(path))
        return prior
    if path.exists():
        raise RuntimeError('Incomplete figure without receipt requires explicit engineering recovery: '+str(path))
    plt = _mpl(); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    fig = draw()
    with temp.open('xb') as handle:
        fig.savefig(handle, format='png')
    plt.close(fig); os.replace(temp, path)
    from PIL import Image
    with Image.open(path) as img:
        size = list(img.size); img.verify()
    out = dict(status='RENDERED_NOT_REVIEWED',identity=identity,image_path=str(path),image_sha256=C.sha(path),pixels=size)
    C.write(receipt,out)
    return out


def render():
    C = _common(); tick = time.perf_counter()
    reg_path = C.OUT/'registration.json'; reg = C.read(reg_path)
    cells = reg['cells']; requests = reg['requests']; references = reg.get('reference_cells', reg.get('references', []))
    if len(cells)!=44 or len(requests)!=11 or len(references)>6:
        raise ValueError('Expected frozen 44 cells, 11 requests, <=6 optional references')
    if len({x['cell_id'] for x in cells}) != 44:
        raise ValueError('Duplicate cell IDs')
    L = float(getattr(C,'L',np.sqrt(3)*2/.3762))
    gallery = C.OUT/'gallery'; gallery.mkdir(parents=True,exist_ok=True)
    identity = dict(version=VERSION,registration_sha256=C.sha(reg_path),render_source_sha256=C.sha(Path(__file__)),common_source_sha256=C.sha(Path(C.__file__)))
    source_files = {}
    for cell in cells + references:
        for field in ('geometry_path', 'input_path', 'condition_source_path', 'query_path'):
            p = cell.get(field)
            if p and C.resolve(p).exists():
                source_files[str(p)] = C.sha(p)
        diag = C.OUT/'diagnostics'/cell['cell_id']/'summary.json'
        if diag.exists():
            source_files[str(diag)] = C.sha(diag)
    identity['source_file_hashes'] = source_files
    final = gallery/'manifest.json'
    if final.exists():
        old = C.read(final)
        if old['identity'] != identity:
            raise ValueError('Gallery implementation/input identity changed')
        for p,h in old['output_hashes'].items():
            if C.sha(p)!=h:raise ValueError('Completed gallery changed: '+p)
        return old
    C.write(gallery/'definition.json',DEFINITION)
    geometry, candidates, array_hashes = {}, {}, {}
    for cell in cells:
        g=C.load_geometry(cell)
        for name in ('C','F','C_original'):
            g[name]=triangles(g[name],name)
        geometry[cell['cell_id']]=g
        candidates[cell['cell_id']]=candidate_information(C,cell,g)
        array_hashes[cell['cell_id']]={name:array_hash(g[name]) for name in ('C','F','C_original')}
    reference_geometry={};reference_by_condition={};reference_skips=[]
    for cell in references:
        if cell.get('status') in ('NA','BLOCKED','MISSING'):
            reference_skips.append(cell);continue
        try:g=C.load_geometry(cell)
        except FileNotFoundError as error:
            reference_skips.append(dict(cell_id=cell['cell_id'],status='BLOCKED',reason=str(error)));continue
        for name in ('C','F','C_original'):g[name]=triangles(g[name],name)
        reference_geometry[cell['cell_id']]=g
        reference_by_condition[cell['condition_id']]=cell
        candidates[cell['cell_id']]=candidate_information(C,cell,g,optional=True)
        array_hashes[cell['cell_id']]={name:array_hash(g[name]) for name in ('C','F','C_original')}
    grouped=collections.defaultdict(list)
    for cell in cells:grouped[cell['condition_id']].append(cell)
    bounds={}
    for cid,cc in grouped.items():
        ref=reference_by_condition.get(cid)
        bounds[cid]=bounds_for_condition([geometry[x['cell_id']] for x in cc],
            reference_geometry[ref['cell_id']] if ref else None,L)
    C.write(gallery/'physical_bounds_and_cuts.json',bounds)
    C.write(gallery/'source_array_hashes.json',array_hashes)
    images=[];review=[];source_images=[];cards=[]
    for request in requests:
        rid=request['request_id']; cc=[x for x in cells if x['request_id']==rid]
        if len(cc)!=4 or {(x.get('label',x.get('model')),x['stage']) for x in cc}!={(m,s) for m in MODEL_ORDER for s in ('RAW','POST')}:
            raise ValueError('Each request must have exactly paired SMALL/CREF RAW/POST')
        cc.sort(key=lambda x:(('RAW','POST').index(x['stage']),MODEL_ORDER.index(x.get('label',x.get('model')))))
        cid=cc[0]['condition_id'];pages=[]
        for page in ('FRONT','SIDE'):
            path=gallery/'images'/(rid+'__'+page+'.png')
            image_identity=dict(run=identity,request_id=rid,page=page,cells=cc,bounds=bounds[cid],
                source_array_hashes={x['cell_id']:array_hashes[x['cell_id']] for x in cc},
                candidates={x['cell_id']:candidates[x['cell_id']] for x in cc})
            rec=save_page(C,gallery,path,image_identity,lambda page=page:request_figure(request,cc,geometry,candidates,bounds[cid],L,page))
            images.append(dict(request_id=rid,condition_id=cid,parent_id=request.get('parent_id',''),page=page,image_path=rec['image_path'],image_sha256=rec['image_sha256']))
            pages.append(rec)
        for cell in cc:
            review.append(dict(cell_id=cell['cell_id'],request_id=rid,condition_id=cid,parent_id=cell.get('parent_id',request.get('parent_id','')),
                label=cell.get('label',cell.get('model')),stage=cell['stage'],selection_group=request.get('selection_reason',request.get('selection_group',request.get('reason',''))),
                review_status='NOT_REVIEWED',reviewer='AI_PROXY',blinded=False,prior_familiarity='UNASSESSED',
                image_paths=json.dumps([p['image_path'] for p in pages]),image_sha256s=json.dumps([p['image_sha256'] for p in pages]),
                reviewed_image_sha256s='',observed_layer_relation='UNCERTAIN',
                **{field:'UNCERTAIN' for field in REVIEW_FIELDS},notes='',limitations='Six fixed cuts can miss off-plane walls; normal sign is not inside/outside; orange is a candidate only.'))
        cards.append('<h2>'+html.escape(rid)+'</h2>'+''.join('<a href="images/'+Path(p['image_path']).name+'"><img loading="lazy" style="width:100%;max-width:2700px" src="images/'+Path(p['image_path']).name+'"></a>' for p in pages))
        print(json.dumps(dict(stage='RENDER',requests_done=len(images)//2,requests_total=11)),flush=True)
    for cid,cell in sorted(reference_by_condition.items()):
        sid=cell['cell_id'];g=reference_geometry[sid];path=gallery/'source_references'/(cid+'__SOURCE_REFERENCE.png')
        ref_identity=dict(run=identity,reference=cell,bounds=bounds[cid],source_array_hashes=array_hashes[sid],candidate=candidates[sid])
        rec=save_page(C,gallery,path,ref_identity,lambda:reference_figure(cell,g,candidates[sid],bounds[cid],L))
        source_images.append(dict(cell_id=sid,condition_id=cid,parent_id=cell.get('parent_id',''),image_path=rec['image_path'],image_sha256=rec['image_sha256']))
    if len(images)!=22 or len(review)!=44:raise ValueError('Figure/review count mismatch')
    C.csvwrite(gallery/'images.csv',images);C.csvwrite(gallery/'AI_PROXY_review_template.csv',review)
    C.csvwrite(gallery/'source_reference_images.csv',source_images)
    C.write(gallery/'source_reference_status.json',dict(available=len(source_images),skipped=reference_skips))
    links=''.join('<li><a href="source_references/'+Path(r['image_path']).name+'">'+html.escape(r['condition_id'])+'</a></li>' for r in source_images)
    content='<!doctype html><meta charset="utf-8"><title>CREF layer audit</title><h1>Offline CREF layer audit: NONBLIND AI_PROXY</h1><p>All 11 selected requests, both models and RAW/POST. Exact finite cuts are fixed from C; candidate colors do not determine judgments. No training, model calls, geometry repair or original scoring rerun.</p><h2>Optional source structural references, not unique GT</h2><ul>'+links+'</ul>'+''.join(cards)
    path=gallery/'index.html'
    if path.exists() and path.read_text(encoding='utf-8')!=content:raise ValueError('Existing HTML differs')
    path.write_text(content,encoding='utf-8')
    paths=[gallery/x for x in ('definition.json','physical_bounds_and_cuts.json','source_array_hashes.json','images.csv','AI_PROXY_review_template.csv','source_reference_images.csv','source_reference_status.json','index.html')]
    paths += [Path(x['image_path']) for x in images+source_images]
    out=dict(status='RENDERED_NOT_REVIEWED',identity=identity,request_pages=22,reference_pages=len(source_images),review_rows=44,
        seconds=time.perf_counter()-tick,output_hashes={str(p):C.sha(p) for p in paths},actual_review='NOT_RUN_BY_RENDERER',model_calls=0,quality_score_calls=0)
    C.write(final,out);return out


def self_test():
    tri=np.asarray([[[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]])
    seg,ids=finite_sections(tri,0,.5)
    assert seg.shape==(1,2,3) and ids.tolist()==[0]
    assert np.all(seg[:,:,0]==.5) and np.all(seg[:,:,2]==0)
    assert np.allclose(np.sort(seg[0,:,1]),[0.,1.5])
    empty,_=finite_sections(tri,0,3.)
    assert empty.shape==(0,2,3)
    coplanar,_=finite_sections(tri,2,0.)
    assert len(coplanar)==3
    print(json.dumps(dict(status='PASS',test='finite_triangle_intersection_cpu_geometry_only',model_calls=0,quality_score_calls=0)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=('render','self-test'))
    args=parser.parse_args()
    if args.stage=='self-test':self_test()
    else:print(json.dumps(render(),ensure_ascii=False),flush=True)

if __name__=='__main__':main()