"""Frozen motif-guidance evaluation; import and CPU commands never import torch.

D0 is available only through the explicit d0_batch entrypoint, after the 200
RAW barrier and lambda lock. It delegates to the unchanged official-D0 runner.
No function generates visual ratings. Calibration reviews must cite viewed PNGs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import csv
import hashlib
import html
import inspect
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
import types

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from . import prepare as P
sys.path.insert(0, str(P.ROOT/'r2/src'))

VERSION = 'bench_full_prior_motif_guidance_original177_evaluation_v1'
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')
ARMS = ('R', 'G0', 'G1', 'G2', 'G3')
LAMBDAS = (0.1, 0.3, 1.0)
I_FIELDS = ('ConflictCov_C', 'ConflictCov_C_upper', 'ConflictCov_C_cross',
            'ConflictCov_C_dup', 'ConflictCov_C_cross_and_dup')


def _cpu():
    if 'torch' in sys.modules:
        raise RuntimeError('Use a fresh CPU process without torch')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _verify(files):
    for path, digest in files.items():
        if P.sha(path) != digest:
            raise ValueError('Frozen file changed: ' + str(path))


def _book(paths):
    return {str(P.resolve(p)): P.sha(p) for p in paths}


def _cached(path):
    path = P.resolve(path)
    if not path.exists():
        return None
    saved = P.read(path)
    if saved.get('source_sha256') != P.sha(__file__):
        raise ValueError('Completed evaluation implementation changed')
    _verify(saved.get('protected', {}))
    _verify(saved.get('files', {}))
    return saved


def _seal(path, data, files, protected):
    result = dict(data, source_sha256=P.sha(__file__), files=_book(files), protected=protected)
    P.write_immutable(path, result)
    return result


def _csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    _text(path, buf.getvalue())
    return P.resolve(path)


def _text(path, value):
    path = P.resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value.encode('utf-8')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refusing to overwrite result: ' + str(path))
    else:
        with path.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())


def _calibration_tasks():
    tasks = [t for t in P.load_tasks() if t['panel_role'] == 'calibration' and t['arm'] != 'G2']
    expected = {'G0': 8, 'G1': 8, 'G3': 24}
    if Counter(t['arm'] for t in tasks) != expected or len({t['cell_id'] for t in tasks}) != 40:
        raise ValueError('Expected the registered forty calibration RAW tasks')
    return tasks


def _calibration_barrier():
    tasks = _calibration_tasks()
    failed = [str(P.resolve(t['output_dir'])/'failure.json') for t in tasks
              if (P.resolve(t['output_dir'])/'failure.json').exists()]
    if failed:
        raise RuntimeError(json.dumps(dict(status='BLOCKED_CALIBRATION_FAILURE_REGISTRATION_REQUIRED',
            failure_receipts=failed,automatic_retries=0,
            reason='Preserve failures. Controller must explicitly register excluded lambda tasks before continuing; no silent replay or denominator change.')))
    for task in tasks:
        P.verify_raw(task)
    return tasks


def _registered(task, selected=False):
    tasks = P.selected_tasks() if selected else P.load_tasks(resolved=_lock_path().exists())
    by = {t['cell_id']: t for t in tasks}
    if task != by.get(task['cell_id']):
        raise ValueError('Task differs from its resolved frozen registration')
    return task


def _lock_path():
    # prepare.py resolves this same lock; no registered task is rewritten.
    return P.OUT/'lock.json'


def freeze_evaluation_code():
    """Independent late freeze; never modifies the generator code registration."""
    path=P.OUT/'evaluation_code.json'
    source=P.ROOT/'experiments/r2_bench_four_routes_v1/exposure/evaluation/score_registration.json'
    original=P.read(source)['source_hashes']
    _verify(original)
    files=dict(original)
    files.update(_book([__file__,Path(__file__).with_name('run.py'),source,
                       P.ROOT/'r2/pilots/bench_c_edgeflip_v1/candidate_post.py',
                       P.ROOT/'r2/pilots/chair_multipart_seam_v1/gallery.py']))
    value=dict(status='FROZEN',version=VERSION,source_score_registration=str(source),
               source_hashes=files,separate_from_generation_registration=True)
    if path.exists():
        saved=P.read(path)
        if saved!=value:raise ValueError('Frozen evaluation code changed')
        _verify(saved['source_hashes'])
        return saved
    P.write_immutable(path,value)
    return value


def score_raw(task):
    """One original RAW177 scan, with the original fixed C query descriptor."""
    _cpu()
    freeze_evaluation_code()
    _registered(task)
    if task['panel_role'] == 'calibration' and task['arm'] != 'G2':
        _calibration_barrier()
    P.verify_raw(task)
    from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
    from r2.pilots.seam_abc_v1 import eval_common as C
    dest = P.OUT/'raw_scores'/task['cell_id']
    receipt = dest/'audit_receipt.json'
    identity = dict(cell_id=task['cell_id'], request_id=task['request_id'], stage='RAW',
                    raw_sha256=P.sha(task['raw_path']), query_sha256=task['q']['query_descriptor']['sha256'],
                    tasks_sha256=P.sha(P.OUT/'tasks.json'), scorer_sha256=P.sha(CPU.M.__file__))
    saved = _cached(receipt)
    if saved:
        if saved['identity'] != identity:
            raise ValueError('Completed RAW score identity changed')
        return saved
    if dest.exists():
        raise RuntimeError('Partial RAW score retained; explicit recovery required: ' + str(dest))
    tick = time.perf_counter()
    payload = CPU.score_current(task['q'], C.arrays(task['raw_path']), dest, task['label'], seam_model=False)
    protected = _book([task['raw_path'], P.raw_receipt(task), task['q']['query_descriptor']['path'],
                       task['q']['inference_npz'], P.OUT/'tasks.json', P.OUT/'evaluation_code.json', CPU.M.__file__])
    return _seal(receipt, dict(status='COMPLETE', identity=identity, row=payload['row'],
                 model_calls=0, D0_calls=0, full_quality_scans=1, new_queries=0,
                 seconds=time.perf_counter()-tick), [p for p in dest.iterdir() if p.is_file()], protected)


def _post_proxy():
    """Clone function globals only; retain the original official D0 implementation."""
    from r2.pilots.bench_c_edgeflip_v1 import candidate_post as old
    ns = dict(vars(old))
    for name, value in vars(old).items():
        if inspect.isfunction(value) and value.__globals__ is vars(old):
            clone = types.FunctionType(value.__code__, ns, value.__name__, value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            ns[name] = clone
    def barrier(tasks_path=None):
        if _cached(_lock_path()) is None:
            raise ValueError('Verified lambda lock required')
        tasks, receipt = P.verify_raw_barrier()
        if len(tasks) != 200 or receipt.get('outputs') != 200:
            raise ValueError('All 200 sealed RAW outputs must precede D0')
        selected = P.selected_tasks()
        if len(selected) != 184 or Counter(t['arm'] for t in selected) != dict.fromkeys(('G0','G1','G2','G3'), 46):
            raise ValueError('Only the selected 184 main-panel tasks may receive D0')
        return selected, receipt
    def identity(task, tasks_path=None):
        return dict(version=VERSION, cell_id=task['cell_id'], arm=task['arm'],
                    lambda_value=task['lambda_value'], request_id=task['request_id'],
                    tasks_sha256=P.sha(P.OUT/'tasks.json'), lambda_lock_sha256=P.sha(_lock_path()),
                    raw_sha256=P.sha(task['raw_path']), source_sha256=P.sha(task['source_npz']),
                    input_sha256=P.sha(task['q']['inference_npz']),
                    query_sha256=task['q']['query_descriptor']['sha256'],
                    evaluator_sha256=P.sha(__file__), delegated_code_sha256=P.sha(old.__file__))
    ns.update(ROOT=P.OUT, TASKS=P.OUT/'tasks.json', VERSION=VERSION,
              load_tasks=lambda tasks_path=None: P.selected_tasks(), verify_raw_barrier=barrier,
              _task_identity=identity, _raw_receipt=P.raw_receipt)
    return types.SimpleNamespace(**ns)


def d0_batch(indices=None, worker='d0_0', estimated_peak_mib=3500):
    """Explicit GPU dispatch only; source runner enforces single D0/no replay."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        raise RuntimeError('D0 must use a fresh GPU process, not a CPU-scoring process')
    freeze_evaluation_code()
    proxy = _post_proxy()
    return proxy.d0_batch(P.OUT/'tasks.json', indices, worker, estimated_peak_mib)


def post(task):
    """Original CPU BASE scoring -> HINGE_OPP015 -> ALPHA -> HOLE -> FINAL."""
    _cpu()
    freeze_evaluation_code()
    _registered(task, selected=True)
    return _post_proxy().post(task, P.OUT/'tasks.json')


def _baseline():
    tasks = P.read(P.OUT/'baseline.json')
    if len(tasks) != 46 or len({t['request_id'] for t in tasks}) != 46:
        raise ValueError('Expected exactly the frozen EXPOSURE46 cache')
    return tasks


def _spec(task, stage):
    if task['arm'] == 'R':
        mesh = P.resolve(task['stage_paths'][stage])
        score = P.resolve(task['score_paths'][stage])
    else:
        mesh = (P.resolve(task['raw_path']) if stage == 'RAW' else
                P.OUT/'post'/task['cell_id']/'post.npz' if stage == 'BASE_POST' else
                P.OUT/'cpu'/task['cell_id']/'CLEAN_FINAL_POST/post.npz')
        score = (P.OUT/'raw_scores'/task['cell_id'] if stage == 'RAW' else
                 P.OUT/'cpu'/task['cell_id']/stage/'scores')
    return dict(task=task, stage=stage, mesh=mesh, score=score)


def _score_read(spec):
    """Authenticate existing score files; never call a detector in collection."""
    score = spec['score']
    done = P.read(score/'complete.json')
    if done.get('status') not in ('PASS', 'COMPLETE'):
        raise ValueError('Unsealed score: ' + str(score))
    _verify(done['files'])
    row = P.read(score/'row.json')
    q = spec['task']['q']
    for key in ('request_id', 'condition_id', 'parent_id'):
        if row[key] != q[key]:
            raise ValueError('Scored request identity changed: ' + key)
    for key in ('N', 'K', 'noise_index'):
        if int(row[key]) != int(q[key]):
            raise ValueError('Scored numeric identity changed: ' + key)
    if row['C_preserved_exact'] is not True:
        raise ValueError('Scored C identity is not exact')
    return row


def _fields():
    fields = P.read(P.OUT/'inputs/metric_fields.json')
    if len(fields) != 177 or len(set(fields)) != 177:
        raise ValueError('Original177 metric dictionary changed')
    return fields


def _metadata(task, stage, main=True):
    q = task['q']
    return dict(cell_id=task['cell_id'], arm=task['arm'], label=task['label'], stage=stage,
                panel_role=task['panel_role'], main_panel=main, lambda_value=task.get('lambda_value'),
                request_id=q['request_id'], parent_id=q['parent_id'], condition_id=q['condition_id'],
                noise_index=int(q['noise_index']), split=q['split'], report_group=q['report_group'],
                N=int(q['N']), K=int(q['K']), result_origin='REUSED_READ_ONLY' if task['arm']=='R' else 'NEW')


def _number(value):
    # Original boolean interface/preservation metrics aggregate as 0/1 rates.
    return isinstance(value, (int, float)) and math.isfinite(value)


def _aggregate(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    result = []
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        out = dict(zip(keys, group));out['denominator_rows'] = len(values)
        for field in fields:
            valid = [r[field] for r in values if _number(r.get(field)) and
                     (field not in I_FIELDS or r.get('I_status') == 'PASS')]
            out[field] = sum(valid)/len(valid) if valid else None
            out[field+'__valid_count'] = len(valid)
        # Parent means qualify only if every noise has original PASS support.
        out['I_status'] = 'PASS' if all(r.get('I_status') == 'PASS' for r in values) else 'NONPASS_RETAINED'
        result.append(out)
    return result


def _pairs(rows, fields):
    indexed = {(r.get('method',r['arm']), r['request_id'], r['stage']): r for r in rows if r['main_panel']}
    if len(indexed)!=sum(bool(r['main_panel']) for r in rows):
        raise ValueError('Paired table contains duplicate method/request/stage identities')
    methods={key[0] for key in indexed}
    contrasts=[('G3','G1'),('G3','G2'),('G3','R'),('G1','G0'),('G2','G1')]
    for method in sorted(methods):
        if method.startswith('G3_L'):
            contrasts.extend((method,reference) for reference in ('G1','G0','R'))
    result = []
    for candidate, reference in contrasts:
        for (arm, rid, stage), row in sorted(indexed.items()):
            if arm != candidate or (reference, rid, stage) not in indexed:
                continue
            base = indexed[reference, rid, stage]
            out = {k: row[k] for k in ('request_id','parent_id','condition_id','noise_index','panel_role','report_group','stage')}
            out.update(candidate=candidate, reference=reference, I_status='PASS' if row['I_status']==base['I_status']=='PASS' else 'NONPASS_RETAINED')
            for field in fields:
                a,b = base.get(field), row.get(field)
                out[field+'__reference'] = a;out[field+'__candidate'] = b
                out[field] = b-a if _number(a) and _number(b) and (field not in I_FIELDS or out['I_status']=='PASS') else None
            result.append(out)
    return result


def collect(mode='final'):
    """Copy original177 fields and derive deterministic parent/matched summaries."""
    _cpu()
    if mode not in ('calibration', 'final'):
        raise ValueError(mode)
    folder = P.OUT/'analysis'/mode
    done = folder/'complete.json'
    saved = _cached(done)
    if saved:return saved
    baseline = _baseline()
    if mode == 'calibration':
        tasks = _calibration_barrier() + [t for t in baseline if t['panel_role']=='calibration']
        specs = [_spec(t, 'RAW') for t in tasks]
        main_ids = {t['cell_id'] for t in tasks}
    else:
        P.verify_raw_barrier()
        tasks = P.selected_tasks() + baseline
        main_ids = {t['cell_id'] for t in tasks}
        specs = [_spec(t, s) for t in tasks for s in STAGES]
        specs += [_spec(t,'RAW') for t in P.load_tasks(resolved=True) if t['cell_id'] not in main_ids]
    fields = _fields();rows=[];protected=_book([P.OUT/'tasks.json', P.OUT/'baseline.json'])
    for spec in specs:
        task=spec['task'];source=_score_read(spec)
        row=dict(_metadata(task,spec['stage'],task['cell_id'] in main_ids),
                 source_label_field=source.get('label'),source_stage_field=source.get('stage'),
                 **{k:source.get(k) for k in fields})
        row['I_status']=source['I_status']
        rows.append(row)
        protected.update(_book([spec['mesh'],spec['score']/'row.json',spec['score']/'complete.json']))
    expected = 48 if mode=='calibration' else 706
    if len(rows)!=expected:raise ValueError('Unexpected stage-row count')
    files=[_csv(folder/'full177_per_output.csv', rows)]
    # Method label retains each calibration lambda; main arm merges selected G3 only.
    main=[dict(r,method=(r['label'] if mode=='calibration' else r['arm'])) for r in rows if r['main_panel']]
    parentkeys=('method','stage','panel_role','report_group','parent_id')
    parents=_aggregate(main,parentkeys,fields)
    files.append(_csv(folder/'full177_per_parent.csv',parents))
    macros=_aggregate(parents,('method','stage','panel_role','report_group'),fields)
    allgroups=_aggregate(parents,('method','stage','panel_role'),fields)
    for row in allgroups:row['report_group']='ALL'
    files.append(_csv(folder/'full177_parent_macro.csv',macros+allgroups))
    paired=_pairs(main,fields)
    files.append(_csv(folder/'paired_per_output.csv',paired))
    pp=_aggregate(paired,('candidate','reference','stage','panel_role','report_group','parent_id'),fields)
    files.append(_csv(folder/'paired_per_parent.csv',pp))
    files.append(_csv(folder/'paired_parent_macro.csv',_aggregate(pp,('candidate','reference','stage','panel_role','report_group'),fields)+
                      [dict(r,report_group='ALL') for r in _aggregate(pp,('candidate','reference','stage','panel_role'),fields)]))
    support=[{k:r.get(k) for k in ('cell_id','arm','label','stage','panel_role','parent_id','request_id','I_status',
              'ConflictCov_C_interior_samples','ConflictCov_C_interior_weight_fraction','C_query_weight_denominator')} for r in rows]
    files.append(_csv(folder/'I_support_and_denominators.csv',support))
    if mode=='final':
        index={(r['cell_id'],r['stage']):r for r in rows if r['main_panel']}
        costs=[]
        for task in tasks:
            raw=index[task['cell_id'],'RAW'];out=_metadata(task,'RAW_TO_POST')
            for stage in ('BASE_POST','CLEAN_FINAL_POST'):
                after=index[task['cell_id'],stage]
                for key in ('actual_free_faces','deleted_free_faces','free_total_surface_area',
                            'free_minimum_angle_lt5_fraction','hinge_diag_isolated_1face_edge_components'):
                    a,b=raw.get(key),after.get(key)
                    out[key+'__RAW']=a;out[key+'__'+stage]=b
                    out[key+'__'+stage+'_minus_RAW']=b-a if _number(a) and _number(b) else None
                a,b=raw.get('free_total_surface_area'),after.get('free_total_surface_area')
                out['free_area__'+stage+'_over_RAW']=b/a if _number(a) and _number(b) and a>0 else None
            costs.append(out)
        files.append(_csv(folder/'stage_retention_and_costs.csv',costs))
    return _seal(done,dict(status='COMPLETE',version=VERSION,stage_rows=len(rows),original_metric_fields=177,
                 new_score_calls=0,model_calls=0,mode=mode,
                 I_policy='Original I values and 2048-weight denominator retained; macro only original PASS; counts explicit.',
                 graph_policy='RAW index components are not visible-fragment labels; no automatic visual judgments.'),files,protected)


def _pages(mode, stages=None):
    baseline=_baseline()
    if mode=='calibration':
        tasks=_calibration_barrier()+[t for t in baseline if t['panel_role']=='calibration']
        stages=('RAW',)
    elif mode=='final':
        tasks=P.selected_tasks()+baseline
        stages=STAGES if stages is None else tuple(stages)
        if not set(stages)<=set(STAGES):raise ValueError('Unknown render stage')
    else:raise ValueError(mode)
    groups=defaultdict(list)
    for task in tasks:groups[task['request_id']].append(task)
    pages=[];mapping=[]
    for rid, group in sorted(groups.items()):
        if len(group)!=(6 if mode=='calibration' else 5):raise ValueError('Wrong column count')
        if mode=='calibration':
            group=sorted(group,key=lambda t:hashlib.sha256(('MOTIF_CALIBRATION_DISPLAY|'+rid+'|'+t['cell_id']).encode()).hexdigest())
        else:group=sorted(group,key=lambda t:ARMS.index(t['arm']))
        reference=next(t['q'] for t in group if t['arm']=='R')
        for stage in stages:
            cells=[]
            for index,task in enumerate(group):
                if (task['q']['gallery_bounds']!=reference['gallery_bounds'] or
                    task['q']['gallery_depth_bounds']!=reference['gallery_depth_bounds']):
                    raise ValueError('Original fixed camera or section coordinates differ')
                spec=_spec(task,stage)
                if not spec['mesh'].is_file():raise FileNotFoundError(spec['mesh'])
                display=chr(65+index) if mode=='calibration' else task['arm']
                cell=dict(display_label=display,cell_id=task['cell_id'],q=task['q'],path=str(spec['mesh']))
                cells.append(cell)
                if mode=='calibration':mapping.append(dict(request_id=rid,display_label=display,cell_id=task['cell_id'],
                     arm=task['arm'],label=task['label'],lambda_value=task.get('lambda_value')))
            pages.append(dict(page_id=stage+'__'+rid,stage=stage,mode=mode,reference_q=reference,cells=cells))
    if mode=='calibration':
        P.write_immutable(P.OUT/'calibration/hidden_method_mapping.json',dict(version=VERSION,records=mapping))
    return pages


def _render_page(page):
    _cpu()
    import numpy as np
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    from r2.pilots.seam_abc_v1 import eval_common as C
    q=page['reference_q'];stage=page['stage'];bounds=q['gallery_bounds'];depth=q['gallery_depth_bounds']
    dest=P.OUT/'gallery'/page['mode']/page['page_id'];png=dest/'comparison.png';receipt=dest/'comparison.json'
    identity=dict(version=VERSION,stage=stage,mode=page['mode'],request_id=q['request_id'],bounds=bounds,depth_bounds=depth,
                  columns=[dict(display_label=c['display_label'],mesh_sha256=P.sha(c['path']),
                                input_sha256=P.sha(c['q']['inference_npz'])) for c in page['cells']])
    saved=_cached(receipt)
    if saved:
        if saved['identity']!=identity:raise ValueError('Rendered identity differs')
        return saved
    if png.exists():raise RuntimeError('Unsealed render retained')
    dest.mkdir(parents=True,exist_ok=True);nc=len(page['cells']);plt=G.H._mpl()
    fig=plt.figure(figsize=(5.3*nc,13.5),dpi=110,facecolor='white')
    grid=fig.add_gridspec(4,nc,left=.025,right=.99,bottom=.055,top=.91,wspace=.15,hspace=.23)
    display=[]
    try:
        for col,cell in enumerate(page['cells']):
            k=int(cell['q']['K']);tri,roles=G.roles(C.arrays(cell['path']),k)
            inp=C.arrays(cell['q']['inference_npz']);original=inp['C_vertices'][inp['C_faces_original']]
            if not np.array_equal(tri[:k],original):raise ValueError('Rendered C differs')
            geometry=dict(C=tri[:k],F=tri[k:],C_original=original,tri=tri)
            facts=dict(display_label=cell['display_label'],stage=stage,K=k,N_requested=int(q['N']),N_output=len(tri),views=[])
            for row,azimuth in enumerate((-60,120)):
                raster,scope=G.role_raster(tri,roles,k,G.D.basis(azimuth,23),np.asarray(depth['center']),depth['span'])
                ax=fig.add_subplot(grid[row,col]);ax.imshow(raster);ax.set_axis_off()
                ax.set_title(cell['display_label']+f' | {stage} | az={azimuth}\nK={k} requestedN={q["N"]} outputN={len(tri)}',fontsize=8)
                facts['views'].append(dict(azimuth=azimuth,**scope))
                if scope['projected_vertices_outside_inherited_frame']:
                    ax.text(.01,.01,'Inherited-frame clipping; see receipt',transform=ax.transAxes,fontsize=7,color='#b34b27')
            for row,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[row,col]);G.H.draw_section(ax,geometry,np.flatnonzero(roles[k:]=='SEAM'),bounds,C.L,view,1)
                position=bounds['front_positions' if view=='front' else 'side_positions'][1]
                ax.set_title(cell['display_label']+' | '+stage+' | '+view+f'={position:.5f}',fontsize=8)
            display.append(facts)
        suffix='Method names and lambda hidden' if page['mode'].startswith('calibration') else 'R / G0 / G1 / G2 / G3'
        fig.suptitle(q['request_id']+' | '+stage+' | '+suffix+'\nBlue: C | Gray: generated free | Original fixed frame and finite sections',fontsize=11)
        fig.text(.5,.018,'Actual AI_PROXY review is separate. Section endpoints and RAW index islands do not certify visible fragments.',ha='center',fontsize=9)
        buf=io.BytesIO();fig.savefig(buf,format='png',dpi=110)
        with png.open('xb') as handle:handle.write(buf.getvalue())
    finally:plt.close(fig)
    return _seal(receipt,dict(status='PENDING_ACTUAL_VIEW',identity=identity,image_path=str(png),image_sha256=P.sha(png),
                 display=display,ratings_generated=0,model_calls=0,scoring_calls=0),[png],_book([G.__file__]))


def render(mode='final', workers=2, stages=None):
    _cpu()
    if workers not in (1,2):raise ValueError('Initial bounded CPU render workers must be 1 or 2')
    pages=_pages(mode,stages)
    main_count=len(pages)
    pages += [dict(page,mode=mode+'_single',page_id=page['page_id']+'__'+cell['display_label'],cells=[cell])
              for page in list(pages) if mode=='calibration' or page['stage'] in ('RAW','CLEAN_FINAL_POST')
              for cell in page['cells']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=list(pool.map(_render_page,pages))
    folder=P.OUT/'gallery'/mode
    name='all' if stages is None else '_'.join(stages)
    index=folder/('index_'+name+'.html')
    body=['<!doctype html><meta charset="utf-8"><title>Fixed views</title>']
    for page,result in zip(pages,receipts):
        relative=('../'+page['mode']+'/' if page['mode'].endswith('_single') else '')+page['page_id']+'/comparison.png'
        body.append('<p>'+html.escape(page['page_id'])+'</p><a href="'+html.escape(relative)+'"><img loading="lazy" width="100%" src="'+html.escape(relative)+'"></a>')
    _text(index,'\n'.join(body))
    return dict(status='COMPLETE_IMAGES_PENDING_ACTUAL_VIEW',pages=main_count,single_cell_pages=len(pages)-main_count,mode=mode,index=str(index),
                files=_book([index]+[r['image_path'] for r in receipts]),ratings_generated=0)


def _normalize_review(record):
    aliases={'body':'overall_shape','adoption':'C_use','support':'support_retention',
             'fragment_evidence':'visible_fragment_or_gap','folding':'layering_or_fold'}
    out=dict(record)
    for key,legacy in aliases.items():
        value=out.get(key,out.get(legacy))
        if value is None:raise ValueError('Review missing '+key)
        out[key]=str(value).upper()
    conversions={'body':{'INCOMPLETE':'IMPAIRED'},'adoption':{'REPLACED_OR_IGNORED':'IMPLAUSIBLE'},
                 'support':{'MISSING_OR_DAMAGED':'MISSING'},
                 'fragment_evidence':{'NO_OBVIOUS':'NOT_CLEAR'},'folding':{'NO_OBVIOUS':'NOT_CLEAR'}}
    for key,lookup in conversions.items():out[key]=lookup.get(out[key],out[key])
    allowed={'body':{'REASONABLE','IMPAIRED','UNREASONABLE','UNCERTAIN','UNKNOWN'},
             'adoption':{'PLAUSIBLE','IMPLAUSIBLE','UNCERTAIN','UNKNOWN'},
             'support':{'PRESENT','MISSING','IMPAIRED','UNCERTAIN','UNKNOWN'},
             'fragment_evidence':{'VISIBLE','NOT_CLEAR','UNCERTAIN','UNKNOWN','YES','NO'},
             'folding':{'VISIBLE','NOT_CLEAR','UNCERTAIN','UNKNOWN','YES','NO'}}
    for key,values in allowed.items():
        if out[key] not in values:raise ValueError('Unknown visual token '+key+': '+out[key])
    return out


def lock_lambda(review_path):
    """Select solely from actual hidden-label RAW reviews and original metrics."""
    _cpu()
    done=_lock_path();saved=_cached(done)
    if saved:return saved
    tasks=_calibration_barrier();summary=collect('calibration')
    mapping_path=P.OUT/'calibration/hidden_method_mapping.json'
    mapping=P.read(mapping_path)['records']
    if len(mapping)!=48:raise ValueError('Expected 48 hidden calibration cells, including R/G0/G1')
    raw=P.read(review_path);records=raw['records'] if isinstance(raw,dict) else raw
    if len(records)!=48:raise ValueError('All 48 calibration cells require actual review')
    by={(r['request_id'],r['display_label']):_normalize_review(r) for r in records}
    if len(by)!=48:raise ValueError('Duplicate calibration review')
    normalized={};protected=_book([review_path,mapping_path,P.OUT/'analysis/calibration/complete.json'])
    for m in mapping:
        review=by.get((m['request_id'],m['display_label']))
        if review is None:raise ValueError('Missing mapped review')
        image=P.resolve(review['source_image'])
        allowed_images=(P.OUT/'gallery/calibration'/('RAW__'+m['request_id'])/'comparison.png',
                        P.OUT/'gallery/calibration_single'/('RAW__'+m['request_id']+'__'+m['display_label'])/'comparison.png')
        if (review.get('actually_viewed') is not True or review.get('stage')!='RAW' or
            review.get('reviewer')!='AI_PROXY' or review.get('method_labels_hidden') is not True or
            image not in allowed_images or review['source_image_sha256']!=P.sha(image)):
            raise ValueError('Calibration review lacks matching actual-view evidence')
        if not str(review.get('observation',review.get('note',''))).strip():
            raise ValueError('Actual visual observation text is required')
        page_receipt=_cached(image.parent/'comparison.json')
        if page_receipt is None or page_receipt['image_sha256']!=review['source_image_sha256']:
            raise ValueError('Review source image is not a sealed fixed-view page')
        normalized[m['cell_id']]=review;protected.update(_book([image]))
    rankings=[]
    for value in LAMBDAS:
        selected=[t for t in tasks if t['arm']=='G3' and float(t['lambda_value'])==value]
        if len(selected)!=8:raise ValueError('Expected eight requests per lambda')
        rows=[];j=0;fragments=0
        for task in selected:
            row=_score_read(_spec(task,'RAW'));v=normalized[task['cell_id']]
            j+=int(v['body']=='REASONABLE' and v['adoption']=='PLAUSIBLE' and v['support']=='PRESENT')
            fragments+=int(v['fragment_evidence'] in ('VISIBLE','YES'))
            row=dict(row,parent_id=task['q']['parent_id']);rows.append(row)
        parent=_aggregate(rows,('parent_id',),('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C'))
        if len(parent)!=4 or sum(p['I_status']=='PASS' for p in parent)!=3:
            raise ValueError('Registered four-parent / three-PASS calibration support changed')
        cf_ff=sum(p['CF_illegal_affected_free_fraction']+p['FF_illegal_affected_free_fraction'] for p in parent)/4
        ivals=[p['ConflictCov_C'] for p in parent if p['I_status']=='PASS'];im=sum(ivals)/3
        if not all(_number(v) for v in (cf_ff,im)):raise ValueError('Nonfinite calibration ranking metric')
        rankings.append(dict(lambda_value=value,J=j,visible_fragment_requests=fragments,
                             parent_macro_CF_plus_FF=cf_ff,parent_macro_I_three_PASS=im,
                             sort_key=[-j,fragments,cf_ff,im,value],engineering_valid=True))
    rankings.sort(key=lambda r:tuple(r['sort_key']))
    rank_path=P.OUT/'calibration/lambda_ranking.json'
    P.write_immutable(rank_path,dict(version=VERSION,rankings=rankings,
                      rule='max J, min explicit fragments, min parent CF+FF, min original I on three PASS parents, min lambda',
                      calibration_only=True,D0_used_for_selection=False))
    protected.update(summary['files']);protected.update(_book([P.OUT/'tasks.json',rank_path]))
    return _seal(done,dict(status='LOCKED',version=VERSION,lambda_value=rankings[0]['lambda_value'],
                 calibration_requests=8,calibration_parents=4,I_PASS_parents=3,
                 main_RAW=184,auxiliary_RAW=16,total_RAW=200,total_D0=184,review_path=str(P.resolve(review_path)),
                 no_independent_human_blindness_claim=True),[rank_path],protected)


def jobs_remaining():
    """Read-only scheduler inventory; no task dispatch, regeneration or retry."""
    locked=_lock_path().is_file()
    tasks=P.load_tasks(resolved=locked)
    initial={t['cell_id'] for t in _calibration_tasks()}
    remaining=[];complete=[];blocked=[]
    for task in tasks:
        receipt=P.raw_receipt(task)
        if P.resolve(receipt).exists():
            P.verify_raw(task);complete.append(task['cell_id'])
        elif P.resolve(task['raw_path']).exists() or (P.resolve(task['output_dir'])/'failure.json').exists():
            blocked.append(dict(cell_id=task['cell_id'],reason='PARTIAL_OR_FAILED_NO_AUTOMATIC_REPLAY'))
        elif locked or task['cell_id'] in initial:
            remaining.append(task)
    return dict(status='BLOCKED' if blocked else 'READY',lambda_locked=locked,total_registered=200,
                complete_RAW=complete,ready_jobs=[P.job(t) for t in remaining],blocked=blocked,
                waiting_for_lambda=[t['cell_id'] for t in tasks if not locked and t['cell_id'] not in initial],
                D0_allowed=locked and len(complete)==200 and not blocked)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('score_raw','post','d0','collect','render','lock_lambda','jobs_remaining'))
    parser.add_argument('--cell-id');parser.add_argument('--indices',nargs='+',type=int)
    parser.add_argument('--mode',choices=('calibration','final'),default='final')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--stages',nargs='+',choices=STAGES)
    parser.add_argument('--review');parser.add_argument('--worker',default='d0_0')
    parser.add_argument('--estimated-peak-mib',type=float,default=3500)
    args=parser.parse_args()
    if args.action in ('score_raw','post'):
        choices=P.selected_tasks() if args.action=='post' else P.load_tasks(resolved=_lock_path().exists())
        task=next(t for t in choices if t['cell_id']==args.cell_id)
        result=score_raw(task) if args.action=='score_raw' else post(task)
    elif args.action=='d0':result=d0_batch(args.indices,args.worker,args.estimated_peak_mib)
    elif args.action=='collect':result=collect(args.mode)
    elif args.action=='render':result=render(args.mode,args.workers,args.stages)
    elif args.action=='lock_lambda':result=lock_lambda(args.review)
    else:result=jobs_remaining()
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','protected','row','ready_jobs')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
