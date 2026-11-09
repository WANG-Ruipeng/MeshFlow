"""CPU contact sheets of completed, saved Phase 2 raw triangle soups.

Run only after Phase 2 finishes:
  python -m experiments.mf_h1_local0.render_results --output <run-directory>

No model import, mesh repair, vertex displacement, sampling or training occurs.
All images use the baseline camera (elev=18, azim=-60, vertical_axis='y').
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

KINDS = ('single_edge', 'single_face', 'mixed_2edges_1face')
SEEDS = (4101, 4102)
ARMS = ('Q_center', 'G_closest')
MODES = ('correct', 'cyclic_mismatch')
CAMERA = dict(elev=18, azim=-60, vertical_axis='y')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def triangles(tokens):
    values = np.asarray(tokens)
    if values.size % 9:
        raise ValueError('Raw triangle-soup array has an invalid coordinate count')
    return values.reshape(-1, 3, 3)


def scalar(arrays, name):
    return arrays[name].item() if name in arrays else None


def preflight(output):
    execution = json.loads((output/'phase2_execution.json').read_text())
    config = json.loads((output/'phase2_resolved_config.json').read_text())
    if execution.get('status') == 'RUNNING' or execution.get('optimizer_updates') != 400:
        raise RuntimeError('Phase 2 has not finished its authorized 400 updates; rendering is deferred')
    if config.get('seed') != 10 or config.get('optimizer_updates_per_arm') != 200:
        raise ValueError('Unexpected training protocol; refuse an inaccurate 200-update title')
    if execution.get('complete_rollouts') != 98:
        raise RuntimeError('The two base and 96 adapted complete raw rollouts must all exist first')
    for arm in ARMS:
        if execution.get('arms', {}).get(arm, {}).get('updates') != 200:
            raise ValueError(f'{arm} does not have exactly 200 completed updates')
    manifest = json.loads((output/'prompt_manifest.json').read_text())
    entries = {(item['parent_index'], item['kind']): item for item in manifest['records']}
    expected = {(i, kind) for i in range(4) for kind in KINDS}
    if set(entries) != expected:
        raise ValueError('Expected the 12 fixed diagnostic prompt records')
    required = [output/'raw'/f'base_bf16_seed{seed}.npz' for seed in SEEDS]
    required += [output/'raw'/f'{arm}_parent{i}_{kind}_seed{seed}_{mode}.npz'
                 for arm in ARMS for i in range(4) for kind in KINDS for seed in SEEDS for mode in MODES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Incomplete saved raw outputs: '+', '.join(missing))
    inputs = []
    fixed = {}
    for key, record in entries.items():
        path = Path(record['path'])
        actual_hash = sha256(path)
        if actual_hash != record['sha256']:
            raise ValueError(f'Fixed input hash changed: {path}')
        fixed[key] = load_arrays(path)
        inputs.append(dict(path=str(path), sha256=actual_hash, role='fixed_target_and_prompt'))
    base = {}
    for seed in SEEDS:
        path = output/'raw'/f'base_bf16_seed{seed}.npz'
        base[seed] = load_arrays(path)
        if len(triangles(base[seed]['tokens'])) != 512:
            raise ValueError(f'Expected 512 raw base face slots: {path}')
        inputs.append(dict(path=str(path),sha256=sha256(path),role='base_raw'))
    generated = {}
    for arm in ARMS:
        for i in range(4):
            for kind in KINDS:
                for seed in SEEDS:
                    for mode in MODES:
                        key = arm, i, kind, seed, mode
                        path = output/'raw'/f'{arm}_parent{i}_{kind}_seed{seed}_{mode}.npz'
                        data = load_arrays(path)
                        actual_i = i if mode == 'correct' else (i+1)%4
                        actual = fixed[(actual_i, kind)]
                        if len(triangles(data['tokens'])) != 512:
                            raise ValueError(f'Expected 512 raw adapted face slots: {path}')
                        if scalar(data,'seed') != seed or scalar(data,'parent_uid') != entries[(i,kind)]['parent_uid']:
                            raise ValueError(f'Raw identity metadata mismatch: {path}')
                        if scalar(data,'condition_parent_uid') != entries[(actual_i,kind)]['parent_uid']:
                            raise ValueError(f'Raw condition identity mismatch: {path}')
                        if not np.array_equal(data['prompt_vertices'],actual['vertices']) or not np.array_equal(data['prompt_arity'],actual['arity']):
                            raise ValueError(f'Actual saved prompt differs from fixed manifest: {path}')
                        if not np.array_equal(data['initial_noise'],base[seed]['initial_noise']):
                            raise ValueError(f'Noise differs from paired base generation: {path}')
                        generated[key] = data
                        inputs.append(dict(path=str(path),sha256=sha256(path),role='adapted_raw'))
    metrics = {}
    with (output/'per_output.csv').open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            key = row['arm'],int(row['parent_index']),row['kind'],int(row['noise_seed']),row['condition_mode']
            metrics[key] = row
    return execution, entries, fixed, base, generated, metrics, inputs


def common_bounds(arrays):
    points = np.concatenate([np.asarray(value,dtype=np.float64).reshape(-1,3) for value in arrays])
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return np.array([[-1.,1.]]*3)
    lower, upper = points.min(axis=0),points.max(axis=0)
    center = (lower+upper)/2
    radius = max(float((upper-lower).max())*.56,1e-6)
    return np.stack([center-radius,center+radius],axis=1)


def metric_label(row):
    if row is None:
        return ''
    try:
        guide=float(row['guide_actual_rms_bbox_pct'])
        sliver=100*float(row['min_angle_lt5_fraction'])
        return f'Guide RMS {guide:.2f}% bbox | min angle <5 deg: {sliver:.1f}%'
    except (TypeError,ValueError,KeyError):
        return 'Metric unavailable; inspect numerical-failure records'


def draw_mesh(ax, tokens, vertices, arity, bounds, title, label):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tri=triangles(tokens)
    finite=np.isfinite(tri).all(axis=(1,2))
    # NaN/Inf triangles have no drawable coordinates; never repair them. Their
    # count remains visible and is retained in the render manifest.
    if finite.any():
        poly=Poly3DCollection(tri[finite],facecolor='#91a9c2',edgecolor='#3a526c',
                              linewidth=.17,alpha=1,zorder=1)
        ax.add_collection3d(poly)
    for v,a in zip(vertices,arity):
        a=int(a)
        if a not in (2,3) or not np.isfinite(v[:a]).all():
            raise ValueError('Invalid fixed primitive for display')
        if a==3:
            patch=Poly3DCollection([v[:3]],facecolor=(1,.77,.10,.30),
                                  edgecolor='none',zorder=8)
            ax.add_collection3d(patch)
            line=np.concatenate([v[:3],v[:1]])
        else:
            line=v[:2]
        # Guides are explicitly overlays, displayed above the raw mesh for
        # visibility while retaining their exact saved coordinates.
        ax.plot(line[:,0],line[:,1],line[:,2],color='#493408',linewidth=3.3,zorder=9)
        ax.plot(line[:,0],line[:,1],line[:,2],color='#ffca28',linewidth=2.2,zorder=10)
        ax.scatter(v[:a,0],v[:a,1],v[:a,2],s=11,c='#ffdc55',edgecolors='#493408',linewidths=.35,zorder=11)
    ax.set_xlim(*bounds[0]);ax.set_ylim(*bounds[1]);ax.set_zlim(*bounds[2])
    ax.set_box_aspect((1,1,1));ax.view_init(**CAMERA);ax.set_axis_off()
    ax.set_title(title,fontsize=10,pad=-4,color='#18324d')
    ax.text2D(.02,.025,label,transform=ax.transAxes,fontsize=7,color='#243d54')
    if not finite.all():
        ax.text2D(.02,.09,f'NUMERICAL FAILURE: {(~finite).sum()} nonfinite raw faces',
                  transform=ax.transAxes,fontsize=8,color='#a40000',weight='bold')
    return dict(raw_faces=len(tri),nonfinite_faces_not_drawable=int((~finite).sum()))


def render(output, figure_subdir="phase2_contacts"):
    if not figure_subdir.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Figure subdirectory must be a simple alphanumeric name")
    output=Path(output).resolve()
    execution,entries,fixed,base,generated,metrics,inputs=preflight(output)
    figure_dir=output/'figures'/figure_subdir
    if figure_dir.exists():
        raise FileExistsError(f'Refusing to overwrite previous contact sheets: {figure_dir}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.facecolor':'white',
                         'savefig.facecolor':'white','axes.unicode_minus':False})
    figure_dir.mkdir(parents=True,exist_ok=False)
    sheets=[]
    for kind in KINDS:
        for seed in SEEDS:
            row_bounds={}
            for i in range(4):
                arrays=[fixed[(i,kind)]['target'],base[seed]['tokens'],fixed[(i,kind)]['vertices'],fixed[((i+1)%4,kind)]['vertices']]
                arrays += [generated[(arm,i,kind,seed,mode)]['tokens'] for arm in ARMS for mode in MODES]
                row_bounds[i]=common_bounds(arrays)
            for layout in ('target_base_correct','correct_vs_mismatch'):
                fig=plt.figure(figsize=(14,14.8))
                fig.subplots_adjust(left=.015,right=.995,bottom=.068,top=.905,wspace=.01,hspace=.035)
                fig.suptitle(f'Phase 2 wiring check | 200 updates per arm | adapter seed 10\n{kind} | evaluation noise {seed} | raw geometry',
                             fontsize=17,color='#17344f',weight='bold',y=.978)
                subtitle=('Target / unconditioned base / Q_center / G_closest; yellow = original guide'
                          if layout=='target_base_correct' else
                          'Q correct / Q cyclic mismatch / G correct / G cyclic mismatch; yellow = actual input guide')
                fig.text(.5,.932,subtitle,ha='center',fontsize=11,color='#405669')
                panels=[]
                for i in range(4):
                    record=entries[(i,kind)];original=fixed[(i,kind)]
                    if layout=='target_base_correct':
                        cells=[('Target',original['target'],original,None),
                               ('Base (no geometry condition)',base[seed]['tokens'],original,metrics.get(('base',i,kind,seed,'none')))]
                        cells += [(arm+' | correct',generated[(arm,i,kind,seed,'correct')]['tokens'],original,
                                   metrics.get((arm,i,kind,seed,'correct'))) for arm in ARMS]
                    else:
                        cells=[]
                        for arm in ARMS:
                            for mode in MODES:
                                guide=fixed[(i if mode=='correct' else (i+1)%4,kind)]
                                name=arm+(' | correct' if mode=='correct' else ' | cyclic mismatch')
                                cells.append((name,generated[(arm,i,kind,seed,mode)]['tokens'],guide,
                                              metrics.get((arm,i,kind,seed,mode))))
                    for column,(name,tokens,guide,row) in enumerate(cells):
                        ax=fig.add_subplot(4,4,i*4+column+1,projection='3d',computed_zorder=False)
                        title=name if i==0 else ''
                        label=metric_label(row) if row is not None else f'{len(triangles(tokens))} '+('original target faces' if layout=='target_base_correct' and column==0 else 'raw faces; metric unavailable')
                        info=draw_mesh(ax,tokens,guide['vertices'],guide['arity'],row_bounds[i],title,label)
                        if column==0:
                            ax.text2D(.02,.91,f'P{i+1}: {record["parent_uid"][:10]}...',transform=ax.transAxes,fontsize=8,color='#18324d')
                        if layout=='correct_vs_mismatch' and column%2==1:
                            ax.text2D(.02,.91,f'Actual guide from P{(i+1)%4+1}',transform=ax.transAxes,fontsize=8,color='#725507')
                        panels.append(dict(parent_index=i,column=column,label=name,bounds_xyz=row_bounds[i].tolist(),**info))
                foot=('Raw saved triangle soups, no repair or postprocessing. Yellow primitives are coordinate-preserving overlays.\n'
                      'Same y-up camera and full-object limits across each row, including both condition modes; no outlier clipping.\n'
                      'Base draws are reused across parent rows, not counted as independent samples. Phase 2 supports wiring/response checks only.')
                fig.text(.5,.021,foot,ha='center',va='bottom',fontsize=8.5,color='#506477',linespacing=1.5)
                path=figure_dir/f'{kind}_seed{seed}_{layout}.png'
                fig.savefig(path,dpi=155)
                plt.close(fig)
                sheets.append(dict(path=str(path),sha256=sha256(path),layout=layout,kind=kind,noise_seed=seed,panels=panels))
                print(json.dumps(dict(stage='render',file=str(path),completed=len(sheets))),flush=True)
    changed=[item['path'] for item in inputs if sha256(item['path'])!=item['sha256']]
    report=dict(status='PASS' if not changed else 'FAIL_INPUT_CHANGED',source_phase2_status=execution['status'],
                renderer_sha256=sha256(__file__),camera=CAMERA,device='CPU matplotlib Agg',mesh_modifications=0,
                actual_model_forwards=0,optimizer_updates=0,inputs=inputs,inputs_changed=changed,
                contact_sheets=sheets,output_png_count=len(sheets),base_independent_draw_count=2,
                nonfinite_faces_policy='Unrenderable coordinates explicitly marked; original raw data are never altered',
                bounds_policy='Each parent/type/noise row uses target, base, Q/G correct and mismatch, and both prompt sets; exact full range plus margin')
    with (figure_dir/'render_manifest.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    if changed:
        raise RuntimeError('An input changed during rendering; see render_manifest.json')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Completed Phase 2 run directory')
    parser.add_argument('--figure-subdir',default='phase2_contacts',help='New directory name below figures; previous images are never overwritten')
    args=parser.parse_args();report=render(args.output,args.figure_subdir)
    print(json.dumps(dict(status=report['status'],png_count=report['output_png_count'],model_forwards=0)))


if __name__=='__main__':main()
