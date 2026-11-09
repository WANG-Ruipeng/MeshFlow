"""Fixed D four-case, five-model panels from existing raw arrays only.

This module imports no torch/model/scorer. The root invokes it after all D
endpoint workers and their original independent scoring have completed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


SEED = 10401
ORDER = ('G0', 'B', 'J', 'R_B_EMA500', 'R_J_EMA500')
TITLES = ('G0 / 2500', 'B raw / 3000', 'J raw / 3000', 'B EMA / 3000', 'J EMA / 3000')
EXPECTED = dict(G0='6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392',
    B='f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b',
    J='b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _read(path, sources):
    path = Path(path)
    data = path.read_bytes()
    sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    return json.loads(data.decode('utf-8-sig'))


def _write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _validate_replay(saved, arm):
    model = arm[-1]
    if (saved.get('status') != 'PASS_EXACT_REPLAY' or saved.get('arm') != arm
            or saved.get('completed_updates') != 500 or saved.get('EMA_evaluation_allowed') is not True
            or saved.get('raw500_bitwise_reference_equal') is not True
            or saved.get('raw500_state_sha256') != EXPECTED[model]):
        raise ValueError('D figure blocked: raw/EMA replay gate missing or not exact: '+arm)
    counts = saved.get('counts', {})
    if any(counts.get(key) != expected for key, expected in (('optimizer_updates', 500),
            ('EMA_updates', 500), ('forward_returns', 4000), ('backward_returns', 4000), ('OT_returns', 4000))):
        raise ValueError('D figure blocked: incomplete replay accounting: '+arm)
    checkpoint = saved['checkpoints']['500']['EMA']
    if (checkpoint.get('arm') != arm or checkpoint.get('kind') != 'EMA'
            or checkpoint.get('completed_updates') != 500 or checkpoint.get('cumulative_updates') != 3000
            or not checkpoint.get('state_sha256')):
        raise ValueError('D figure blocked: EMA checkpoint identity incomplete: '+arm)
    return checkpoint


def _worker_gate(out, label, expected_state, sources):
    path = out/'A'/label/'results.json' if label in ('B', 'J') else out/'D_evaluation'/label/'results.json'
    saved = _read(path, sources)
    if (saved.get('status') != 'PASS' or saved.get('state_before') != expected_state
            or saved.get('state_after') != expected_state
            or saved.get('model_identity', {}).get('state_sha256') != expected_state):
        raise ValueError('Frozen completed worker identity differs: '+label)
    expected_count = 84 if label in ('B', 'J') else 4 if label == 'G0' else 20
    if saved.get('formal_completed') != expected_count:
        raise ValueError('Figure requires the complete predetermined worker: '+label)
    return saved


def _select(out, sources):
    protocol = _read(out/'registration/protocol.json', sources)
    seeds = protocol['generation_seeds']
    if seeds != [10401, 10402]:
        raise ValueError('Fixed figure seed is not the registered first noise')
    ema = {}
    for arm in ('R_B', 'R_J'):
        saved = _read(out/'stage_d'/arm/'replay.json', sources)
        ema[arm+'_EMA500'] = _validate_replay(saved, arm)
    chosen = {}
    for label in ORDER:
        expected = EXPECTED[label] if label in EXPECTED else ema[label]['state_sha256']
        worker = _worker_gate(out, label, expected, sources)
        phase = 'A_'+label if label in ('B', 'J') else 'D_'+label
        score = _read(out/'scores'/phase/'results.json', sources)
        if score.get('status') != 'PASS' or score.get('phase') != phase:
            raise ValueError('Figure requires completed original scoring: '+phase)
        rows = [row for row in score['rows'] if row.get('precision') == 'bf16' and row.get('nfe') == 50]
        expected_grid = {(p, j, seed) for p in (0, 1)
                         for j in ((0, 4) if label == 'G0' else range(5))
                         for seed in ([SEED] if label == 'G0' else seeds)}
        grid = {(row['parent'], row['patch'], row['seed']) for row in rows}
        if len(rows) != len(expected_grid) or grid != expected_grid or any(row.get('model') != label for row in rows):
            raise ValueError('Score grid differs: '+label)
        worker_rows = {(row['parent'], row['patch'], row['seed'], row['precision'], row['nfe']): row for row in worker['rows']}
        for row in rows:
            if row['seed'] != SEED or row['patch'] not in (0, 4):
                continue
            key = (row['parent'], row['patch'], SEED, 'bf16', 50)
            original = worker_rows[key]
            if row['path'] != original['path'] or row['sha256'] != original['sha256']:
                raise ValueError('Score raw identity differs from worker record')
            chosen[label, row['parent'], row['patch']] = row
    if len(chosen) != 20:
        raise ValueError('Fixed four cases require all five models')
    return chosen, ema


def _load_raw(out, row, sources):
    path = (out/row['path']).resolve()
    if not path.is_relative_to(out.resolve()) or not path.is_file() or _sha(path) != row['sha256']:
        raise ValueError('Raw path/hash changed')
    with np.load(path, allow_pickle=False) as z:
        raw = z['raw'].copy()
        context = z['C'].copy()
    if (raw.dtype != np.float32 or raw.shape not in ((112, 9), (112, 3, 3))
            or context.dtype != np.float32 or context.size != 108
            or not np.isfinite(raw).all() or not np.isfinite(context).all()):
        raise ValueError('Fixed FP32 N112/K12 raw geometry differs')
    raw = raw.reshape(112, 3, 3)
    context = context.reshape(12, 3, 3)
    if raw[:12].tobytes() != context.tobytes():
        raise ValueError('Known C differs in saved endpoint')
    raw.setflags(write=False)
    sources.append(dict(path=str(path), sha256=row['sha256'], bytes=path.stat().st_size,
                        role='unchanged raw archive; only read'))
    return raw, context


def _joint_bounds(arrays):
    points = np.concatenate([raw.reshape(-1, 3) for raw in arrays]).astype(np.float64)[:, [0, 2, 1]]
    center = (points.min(0)+points.max(0))*.5
    radius = float(np.ptp(points, axis=0).max())*.54
    if not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Joint plot bounds are not finite and nonzero')
    return center, radius


def run(out):
    out = Path(out).resolve()
    sources = []
    chosen, ema = _select(out, sources)
    folder = out/'figures'
    manifest = folder/'D_manifest.json'
    paths = {(p, j): folder/f'D_P{p}_patch{j}_seed{SEED}.png' for p in (0, 1) for j in (0, 4)}
    if manifest.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError('Never overwrite fixed D figures or their manifest')
    panels = {}
    for parent in (0, 1):
        for patch in (0, 4):
            entries = []
            context = None
            for label in ORDER:
                row = chosen[label, parent, patch]
                raw, C = _load_raw(out, row, sources)
                if context is not None and context.tobytes() != C.tobytes():
                    raise ValueError('Models received different known coordinates')
                context = C
                entries.append((label, row, raw, hashlib.sha256(raw.tobytes()).hexdigest()))
            panels[parent, patch] = entries
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(status='RUNNING', schema='jedge5_D_fixed_figures_v1', records=[], sources=sources,
        seed=SEED, fixed_cases=[[p, j] for p in (0, 1) for j in (0, 4)], model_order=list(ORDER),
        view=dict(axis_display_order=['X', 'Z', 'Y'], elev=24, azim=-54,
                  bounds='joint bounds of all five raw outputs, reused for both rows within each panel'),
        columns=list(TITLES), rows=['full: orange C and blue free', 'free only: blue free'],
        raw_replay_gate='Both R_B/R_J PASS_EXACT_REPLAY and original state hashes, counts and corresponding EMA worker identity required',
        EMA_state_identities={label: record['state_sha256'] for label, record in ema.items()},
        interpretation='Four fixed diagnostic cases, not representative outputs; no success-based selection.',
        counts=dict(figures=0, raw_archives_read=20, new_model_calls=0, new_geometry_scores=0,
                    new_OT_calls=0, new_autograd_calls=0, repaired_vertices=0),
        visual_review='TO_BE_PERFORMED_BY_ROOT')
    tick = time.perf_counter()
    _write(manifest, result)
    try:
        for (parent, patch), entries in panels.items():
            panel_start = time.perf_counter()
            center, radius = _joint_bounds([entry[2] for entry in entries])
            fig = plt.figure(figsize=(18, 7), layout='constrained')
            try:
                for column, (label, row, raw, _) in enumerate(entries):
                    triangles = raw[:, :, [0, 2, 1]]  # display-only copy; archived coordinates never changed
                    for part in range(2):
                        ax = fig.add_subplot(2, 5, part*5+column+1, projection='3d')
                        ax.add_collection3d(Poly3DCollection(triangles[12:], facecolors='#91a9c2',
                            edgecolors='#30445a', linewidths=.35, alpha=.86))
                        if part == 0:
                            ax.add_collection3d(Poly3DCollection(triangles[:12], facecolors='#f6aa3d',
                                edgecolors='#a45309', linewidths=.7, alpha=.95))
                        ax.set_xlim(center[0]-radius, center[0]+radius)
                        ax.set_ylim(center[1]-radius, center[1]+radius)
                        ax.set_zlim(center[2]-radius, center[2]+radius)
                        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=24, azim=-54); ax.set_axis_off()
                        ax.set_title(TITLES[column]+(' | full' if part == 0 else ' | free only'), fontsize=11)
                fig.suptitle(f'P{parent} patch{patch} / fixed seed {SEED} (not representative)\n'
                    'BF16 Euler50; orange known C, blue free; shared bounds/view; no postprocessing.', fontsize=13)
                path = paths[parent, patch]
                fig.savefig(path, dpi=150)
            finally:
                plt.close(fig)
            if any(hashlib.sha256(raw.tobytes()).hexdigest() != before for _, _, raw, before in entries):
                raise AssertionError('Rendering altered loaded source coordinates')
            result['records'].append(dict(parent=parent, patch=patch, seed=SEED,
                path=str(path.relative_to(out)), sha256=_sha(path), bytes=path.stat().st_size,
                model_order=list(ORDER), center=center.tolist(), radius=radius,
                input_array_bytes_unchanged=True, render_seconds=time.perf_counter()-panel_start,
                raw_sources=[dict(model=label, path=row['path'], sha256=row['sha256']) for label, row, _, _ in entries]))
            result['counts']['figures'] += 1
            _write(manifest, result)
        result['status'] = 'PASS'
    except BaseException as error:
        result.update(status='STOPPED', error=repr(error))
        raise
    finally:
        result['seconds'] = time.perf_counter()-tick
        _write(manifest, result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args=parser.parse_args(argv)
    result=run(args.out)
    print(json.dumps(dict(status=result['status'], counts=result['counts'])))


if __name__ == '__main__':
    main()
