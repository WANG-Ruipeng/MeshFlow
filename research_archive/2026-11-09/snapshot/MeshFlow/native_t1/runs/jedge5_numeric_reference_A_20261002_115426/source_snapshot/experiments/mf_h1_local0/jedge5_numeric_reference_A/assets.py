"""Window A frozen assets. Only exact archived arrays, no noise draws or OT."""
from native_t1 import runtime as _runtime  # sets CUBLAS before a possible later CUDA load
from pathlib import Path
import json
import os
import sys
import numpy as np
import torch
from native_t1.artifacts import file_sha256, state_sha256, array_hash
from native_t1.geometry_checkpoint import load_geometry_checkpoint

ROOT = Path(__file__).resolve().parents[3]
EVALUATION = ROOT/'native_t1/runs/edge_factorial_20261001_123000/registration/evaluation_inputs.npz'
EVALUATION_SHA = '2154881d77b946a2a18959503930fd1d405f88f0d1f56d64a8f0d86710ed01b4'
CASE_SPECS = ((0,0,10401),(1,0,10401),(0,4,10401),(0,4,10402))
ASSETS = {
 'B': dict(path=ROOT/'native_t1/checkpoints/ablations/fm_cumulative3000.pt', recipe='fm',
           state='f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b',
           file='0f22ce498ca2c27dc74b43890fb8aad20f0f3e7153aa74ee6d3c14ea445a2660'),
 'J': dict(path=ROOT/'native_t1/checkpoints/jedge5_cumulative3000.pt', recipe='edge5',
           state='b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3',
           file='cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9')}


def _json(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, default=_json, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def case_id(parent, patch, seed):
    return f'P{parent}_patch{patch}_seed{seed}'


def load_model(label, device='cpu'):
    spec = ASSETS[label]
    if file_sha256(spec['path']) != spec['file']:
        raise ValueError(f'{label} checkpoint file identity changed')
    with torch.random.fork_rng(devices=[]):
        model, audit = load_geometry_checkpoint(spec['path'], device=device, expected_recipe=spec['recipe'])
    if audit['state_sha256'] != spec['state'] or audit['cumulative_updates'] != 3000:
        raise ValueError(f'{label} generator/cumulative identity mismatch')
    if not audit['strict'] or audit['missing_keys'] or audit['unexpected_keys']:
        raise ValueError('Strict checkpoint loading contract failed')
    return model, audit


def discover_archive():
    """Select latest matching report using content identities, not a timestamp literal."""
    candidates = []; rejected = []
    for report in (ROOT/'native_t1/runs').rglob('JEDGE5_DIAGNOSIS_REPORT.md'):
        folder = report.parent
        try:
            protocol = read_json(folder/'registration/protocol.json')
            if (protocol.get('schema') != 'jedge5_diagnosis_v1'
                or protocol.get('generation_seeds') != [10401,10402]
                or protocol.get('primary_order') != ['A','B','C','D']):
                raise ValueError('Protocol/seeds/stages differ')
            identities = {r['label']:r for r in protocol['checkpoint_identities']}
            for label, spec in ASSETS.items():
                if identities[label]['state_sha256'] != spec['state']:
                    raise ValueError(label+' identity differs')
                replay = read_json(folder/'stage_d'/('R_'+label)/'replay.json')
                if (replay.get('status') != 'PASS_EXACT_REPLAY'
                    or replay.get('raw500_state_sha256') != spec['state']
                    or replay.get('raw500_bitwise_reference_equal') is not True):
                    raise ValueError(label+' replay identity differs')
                a = read_json(folder/'A'/label/'results.json')
                if a.get('status') != 'PASS' or a.get('state_after') != spec['state']:
                    raise ValueError(label+' A phase incomplete')
            if not (folder/'B').is_dir() or not (folder/'stage_c').is_dir():
                raise ValueError('Missing B/C archive stage')
            candidates.append(dict(path=str(folder),report=str(report),mtime_ns=report.stat().st_mtime_ns))
        except (ValueError, KeyError, OSError) as exc:
            rejected.append(dict(path=str(folder),reason=str(exc)))
    if not candidates:
        raise FileNotFoundError('No ABCD archive with the prescribed B/J replay and seeds; '+str(rejected))
    candidates.sort(key=lambda row:(row['mtime_ns'],row['path']), reverse=True)
    return Path(candidates[0]['path']), dict(selection='latest report mtime among identity-matching ABCD archives',
                                            candidates=candidates,rejected=rejected)


def load_registered_cases(out):
    """Return the four registered input records; every continuous array is archived."""
    out = Path(out); p = read_json(out/'registration/protocol.json')
    source = out/'registration/inputs.npz'
    if file_sha256(source) != p['registered_inputs_sha256']:
        raise ValueError('Registered input archive changed')
    rows = []
    with np.load(source, allow_pickle=False) as z:
        for meta in p['cases']:
            row = dict(meta)
            for name in ('noise','C','known','valid','y','known_slots','free_slots','initial_full','initial_free'):
                row[name] = z[meta['id']+'__'+name].copy()
                if array_hash(row[name]) != meta['array_hashes'][name]:
                    raise ValueError('Registered array changed: '+meta['id']+'/'+name)
            rows.append(row)
    return rows


def load_case(out, identifier):
    rows = load_registered_cases(out)
    if isinstance(identifier, int): return rows[identifier]
    return next(row for row in rows if row['id'] == identifier)


def noise_bank(out):
    rows = load_registered_cases(out)
    return {row['seed']:row['noise'].copy() for row in rows}


def actually_imported_sources():
    """Hash only local imported source modules, excluding either window's output/code."""
    own = Path(__file__).resolve().parent
    paths = set()
    for module in tuple(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if not file: continue
        path = Path(file).resolve()
        if not path.is_file() or path.suffix != '.py' or ROOT not in path.parents or own in path.parents: continue
        # No other window's runtime/output files become immutable by association.
        rel = str(path.relative_to(ROOT)).replace('\\','/')
        if 'jedge5_local' in rel or 'jedge5_numeric_reference_A_' in rel: continue
        paths.add(path)
    return [dict(path=str(p),sha256=file_sha256(p),kind='read_source') for p in sorted(paths)]


def verify_protected(out):
    """Check only registered read assets; unrelated window outputs are ignored."""
    registration = Path(out)/'registration/protected_assets.json'
    payload = read_json(registration); rows=[]
    for record in payload['files']:
        path=Path(record['path']); actual=file_sha256(path) if path.is_file() else None
        rows.append(dict(path=str(path),expected=record['sha256'],actual=actual,
                         unchanged=actual==record['sha256'],kind=record['kind']))
    return dict(status='PASS' if all(r['unchanged'] for r in rows) else 'FAIL',
                checked=len(rows),changed=[r for r in rows if not r['unchanged']],
                scope=payload['scope'])

