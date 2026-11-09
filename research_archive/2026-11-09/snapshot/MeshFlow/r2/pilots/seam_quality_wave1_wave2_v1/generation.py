"""Five fixed exports, unchanged LONG layouts, per-wave all-RAW D0 barrier.

The established rollout and D0 routines execute in private namespaces. This
module changes model identity and output locations, not their arithmetic.
Nothing starts on import. The parent supplies a frozen per-wave GO receipt.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import json
from pathlib import Path
from r2.pilots.seam_abc_v1 import generate as G
from .evaluation import C, ROOT, OUT, ARMS, WAVES, wave_for, private_module, requests

VERSION = 'seam_quality_wave12_exact_cached_LONG_generation_v1'
PAYLOAD_ARMS = {'CLEAN_BULK_FM_S1000': 'CLEAN_BULK_FM',
                'SURGICAL_LATE4_S1000': 'SURGICAL_LATE4'}
STEPS = {'PLAIN_S1000': (1000, 14000), 'PLAIN_S2000': (2000, 15000),
         'CLEAN_BULK_FM_S1000': (1000, 14000), 'SURGICAL_LATE4_S1000': (1000, 14000)}


def registration_path(wave):
    if wave not in WAVES:
        raise ValueError('Unknown wave')
    return OUT / ('generation_' + wave + '_registration.json')


def prepare(wave):
    path = registration_path(wave)
    if path.exists():
        return contract(wave)
    protocol = C.read(OUT / 'protocol.json')
    # Stage-0 cleanup and scientific-contract release are parent responsibilities;
    # prepare never grants permission to dispatch a model.
    protected = {str(OUT / 'protocol.json'): C.sha(OUT / 'protocol.json')}
    qq = requests()
    layouts = protocol['evaluation']['layouts']
    for q in qq:
        for key in ('inference_npz', 'metadata_npz', 'noise_npz'):
            protected[q[key]] = C.sha(q[key])
        desc = q['query_descriptor']
        if C.sha(desc['path']) != desc['sha256']:
            raise ValueError('Fixed evaluation queries changed')
        protected[desc['path']] = desc['sha256']
        item = layouts[q['condition_id']]
        for key in ('path', 'source_path'):
            protected[item[key]] = C.sha(item[key])
    models = {}
    for arm in WAVES[wave]:
        fp = OUT / 'inference' / (PAYLOAD_ARMS.get(arm, arm) + '_generator.pt')
        meta_path = fp.with_suffix('.json')
        meta = C.read(meta_path)
        if C.sha(fp) != meta['sha256']:
            raise ValueError('Generator export identity changed: ' + arm)
        models[arm] = dict(path=str(fp), sha256=meta['sha256'],
                           metadata_path=str(meta_path), metadata_sha256=C.sha(meta_path))
        protected[str(fp)] = meta['sha256']
        protected[str(meta_path)] = C.sha(meta_path)
    old = C.read(ROOT / 'experiments/r2_seam_abc_v1/registration.json')
    d0 = old['d0_checkpoint']
    protected[d0['path']] = d0['sha256']
    n = 28 * len(WAVES[wave])
    reg = dict(schema=VERSION, status='FROZEN', wave=wave, arms=list(WAVES[wave]),
        requests=qq, layouts=layouts, models=models, d0_checkpoint=d0,
        budget=dict(training_updates=0, head_forwards=0, RAW_outputs=n,
                    generator_forwards=n*50, D0_forwards=n, GEO_calls=n),
        resources=dict(generator_estimate_mib=2600, D0_estimate_mib=3500),
        barrier='Every registered RAW in this wave must be complete before any D0 in this wave',
        interpretation='Fixed cached LONG C-only layout; no source-occupancy labels and no head forward',
        protected_files=protected)
    G.verify_files(protected)
    C.write(path, reg)
    return reg


def contract(wave, require_go=False):
    path = registration_path(wave)
    reg = C.read(path)
    if reg['status'] != 'FROZEN' or reg['arms'] != list(WAVES[wave]) or reg['wave'] != wave:
        raise ValueError('Generation contract differs')
    G.verify_files(reg['protected_files'])
    if require_go:
        go = C.read(OUT / ('generation_' + wave + '_dispatch_go.json'))
        if go.get('authorized') is not True or go['registration_sha256'] != C.sha(path):
            raise ValueError('Parent dispatch release missing or changed')
        G.verify_files(go['code_hashes'])
    return reg


def load_generator(_root, arm):
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.models.native import build_model
    reg = C.read(registration_path(wave_for(arm)))
    item = reg['models'][arm]
    if C.sha(item['path']) != item['sha256']:
        raise ValueError('Model file changed')
    payload = torch.load(C.resolve(item['path']), map_location='cpu', weights_only=True, mmap=True)
    if payload['arm'] not in (arm, PAYLOAD_ARMS.get(arm, arm)) or payload['EMA'] is not False:
        raise ValueError('Wrong live-generator export identity')
    if arm in STEPS:
        local, global_step = STEPS[arm]
        if payload['local_step'] != local or payload['global_step'] != global_step:
            raise ValueError('Unregistered training endpoint')
    else:
        # This model is a frozen FP32 chord midpoint, never a training step.
        meta = C.read(item['metadata_path'])
        if payload.get('interpolation_alpha', payload.get('alpha', meta.get('alpha'))) != .5:
            raise ValueError('WISE requires preregistered alpha=0.5')
    if any(k in payload for k in ('optimizer', 'rng', 'head_optimizer')):
        raise ValueError('Sampling requires independent generator-only export')
    state = payload['generator_state']
    if len(state) != 194 or any(v.dtype != torch.float32 for v in state.values()):
        raise ValueError('Expected original 194 FP32 generator tensors')
    if state_hash(state) != payload['generator_state_sha256']:
        raise ValueError('Generator tensor hash differs')
    model = build_model(payload['model_config'], readout_mode='none', trainable=False)
    model.load_state_dict(state, strict=True)
    model.to('cuda').set_trainable(False).eval()
    return model, payload, dict(path=item['path'], sha256=item['sha256'],
        generator_state_sha256=payload['generator_state_sha256'], arm=arm)


def all_raw(wave, reg):
    files = {}
    for arm in WAVES[wave]:
        aggregate = G.checked_receipt(OUT / 'evaluation' / arm / 'raw_complete.json')
        if aggregate['outputs'] != 28 or aggregate['generator_forwards'] != 1400:
            raise ValueError('Incomplete RAW arm budget')
        for q in reg['requests']:
            path = OUT / 'evaluation' / arm / q['request_id'] / 'raw/receipt.json'
            r = G.checked_receipt(path)
            if r['arm'] != arm or r['request_id'] != q['request_id'] or r['generator_forwards'] != 50:
                raise ValueError('RAW identity or call budget differs')
            files[str(path)] = C.sha(path)
    if len(files) != 28 * len(WAVES[wave]):
        raise ValueError('Per-wave all-RAW barrier failed')
    return files


def adapter(wave):
    mod = private_module(ROOT / 'r2/pilots/owner_layout_v1/generation.py', '_quality_' + wave + '_runtime')
    api = private_module(ROOT / 'r2/pilots/seam_abc_v1/generate.py', '_quality_' + wave + '_api')
    api._load_generator = load_generator
    original_write = api.write
    def exact_barrier_write(path, value, *args, **kwargs):
        if Path(path).name == 'all_RAW_barrier.json':
            value = {**value, 'RAW': 28*len(WAVES[wave]), 'wave': wave}
        return original_write(path, value, *args, **kwargs)
    api.write = exact_barrier_write
    mod.G = api
    mod.OUT = OUT
    mod.ARMS = {arm: arm for arm in WAVES[wave]}
    mod.VERSION = VERSION
    mod.contract = lambda require_go=False: contract(wave, require_go)
    mod.all_raw = lambda reg: all_raw(wave, reg)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'raw', 'post'))
    parser.add_argument('--wave', choices=tuple(WAVES))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if not args.wave:
            parser.error('prepare requires --wave')
        result = prepare(args.wave)
    else:
        if not args.arm:
            parser.error('raw/post requires --arm')
        wave = wave_for(args.arm)
        if args.wave and args.wave != wave:
            parser.error('arm/wave mismatch')
        runtime = adapter(wave)
        result = (runtime.raw if args.stage == 'raw' else runtime.post)(args.arm)
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('protected_files', 'requests', 'layouts', 'files')}, default=str), flush=True)


if __name__ == '__main__':
    main()
