"""CPU-only frozen inputs/layout cache and experiment registration."""
import json
from pathlib import Path
import numpy as np
from .common import *
from .eval_plan import BASELINE_ROOT, SECONDARY_ROOT, cached_layout, P, arrays

def run():
    original = read(SECONDARY_ROOT / 'eval_manifest.json')
    assert original == read(BASELINE_ROOT / 'eval_manifest.json')
    protected = dict(original.get('protected', {}))
    protected[str(SECONDARY_ROOT / 'eval_manifest.json')] = sha(SECONDARY_ROOT / 'eval_manifest.json')
    protected[str(BASELINE_ROOT / 'eval_manifest.json')] = sha(BASELINE_ROOT / 'eval_manifest.json')
    rows = []
    for original_q in original['requests']:
        q = dict(original_q)
        raw = SECONDARY_ROOT / 'evaluation/JOINT_SHIFT_S1500' / q['request_id'] / 'raw'
        r = read(raw / 'receipt.json')
        for key, name in [('trajectory','trajectory.npz'),('receipt','receipt.json'),('raw','raw.npz')]:
            path = raw / name
            digest = sha(path)
            if key != 'receipt':
                expected = next(h for p,h in r['files'].items() if resolve(p) == path)
                assert digest == expected
            q['prefix_'+key+'_path'] = str(path)
            q['prefix_'+key+'_sha256'] = digest
            protected[str(path)] = digest
        state = arrays(raw/'trajectory.npz')['state']
        assert state.shape == (51,q['N'],3,3) and state.dtype == np.float32 and np.isfinite(state).all()
        layout, logits, source = cached_layout(q)
        dest = OUT / 'evaluation' / LABEL / 'layouts' / q['condition_id']
        if not (dest/'receipt.json').exists():
            dest.mkdir(parents=True, exist_ok=False)
            P.array_file(dest/'layout.npz', {**layout,'occupancy_logits':logits})
            P.array_file(dest/'source_condition.npz', source)
            P.immutable(dest/'receipt.json', dict(status='COMPLETE',condition_id=q['condition_id'],label=LABEL,
                old_layout_sha256=q['old_layout_sha256'],head_forwards=0,source_labels_read=False,
                files={str(p):sha(p) for p in dest.iterdir() if p.is_file()}))
        for p in dest.iterdir():
            if p.is_file(): protected[str(p)] = sha(p)
        rows.append(q)
    manifest = {**original, 'requests':rows}
    P.immutable(OUT/'eval_manifest.json',manifest)
    gen = BASELINE_ROOT / 'training/RELATION_V1_SEAM/generator_step1500.pt'
    for p in (gen,gen.with_suffix('.json'),OUT/'eval_manifest.json',OUT/'PROTOCOL.md'):
        protected[str(p)] = sha(p)
    from r2.pilots.bench_control_transfer_v1 import sample as S
    from r2.pilots.seam_slot_c200_v1 import generate as G
    from meshflow_control.seam import model_adapter as A
    files = list(Path(__file__).parent.glob('*.py')) + [Path(S.__file__),Path(G.__file__),Path(A.__file__)]
    value = dict(status='FROZEN',experiment=EXPERIMENT,evaluation_labels={LABEL:dict(arm=LABEL,
        source_arm='RELATION_V1_SEAM',subset='full',local_step=1500,global_step=13500,generator_path=str(gen),
        source_prefix_root=str(SECONDARY_ROOT),source_prefix_label='JOINT_SHIFT_S1500')},
        split_step=25,absolute_times='.50,.52,...,.98',budget=dict(training_updates=0,generator_forwards=1200,
        D0_forwards=48,new_RAW=48,new_scores=144,new_noise=0,new_OT=0,head_forwards=0),
        protected=protected,source_hashes={str(p):sha(p) for p in files},
        blind_gate_required=True,default_changed=False,automatic_retry=False)
    P.immutable(OUT/'registration.json',value)
    snap = resources()
    P.immutable(OUT/'resources_initial.json',snap)
    return dict(status='FROZEN',requests=len(rows),protected=len(protected),model_calls=0,
                gpu=snap['gpu'],CPU_cores=snap['CPU_cores'],RAM_available=snap['RAM_available'])

if __name__ == '__main__': print(json.dumps(run(),ensure_ascii=False))
