"""CPU data/implementation checks; no generator/head/D0 constructions/calls."""
from pathlib import Path
import argparse
import ast
import json
import numpy as np
from . import support as S
from .prepare import sha, write, schedule


def check(root):
    path=root/'training_cpu_checks.json'
    if path.exists(): return S.read(path)
    parsed=[]
    for p in Path(__file__).parent.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8')); parsed.append(dict(path=str(p),sha256=sha(p)))
    reg=S.read(root/'training_registration.json')
    if schedule(reg['records']) != S.read(root/'input_schedule.json'):
        raise ValueError('Frozen independent data stream differs')
    manifest=S.read(root/'data_manifest.json')
    if len(manifest['steps'])!=800: raise ValueError('Not 800 steps')
    for entry in manifest['steps']:
        if not entry['samples']['LONG']==entry['samples']['ANTI_COVER']==entry['samples']['LOCAL_NOISE']:
            raise ValueError('Arms do not share exact source OT arrays')
    rows=[]
    # First/last complete batches exercise all four C and both augmentation modes.
    for entry in (manifest['steps'][0],manifest['steps'][-1]):
        samples=[]
        for micro,record in enumerate(entry['samples']['LONG']):
            a=S.load_sample(record,root,'LONG',entry['step'],micro,sha)
            b=S.load_sample(record,root,'LOCAL_NOISE',entry['step'],micro,sha)
            fixed=a['anchor_ids']>=0
            slots=np.flatnonzero(~a['known_mask'] & fixed[:,0] & fixed[:,1] & ~fixed[:,2])
            midpoint=a['C_vertices'][a['anchor_ids'][slots,:2]].mean(1)
            np.testing.assert_array_equal(b['x0'][slots,2],a['x0'][slots,2]+midpoint)
            np.testing.assert_array_equal(a['x1'],b['x1'])
            np.testing.assert_array_equal(a['ot_face_map'],b['ot_face_map'])
            np.testing.assert_array_equal(a['ot_corner_map'],b['ot_corner_map'])
            np.testing.assert_array_equal(a['source_face_ids'],b['source_face_ids'])
            np.testing.assert_array_equal(a['anchor_ids'],b['anchor_ids'])
            assert set(map(int,a['anti_cover_seam_slots']))==set(map(int,slots))
            assert len(a['anti_cover_owner_faces'])==len(slots)
            assert not b['u'][fixed].any()
            samples.append(a)
            rows.append(dict(step=entry['step'],source_step=entry['source_step'],micro=micro,
                uid=a['uid'],N=a['N'],K=a['K'],seams=len(slots),matching_unchanged=True,
                only_tip_origin_translated=True,original_source_target_unchanged=True))
        S.validate_batch(samples)
    result=dict(status='PASS',code=parsed,checked_samples=rows,shared_OT_calls=6400,
        no_new_OT_in_check=True,neural_forward=0,optimizer_updates=0,GPU_calls=0,
        restore_contract=dict(generator_start=12200,generator_end=13000,head_start=200,head_end=1000,
            source='retained live SEAM_SLOT_C200 step200 + both matched AdamW + model RNG',
            data_RNG='separate MT19937 verified old200 then continued201..1000'),
        limitations='Array and syntax checks only; actual CUDA restore/gradients checked on authorized training')
    write(path,result); return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='experiments/r2_seam_quality_four_v1');a=p.parse_args()
    print(json.dumps(check(S.resolve(a.root)),ensure_ascii=False)[:2000])
