"""Freeze evaluation inputs, reuse exact LONG TRAIN artifacts; no model calls."""
from pathlib import Path
import argparse
import ast
import json
import shutil
from . import eval_common as C

OLD = C.ROOT/'experiments/r2_seam_quality_four_v1'

def copy_stage(src, dest, stage):
    old = C.read(src/'receipt.json')
    for p, h in old['files'].items():
        if C.sha(p) != h:
            raise ValueError('Protected cache changed: '+p)
    dest.mkdir(parents=True, exist_ok=True)
    copied = {}
    for p in src.iterdir():
        if not p.is_file() or p.name == 'receipt.json':
            continue
        q = dest/p.name
        if q.exists():
            if C.sha(q) != C.sha(p): raise ValueError('Cache collision '+str(q))
        else:
            shutil.copyfile(p, q)
        copied[str(q)] = C.sha(q)
    value = dict(old, files=copied, cache_origin=str(src), cache_origin_receipt_sha256=C.sha(src/'receipt.json'),
                 newly_generated=False, new_rollouts=0, new_D0_forwards=0)
    for key in ('raw_path','post_path','pre_geo_path'):
        if key in value and Path(value[key]).parent == src:
            value[key] = str(dest/Path(value[key]).name)
    C.write(dest/'receipt.json', value)

def prepare():
    root=C.OUT
    old=C.read(OLD/'eval_manifest.json')
    extra=C.read(root/'B_EVAL_DEV_ASSETS.json')
    conditions=old['conditions']+extra['conditions']
    requests=old['requests']+extra['requests']
    if len(conditions)!=7 or len(requests)!=28 or len({r['request_id'] for r in requests})!=28:
        raise ValueError('Need 4 TRAIN + 3 DEV, four existing noises each')
    manifest=dict(schema='seam_abc_eval_v1', arms=list(C.ARMS), conditions=conditions, requests=requests,
                  original_manifest=str(OLD/'eval_manifest.json'), original_manifest_sha256=C.sha(OLD/'eval_manifest.json'),
                  dev_manifest_sha256=C.sha(root/'B_EVAL_DEV_ASSETS.json'))
    C.write(root/'eval_manifest.json',manifest)
    # Only artifact copies; no rescoring, new noise, or geometry mutation.
    for r in old['requests']:
        for stage in ('raw','post'):
            copy_stage(OLD/'evaluation/LONG'/r['request_id']/stage,
                       root/'evaluation/LONG'/r['request_id']/stage,stage)
    oldlayouts={C.read(p/'receipt.json')['condition_id']:p for p in (OLD/'evaluation/LONG/layouts').iterdir() if p.is_dir()}
    for ci, cid in enumerate(sorted(r['condition_id'] for r in conditions)):
        if cid not in oldlayouts:continue
        src=oldlayouts[cid];dest=root/'evaluation/LONG/layouts'/f'C{ci:02d}'
        dest.mkdir(parents=True,exist_ok=True)
        for name in ('layout.npz','source_condition.npz'):
            p=dest/name
            if p.exists():
                if C.sha(p)!=C.sha(src/name):raise ValueError('Layout cache collision')
            else:shutil.copyfile(src/name,p)
        receipt=C.read(src/'receipt.json')
        receipt.update(occupancy_forward_calls=0,cache_origin_receipt=str(src/'receipt.json'),
                       files={str(dest/n):C.sha(dest/n) for n in ('layout.npz','source_condition.npz')})
        C.write(dest/'receipt.json',receipt)
    prior=C.read(OLD/'registration.json')
    reg=dict(schema='R2_SEAM_ABC_V1',status='FROZEN',arms=list(C.ARMS),updates_per_arm=800,
             source_checkpoint=prior['source_checkpoint'],source_checkpoint_sha256=prior['source_checkpoint_sha256'],
             d0_checkpoint=prior['d0_checkpoint'],eval_manifest_sha256=C.sha(root/'eval_manifest.json'),
             generator_Euler_steps=50,new_rollouts=96,new_D0_forwards=96,cached_LONG_TRAIN=16,
             expected_occupancy_forwards=24,expected_pointer_forwards=28,
             all_RAW_before_D0=True,resources=prior['resources'],
             scope='4 seen TRAIN C and 3 existing DEV C; all four fixed old noises, no result selection',
             comparison='Each ABC arm versus LONG at cumulative1000; historical SEAM200 only on TRAIN16',
             protocol_sha256=C.sha(root/'PROTOCOL.md'), source_labels_in_inference=False)
    C.write(root/'registration.json',reg)
    print(json.dumps(dict(status='REGISTERED',requests=28,arms=4,new_RAW=96,new_D0=96,cached=16)))

def freeze():
    root=C.OUT;manifest=C.read(root/'eval_manifest.json')
    code={}
    for name in ('generate.py','postprocess.py','eval_common.py','shared_tip.py','pointer.py'):
        p=Path(__file__).parent/name
        ast.parse(p.read_text(encoding='utf-8'));code[str(p)]=C.sha(p)
    for folder in ('r2/prototypes/seam_slot_c200_v1','r2/src/meshflow_control/seam'):
        code.update({str(p):C.sha(p) for p in (C.ROOT/folder).glob('*.py')})
    inputs={str(root/'eval_manifest.json'):C.sha(root/'eval_manifest.json'),str(root/'registration.json'):C.sha(root/'registration.json')}
    for q in manifest['requests']:
        for k in ('inference_npz','noise_npz','metadata_npz'):
            inputs[q[k]]=C.sha(q[k])
        inputs[q['query_descriptor']['path']]=q['query_descriptor']['sha256']
    C.write(root/'execution_freeze.json',dict(status='FROZEN',code_hashes=code,input_hashes=inputs,
         scope='fixed endpoint generation/postprocessing; training has separate per-arm freezes'))
    print(json.dumps(dict(status='FROZEN',code_files=len(code),inputs=len(inputs))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','freeze'),required=True)
    a=p.parse_args();prepare() if a.stage=='prepare' else freeze()
