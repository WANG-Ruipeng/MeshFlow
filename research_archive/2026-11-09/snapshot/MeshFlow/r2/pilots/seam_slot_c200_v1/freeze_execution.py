"""Seal implemented stage code and all inference inputs before actual training."""
from . import common as C

def run():
    path=C.OUT/'execution_freeze.json'
    if path.exists():
        r=C.read(path)
        for p,h in {**r['code_hashes'],**r['input_hashes']}.items():assert C.sha(p)==h,p
        return r
    code=list((C.ROOT/'r2/pilots/seam_slot_c200_v1').glob('*.py'))
    code+=list((C.ROOT/'r2/prototypes/seam_slot_c200_v1').glob('*.py'))
    code+=list((C.ROOT/'r2/src/meshflow_control').rglob('*.py'))
    inputs=list((C.OUT/'inputs').rglob('*.npz'))+[C.OUT/n for n in ('registration.json','eval_manifest.json','input_schedule.json','data_manifest.json','PROTOCOL.md','dispatch_policy.json')]
    r=dict(schema='seam_c200_execution_freeze_v1',utc=C.utc(),registration_sha256=C.sha(C.OUT/'registration.json'),
        code_hashes={str(p):C.sha(p) for p in sorted(set(code))},input_hashes={str(p):C.sha(p) for p in inputs},
        training_data='All3200cached NPZ hashes are bound individually in data_manifest and verified by each loader',
        source='Original weights/data/scorers protected by their existing identity and source hashes; no writes')
    C.write(path,r);C.event('EXECUTION_FROZEN',sha256=C.sha(path),code_files=len(code),input_files=len(inputs));return r

if __name__=='__main__':
    r=run();print('EXECUTION_FROZEN',len(r['code_hashes']),len(r['input_hashes']))
