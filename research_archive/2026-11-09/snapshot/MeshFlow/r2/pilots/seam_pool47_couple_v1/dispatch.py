"""Freeze authorized execution code and resource policy, without launching jobs."""
from pathlib import Path
import argparse,json
from r2.pilots.seam_abc_v1 import generate as G
from .protocol import OUT,ROOT

def dependencies():
    # Include actual repo definitions, not installed runtime packages/weights.
    paths=set((ROOT/'r2/src/meshflow_control').rglob('*.py'))
    for folder in ('r2/prototypes/seam_slot_c200_v1','r2/pilots/seam_slot_c200_v1',
        'r2/pilots/seam_abc_v1','r2/pilots/user_condition_finetune_v1',
        'r2/pilots/full_seat_learning_curve_v1'):
        paths.update((ROOT/folder).rglob('*.py'))
    for name in ('training.py','support.py','data.py','protocol.py'):
        paths.add(Path(__file__).with_name(name))
    return {str(p):G.sha(p) for p in sorted(paths)}

def training():
    review=G.read(OUT/'implementation_review.json')
    if review['status']!='PASS':raise ValueError('Independent implementation review not passed')
    for path,digest in review['reviewed_code_hashes'].items():
        p=Path(path);p=p if p.is_absolute() else ROOT/p
        if G.sha(p)!=digest:raise ValueError('Reviewed implementation changed: '+str(p))
    reg=G.read(OUT/'training_registration.json');cache=G.read(OUT/'shared_data_manifest.json')
    if cache['status']!='COMPLETE' or cache['samples']!=24000:raise ValueError('Full common cache required')
    G.verify_files(reg['protected_hashes']);G.verify_files(reg['cache_code_hashes'])
    path=OUT/'training_dispatch_go.json'
    if path.exists():return G.read(path)
    snap=G.resources()
    go=dict(authorized=True,training_registration_sha256=G.sha(OUT/'training_registration.json'),
        shared_data_manifest_sha256=G.sha(OUT/'shared_data_manifest.json'),
        protocol_sha256=G.sha(OUT/'protocol.json'),review_sha256=G.sha(OUT/'implementation_review.json'),
        code_hashes=dependencies(),resources=snap,
        plan='Initially one already-authorized arm; measure real backward/AdamW peak before another worker. Honor user VRAM preference, >=2048MiB headroom and measured compute utilization. No extra probe updates.',
        initial_user_training_budget_mib=6144,initial_workers=1,updates_per_arm=3000)
    G.write(OUT/'dispatch_policy.json',dict(minimum_free_mib_before_load=6963,
        estimate='Historical comparable allocator max3174MiB; conservative worker allowance4096MiB +20% +2048MiB headroom. Process memory measured after actual updates; allocator is not total VRAM.',
        initial_workers=1,training_budget_mib=6144))
    G.write(path,go);G.event(OUT/'ledger.jsonl','TRAINING_RELEASE',workers=1,updates_per_arm=3000)
    return go

def generation():
    from .generation import prepare
    reg=prepare();path=OUT/'generation_dispatch_go.json'
    if path.exists():return G.read(path)
    deps=dependencies()
    for name in ('generation.py','dispatch.py'):
        p=Path(__file__).with_name(name);deps[str(p)]=G.sha(p)
    for rel in ('r2/pilots/owner_layout_v1/generation.py','r2/pilots/seam_followup_v1/layout.py'):
        p=ROOT/rel;deps[str(p)]=G.sha(p)
    result=dict(authorized=True,registration_sha256=G.sha(OUT/'generation_registration.json'),code_hashes=deps,
        resources=G.resources(),RAW=56,D0=56,head_forwards=0,all_RAW_before_D0=True,
        plan='Recheck resources before each incremental generator/D0 worker; original N/batch/precision fixed.')
    G.write(path,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['training','generation']);a=p.parse_args()
    r=training() if a.stage=='training' else generation()
    print(json.dumps({k:v for k,v in r.items() if k!='code_hashes'},indent=2))
