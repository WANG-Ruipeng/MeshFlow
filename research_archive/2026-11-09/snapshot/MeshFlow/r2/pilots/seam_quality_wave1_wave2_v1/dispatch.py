"""Explicit dispatch releases after verified preparation, never a scheduler probe."""
import argparse,json,time
from .protocol import ROOT,OUT
from .resources import snapshot
from r2.pilots.seam_abc_v1 import generate as G

def release(stage):
    r=snapshot(stage+'_release')
    if stage in ('wave1','wave2'):
        from .generation import contract
        reg=contract(stage)
        paths=[Path for Path in (ROOT/'r2/pilots/seam_quality_wave1_wave2_v1').glob('*.py') if Path.name in ('generation.py','evaluation.py','protocol.py')]
        paths += [ROOT/'r2/pilots/owner_layout_v1/generation.py',ROOT/'r2/pilots/seam_abc_v1/generate.py',
                  ROOT/'r2/pilots/seam_quality_four_v1/postprocess.py',ROOT/'r2/src/meshflow_control/denoise_model.py']
        go=dict(authorized=True,registration_sha256=G.sha(OUT/f'generation_{stage}_registration.json'),
            code_hashes={str(p):G.sha(p) for p in paths},resources=r,
            plan='Incremental workers; first actual task measured before additional dispatch. At least2048MiB spare and1.2 process-peak margin; compute utilization also considered.',
            budget=reg['budget'])
        p=OUT/f'generation_{stage}_dispatch_go.json'
    elif stage=='training':
        summary=G.read(OUT/'results/wave1/completion.json')
        if summary['rows']!=504 or summary['numeric_fields']!=124:
            raise ValueError('All Wave1 numerical comparisons must finish before Wave2')
        reg=G.read(OUT/'training_registration.json')
        dependencies=G.read(OUT/'wave2_inputs/source_code_dependencies.json')
        hashes={**reg['code_hashes'],**dependencies['code_hashes']}
        G.verify_files(hashes)
        go=dict(authorized=True,training_registration_sha256=G.sha(OUT/'training_registration.json'),
            shared_data_manifest_sha256=G.sha(reg['shared_data_manifest']),code_hashes=hashes,
            wave1_numerical_completion_sha256=G.sha(OUT/'results/wave1/completion.json'),
            resources=r,budget=dict(arms=reg['arms'],updates_each=1000,neural_forward_each=8000,neural_backward_each=8000),
            decision='Preauthorized both arms irrespective of Wave1 outcome; exact PLAIN_S1000 control; no extra probe or extension')
        p=OUT/'training_dispatch_go.json'
        policy=OUT/'dispatch_policy.json'
        if not policy.exists():
            G.write(policy,dict(minimum_free_mib_before_load=6963,initial_workers=1,
                minimum_runtime_free_mib=2048,conservative_worker_allowance_mib=4096,
                reason='Comparable full-SEAM BF16 micro1 allocator peak3174MiB; conservatively4096MiB process allowance +20% +2048MiB spare. Observe real new update before increasing workers.',
                user_resource_policy='Use currently free VRAM with safety margin; former6GB cap superseded'))
    else:raise ValueError('Unknown stage')
    if p.exists():
        old=G.read(p)
        key='training_registration_sha256' if stage=='training' else 'registration_sha256'
        if old[key]!=go[key] or old['code_hashes']!=go['code_hashes']:raise ValueError('Release definition changed')
        return old
    G.write(p,go);return go

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['wave1','wave2','training'])
    a=parser.parse_args();r=release(a.stage);print(json.dumps(dict(authorized=r['authorized'],budget=r['budget'])))
