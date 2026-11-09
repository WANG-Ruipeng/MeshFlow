"""Explicit parent releases after input and resource gates; no probe jobs."""
import argparse,json
from .protocol import ROOT,OUT
from .resources import snapshot
from r2.pilots.seam_abc_v1 import generate as G
def release(stage):
    resources=snapshot(stage+'_release')
    if stage=='training':
        regpath=OUT/'training_registration.json';reg=G.read(regpath)
        audit=G.read(OUT/'training_inputs/transport_audit.json');assert audit['status']=='PASS' and audit['samples']==8000
        deps=G.read(ROOT/'experiments/r2_seam_quality_wave1_wave2_v1/wave2_inputs/source_code_dependencies.json')['code_hashes']
        code={**deps,**reg['code_hashes']}
        # Full inherited code definition stays protected even when only its helper is used.
        G.verify_files(code)
        path=OUT/'training_dispatch_go.json'
        value=dict(authorized=True,training_registration_sha256=G.sha(regpath),
            shared_data_manifest_sha256=G.sha(reg['shared_data_manifest']),code_hashes=code,resources=resources,
            budget=dict(arms=reg['arms'],updates_each=1000,neural_forward_each=8000,neural_backward_each=8000),
            initial_workers=1,decision='Start first authorized arm; measure actual backward/optimizer before additional dispatch; no added probes')
        policy=OUT/'dispatch_policy.json'
        if not policy.exists():G.write(policy,dict(minimum_free_mib_before_load=6963,minimum_runtime_free_mib=2048,
            worker_process_allowance_mib=4096,margin_factor=1.2,initial_workers=1,
            evidence='Comparable fullSEAM allocatorpeak~3184MiB, processVRAM unavailable underWDDM; conservative4096MiB allowance then observe real updates',
            user_policy='Use currently free memory, retain safety margin; no fixed6GB cap'))
        G.dispatch_guard(resources,4096)
    elif stage=='generation':
        from .generation import contract
        reg=contract();regpath=OUT/'generation_registration.json';path=OUT/'generation_dispatch_go.json'
        files=[ROOT/'r2/pilots/seam_hinge_quality_v1'/n for n in ('generation.py','training_data.py','protocol.py')]
        files += [ROOT/rel for rel in ('r2/pilots/owner_layout_v1/generation.py','r2/pilots/seam_abc_v1/generate.py',
            'r2/pilots/seam_abc_v1/postprocess.py','r2/pilots/seam_quality_four_v1/postprocess.py',
            'r2/src/meshflow_control/denoise_model.py')]
        value=dict(authorized=True,registration_sha256=G.sha(regpath),code_hashes={str(f):G.sha(f) for f in files},
            resources=resources,budget=reg['budget'],initial_workers=1,
            decision='Incremental measured generationworkers, all56RAW beforeD0, D0memory sizedseparately')
        G.dispatch_guard(resources,reg['resources']['generator_estimate_mib'])
    else:raise ValueError(stage)
    if path.exists():
        old=G.read(path);assert old['code_hashes']==value['code_hashes'];return old
    G.write(path,value);return value
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['training','generation']);r=release(p.parse_args().stage)
    print(json.dumps(dict(authorized=r['authorized'],budget=r['budget'])))
