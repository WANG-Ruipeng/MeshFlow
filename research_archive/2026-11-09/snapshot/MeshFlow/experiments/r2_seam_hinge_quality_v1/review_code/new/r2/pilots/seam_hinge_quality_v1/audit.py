"""Independent final read-only budget, identity and geometry audit."""
import csv,json,collections
from pathlib import Path
import numpy as np
from .protocol import ROOT,OUT,ARMS
from r2.pilots.seam_abc_v1 import generate as G

def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def run():
    p=G.read(OUT/'protocol.json');G.verify_files(p['protected_files']);results={}
    train=G.read(OUT/'training_registration.json');G.verify_files(train['protected_hashes'])
    manifests=G.read(train['shared_data_manifest'])['steps']
    for arm in ARMS:
        folder=OUT/'training'/arm;status=G.read(folder/'status.json')
        assert status['status']=='COMPLETE' and status['local_step']==1000 and status['global_step']==15000
        logs=lines(folder/'training_log.jsonl');exposure=lines(folder/'exposure.jsonl')
        assert len(logs)==1000 and len(exposure)==8000
        assert [r['local_step'] for r in logs]==list(range(1,1001))
        for i,row in enumerate(exposure):
            expected=manifests[i//8]['samples'][i%8]
            assert row['local_step']==i//8+1 and row['microbatch']==i%8
            assert row['uid']==expected['uid'] and row['input_sha256']==expected['sha256']
        rg=G.checked_receipt(OUT/'evaluation'/arm/'raw_complete.json')
        pg=G.checked_receipt(OUT/'evaluation'/arm/'post_complete.json')
        assert rg['outputs']==pg['outputs']==28 and rg['generator_forwards']==1400 and pg['D0_forwards']==28
        prior_checks=[]
        for q in p['evaluation']['requests']:
            base=OUT/'evaluation'/arm/q['request_id'];raw=G.load_npz(base/'raw/raw.npz')
            rr=G.checked_receipt(base/'raw/receipt.json');pr=G.checked_receipt(base/'post/receipt.json')
            assert rr['generator_forwards']==50 and pr['D0_forwards']==1
            noise=G.load_npz(base/'raw/actual_initial_noise.npz');old=G.read_noise(q)
            assert np.array_equal(noise['original_z'],old)
            from .training_data import transform_bulk_noise
            expected=transform_bulk_noise(old,noise['bulk_mask'],.25) if arm=='SMOOTH_PRIOR' else old
            assert np.array_equal(noise['z'],expected)
            tr=G.load_npz(base/'raw/trajectory.npz')['state'];mask=raw['raw_anchor_ids']>=0
            assert len(tr)==51 and all(np.array_equal(s[mask],tr[0][mask]) for s in tr)
            post=G.load_npz(base/'post/post.npz');assert np.array_equal(post['C'],raw['C'])
            assert np.array_equal(post['vertices'][post['faces']],post['output'])
            prior_checks.append(q['request_id'])
        results[arm]=dict(updates=1000,exposures=8000,generator_forwards=1400,D0=28,prior_and_anchor_checks=prior_checks)
    for arm in ('PLAIN_S2000',*ARMS):
        h=G.read(OUT/'hinge'/(arm+'_completion.json'));assert h['status']=='COMPLETE' and h['cells']==56
        G.verify_files(h['files'])
    # Rendering completion is not a visual judgement; actual review is audited separately.
    for arm in ARMS:
        comp=G.read(OUT/'sections'/arm/'completion.json');assert len(comp['images'])==28
        for im in comp['images']:
            G.verify_files(im['source_hashes']);G.verify_files(im['outputs'])
    r=dict(status='PASS',training=results,training_updates=2000,training_exposures=16000,
        new_RAW=56,generator_forwards=2800,D0=56,hinge_cells=168,source_assets_unchanged=len(p['protected_files']),
        no_semantic_correspondence_claim_across_generated_slot_ids=True)
    G.write(OUT/'audit/execution.json',r);return r
if __name__=='__main__':print(json.dumps(run(),indent=2))
