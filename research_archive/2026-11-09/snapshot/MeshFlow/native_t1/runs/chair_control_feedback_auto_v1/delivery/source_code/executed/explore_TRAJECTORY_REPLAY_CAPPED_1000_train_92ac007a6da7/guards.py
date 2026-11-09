"""Positive DEV and frozen identities are required for later GPU stages."""
from .common import *

def repeat(out,arm):
    p=Path(out)/'decisions/repeat_authorization.json'
    if not p.exists():raise RuntimeError('Independent repeat is gated on favorable reviewed DEV')
    r=read_json(p)
    assert r['status']=='FROZEN' and arm in r['arms'] and r['START_sha256']==START_SHA
    pair={r['candidate_arm'],r['matched_arm']}
    assert len(pair)==2 and set(r['arms'])==pair and set(r['exploration_checkpoints'])==pair
    assert r['candidate_arm'] in ARMS and r['candidate_arm']!='FM_CONTROL' and r['matched_arm']=='REPLAY_FM_CONTROL'
    assert all(v.get('path') and v.get('sha256') for v in r['exploration_checkpoints'].values())
    assert digest(r['DEV_gate_path'])==r['DEV_gate_sha256']
    gate=read_json(r['DEV_gate_path']);assert gate['status']=='PASS' and gate['numeric_status']=='PASS'
    for model in r['exploration_checkpoints'].values():assert digest(model['path'])==model['sha256']
    return r

def confirm(out,label,sha,release):
    if not release:raise RuntimeError('CONFIRM remains sealed')
    r=read_json(release)
    assert r['status']=='FROZEN' and len(r['models'])==5
    assert r['models'][label]['sha256']==sha and r['START_sha256']==START_SHA
    assert r['CONFIRM_manifest_sha256']==digest(Path(out)/'manifests/CONFIRM.json')
    assert r['candidate_checkpoint_sha256']
    authorization=read_json(Path(out)/'decisions/repeat_authorization.json')
    assert set(r['repeat_arms'])=={authorization['candidate_arm'],authorization['matched_arm']} and len(r['repeat_arms'])==2
    assert {m['role'] for m in r['models'].values()}=={'START','explore_candidate','explore_matched','repeat_candidate','repeat_matched'}
    assert next(m['sha256'] for m in r['models'].values() if m['role']=='START')==START_SHA
    assert next(m['sha256'] for m in r['models'].values() if m['role']=='explore_candidate')==r['candidate_checkpoint_sha256']
    for arm in r['repeat_arms']:
        repeat(out,arm)
        done=read_json(Path(out)/'training'/f'repeat_{arm}_complete.json')
        assert done['status']=='PASS' and done['steps']==1000
    for model in r['models'].values():assert digest(model['path'])==model['sha256']
    return r
