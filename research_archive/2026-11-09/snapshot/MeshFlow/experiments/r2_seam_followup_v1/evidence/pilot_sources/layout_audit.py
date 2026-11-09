"""Read-only final budget/identity validation for the fixed layout intervention."""
import collections
import json
from . import layout as L
import numpy as np


def main():
    G=L.G;reg=L.contract();go=G.read(L.OUT/'dispatch_go.json');G.verify_files(go['dependency_hashes'])
    raw=G.checked_receipt(L.OUT/'raw_complete.json');post=G.checked_receipt(L.OUT/'post_complete.json')
    counts=collections.Counter();rows=[];raw_returns=[];d0_starts=[]
    for q in reg['requests']:
        dest=L.OUT/'evaluation'/L.LABEL/q['request_id'];a=G.load_npz(dest/'raw/raw.npz')
        layout,_,source,item=L.load_layout(reg,q);traj=G.load_npz(dest/'raw/trajectory.npz')['state']
        initial=np.zeros((q['N'],3,3),np.float32);initial[q['K']:]=G.read_noise(q)
        fixed=layout['anchor_ids']>=0;initial[fixed]=layout['C_vertices'][layout['anchor_ids'][fixed]]
        assert G.same(traj[0],initial) and G.same(traj[-1],a['raw'])
        assert all(G.same(p[fixed],initial[fixed]) for p in traj)
        b=G.load_npz(dest/'post/post.npz');assert G.same(b['C'],source['C_original'])
        assert G.same(a['occupied'],layout['occupied'])
        assert G.same(a['raw_anchor_ids'],layout['anchor_ids'])
        anchors=G.read(dest/'post/anchor_identity.json');assert anchors['status']=='PASS'
        for stage in ('raw','post'):
            G.checked_receipt(dest/stage/'receipt.json')
            events=[json.loads(x) for x in (dest/stage/'calls.jsonl').read_text().splitlines()]
            counts.update(x['event'] for x in events)
            raw_returns.extend(x['unix'] for x in events if x['event']=='FORWARD_RETURNED')
            d0_starts.extend(x['unix'] for x in events if x['event']=='D0_FORWARD_STARTED')
        old=G.load_npz(L.OLD/'evaluation/B_POOL47'/q['request_id']/'raw/raw.npz')
        same_layout=G.same(old['raw_anchor_ids'],a['raw_anchor_ids'])
        if same_layout:
            assert G.same(old['raw'],a['raw'])
            oldpost=G.load_npz(L.OLD/'evaluation/B_POOL47'/q['request_id']/'post/post.npz')
            assert G.same(oldpost['output'],b['output'])
        rows.append(dict(request_id=q['request_id'],split=q['split'],active_seams=int(a['occupied'].sum()),
            original_gaussian_exact=True,C_and_seam_fixed_all51=True,source_C_exact=True,post_anchor_identity='PASS',
            same_layout_as_B_own=same_layout,unchanged_layout_RAW_and_POST_bitwise='PASS' if same_layout else 'NA_CHANGED_LAYOUT'))
    assert counts==dict(FORWARD_STARTED=1400,FORWARD_RETURNED=1400,D0_FORWARD_STARTED=28,D0_FORWARD_RETURNED=28)
    assert max(raw_returns)<min(d0_starts)
    summary=G.read(L.OUT/'evaluation/summary_completion.json');G.verify_files(summary['files'])
    assert summary['numeric_fields']==124 and summary['new_scans']==56 and summary['reused_scores']==112
    score=G.read(L.OUT/'evaluation/score_completion.json')
    for rec in score['receipts']:G.verify_files({rec['path']:rec['sha256']});G.verify_files(G.read(rec['path'])['files'])
    value=dict(status='PASS',version='seam_followup_layout_identity_budget_audit_v1',requests=28,
        generator_forwards=1400,D0_forwards=28,training_updates=0,head_forwards=0,pointer_forwards=0,
        events=dict(counts),rows=rows,unchanged_layout_requests=sum(r['same_layout_as_B_own'] for r in rows),
        failure_disclosure='One pre-forward setup TypeError; model loaded, zero forward/outputs. Original attempt and code hash retained; scalar-only engineering correction.',
        source_and_code_hashes_unchanged=True,all_RAW_before_D0=True,
        last_RAW_forward_return_unix=max(raw_returns),first_D0_start_unix=min(d0_starts),
        files={str(L.OUT/name):G.sha(L.OUT/name) for name in ('registration.json','dispatch_go.json','raw_complete.json','post_complete.json','evaluation/summary_completion.json','evaluation/score_completion.json')})
    G.write(L.OUT/'audit.json',value);print(json.dumps({k:v for k,v in value.items() if k not in ('rows','files')}))

if __name__=='__main__':main()
