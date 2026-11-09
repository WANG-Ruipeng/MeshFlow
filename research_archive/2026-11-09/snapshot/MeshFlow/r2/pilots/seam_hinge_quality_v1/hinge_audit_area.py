"""Area-tail explanation from already saved original/final POST geometry."""
import json
import numpy as np
from . import hinge as H
from .hinge_audit import geometry


def main():
    rows=[]
    for q in H.requests():
        base=H.C.arrays(H.paths('PLAIN_S2000',q)['post']);a=geometry(base['output']);K=q['K'];sf=base['source_face_ids']
        for d in H.DELTAS:
            label=H.label('PLAIN_S2000',d);path=H.OUT/'cells'/label/q['request_id']/'post.npz';other=H.C.arrays(path);b=geometry(other['output']);ratio=b['area']/a['area']
            for i in np.flatnonzero((np.arange(len(sf))>=K)&((ratio<.6)|(ratio>3))):
                rows.append(dict(label=label,request_id=q['request_id'],raw_slot=int(sf[i]),area_ratio=float(ratio[i]),
                    area_before=float(a['area'][i]),area_after=float(b['area'][i]),area_delta_L2=float((b['area'][i]-a['area'][i])/H.C.L**2),
                    angle_before=float(a['angle'][i]),angle_after=float(b['angle'][i]),q_before=float(a['q'][i]),q_after=float(b['q'][i]),
                    original_post_sha=H.C.sha(H.paths('PLAIN_S2000',q)['post']),candidate_post_sha=H.C.sha(path)))
    r=dict(status='COMPLETE_SAVED_ARRAYS_ONLY',rows=rows,new_scans=0,source_sha=H.C.sha(__file__),
        interpretation='Relative area changes can be large even with preserved face identities and shape guards. Absolute area and angles shown; no causal or visual-quality inference.')
    H.C.write(H.OUT/'baseline_audit_area_tails.json',r);print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
