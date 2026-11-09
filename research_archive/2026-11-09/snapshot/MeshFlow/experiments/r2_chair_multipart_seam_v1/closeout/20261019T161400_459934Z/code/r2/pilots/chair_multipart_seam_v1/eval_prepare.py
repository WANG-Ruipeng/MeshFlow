"""Freeze the existing 28-seat/16-back requests without resampling."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'r2/src'))
from r2.pilots.seam_slot_c200_v1 import generate as G

ROOT=G.REPOSITORY
OUT=ROOT/'experiments/r2_chair_multipart_seam_v1'
LABEL='MULTIPART_S1500'


def run():
    dest=OUT/'eval_manifest.json'
    if dest.exists():
        m=G.read(dest);G.verify_files(m['protected']);return m
    seatpath=ROOT/'experiments/r2_seam_pool47_couple_v1/protocol.json'
    backpath=ROOT/'experiments/r2_chair_back_transfer_v1/eval_manifest.json'
    seat=G.read(seatpath)['evaluation']['requests'];back=G.read(backpath)['requests']
    assert len(seat)==28 and len(back)==16
    rows=[];protected={str(p):G.sha(p) for p in (seatpath,backpath)}
    for part,qs in (('SEAT_FULL',seat),('BACK_FULL',back)):
        for original in qs:
            q=dict(original);q['part_type']=part;q['original_split']=q['split']
            for key in ('inference_npz','noise_npz','metadata_npz','source_npz'):
                if key not in q:continue
                digest=G.sha(q[key]);q[key+'_sha256']=digest;protected[q[key]]=digest
            desc=q['query_descriptor'];assert G.sha(desc['path'])==desc['sha256']
            protected[desc['path']]=desc['sha256']
            bundle=G.read_bundle(q);G.source_from_bundle(q,bundle);G.read_noise(q)
            rows.append(q)
    assert len({q['request_id'] for q in rows})==44
    assert len({q['condition_id'] for q in rows})==15
    m=dict(status='FROZEN',label=LABEL,requests=rows,protected=protected,
        new_noise_arrays=0,new_query_arrays=0,raw_outputs=44,D0_outputs=44,
        generator_forwards=2200,head_forward_calls=15,
        original_panels=dict(SEAT_FULL=28,BACK_FULL=16),
        scope='BACK FAMILIAR becomes seen-condition adaptation under the new pool; DEV held out. Preserve original split labels and never combine the two panels to hide regression.',
        source_style_success_criterion=False)
    G.write(dest,m);return m


if __name__=='__main__':
    m=run();print({'status':m['status'],'requests':len(m['requests']),'noise':0})
