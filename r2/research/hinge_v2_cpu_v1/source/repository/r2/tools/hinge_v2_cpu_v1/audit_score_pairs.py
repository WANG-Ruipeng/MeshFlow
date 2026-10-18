"""Compare cached geometric decisions with completed endpoint scoring scans.

No contact detection or mesh scoring is run here. In particular, equivalence of
BRIDGE incremental scans is certified only if score.json explicitly records a
fresh full-pair scan at that endpoint, never by comparing inherited copies.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
from .diagnose import read, sha, write_csv

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_hinge_v2_cpu_v1'
BASE=ROOT/'experiments/r2_hinge_opposite_v1/cells/HINGE_OPP015'
STAGES=('BASELINE','MIDPOINT_REPLAY','ALPHA','HOLE','ALPHA_HOLE')


def contacts(scan):
    result={}
    for r in scan['contacts']:
        key=tuple(map(int,r['pair']));assert key not in result
        detector=r.get('detector',{})
        result[key]=(r['classification'],detector.get('resolved') is False,detector.get('verified') is False)
    return result


def run():
    start=time.perf_counter();reg=read(OUT/'protocol.json');records=[];input_hashes={}
    for q in reg['requests']:
        rid=q['request_id']
        for stage in STAGES:
            score=OUT/'scores'/stage/rid;assert (score/'complete.json').exists(),str(score)
            old=BASE/rid if stage=='BASELINE' else OUT/'cells'/stage/rid
            a=read(old/'pairs.json');b=read(score/'pairs.json');result=read(score/'score.json')
            row=read(score/'row.json');bridges=int(row.get('new_bridge_faces',0))
            origin=result['pair_origin']
            if bridges:
                assert 'INHERIT' not in origin.upper(),('bridge cache compared to inherited copy',stage,rid,origin)
                assert 'FULL' in origin.upper(),origin
            compared=0
            for kind in ('CF','FF'):
                aa,bb=contacts(a[kind]),contacts(b[kind]);assert aa==bb,(stage,rid,kind,'contact classification mismatch')
                assert a[kind]['expected_full_pairs']==b[kind]['expected_full_pairs']
                assert a[kind]['tested_pairs']==b[kind]['tested_pairs']==b[kind]['expected_full_pairs']
                compared+=len(aa)
            records.append(dict(request_id=rid,split=q['split'],stage=stage,bridge_faces=bridges,
                pair_origin=origin,contact_rows_compared=compared,
                exact_contact_classification_and_unresolved_flags_equal=True,
                bridge_incremental_vs_independent_full_scan_verified=bridges>0))
            for p in (old/'pairs.json',score/'pairs.json',score/'complete.json'):input_hashes[str(p)]=sha(p)
    assert len(records)==140
    dest=OUT/'independent_audit';dest.mkdir(parents=True,exist_ok=True)
    write_csv(dest/'score_pair_equivalence.csv',records)
    result=dict(status='PASS_SAVED_SCORE_PAIR_EQUIVALENCE',outputs=len(records),
        bridge_outputs_independently_full_scanned=sum(r['bridge_faces']>0 for r in records),
        bridge_faces_in_those_outputs=sum(r['bridge_faces'] for r in records),
        contact_rows_compared=sum(r['contact_rows_compared'] for r in records),
        mismatched_classifications=0,mismatched_unresolved_flags=0,
        new_geometry_scans=0,new_model_calls=0,seconds=time.perf_counter()-start,
        caveat='No-bridge inherited caches are identity checks, not independent full rescans.',
        source_hashes=input_hashes,code_sha256=sha(__file__),
        files={str(dest/'score_pair_equivalence.csv'):sha(dest/'score_pair_equivalence.csv')})
    (OUT/'independent_score_pair_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','files')},indent=2));return result


def main():return run()
if __name__=='__main__':main()
