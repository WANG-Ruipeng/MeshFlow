"""Exactly one official Bench D0 per new output, after all96 RAW are saved."""
from __future__ import annotations
import argparse,json
from .eval_prepare import OUT,LABELS,P,verify

def run(label,estimated_peak_mib=3500):
    if label not in LABELS:raise ValueError('Unregistered new layout label')
    from r2.pilots.bench_parts_convergence_v1 import postprocess as B
    from .eval_prepare import requests
    B.requests=requests
    qs,receipts=B.verify_raw_barrier(OUT,list(LABELS),'full')
    assert len(qs)==48 and len(receipts)==96
    return B.run(OUT,label,list(LABELS),'full',B.OLD.D0,estimated_peak_mib)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',choices=LABELS,required=True)
    parser.add_argument('--estimated-peak-mib',type=float,default=3500);a=parser.parse_args();print(json.dumps(run(a.label,a.estimated_peak_mib)),flush=True)
