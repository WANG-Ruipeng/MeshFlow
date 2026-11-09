"""Unchanged official single D0 / GEO_C / source winding / CPU cascade."""
from __future__ import annotations
import argparse
import json
from . import evaluation as E


def d0(label,estimated_peak_mib=3500):
    if label not in E.NEW_LABELS: raise ValueError('Baseline D0 rerun forbidden')
    E.all_raw_barrier()
    return E.engine().d0(label,estimated_peak_mib)


def cpu(mode,labels=None,workers=2):
    if mode not in ('raw','post') or workers not in (1,2): raise ValueError('Fixed CPU scope')
    E.all_raw_barrier()
    return E.run_cpu(list(E.NEW_LABELS if labels is None else labels),mode,workers)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=('verify','d0','raw','post'))
    p.add_argument('--label',choices=E.NEW_LABELS); p.add_argument('--labels',nargs='+',choices=E.NEW_LABELS)
    p.add_argument('--workers',type=int,choices=(1,2),default=2); p.add_argument('--estimated-peak-mib',type=float,default=3500)
    a=p.parse_args()
    if a.action=='verify': r=dict(status='PASS',RAW=len(E.all_raw_barrier()))
    elif a.action=='d0':
        if not a.label: p.error('--label required')
        r=d0(a.label,a.estimated_peak_mib)
    else: r=cpu(a.action,a.labels,a.workers)
    print(json.dumps({k:v for k,v in r.items() if k!='files'},ensure_ascii=False))


if __name__=='__main__': main()
