"""Explicit engineering correction of extra sharp-vertex over-freezing.

The original 30 results remain immutable. Only the 20 first-pass candidates
blocked for insufficient reduction are eligible; the passing source and N-floor
cases are not rerun. All scientific target/admission thresholds stay identical.
"""
from . import common as c
from . import run
import argparse
import copy

BASE=c.OUT


def setup():
    old=c.read(BASE/'protocol.json')
    original=c.read(BASE/'numeric_results.json')
    eligible={r['parent_id'] for r in original['rows'] if r['status']=='BLOCKED_INSUFFICIENT_REDUCTION'}
    c.OUT=BASE/'engineering_v2'
    run.SIMPLIFIER_MODULE='simplifier_v2'
    p=copy.deepcopy(old)
    p.pop('max_candidate_paths_per_parent',None)
    p.update(schema='BENCH_FIXED_C_MULTILOD_V1_ENGINEERING_V2',
        rows=[r for r in old['rows'] if r['parent_id'] in eligible],parents=len(eligible),
        original_protocol_sha256=c.sha(BASE/'protocol.json'),
        original_numeric_results_sha256=c.sha(BASE/'numeric_results.json'),
        engineering_reason='v1 froze every endpoint of sharp free edges and disallowed collapse into a fixed endpoint. 19/21 proposals had all vertices frozen; this extra restriction was not the fixed-C/fidelity task contract.',
        correction='Keep C/interface, boundary and existing-defect vertices fixed. Let QEM and original 30-degree face-normal limits handle free sharp edges. Allow a free endpoint to collapse into an unchanged fixed endpoint. Independent admission thresholds are unchanged.',
        max_new_versions_per_parent=1,max_candidate_paths_this_revision=1,
        total_attempt_policy='At most two recorded deterministic paths for these failed parents; retain original proposals/results. Already-passing v1 parent is not rerun or selected post hoc.',
        source_fidelity_gate_changed=False)
    c.write(c.OUT/'protocol.json',p)
    return p


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','propose','build','render','status'])
    parser.add_argument('--workers',type=int,choices=[1,2],default=2);args=parser.parse_args()
    p=setup()
    if args.stage=='freeze': print(c.json.dumps(dict(parents=p['parents'])))
    elif args.stage in ('propose','build'):run.build(args.workers,proposals_only=args.stage=='propose')
    elif args.stage=='render':
        from .render import main
        main()
    else:run.status()
