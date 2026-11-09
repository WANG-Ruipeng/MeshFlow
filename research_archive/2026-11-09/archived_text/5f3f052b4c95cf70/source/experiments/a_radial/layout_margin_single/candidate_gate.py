"""Apply root's pre-final exploratory compute-allocation rule, without new work."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .prepare import UIDS,read,require
from ..io_utils import file_hash,write_json

POLICY_SHA256='8d7509ed3e69b8b9fd59acf6203e927b5d357f4e901bc0f4a4423149b522dfc6'


def improvement(a,b):
    if a is None or b is None or a.get('status')!='success' or b.get('status')!='success':return None
    x=np.array([m['rms_bbox_percent'] for m in a['metrics']],np.float64)
    y=np.array([m['rms_bbox_percent'] for m in b['metrics']],np.float64)
    require(x.shape==y.shape==(2,) and np.isfinite(x).all() and np.isfinite(y).all() and (y>0).all(),'Invalid fixed two-MC pair')
    require([m['seed'] for m in a['metrics']]==[m['seed'] for m in b['metrics']],'Paired MC seeds differ')
    return (1-x/y).tolist()


def classify(attempts):
    keys=[(r['uid'],r['endpoint']) for r in attempts]
    require(len(keys)==len(set(keys)),'Duplicate endpoint')
    rows=dict(zip(keys,attempts));old=set();new=set();pairwins=[];records=[];geometry=[]
    for uid in UIDS:
        f=rows.get((uid,'fixed'));a=rows.get((uid,'old_final'));b=rows.get((uid,'margin_final'))
        old_gain=improvement(a,f);new_gain=improvement(b,f);paired_gain=improvement(b,a)
        old_success=bool(a and a.get('qualified_original',{}).get('qualified') and old_gain is not None and min(old_gain)>=.02)
        new_success=bool(b and b.get('qualified_original',{}).get('qualified') and new_gain is not None and min(new_gain)>=.02)
        exact=bool(b and b.get('status')=='success' and b.get('audit',{}).get('valid') is True
            and b.get('audit',{}).get('exact_executed') is True and b.get('quality',{}).get('face_count')==512)
        win=paired_gain is not None and min(paired_gain)>=.02
        if old_success:old.add(uid)
        if new_success:new.add(uid)
        if win:pairwins.append(uid)
        geometry.append(exact)
        records.append({'uid':uid,'margin_final_exact_valid_F512':exact,'old_joint_success':old_success,'margin_joint_success':new_success,
            'old_gain_over_Fixed_fraction':old_gain,'margin_gain_over_Fixed_fraction':new_gain,
            'margin_gain_over_old_fraction':paired_gain,'paired_dual_MC_ge2pct_without_quality_condition':win})
    conditions={'all_four_new_finals_exact_valid_and_F512':all(geometry),
        'net_at_least_one_joint_success_and_no_old_success_lost':len(new)>=len(old)+1 and old<=new,
        'at_least_two_new_vs_old_dual_MC_ge2pct':len(pairwins)>=2}
    return {'conditions':conditions,'candidate_go':all(conditions.values()),'old_joint_success_uids':sorted(old),
        'new_joint_success_uids':sorted(new),'lost_old_joint_success_uids':sorted(old-new),
        'paired_dual_MC_ge2pct_uids':pairwins,'per_uid':records,
        'meaning':'predeclared exploratory compute-allocation rule, not significance or paper acceptance',
        'training_authorized_by_this_report':False,'no_new_training_started':True}


def run_gate(run,policy):
    run=Path(run).resolve();policy=Path(policy).resolve();require(file_hash(policy)==POLICY_SHA256,'Pre-final policy changed')
    ledger=read(run/'results.json');require(ledger['planned_slots']==ledger['recorded_slots']==20 and not ledger['unattempted'],'All20 final slots must be attempted')
    value=classify(ledger['all_attempts']);value.update(policy_sha256=POLICY_SHA256,policy_path=str(policy),
        results_sha256=file_hash(run/'results.json'),protocol_sha256=file_hash(run/'RUN_PROTOCOL.json'),source_sha256=file_hash(__file__))
    write_json(run/'EXPLORATORY_JOINT_GATE.json',value);return value


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--run',required=True);a.add_argument('--policy',required=True);args=a.parse_args()
    print(json.dumps(run_gate(args.run,args.policy),indent=2))


if __name__=='__main__':main()
