"""Summarize every preregistered attempt; no successful-subset ranking."""
import argparse, csv, json, statistics
from pathlib import Path
import numpy as np
from ..io_utils import write_json,file_hash
from .geometry import evaluate_mesh
from .visualization import render_report

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--recheck',action='store_true')
    p.add_argument('--plots',action='store_true');a=p.parse_args();run=Path(a.run)
    protocol=json.loads((run/'RUN_PROTOCOL.json').read_text());rows=json.loads((run/'results.json').read_text())['rows']
    lookup={(r['uid'],r['arm']):r for r in rows}
    missing=[(uid,arm) for uid in protocol['uids'] for arm in protocol['arms'] if (uid,arm) not in lookup]
    failed=[(r['uid'],r['arm'],r.get('error')) for r in rows if r['status']!='success']
    rechecks={}
    if a.recheck:
        for index,uid in enumerate(protocol['uids']):
            with np.load(run/'inputs'/f'{uid}.npz',allow_pickle=False) as z:rv=z['vertices'];rf=z['faces']
            for arm in protocol['arms']:
                row=lookup.get((uid,arm))
                if not row or row['status']!='success':continue
                dest=run/'rechecks'/uid/(arm+'.json')
                if dest.exists():check=json.loads(dest.read_text())
                else:
                    with np.load(run/row['mesh_file'],allow_pickle=False) as z:v=z['vertices'];f=z['faces']
                    check=evaluate_mesh(v,f,rv,rf,points=8192,seed=protocol['evaluation']['recheck_seed_base']+index)
                    write_json(dest,check)
                rechecks[(uid,arm)]=check
    summary=[];per_sample=[]
    for arm in protocol['arms']:
        rr=[r for r in rows if r['arm']==arm];ok=[r for r in rr if r['status']=='success']
        ratios=[];qual_ratios=[];targets=[];targets_alt=[];targets_both=[];times=[];changes=[];first_times=[];ratio_seed_stable=[]
        for r in ok:
            fixed=lookup.get((r['uid'],'fixed'))
            if not fixed or fixed['status']!='success':continue
            base=fixed['metrics']['rms_bbox_percent'];rms=r['metrics']['rms_bbox_percent']
            ratio=rms/base if base>1e-12 else None
            if ratio is not None:ratios.append(ratio)
            if r['qualified'] and ratio is not None:qual_ratios.append(ratio)
            met=r['qualified'] and rms<=.9*base
            targets.append(met);times.append(r['algorithm_seconds'])
            if rechecks:
                alt=rechecks.get((r['uid'],arm));altfixed=rechecks.get((r['uid'],'fixed'))
                if alt and altfixed:
                    met_alt=bool(r['qualified'] and alt['rms_bbox_percent']<=.9*altfixed['rms_bbox_percent'])
                    targets_alt.append(met_alt);targets_both.append(bool(met and met_alt))
                    changes.append(abs(alt['rms_bbox_percent']-rms))
                    ratio_seed_stable.append((rms<base)==(alt['rms_bbox_percent']<altfixed['rms_bbox_percent']))
            trajectory=list(r.get('trajectory',[]))+[{'elapsed_seconds':r['algorithm_seconds'],'qualified':r['qualified'],'metrics':r['metrics']}]
            tt=[t['elapsed_seconds'] for t in trajectory if t.get('qualified') and t['metrics']['rms_bbox_percent']<=.9*base]
            if tt:first_times.append(min(tt))
            per_sample.append({'uid':r['uid'],'arm':arm,'status':r['status'],'error':None,'rms_bbox_percent':rms,'rms_over_fixed':ratio,
                'q05_deg':r['quality']['angle_q05_deg'],'min_angle_deg':r['quality']['min_angle_deg'],
                'actual_faces':r['actual_faces'],'valid':r['audit']['valid'],'quality_pass':r['quality_pass'],
                'budget_match':r['budget_match'],'qualified':r['qualified'],'target_reached':met,
                'algorithm_seconds':r['algorithm_seconds'],'target_first_sampled_seconds':min(tt) if tt else None})
        med=lambda x:float(np.median(x)) if x else None
        summary.append({'arm':arm,'expected_attempts':len(protocol['uids']),'completed_attempts':len(rr),
          'execution_success':len(ok),'execution_errors':len(rr)-len(ok),
          'valid_count':sum(bool(r['audit']['valid']) for r in ok),
          'budget_match_count':sum(r['budget_match'] for r in ok),
          'quality_pass_count':sum(r['quality_pass'] for r in ok),
          'qualified_count':sum(r['qualified'] for r in ok),
          'target_count':sum(targets),'target_count_second_seed':sum(targets_alt),'target_count_both_seeds':sum(targets_both),'median_rms_bbox_percent':med([r['metrics']['rms_bbox_percent'] for r in ok]),
          'median_rms_over_fixed_all_readable_NOT_equal_budget_ranking':med(ratios),
          'rms_improved_count':sum(x<1 for x in ratios),
          'median_q05_deg':med([r['quality']['angle_q05_deg'] for r in ok]),
          'median_algorithm_seconds':med(times),
          'actual_faces_min':min([r['actual_faces'] for r in ok],default=None),
          'actual_faces_max':max([r['actual_faces'] for r in ok],default=None),
          'recheck_rms_absolute_difference_median':med(changes),
          'recheck_improvement_sign_consistent_count':sum(ratio_seed_stable),
          'target_time_median_among_reached_only':med(first_times)})
    for failure in rows:
        if failure['status']!='success':
            per_sample.append({**({k:None for k in per_sample[0]} if per_sample else {}),'uid':failure['uid'],'arm':failure['arm'],'status':failure['status'],'error':failure.get('error'),'qualified':False})
    result={'experiment':protocol['experiment'],'expected_attempts':96,'recorded_attempts':len(rows),
      'missing':missing,'execution_failures':failed,'summary':summary,'per_sample':per_sample,
      'qualification':'Seen16 fixed-field diagnostic only; no learned-weight update, no heldout or MeshFlow superiority claim',
      'limitations':['Finite local optimizer, not global bound','Fixed-relative min/q05 floors are a strict diagnostic condition, not an application standard',
        'A is original frozen teacher-path network, not trained for these quality constraints',
        'CGAL6.1.1 classic-plane with custom guarded collapse; global q05 endpoint-only; queue exhaustion does not prove infeasibility',
        'Reverse optimization quadrature approximates surface integral; independent RMS takes priority',
        'Single trajectory timing; cold/runtime implementation overhead included; platform performance not asymptotic algorithm claim']}
    write_json(run/'SUMMARY.json',result,replace=(run/'SUMMARY.json').exists())
    with (run/'PER_SAMPLE.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_sample[0]) if per_sample else ['uid']);writer.writeheader();writer.writerows(per_sample)
    if a.plots:
        print(json.dumps(render_report(run),ensure_ascii=False,default=str))
    print(json.dumps({'summary':summary,'missing':len(missing),'errors':len(failed)},ensure_ascii=False))
if __name__=='__main__':main()




