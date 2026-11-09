"""Read only completed scores; retain all fields and paired quality costs."""
from __future__ import annotations
import collections,json
import numpy as np
from . import run as R
C=R.C
STAGES=('OLD_BASE','OLD_FINAL','GUARD_GEO','FINAL')
def number(x):return isinstance(x,(int,float,bool)) and np.isfinite(float(x))

def aggregate(root=R.ROOT):
    root=C.resolve(root);reg=C.read(root/'registration.json');complete=C.read(root/'completion.json');R.E.verify(complete['files'])
    rows=[];ledgers=[];faces=[];sources={};output=root/'tables'
    for job in reg['jobs']:
        d=root/'cells'/job['panel']/job['q']['request_id'];r=C.read(d/'complete.json');R.E.verify(r['files']);sources[str(d/'complete.json')]=C.sha(d/'complete.json')
        for row in C.read(d/'metric_rows.json'):
            row.update(part_type=job['q'].get('part_type','BACK_FULL' if 'BACK' in job['panel'] else 'SEAT_FULL'),closed_C=job['q']['boundary_edges']==0)
            rows.append(row)
        for stage,p in (('GUARD_GEO',d/'GUARD_GEO/old_base_paired_ledger.json'),('FINAL',d/'old_base_final_paired_ledger.json')):
            ledgers.append(dict(panel=job['panel'],request_id=job['q']['request_id'],parent_id=job['q']['parent_id'],split=job['q']['split'],stage=stage,**C.read(p)['summary']))
        for f in C.read(d/'floor_faces.json'):
            faces.append(dict(panel=job['panel'],request_id=job['q']['request_id'],parent_id=job['q']['parent_id'],split=job['q']['split'],**{k:v for k,v in f.items() if k!='quadratics'}))
    meta=set(R.E.META)|{'label','stage','panel','part_type','old_final_chain','closed_C','C_boundary_edges'}
    original=set(C.read(C.ROOT/'r2/tools/hinge_opposite_v1/metric_policy.json')['directions'])
    fields=sorted(original|{k for r in rows for k,v in r.items() if k not in meta and number(v)})
    missing=[dict(panel=r['panel'],stage=r['stage'],request_id=r['request_id'],fields=sorted(original-set(r))) for r in rows if original-set(r)]
    if missing:raise AssertionError(missing)
    summary=[];parent=[];paired=[];common=[]
    for panel in sorted({r['panel'] for r in rows}):
        for stage in STAGES:
            rr=[r for r in rows if r['panel']==panel and r['stage']==stage]
            for grouping,groups in (('split',['ALL',*sorted({r['split'] for r in rr})]),('part_type',sorted({r['part_type'] for r in rr}))):
                for group in groups:
                    xx=[r for r in rr if group=='ALL' or r[grouping]==group]
                    for field in fields:
                        byc=collections.defaultdict(list)
                        for r in xx:
                            if number(r.get(field)):byc[r['parent_id'],r['condition_id']].append(float(r[field]))
                        byp=collections.defaultdict(list)
                        for (pid,cid),vv in byc.items():byp[pid].append(float(np.mean(vv)))
                        pv={pid:float(np.mean(vv)) for pid,vv in byp.items()};vv=[float(r[field]) for r in xx if number(r.get(field))]
                        summary.append(dict(panel=panel,stage=stage,grouping=grouping,group=group,metric=field,
                            mean_parent_equal=float(np.mean(list(pv.values()))) if pv else None,mean_output=float(np.mean(vv)) if vv else None,
                            n_requests=len(xx),n_valid=len(vv),n_valid_parents=len(pv),closed_C_requests=sum(r['closed_C'] for r in xx)))
                        if grouping=='split' and group!='ALL':
                            parent.extend(dict(panel=panel,stage=stage,split=group,parent_id=pid,metric=field,value=value) for pid,value in pv.items())
    lookup={(r['panel'],r['request_id'],r['stage']):r for r in rows}
    for job in reg['jobs']:
        q=job['q']
        for a,b in (('OLD_BASE','GUARD_GEO'),('OLD_BASE','FINAL'),('OLD_FINAL','FINAL'),('GUARD_GEO','FINAL')):
            x,y=lookup[job['panel'],q['request_id'],a],lookup[job['panel'],q['request_id'],b]
            for field in fields:
                av,bv=x.get(field),y.get(field)
                paired.append(dict(panel=job['panel'],request_id=q['request_id'],condition_id=q['condition_id'],parent_id=q['parent_id'],split=q['split'],
                    baseline=a,candidate=b,metric=field,baseline_value=av,candidate_value=bv,delta=float(bv)-float(av) if number(av) and number(bv) else None,
                    status='PASS' if number(av) and number(bv) else 'NA_NO_COMMON_VALUE'))
        if job['panel']=='BENCH_SEAM_S15000':
            for stage in STAGES:
                x,y=lookup[job['panel'],q['request_id'],stage],lookup['BENCH_SEAM_S21000',q['request_id'],stage]
                for field in fields:
                    av,bv=x.get(field),y.get(field);common.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],split=q['split'],stage=stage,metric=field,
                        step15000=av,step21000=bv,delta=float(bv)-float(av) if number(av) and number(bv) else None))
    checks=[]
    for job in reg['jobs']:
        q=job['q'];d=root/'cells'/job['panel']/q['request_id'];fr=C.read(d/'floor_faces.json');g=C.read(d/'geo.json')
        old=lookup[job['panel'],q['request_id'],'OLD_BASE'];new=lookup[job['panel'],q['request_id'],'FINAL']
        checks.append(dict(panel=job['panel'],request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],C_boundary_edges=q['boundary_edges'],
            floor_triggered_faces=sum(x['floor_triggered'] for x in fr),anti_protected_faces=sum(x['anti_collapse_protected'] for x in fr),
            anti_rejected_assignments=g['stats']['anti_collapse_rejections'],protected_deleted_faces=g['stats']['protected_deleted_faces'],
            old_deleted_free=old['deleted_free_faces'],new_deleted_free=new['deleted_free_faces'],new_bridge_faces=new.get('new_bridge_faces'),
            retained_free_change=new['actual_free_faces']-old['actual_free_faces'],C_preserved_exact=new['C_preserved_exact'],
            old_bulk_reach=old['free_only_reached_bulk_area_fraction'],new_bulk_reach=new['free_only_reached_bulk_area_fraction'],
            old_bulk_ge30=old['bulk_reach_ge_0_30_descriptive'],new_bulk_ge30=new['bulk_reach_ge_0_30_descriptive']))
    outputs={}
    for name,value in (('per_output',rows),('summary_parent_equal',summary),('per_parent',parent),('paired_per_output',paired),('bench_same_request_step_pairing',common),('connectivity_ledger',ledgers),('floor_per_face',faces),('request_guard_counts',checks)):
        p=output/(name+'.csv');C.csvwrite(p,value);outputs[str(p)]=C.sha(p)
    rec=dict(status='COMPLETE',rows=len(rows),requests=len(checks),fields=fields,n_fields=len(fields),all_original135_present=True,
        original179_extra_policy='Old Bench/base original135 is retained; bridge/fixed-parent extension absent there is NA, not invented or rescored',
        units='All original fractions remain0..1; count and model-length fields unscaled; display percentages multiply fractions by100',
        applicability='Closed C seam/boundary metrics NA; bulk≥30 denominator includes only applicable nonempty BULK values',
        aggregation='Noise -> condition -> parent equal; mixed category/endpoint panels are not pooled into one causal score',
        ledger_scope='OLD_BASE to GUARD/FINAL raw-slot lineage; BRIDGE uses separate negative identity, never original face lineage',
        comparisons='Guard pair isolates both guards together. FINAL includes HINGE/ALPHA/HOLE; Bench OLD_FINAL previously HINGE only. Chair OLD_FINAL is matched full chain.',
        source_receipts=sources,files=outputs,code_sha256=C.sha(__file__),metric_calls=0,model_calls=0)
    C.write(output/'completion.json',rec);print(json.dumps({k:v for k,v in rec.items() if k not in ('fields','files','source_receipts')}));return rec
if __name__=='__main__':aggregate()
