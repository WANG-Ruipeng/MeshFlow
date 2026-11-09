"""Supplement base-C audit with recorded augmentation ranges and pair near-duplicate checks.

CPU arrays only. This does not change the frozen input-only matching.
"""
from pathlib import Path
import csv,json,hashlib,math,time
from collections import defaultdict,Counter
import numpy as np
from scipy.spatial import cKDTree
from coverage_audit import ROOT,OLD,OUT,L,FEATURES,sha
DEST=OUT/'supplement_v1'
TOL=1e-12

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(name,rs):
    with (DEST/name).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rs for k in r)),lineterminator='\n');w.writeheader();w.writerows(rs)
def jdump(name,obj):
    with (DEST/name).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def canon(t):
    return np.array([x[np.lexsort((x[:,2],x[:,1],x[:,0]))] for x in t]).reshape(len(t),9)
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    definition=dict(schema='actual_augmentation_and_input_similarity_v1',fixed_L=L,range_tolerance=TOL,
        tolerance_reason='dimensionless arithmetic-edge tolerance; base audit used literal comparisons and is preserved. This is not geometry acceptance slack.',
        coverage='min/max of actual logged augmentation values applied to fixed C for each arm and type, no new random draws; compare DEV to marginal ranges only.',
        similarity='Read source C and M only for all eight selected pairs. Identity, X reflection, Z reflection, XZ reflection around model origin; no optimized alignment. Report symmetric vertex/face distances; descriptive, not identity adjudication.',
        pairing='Unchanged frozen base input assignment. Similarity follow-up cannot exclude or replace a pair.',
        zero_model_forward=True,zero_generated_geometry_read=True)
    jdump('definition.json',definition)
    geom=rows(OUT/'condition_geometry.csv');byid={r['condition_id']:r for r in geom};pairs=rows(OUT/'selected_train_dev_pairs.csv')
    exposures=rows(OLD/'training_analysis/final/exposures.csv');data={};mins={};maxs={};counter=Counter()
    for r in geom:
        if r['split']!='TRAIN':continue
        with np.load(r['C_source'],allow_pickle=False) as z:c=z['C'].astype(float)
        cr=np.cross(c[:,1]-c[:,0],c[:,2]-c[:,0]);a=np.linalg.norm(cr,axis=1)/2;n=cr/(2*a[:,None]);data[r['condition_id']]=(a,n)
    for e in exposures:
        cid=e['task_id'];r=byid[cid];a,n=data[cid];alpha=float(e['alpha']);key=(e['arm'],cid)
        if key not in mins:mins[key]=np.full(len(FEATURES),np.inf);maxs[key]=np.full(len(FEATURES),-np.inf)
        factor=np.sqrt(n[:,0]**2+alpha**2*(n[:,1]**2+n[:,2]**2));area=a*factor
        normal_y=np.clip(np.abs(alpha*n[:,1]/factor),0.,1.)
        tilt=float(np.average(np.arccos(normal_y),weights=area))/(np.pi/2)
        v=np.array([float(r[k]) for k in FEATURES]);v[0]*=alpha;v[3]*=alpha;v[6]=math.sqrt(float(area.sum()))/L;v[7]=tilt
        mins[key]=np.minimum(mins[key],v);maxs[key]=np.maximum(maxs[key],v);counter[key]+=1
    rr=[]
    for key in sorted(mins):
        arm,cid=key
        for j,f in enumerate(FEATURES):rr.append(dict(arm=arm,condition_id=cid,condition_type=byid[cid]['condition_type'],feature=f,actual_min=float(mins[key][j]),actual_max=float(maxs[key][j]),completed_exposures=counter[key]))
    dump('actual_exposure_feature_ranges_per_condition.csv',rr)
    coverage=[]
    for d in [r for r in geom if r['split']=='DEV']:
        for arm in ('USER_SMALL','USER_EXPANDED_67'):
            ids=[key for key in mins if key[0]==arm and byid[key[1]]['condition_type']==d['condition_type']]
            low=np.min([mins[k] for k in ids],axis=0);high=np.max([maxs[k] for k in ids],axis=0)
            for j,f in enumerate(FEATURES):
                x=float(d[f]);coverage.append(dict(arm=arm,dev_condition_id=d['condition_id'],condition_type=d['condition_type'],feature=f,value=x,actual_train_min=float(low[j]),actual_train_max=float(high[j]),within_range_tol=bool(low[j]-TOL<=x<=high[j]+TOL),below_by=max(0.,float(low[j]-x)),above_by=max(0.,float(x-high[j]))))
    dump('dev_actual_training_range_coverage.csv',coverage)
    base=rows(OUT/'dev_train_feature_ranges.csv');countbase={};countactual={}
    for arm in ('USER_SMALL','USER_EXPANDED_67'):
        countbase[arm]=len({r['condition_id'] for r in base if r['pool']==arm and not float(r['train_min'])-TOL<=float(r['value'])<=float(r['train_max'])+TOL})
        countactual[arm]=len({r['dev_condition_id'] for r in coverage if r['arm']==arm and not r['within_range_tol']})
    checks=[]
    for pair in pairs:
        dr=byid[pair['dev_condition_id']];tr=byid[pair['train_condition_id']]
        for domain in ('C','M'):
            key='C' if domain=='C' else 'full_target';paths=[r['C_source'] if domain=='C' else r['parent_source'] for r in (dr,tr)]
            arrays=[]
            for p in paths:
                with np.load(p,allow_pickle=False) as z:arrays.append(z[key].reshape(-1,3,3).astype(np.float64))
            dt,tt=arrays
            for name,sign in [('IDENTITY',[1,1,1]),('MIRROR_X',[-1,1,1]),('MIRROR_Z',[1,1,-1]),('MIRROR_XZ',[-1,1,-1])]:
                transform=tt*np.array(sign);dv=dt.reshape(-1,3);tv=transform.reshape(-1,3)
                d1=cKDTree(tv).query(dv)[0]/L;d2=cKDTree(dv).query(tv)[0]/L
                dc=canon(dt);tc=canon(transform);f1=cKDTree(tc).query(dc)[0]/(3*L);f2=cKDTree(dc).query(tc)[0]/(3*L)
                checks.append(dict(pair_id=pair['pair_id'],domain=domain,transform=name,dev_parent=dr['parent_id'],train_parent=tr['parent_id'],dev_group=dr['group_uid'],train_group=tr['group_uid'],known_groups_equal=dr['group_uid']==tr['group_uid'],dev_faces=len(dt),train_faces=len(tt),vertex_sym_mean_L=float((d1.mean()+d2.mean())/2),vertex_sym_p95_L=float(max(np.quantile(d1,.95),np.quantile(d2,.95))),vertex_sym_max_L=float(max(d1.max(),d2.max())),face_corner_rms_sym_mean_L=float((f1.mean()+f2.mean())/2),face_corner_rms_sym_max_L=float(max(f1.max(),f2.max())),dev_faces_with_nearest_rms_below_1e_5_L=int((f1<1e-5).sum()),train_faces_with_nearest_rms_below_1e_5_L=int((f2<1e-5).sum())))
    dump('selected_pair_source_geometry_similarity.csv',checks)
    best=[]
    for pair in pairs:
        for domain in ('C','M'):
            x=min([r for r in checks if r['pair_id']==pair['pair_id'] and r['domain']==domain],key=lambda r:r['face_corner_rms_sym_mean_L']);best.append(x)
    dump('selected_pair_best_fixed_reflection.csv',best)
    jdump('summary.json',dict(base_C_outside_with_tolerance=countbase,actual_exposure_outside_with_tolerance=countactual,all_scales_fixed=True,pairing_unchanged=True,near_pair04=[r for r in best if r['pair_id']=='PAIR04'],source_data_hashes_unchanged=all(sha(p)==h for p,h in json.loads((OUT/'source_hashes.json').read_text()).items())))
    text=['# Supplement: actual augmentation coverage and source similarity','',f'Numerical range comparisons allow only {TOL} in dimensionless features. The base audit literal-comparison tables are preserved.',f'Base C outside one or more same-type marginal ranges: {countbase}. Actual logged training augmentations: {countactual}. These are marginal range diagnostics, not IID tests.',
        '', 'The original assignment is unchanged. All eight pairs were checked on source C and M under four fixed reflections; no generated geometry or quality was read. Review PAIR04 separately: similar or duplicate source geometry can weaken any claim of independent new-shape generalization. Known parent/group IDs are metadata, not a complete near-duplicate detector.']
    (DEST/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    jdump('completion.json',dict(status='COMPLETE',script_sha256=sha(__file__),output_hashes={p.name:sha(p) for p in sorted(DEST.iterdir()) if p.is_file()}))
    print(json.dumps(dict(base=countbase,actual=countactual,best04=[r for r in best if r['pair_id']=='PAIR04']),indent=2))
if __name__=='__main__':main()
