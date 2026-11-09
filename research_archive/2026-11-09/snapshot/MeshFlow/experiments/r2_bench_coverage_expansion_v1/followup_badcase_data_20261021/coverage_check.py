"""Read frozen metadata/exposure/review only; no mesh/model/scorer calls."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def writecsv(name, data):
    keys = list(dict.fromkeys(k for r in data for k in r))
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(data)

def feats(r):
    b = r.get('bbox_normalized')
    out = dict(K=float(r['K']), N=float(r['N']), K_over_N=r['K']/r['N'],
               area_fraction=r.get('area_fraction'))
    if b is None or len(b)!=2:
        out['bbox_status']='NA_MISSING'; return out
    for i, ax in enumerate('xyz'):
        out[f'bbox_{ax}_min']=b[0][i]
        out[f'bbox_{ax}_max']=b[1][i]
        out[f'bbox_{ax}_center']=(b[0][i]+b[1][i])/2
        out[f'bbox_{ax}_span']=b[1][i]-b[0][i]
    out['horizontal_max_span']=max(out['bbox_x_span'], out['bbox_z_span'])
    return out

def quantile(vals, q):
    vals=sorted(vals)
    if not vals:return None
    t=(len(vals)-1)*q; a=int(t); b=min(a+1,len(vals)-1)
    return vals[a]*(1-(t-a))+vals[b]*(t-a)

def main():
    inputs=[ROOT/'data/manifest.json', ROOT/'data/condition_exposure.csv',
            ROOT/'training/COVERAGE_SEAM/exposure.jsonl',ROOT/'visual_review/per_cell.csv']
    manifest=read(inputs[0]); allrows=manifest['rows']
    byid={r['condition_id']:r for r in allrows}
    training=[r for r in allrows if r['group']=='TRAIN']
    fs={k:feats(r) for k,r in byid.items()}
    exp=Counter(); augmented=defaultdict(lambda:defaultdict(list))
    for line in inputs[2].read_text(encoding='utf-8').splitlines():
        e=json.loads(line); assert 1<=e['local_step']<=1500
        cid=e['condition_id']; exp[cid]+=1
        for key,v in fs[cid].items():
            if key.startswith('bbox_') and isinstance(v,(int,float)):
                axis='xyz'.index(key.split('_')[1]); v*=e['scale_xyz'][axis]
                augmented[byid[cid]['part_type']][key].append(v)
    assert sum(exp.values())==12000 and len(exp)==321
    exposure={r['condition_id']:r for r in rows(inputs[1])}
    assert all(exp[k]==int(exposure[k]['planned_exposures']) for k in exp)
    reviews=[r for r in rows(inputs[3]) if r['stage']=='CLEAN_FINAL_POST']
    assert len(reviews)==96
    joined=[]
    for r in reviews:
        c=byid[r['condition_id']]
        joined.append(dict(**r, **fs[c['condition_id']], extent_description=c.get('extent_description'),
                           actual_train_exposures=exp[c['condition_id']],
                           comparator_train_exposures=exposure.get(c['condition_id'],{}).get('comparator_exposures',0)))
    writecsv('coverage_final_review_conditions.csv',joined)
    grouped=[]
    for split,part,label in sorted({(r['split'],r['part_type'],r['label']) for r in reviews}):
        subset=[r for r in reviews if (r['split'],r['part_type'],r['label'])==(split,part,label)]
        train=[r for r in training if r['part_type']==part]
        g=dict(split=split,part_type=part,label=label,n_outputs=len(subset),n_eval_conditions=len({r['condition_id'] for r in subset}),
               n_eval_parents=len({r['parent_id'] for r in subset}),train_conditions=len(train),train_parents=len({r['parent_id'] for r in train}),
               actual_train_exposures=sum(exp[r['condition_id']] for r in train),
               comparator_train_exposures=sum(int(exposure[r['condition_id']]['comparator_exposures']) for r in train))
        for field,levels in [('C_use',['YES','NO','UNCERTAIN']),('whole_shape',['COMPLETE','PARTIAL','UNCERTAIN'])]:
            for level in levels:g[f'{field}_{level}']=sum(r[field]==level for r in subset)
        grouped.append(g)
    writecsv('coverage_final_by_part.csv',grouped)
    specs=[('9fd5','SEAT_SURFACE'),('459f','SEAT_SURFACE'),('5dda','SEAT_SURFACE'),('f5d3','BACK_SURFACE'),('1dd7','SEAT_SURFACE'),('2022','SEAT_SUPPORT'),('9d5608','BACK_SURFACE'),('fd355','LEG_SUPPORT'),('459f','SEAT_SUPPORT')]
    comparisons=[]; cases=[]
    for prefix,part in specs:
        hits=[r for r in allrows if r['parent_id'].startswith(prefix) and r['part_type']==part]
        assert len(hits)==1,(prefix,part)
        c=hits[0]; cid=c['condition_id']; train=[r for r in training if r['part_type']==part]
        cases.append(dict(condition_id=cid,group=c['group'],part_type=part,extent_description=c.get('extent_description'),
                          features=fs[cid],actual_train_exposures=exp[cid],comparator_train_exposures=exposure.get(cid,{}).get('comparator_exposures',0),
                          final_reviews=[dict(request_id=r['request_id'],label=r['label'],C_use=r['C_use'],whole_shape=r['whole_shape'],pair_preference=r['pair_preference'],note=r['note']) for r in reviews if r['condition_id']==cid]))
        for feature,value in fs[cid].items():
            if not isinstance(value,(int,float)):continue
            vals=[fs[r['condition_id']].get(feature) for r in train]
            vals=[x for x in vals if isinstance(x,(int,float))]
            if not vals:continue
            weighted=sum(exp[r['condition_id']] for r in train if fs[r['condition_id']].get(feature) is not None and fs[r['condition_id']][feature]<=value)
            wden=sum(exp[r['condition_id']] for r in train if fs[r['condition_id']].get(feature) is not None)
            a=augmented[part].get(feature,[])
            comparisons.append(dict(condition_id=cid,part_type=part,group=c['group'],feature=feature,value=value,
                training_n=len(vals),train_min=min(vals),train_q25=quantile(vals,.25),train_median=quantile(vals,.5),train_q75=quantile(vals,.75),train_max=max(vals),
                marginal_range_status='WITHIN' if min(vals)<=value<=max(vals) else 'BELOW' if value<min(vals) else 'ABOVE',
                train_fraction_le=sum(x<=value for x in vals)/len(vals),exposure_weighted_fraction_le=weighted/wden,
                actual_augmented_min=min(a) if a else None,actual_augmented_max=max(a) if a else None,
                actual_augmented_fraction_le=sum(x<=value for x in a)/len(a) if a else None))
    writecsv('coverage_badcase_ranges.csv',comparisons)
    summary=dict(schema='READ_ONLY_BADCASE_COVERAGE_V1',actual_exposure_rows=12000,train_conditions=321,train_parents=79,
        by_part_final=grouped,badcases=cases,
        limitations=['No new training/generation/scoring or source mesh scan. Existing frozen metadata and actual completed exposure only.',
        'Bbox units are original official-normalized object coordinates, not model units. Ranges are source-condition marginals, not a joint/semantic coverage certificate.',
        'Actual augmentation bbox values are obtained from saved positive XYZ scales; source area fraction is NOT recomputed under anisotropic scaling.',
        'Area fraction denominator sums all original source faces including duplicates. Semantic part labels include different extents.',
        'TRAIN cases occur in their own same-part reference distribution; within-range is not independent generalization evidence.',
        'Two noises per condition; outputs are not independent parents. Existing AI_PROXY nonblind judgements, not human blind labels.',
        'No new thresholds, nearest-neighbour distance metric, significance test, or causal conclusion.'],
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/'coverage_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',by_part_final=grouped,cases=[dict(condition_id=c['condition_id'],features=c['features'],exposures=c['actual_train_exposures']) for c in cases],outside_ranges=[r for r in comparisons if r['marginal_range_status']!='WITHIN']),ensure_ascii=False))

if __name__=='__main__':main()
