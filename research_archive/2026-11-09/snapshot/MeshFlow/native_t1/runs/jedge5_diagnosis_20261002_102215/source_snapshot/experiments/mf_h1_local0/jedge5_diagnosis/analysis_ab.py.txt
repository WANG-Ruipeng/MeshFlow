"""Read saved endpoint scalars/arrays; never call a network or geometry scorer.

All differences are formed within a registered parent/patch/seed before any
aggregation. Coordinate differences use unchanged slots and are numerical, not
semantic correspondences. There is no model selection or significance test.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import numpy as np

DESCRIPTIVE = frozenset(('near_C_area_fraction','near_C_boundary_band_area_fraction',
    'near_C_outside_boundary_band_area_fraction','C_surface_area_fraction',
    'exact_C_free_shared_edge_count','exact_C_free_shared_vertex_count',
    'C_inherited_nonmanifold_vertex_count','C_current_winding_conflict_count',
    'shared_edge_component_count','free_total_surface_area'))
HIGHER = frozenset(('fixed_R12_coverage_within_1pct_fraction','boundary_coverage_fraction','free_angle_p05_degrees'))
SETTINGS=(('bf16',50),('bf16',100),('fp32',50),('fp32',100))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def _finite(x):
    return isinstance(x,(int,float,np.number)) and not isinstance(x,(bool,np.bool_)) and np.isfinite(x)


def _direction(name):
    if name in DESCRIPTIVE or ('area' in name.lower() and 'degenerate' not in name.lower() and 'zero' not in name.lower()):
        return 'descriptive_only'
    if name in HIGHER or ('coverage' in name and name.endswith('_fraction')):
        return 'higher_is_better'
    return 'lower_is_better'


def _record(path):
    path=Path(path);return dict(path=str(path),bytes=path.stat().st_size,sha256=_sha(path))


def _read_scores(out,name):
    path=Path(out)/'scores'/name/'results.json';value=json.loads(path.read_text())
    if value.get('status')!='PASS' or not value.get('rows'):
        raise RuntimeError('Analysis requires a complete scored phase: '+name)
    if any('scalars' not in x for x in value['rows']):
        raise ValueError('Saved score row lacks original scalars')
    return value['rows'],_record(path)


def _case_id(row):
    return f"P{row['parent']}_patch{row['patch']}_seed{row['seed']}"


def _endpoint_id(row):
    model=row.get('model');src=row.get('source_model');cut=row.get('start_step')
    prefix=f'{src}_to_{model}_k{cut}' if src is not None else str(model)
    return prefix+'_'+_case_id(row)+f"_{row['precision']}_Euler{row['nfe']}"


def _identity_A(row):
    return row['model'],int(row['parent']),int(row['patch']),int(row['seed']),row['precision'],int(row['nfe'])


def _index_A(rows,seeds):
    expected={(m,p,k,int(seed),precision,nfe) for m in ('B','J') for p in (0,1) for k in range(5)
              for seed in seeds for precision,nfe in SETTINGS}
    expected|={(m,p,k,int(seeds[0]),'fp32',200) for m in ('B','J') for p in (0,1) for k in (0,4)}
    index={_identity_A(x):x for x in rows}
    if len(rows)!=168 or len(index)!=168 or set(index)!=expected:
        raise ValueError('A scored matrix differs from all168 registered endpoints')
    return index


class RawCache:
    def __init__(self,out,cases):
        self.out=Path(out);self.cases=cases;self.cache={};self.records=[]
    def get(self,row):
        relative=row.get('path') or row.get('raw_path');path=Path(relative)
        if not path.is_absolute():path=self.out/path
        key=str(path);expected=row.get('sha256') or row.get('raw_sha256')
        if key not in self.cache:
            if not expected or _sha(path)!=expected:
                raise ValueError('Coordinate comparison archive identity changed')
            with np.load(path,allow_pickle=False) as z:raw=np.asarray(z['raw']).copy()
            if raw.dtype!=np.float32 or raw.size!=112*9 or not np.isfinite(raw).all():
                raise ValueError('Finite unchanged FP32 raw coordinates required')
            raw=raw.reshape(112,9)
            C=np.asarray(self.cases[int(row['parent']),int(row['patch'])]['C']).reshape(12,9)
            if raw[:12].tobytes()!=C.tobytes():
                raise ValueError('Coordinate comparison did not preserve registered C')
            self.cache[key]=(raw,expected);self.records.append(dict(path=relative,sha256=expected))
        elif self.cache[key][1]!=expected:
            raise ValueError('One coordinate archive has inconsistent row hashes')
        return self.cache[key][0]


def coordinate_change(arrays,coefficients,L):
    """FP64 coordinate arithmetic only, with free-slot and all-slot summaries."""
    if not np.isfinite(L) or L<=0 or len(arrays)!=len(coefficients):
        raise ValueError('Coordinate comparison requires fixed positive scoring bbox')
    diff=np.zeros((112,9),dtype=np.float64)
    for x,c in zip(arrays,coefficients):diff+=float(c)*np.asarray(x,dtype=np.float64).reshape(112,9)
    result={}
    for label,x in (('free',diff[12:]),('all',diff)):
        rms=float(np.sqrt(np.mean(x*x)));vertex=float(np.sqrt(np.mean(np.sum(x.reshape(-1,3)**2,axis=1))))
        result.update({label+'_max_abs':float(np.max(np.abs(x))),label+'_coordinate_RMS':rms,
            label+'_coordinate_RMS_bbox_fraction':rms/L,label+'_coordinate_RMS_bbox_pct':100*rms/L,
            label+'_vertex_RMS':vertex,label+'_vertex_RMS_bbox_pct':100*vertex/L})
    result['known_max_abs']=float(np.max(np.abs(diff[:12])))
    return result


def _linear_scalars(endpoints,coefficients):
    keys=sorted(set().union(*(x['scalars'] for x in endpoints)));values={}
    for key in keys:
        v=[x['scalars'].get(key) for x in endpoints]
        values[key]=float(sum(c*float(q) for c,q in zip(coefficients,v))) if all(_finite(q) for q in v) else None
    return values


def _pair(family,comparison,endpoints,coefficients,cache,cases,*,meaning):
    first=endpoints[0];case=(int(first['parent']),int(first['patch']),int(first['seed']))
    if any((int(x['parent']),int(x['patch']),int(x['seed']))!=case for x in endpoints):
        raise ValueError('Pair comparisons must share parent, patch and Gaussian seed')
    L=float(cases[case[0],case[1]]['bbox'])
    coordinates=coordinate_change([cache.get(x) for x in endpoints],coefficients,L)
    return dict(family=family,comparison=comparison,parent=case[0],patch=case[1],seed=case[2],case_id=_case_id(first),
        context_group='training_C' if case[1]==0 else 'other_C',meaning=meaning,
        endpoint_ids=[_endpoint_id(x) for x in endpoints],coefficients=list(coefficients),
        endpoint_scalar_values=[x['scalars'] for x in endpoints],metric_delta=_linear_scalars(endpoints,coefficients),
        coordinate_difference=coordinates,bbox=L)


def _stats(values,total,*,direction):
    a=np.asarray([float(x) for x in values if _finite(x)],dtype=np.float64)
    if not len(a):
        return dict(n=0,missing=total,direction=direction,mean=None,median=None,minimum=None,maximum=None,
                    positive=0,negative=0,zero=0,favorable_count=None,worse_count=None)
    pos=int((a>0).sum());neg=int((a<0).sum());zero=int((a==0).sum())
    return dict(n=len(a),missing=total-len(a),direction=direction,mean=float(a.mean()),median=float(np.median(a)),
        minimum=float(a.min()),maximum=float(a.max()),positive=pos,negative=neg,zero=zero,
        favorable_count=None if direction=='descriptive_only' else (pos if direction=='higher_is_better' else neg),
        worse_count=None if direction=='descriptive_only' else (neg if direction=='higher_is_better' else pos))


def _groups(rows):
    groups={'all':rows}
    for p in (0,1):groups['parent_'+str(p)]=[r for r in rows if r['parent']==p]
    for k in range(5):groups['patch_'+str(k)]=[r for r in rows if r['patch']==k]
    for name in ('training_C','other_C'):groups[name]=[r for r in rows if r['context_group']==name]
    for seed in sorted({r['seed'] for r in rows}):groups['seed_'+str(seed)]=[r for r in rows if r['seed']==seed]
    for p in (0,1):
        for k in range(5):groups[f'parent_{p}_patch_{k}']=[r for r in rows if r['parent']==p and r['patch']==k]
    return {key:value for key,value in groups.items() if value}


def summarize_pairs(rows):
    out={}
    for comparison in sorted({r['comparison'] for r in rows}):
        selected=[r for r in rows if r['comparison']==comparison];groups={}
        for group,subset in _groups(selected).items():
            metrics=sorted(set().union(*(x['metric_delta'] for x in subset)))
            coords=sorted(set().union(*(x['coordinate_difference'] for x in subset)))
            groups[group]=dict(cases=len(subset),metrics={key:_stats([x['metric_delta'].get(key) for x in subset],len(subset),direction=_direction(key)) for key in metrics},
                coordinates={key:_stats([x['coordinate_difference'][key] for x in subset],len(subset),direction='descriptive_only') for key in coords})
        out[comparison]=dict(family=selected[0]['family'],meaning=selected[0]['meaning'],groups=groups)
    return out


def _flatten(value,prefix=''):
    row={}
    for key,item in value.items():
        name=prefix+str(key)
        if isinstance(item,dict):row.update(_flatten(item,name+'.'))
        elif isinstance(item,list):row[name]=json.dumps(item,ensure_ascii=False,allow_nan=False)
        else:row[name]=item
    return row


def _csv(path,rows):
    flat=[_flatten(x) for x in rows];fields=sorted(set().union(*(x.keys() for x in flat)))
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    return _record(path)


def _json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def _endpoint_rows(rows):
    return [dict(endpoint_id=_endpoint_id(x),model=x['model'],parent=x['parent'],patch=x['patch'],seed=x['seed'],
                 precision=x['precision'],nfe=x['nfe'],context_group='training_C' if x['patch']==0 else 'other_C',
                 source_model=x.get('source_model'),continuation_model=x.get('continuation_model'),
                 start_step=x.get('start_step'),path=x.get('path') or x.get('raw_path'),scalars=x['scalars']) for x in rows]


def _shared(out):
    # This only validates/retrieves stored assets; no model loader is invoked.
    from .common import load_cases,registered_seeds
    cases=load_cases();seeds=list(registered_seeds(out))
    if len(seeds)!=2:
        raise ValueError('Exactly two registered generation seeds required')
    rows_B,manifest_B=_read_scores(out,'A_B');rows_J,manifest_J=_read_scores(out,'A_J')
    rows=rows_B+rows_J;index=_index_A(rows,seeds)
    return cases,seeds,rows,index,[manifest_B,manifest_J]


def _conventions():
    return dict(delta='Linear combination recorded per case, before mean/median; no difference of group medians',
        favorable='Sign relative to metric direction; DID favorable means a relative J-minus-B gap shift, not every endpoint improved',
        area='Descriptive only; overlapping triangle areas are counted by the original scorer',
        coordinate_RMS='sqrt(mean squared scalar coordinate difference); free slots12:112 primary',
        vertex_RMS='sqrt(mean squared 3D point difference), also provided explicitly',
        correspondence='Same numerical slots only; cross-model slots are not verified semantic parts',
        convergence='Euler200 is a finer numerical reference for only four cases, not exact ODE truth',
        uncertainty='Two noises and two already observed parents; no significance, success-probability or new-object inference')


def analyze_A(out):
    out=Path(out);folder=out/'analysis'/'A';folder.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    cases,seeds,allrows,index,sources=_shared(out);cache=RawCache(out,cases);pairs=[]
    def get(m,p,k,s,precision,nfe):return index[m,p,k,s,precision,nfe]
    def add(fam,name,ends,coef,meaning):pairs.append(_pair(fam,name,ends,coef,cache,cases,meaning=meaning))
    for p in (0,1):
        for k in range(5):
            for seed in seeds:
                for m in ('B','J'):
                    for precision in ('bf16','fp32'):
                        add('step_refinement',f'{m}_{precision}_100_minus_50',[get(m,p,k,seed,precision,100),get(m,p,k,seed,precision,50)],[1,-1],'Same-model Euler100 minus Euler50')
                    for nfe in (50,100):
                        add('precision',f'{m}_Euler{nfe}_FP32_minus_BF16',[get(m,p,k,seed,'fp32',nfe),get(m,p,k,seed,'bf16',nfe)],[1,-1],'Same-model FP32 minus BF16 at equal NFE')
                for precision,nfe in SETTINGS:
                    add('model_gap',f'J_minus_B_{precision}_Euler{nfe}',[get('J',p,k,seed,precision,nfe),get('B',p,k,seed,precision,nfe)],[1,-1],'J minus B with equal precision/NFE')
                for precision in ('bf16','fp32'):
                    add('step_gap_difference_of_differences',f'J_B_gap_Euler100_minus50_{precision}',
                        [get('J',p,k,seed,precision,100),get('B',p,k,seed,precision,100),get('J',p,k,seed,precision,50),get('B',p,k,seed,precision,50)],
                        [1,-1,-1,1],'[E(J,100)-E(B,100)]-[E(J,50)-E(B,50)]; per-case gap sensitivity')
                for nfe in (50,100):
                    add('precision_gap_difference_of_differences',f'J_B_gap_FP32_minusBF16_Euler{nfe}',
                        [get('J',p,k,seed,'fp32',nfe),get('B',p,k,seed,'fp32',nfe),get('J',p,k,seed,'bf16',nfe),get('B',p,k,seed,'bf16',nfe)],
                        [1,-1,-1,1],'[E(J,FP32)-E(B,FP32)]-[E(J,BF16)-E(B,BF16)]; per-case gap sensitivity')
    for m in ('B','J'):
        for p in (0,1):
            for k in (0,4):
                add('four_case_reference',f'{m}_FP32_200_minus100',[get(m,p,k,seeds[0],'fp32',200),get(m,p,k,seeds[0],'fp32',100)],
                    [1,-1],'Fixed four-case FP32 Euler200 minus100; no convergence guarantee')
    if len(pairs)!=328:raise AssertionError('A comparison matrix size changed')
    convergence=[]
    pair_index={(x['comparison'],x['parent'],x['patch'],x['seed']):x for x in pairs}
    for m in ('B','J'):
        for p in (0,1):
            for k in (0,4):
                first=pair_index[f'{m}_fp32_100_minus_50',p,k,seeds[0]]
                second=pair_index[f'{m}_FP32_200_minus100',p,k,seeds[0]]
                d1=first['coordinate_difference']['free_coordinate_RMS']
                d2=second['coordinate_difference']['free_coordinate_RMS']
                convergence.append(dict(model=m,parent=p,patch=k,seed=seeds[0],free_coordinate_RMS_50_to100=d1,
                    free_coordinate_RMS_100_to200=d2,ratio_second_to_first=d2/d1 if d1>0 else None,
                    coordinate_increment_shrank=d2<d1,metric_delta_100_minus50=first['metric_delta'],
                    metric_delta_200_minus100=second['metric_delta'],interpretation='Descriptive fixed four-case refinement; neither exact ODE truth nor proof of monotonic geometry quality'))
    endpoint_csv=_csv(folder/'endpoints.csv',_endpoint_rows(allrows));pair_csv=_csv(folder/'paired_cases.csv',pairs)
    convergence_csv=_csv(folder/'four_case_convergence.csv',convergence)
    result=dict(status='PASS',stage='A',seeds=seeds,endpoint_count=len(allrows),paired_case_count=len(pairs),four_case_convergence=convergence,
        source_manifests=sources,coordinate_archive_reads=cache.records,paired_rows=pairs,summary=summarize_pairs(pairs),
        conventions=_conventions(),artifacts=[endpoint_csv,pair_csv,convergence_csv],seconds=time.perf_counter()-tick,
        model_forwards=0,geometry_scoring_calls=0,OT_calls=0,autograd_calls=0,optimizer_updates=0)
    _json(folder/'results.json',result);return result


def analyze_B(out):
    out=Path(out);folder=out/'analysis'/'B';folder.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    cases,seeds,Arows,Aindex,sources=_shared(out);cache=RawCache(out,cases)
    rows_B,manifest_B=_read_scores(out,'B_B');rows_J,manifest_J=_read_scores(out,'B_J');cross=rows_B+rows_J;sources.extend([manifest_B,manifest_J])
    index={(x['source_model'],x['continuation_model'],int(x['parent']),int(x['patch']),int(x['seed']),int(x['start_step'])):x for x in cross}
    expected={(src,dst,p,k,int(seed),cut) for src,dst in (('B','J'),('J','B')) for p in (0,1) for k in (0,2,4) for seed in seeds for cut in (20,50,80)}
    if len(cross)!=72 or set(index)!=expected or any(x['precision']!='fp32' or x['nfe']!=100 for x in cross):
        raise ValueError('B scored matrix differs from the72 registered cross suffixes')
    pairs=[];four=[]
    for p in (0,1):
        for k in (0,2,4):
            for seed in seeds:
                for cut in (20,50,80):
                    cells={'BB':Aindex['B',p,k,seed,'fp32',100], 'BJ':index['B','J',p,k,seed,cut],
                           'JB':index['J','B',p,k,seed,cut], 'JJ':Aindex['J',p,k,seed,'fp32',100]}
                    four.append(dict(parent=p,patch=k,seed=seed,start_step=cut,switch_time=cut/100,
                        context_group='training_C' if k==0 else 'other_C',case_id=_case_id(cells['BB']),
                        definition='First letter is source state; second is continuation field. BB/JJ endpoints reused from original A.',
                        cells={key:dict(endpoint_id=_endpoint_id(value),path=value['path'],scalars=value['scalars']) for key,value in cells.items()}))
                    for src,left,right in (('B','BJ','BB'),('J','JJ','JB')):
                        q=_pair('same_source_continuation',f'source_{src}_J_minus_B_at_{cut}',[cells[left],cells[right]],[1,-1],cache,cases,
                            meaning=f'Same {src}-generated state at absolute t={cut/100}; J continuation minus B continuation')
                        q.update(source_model=src,start_step=cut,switch_time=cut/100);pairs.append(q)
    if len(four)!=36 or len(pairs)!=72:raise AssertionError('B four-cell count changed')
    four_csv=_csv(folder/'four_cells.csv',four);pair_csv=_csv(folder/'paired_cases.csv',pairs);endpoint_csv=_csv(folder/'cross_endpoints.csv',_endpoint_rows(cross))
    gates=[]
    for label in ('B','J'):
        path=out/'B'/label/'gate_results.json';g=json.loads(path.read_text())
        if g.get('status')!='PASS' or len(g.get('same_model_replays',[]))!=3 or not all(x.get('same_model_full_suffix_bitwise') for x in g['same_model_replays']):
            raise ValueError('Cannot interpret cross suffixes without all six same-model repeat proofs')
        gates.append(dict(model=label,manifest=_record(path),counts=g['counts'],same_model_suffixes_bitwise=True))
    conventions=_conventions();conventions['cross_suffix']='Hybrid trained fields, not standard B/J sampling and not a strict causal decomposition'
    result=dict(status='PASS',stage='B',seeds=seeds,cross_suffix_count=72,reused_original_endpoints=24,
        four_cell_cases=four,paired_rows=pairs,summary=summarize_pairs(pairs),same_model_gate_evidence=gates,
        source_manifests=sources,coordinate_archive_reads=cache.records,conventions=conventions,
        artifacts=[four_csv,pair_csv,endpoint_csv],seconds=time.perf_counter()-tick,
        model_forwards=0,geometry_scoring_calls=0,OT_calls=0,autograd_calls=0,optimizer_updates=0)
    _json(folder/'results.json',result);return result


def run_cpu_fixtures():
    d=np.array([10.,-80.,10.]);s=_stats(d,3,direction='lower_is_better')
    assert s['median']==10 and s['favorable_count']==1 and s['worse_count']==2
    assert _stats([0,None,2],3,direction='descriptive_only')['missing']==1
    assert _stats([0,None,2],3,direction='descriptive_only')['zero']==1
    assert _direction('free_total_surface_area')=='descriptive_only'
    assert _direction('free_near_degenerate_count')=='lower_is_better'
    assert _direction('Gamma_to_free_edges_coverage_0.001_fraction')=='higher_is_better'
    assert _direction('fixed_R12_coverage_RMS_bbox_pct')=='lower_is_better'
    x=np.ones((112,9),np.float32);x[:12]=0;zero=np.zeros_like(x)
    c=coordinate_change([x,zero],[1,-1],2)
    assert c['free_coordinate_RMS']==1 and c['free_coordinate_RMS_bbox_pct']==50
    assert c['known_max_abs']==0 and np.isclose(c['free_vertex_RMS'],np.sqrt(3))
    rows=[dict(scalars={'error':x}) for x in (4,1,8,2)]
    assert _linear_scalars(rows,[1,-1,-1,1])['error']==-3
    return dict(status='PASS',fixture_groups=4,description=['paired_median_signs','missing_zero_and_metric_direction','coordinate_RMS_units','difference_of_differences'],
                model_forwards=0,geometry_scoring_calls=0,OT_calls=0,autograd_calls=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--stage',choices=('A','B'),required=True)
    args=parser.parse_args();result=(analyze_A if args.stage=='A' else analyze_B)(args.out)
    print(json.dumps(dict(status=result['status'],stage=args.stage,seconds=result['seconds']),allow_nan=False))


if __name__=='__main__':main()
