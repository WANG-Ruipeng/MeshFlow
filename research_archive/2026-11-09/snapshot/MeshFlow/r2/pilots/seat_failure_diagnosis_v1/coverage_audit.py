"""CPU-only C coverage/exposure audit. No torch, scoring, generated geometry reads.

Only the new experiment coverage_audit directory is writable. Matching uses C
and requested N only; source M appears solely in descriptive coverage columns.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
from pathlib import Path
import csv, json, hashlib, time, math
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path('/mnt/e/MeshFlow')
OLD=ROOT/'experiments/r2_user_data_scaling_32_67_v1'
WIDTH=ROOT/'experiments/r2_condition_width_response_v1'
OUT=ROOT/'experiments/r2_seat_failure_diagnosis_v1/coverage_audit'
L=9.208138264587333
MODEL_SHA='2e2a40bcacda7e5ee04a5754018f71986ef5d378345058075ba412bec808f925'
FEATURES=['center_x_L','center_y_L','center_z_L','span_x_L','span_y_L','span_z_L',
          'sqrt_area_L','inclination_to_horizontal_90','K_over_N','N_over_256']
DEFINITION=dict(schema='r2_seat_failure_C_coverage_v1',fixed_length=L,
    coordinate_frame='Existing model XYZ; Y is vertical. No recenter, rotation, per-shape rescale or ICP.',
    centroid='Triangle-area-weighted triangle centroids, FP64',
    area='Sum triangle areas, not union area; duplicates would count with multiplicity.',
    inclination='Area-weighted arccos(abs(unit_normal_y)) in degrees; winding invariant, 0 horizontal / 90 vertical.',
    matching_features=FEATURES,matching_distance='sqrt(mean((a-b)^2)) over the ten listed dimensionless features; equal weights, no fitted scales.',
    area_context='C/M and C/R source-area ratios descriptive only; full M/Y, generated output and quality never enter matching.',
    parent_relative_height='(C area-centroid y - source M ymin)/(source M ymax-ymin), descriptive only.',
    selection='Retain width8 HALF at original selection_order 0,2,3 (K=1,4,13), both FULL and two BACK; add lower median K of remaining DEV SEAT_FULL with unused parent, sort K,parent,condition.',
    assignment='Within exact condition type, minimum total squared feature distance over distinct TRAIN67 parents using linear_sum_assignment; best condition per parent chosen input-only, lexicographic tie order.',
    coverage='For each DEV condition and same-type train pool, descriptive nearest distance and per-feature train min/max inclusion. Marginal inclusion is not joint coverage or IID.',
    exposure='Recount completed formal sample records; each arm exactly48000, unique sample_index0..47999 and step1..6000. No FM fixture is a generated mesh.',
    generation_inventory='Identity-only scan existing experiments/*/evaluation/raw_index.json and post_index.json, excluding this new run; count only pinned EXP67 or explicit USER_EXPANDED_67 label, verify paths. Not a claim about unindexed/unidentifiable files.',
    rng_recommendation='Reuse registered DEV Gaussian. For matched TRAIN use the same MT19937 seed, randn(Ntrain-Ktrain,3,3).astype(float32); check shared prefix exactly against saved DEV. Preserve each request own N and K. Different sizes are not identical full arrays.',
    limitation='Input matching neither equalizes task difficulty nor proves IID; source parent seen in conditional training differs intentionally. No training, model forward, generation or quality scoring in this script.')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

PROTECTED={}
def protect(p,expected=None):
    p=Path(p).resolve(); h=sha(p)
    if expected and h!=expected:raise ValueError('Source hash differs '+str(p))
    if str(p) in PROTECTED and PROTECTED[str(p)]!=h:raise ValueError('Source changed')
    PROTECTED[str(p)]=h
    return p

def read_json(p):return json.loads(protect(p).read_text(encoding='utf-8-sig'))
def read_csv(p):
    with protect(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_json(name,value):
    p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,sort_keys=True)+'\n'
    if p.exists():
        if p.read_text(encoding='utf8')!=text:raise ValueError('Refuse to overwrite different audit '+str(p))
    else:p.write_text(text,encoding='utf8')

def write_csv(name,rows):
    if not rows:raise ValueError('Empty expected table '+name)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    import io
    buf=io.StringIO(newline=''); writer=csv.DictWriter(buf,fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    p=OUT/name;text=buf.getvalue()
    if p.exists():
        if p.read_text(encoding='utf8')!=text:raise ValueError('Refuse different CSV '+str(p))
    else:p.write_text(text,encoding='utf8',newline='')

def areas(tri):return np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2

def descriptor(t,parent,split,pool):
    cp=protect(t['condition_source'],t['condition_source_sha256'])
    pp=protect(parent['npz'],parent['sha256'])
    with np.load(cp,allow_pickle=False) as z:
        C=z['C'].astype(np.float64); ids=z['source_face_ids']; rid=z['region_source_face_ids']; N=int(z['N']);K=int(z['K'])
        assert (K,N)==(int(t['K']),int(t['N']))
    with np.load(pp,allow_pickle=False) as z: M=z['full_target'].reshape(N,3,3).astype(np.float64)
    assert np.array_equal(C,M[ids]) and len(np.unique(ids))==K
    A=areas(C); total=float(A.sum()); MA=areas(M); Rarea=float(MA[rid].sum())
    center=np.average(C.mean(1),axis=0,weights=A); low=C.min(axis=(0,1));high=C.max(axis=(0,1));span=high-low
    cross=np.cross(C[:,1]-C[:,0],C[:,2]-C[:,0]); normals=cross/(2*A[:,None])
    tilt=float(np.average(np.degrees(np.arccos(np.clip(np.abs(normals[:,1]),0,1))),weights=A))
    mlo=M.min(axis=(0,1));mhi=M.max(axis=(0,1));height=float(mhi[1]-mlo[1]);area_fraction=total/float(MA.sum())
    if not math.isclose(area_fraction,float(t['area_C_over_M']),rel_tol=1e-9,abs_tol=1e-11):
        raise ValueError('Saved area ratio differs '+t['condition_id'])
    row=dict(split=split,pool_membership=pool,parent_id=t['uid'],group_uid=t.get('group_uid',t['uid']),condition_id=t['condition_id'],condition_type=t['condition_type'],N=N,K=K,free_faces=N-K,K_over_N=K/N,
        area_C=total,area_M=float(MA.sum()),area_region=Rarea,area_C_over_M=area_fraction,area_C_over_region=total/Rarea,
        C_centroid_height_y=float(center[1]),C_centroid_height_source_fraction=float((center[1]-mlo[1])/height),source_M_height=height,
        C_inclination_to_horizontal_deg=tilt,source_quality=t['source_quality'],source_FF_illegal=t['CC_CF_FF']['FF']['illegal'],source_FF_unknown=t['CC_CF_FF']['FF']['unknown'],
        C_source=str(cp),C_sha256=PROTECTED[str(cp)],parent_source=str(pp),parent_sha256=PROTECTED[str(pp)],
        sqrt_area_L=math.sqrt(total)/L,inclination_to_horizontal_90=tilt/90,N_over_256=N/256)
    for j,axis in enumerate('xyz'):
        row.update({f'C_centroid_{axis}':float(center[j]),f'C_span_{axis}':float(span[j]),f'center_{axis}_L':float(center[j]/L),f'span_{axis}_L':float(span[j]/L),f'area_mean_abs_normal_{axis}':float(np.average(np.abs(normals[:,j]),weights=A))})
    return row,dict(areas=A,normals=normals)

def vec(r):return np.array([r[k] for k in FEATURES],dtype=np.float64)

def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists():
        done=json.loads((OUT/'completion.json').read_text())
        assert all(sha(OUT/p)==h for p,h in done['output_hashes'].items())
        print(json.dumps(dict(status='REUSED',source=done['schema'])));return
    write_json('definition.json',DEFINITION)
    train=read_json(OLD/'data/train_manifest.json'); dev=read_json(OLD/'data/panel_manifest.json')
    small=set(train['pools']['USER_SMALL']); big=set(train['pools']['USER_EXPANDED_67'])
    assert len(small)==32 and len(big)==67 and small<big
    devp={p['uid']:p for p in dev['dev_parents']}; assert len(devp)==32 and not big&set(devp)
    rows=[];geometries={}
    for t in sorted(train['tasks'].values(),key=lambda x:x['condition_id']):
        r,g=descriptor(t,train['parents'][t['uid']],'TRAIN','SMALL32+EXP67' if t['uid'] in small else 'EXP67_ONLY');rows.append(r);geometries[r['condition_id']]=g
    for t in sorted(dev['conditions'],key=lambda x:x['condition_id']):
        r,g=descriptor(t,devp[t['uid']],'DEV','NEW32');rows.append(r);geometries[r['condition_id']]=g
    assert len(rows)==258 and len({r['condition_id'] for r in rows})==258
    byid={r['condition_id']:r for r in rows};write_csv('condition_geometry.csv',rows)
    exposures=read_csv(OLD/'training_analysis/final/exposures.csv')
    counts=defaultdict(Counter);parents=defaultdict(Counter);clean=defaultdict(Counter);free=Counter();alpha_area=defaultdict(list)
    for arm in ('USER_SMALL','USER_EXPANDED_67'):
        ex=[r for r in exposures if r['arm']==arm]
        assert len(ex)==48000 and {int(r['sample_index']) for r in ex}==set(range(48000))
        assert Counter(int(r['step']) for r in ex)==Counter({k:8 for k in range(1,6001)})
        for e in ex:
            assert e['optimizer_update_completed']=='True'
            r=byid[e['task_id']]; assert r['parent_id']==e['uid'] and (int(e['N']),int(e['K']))==(r['N'],r['K'])
            assert e['uid'] in train['pools'][arm]
            counts[arm][e['task_id']]+=1;parents[arm][e['uid']]+=1;alpha=float(e['alpha']);clean[arm][e['task_id']]+=int(alpha==1)
            free[arm]+=int(e['free_coordinate_count']); assert int(e['free_coordinate_count'])==9*(r['N']-r['K'])
            g=geometries[e['task_id']];n=g['normals'];a=g['areas']
            changed_area=float(np.sum(a*np.sqrt(n[:,0]**2+alpha**2*(n[:,1]**2+n[:,2]**2))))
            alpha_area[(arm,e['task_id'])].append(changed_area)
    exrows=[];parentrows=[];summaries=[]
    for arm,pool in train['pools'].items():
        poolrows=[r for r in rows if r['split']=='TRAIN' and r['parent_id'] in pool]
        for r in poolrows:
            cid=r['condition_id'];n=counts[arm][cid]
            exrows.append(dict(arm=arm,parent_id=r['parent_id'],condition_id=cid,condition_type=r['condition_type'],N=r['N'],K=r['K'],K_over_N=r['K_over_N'],area_C_over_M=r['area_C_over_M'],source_area_C=r['area_C'],completed_exposures=n,clean_exposures=clean[arm][cid],width_exposures=n-clean[arm][cid],mean_actual_augmented_C_area=float(np.mean(alpha_area[(arm,cid)]))))
        for uid in sorted(pool):
            pr=[r for r in poolrows if r['parent_id']==uid]
            parentrows.append(dict(arm=arm,parent_id=uid,conditions=len(pr),completed_exposures=parents[arm][uid],N=pr[0]['N'],K_min=min(r['K'] for r in pr),K_max=max(r['K'] for r in pr)))
        for typ in ['ALL']+sorted({r['condition_type'] for r in poolrows}):
            pr=[r for r in poolrows if typ=='ALL' or r['condition_type']==typ];n=sum(counts[arm][r['condition_id']] for r in pr)
            d=dict(pool=arm,condition_type=typ,parents=len({r['parent_id'] for r in pr}),conditions=len(pr),completed_exposures=n,exposure_fraction=n/48000)
            for k in ['N','K','K_over_N','area_C_over_M','C_centroid_height_y','C_inclination_to_horizontal_deg']:
                vals=[r[k] for r in pr];d.update({k+'_min':min(vals),k+'_median':float(np.median(vals)),k+'_max':max(vals),k+'_condition_mean':float(np.mean(vals)),k+'_exposure_mean':sum(r[k]*counts[arm][r['condition_id']] for r in pr)/n})
            summaries.append(d)
    for typ in ['ALL']+sorted({r['condition_type'] for r in rows if r['split']=='DEV'}):
        pr=[r for r in rows if r['split']=='DEV' and (typ=='ALL' or r['condition_type']==typ)]
        d=dict(pool='DEV32',condition_type=typ,parents=len({r['parent_id'] for r in pr}),conditions=len(pr),completed_exposures=0,exposure_fraction=0)
        for k in ['N','K','K_over_N','area_C_over_M','C_centroid_height_y','C_inclination_to_horizontal_deg']:
            vals=[r[k] for r in pr];d.update({k+'_min':min(vals),k+'_median':float(np.median(vals)),k+'_max':max(vals),k+'_condition_mean':float(np.mean(vals)),k+'_exposure_mean':''})
        summaries.append(d)
    write_csv('exposure_per_condition.csv',exrows);write_csv('exposure_per_parent.csv',parentrows);write_csv('coverage_summary.csv',summaries)
    ranges=[];nearest=[];pairdist=[]
    for arm,pool in train['pools'].items():
        for d in [r for r in rows if r['split']=='DEV']:
            opts=[r for r in rows if r['split']=='TRAIN' and r['parent_id'] in pool and r['condition_type']==d['condition_type']]
            distances=[float(np.sqrt(np.mean((vec(d)-vec(t))**2))) for t in opts];order=sorted(range(len(opts)),key=lambda i:(distances[i],opts[i]['condition_id']))
            lo=np.min([vec(t) for t in opts],axis=0);hi=np.max([vec(t) for t in opts],axis=0)
            bad=[]
            for j,f in enumerate(FEATURES):
                inside=bool(lo[j]<=d[f]<=hi[j]);
                if not inside:bad.append(f)
                ranges.append(dict(pool=arm,condition_id=d['condition_id'],condition_type=d['condition_type'],feature=f,value=d[f],train_min=float(lo[j]),train_max=float(hi[j]),within_marginal_range=inside,below_by=max(0.,float(lo[j]-d[f])),above_by=max(0.,float(d[f]-hi[j]))))
            best=opts[order[0]];nearest.append(dict(pool=arm,dev_condition_id=d['condition_id'],condition_type=d['condition_type'],nearest_train_condition_id=best['condition_id'],nearest_train_parent=best['parent_id'],distance=distances[order[0]],train_conditions=len(opts),outside_feature_count=len(bad),outside_features=';'.join(bad),all_marginal_ranges=not bad))
            if arm=='USER_EXPANDED_67':
                for i in order:pairdist.append(dict(dev_condition_id=d['condition_id'],train_condition_id=opts[i]['condition_id'],train_parent=opts[i]['parent_id'],condition_type=d['condition_type'],distance=distances[i]))
    write_csv('dev_train_nearest.csv',nearest);write_csv('dev_train_feature_ranges.csv',ranges);write_csv('dev_train_all_same_type_distances.csv',pairdist)
    original=read_csv(WIDTH/'selected_conditions.csv'); chosen=[]
    for x in original:
        if x['condition_type']!='SEAT_HALF' or int(x['selection_order']) in (0,2,3):chosen.append(byid[x['condition_id']])
    used={r['parent_id'] for r in chosen};options=sorted([r for r in rows if r['split']=='DEV' and r['condition_type']=='SEAT_FULL' and r['parent_id'] not in used],key=lambda r:(r['K'],r['parent_id'],r['condition_id']))
    extra=options[(len(options)-1)//2];chosen.append(extra)
    chosen=sorted(chosen,key=lambda r:({'SEAT_HALF':0,'SEAT_FULL':1,'BACK_HALF':2,'BACK_FULL':3}[r['condition_type']],r['K'],r['condition_id']))
    assert len(chosen)==8 and len({r['parent_id'] for r in chosen})==8
    orderedparents=sorted(big);cost=np.full((8,len(orderedparents)),1e8);candidates={}
    for i,d in enumerate(chosen):
        for j,pid in enumerate(orderedparents):
            ts=[r for r in rows if r['split']=='TRAIN' and r['parent_id']==pid and r['condition_type']==d['condition_type']]
            if ts:
                t=min(ts,key=lambda t:(float(np.mean((vec(d)-vec(t))**2)),t['condition_id']));cost[i,j]=float(np.mean((vec(d)-vec(t))**2));candidates[(i,j)]=t
    ii,jj=linear_sum_assignment(cost);pairs=[]
    for i,j in zip(ii,jj):
        assert cost[i,j]<1e8;d=chosen[i];t=candidates[(i,j)]
        row=dict(pair_id=f'PAIR{i:02d}',condition_type=d['condition_type'],dev_parent_id=d['parent_id'],dev_condition_id=d['condition_id'],train_parent_id=t['parent_id'],train_condition_id=t['condition_id'],distance=math.sqrt(cost[i,j]),dev_N=d['N'],dev_K=d['K'],train_N=t['N'],train_K=t['K'],dev_free=d['N']-d['K'],train_free=t['N']-t['K'],train_exposures=counts['USER_EXPANDED_67'][t['condition_id']],dev_was_width8=any(x['condition_id']==d['condition_id'] for x in original),dev_condition_source=d['C_source'],train_condition_source=t['C_source'],dev_condition_sha256=d['C_sha256'],train_condition_sha256=t['C_sha256'])
        for f in FEATURES:row['dev_'+f]=d[f];row['train_'+f]=t[f];row['delta_'+f]=t[f]-d[f]
        pairs.append(row)
    assert len({r['train_parent_id'] for r in pairs})==8;write_csv('selected_train_dev_pairs.csv',pairs)
    write_json('selected_train_dev_manifest.json',dict(definition_sha256=sha(OUT/'definition.json'),pairs=pairs,extra_dev_FULL_candidates=[r['condition_id'] for r in options],selected_extra_full=extra['condition_id'],selection_used_generated_quality=False,assignment_total_squared_cost=float(sum(cost[i,j] for i,j in zip(ii,jj)))))
    # Inventory is deliberately after selection; only identity fields are used.
    inventory=[];scan=[]
    for stage in ('raw','post'):
        for p in sorted((ROOT/'experiments').glob('*/evaluation/'+stage+'_index.json')):
            if p.is_relative_to(OUT.parent):continue
            obj=read_json(p);rs=obj.get('rows',[]) if isinstance(obj,dict) else obj
            if not isinstance(rs,list):continue
            hits=[]
            for r in rs:
                if r.get('checkpoint_sha256')!=MODEL_SHA and r.get('label')!='USER_EXPANDED_67':continue
                pid=r.get('parent_id','');paths=[(k,v) for k,v in r.items() if k in ('raw_path','post_path','ply_path','result_path','canonical_path') and isinstance(v,str)]
                hit=dict(index_path=str(p),stage=stage,request_id=r.get('request_id',''),parent_id=pid,condition_id=r.get('condition_id',''),label=r.get('label',''),checkpoint_sha256=r.get('checkpoint_sha256',''),is_EXP67_train_parent=pid in big,is_DEV32_parent=pid in devp,geometry_paths=json.dumps(dict(paths),sort_keys=True),all_indexed_geometry_paths_exist=bool(paths) and all(Path(v).is_file() for _,v in paths))
                inventory.append(hit);hits.append(hit)
            scan.append(dict(index_path=str(p),stage=stage,rows=len(rs),EXP67_rows=len(hits),EXP67_train_parent_rows=sum(r['is_EXP67_train_parent'] for r in hits)))
    write_csv('existing_generation_inventory.csv',inventory);write_csv('generation_index_search.csv',scan)
    requests=read_csv(OLD/'evaluation/requests.csv');reqby={(r['condition_id'],int(r['candidate'])):r for r in requests}
    reqs=[]
    for p in pairs:
        for noise in (0,1):
            r=reqby[(p['dev_condition_id'],noise)]
            reqs.append(dict(pair_id=p['pair_id'],candidate=noise,dev_request_id=r['request_id'],noise_seed=r['noise_seed'],dev_input_path=r['input_path'],dev_input_sha256=r['input_sha256'],dev_Gaussian_sha256=r['Gaussian_sha256'],train_N=p['train_N'],train_K=p['train_K'],train_free_faces=p['train_free'],dev_free_faces=p['dev_free'],rng_rule='same stored seed / NumPy RandomState.randn(count,3,3).astype(float32); assert shared prefix against DEV; no force N equality'))
    write_csv('paired_request_reuse_and_rng.csv',reqs)
    for p,h in PROTECTED.items():
        if sha(p)!=h:raise ValueError('Protected source modified '+p)
    write_json('source_hashes.json',PROTECTED)
    stats=dict(source_conditions=len(rows),source_parents=len(big|set(devp)),formal_exposures={a:sum(c.values()) for a,c in counts.items()},free_scalars=dict(free),selected_pairs=len(pairs),selected_types=dict(Counter(r['condition_type'] for r in pairs)),selected_dev_width8=sum(r['dev_was_width8'] for r in pairs),EXP67_indexed_train_RAW=sum(r['stage']=='raw' and r['is_EXP67_train_parent'] for r in inventory),EXP67_indexed_train_POST=sum(r['stage']=='post' and r['is_EXP67_train_parent'] for r in inventory),marginal_outside_counts={arm:sum(r['outside_feature_count']>0 for r in nearest if r['pool']==arm) for arm in train['pools']},zero_model_calls=True,source_hashes_unchanged=True)
    write_json('summary.json',stats)
    lines=['# Condition coverage and exposure audit','',
        'All source files are read-only. This stage used CPU arrays and identity metadata only; zero model forwards, new meshes, quality rescoring or training.',
        '',f'Unique inputs: {len(rows)} conditions / {len(big|set(devp))} parents. TRAIN67=194, SMALL32=91 nested, DEV32=64. Completed formal exposures: {dict(stats["formal_exposures"])}.',
        '',f'Within the scanned EXP67 RAW/POST indices, training-parent generated outputs found: RAW={stats["EXP67_indexed_train_RAW"]}, POST={stats["EXP67_indexed_train_POST"]}. Fixed FM diagnostic fixtures are velocity regression, not rollout or qualitative generation. Unindexed unidentified files remain outside this claim.',
        '',f'DEV conditions outside at least one same-type marginal TRAIN feature range: {stats["marginal_outside_counts"]}. A condition inside all scalar ranges is not thereby jointly covered, equally difficult or IID.',
        '', 'Matching is fixed before any generated-quality inspection: 3 SEAT_HALF + 3 SEAT_FULL + 2 BACK, eight unique DEV and TRAIN parents. Seven DEV inputs reuse width8; one extra FULL comes from input-only lower-median K and also has frozen original EXP67 outputs.',
        '', 'Model Y is the vertical axis. Centroids are area weighted; tilt is unsigned inclination from the horizontal plane. The fixed L is inherited, not estimated from outputs. C/M and source-relative heights are descriptive and do not enter matching.',
        '', 'Use each request own N/K. Reuse DEV saved Gaussian arrays. The old sampler uses NumPy MT19937; the same seed at TRAIN free count should share the exact prefix, which the execution owner must assert. Different free counts prevent claiming the full noise arrays are identical.',
        '', 'Matching does not equalize training exposure, complete-chair structure, semantic ambiguity, topology or expected completion. A small TRAIN-vs-DEV result supports diagnosis only, not a generalization benchmark or causal architecture conclusion.',
        '', 'Files: condition_geometry.csv; exposure_per_condition.csv; exposure_per_parent.csv; coverage_summary.csv; dev_train_nearest.csv; dev_train_feature_ranges.csv; dev_train_all_same_type_distances.csv; selected_train_dev_pairs.csv; paired_request_reuse_and_rng.csv; generation_index_search.csv; existing_generation_inventory.csv; definition.json; source_hashes.json.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='completion.json'}
    write_json('completion.json',dict(schema=DEFINITION['schema'],status='COMPLETE',wall_seconds=time.time()-start,source_count=len(PROTECTED),source_hashes_unchanged=True,script_sha256=sha(__file__),output_hashes=outputs,stats=stats))
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
