"""Independent CPU audit of fixed S1->S5 student denoising; zero model calls.

Reads immutable RAW/S1 and actual S2..S5 arrays/receipts/ledger. Only writes
experiments/r2_student_multistep_denoise_v1/audit. No torch/GPU imports.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).parent
OUT=ROOT/'experiments/r2_student_multistep_denoise_v1'
SCALE=.95*.3762
STUDENT_SHA='2b7bd5536ec9c4b32ca36d3d8c32c7d0510b74eddcfd73e23287fdafce38c34d'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def exact(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def write(path,value):
    path=Path(path).absolute()
    if not path.is_relative_to((OUT/'audit').absolute()):raise ValueError('Audit output outside owned directory')
    for p in (path,*path.parents):
        if p.is_symlink():raise ValueError('Symlink in audit output path')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def csvwrite(path,rows):
    path=Path(path)
    if not path.absolute().is_relative_to((OUT/'audit').absolute()):raise ValueError('CSV outside owned directory')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def triangles(array,N,name):
    a=np.asarray(array)
    if a.dtype!=np.float32 or a.size!=9*N or not np.isfinite(a).all():
        raise AssertionError(name+' must be finite FP32 triangles with registered N')
    return a.reshape(N,3,3)

def displacement_check(saved,output,reference):
    if saved.dtype not in (np.float32,np.float64):raise AssertionError('Displacement dtype')
    expected=np.subtract(output.astype(saved.dtype),reference.astype(saved.dtype),dtype=saved.dtype)
    if not exact(saved,expected):raise AssertionError('Displacement does not match declared origin')
    return str(saved.dtype)

def run(registration_path=OUT/'registration.json'):
    registration_path=Path(registration_path)
    reg=read(registration_path);registration_sha=sha(registration_path)
    records=reg['records']
    assert len(records)==12 and len({r['request_id'] for r in records})==12
    assert all(r['label']=='R2_USER_MIX' for r in records)
    assert reg['student_checkpoint_sha256']==STUDENT_SHA
    assert sha(reg['student_checkpoint_path'])==STUDENT_SHA
    sources=reg.get('runner_hashes',reg.get('source_hashes',{}))
    assert sources,'Frozen runner/source hashes are required for independent audit'
    for p,h in sources.items():assert sha(p)==h,('Frozen source changed',p)
    pre_sources=dict(sources)
    rows=[];reuse=[];expected_calls=set()
    for row in records:
        rid=row['request_id'];N,K=int(row['N']),int(row['K'])
        assert 0<K<N<=800
        assert sha(row['raw_path'])==row['raw_sha256']
        assert sha(row['S1_denoised_path'])==row['S1_denoised_sha256']
        raw_data=arrays(row['raw_path']);raw=triangles(raw_data['raw'],N,'RAW');c=triangles(raw_data['C'],K,'C')
        s1=arrays(row['S1_denoised_path']);one=triangles(s1['output'],N,'S1')
        s1proof=read(row['S1_receipt_path'])
        assert s1proof['status']=='PASS'
        assert s1proof['request_id']==rid and s1proof['raw_sha256']==row['raw_sha256']
        assert s1proof['output_sha256']==row['S1_denoised_sha256']
        assert s1proof['student_checkpoint_sha256']==STUDENT_SHA
        assert s1proof.get('teacher_instance_used') is False
        assert exact(raw[:K],c) and exact(one[:K],c)
        previous=one;previous_path=Path(row['S1_denoised_path'])
        reuse.append(dict(request_id=rid,raw_sha256=row['raw_sha256'],S1_sha256=row['S1_denoised_sha256'],S1_receipt_sha256=sha(row['S1_receipt_path'])))
        for number in (2,3,4,5):
            path=OUT/'denoised'/f'S{number}'/rid/'denoised.npz';receipt_path=path.with_name('receipt.json')
            proof=read(receipt_path);a=arrays(path)
            assert proof['status']=='PASS' and proof['request_id']==rid and proof['round']==number
            assert Path(proof['input_path']).resolve()==previous_path.resolve()
            assert proof['input_sha256']==sha(previous_path)
            assert proof['raw_sha256']==row['raw_sha256']
            assert proof['registration_sha256']==registration_sha
            assert proof['checkpoint_sha256']==STUDENT_SHA and proof['output_sha256']==sha(path)
            assert proof['N']==N and proof['K']==K
            assert proof.get('no_rng_consumed') is True
            assert proof.get('no_intermediate_GEO') is True
            output=triangles(a['output'],N,'output');inp=triangles(a['input_model'],N,'input_model')
            xd=triangles(a['input_d'],N,'input_d');pd=triangles(a['proposal_d'],N,'proposal_d');pm=triangles(a['proposal_model'],N,'proposal_model')
            assert exact(inp,previous),('Previous mesh chain differs',rid,number)
            assert exact(triangles(a['C'],K,'C'),c) and exact(output[:K],c)
            assert exact(xd,np.multiply(previous,np.float32(SCALE),dtype=np.float32))
            # Verified PyTorch scalar-division kernel uses FP32 reciprocal multiplication.
            expected_pm=np.multiply(pd,np.float32(1./SCALE),dtype=np.float32)
            assert exact(pm,expected_pm),('FP32 scale inverse differs',rid,number)
            effective=np.subtract(pd,xd,dtype=np.float32)
            assert exact(a['residual_d'],effective) and exact(a['effective_residual_d'],effective)
            restored=pm.copy();restored[:K]=c
            assert exact(output,restored)
            for key,reference in (('displacement_from_previous',previous),('displacement_from_RAW',raw),('displacement_from_S1',one)):
                displacement_check(a[key],output,reference)
            naive=np.divide(pd,np.float32(SCALE),dtype=np.float32)
            step=output[K:].astype(np.float64)-previous[K:].astype(np.float64)
            total=output[K:].astype(np.float64)-raw[K:].astype(np.float64)
            rows.append(dict(request_id=rid,round=number,N=N,K=K,free_faces=N-K,status='PASS',
                input_path=str(previous_path),input_sha256=proof['input_sha256'],output_path=str(path),output_sha256=proof['output_sha256'],
                receipt_sha256=sha(receipt_path),C_bitwise=True,N_preserved=True,input_chain_bitwise=True,effective_residual_bitwise=True,
                proposal_inverse_scale='Exact FP32 reciprocal multiply, same as original Torch scalar division',
                naive_CPU_divide_max_abs_difference=float(np.max(np.abs(pm.astype(np.float64)-naive.astype(np.float64)))),
                mean_step_free_corner_distance=float(np.mean(np.linalg.norm(step,axis=-1))),
                max_step_free_corner_distance=float(np.max(np.linalg.norm(step,axis=-1))),
                mean_total_free_corner_distance=float(np.mean(np.linalg.norm(total,axis=-1)))))
            previous=output;previous_path=path;expected_calls.add((rid,number))
    ledger_path=OUT/'denoise_ledger.jsonl'
    ledger=[json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    starts=[r for r in ledger if r['event']=='STUDENT_STARTED'];ends=[r for r in ledger if r['event']=='STUDENT_COMPLETE']
    failures=[r for r in ledger if r['event']=='STUDENT_FAILED']
    keys=lambda rr:[(r['request_id'],int(r['round'])) for r in rr]
    assert len(starts)==len(ends)==48 and not failures
    assert len(set(keys(starts)))==len(set(keys(ends)))==48
    assert set(keys(starts))==set(keys(ends))==expected_calls
    # Each closure must follow its unique start, and later rounds must follow the prior closure.
    positions={(r['event'],r.get('request_id'),r.get('round')):i for i,r in enumerate(ledger) if r['event'] in ('STUDENT_STARTED','STUDENT_COMPLETE')}
    for rid,number in expected_calls:
        assert positions[('STUDENT_STARTED',rid,number)]<positions[('STUDENT_COMPLETE',rid,number)]
        if number>2:assert positions[('STUDENT_COMPLETE',rid,number-1)]<positions[('STUDENT_STARTED',rid,number)]
    for p,h in pre_sources.items():assert sha(p)==h,('Source changed during audit',p)
    for p,h in reg['protected'].items():assert sha(p)==h,('Protected file changed',p)
    assert sha(registration_path)==registration_sha
    completion=OUT/'denoise_completion.json'
    comp=read(completion)
    assert comp['status']=='PASS' and comp['total_student_forwards']==48
    assert comp['teacher_forwards']==comp['generator_forwards']==comp['training_updates']==0
    assert comp['student_state_before']==comp['student_state_after']=='b8a16c1f0ffef14e63809e7110dc3c05dbe4cba19f863e924cc545e13449acfa'
    assert comp['student_checkpoint_sha256']==STUDENT_SHA
    assert comp['no_new_noise'] and comp['C_bitwise_all'] and comp['N_fixed']
    assert comp['intermediate_GEO_C']==0
    summary=dict(status='PASS',registration_sha256=registration_sha,requests=12,reused_S1=12,new_student_started=48,new_student_completed=48,
        per_round_outputs=48,all_C_bitwise=True,all_N_preserved=True,all_chains_bitwise=True,no_intermediate_GEO=True,
        all_effective_residuals_bitwise=True,all_displacement_origins_verified=True,no_failed_or_replayed_call=True,
        runner_hashes_verified=pre_sources,protected_hashes_verified=len(reg['protected']),student_checkpoint_sha256=STUDENT_SHA,
        audit_model_calls=0,audit_GPU_calls=0,audit_new_noise=0,audit_source_sha256=sha(__file__),
        denoise_completion_sha256=sha(completion) if completion.exists() else None,reused_S1_identities=reuse,
        limitations='Saved-array/receipt/ledger/source consistency audit; no extra forward or independent regeneration. POST geometry metrics remain separately scored.')
    csvwrite(OUT/'audit/chain_per_round.csv',rows);write(OUT/'audit/chain_audit.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','requests','reused_S1','new_student_started','new_student_completed','all_C_bitwise','all_chains_bitwise','protected_hashes_verified')}))
    return summary


def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def number(v):
    if v is None or v in ('','None','null'):return None
    if isinstance(v,str) and v.lower() in ('true','false'):return float(v.lower()=='true')
    return float(v)

def equivalent(actual,expected):
    if expected is None:return actual in ('',None,'None','null')
    if isinstance(expected,(int,float,bool,np.generic)):
        v=number(actual)
        return v is not None and math.isclose(v,float(expected),rel_tol=1e-12,abs_tol=1e-15)
    return str(actual)==str(expected)

def ply_arrays(path):
    with Path(path).open('rb') as f:
        lines=[]
        for _ in range(32):
            line=f.readline(1024).decode('ascii').strip()
            if not line.startswith('comment '):lines.append(line)
            if line=='end_header':break
        assert lines[:2]==['ply','format binary_little_endian 1.0'] and len(lines)==9
        assert lines[3:6]==['property float x','property float y','property float z']
        nv=int(lines[2].split()[-1]);nf=int(lines[6].split()[-1]);data=f.read()
    face_dtype=np.dtype([('count','u1'),('indices','<i4',(3,))])
    assert len(data)==nv*12+nf*face_dtype.itemsize
    vertices=np.frombuffer(data,dtype='<f4',count=nv*3).reshape(nv,3).copy()
    faces=np.frombuffer(data,dtype=face_dtype,count=nf,offset=nv*12)
    assert np.all(faces['count']==3)
    return vertices,faces['indices'].astype(np.int64)

def check_summary(table,summary,basekeys):
    field_checks=0
    extras={'ALL':(), 'REGION':('region_type',), 'PARENT':('parent_id',),
            'NOISE':('candidate',), 'PARENT_NOISE':('parent_id','candidate')}
    seen=set()
    for result in summary:
        keys=tuple(basekeys)+extras[result['grouping']]
        token=(result['grouping'],tuple(result[k] for k in keys))
        assert token not in seen;seen.add(token)
        group=[r for r in table if all(str(r.get(k,''))==str(result[k]) for k in keys)]
        assert group and int(result['outputs'])==len(group)
        # Registered cases all have two noises per parent; parent/noise keys must stay separate.
        for name,value in result.items():
            if not name.endswith('_valid_n'):continue
            field=name[:-8]
            vals=[number(r.get(field)) for r in group if number(r.get(field)) is not None]
            assert int(value)==len(vals)
            expected={'mean':float(np.mean(vals)) if vals else None,
                      'median':float(np.median(vals)) if vals else None,
                      'p95':float(np.quantile(vals,.95)) if vals else None}
            for statistic,v in expected.items():
                assert equivalent(result[field+'_'+statistic],v),(field,statistic,result['grouping'],result)
                field_checks+=1
    expected={(grouping,tuple(str(r.get(k,'')) for k in tuple(basekeys)+extra))
              for grouping,extra in extras.items() for r in table}
    assert seen==expected,('Missing/extra summary groups',len(seen),len(expected))
    return dict(rows=len(summary),numeric_statistic_checks=field_checks,all_expected_groups=True)

def score_audit():
    reg=read(OUT/'registration.json');sreg=read(OUT/'score_registration.json')
    completion=read(OUT/'score_aggregate_completion.json');sc=read(OUT/'score_completion.json')
    assert completion['status']==sc['status']=='PASS'
    assert (completion['new_GEO'],completion['new_quality'],completion['reused_POST'])==(24,24,36)
    assert completion['model_calls']==completion['new_queries']==0
    assert sha(CODE/'score.py')==sreg['score_source_sha256']
    assert sreg['parent_registration_sha256']==sha(OUT/'registration.json')
    for mapping in (reg['protected'],reg['runner_hashes'],sreg['protected']):
        for p,h in mapping.items():assert sha(p)==h,('Protected/source file changed',p)
    stages=('IDENTITY_GEO','D1','S1','S3','S5')
    index=readcsv(OUT/'post_index.csv');ridset={r['request_id'] for r in reg['records']}
    assert len(index)==72
    assert {(r['stage'],r['request_id']) for r in index}=={(s,r) for s in (*stages,'RAW') for r in ridset}
    by_record={r['request_id']:r for r in reg['records']}
    baseline={(r['stage'],r['request_id']):r for r in sreg['baselines']}
    assert len(baseline)==36
    old_dist=ROOT/'experiments/r2_single_step_denoiser_distillation_v1/evaluation/post_index.csv'
    old_index={(r['method'],r['request_id']):r for r in readcsv(old_dist) if r['label']=='R2_USER_MIX'}
    alias={'IDENTITY_GEO':'IDENTITY_GEO','D1':'D0_GEO','S1':'STUDENT_GEO'}
    loaded={};mesh_rows=[]
    for row in index:
        stage,rid=row['stage'],row['request_id'];orig=by_record[rid];N,K=int(row['N']),int(row['K'])
        assert sha(row['post_path'])==row['post_sha256']
        raw=triangles(arrays(orig['raw_path'])['raw'],N,'RAW')
        if stage=='RAW':
            assert Path(row['post_path']).resolve()==Path(orig['raw_path']).resolve()
            continue
        a=arrays(row['post_path']);output=a['output']
        assert output.dtype==np.float32 and output.ndim==3 and output.shape[1:]==(3,3) and np.isfinite(output).all()
        assert exact(output[:K],raw[:K]) and exact(a['C'],raw[:K])
        assert exact(a['vertices'][a['faces']],output)
        v,f=ply_arrays(row['post_ply']);assert exact(v,a['vertices']) and exact(f,a['faces'])
        assert len(output)==int(row['output_N'])
        d=read(row['diagnostic_path']);loaded[stage,rid]=d
        assert d['status']==d['quality']['status']=='PASS'
        assert d['identity']['post_sha256']==row['post_sha256']
        assert d['quality']['identity']['processed_sha256']==row['post_sha256']
        assert d['quality']['free_face_denominator']==len(output)-K
        for mapping in (d['files'],d['quality']['files']):
            for p,h in mapping.items():assert sha(p)==h,p
        for k in ('parent_id','condition_id','region_type','tier','candidate','N','K'):
            assert equivalent(row[k],orig[k])
        if stage in alias:
            old=old_index[alias[stage],rid]
            for k in ('post_path','post_sha256','diagnostic_path','denoised_path','pre_GEO_key'):
                assert row[k]==old[k],('Baseline not exact reuse',stage,rid,k)
        else:
            pp=read(Path(row['post_path']).with_name('postprocess.json'))
            assert pp['status']=='PASS' and pp['input']['file_sha256']==sha(row['denoised_path'])
            assert Path(row['denoised_path'])==OUT/'denoised'/stage/rid/'denoised.npz'
            for name,item in pp['files'].items():assert sha(Path(row['post_path']).parent/name)==item['sha256']
            for kind in ('GEO','QUALITY'):
                for suffix in ('STARTED','COMPLETE'):
                    e=read(OUT/'score_events'/stage/rid/(kind+'_'+suffix+'.json'))
                    assert e['event']==kind+'_'+suffix and e['request_id']==rid and e['stage']==stage
        mesh_rows.append(dict(stage=stage,request_id=rid,N_raw=N,N_POST=len(output),K=K,C_bitwise=True,
                             indexed_output_exact=True,PLY_exact=True,reused=stage in alias,post_sha256=row['post_sha256']))
    tables={name:readcsv(OUT/(name+'.csv')) for name in ('quality_per_output','topology_per_output','legacy_indexed_topology_per_output',
                 'per_output','source_supported_reference_per_output','paired_per_output','AVMR_paired_per_output','modifications_per_output')}
    assert len(tables['quality_per_output'])==len(tables['topology_per_output'])==60
    vm=tables['per_output'];assert len(vm)==960 and collections.Counter(r['phase'] for r in vm)=={'POST':480,'PRE_GEO_C_RESTORED':480}
    setting_names={r['name'] for r in reg['settings']};assert len(setting_names)==8
    settings={r['name']:r for r in reg['settings']}
    seen=set();metric_checks=0
    for row in vm:
        stage,rid,phase,setting=(row[k] for k in ('stage','request_id','phase','setting'))
        token=(stage,rid,phase,setting);assert token not in seen;seen.add(token)
        d=loaded[stage,rid];items=d['metrics'] if phase=='POST' else d['pre_GEO_metrics']
        source=next(v for v in items if v['domain']=='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY' and v['setting']==setting)
        for k,v in source.items():assert equivalent(row[k],v),(token,k)
        assert equivalent(row['threshold_model'],settings[setting]['threshold_model'])
        vv=d['vertices'] if phase=='POST' else d['pre_GEO_vertices'];dist=[v['distance_model'] for v in vv]
        assert len(dist)==int(row['boundary_vertex_count'])
        n=len(dist);matched=sum(v is not None and float(v)<float(row['threshold_model']) for v in dist)
        assert int(row['matched_count'])==matched
        assert equivalent(row['VMR'],matched/n if n else None)
        assert equivalent(row['perfect'],int(matched==n) if n else None)
        metric_checks+=3
    assert seen=={(s,r,p,t) for s in stages for r in ridset for p in ('POST','PRE_GEO_C_RESTORED') for t in setting_names}
    for table,key in ((tables['quality_per_output'],'quality'),(tables['topology_per_output'],'topology')):
        for row in table:
            d=loaded[row['stage'],row['request_id']]
            values=d['quality']['metrics'] if key=='quality' else d['topology']
            for name,v in values.items():
                if name in row:assert equivalent(row[name],v),(key,name)
    maps={name:{(r['stage'],r['request_id']):r for r in table} for name,table in tables.items()
          if name in ('quality_per_output','topology_per_output','legacy_indexed_topology_per_output')}
    paired=tables['paired_per_output'];assert len(paired)==36
    assert {(r['comparison'],r['request_id']) for r in paired}=={(c,r) for c in ('S3_minus_S1','S5_minus_S1','S5_minus_S3') for r in ridset}
    pair_checks=0
    for row in paired:
        new,old=row['comparison'].split('_minus_');rid=row['request_id']
        for field,val in row.items():
            if not field.endswith('_delta'):continue
            key=field[:-6]
            if key.startswith('legacy_'):mapping=maps['legacy_indexed_topology_per_output'];key=key[7:]
            elif key in maps['topology_per_output'][new,rid]:mapping=maps['topology_per_output']
            else:mapping=maps['quality_per_output']
            a,b=number(mapping[new,rid][key]),number(mapping[old,rid][key])
            assert equivalent(val,a-b if a is not None and b is not None else None),(row['comparison'],rid,field)
            pair_checks+=1
        orig=by_record[rid]
        assert row['parent_id']==orig['parent_id'] and int(row['candidate'])==int(orig['candidate'])
    lookup={(r['stage'],r['request_id'],r['phase'],r['setting']):r for r in vm}
    avpaired=tables['AVMR_paired_per_output'];assert len(avpaired)==576
    for row in avpaired:
        new,old=row['comparison'].split('_minus_');rid,phase,setting=(row[k] for k in ('request_id','phase','setting'))
        a,b=lookup[new,rid,phase,setting],lookup[old,rid,phase,setting]
        for key in ('VMR','perfect'):
            x,y=number(a[key]),number(b[key])
            assert equivalent(row[key+'_delta'],x-y if x is not None and y is not None else None)
            pair_checks+=1
    summary_checks={}
    for name,table,keys in (
        ('quality',tables['quality_per_output'],('stage','label')),
        ('topology',tables['topology_per_output'],('stage','label')),
        ('modifications',tables['modifications_per_output'],('stage','label')),
        ('AVMR',vm,('stage','label','phase','setting')),
        ('paired',paired,('comparison','label')),
        ('AVMR_paired',avpaired,('comparison','label','phase','setting'))):
        summary_checks[name]=check_summary(table,readcsv(OUT/(name+'_summary.csv')),keys)
    # Explicitly preserve the paired parent/two-noise design rather than treating images as independent parents.
    parent_noises=collections.defaultdict(set)
    for row in reg['records']:parent_noises[row['parent_id']].add(int(row['candidate']))
    assert len(parent_noises)==6 and all(v=={0,1} for v in parent_noises.values())
    for kind in ('GEO','QUALITY'):
        for suffix in ('STARTED','COMPLETE'):
            assert len(list((OUT/'score_events').glob('*/*/'+kind+'_'+suffix+'.json')))==24
    for mapping in (reg['protected'],reg['runner_hashes'],sreg['protected']):
        for p,h in mapping.items():assert sha(p)==h
    summary=dict(status='PASS',scope='Saved arrays/receipts and independent CSV reductions only; no new quality scoring/model/query',
        new_GEO_POSTs=24,new_quality_receipts=24,reused_POSTs=36,RAW_references=12,all60_POST_C_bytes=True,all60_indexed_NPZ_PLY_exact=True,
        reused_baselines={'IDENTITY_GEO':12,'D1':12,'S1':12},AVMR_main_rows=960,AVMR_thresholds=8,strict_less_than_count_checks=metric_checks,
        paired_per_output_rows=36,AVMR_pair_rows=576,independent_delta_checks=pair_checks,summary_checks=summary_checks,
        parent_ids=6,noise_ids_per_parent=[0,1],master_protected_hashes=len(reg['protected']),score_protected_hashes=len(sreg['protected']),
        no_new_models=True,no_new_quality_scores=True,source_sha256=sha(__file__),score_source_sha256=sha(CODE/'score.py'),
        score_aggregate_completion_sha256=sha(OUT/'score_aggregate_completion.json'),
        source_csv_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
    csvwrite(OUT/'audit/POST_contract_per_output.csv',mesh_rows)
    write(OUT/'audit/score_table_audit.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','new_GEO_POSTs','reused_POSTs','all60_POST_C_bytes','AVMR_main_rows','independent_delta_checks','master_protected_hashes','score_protected_hashes')}))
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registration',type=Path,default=OUT/'registration.json')
    ap.add_argument('--check-only',action='store_true',help='Import/AST availability only; no result audit or model calls.')
    ap.add_argument('--stage',choices=('chain','scores','all'),default='chain')
    args=ap.parse_args()
    if args.check_only:print(json.dumps(dict(status='PASS_IMPORT_CPU_ONLY',model_calls=0,GPU_calls=0)))
    else:
        if args.stage in ('chain','all'):run(args.registration)
        if args.stage in ('scores','all'):score_audit()
