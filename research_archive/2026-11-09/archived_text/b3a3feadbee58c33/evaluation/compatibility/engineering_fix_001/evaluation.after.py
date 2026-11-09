"""Isolated CPU evaluation: 48 fixed RAW x identity/D0/student, frozen scorers.

Invoke as a separate CPU process. D0 quality/topology are verified old caches;
96 new GEO and quality jobs write only below this pilot. Scientific arithmetic
is imported read-only from the audited anchor evaluation; no globals change.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, concurrent.futures, csv, json, multiprocessing, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.single_step_denoiser_distillation_v1 import common as C
from r2.pilots.single_step_denoiser_distillation_v1 import distill_metrics as DM
from r2.pilots.anchor_aware_denoiser_v1 import evaluation as E

OUT=C.OUT/'evaluation'
METHODS=('IDENTITY_GEO','D0_GEO','STUDENT_GEO')
VERSION='single_step_frozen_geometry_evaluation_v1'
SOURCE=Path(__file__).resolve()
ANCHOR_EVALUATION_SHA='ea0903dde2d96cdb11628b9e864f9b2ce2e827178628b2003a2dbd44fe7afab5'


def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def csvwrite(path,rows,empty_fields=()):
    if rows:C.csvwrite(path,E.clean(rows))
    else:
        if not empty_fields:raise ValueError('Empty table lacks schema')
        C.atomic(path,lambda f:csv.writer(f).writerow(empty_fields))


def write(path,obj):C.write(path,E.clean(obj))
def event(kind,**kw):
    if all(name in kw for name in ('method','label','request_id')):
        path=C.checked(OUT/'events'/kw['method']/kw['label']/kw['request_id']/(kind+'.json'))
        if path.exists():raise RuntimeError('Duplicate per-job event: '+str(path))
        write(path,dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**kw))
    else:C.event(kind,ledger='evaluation/ledger.jsonl',**kw)
def canonical(row):return E.canonical_identity(row)
def meta(row):return dict(E.meta(canonical(row)),method=row['method'],denoiser=row['method'])
def jobid(row):return dict(method=row['method'],label=row['label'],request_id=row['request_id'])
def key(row):return row['method'],row['label'],row['request_id']
def same(a,b):return E.same(a,b)
def payload(path):return E.payload(path)


def job_open(stage,row):
    path=OUT/'ledger.jsonl'
    events=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    events += [C.read(p) for p in (OUT/'events'/row['method']/row['label']/row['request_id']).glob('*.json')]
    relevant=[x['event'] for x in events if all(x.get(k)==v for k,v in jobid(row).items()) and x['event'].startswith(stage+'_')]
    if relevant:raise RuntimeError('Existing attempted '+stage+' lacks a validated completed artifact; no silent retry: '+str(jobid(row)))
    event(stage+'_STARTED',**jobid(row))


def old_paths(r):
    base=(r['label'],r['request_id'])
    den=C.OLD.joinpath('denoised',*base,'denoised.npz')
    post=C.OLD.joinpath('outputs',*base,'post.npz')
    score=C.OLD.joinpath('scores',*base,'score.json')
    diag=C.ANCHOR.joinpath('evaluation/diagnostics/D0_OFFICIAL',*base,'diagnostic.json')
    return den,post,score,diag


def prepare():
    target=OUT/'registration.json'
    if target.exists():
        reg=C.read(target);C.verify(reg['protected'])
        if C.sha(SOURCE)!=reg['evaluation_source_sha256']:
            import ast
            fix=C.read(OUT/'compatibility/engineering_fix_001.json')
            assert fix['original_source_sha256']==reg['evaluation_source_sha256']
            assert fix['revised_source_sha256']==C.sha(SOURCE) and fix['registration_sha256']==C.sha(target)
            for path,h in fix['files'].items():assert C.sha(path)==h
            def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            before=functions(Path(fix['original_source_snapshot']).read_text());after=functions(SOURCE.read_text())
            changed=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
            assert changed==['event','fidelity','job_open','prepare']
            assert fix['no_scientific_definition_change'] and fix['no_model_or_quality_replay']
        return reg
    A,Q,_,config=E.helpers()
    assert C.sha(Path(E.__file__))==ANCHOR_EVALUATION_SHA
    old=C.read(C.OLD/'registration.json')
    assert len(old['records'])==48 and old['postprocess_config']==config
    assert old['settings']==[dict(name=n,threshold_model=t,scale=s,tau=v) for n,t,s,v in A.SETTINGS]
    protected={}
    def protect(p):
        p=Path(p);protected[str(p)]=C.sha(p)
    for p in [Path(E.__file__),Path(DM.__file__),C.OLD/'registration.json',C.ANCHOR/'evaluation/registration.json',
              C.ROOT/'r2/tools/avmr_pmr_v1/run.py',C.ROOT/'r2/tools/postprocess_avmr_pmr_v1/quality.py',
              C.ROOT/'native_t1/postprocessing.py',C.ROOT/'native_t1/postprocess_cli.py',
              C.ANCHOR/'data_audit/dev_conditions.json',
              C.ROOT/'experiments/r2_user_mix_from_official_v1/evaluation/per_output.csv',
              C.ROOT/'experiments/r2_avmr_pmr_offline_v1/per_output.csv',*Q.protected_paths()]:protect(p)
    for interface in C.read(C.ANCHOR/'data_audit/dev_conditions.json')['rows']:protect(interface['source_interface_json'])
    for r in old['records']:
        den,post,score,diag=old_paths(r)
        s,d=C.read(score),C.read(diag)
        assert s['status']==d['status']=='PASS'
        assert s['identity']['raw_sha256']==r['raw_sha256'] and s['identity']['post_sha256']==C.sha(post)
        assert d['identity']['post_sha256']==C.sha(post) and d['identity']['denoised_sha256']==C.sha(den)
        assert d['identity']['label']==r['label'] and d['identity']['request_id']==r['request_id']
        assert d['quality']==s['quality']
        for mapping in (s['files'],d['files'],d['quality']['files']):
            E.verify_files(mapping)
            for p in mapping:protect(p)
        for p in (den,den.with_name('receipt.json'),post,post.with_name('post.ply'),post.with_name('postprocess.json'),
                  score,diag,Path(r['raw_path']),Path(r['condition_source'])):protect(p)
    reg=dict(version=VERSION,status='REGISTERED',records=old['records'],methods=METHODS,
        expected_POST=144,new_GEO=96,new_quality=96,reused_D0_POST_quality_topology=48,
        settings=old['settings'],postprocess_config=config,L=A.L,quality_definition=Q.definition(),
        quality_source_hashes=Q.source_hashes(),evaluation_source_sha256=C.sha(SOURCE),protected=protected,
        source_face_identity='Original fixed RAW face ordinal; source-corner identities preserved by GEO',
        topology_definition=C.read(C.ANCHOR/'evaluation/registration.json')['valid_edge_definition'],
        reachable_area_definition=C.read(C.ANCHOR/'evaluation/registration.json')['reachable_area_definition'],
        additional_GEO_sensitivity='Source-face deletion, exact C-endpoint valid-edge loss/new, actual corner target/status changes, and input point-pair radius membership crossings. Crossings explain sensitivity; do not attribute causality through greedy representative selection.',
        no_unique_source_reconstruction=True,distillation=DM.definition(),CPU_workers=2,threads_per_worker=2)
    write(target,reg);write(OUT/'resources_before.json',C.resources());event('REGISTERED',new_quality=96,new_GEO=96)
    return reg


def pre_input(r,method):
    if method=='IDENTITY_GEO':return Path(r['raw_path']),'raw','IDENTITY_NO_MODEL'
    if method=='D0_GEO':return old_paths(r)[0],'output',C.EXPECTED
    p=C.OUT/'inference'/C.STUDENT/r['label']/r['request_id']/'denoised.npz'
    rec=C.read(p.with_name('denoised.json'));ident=rec.get('identity',rec)
    assert rec['status']=='PASS' and rec.get('output_sha256',ident.get('output_sha256'))==C.sha(p)
    for name in ('label','request_id','raw_sha256'):assert ident[name]==r[name]
    h=ident.get('model_checkpoint_sha256',ident.get('checkpoint_sha256'))
    assert h
    a=payload(p);raw=payload(r['raw_path'])['raw'].reshape(-1,3,3);K=int(r['K'])
    assert same(a['output'][:K],raw[:K]) and same(a['C'],raw[:K])
    assert a['output'].shape==raw.shape and a['output'].dtype==np.float32 and np.isfinite(a['output']).all()
    proposal=a.get('proposal_S_model',a.get('proposal_model',a.get('proposal')))
    assert same(proposal[K:],a['output'][K:])
    return p,'output',h


def postprocess(methods=METHODS):
    reg=prepare();_,_,PC,_=E.helpers();rows=[]
    for method in methods:
        for orig in reg['records']:
            row=dict(orig,method=method,denoiser=method)
            den,input_key,modelsha=pre_input(row,method)
            assert C.sha(row['raw_path'])==row['raw_sha256']
            if method=='D0_GEO':post,diag=old_paths(row)[1],old_paths(row)[3]
            else:
                post=OUT/'outputs'/method/row['label']/row['request_id']/'post.npz'
                diag=OUT/'diagnostics'/method/row['label']/row['request_id']/'diagnostic.json'
                dest=C.checked(post.parent)
                if (dest/'postprocess.json').exists():
                    pp=C.read(dest/'postprocess.json')
                    assert pp['status']=='PASS' and pp['input']['file_sha256']==C.sha(den)
                    for name,v in pp['files'].items():assert C.sha(dest/name)==v['sha256']
                else:
                    stage=C.checked(dest.with_name(dest.name+'.staging'))
                    if dest.exists() or stage.exists():raise RuntimeError('Incomplete GEO artifact; no implicit retry')
                    job_open('GEO',row)
                    pp=PC.run(den,int(row['K']),stage,input_key=input_key)
                    write(stage/'postprocess.json',E.rewrite_paths(pp,stage,dest));os.replace(stage,dest)
                    event('GEO_COMPLETE',**jobid(row),seconds=pp['seconds'])
            postdata=payload(post);raw=payload(row['raw_path'])['raw'].reshape(-1,3,3)
            assert same(postdata['output'][:int(row['K'])],raw[:int(row['K'])])
            assert same(postdata['vertices'][postdata['faces']],postdata['output'])
            vertices,faces=PC.read_indexed_ply(post.with_name('post.ply'))
            assert same(vertices,postdata['vertices']) and same(faces,postdata['faces'])
            row.update(denoised_path=str(den),pre_GEO_path=str(den),pre_GEO_key=input_key,post_path=str(post),
                post_ply=str(post.with_name('post.ply')),post_sha256=C.sha(post),output_N=len(postdata['output']),
                diagnostic_path=str(diag),denoiser_checkpoint_sha256=modelsha,stage='POST')
            rows.append(row)
    expected=48*len(methods);assert len(rows)==expected
    csvwrite(OUT/'post_index.csv',rows)
    complete=tuple(methods)==METHODS
    name='post_completion.json' if complete else 'post_identity_completion.json'
    write(OUT/name,dict(status='PASS',scope='FULL' if complete else 'IDENTITY_AND_D0_ONLY',primary_POST=expected,
        new_GEO=48*sum(m!='D0_GEO' for m in methods),reused_D0=48*int('D0_GEO' in methods),C_and_indexed_PLY_bitwise=True))
    return rows


def score_one(row):
    row=canonical(row);diag=Path(row['diagnostic_path']);post=Path(row['post_path']);den=Path(row['denoised_path'])
    if row['method']=='D0_GEO':
        d=C.read(diag);assert d['status']=='PASS' and d['identity']['post_sha256']==C.sha(post)
        E.verify_files(d['files']);E.verify_files(d['quality']['files']);return str(diag)
    ident=dict(meta(row),post_sha256=C.sha(post),input_sha256=C.sha(den),registration_sha256=C.sha(OUT/'registration.json'))
    if diag.exists():
        d=C.read(diag);assert d['status']=='PASS' and canonical(d['identity'])==canonical(ident)
        E.verify_files(d['files']);E.verify_files(d['quality']['files']);return str(diag)
    A,Q,_,_=E.helpers();score=OUT/'scores'/row['method']/row['label']/row['request_id']/'score.json'
    if score.exists():
        quality=C.read(score);assert quality['status']=='PASS' and quality['identity']['processed_sha256']==C.sha(post)
        E.verify_files(quality['files'])
    else:
        dest=C.checked(score.parent);stage=C.checked(dest.with_name(dest.name+'.staging'))
        if dest.exists() or stage.exists():raise RuntimeError('Unfinished quality output; no implicit retry')
        job_open('QUALITY',row);quality=Q.score_quality(row,post,C.checked(stage/'quality'))
        quality=E.rewrite_paths(quality,stage,dest);write(stage/'score.json',quality);os.replace(stage,dest)
        E.verify_files(quality['files']);event('QUALITY_COMPLETE',**jobid(row),seconds=quality['seconds'])
    a=payload(post);before=payload(den)[row['pre_GEO_key']].reshape(-1,3,3);K=int(row['K'])
    topology,edges,reached=E.actual_valid_graph(a['vertices'],a['faces'],K,C.read(quality['pair_audit_path']))
    verts,metrics=E.boundary_values(row,a['output'],A);pv,pm=E.boundary_values(row,before,A)
    shape=Q._old().cpu_math()['shape'].shape_quality(before[K:],A.L)
    out=C.checked(diag.parent)
    write(out/'pre_GEO_shape_per_face.json',shape)
    csvwrite(out/'valid_edge_details.csv',edges,('vertex_u','vertex_v','valid_edge'))
    d=dict(status='PASS',identity=ident,quality=quality,topology=topology,vertices=verts,metrics=metrics,
        pre_GEO_vertices=pv,pre_GEO_metrics=pm,
        pre_GEO_shape=dict(free_faces=len(before)-K,free_total_surface_area=shape['total_area'],
            free_minimum_angle_lt5_fraction=shape['min_angle_lt5_fraction'],free_q_gt10_fraction=shape['q_actual_gt10_fraction'],
            free_exact_zero_area_count=shape['exact_zero_area_count'],free_near_degenerate_count=shape['near_degenerate_area_count']),
        reached_free_output_face_ids=reached,source_face_ids=a['source_face_ids'].tolist(),
        postprocess=C.read(post.with_name('postprocess.json')),files={str(p):C.sha(p) for p in out.iterdir() if p.is_file()})
    write(diag,d);return str(diag)


def scores(require_full=True):
    reg=prepare();rows=readcsv(OUT/'post_index.csv');assert len(rows)==(144 if require_full else 96)
    if not require_full:assert {r['method'] for r in rows}=={'IDENTITY_GEO','D0_GEO'}
    write(OUT/'resources_before_score.json',C.resources());start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=list(pool.map(score_one,rows,chunksize=1))
    C.verify(reg['protected'])
    name='score_completion.json' if require_full else 'score_identity_completion.json'
    write(OUT/name,dict(status='PASS',scope='FULL' if require_full else 'IDENTITY_AND_D0_ONLY',receipts=receipts,
        new_quality=96 if require_full else 48,reused_D0=48,model_calls=0,new_queries=0,CPU_workers=2,
        threads_per_worker=2,seconds=time.perf_counter()-start))


def _edge_keys(post,diagnostic):
    a=payload(post);K=int(diagnostic['identity']['K'])
    _,edges,_=E.actual_valid_graph(a['vertices'],a['faces'],K,C.read(diagnostic['quality']['pair_audit_path']))
    result={}
    for e in edges:
        if e['valid_edge'] and e['is_C_F']:
            # A valid C/F shared edge has immutable C endpoints. Byte coordinate
            # keys compare it across output vertex reindexing and deleted faces.
            bits=sorted(a['vertices'][i].tobytes().hex() for i in (e['vertex_u'],e['vertex_v']))
            result[tuple(bits)]=e
    return result


def sensitivity(row,diagnostic,baseline_row,baseline):
    from scipy.spatial import cKDTree
    a,b=payload(row['post_path']),payload(baseline_row['post_path']);K=int(row['K'])
    m=meta(row);faces=[];corners=[];events=[];crossings=[]
    for i,(keep_s,keep_t) in enumerate(zip(a['source_face_kept'],b['source_face_kept'])):
        if i<K:continue
        faces.append(dict(**m,raw_source_face=i,method_kept=bool(keep_s),D0_kept=bool(keep_t),
            method_reason=str(a['source_face_reason'][i]),D0_reason=str(b['source_face_reason'][i]),
            fate='BOTH_KEPT' if keep_s and keep_t else ('METHOD_ONLY_DELETION' if keep_t else ('D0_ONLY_DELETION' if keep_s else 'BOTH_DELETED'))))
    for i in range(K*3,len(a['corner_status'].reshape(-1))):
        af,bf=a['corner_status'].reshape(-1),b['corner_status'].reshape(-1)
        at,bt=a['corner_target_input_ids'].reshape(-1),b['corner_target_input_ids'].reshape(-1)
        if af[i]!=bf[i] or at[i]!=bt[i]:
            corners.append(dict(**m,raw_corner_id=i,method_status=str(af[i]),D0_status=str(bf[i]),method_target_input_id=int(at[i]),D0_target_input_id=int(bt[i])))
    es,et=_edge_keys(row['post_path'],diagnostic),_edge_keys(baseline_row['post_path'],baseline)
    for edge in sorted(set(es)|set(et)):
        events.append(dict(**m,C_endpoint_a_bits=edge[0],C_endpoint_b_bits=edge[1],
            fate='RETAINED' if edge in es and edge in et else ('D0_VALID_EDGE_LOST' if edge in et else 'NEW_VALID_EDGE'),
            method_winding_opposite=es.get(edge,{}).get('winding_opposite'),D0_winding_opposite=et.get(edge,{}).get('winding_opposite')))
    before_s=payload(row['denoised_path'])[row['pre_GEO_key']].reshape(-1,3).astype(np.float64)
    before_t=payload(baseline_row['denoised_path'])[baseline_row['pre_GEO_key']].reshape(-1,3).astype(np.float64)
    radius=C.read(OUT/'registration.json')['postprocess_config']['radius_model']
    # Candidate union only changes CPU speed. Membership is decided by explicit
    # FP64 squared distances <= frozen radius squared, same numerical comparator.
    candidates=set(cKDTree(before_s).query_pairs(radius*(1+1e-12)))|set(cKDTree(before_t).query_pairs(radius*(1+1e-12)))
    for i,j in sorted(candidates):
        if j<K*3:continue
        ds=float(np.sum((before_s[i]-before_s[j])**2));dt=float(np.sum((before_t[i]-before_t[j])**2))
        within_s,within_t=ds<=radius*radius,dt<=radius*radius
        if within_s!=within_t:
            crossings.append(dict(**m,raw_corner_a=i,raw_corner_b=j,domain='C_F' if i<K*3 else 'F_F',
                method_distance=float(np.sqrt(ds)),D0_distance=float(np.sqrt(dt)),radius_model=radius,
                method_within=within_s,D0_within=within_t))
    fc=collections.Counter(x['fate'] for x in faces);ec=collections.Counter(x['fate'] for x in events)
    summary=dict(**m,**{k:fc[k] for k in ('BOTH_KEPT','METHOD_ONLY_DELETION','D0_ONLY_DELETION','BOTH_DELETED')},
        **{k:ec[k] for k in ('RETAINED','D0_VALID_EDGE_LOST','NEW_VALID_EDGE')},
        corner_target_or_status_changes=len(corners),radius_membership_crossings=len(crossings),
        radius_C_F_crossings=sum(x['domain']=='C_F' for x in crossings))
    return summary,faces,corners,events,crossings


def aggregate():
    reg=prepare();rows=[canonical(r) for r in readcsv(OUT/'post_index.csv')]
    ds={key(r):C.read(r['diagnostic_path']) for r in rows};rs={key(r):r for r in rows}
    quality=[];topology=[];vm=[];supported=[];vertices=[];preshape=[];mods=[];distance=[]
    fate=[];fateevents=[];sens=[];faceevents=[];cornerevents=[];edgeevents=[];crossings=[]
    for r in rows:
        d=ds[key(r)];m=meta(r);K=r['K']
        quality.append(dict(**m,stage='POST',**d['quality']['metrics'],free_face_denominator=d['quality']['free_face_denominator']))
        topology.append(dict(**m,stage='POST',**d['topology']))
        preshape.append(dict(**m,stage='PRE_GEO_C_RESTORED',**d['pre_GEO_shape']))
        for stage,mk,vk in (('POST','metrics','vertices'),('PRE_GEO_C_RESTORED','pre_GEO_metrics','pre_GEO_vertices')):
            for v in d[mk]:(vm if v['domain']=='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY' else supported).append(dict(**m,stage=stage,**v))
            vertices.extend(dict(**m,stage=stage,**v) for v in d[vk])
            vv=[v['distance_model'] for v in d[vk]];valid=bool(vv) and all(x is not None for x in vv)
            distance.append(dict(**m,stage=stage,boundary_vertices=len(vv),status='PASS' if valid else 'NA_EMPTY_BOUNDARY_OR_FREE',
                distance_mean_model=float(np.mean(vv)) if valid else None,distance_median_model=float(np.median(vv)) if valid else None,
                distance_p95_model=float(np.quantile(vv,.95)) if valid else None))
        pre=payload(r['denoised_path'])[r['pre_GEO_key']].reshape(-1,3,3);post=payload(r['post_path'])
        raw=payload(r['raw_path'])['raw'].reshape(-1,3,3);drift=np.linalg.norm(pre[K:].astype(float)-raw[K:],axis=-1)
        mods.append(dict(**m,raw_free_faces=len(raw)-K,POST_free_faces=len(post['output'])-K,
            deleted_free_faces=len(raw)-len(post['output']),deleted_free_fraction=(len(raw)-len(post['output']))/(len(raw)-K),
            denoise_free_corner_displacement_mean=float(drift.mean()),denoise_free_corner_displacement_p95=float(np.quantile(drift,.95)),
            denoise_free_corner_displacement_max=float(drift.max()),GEO_free_corner_displacement_mean=float(post['corner_displacement'][K:].mean()),
            GEO_free_corner_displacement_max=float(post['corner_displacement'][K:].max()),POST_free_surface_area=d['quality']['metrics']['free_total_surface_area']))
        if r['method']!='D0_GEO':
            bk=('D0_GEO',r['label'],r['request_id']);fr,fe=E.ff_fates(r,d,ds[bk]);fr['method']=r['method']
            fate.append(fr);fateevents.extend(fe)
            s,f,c,e,x=sensitivity(r,d,rs[bk],ds[bk]);sens.append(s);faceevents.extend(f);cornerevents.extend(c);edgeevents.extend(e);crossings.extend(x)
    tables={'quality_per_output':quality,'topology_per_output':topology,'per_output':vm,'source_supported_reference_per_output':supported,
        'per_boundary_vertex':vertices,'pre_GEO_shape_per_output':preshape,'modifications_per_output':mods,'boundary_distance_per_output':distance,
        'FF_fates_per_output':fate,'FF_pair_fates':fateevents,'GEO_sensitivity_per_output':sens,'GEO_source_face_events':faceevents,
        'GEO_corner_changes':cornerevents,'GEO_valid_edge_events':edgeevents,'GEO_radius_membership_crossings':crossings}
    for name,table in tables.items():csvwrite(OUT/(name+'.csv'),table,('method','label','request_id','status'))
    qnum=sorted({k for r in quality for k,v in r.items() if k not in E.META and isinstance(v,(int,float))})
    tnum=sorted({k for r in topology for k,v in r.items() if k not in E.META and isinstance(v,(int,float))})
    for name,table,numeric,base in (('quality',quality,qnum,('method','label')),('topology',topology,tnum,('method','label')),
        ('AVMR',vm,('VMR','perfect'),('method','label','stage','setting'))):
        sums=[]
        for group,extra in (('ALL',()),('REGION',('region_type',)),('PARENT',('parent_id',)),('NOISE',('candidate',)),('PARENT_NOISE',('parent_id','candidate'))):
            sums.extend(dict(grouping=group,**r) for r in E.summaries(table,base+extra,numeric))
        csvwrite(OUT/(name+'_summary.csv'),sums)
    paired=[];vpaired=[]
    qmap={key(r):r for r in quality};tmap={key(r):r for r in topology}
    for newer,older in (('STUDENT_GEO','D0_GEO'),('IDENTITY_GEO','D0_GEO'),('STUDENT_GEO','IDENTITY_GEO')):
        for r in quality:
            if r['method']!=newer:continue
            oldkey=(older,r['label'],r['request_id']);p=dict(E.meta(r),comparison=newer+'_minus_'+older)
            for num,new,old in ((qnum,r,qmap[oldkey]),(tnum,tmap[key(r)],tmap[oldkey])):
                for field in num:p[field+'_delta']=float(new[field])-float(old[field]) if new[field] is not None and old[field] is not None else None
            paired.append(p)
        refs={(r['label'],r['request_id'],r['stage'],r['setting']):r for r in vm if r['method']==older}
        for r in vm:
            if r['method']!=newer:continue
            base=refs[(r['label'],r['request_id'],r['stage'],r['setting'])]
            vpaired.append(dict(E.meta(r),comparison=newer+'_minus_'+older,stage=r['stage'],setting=r['setting'],
                **{v+'_delta':r[v]-base[v] if r[v] is not None and base[v] is not None else None for v in ('VMR','perfect')}))
    csvwrite(OUT/'paired_per_output.csv',paired)
    csvwrite(OUT/'paired_summary.csv',E.summaries(paired,('comparison','label'),[k for k in paired[0] if k.endswith('_delta')]))
    csvwrite(OUT/'AVMR_paired_per_output.csv',vpaired)
    csvwrite(OUT/'AVMR_paired_summary.csv',E.summaries(vpaired,('comparison','label','stage','setting'),('VMR_delta','perfect_delta')))
    csvwrite(OUT/'RAW_quality_reference.csv',readcsv(C.ROOT/'experiments/r2_user_mix_from_official_v1/evaluation/per_output.csv'))
    csvwrite(OUT/'RAW_AVMR_reference.csv',readcsv(C.ROOT/'experiments/r2_avmr_pmr_offline_v1/per_output.csv'))
    assert len(quality)==len(topology)==144 and len(vm)==2304 and len(vertices)==2976
    C.verify(reg['protected'])
    write(OUT/'completion.json',dict(status='PASS',primary_POST=144,new_quality=96,new_GEO=96,D0_reused=48,
        primary_AVMR_rows=len(vm),boundary_vertices=len(vertices),FF_events=len(fateevents),GEO_face_events=len(faceevents),
        GEO_edge_events=len(edgeevents),GEO_radius_crossings=len(crossings),model_calls=0,new_queries=0,protected_files=len(reg['protected'])))
    event('EVALUATION_COMPLETE',primary_POST=144,new_quality=96)


def fidelity():
    manifest=C.read(C.OUT/'manifest/data.json');rows=[]
    complete=C.read(C.OUT/'inference/completion.json')
    assert complete['status']=='COMPLETE' and len(complete['rows'])==66
    student_index={r['sample_id']:r for r in complete['rows']}
    for domain in ('dev_real','dev_synthetic'):
        for item in manifest[domain]:
            sample=item['sample_id'];teacher=Path(item['teacher_cache_path'])
            sr=student_index[sample];student=Path(sr['student_cache_path'])
            assert C.sha(student)==sr['student_cache_sha256'] and C.sha(sr['student_receipt_path'])==sr['student_receipt_sha256']
            tr=C.read(teacher.with_suffix('.json'))
            assert tr['status']=='PASS' and C.sha(teacher)==tr['sha256'] and C.sha(item['input_path'])==tr['input_sha256']
            t,s=payload(teacher),payload(student)
            assert same(t['input_d'],s['input_d'])
            target=None
            if domain=='dev_synthetic':
                assert C.sha(item['source_input_path'])==item['source_input_sha256']
                target=payload(item['source_input_path'])['target_d']
            metric=DM.residual_metrics(t['input_d'],t['proposal_T_d'],s.get('proposal_S_d',s.get('proposal_d')),int(item['K']),
                t['proposal_T_model'],s.get('proposal_S_model',s.get('proposal_model')),target)
            row={k:item.get(k) for k in ('sample_id','uid','condition_id','label','request_id','region_type','tier','candidate','variant','domain')}
            row.update(phase=domain,teacher_path=str(teacher),teacher_sha256=C.sha(teacher),student_path=str(student),student_sha256=C.sha(student),**metric)
            rows.append(row)
    csvwrite(OUT/'distillation_per_output.csv',rows)
    sums=[]
    for grouping,keys in (('ALL',('phase',)),('GENERATOR',('phase','label')),('PARENT',('phase','label','uid')),
                          ('NOISE',('phase','label','candidate')),('VARIANT',('phase','variant'))):
        sums.extend(dict(grouping=grouping,**r) for r in DM.summarize(rows,keys))
    csvwrite(OUT/'distillation_summary.csv',sums)
    write(OUT/'fidelity_completion.json',dict(status='PASS',outputs=len(rows),real_outputs=48,synthetic_outputs=18,definition=DM.definition()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','post','score','aggregate','fidelity','self-check','identity','all'),required=True);args=p.parse_args()
    if args.stage=='self-check':
        C.safety_checks();write(OUT/'arithmetic_self_check.json',DM.self_check());print('PASS CPU arithmetic and output guard checks; zero model/scorer calls')
    elif args.stage=='identity':
        postprocess(methods=('IDENTITY_GEO','D0_GEO'));scores(require_full=False)
    elif args.stage=='all':
        postprocess();scores();aggregate();fidelity()
    else:globals()[{'post':'postprocess','score':'scores'}.get(args.stage,args.stage)]()


if __name__=='__main__':main()
