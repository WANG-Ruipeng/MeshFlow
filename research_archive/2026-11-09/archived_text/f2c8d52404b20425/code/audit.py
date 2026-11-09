"""R2 source-interface audit. CPU, frozen functions, no model or mesh modification."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,ast,collections,csv,hashlib,json,sys,time,traceback,resource,subprocess,functools,concurrent.futures,multiprocessing
import numpy as np
OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
MINI=ROOT/'deliverables/R2_32_67_ANALYSIS_MINI_V1'
ORIG=MINI/'protocol_and_code/original'
DOMAINS=('SOURCE_INDEX','EXACT_COORD_UINT32')
ROLE=('COMMON_SMALL32','EXP_ONLY35','NEW_DEV32')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x
def write(p,x):
    p=Path(p);assert p.resolve().is_relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
    b=(json.dumps(clean(x),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if p.exists():
        if p.read_bytes()!=b:raise FileExistsError(p)
        return
    tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(b);tmp.replace(p)
def csvwrite(p,rs):
    rs=list(rs);fields=list(dict.fromkeys(k for r in rs for k in r))
    import io
    f=io.StringIO(newline='');w=csv.DictWriter(f,fields);w.writeheader()
    for r in rs:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);b=f.getvalue().encode('utf-8-sig')
    if p.exists() and p.read_bytes()!=b:raise FileExistsError(p)
    p.write_bytes(b)
def event(kind,**kw):
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(clean(dict(utc=time.time(),event=kind,**kw)),ensure_ascii=False)+'\n')
def path(p):return Path(p.replace('E:\\MeshFlow','/mnt/e/MeshFlow').replace('E:/MeshFlow','/mnt/e/MeshFlow').replace('\\','/'))
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
@functools.lru_cache(None)
def api():
    namespace={'np':np,'collections':collections};defs={}
    selections={'avmr_pmr.py':{'bit_equal','topology','nearest','metric','L','SETTINGS'},
        'valid_graph.py':{'actual_valid_graph','ILLEGAL','LEGAL'},'opposite_reach.py':{'opposite_reach'},
        'winding_solver.py':{'definition','orient_fixed_C','VERSION'}}
    for name,wanted in selections.items():
        p=OUT/'code/frozen'/name
        src=p.read_text('utf-8');tree=ast.parse(src);nodes=[]
        for n in tree.body:
            names=set()
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):names={n.name}
            elif isinstance(n,ast.Assign):names={t.id for t in n.targets if isinstance(t,ast.Name)}
            elif isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name):names={n.target.id}
            if names&wanted:nodes.append(n)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),namespace)
        defs[name]=[dict(symbol=getattr(n,'name',None),line=n.lineno,end=n.end_lineno) for n in nodes]
    return namespace,defs
def resources():
    import psutil
    return dict(utc=time.time(),logical_cores=os.cpu_count(),affinity=len(os.sched_getaffinity(0)),
        load=list(os.getloadavg()),available_RAM=psutil.virtual_memory().available,
        peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,threads=1,
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv'],capture_output=True,text=True).stdout)
def prepare():
    regpath=OUT/'protocol/registration.json'
    if regpath.exists():
        reg=read(regpath)
        for p,h in reg['protected'].items():assert sha(p)==h,('Protected input changed',p)
        assert sha(__file__)==reg['entry_sha256']
        return reg
    cc=[r for r in rows(MINI/'conditions.csv') if r['parent_role'] in ROLE]
    pp={r['parent_uid']:r for r in rows(MINI/'parents.csv') if r['parent_role'] in ROLE}
    qq={r['condition_id']:r for r in rows(MINI/'source_quality_and_interfaces.csv')}
    assert len(cc)==258 and len(pp)==99
    assert collections.Counter(c['parent_role'] for c in cc)=={'COMMON_SMALL32':91,'EXP_ONLY35':103,'NEW_DEV32':64}
    train={p['group_id'] for p in pp.values() if p['parent_role']!='NEW_DEV32'}
    dev={p['group_id'] for p in pp.values() if p['parent_role']=='NEW_DEV32'};assert not train&dev
    protected={str(MINI/n):sha(MINI/n) for n in ('parents.csv','conditions.csv','source_quality_and_interfaces.csv','metrics_per_output.csv','AVMR_PMR_all8.csv')}
    functions={}
    for name in ('avmr_pmr.py','valid_graph.py','opposite_reach.py','winding_solver.py','scorers/pair_audit.py','scorers/intersections.py','scorers/seams.py'):
        src=ORIG/name;dst=OUT/'code/frozen'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        b=src.read_bytes()
        if dst.exists():assert dst.read_bytes()==b
        else:dst.write_bytes(b)
        protected[str(src)]=sha(src);protected[str(dst)]=sha(dst)
        functions[name]=dict(original_snapshot=str(src),sha256=sha(src),local_snapshot=str(dst))
    jobs=[]
    for c in sorted(cc,key=lambda r:(ROLE.index(r['parent_role']),r['parent_uid'],r['condition_id'])):
        p=pp[c['parent_uid']];q=qq[c['condition_id']]
        inputs={'parent':path(p['source_npz']),'condition':path(c['condition_source']),'quality':path(q['quality_path'])}
        expected={'parent':p['source_existing_sha256'],'condition':c['condition_existing_sha256'],'quality':q['quality_existing_sha256']}
        actual={}
        for key,fp in inputs.items():
            h=protected.get(str(fp))
            if h is None:h=sha(fp);protected[str(fp)]=h
            assert h==expected[key],(key,str(fp),'Identity mismatch')
            actual[key]=dict(path=str(fp),sha256=h)
        jobs.append(dict(condition_id=c['condition_id'],parent_uid=c['parent_uid'],parent_role=c['parent_role'],
            condition_type=c['condition_type'],N=int(c['N']),K=int(c['K']),source_quality=c['source_quality'],
            in_small=c['in_small'],in_expanded=c['in_expanded'],inputs=actual))
    a,index=api()
    reg=dict(version='R2_SOURCE_INTERFACE_AUDIT_V1',created_utc=time.time(),jobs=jobs,protected=protected,
        entry_sha256=sha(__file__),frozen_code=functions,selected_AST_functions=index,settings=a['SETTINGS'],L=a['L'],
        primary_scope='194 unique training C plus64 NEW_DEV C; no old6 or generated mesh rescoring',
        topology_domains=list(DOMAINS),source_coordinate_contract='model_vertices/source_faces in original winding; canonical full_target proven byte equal through saved corner permutation',
        direction='AS_STORED_SOURCE and same frozen fixed-C free orientation used for export; diagnostic in memory only, no saved mesh mutation',
        boundary_domain='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY; all Y vertices only; no C leakage; strict <; same8 thresholds',
        graph='Same original actual_valid_graph and opposite_reach; original-index and bitwise-coordinate index separate, no tolerance weld or GEO',
        pair_legality='Reuse verified full source pair_contacts only after mesh bytes and frozen detector source hashes agree. Remap original face row IDs to C/Y-local IDs. No all-pair rescan.',
        budget=dict(conditions=258,topology_domains_per_condition=2,original_graph_and_oriented_graph_per_domain=2,
            orientation_calls=516,new_triangle_pair_tests=0,models=0,training=0,D0=0,GEO=0,new_noise=0,render=0),
        interpretation='Reference completion diagnoses the training example; not unique-GT style accuracy, not user-required boundary success.')
    write(regpath,reg);write(OUT/'protocol/resources_wsl_before.json',resources())
    event('REGISTERED',conditions=258,parents=99,source_files=len(protected))
    return reg
@functools.lru_cache(8)
def parent_data(p,quality):
    with np.load(p,allow_pickle=False) as z:d={k:z[k].copy() for k in ('full_target','source_faces','source_face_vertex_ids','source_corner_permutations','model_vertices','retained_to_original_face')}
    q=read(quality);tri=d['full_target'].reshape(-1,3,3)
    assert sha(p)==q['input_npz_sha256']
    assert hashlib.sha256(tri.tobytes()).hexdigest()==q['full_target_sha256']
    assert q['independent_verification'] is True
    assert bits(d['model_vertices'][d['source_face_vertex_ids']],tri)
    assert bits(np.take_along_axis(d['model_vertices'][d['source_faces']],d['source_corner_permutations'][...,None],axis=1),tri)
    for fp,h in q['identity']['detector_source_hashes'].items():
        assert sha(OUT/'code/frozen/scorers'/Path(fp).name)==h,('Detector source mismatch',fp)
    assert q['L']==api()[0]['L']
    assert q['receipt']['new_unique_pair_tests']==len(tri)*(len(tri)-1)//2
    return d,q
def partition_pairs(q,cids,yids):
    ci={int(v):i for i,v in enumerate(cids)};yi={int(v):i for i,v in enumerate(yids)}
    out={'CF':{'contacts':[]},'FF':{'contacts':[]}}
    for row in q['pair_contacts']:
        x,y=map(int,row['pair'])
        if x in ci and y in yi:p=[ci[x],yi[y]];which='CF'
        elif y in ci and x in yi:p=[ci[y],yi[x]];which='CF'
        elif x in yi and y in yi:p=sorted([yi[x],yi[y]]);which='FF'
        else:continue
        out[which]['contacts'].append(dict(pair=p,classification=row['classification'],source_face_pair=[x,y]))
    return out
def score(job):
    tick=time.perf_counter();dest=OUT/'conditions'/job['condition_id'];rp=dest/'result.json'
    identity=dict(job=job,entry_sha256=sha(__file__))
    if rp.exists():
        r=read(rp);assert r['identity']==identity;return dict(condition_id=job['condition_id'],status=r['status'],reused=True,seconds=0)
    if dest.exists():raise RuntimeError('Interrupted condition requires explicit inspection: '+str(dest))
    write(dest/'started.json',dict(identity=identity,utc=time.time(),pid=os.getpid()))
    try:
        a,_=api()
        d,q=parent_data(job['inputs']['parent']['path'],job['inputs']['quality']['path'])
        with np.load(job['inputs']['condition']['path'],allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
        cids=np.asarray(c['source_face_ids'],dtype=np.int64);yids=np.asarray(c['free_source_ids'],dtype=np.int64)
        N=job['N'];K=job['K'];order=np.r_[cids,yids];tri=d['full_target'].reshape(N,3,3)
        assert len(cids)==K and np.array_equal(np.sort(order),np.arange(N))
        V=d['model_vertices'];F=d['source_faces'][order];original=V[F]
        assert bits(c['C'],tri[cids]) and bits(original[:K],c['C_original'])
        assert np.array_equal(F[:K],c['C_original_source_vertex_ids'])
        assert np.array_equal(d['source_face_vertex_ids'][cids],c['C_source_vertex_ids'])
        assert np.isfinite(original).all() and original.dtype==np.float32
        assert bits(np.take_along_axis(original,d['source_corner_permutations'][order][...,None],axis=1),tri[order])
        pairs=partition_pairs(q,cids,yids)
        bid,pts,ba=a['topology'](c['C'],c['C_source_vertex_ids'])
        dist,nearest=a['nearest'](pts,tri[yids])
        boundary=[dict(source_vertex_id=int(v),distance_model=float(ds),distance_L_fraction=float(ds/a['L']),
            nearest_Y_source_face=int(yids[int(n)//3]),nearest_Y_canonical_corner=int(n)%3) for v,ds,n in zip(bid,dist,nearest)]
        vm=[dict(setting=n,threshold_model=t,scale=s,tau=v,**a['metric'](dist,t)) for n,t,s,v in a['SETTINGS']]
        graphs={}
        for domain in DOMAINS:
            if domain=='SOURCE_INDEX':v,f=V,F
            else:
                flat=np.ascontiguousarray(original).reshape(-1,3)
                _,first,inverse=np.unique(flat.view(np.uint32).reshape(-1,3),axis=0,return_index=True,return_inverse=True)
                v,f=flat[first],inverse.reshape(-1,3)
                assert bits(v[f],original)
            before,edges,reached=a['actual_valid_graph'](v,f,K,pairs)
            before.update(a['opposite_reach'](v[f],K,edges))
            # Same export orientation routine; geometry is never saved/repaired.
            orient=a['orient_fixed_C'](v,f,K);ff=orient['faces']
            assert np.array_equal(ff[:K],f[:K])
            assert np.array_equal(np.sort(ff,axis=1),np.sort(f,axis=1))
            after,ae,ar=a['actual_valid_graph'](v,ff,K,pairs);after.update(a['opposite_reach'](v[ff],K,ae))
            for key in before:
                if key not in ('C_F_valid_opposite_winding_edge_count','opposite_winding_reached_free_faces','opposite_winding_reached_free_area','opposite_winding_reached_free_area_fraction'):
                    assert before[key]==after[key],('Orientation changed invariant',domain,key)
            for e in ae:
                e['source_face_ids']=[int(order[i]) for i in e['incident_face_ids']]
            cf=[e for e in ae if e['is_C_F']]
            # Boundary coverage is descriptive, never a declaration that all C edges must join.
            edge_counts=collections.Counter(tuple(sorted((int(face[i]),int(face[(i+1)%3])))) for face in f[:K] for i in range(3))
            cb={e for e,n in edge_counts.items() if n==1}
            denom=sum(float(np.linalg.norm(v[u].astype(float)-v[w])) for u,w in cb)
            linked=[e for e in cf if e['valid_edge'] and (e['vertex_u'],e['vertex_v']) in cb]
            coverage=sum(e['length_model'] for e in linked)
            extra=dict(C_boundary_edge_count=len(cb),C_boundary_length_model=denom,valid_interface_on_C_boundary_count=len(linked),
                valid_interface_boundary_length_model=coverage,valid_interface_boundary_length_fraction=coverage/denom if denom else None,
                orientation_flips=int(np.count_nonzero(orient['flips'])),
                same_unordered_faces_after_orientation=True,C_original_order_unchanged=True)
            graphs[domain]=dict(as_stored=before,export_oriented=after,diagnostic=extra,
                orientation_summary=orient['summary'],edges=ae,reached_Y_source_faces=[int(order[i]) for i in ar],
                reference_boundary_not_user_required=True)
        # Every actual source shared boundary vertex must be available among Y vertices.
        source_y_vertices=set(map(int,F[K:].reshape(-1)))
        source_match=np.asarray([int(v) in source_y_vertices for v in bid])
        assert np.all(dist[source_match]==0)
        r=dict(status='PASS',identity=identity,metadata=job,boundary_audit=ba,boundary=boundary,AVMR=vm,graphs=graphs,
            source_index_boundary_vertices_in_Y=int(source_match.sum()),source_index_boundary_vertex_denominator=len(bid),
            cached_CF_contact_faces=len(pairs['CF']['contacts']),cached_FF_contact_faces=len(pairs['FF']['contacts']),
            pair_legality='INHERITED_COMPLETE_VERIFIED_SOURCE_SCAN',new_pair_tests=0,
            wall_seconds=time.perf_counter()-tick,peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if 'torch' in sys.modules:raise RuntimeError('Model framework imported')
        write(rp,r);return dict(condition_id=job['condition_id'],status='PASS',reused=False,seconds=r['wall_seconds'],peak_RSS=r['peak_RSS'])
    except Exception as exc:
        r=dict(status='BLOCKED',identity=identity,error=repr(exc),traceback=traceback.format_exc())
        write(rp,r)
        return dict(condition_id=job['condition_id'],status='BLOCKED',error=repr(exc),seconds=time.perf_counter()-tick)
def run():
    reg=prepare();jobs=reg['jobs'];start=time.perf_counter()
    first=score(jobs[0]);res=resources()
    workers=2 if first['status']=='PASS' and res['affinity']>=4 and res['available_RAM']>max(4<<30,4*first['peak_RSS']) and res['load'][0]<res['affinity']-2 else 1
    dispatch=dict(first_authorized_job=first,resources=res,CPU_workers=workers,threads_per_worker=1,GPU_workers=0,
        rationale='Measured actual first registered condition, RAM/load headroom; independent statistics in parallel; no extra probe.')
    write(OUT/'protocol/dispatch.json',dispatch);event('FIRST_CONDITION',**first)
    if first['status']!='PASS':
        print(json.dumps(first));return
    receipts=[first]
    if workers==1:
        for j in jobs[1:]:
            x=score(j);receipts.append(x);event('CONDITION',**x)
            if len(receipts)%25==0:print(json.dumps(dict(completed=len(receipts),last=x)),flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for x in pool.map(score,jobs[1:],chunksize=1):
                receipts.append(x);event('CONDITION',**x)
                if len(receipts)%25==0:print(json.dumps(dict(completed=len(receipts),last=x)),flush=True)
    after={p:sha(p) for p in reg['protected']};assert after==reg['protected']
    write(OUT/'protocol/protected_after.json',dict(status='PASS',files=after))
    write(OUT/'completion.json',dict(status='COMPLETE' if all(r['status']=='PASS' for r in receipts) else 'PARTIAL',
        condition_count=len(receipts),pass_count=sum(r['status']=='PASS' for r in receipts),receipts=receipts,wall_seconds=time.perf_counter()-start,
        workers=workers,model_loads=0,forwards=0,training=0,D0=0,GEO=0,new_pair_tests=0,modified_source_meshes=0))
    write(OUT/'protocol/resources_wsl_after.json',resources())
    aggregate()
    print(json.dumps(read(OUT/'completion.json'))[:1200])
def aggregate():
    reg=read(OUT/'protocol/registration.json');allrows=[];vm=[];bv=[];edge_rows=[];blocked=[]
    for job in reg['jobs']:
        rp=OUT/'conditions'/job['condition_id']/'result.json'
        if not rp.exists():blocked.append(dict(condition_id=job['condition_id'],status='NOT_RUN'));continue
        r=read(rp);m={k:job[k] for k in ('condition_id','parent_uid','parent_role','condition_type','N','K','source_quality')}
        if r['status']!='PASS':blocked.append(dict(m,status=r['status'],error=r['error']));continue
        for v in r['AVMR']:vm.append(dict(m,**v))
        for v in r['boundary']:bv.append(dict(m,**v))
        for domain,g in r['graphs'].items():
            for direction,key in (('AS_STORED_SOURCE','as_stored'),('EXPORT_ORIENTED','export_oriented')):
                allrows.append(dict(m,topology_domain=domain,direction=direction,**g[key],**g['diagnostic'],
                    source_index_boundary_vertices_in_Y=r['source_index_boundary_vertices_in_Y'],
                    source_boundary_vertex_count=r['source_index_boundary_vertex_denominator']))
            for e in g['edges']:
                if e['is_C_F']:edge_rows.append(dict(m,topology_domain=domain,**e))
    for name,table in [('source_interfaces',allrows),('AVMR_PMR_all8',vm),('boundary_vertices',bv),('C_Y_edge_details',edge_rows),('blocked',blocked)]:
        csvwrite(OUT/(name+'.csv'),table)
    def means(table,fields,groupkeys):
        groups=collections.defaultdict(list)
        for r in table:groups[tuple(r[k] for k in groupkeys)].append(r)
        out=[]
        for group,rs in sorted(groups.items()):
            parents=sorted({r['parent_uid'] for r in rs});r=dict(zip(groupkeys,group))
            r.update(kind='DERIVED',conditions=len(rs),parents=len(parents))
            for f in fields:
                pv=[np.mean([x[f] for x in rs if x['parent_uid']==p and x[f] is not None]) for p in parents if any(x['parent_uid']==p and x[f] is not None for x in rs)]
                r[f+'_parent_equal_mean']=float(np.mean(pv)) if pv else None
                r[f+'_condition_nonzero']=sum(x[f] is not None and x[f]>0 for x in rs)
                r[f+'_valid_conditions']=sum(x[f] is not None for x in rs)
            out.append(r)
        return out
    gfields=['C_F_has_valid_edge','C_F_valid_edge_count','C_F_valid_edge_total_length','reached_free_area_fraction','opposite_winding_reached_free_area_fraction','valid_interface_boundary_length_fraction','C_F_rejected_duplicate_edge_count','C_F_nonmanifold_edge_count']
    for typ in (False,True):
        keys=['parent_role','topology_domain','direction']+(['condition_type'] if typ else [])
        csvwrite(OUT/('interface_summary_by_type.csv' if typ else 'interface_summary.csv'),means(allrows,gfields,keys))
        csvwrite(OUT/('AVMR_summary_by_type.csv' if typ else 'AVMR_summary.csv'),means(vm,['VMR','perfect'],['parent_role','setting']+(['condition_type'] if typ else [])))
    # Compare only the same64 new C to the existing generated POST; no new generated score.
    source={(r['condition_id'],r['topology_domain'],r['direction']):r for r in allrows}
    vms={(r['condition_id'],r['setting']):r for r in vm};paired=[]
    output_fields=['C_F_has_valid_edge','C_F_valid_edge_count','C_F_valid_edge_total_length','reached_free_area_fraction','opposite_winding_reached_free_area_fraction']
    for r in rows(MINI/'metrics_per_output.csv'):
        if r['panel']!='NEW32' or r['stage']!='POST':continue
        for domain in DOMAINS:
            s=source.get((r['condition_id'],domain,'EXPORT_ORIENTED'))
            if not s:continue
            p=dict(kind='DERIVED_REFERENCE_COMPARISON_NOT_RECONSTRUCTION',parent_uid=r['parent_id'],condition_id=r['condition_id'],
                condition_type=r['region_type']+'_'+r['tier'],request_id=r['request_id'],noise_id=r['candidate'],label=r['label'],topology_domain=domain)
            for field in output_fields:p['source_'+field]=s[field];p['generated_'+field]=r[field]
            for setting in ('raw_tau_1e-6','L_tau_0.01','L_tau_0.02'):
                p['source_VMR_'+setting]=vms[r['condition_id'],setting]['VMR'];p['generated_VMR_'+setting]=r['A_VMR__'+setting]
            paired.append(p)
    csvwrite(OUT/'DEV_source_vs_existing_POST.csv',paired)
    write(OUT/'aggregation.json',dict(status='COMPLETE' if not blocked else 'PARTIAL',source_interface_rows=len(allrows),vm_rows=len(vm),
        boundary_vertex_rows=len(bv),C_Y_edge_rows=len(edge_rows),DEV_reference_pair_rows=len(paired),blocked=blocked,
        hierarchy='C within parent; parent equal; cohorts and4 condition types separate; source has no noise, duplicated only in explicit output-pair table',
        all_source_files_unchanged=True,new_generated_scores=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run','aggregate'],required=True);a=p.parse_args()
    if a.stage=='prepare':print(json.dumps(dict(prepared_conditions=len(prepare()['jobs']))))
    elif a.stage=='run':run()
    else:aggregate()
