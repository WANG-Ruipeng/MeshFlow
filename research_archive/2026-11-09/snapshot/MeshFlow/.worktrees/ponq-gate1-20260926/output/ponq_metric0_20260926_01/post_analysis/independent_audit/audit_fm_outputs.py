"""Independent read-only FM artifact audit; write only supplemental post-analysis."""
from pathlib import Path
import argparse, collections, hashlib, json, time, shutil
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def write(p, value): Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def valid(r):
    g=r.get('graph_capacity_stats',{})
    return bool(r.get('status')=='EXTRACTED' and r.get('closed_oriented_2manifold') and
                r.get('strict_all_vertex_links_simple_cycles') and
                [r.get('beta'+str(i)) for i in range(3)]==[1,0,1] and
                all(r.get(k)==0 for k in ['exact_intersection_pairs','exact_degenerate_faces','duplicate_faces']) and
                g.get('nonfinite_count')==0 and g.get('negative_count')==0)
def failure_tags(r):
    tags=[]
    if r.get('status')!='EXTRACTED':tags.append('pipeline_incomplete')
    if not r.get('closed_oriented_2manifold'):tags.append('not_closed_oriented_2manifold')
    if not r.get('strict_all_vertex_links_simple_cycles'):tags.append('vertex_links_not_cycles')
    if [r.get('beta'+str(i)) for i in range(3)]!=[1,0,1]:tags.append('betti_mismatch')
    for k in ['exact_intersection_pairs','exact_degenerate_faces','duplicate_faces']:
        if r.get(k)!=0:tags.append(k)
    for k in ['nonfinite_count','negative_count']:
        if r.get('graph_capacity_stats',{}).get(k)!=0:tags.append('graph_'+k)
    return tags
def tensor_equal(a,b): return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
def stats(x):
    a=np.asarray(x,dtype=float)
    return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),maximum=float(a.max()),mean=float(a.mean())) if len(a) else dict(count=0)
def site_cd(a,b): return float(.5*(np.mean(cKDTree(a).query(b)[0]**2)+np.mean(cKDTree(b).query(a)[0]**2)))


def independent_topology(v,f):
    edge=collections.defaultdict(list); links=collections.defaultdict(list)
    for a,b,c in f:
        for x,y in [(a,b),(b,c),(c,a)]: edge[tuple(sorted((int(x),int(y))))].append((int(x),int(y)))
        for x,y,z in [(a,b,c),(b,c,a),(c,a,b)]: links[int(x)].append((int(y),int(z)))
    badlinks=[]
    for vid,pairs in links.items():
        adj=collections.defaultdict(list)
        for a,b in pairs:adj[a].append(b);adj[b].append(a)
        seen=set(); stack=[next(iter(adj))]
        while stack:
            x=stack.pop()
            if x in seen:continue
            seen.add(x);stack.extend(adj[x])
        if any(len(n)!=2 for n in adj.values()) or len(seen)!=len(adj):badlinks.append(vid)
    adj=collections.defaultdict(set)
    for a,b in edge:adj[a].add(b);adj[b].add(a)
    seen=set();components=0
    for x in adj:
        if x in seen:continue
        components+=1;stack=[x]
        while stack:
            u=stack.pop()
            if u in seen:continue
            seen.add(u);stack.extend(adj[u])
    boundary=sum(len(e)==1 for e in edge.values()); nonmanifold=sum(len(e)>2 for e in edge.values())
    orientation=sum(len(e)==2 and e[0]!=e[1][::-1] for e in edge.values())
    chi=len(adj)-len(edge)+len(f)
    closed=boundary==nonmanifold==orientation==0 and not badlinks
    return dict(used_vertices=len(adj),faces=len(f),edges=len(edge),components=components,euler=chi,
                boundary_edges=boundary,nonmanifold_edges=nonmanifold,inconsistent_oriented_edges=orientation,
                bad_vertex_links=badlinks,closed_oriented_strict=closed,
                betti_if_closed_oriented=[components,2*components-chi,components] if closed else None)


def main(root):
    start=time.perf_counter();root=Path(root).resolve();out=root/'post_analysis'/'independent_audit'
    if out.exists():raise RuntimeError('Refuse existing post_analysis directory')
    checks=[]
    def check(name,ok,detail=None): checks.append(dict(name=name,passed=bool(ok),detail=detail))
    protocol=read(root/'RUN_PROTOCOL.json');execution=read(root/'EXECUTION.json')
    check('execution_complete',execution['status']=='COMPLETE')
    manifest=read(root/'MANIFEST.json');manifest_bad=[]
    for item in manifest['files']:
        p=root/item['path']
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:manifest_bad.append(item['path'])
    check('original_manifest',not manifest_bad,dict(files=len(manifest['files']),bad=manifest_bad))
    check('run_protocol_sha256',sha(root/'RUN_PROTOCOL.json')==read(root/'RUN_PROTOCOL_SHA256.json')['sha256'])
    check('source_snapshot',all(sha(root/i['path'])==i['sha256'] for i in read(root/'SOURCE_SNAPSHOT.json')['files']))
    source_hashes={str(p.relative_to(root)):sha(p) for p in [root/'RUN_PROTOCOL.json',root/'MANIFEST.json',root/'LEDGER.json',root/'SUMMARY_FM.json',root/'data.npz']}
    data=arrays(root/'data.npz');states=data['states'];check('target_shape',states.shape==(24,256,22))
    arms=['direct','transport_residual'];results={};traces={};initial={};ckpts={};training={}
    for arm in arms:
        d=root/'training'/arm;r=read(d/'RESULT.json');p=read(d/'PROTOCOL.json')
        trace=[json.loads(l) for l in (d/'TRAIN_TRACE.jsonl').read_text().splitlines()]
        ck=torch.load(d/'final.pt',map_location='cpu',weights_only=False)
        ini=torch.load(d/'initial_model.pt',map_location='cpu',weights_only=True)
        results[arm]=r;traces[arm]=trace;initial[arm]=ini;ckpts[arm]=ck
        check(arm+'_3000_updates',r['status']=='COMPLETED' and r['completed_steps']==3000 and r['attempted_batches']==3000 and len(trace)==3000 and [t['step'] for t in trace]==list(range(1,3001)))
        check(arm+'_checkpoint_binding',sha(d/'final.pt')==r['final_checkpoint_sha256'] and sha(d/'PROTOCOL.json')==r['protocol_sha256']==ck['protocol_sha256'] and sha(d/'initial_model.pt')==r['initial_model_sha256'])
        check(arm+'_checkpoint_completed',ck['completed_steps']==3000 and ck['status']=='COMPLETED' and all(int(s['step'])==3000 for s in ck['optimizer_state_dict']['state'].values()))
        check(arm+'_data_and_chart_binding',p['data_sha256']==sha(root/'data.npz') and np.array_equal(ck['mean'].numpy(),data['mean'].astype(np.float32)) and np.array_equal(ck['scale'].numpy(),data['scale'].astype(np.float32)))
        digest=bytes(32)
        for t in trace:digest=hashlib.sha256(digest+bytes.fromhex(t['batch_sha256'])).digest()
        check(arm+'_trace_stream_digest',digest.hex()==r['attempted_stream_sha256']==ck['attempted_stream_sha256'])
        check(arm+'_finite_changed_model',all(torch.isfinite(v).all() for v in ck['model_state_dict'].values()) and not tensor_equal(ini,ck['model_state_dict']))
        check(arm+'_timing_monotone',all(b['elapsed_seconds']>=a['elapsed_seconds'] for a,b in zip(trace,trace[1:])) and r['training_elapsed_seconds']>=trace[-1]['elapsed_seconds'] and r['total_wall_seconds_including_save']>=r['training_elapsed_seconds'])
        training[arm]=dict(completed_steps=r['completed_steps'],training_elapsed_seconds=r['training_elapsed_seconds'],total_wall_seconds_including_save=r['total_wall_seconds_including_save'],exposure_counts=r['exposure_counts'],initial_loss=trace[0]['loss'],final_loss=trace[-1]['loss'],final_x_mse=trace[-1]['x_mse'],final_n_mse=trace[-1]['n_mse'],final_F_mse=trace[-1]['F_mse'],checkpoint_sha256=sha(d/'final.pt'),stream_sha256=digest.hex())
    check('initial_tensor_exact_equality',tensor_equal(initial[arms[0]],initial[arms[1]]))
    check('all_3000_batch_hashes_paired',[r['batch_sha256'] for r in traces[arms[0]]]==[r['batch_sha256'] for r in traces[arms[1]]])
    # Replay all RNG/schedule transitions and four selected Hungarian batches.
    gen=torch.Generator(device='cpu');gen.manual_seed(260927);rng=np.random.default_rng(260928)
    order=[];cursor=0;exposure=np.zeros(24,dtype=int);selected=[0,999,1999,2999];replay=[]
    for step in range(3000):
        if cursor==len(order):order=rng.permutation(24).tolist();cursor=0
        ids=order[cursor:cursor+8];cursor+=8;exposure[ids]+=1
        noise=torch.randn(8,256,22,generator=gen,dtype=torch.float32);tt=torch.rand(8,1,1,generator=gen,dtype=torch.float32)
        if step in selected:
            sx=noise.numpy()[...,:3]*data['scale'][:3]+data['mean'][:3];targets=np.empty((8,256,22),np.float32)
            for b,j in enumerate(ids):
                cost=((sx[b,:,None,:]-states[j,None,:,:3])**2).sum(axis=-1);ii,jj=linear_sum_assignment(cost)
                targets[b]=((states[j,jj]-data['mean'])/data['scale']).astype(np.float32)
            h=hashlib.sha256(noise.numpy().tobytes()+tt.numpy().tobytes()+targets.tobytes()+np.asarray(ids,np.int64).tobytes()).hexdigest()
            ok=h==traces['direct'][step]['batch_sha256'];replay.append(dict(step=step+1,sha256=h,matched=ok));check('independent_batch_replay_'+str(step+1),ok)
    check('balanced_exposure_replay',all(r['exposure_counts']==exposure.tolist() for r in results.values()))
    saved_noise=arrays(root/'sample_noise.npz')['noise'];expected_noise=np.random.default_rng(260927).standard_normal((32,256,22)).astype(np.float32)
    check('32_noise_arrays_from_frozen_seed',np.array_equal(saved_noise,expected_noise))
    rows=read(root/'LEDGER.json');expected={(arm,i) for arm in arms for i in range(32)}
    check('complete_unique_64_slots',len(rows)==64 and {(r['arm'],r['sample_id']) for r in rows}==expected)
    summary=read(root/'SUMMARY_FM.json');observations=[];by_arm={}
    for arm in arms:
        endpoints=arrays(root/(arm+'_endpoints.npz'))['states'];rr=[r for r in rows if r['arm']==arm]
        check(arm+'_endpoint_slots',endpoints.shape==(32,256,22))
        for r in rr:
            i=r['sample_id'];d=root/'samples'/f'{arm}_{i:03d}';state=endpoints[i];o=dict(arm=arm,sample_id=i,joint_valid=valid(r))
            check(f'{arm}_{i}_bindings',r['paired_noise_sha256']==sha(root/'sample_noise.npz') and r['checkpoint_sha256']==training[arm]['checkpoint_sha256'] and read(d/'result.json')==r)
            check(f'{arm}_{i}_NFE',r['requested_NFE']==64 and 0<r['NFE']<=64 and (r['rollout_failure'] is not None or r['NFE']==64))
            if np.isfinite(state).all():
                cds=[site_cd(state[:,:3],s[:,:3]) for s in states];j=int(np.argmin(cds))
                check(f'{arm}_{i}_closed_target_selection',r['nearest_target_by_site_cd']==str(data['object_names'][j]) and np.allclose(cds,r['all_target_site_cd'],rtol=0,atol=1e-14))
                if (d/'parameters.npz').is_file():
                    pars=arrays(d/'parameters.npz');F=state[:,6:].astype(np.float64).reshape(-1,4,4);T=np.broadcast_to(np.eye(4),(256,4,4)).copy();T[:,:3,3]=-state[:,:3].astype(np.float64)
                    Q=T.transpose(0,2,1)@(F@F.transpose(0,2,1))@T
                    check(f'{arm}_{i}_decode_binding',np.array_equal(pars['v'],state[:,:3]) and np.array_equal(pars['p'],state[:,:3]) and np.array_equal(pars['n'],state[:,3:6]) and np.array_equal(pars['Q'],Q))
            if (d/'mesh.obj').is_file():
                check(f'{arm}_{i}_mesh_hash',r.get('mesh_sha256')==sha(d/'mesh.obj'))
                mesh=arrays(d/'mesh.npz');topo=independent_topology(mesh['vertices'],mesh['faces']);o['independent_topology']=topo
                check(f'{arm}_{i}_topology',topo['faces']==r['faces'] and len(topo['bad_vertex_links'])==r['strict_noncycle_vertex_count'] and topo['boundary_edges']==r['boundary_edges'] and topo['nonmanifold_edges']==r['nonmanifold_edges'] and (topo['betti_if_closed_oriented'] is None or topo['betti_if_closed_oriented']==[r['beta'+str(k)] for k in range(3)]))
            o.update(status=r['status'],faces=r.get('faces'),valid_field=r['valid'],error=r.get('error'),failure_tags=failure_tags(r),geometry_vs_site_selected_target=r.get('geometry_vs_site_selected_target'))
            observations.append(o)
        declared=summary['by_arm'][arm]
        by_arm[arm]=dict(requested=32,finite_endpoints=int(np.isfinite(endpoints).all(axis=(1,2)).sum()),extracted=sum(bool(r['extraction_completed']) for r in rr),status_counts=dict(collections.Counter(r['status'] for r in rr)),joint_valid=sum(valid(r) for r in rr),valid_field_count=sum(r['valid'] for r in rr),failures=[r['sample_id'] for r in rr if not valid(r)],actual_faces=[r.get('faces') for r in rr],invalid_reasons=dict(collections.Counter(r.get('error','geometry audit invalid') for r in rr if not valid(r))),sampling=declared['timing'],decode_extract_audit_evaluate_seconds=sum(r['decode_extract_audit_evaluate_seconds'] for r in rr),nearest_target_counts=dict(collections.Counter(r.get('nearest_target_by_site_cd','unavailable') for r in rr)),valid_target_rms=stats([r['geometry_vs_site_selected_target']['rms_pct_bboxdiag'] for r in rr if valid(r) and 'geometry_vs_site_selected_target' in r]))
        check(arm+'_summary_counts',by_arm[arm]['joint_valid']==declared['valid_samples'] and by_arm[arm]['extracted']==declared['extracted'])
        by_arm[arm]['failure_tag_counts']=dict(collections.Counter(tag for r in rr for tag in failure_tags(r)))
        by_arm[arm]['vertices_stats']=stats([r['vertices'] for r in rr if 'vertices' in r])
        by_arm[arm]['faces_stats']=stats([r['faces'] for r in rr if 'faces' in r])
        by_arm[arm]['all_extracted_target_rms']=stats([r['geometry_vs_site_selected_target']['rms_pct_bboxdiag'] for r in rr if 'geometry_vs_site_selected_target' in r])
    chart=read(root/'DATA_CHECK.json');precision=read(root/'FLOAT32_ORACLE.json')
    check('float64_target_oracle_all24',chart['passed'] and len(chart['records'])==24 and all(r['valid'] and r['same_surface_input_faces'] for r in chart['records']))
    check('float32_target_oracle_all24',len(precision['rows'])==24 and all(r['valid'] and r['same_input_id_surface'] for r in precision['rows']))
    oracles=dict(float64_valid=sum(r['valid'] for r in chart['records']),float64_same_input_faces=sum(r['same_surface_input_faces'] for r in chart['records']),float64_Q_max_relative=max(r['Q_world_relative'] for r in chart['records']),float32_valid=sum(r['valid'] for r in precision['rows']),float32_same_input_faces=sum(r['same_input_id_surface'] for r in precision['rows']),float32_state_max_absolute_change=max(r['state_max_absolute_change'] for r in precision['rows']))
    result=dict(passed=all(c['passed'] for c in checks),check_count=len(checks),failed_checks=[c for c in checks if not c['passed']],checks=checks,source_hashes=source_hashes,training=training,batch_replay=replay,by_arm=by_arm,target_oracles=oracles,observations=observations,execution=execution,audit_seconds=time.perf_counter()-start,scope='Artifact/hash binding, independent selected batch replay, full pairing and noise replay, independent endpoint decode/site selection/topology. Exact intersection findings are existing audited artifacts, not rerun here. No network training or sampling rerun.',interpretation=['Sampling source is independent full-chart Gaussian; only FM time and current state enter velocity. Training-wide chart mean/scale are retained. Target lookups occur after endpoint generation in the reviewed source.','Input-ID surface equality is target representation roundtrip only; it is not matching learned output IDs to targets.','Closed nearest of 24 correlated target states from three parents is not independent generalization.','Transport arm has additional backward dependencies; 3000 matched updates do not imply matched wall cost.','No claim of stable superiority from one paired training seed and 32 sample noises.'])
    out.mkdir(parents=True);write(out/'FM_OUTPUT_AUDIT.json',result);shutil.copy2(Path(__file__),out/'audit_fm_outputs.py')
    supp=dict(original_manifest_sha256=sha(root/'MANIFEST.json'),analysis_source=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__))),files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()])
    write(out/'SUPPLEMENTAL_MANIFEST.json',supp)
    print(json.dumps({k:result[k] for k in ['passed','check_count','failed_checks','training','by_arm','target_oracles','audit_seconds']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);main(p.parse_args().root)
