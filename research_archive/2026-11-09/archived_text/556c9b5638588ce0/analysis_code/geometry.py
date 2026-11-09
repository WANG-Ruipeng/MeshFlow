from core import *
from scalars import advantages
import importlib.util,resource,concurrent.futures,multiprocessing

def scorer():
 p=ROOT/'experiments/mf_h1_local0/evaluate.py'
 spec=importlib.util.spec_from_file_location('original_cpu_surface',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert 'torch' not in sys.modules
 return m

def fixtures():
 results=[]
 def done(name,**kw):results.append(dict(family=name,status='PASS',**kw))
 M=np.tile([1.,2.,3.],(3,1));c,r=advantages(M);assert np.array_equal(c,[0,0,0]) and np.array_equal(r,[1.5,0,-1.5]);done('identical_rows',column=c,row=r)
 M=np.ones((3,3))-np.eye(3);c,r=advantages(M);assert np.all(c>0);done('diagonal_zero',column=c)
 M=np.array([[0.,3,7],[2,0,8],[4,6,0]]);rp=[2,0,1];cp=[1,2,0];permuted=M[np.ix_(rp,cp)]
 tagged={(ALPHAS[rp[i]],ALPHAS[cp[j]]):permuted[i,j] for i in range(3) for j in range(3)}
 restored=np.array([[tagged[(a,b)] for b in ALPHAS] for a in ALPHAS]);assert np.array_equal(restored,M);assert not np.array_equal(M,M.T);done('label_permutation',restored=restored)
 c,r=advantages(np.ones((3,3)));assert np.all(c==0) and np.all(r==0);done('all_equal_ties',strict_wins=0)
 actual=readcsv(OUT/'no_response_direction_audit.csv');assert len(actual)==216 and all(x['A_column']==0 and not x['column_strict_win'] for x in actual);done('actual_F1_no_response',groups=len(actual)//3,geometry_calls=0)
 p,g,pa,a,L=2.,4.,2.3,1.1,5.;e=(pa-a*g)/L;base=a*(p-g)/L;resp=(pa-a*p)/L
 assert abs(e-base-resp)<1e-15 and abs(e*e-base*base-resp*resp-2*base*resp)<1e-15;done('decomposition_cancellation',e=e,base=base,response=resp)
 m=scorer();tri=np.array([[[0.,0,0],[1.,0,0],[0,1.,0]]]);q=np.array([[.2,.2,1],[.2,.2,0],[2.,0,0]])
 d=m.surface_distance_squared(q,tri);assert np.allclose(d,[1,0,1],atol=1e-14,rtol=0);done('known_point_triangle_distance',squared_distances=d,geometry_calls=1)
 transformed=tri.copy();transformed[...,0]*=1.1;pulled=transformed.copy();pulled[...,0]/=1.1;assert np.array_equal(pulled,tri);done('affine_pullback_identity',coordinate_residual=float(abs(pulled-tri).max()))
 savej(OUT/'cpu_fixture_results.json',dict(families=results,new_geometry_calls=1,random_scans=0,model_calls=0,torch_imported='torch' in sys.modules))
 return dict(fixture_families=len(results),status='PASS',new_point_triangle_calls=1),['cpu_fixture_results.json']

def evaluate_pullback(job):
 # Each candidate owns durable result and distance buffers. No shared mutable RNG.
 name=job['candidate_id'];dest=OUT/'pullback_cache'/name;dest.mkdir(parents=True,exist_ok=True)
 report=dest/'result.json'
 if report.exists():
  r=js(report)
  for p,h in r['cache_hashes'].items():assert digest(dest/p)==h
  return r
 start=time.monotonic();attempt=dest/'attempt.json'
 if attempt.exists():
  a=js(attempt)
  # A finished distance type is reused. An interrupted uncommitted score never silently replays.
  for typ in ('symmetric','fixed_GT'):
   if a.get(typ)=='RUNNING' and not (dest/(typ+'.npz')).exists():raise RuntimeError('Interrupted distance type requires explicit audit: '+name+'/'+typ)
 else:a=dict(candidate_id=name,status='RUNNING',model_calls=0,new_Gaussian=0)
 savej(attempt,a)
 f,b,raw=raw_for(job['row']);f1,b1,raw1=raw_for(job['base_row']);L=float(b1['L0']);alpha=job['alpha'];factor=float(np.float32(alpha))
 pulled=f.astype(np.float64).copy();pulled[...,0]/=factor
 q=arrays(recs()[job['base_row']['case_id']]['control_points_path'])['points']
 m=scorer()
 cache={}
 sym=dest/'symmetric.npz'
 if not sym.exists():
  a['symmetric']='RUNNING';savej(attempt,a)
  # Independent identical registered RNG stream for the two area measures.
  p1=m.sample_mesh_surface(f1,512,np.random.RandomState(9911));pp=m.sample_mesh_surface(pulled,512,np.random.RandomState(9911))
  d1=m.surface_distance_squared(p1,pulled);d2=m.surface_distance_squared(pp,f1)
  assert np.isfinite(d1).all() and np.isfinite(d2).all()
  tmp=dest/'symmetric.tmp.npz';np.savez_compressed(tmp,F1_samples=p1,pullback_samples=pp,F1_to_pullback_d2=d1,pullback_to_F1_d2=d2);os.replace(tmp,sym)
  a['symmetric']='COMPLETE';savej(attempt,a)
 else:
  with np.load(sym) as z:d1=z['F1_to_pullback_d2'];d2=z['pullback_to_F1_d2']
 gp=dest/'fixed_GT.npz'
 if not gp.exists():
  a['fixed_GT']='RUNNING';savej(attempt,a)
  dg=m.surface_distance_squared(q,pulled);assert np.isfinite(dg).all()
  tmp=dest/'fixed_GT.tmp.npz'
  np.savez_compressed(tmp,GT_Q1=q,GT_to_pullback_d2=dg);os.replace(tmp,gp)
  a['fixed_GT']='COMPLETE';savej(attempt,a)
 else:
  with np.load(gp) as z:dg=z['GT_to_pullback_d2']
 result=dict(candidate_id=name,curve_id=job['curve_id'],arm=job['row']['arm'],phase_step=3000,uid=job['row']['uid'],noise_index=job['row']['noise_index'],split=job['row']['split'],alpha=alpha,inverse_factor=factor,L0=L,
  raw_sha256=job['row']['raw_sha256'],F1_raw_sha256=job['base_row']['raw_sha256'],query_sha256=ah(q),raw_free_sha256=ah(f),pullback_sha256=ah(pulled),
  symmetric_RMS_L0=float(np.sqrt((d1.mean()+d2.mean())/2)/L),F1_to_pullback_RMS_L0=float(np.sqrt(d1.mean())/L),pullback_to_F1_RMS_L0=float(np.sqrt(d2.mean())/L),
  fixed_GT_RMS_L0=float(np.sqrt(dg.mean())/L),fixed_GT_coverage_fraction=float(np.mean(dg<=(.01*L)**2)),fixed_GT_p50_L0=float(np.quantile(np.sqrt(dg),.5)/L),fixed_GT_p90_L0=float(np.quantile(np.sqrt(dg),.9)/L),fixed_GT_p95_L0=float(np.quantile(np.sqrt(dg),.95)/L),
  F1_fixed_GT_RMS_L0=job['F1_E'],F1_fixed_GT_coverage_fraction=job['F1_cov'],fixed_GT_RMS_minus_F1=float(np.sqrt(dg.mean())/L-job['F1_E']),fixed_GT_coverage_minus_F1=float(np.mean(dg<=(.01*L)**2)-job['F1_cov']),
  same_slot_coordinate_RMS_L0_AUX=float(np.sqrt(np.mean((pulled-f1)**2))/L),same_slot_scope='predicted slot only, no GT assignment; not a surface metric',
  sample_points_each_direction=512,fixed_GT_points=len(q),geometry_types=2,point_triangle_calls=3,geometry_scope='OFFLINE_DIAGNOSTIC_NOT_NEW_GENERATION',seconds=time.monotonic()-start,
  peak_process_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),peak_scope='whole CPU worker process lifetime, includes imports and geometry buffers',pid=os.getpid(),torch_imported='torch' in sys.modules,
  cache_hashes={p.name:digest(p) for p in (sym,gp)})
 assert not result['torch_imported']
 savej(report,result);a['status']='COMPLETE';savej(attempt,a)
 return result

def pullback():
 gen=[r for r in generated() if r['table']=='B' and r['phase_step']==3000 and r['status']=='PASS'];cx=readcsv(OUT/'cross_matrix.csv');ix={(r['arm'],r['uid'],r['noise_index']):r for r in cx if r['phase_step']==3000 and r['alpha_in']==1 and r['beta_target']==1}
 jobs=[]
 for key,rs in groups(gen,['arm','uid','noise_index'] ).items():
  aa={r['alpha']:r for r in rs};cid=f'{key[0]}_phase3000_{key[1]}_n{key[2]}';c=ix[key]
  for alpha in (.9,1.1):jobs.append(dict(candidate_id=cid+'_a'+str(int(round(alpha*100))),curve_id=cid,alpha=alpha,row=aa[alpha],base_row=aa[1.],F1_E=c['E_RMS_L0'],F1_cov=c['coverage_fraction']))
 assert len(jobs)<=96
 pending=[j for j in jobs if not (OUT/'pullback_cache'/j['candidate_id']/'result.json').exists()];completed=[]
 resources('before_authorized_first_pullback')
 started=time.monotonic();workers=1;observed=None;dispatch=[]
 if pending:
  first=pending.pop(0);r=evaluate_pullback(first);completed.append(r);observed=r['peak_process_RSS_bytes']
  snap=resources('after_first_actual_candidate');free_cpu=max(1,int(snap['cpu_count']*(1-snap['cpu_percent']/100)))
  # CPU workers cap at four for this bounded audit, further reduced from measured RSS and current CPU/RAM.
  allowed=min(4,max(1,free_cpu//2),max(1,int((snap['ram_available']-2*1024**3)/(1.5*max(observed,128*1024**2)))),len(pending))
  pools=[];running={};idx=0
  try:
   for slot in range(allowed):
    s=resources('before_CPU_worker_'+str(slot+1))
    if s['ram_available']<2*1024**3+1.5*max(observed,128*1024**2) or s['cpu_percent']>90:break
    pool=concurrent.futures.ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'));pools.append(pool)
    job=pending[idx];idx+=1;f=pool.submit(evaluate_pullback,job);running[f]=(pool,job)
    # Incremental startup: observe representative candidate before adding another worker.
    rr=f.result();completed.append(rr);running.pop(f)
    dispatch.append(dict(slot=slot+1,first_candidate=job['candidate_id'],seconds=rr['seconds'],observed_RSS=rr['peak_process_RSS_bytes'],resource=s))
    if idx<len(pending):
     job=pending[idx];idx+=1;running[pool.submit(evaluate_pullback,job)]=(pool,job)
   workers=max(1,len(pools))
   if not pools:
    for j in pending:completed.append(evaluate_pullback(j))
   while running:
    done,_=concurrent.futures.wait(running,timeout=20,return_when=concurrent.futures.FIRST_COMPLETED)
    for f in done:
     pool,job=running.pop(f);rr=f.result();completed.append(rr)
     if len(completed)%16==0:print('pullback complete',len(completed),'of',len(jobs),flush=True)
     if idx<len(pending):
      job=pending[idx];idx+=1;running[pool.submit(evaluate_pullback,job)]=(pool,job)
  finally:
   for p in pools:p.shutdown(wait=True)
 resources('pullback_finished')
 result=[evaluate_pullback(j) for j in jobs] # cache reads only, verified hashes
 metrics=['symmetric_RMS_L0','fixed_GT_RMS_L0','fixed_GT_coverage_fraction','F1_fixed_GT_RMS_L0','fixed_GT_RMS_minus_F1','fixed_GT_coverage_minus_F1','fixed_GT_p95_L0','same_slot_coordinate_RMS_L0_AUX']
 per,summ=parent_summary(result,['arm','split'],metrics)
 outs=[writecsv('pullback_per_case.csv',result),writecsv('pullback_per_parent.csv',per),writecsv('pullback_summary.csv',summ)]
 savej(OUT/'CPU_dispatch.json',dict(actual_parallel_workers=workers,first_actual_candidate_peak_RSS=observed,incremental_dispatch=dispatch,seconds=time.monotonic()-started,candidates=len(jobs),new_candidates_this_entry=len(completed),reused_candidates_this_entry=len(jobs)-len(completed),throughput_candidates_per_second=len(completed)/(time.monotonic()-started),phase_timing_scope='concurrent CPU wall time including imports and IO, not isolated compute benchmark',GPU_workers=0))
 outs.append('CPU_dispatch.json')
 return dict(candidates=len(result),geometry_types_per_candidate=2,new_candidates_this_entry=len(completed),point_triangle_calls_total=sum(r['point_triangle_calls'] for r in result),workers=workers),outs
