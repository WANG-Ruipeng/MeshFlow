from core import *

QUALITY=['remainder_symmetric_RMS_bbox_pct','R12_to_free_RMS_bbox_pct','free_to_R12_RMS_bbox_pct','Gamma_to_free_edges_RMS_bbox_pct','Gamma_to_free_edges_p95_bbox_pct','boundary_RMS_bbox_pct','boundary_p95_bbox_pct','CF_proper_pair_count','FF_proper_pair_count','FF_illegal_pair_count','CF_proper_affected_free_fraction','FF_proper_affected_free_fraction','FF_illegal_affected_free_fraction','free_minimum_angle_lt5_fraction','free_angle_p05_degrees','free_area_over_GT_free_area','free_q_gt10_fraction']

def coverage():
 records=recs();rr=[r for r in generated() if r['table']=='A'];scores={};vectors={};per=[];missing=[];recon=[];points={}
 for r in rr:
  rec=records[r['case_id']];key=(r['arm'],r['phase_step'],r['case_id'])
  if r['status']!='PASS':missing.append(dict(arm=r['arm'],phase_step=r['phase_step'],case_id=r['case_id'],uid=r['uid'],split=r['split'],ratio=rec['ratio'],status='NA',reason=r.get('reason')));continue
  try:
   sp=resolve(r['score_path']);s=js(sp);assert digest(sp)==r['score_sha256']
   bank=arrays(rec['input_path']);ref=arrays(rec['ref_npz']);refj=js(resolve(rec['ref_json']));L=float(bank['L'])
   q=ref['coverage_points'];d=np.array(s['detail']['fixed_GT_coverage']['per_point_distances'],np.float64)/L
   assert len(q)==len(d)==512 and np.isfinite(d).all()
   assert ah(q)==s['detail']['fixed_GT_coverage']['points_sha256']==refj['array_sha256']['coverage_points']
   assert s['detail']['reference']['GT_free_sha256']==ah(bank['gt_free'].reshape(-1,3,3))
   assert ah(ref['pairwise_target_points'])==s['detail']['remainder']['target_sample_sha256']
   assert ah(q)!=ah(ref['pairwise_target_points'])
   cov=float(np.mean(d<=.01));mse=float(np.mean(d*d));m=s['metrics']
   assert close(cov,m['fixed_R12_coverage_within_1pct_fraction']) and close(np.sqrt(mse),m['fixed_R12_coverage_RMS_bbox_pct']/100)
   row=dict(arm=r['arm'],phase_step=r['phase_step'],case_id=r['case_id'],uid=r['uid'],split=r['split'],ratio=rec['ratio'],noise_index=r['noise_index'],task_id=rec['task_id'],L=L,N=int(bank['N']),K=int(bank['K']),Q=len(d),query_sha256=ah(q),GT_sha256=ah(bank['gt_free']),noise_sha256=ah(bank['z']),input_sha256=rec['input_sha256'],source_score_sha256=r['score_sha256'],fixed_query_MSE=mse,fixed_query_RMS_L=np.sqrt(mse),coverage_fraction=cov,d_p50_L=np.quantile(d,.5),d_p90_L=np.quantile(d,.9),d_p95_L=np.quantile(d,.95),d_max_L=d.max(),evidence='REUSED_ORIGINAL_DISTANCE_VECTOR',pairwise_and_coverage_queries_identical=False,**{k:m.get(k) for k in QUALITY})
   per.append(row);scores[key]=row;vectors[key]=d
   pkey=rec['task_id'];points[pkey]=q
   vpath=OUT/'coverage_vectors'/f"{r['arm']}_phase{r['phase_step']}_{r['case_id']}.npz";vpath.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(vpath,points=q,distances_L=d,L=np.array(L),query_sha256=np.array(ah(q)))
   recon.append(dict(arm=r['arm'],phase_step=r['phase_step'],case_id=r['case_id'],coverage_diff=cov-m['fixed_R12_coverage_within_1pct_fraction'],fixed_RMS_L_diff=np.sqrt(mse)-m['fixed_R12_coverage_RMS_bbox_pct']/100,status='MATCH',coverage_query_sha256=ah(q),pairwise_query_sha256=ah(ref['pairwise_target_points'])))
  except FileNotFoundError as e:missing.append(dict(arm=r['arm'],phase_step=r['phase_step'],case_id=r['case_id'],status='BLOCKED',reason=str(e)))
 assert len(per)<=270
 pairs=[];parts=[];pairplan=[]
 for r in per:
  if r['phase_step']==3000:
   if r['arm']=='Geo':pairplan.append((('T1',3000,r['case_id']),('Geo',3000,r['case_id']),'Geo-Native'))
   if r['arm']=='Full':pairplan.append((('Geo',3000,r['case_id']),('Full',3000,r['case_id']),'Joint-Geo'))
   if r['noise_index']==0 and (r['arm'],0,r['case_id']) in scores:pairplan.append(((r['arm'],0,r['case_id']),(r['arm'],3000,r['case_id']),r['arm']+'_phase3000-phase0'))
 for ka,kb,name in pairplan:
  if ka not in scores or kb not in scores:continue
  a,b=scores[ka],scores[kb];x,y=vectors[ka],vectors[kb]
  for f in ('query_sha256','GT_sha256','noise_sha256','input_sha256','Q','L','N','K'):assert a[f]==b[f],(f,ka,kb)
  withinx=x<=.01;withiny=y<=.01;delta=y*y-x*x;Q=len(x)
  row=dict(comparison=name,old_arm=ka[0],new_arm=kb[0],old_phase=ka[1],new_phase=kb[1],case_id=b['case_id'],uid=b['uid'],split=b['split'],ratio=b['ratio'],noise_index=b['noise_index'],query_sha256=b['query_sha256'],noise_sha256=b['noise_sha256'],Q=Q)
  for m in ['coverage_fraction','fixed_query_MSE','fixed_query_RMS_L','d_p50_L','d_p90_L','d_p95_L','d_max_L']+QUALITY:
   row[m+'_old']=a[m];row[m+'_new']=b[m];row[m+'_delta']=b[m]-a[m] if a[m] is not None and b[m] is not None else None
  row['coverage_up_fixed_MSE_worse']=bool(row['coverage_fraction_delta']>0 and row['fixed_query_MSE_delta']>0)
  row['coverage_up_R12_RMS_worse']=bool(row['coverage_fraction_delta']>0 and row['R12_to_free_RMS_bbox_pct_delta']>0)
  maskgroups={'both_in':withinx&withiny,'enter':~withinx&withiny,'leave':withinx&~withiny,'both_out':~withinx&~withiny}
  total=0
  labels=np.zeros(Q,np.int8)
  for i,(group,mask) in enumerate(maskgroups.items()):
   contribution=float(delta[mask].sum()/Q);total+=contribution;labels[mask]=i
   part=dict(comparison=name,case_id=b['case_id'],uid=b['uid'],split=b['split'],ratio=b['ratio'],noise_index=b['noise_index'],group=group,count=int(mask.sum()),fraction=float(mask.mean()),Q=Q,old_mean_distance_L=mean(x[mask]),new_mean_distance_L=mean(y[mask]),fixed_query_MSE_contribution=contribution,total_MSE_delta=float(delta.mean()),evidence='ORIGINAL_FIXED_QUERY_VECTORS',query_sha256=b['query_sha256'])
   parts.append(part);row[group+'_fraction']=part['fraction'];row[group+'_MSE_contribution']=contribution
  row['partition_sum_residual']=total-float(delta.mean());assert abs(row['partition_sum_residual'])<1e-14
  row['entered_minus_left_residual']=row['enter_fraction']-row['leave_fraction']-row['coverage_fraction_delta'];assert abs(row['entered_minus_left_residual'])<1e-14
  pairs.append(row)
  target=OUT/'coverage_pair_vectors'/f'{name}_{b["case_id"]}.npz';target.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(target,points=points[b['task_id']],old_distance_L=x,new_distance_L=y,partition=labels,squared_delta=delta)
 metrics=[k for k in pairs[0] if k.endswith('_delta') or k.endswith('_contribution') or k.endswith('_fraction')]+['coverage_up_fixed_MSE_worse','coverage_up_R12_RMS_worse']
 pp,summary=parent_summary(pairs,['comparison','split','ratio'],metrics)
 # Parent descriptive direction counts use average within each parent's noise first.
 for s in summary:
  ps=[r for r in pp if all(r[k]==s[k] for k in ('comparison','split','ratio'))]
  for m in ('coverage_fraction_delta','fixed_query_MSE_delta','R12_to_free_RMS_bbox_pct_delta','Gamma_to_free_edges_RMS_bbox_pct_delta','FF_illegal_affected_free_fraction_delta'):
   s[m+'_positive_parents']=sum(r[m]>0 for r in ps);s[m+'_negative_parents']=sum(r[m]<0 for r in ps)
  s['coverage_up_MSE_worse_parents']=sum(r['coverage_fraction_delta']>0 and r['fixed_query_MSE_delta']>0 for r in ps)
  s['coverage_up_R12_worse_parents']=sum(r['coverage_fraction_delta']>0 and r['R12_to_free_RMS_bbox_pct_delta']>0 for r in ps)
 ppar,partsum=parent_summary(parts,['comparison','split','ratio','group'],['count','fraction','old_mean_distance_L','new_mean_distance_L','fixed_query_MSE_contribution','total_MSE_delta'])
 # Intersect parent sets only for availability table, never interpret across ratios causally.
 availability=[]
 for split in ('train','validation'):
  a20={r['uid'] for r in per if r['phase_step']==3000 and r['split']==split and r['ratio']==.2}
  a40={r['uid'] for r in per if r['phase_step']==3000 and r['split']==split and r['ratio']==.4}
  for u in sorted(a20|a40):availability.append(dict(uid=u,split=split,A20=u in a20,A40=u in a40,common=u in a20&a40,causal_information_effect='NOT_IDENTIFIED; C/free complement differs'))
 outs=[]
 for n,r in [('A_distance_per_output.csv',per),('A_distance_reconciliation.csv',recon),('coverage_partition.csv',parts),('coverage_partition_per_parent.csv',ppar),('coverage_partition_summary.csv',partsum),('coverage_paired.csv',pairs),('coverage_per_parent.csv',pp),('coverage_summary.csv',summary),('A_availability.csv',availability),('A_missing_NA.csv',missing)]:outs.append(writecsv(n,r))
 return dict(original_A_vectors_reused=len(per),new_A_distance_evaluations=0,new_pair_intersection_scans=0,paired_cases=len(pairs),partition_rows=len(parts),NA=sum(r['status']=='NA' for r in missing),blocked=sum(r['status']=='BLOCKED' for r in missing),max_partition_residual=max(abs(r['partition_sum_residual']) for r in pairs)),outs

def teacher():
 rows=readcsv(SRC/'teacher_per_case.csv');assert len(rows)<=288
 metrics=['FM_error','D_edge','L5','Yhat_free_area']
 summaries=aggregate(rows,['arm','phase_step','split','t'],metrics)
 original=readcsv(SRC/'teacher_summary.csv');check=[]
 oi={tuple(r[k] for k in ('arm','phase_step','split','t')):r for r in original}
 for s in summaries:
  o=oi[tuple(s[k] for k in ('arm','phase_step','split','t'))]
  for m in metrics:
   check.append(dict(arm=s['arm'],phase_step=s['phase_step'],split=s['split'],t=s['t'],metric=m,recomputed=s[m],original=o[m],difference=s[m]-o[m],match=close(s[m],o[m])))
 assert all(r['match'] for r in check)
 pairs=[];trajectories=[]
 for key,rs in groups(rows,['case_id','input_sha256'] ).items():
  ix={(r['arm'],r['phase_step']):r for r in rs}
  assert len(rs)==12
  for phase in (0,1000,2500,3000):
   for a,b in (('T1','Geo'),('Geo','Full')):
    aa,bb=ix[(a,phase)],ix[(b,phase)]
    pairs.append(dict(comparison=b+'-'+a,phase_step=phase,case_id=key[0],input_sha256=key[1],uid=bb['uid'],split=bb['split'],t=bb['t'],**{m+'_delta':bb[m]-aa[m] for m in metrics}))
  for arm in ARMS:
   a,b=ix[(arm,0)],ix[(arm,3000)]
   tr=dict(arm=arm,case_id=key[0],input_sha256=key[1],uid=b['uid'],split=b['split'],t=b['t'])
   for m in metrics:
    vals=[ix[(arm,p)][m] for p in (0,1000,2500,3000)]
    for p,v in zip((0,1000,2500,3000),vals):tr[m+'_phase'+str(p)]=v
    tr[m+'_delta']=b[m]-a[m];tr[m+'_relative_change_pct']=100*(b[m]/a[m]-1) if a[m] else None
    tr[m+'_monotonic']=bool(np.all(np.diff(vals)>=0) or np.all(np.diff(vals)<=0))
   trajectories.append(tr)
 # Generation curve reproduction from real A20 noise0 records, preserving NA train middle phases.
 A=readcsv(OUT/'A_distance_per_output.csv');gr=[r for r in A if r['ratio']==.2 and r['noise_index']==0]
 gm=['remainder_symmetric_RMS_bbox_pct','R12_to_free_RMS_bbox_pct','Gamma_to_free_edges_RMS_bbox_pct','FF_illegal_affected_free_fraction','coverage_fraction']
 gps,gs=parent_summary(gr,['arm','phase_step','split'],gm)
 gcheck=[];og=readcsv(SRC/'shared_input_generation_curves.csv')
 for r in og:
  s=next(x for x in gs if all(x[k]==r[k] for k in ('arm','phase_step','split')))
  for m in gm:
   om='fixed_R12_coverage_within_1pct_fraction' if m=='coverage_fraction' else m
   gcheck.append(dict(arm=r['arm'],phase_step=r['phase_step'],split=r['split'],metric=m,old=r[om],recomputed=s[m],match=close(r[om],s[m])))
 assert all(r['match'] for r in gcheck)
 for arm in ARMS:
  for phase in (1000,2500):gs.append(dict(arm=arm,phase_step=phase,split='train',n_parents=0,status='NA',reason='No original intermediate train generation'))
 outs=[]
 for n,r in [('teacher_readonly_per_case.csv',rows),('teacher_recomputed_summary.csv',summaries),('teacher_reconciliation.csv',check),('teacher_input_paired.csv',pairs),('teacher_per_input_curves.csv',trajectories),('generation_readonly_curves.csv',gs),('generation_curve_reconciliation.csv',gcheck)]:outs.append(writecsv(n,r))
 return dict(original_teacher_rows=len(rows),input_trajectories=len(trajectories),teacher_summary_checks=len(check),generation_curve_checks=len(gcheck),new_model_calls=0),outs
