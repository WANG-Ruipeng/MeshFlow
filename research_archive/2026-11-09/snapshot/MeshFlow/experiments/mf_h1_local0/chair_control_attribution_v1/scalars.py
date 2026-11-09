from core import *
CK=['arm','phase_step','uid','noise_index']
def ext(a):
 p=np.asarray(a,dtype=np.float64).reshape(-1,3)
 return np.quantile(p,.95,axis=0,method='linear')-np.quantile(p,.05,axis=0,method='linear'),np.median(p,axis=0)

def width():
 original=readcsv(SRC/'B_width_response.csv');gen={(r['arm'],r['phase_step'],r['case_id']):r for r in generated() if r['table']=='B'}
 panels=recs();out=[];curves=[];audit=[];missing=[];analytic_audit=[]
 analytic=readcsv(SRC/'B_analytic_baselines.csv')
 ai={(r['arm'],r['phase_step'],r['uid'],r['noise_index'],r['alpha'],r['candidate']):r for r in analytic}
 for key,rs in groups(original,CK).items():
  base=dict(zip(CK,key));base['split']=rs[0]['split'];base['display_model']=NAMES[base['arm']]
  state={};complete=True
  for r in rs:
   gr=gen[(*key[:2],r['case_id'])];rec=panels[r['case_id']]
   try:
    f,b,raw=raw_for(gr)
    assert digest(resolve(gr['raw_path']))==gr['raw_sha256']
    assert digest(resolve(rec['input_path']))==rec['input_sha256']
    assert np.array_equal(raw['noise'],b['z'])
    assert np.array_equal(raw['raw'][:int(b['K'])].reshape(-1,3,3),b['C'])
    assert np.all(b['valid_mask']) and np.array_equal(raw['known_mask'],b['known_mask'])
    s,m=ext(f);gs,gm=ext(b['gt_free'])
    state[r['alpha']]=dict(r=r,gr=gr,rec=rec,free=f,bank=b,raw=raw,span=s,median=m,gspan=gs,gmedian=gm)
   except FileNotFoundError as e:
    complete=False;missing.append(dict(module='width_raw',key=str(key),status='BLOCKED',reason=str(e)))
  if not complete or set(state)!=set(ALPHAS):continue
  z=state[1.];p=z['span'][0];g=z['gspan'][0];L=float(z['bank']['L0']);N=int(z['bank']['N']);K=int(z['bank']['K'])
  noise=ah(z['raw']['noise']);ids=ah(z['bank']['source_face_ids']);slots=ah(z['bank']['free_source_ids'])
  cid=f'{key[0]}_phase{key[1]}_{key[2]}_n{key[3]}'
  base.update(curve_id=cid,task_id=z['rec']['task_id'],noise_sha256=noise,C_source_ids_sha256=ids,free_slot_ids_sha256=slots,checkpoint_sha256=z['gr']['checkpoint_sha256'],N=N,K=K,gaussian_seed=z['rec']['gaussian_seed'],L0=L)
  goodp=p>1e-8*L;goodg=g>1e-8*L
  lo,hi=state[.9]['span'][0],state[1.1]['span'][0]
  bself=(hi-lo)/(.2*p) if goodp else None;bgt=(hi-lo)/(.2*g) if goodg else None
  sm=(p-lo)/(.1*p) if goodp else None;sp=(hi-p)/(.1*p) if goodp else None
  c=dict(base,p=p,g=g,p_over_g=p/g if goodg else None,base_signed_L0=(p-g)/L,base_abs_L0=abs(p-g)/L,b_self=bself,b_GT=bgt,
   b_self_abs_error=abs(bself-1) if goodp else None,b_GT_abs_error=abs(bgt-1) if goodg else None,slope_narrow=sm,slope_widen=sp,
   one_sided_difference=sp-sm if goodp else None,change_narrow=p-lo,change_widen=hi-p,change_total=hi-lo,
   monotonic=bool(lo<=p<=hi),strict_monotonic=bool(lo<p<hi),reverse_narrow=bool(p<lo),reverse_widen=bool(hi<p),net_reverse=bool(hi<lo),tie_narrow=bool(p==lo),tie_widen=bool(hi==p),
   under_response=bool(bself<1) if goodp else None,over_response=bool(bself>1) if goodp else None,response_status='DEFINED' if goodp else 'UNDEFINED_SMALL_BASE_WIDTH',
   slope_identity_residual=bgt-bself*p/g if goodp and goodg else None)
  curve_rows=[]
  for a,v in sorted(state.items()):
   b=v['bank'];gr=v['gr'];r=v['r'];pa=v['span'][0];ga=v['gspan'][0]
   assert noise==ah(v['raw']['noise']) and ids==ah(b['source_face_ids']) and slots==ah(b['free_source_ids'])
   assert N==int(b['N']) and K==int(b['K'])
   expect=z['bank']['full_target'].copy().reshape(-1,3);expect[:,0]*=np.float32(a)
   assert np.array_equal(expect,b['full_target'].reshape(-1,3))
   assert np.array_equal(b['gt_free'].reshape(-1,3)[:,1:],z['bank']['gt_free'].reshape(-1,3)[:,1:])
   assert close(float(np.linalg.norm(np.ptp(z['bank']['full_target'].reshape(-1,3),axis=0))),L)
   bt=a*(p-g)/L;resp=(pa-a*p)/L;e=(pa-a*g)/L;gtcorr=(a*g-ga)/L
   row=dict(base,case_id=r['case_id'],alpha=a,alpha_float32=float(np.float32(a)),raw_path=gr['raw_path'],raw_sha256=gr['raw_sha256'],input_path=v['rec']['input_path'],raw_dtype=str(v['free'].dtype),statistic_dtype='float64',evidence='RAW_RECOMPUTED',L_alpha=float(b['L']),p=p,g=g,p_alpha=pa,g_alpha=ga,
     p_over_g=c['p_over_g'],e=e,base=bt,response=resp,exact_residual=e-bt-resp,e2=e*e,base2=bt*bt,response2=resp*resp,cross_term=2*bt*resp,square_residual=e*e-bt*bt-resp*resp-2*bt*resp,
     abs_e=abs(e),abs_base=abs(bt),abs_response=abs(resp),cancellation=abs(bt)+abs(resp)-abs(e),opposing_terms=bool(bt*resp<0),
     actual_GT_signed_error_L0=(pa-ga)/L,actual_GT_abs_error_L0=abs(pa-ga)/L,GT_affine_roundoff_L0=gtcorr,GT_affine_identity_within_tolerance=close(ga/L,a*g/L),
     r_alpha=pa/p-a if goodp else None,response_via_relative=p*(pa/p-a)/L if goodp else None,b_self=bself,b_GT=bgt,
     old_X_span_abs_error_L0=r['X_span_abs_error_L0'],old_slope=r['slope'])
   for j,ax in enumerate('XYZ'):
    for field,new in ((ax+'_span',v['span'][j]),(ax+'_GT_span',v['gspan'][j]),(ax+'_median',v['median'][j]),(ax+'_GT_median',v['gmedian'][j])):
     row[field]=new;audit.append(dict(curve_id=cid,alpha=a,field=field,old=r[field],recomputed=new,difference=new-r[field],match=close(new/L,r[field]/L)))
    row[ax+'_span_signed_error_L0']=(v['span'][j]-v['gspan'][j])/L
    row[ax+'_median_signed_error_L0']=(v['median'][j]-v['gmedian'][j])/L
   audit.append(dict(curve_id=cid,alpha=a,field='slope',old=r['slope'],recomputed=bself,difference=bself-r['slope'],match=close(bself,r['slope'])))
   assert abs(row['exact_residual'])<1e-12 and abs(row['square_residual'])<1e-12
   curve_rows.append(row);out.append(row)
   for name in ('No-response','Post-scale'):
    ar=ai[(*key,a,name)];craw=arrays(ar['candidate_raw_path'])['raw'];cf=craw[K:].reshape(-1,3,3)
    x=z['bank']['C'][...,0].astype(np.float64).ravel();xx=b['C'][...,0].astype(np.float64).ravel();hat=float(x@xx/(x@x))
    expected=z['free'].astype(np.float64).copy()
    if name=='Post-scale':expected[...,0]*=ar['alpha_hat']
    assert np.array_equal(cf,expected) and digest(resolve(ar['candidate_raw_path']))==ar['candidate_raw_sha256']
    assert close(hat,ar['alpha_hat']) and ar['source_alpha1_raw_sha256']==z['gr']['raw_sha256']
    analytic_audit.append(dict(base,alpha=a,candidate=name,alpha_hat=ar['alpha_hat'],hat_recomputed=hat,hat_roundoff=hat-ar['alpha_hat'],hat_source='only C1 and Calpha dot fit; no GT free',source_F1_sha256=z['gr']['raw_sha256'],coordinate_exact=True,width=ext(cf)[0][0],absolute_error_L0=ar['X_span_abs_error_L0'],response_residual=ar['response_residual'],slope=ar['slope']))
  for m in ('e','base','response','e2','base2','response2','cross_term','abs_e','abs_base','abs_response','cancellation','actual_GT_abs_error_L0'):
   c['mean_'+m]=mean(r[m] for r in curve_rows)
  c['mean_abs_r']=mean(abs(r['r_alpha']) for r in curve_rows)
  curves.append(c)
 if not all(r['match'] for r in audit):raise RuntimeError('Width reconciliation failed')
 phasepairs=[]
 for key,rs in groups([r for r in curves if r['noise_index']==0],['arm','uid','task_id','noise_sha256']).items():
  ix={r['phase_step']:r for r in rs}
  if set(ix)!={0,3000}:continue
  row=dict(zip(['arm','uid','task_id','noise_sha256'],key),split=rs[0]['split'],noise_index=0)
  for m in ('p_over_g','base_signed_L0','base_abs_L0','b_self','b_GT','b_self_abs_error','b_GT_abs_error','mean_abs_r','mean_e','mean_base','mean_response','mean_e2','mean_base2','mean_response2','mean_cross_term','mean_abs_e','mean_abs_base','mean_abs_response','mean_cancellation','mean_actual_GT_abs_error_L0'):
   row[m+'_phase0']=ix[0][m];row[m+'_phase3000']=ix[3000][m];row[m+'_delta']=ix[3000][m]-ix[0][m]
  phasepairs.append(row)
 noisepairs=[]
 for key,rs in groups([r for r in curves if r['phase_step']==3000],['arm','uid']).items():
  ix={r['noise_index']:r for r in rs}
  if set(ix)!={0,1}:continue
  row=dict(arm=key[0],uid=key[1],split=rs[0]['split'],noise0_sha256=ix[0]['noise_sha256'],noise1_sha256=ix[1]['noise_sha256'],distinct_gaussian_bytes=ix[0]['noise_sha256']!=ix[1]['noise_sha256'])
  for m in ('p_over_g','b_self','b_GT','b_self_abs_error','b_GT_abs_error','slope_narrow','slope_widen','one_sided_difference','strict_monotonic','net_reverse','under_response','over_response'):
   row[m+'_noise0']=ix[0][m];row[m+'_noise1']=ix[1][m]
  row['b_noise1_minus_noise0']=ix[1]['b_self']-ix[0]['b_self'];row['abs_b_noise_difference']=abs(row['b_noise1_minus_noise0'])
  row['crosses_ideal_one']=bool((ix[0]['b_self']-1)*(ix[1]['b_self']-1)<0);row['slope_changes_sign']=bool(ix[0]['b_self']*ix[1]['b_self']<0)
  noisepairs.append(row)
 metrics=['p_over_g','base_signed_L0','base_abs_L0','b_self','b_GT','b_self_abs_error','b_GT_abs_error','slope_narrow','slope_widen','strict_monotonic','monotonic','under_response','over_response','net_reverse','mean_abs_e','mean_abs_base','mean_abs_response','mean_cross_term','mean_cancellation','mean_abs_r']
 parents,summary=parent_summary(curves,['arm','phase_step','split'],metrics)
 for r in summary:r['abs_mean_b_minus_one']=abs(r['b_self']-1)
 seed_summary=aggregate(curves,['arm','phase_step','split','noise_index'],metrics)
 decomp_m=['e','base','response','e2','base2','response2','cross_term','abs_e','abs_base','abs_response','cancellation','actual_GT_abs_error_L0','GT_affine_roundoff_L0']
 dp,ds=parent_summary(out,['arm','phase_step','split'],decomp_m)
 ac=[]
 for key,rs in groups(analytic_audit,CK).items():
  i=groups(rs,['candidate']);row=dict(zip(CK,key),split=rs[0]['split'])
  for (name,),v in i.items():
   row[name+'_mean_absolute_error_L0']=mean(x['absolute_error_L0'] for x in v)
   row[name+'_slope']=mean(x['slope'] for x in v)
  row['post_minus_no_mean_error_L0']=row['Post-scale_mean_absolute_error_L0']-row['No-response_mean_absolute_error_L0'];ac.append(row)
 outs=[]
 for n,r in [('width_per_output.csv',out),('width_per_curve.csv',curves),('width_raw_reconciliation.csv',audit),('width_phase_paired.csv',phasepairs),('noise_paired.csv',noisepairs),('width_per_parent.csv',parents),('width_summary.csv',summary),('width_by_noise.csv',seed_summary),('error_decomposition_summary.csv',dp),('error_decomposition_macro.csv',ds),('analytic_source_audit.csv',analytic_audit),('analytic_curve_comparison.csv',ac),('width_missing.csv',missing)]:outs.append(writecsv(n,r))
 return dict(outputs=len(out),curves=len(curves),raw_numeric_checks=len(audit),max_width_difference=max(abs(r['difference']) for r in audit),max_exact_residual=max(abs(r['exact_residual']) for r in out),max_target_FP32_affine_residual_L0=max(abs(r['GT_affine_roundoff_L0']) for r in out),missing=len(missing)),outs

def advantages(M):
 return np.array([np.mean(np.delete(M[:,j],j))-M[j,j] for j in range(3)]),np.array([np.mean(np.delete(M[i,:],i))-M[i,i] for i in range(3)])

def cross():
 rows=readcsv(SRC/'B_cross_condition_scores.csv');wr=readcsv(SRC/'B_width_response.csv');original={(r['arm'],r['phase_step'],r['uid'],r['noise_index'],r['alpha']):r for r in wr}
 aa=readcsv(SRC/'B_analytic_baselines.csv');ar={(r['arm'],r['phase_step'],r['uid'],r['noise_index'],r['alpha']):r for r in aa if r['candidate']=='No-response'}
 out=[];audit=[];no=[];noaudit=[];reconcile=[]
 for key,rs in groups(rows,CK).items():
  cid=f'{key[0]}_phase{key[1]}_{key[2]}_n{key[3]}'
  saved=js(SRC/'analysis/cross_condition_group'/f'{cid}_cross.json')['rows'];saved={(r['alpha'],r['target_beta']):r for r in saved}
  M=np.full((3,3),np.nan)
  for r in rs:
   a,b=r['alpha'],r['target_beta'];val=r['corresponding_GT_to_free_RMS_L0'];s=saved[(a,b)]
   assert close(val,s['corresponding_GT_to_free_RMS_L0'])
   M[ALPHAS.index(a),ALPHAS.index(b)]=val
   out.append(dict(zip(CK,key),curve_id=cid,split=r['split'],alpha_in=a,beta_target=b,E_RMS_L0=val,coverage_fraction=r['corresponding_GT_coverage_1pct_L0_fraction'],L0=r['L0'],evidence='ORIGINAL_CSV_AND_SCORE_JSON',source_json=f'analysis/cross_condition_group/{cid}_cross.json'))
   reconcile.append(dict(curve_id=cid,alpha_in=a,beta_target=b,CSV_minus_JSON=val-s['corresponding_GT_to_free_RMS_L0'],match=True))
  assert np.isfinite(M).all()
  cols,rr=advantages(M);N=np.tile(M[1,:],(3,1));ncols,nrows=advantages(N)
  assert np.max(abs(ncols))<1e-14
  for j,a in enumerate(ALPHAS):
   old=original[(*key,a)];wrongc=np.delete(M[:,j],j);wrongr=np.delete(M[j,:],j);tol=TOL['atol']+TOL['rtol']*abs(M[j,j])
   assert close(rr[j],old['wrong_mean_minus_diagonal_RMS_L0'])
   assert bool(M[j,j]<wrongr.min())==old['diagonal_strictly_better_than_both_wrong']
   assert bool(M[j,j]<=wrongr.min())==old['diagonal_no_worse_than_both_wrong']
   wa=[x for x in ALPHAS if x!=a]
   audit.append(dict(zip(CK,key),curve_id=cid,split=rs[0]['split'],alpha_beta=a,old_field='wrong_mean_minus_diagonal_RMS_L0',old_formula='mean E[a,b!=a]-E[a,a]',old_actual_axis='ROW_FIXED_OUTPUT',old_value=old['wrong_mean_minus_diagonal_RMS_L0'],A_row=rr[j],A_column=cols[j],
    wrong_input_1=wa[0],wrong_input_2=wa[1],wrong1_minus_diagonal=wrongc[0]-M[j,j],wrong2_minus_diagonal=wrongc[1]-M[j,j],
    old_row_strict_win=old['diagonal_strictly_better_than_both_wrong'],column_strict_win=bool(M[j,j]<wrongc.min()),column_no_worse=bool(M[j,j]<=wrongc.min()),column_strict_win_beyond_tolerance=bool(np.all(wrongc-M[j,j]>tol)),column_ties_within_tolerance=int(np.sum(abs(wrongc-M[j,j])<=tol)),diagonal_rank=1+int(np.sum(wrongc<M[j,j])),
    advantage_sign_changed=bool(np.sign(rr[j])!=np.sign(cols[j])),strict_win_changed=bool((M[j,j]<wrongr.min())!=(M[j,j]<wrongc.min())),audit_result='OLD_VALUES_MATCH_ROW; COLUMN_ADDED',source='width_analytics.py:matching selects alpha, off selects target_beta'))
   nar=ar[(*key,a)];assert close(nar['corresponding_GT_to_free_RMS_L0'],M[1,j])
   noaudit.append(dict(zip(CK,key),curve_id=cid,split=rs[0]['split'],beta_target=a,A_column=ncols[j],A_row=nrows[j],column_strict_win=False,column_ties=2,source_F1_sha256=nar['source_alpha1_raw_sha256'],actual_analytic_score_minus_F1_matrix=nar['corresponding_GT_to_free_RMS_L0']-M[1,j],evidence='original No-response candidate raw verified in width stage + original scores reused'))
   for i,ain in enumerate(ALPHAS):no.append(dict(zip(CK,key),curve_id=cid,alpha_in=ain,beta_target=a,E_RMS_L0=N[i,j],evidence='ANALYTIC_F1_REPEATED_EXISTING_DISTANCES'))
  assert close(cols.mean(),rr.mean())
 par,summ=parent_summary(audit,['arm','phase_step','split','alpha_beta'],['A_column','A_row','column_strict_win','old_row_strict_win','column_ties_within_tolerance','diagonal_rank'])
 allpar,allsumm=parent_summary(audit,['arm','phase_step','split'],['A_column','A_row','column_strict_win','old_row_strict_win'])
 pairs=[]
 for key,rs in groups([r for r in audit if r['phase_step']==3000],['uid','noise_index','alpha_beta']).items():
  ix={r['arm']:r for r in rs}
  for old,new in (('T1','Geo'),('Geo','Full')):
   pairs.append(dict(uid=key[0],noise_index=key[1],alpha_beta=key[2],split=rs[0]['split'],comparison=new+'-'+old,column_advantage_delta=ix[new]['A_column']-ix[old]['A_column'],column_win_delta=int(ix[new]['column_strict_win'])-int(ix[old]['column_strict_win'])))
 outs=[]
 for n,r in [('cross_matrix.csv',out),('cross_direction_audit.csv',audit),('cross_original_reconciliation.csv',reconcile),('no_response_matrix.csv',no),('no_response_direction_audit.csv',noaudit),('cross_per_parent.csv',par),('cross_summary.csv',summ),('cross_all_targets_summary.csv',allsumm),('cross_model_paired.csv',pairs)]:outs.append(writecsv(n,r))
 return dict(original_cells_reused=len(out),new_geometry_cells=0,direction_rows=len(audit),changed_signs=sum(r['advantage_sign_changed'] for r in audit),changed_strict_wins=sum(r['strict_win_changed'] for r in audit),no_response_groups=len(noaudit)//3),outs
