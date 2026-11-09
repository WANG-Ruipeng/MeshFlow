"""Explicit metric domains, scales, applicability and denominators."""
VERSION='native_mesh_direct_context_readout_metrics_v1'
UNIMPLEMENTED=('whole_mesh_Ri','watertight_certification','nonmanifold_vertices','human_usability','FID','KID')
REFERENCE=('SeamCov','ExposedRetain','GT_','F_to_GT','D_sym','Precision','Recall','Fscore','fixed_query','coverage_fraction','remainder_','free_to_R12','R12_to_free','boundary_','Gamma','free_area_over_GT')
def describe(name,panel,N,K,L):
    ref=any(name.startswith(p) for p in REFERENCE) or 'minus_GT' in name or name.startswith('ConflictCov_C_GT') or name.startswith('reference_')
    domain='free' if name.startswith(('free_','FF')) else 'C_vs_free' if name.startswith(('CF','Conflict')) else 'full_raw'
    if name.startswith(('Seam','Gamma','boundary')):domain='source_Gamma_to_free'
    if name.startswith('Exposed'):domain='reference_visible_C'
    if name.startswith(('GT_','F_to_GT','D_sym','Precision','Recall','Fscore','fixed_query','coverage','remainder','R12','free_to_R12')):domain='free_vs_reference_free'
    denom=N-K if domain=='free' else 2048 if name.startswith(('Conflict','Exposed')) else 512 if name.startswith(('fixed_query','coverage')) else None
    thresholds={'S_taus':[.005,.01,.02],'S_angle_max_degrees':20,'I_cross_band':.005,'I_duplicate_band':.005,'I_C_border':.01,'I_normal_max_degrees':15,
                'coverage512':dict(tau=.01,comparison='<='),'PR2048':dict(taus=[.005,.01,.02,.05],comparison='<'),
                'pair_distance_epsilon':1e-9,'pair_primary_area_epsilon':1e-12,'pair_independent_area_floor':1e-11,'pair_coplanar_area_epsilon':1e-12,
                'free_shape_near_area':1e-12,'small_angle_degrees':5,'q_actual_threshold':10,'finite_Gamma_coverage_tau':.001}
    return dict(metric_id=name,version=VERSION,output_stage='RAW',domain=domain,
                reference_type='SOURCE_COMPLETION' if ref else 'NONE',requires_GT=ref,
                scale=L,scale_source='source_parent_L0' if panel=='P1' else 'fixed_canonical_training_cube',
                thresholds=thresholds,denominator=denom)
def records(row):
    out=[];m=row['metrics'];N,K,L=row['N'],row['K'],row['L0']
    for name,value in m.items():
        if not isinstance(value,(int,float,bool)) and value is not None:continue
        spec=describe(name,row['panel'],N,K,L)
        if spec['requires_GT'] and row['panel']=='P2':status='NA_NOT_APPLICABLE'
        else:
            family=next((p for p in ('SeamCov','ConflictCov_C','ExposedRetain') if name.startswith(p)),name)
            status=row.get('metric_status',{}).get(name,row.get('metric_status',{}).get(family,'PASS' if value is not None else 'NOT_RUN'))
        lower=value;upper=value
        if name=='SeamCov':upper=m.get('SeamCov_upper')
        elif name=='ConflictCov_C':upper=m.get('ConflictCov_C_upper')
        elif name=='ExposedRetain':upper=m.get('ExposedRetain_upper')
        if name.startswith('SeamCov'):spec['denominator']=m.get('SeamCov_gamma_total_length')
        elif name.startswith('ExposedRetain'):spec['denominator']=m.get('reference_exposed_points')
        elif '_pair_count' in name or 'pairs' in name:spec['denominator']=K*(N-K) if name.startswith(('CF','Conflict')) else (N-K)*(N-K-1)//2
        elif spec['denominator'] is None:spec['denominator']='defined statistic domain; see metric_id and sampling'
        spec['unit']='percent_L' if name.endswith('_bbox_pct') else 'L_squared_normalized' if 'MSE' in name or 'D_sym' in name else 'fraction_or_count_as_named'
        out.append(dict(request_id=row['request_id'],label=row['label'],panel=row['panel'],cohort=row['cohort'],
            condition_id=row['condition_id'],parent_id=row['parent_id'],candidate=row['candidate'],
            **spec,value=value,lower=lower,upper=upper,status=status,reason=row.get('reason') if status!='PASS' else None))
    for name in UNIMPLEMENTED:
        out.append(dict(request_id=row['request_id'],label=row['label'],panel=row['panel'],cohort=row['cohort'],
            condition_id=row['condition_id'],parent_id=row['parent_id'],candidate=row['candidate'],
            **describe(name,row['panel'],N,K,L),value=None,lower=None,upper=None,status='NOT_RUN' if name=='human_usability' else 'NOT_IMPLEMENTED',
            reason='Not implemented/authorized as a performance score in this fixed pilot'))
    return out
