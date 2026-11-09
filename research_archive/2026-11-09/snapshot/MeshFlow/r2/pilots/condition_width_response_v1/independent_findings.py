"""Independent CPU aggregation of frozen response/review tables and execution metadata."""
import collections,csv,json,math,statistics
from pathlib import Path
from . import common as C

SOURCES={}
def read_json(path):
    p=C.resolve(path);SOURCES[str(p)]=C.sha(p);return C.read(p)
def rows(path):
    p=C.resolve(path);SOURCES[str(p)]=C.sha(p)
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def summary(values):
    v=[float(x) for x in values if x not in ('',None,'NA')]
    return dict(n=len(v),min=min(v),mean=statistics.mean(v),median=statistics.median(v),max=max(v)) if v else dict(n=0)
def select(rs,group):
    if group=='ALL':return rs
    if group=='SEAT_HALF':return [r for r in rs if r['region_type']=='SEAT' and r['tier']=='HALF']
    if group=='OTHER':return [r for r in rs if not(r['region_type']=='SEAT' and r['tier']=='HALF')]
    return [r for r in rs if r['tier']==('HALF' if group=='HALF_ALL' else 'FULL')]
def parent_mean(rs,key):
    d=collections.defaultdict(list)
    for r in rs:
        if r.get(key) not in ('',None,'NA'):d[r['parent_id']].append(float(r[key]))
    return summary([statistics.mean(v) for v in d.values()])
def axis_stats(rs,prefix):
    values=[[float(r[prefix+a]) for a in 'xyz'] for r in rs]
    return dict(component_RMS={a:summary(v[j] for v in values) for j,a in enumerate('xyz')},
        dominant_component_counts=dict(collections.Counter('xyz'[max(range(3),key=lambda j:v[j])] for v in values)),
        mean_per_record_squared_component_fraction={a:statistics.mean(v[j]**2/sum(x*x for x in v) for v in values) for j,a in enumerate('xyz')},
        squared_fraction_weighting='Equal record mean of component RMS squared / sum over xyz; not pooled corner counts.')
def main():
    review_receipt=read_json(C.OUT/'evaluation/gallery/output_review_receipt.json')
    assert review_receipt['status']=='COMPLETE' and C.sha(review_receipt['review_csv'])==review_receipt['review_csv_sha256']
    probes=rows(C.OUT/'analysis/per_fixed_state_response.csv');curves=rows(C.OUT/'analysis/per_curve_response.csv')
    trajectory=rows(C.OUT/'analysis/per_trajectory_response.csv');face=rows(C.OUT/'analysis/per_free_face_velocity_response.csv')
    review=rows(review_receipt['review_csv']);quality=rows(C.OUT/'evaluation/quality_per_output.csv')
    topology=rows(C.OUT/'evaluation/topology_per_output.csv')
    reg=read_json(C.OUT/'registration.json');assert len(probes)==64 and len(curves)==32 and len(review)==96
    probe_stats=[]
    for group in ('ALL','SEAT_HALF','OTHER','HALF_ALL','FULL_ALL'):
        for step in (None,5,15,30,45):
            rr=select(probes,group);rr=[r for r in rr if step is None or int(r['step'])==step]
            probe_stats.append(dict(group=group,step=step,time=None if step is None else step/50,
                probe_pairs=len(rr),parents=len({r['parent_id'] for r in rr}),
                relative_velocity_response=summary(r['relative_velocity_response'] for r in rr),
                delta_velocity_RMS=summary(r['delta_velocity_rms'] for r in rr),
                exact_zero_pairs=sum(r['exact_zero_velocity']=='True' for r in rr),
                axis=axis_stats(rr,'RMS_delta_v')))
    curve_stats=[]
    for group in ('ALL','SEAT_HALF','OTHER','HALF_ALL','FULL_ALL'):
        for stage in ('RAW','POST'):
            rr=[r for r in select(curves,group) if r['stage']==stage]
            z=dict(group=group,stage=stage,curves=len(rr),parents=len({r['parent_id'] for r in rr}),
                monotone_three_point_count=sum(r['width_monotone']=='True' for r in rr),
                positive_endpoint_width_change_count=sum(float(r['width_delta'])>0 for r in rr),
                width_slope=summary(r['width_slope'] for r in rr),width_slope_parent_equal=parent_mean(rr,'width_slope'),
                nonmonotone_ids=[r['base_request_id'] for r in rr if r['width_monotone']!='True'])
            if stage=='RAW':
                z.update(exact_zero_final_free_count=sum(r['exact_zero_response']=='True' for r in rr),
                    free_corner_displacement_over_L=summary(r['free_corner_displacement_L'] for r in rr),
                    displacement_over_C=summary(r['displacement_over_C'] for r in rr),axis=axis_stats(rr,'displacement_RMS_'))
            curve_stats.append(z)
    temporal=[]
    for step in range(51):
        rr=[r for r in trajectory if int(r['step'])==step]
        temporal.append(dict(step=step,time=step/50,free_corner_displacement_over_L=summary(r['corner_displacement_L'] for r in rr)))
    trajmono=[]
    for rid in sorted({r['base_request_id'] for r in trajectory}):
        rr=sorted((r for r in trajectory if r['base_request_id']==rid),key=lambda r:int(r['step']))
        v=[float(r['corner_displacement_L']) for r in rr]
        trajmono.append(dict(base_request_id=rid,all_50_displacement_increments_nonnegative=all(a<=b for a,b in zip(v,v[1:]))))
    # Existing per-face mean vx is the velocity change of that face's centroid.
    # Do not rename its RMS as the unsaved per-corner x-component RMS.
    request_parent={q['base_request_id']:q['parent_id'] for q in reg['requests']}
    grouped=collections.defaultdict(list)
    for r in face:grouped[(r['base_request_id'],int(r['step']),r['spatial_x_bin'])].append(r)
    spatial_curve=[]
    for (rid,step,binname),rr in sorted(grouped.items()):
        vx=[float(r['mean_delta_vx']) for r in rr];vrms=[float(r['delta_velocity_rms']) for r in rr]
        pos=sum(v>0 for v in vx);neg=sum(v<0 for v in vx);zero=sum(v==0 for v in vx)
        spatial_curve.append(dict(base_request_id=rid,parent_id=request_parent[rid],step=step,time=step/50,
            spatial_x_bin=binname,free_faces_in_bin=len(rr),delta_centroid_vx_mean=statistics.mean(vx),
            delta_centroid_vx_RMS=math.sqrt(statistics.mean(v*v for v in vx)),
            delta_velocity_all_coordinates_RMS=math.sqrt(statistics.mean(v*v for v in vrms)),
            centroid_vx_positive_fraction=pos/len(vx),centroid_vx_negative_fraction=neg/len(vx),centroid_vx_exact_zero_fraction=zero/len(vx),
            outward_sign_fraction=neg/len(vx) if binname=='LEFT_OF_C' else pos/len(vx) if binname=='RIGHT_OF_C' else None))
    spatial=[]
    metrics=('delta_centroid_vx_mean','delta_centroid_vx_RMS','delta_velocity_all_coordinates_RMS',
             'centroid_vx_positive_fraction','centroid_vx_negative_fraction','centroid_vx_exact_zero_fraction','outward_sign_fraction')
    for step in (5,15,30,45):
        for b in ('LEFT_OF_C','WITHIN_C_X_SPAN','RIGHT_OF_C'):
            rr=[r for r in spatial_curve if r['step']==step and r['spatial_x_bin']==b]
            d=dict(step=step,time=step/50,spatial_x_bin=b,valid_curves=len(rr),NA_empty_curves=16-len(rr),
                valid_parents=len({r['parent_id'] for r in rr}),NA_empty_parents=8-len({r['parent_id'] for r in rr}),
                saved_face_rows=sum(r['free_faces_in_bin'] for r in rr))
            for key in metrics:
                ss=parent_mean(rr,key);d[key+'_parent_equal_mean']=ss.get('mean');d[key+'_valid_parents']=ss['n']
            spatial.append(d)
    visual=[]
    quality_summary=[]
    for group in ('ALL','SEAT_HALF','OTHER'):
        for stage in ('RAW','POST'):
            for variant in ('m095','base','p105'):
                rr=[r for r in select(review,group) if r['stage']==stage and r['variant']==variant]
                visual.append(dict(group=group,stage=stage,variant=variant,outputs=len(rr),
                    judgments={k:{z:sum(r[k]==z for r in rr) for z in ('YES','NO','UNCERTAIN')} for k in ('C_adoption','C_wrong_layer','whole_body_complete','support_missing')}))
                qr=[r for r in select(quality,group) if r['stage']==stage and r['variant']==variant]
                tt=[r for r in select(topology,group) if r['stage']==stage and r['variant']==variant]
                quality_summary.append(dict(group=group,stage=stage,variant=variant,outputs=len(qr),
                    parent_equal_means={k:parent_mean(qr,k)['mean'] for k in ('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','deleted_free_faces')},
                    valid_CF_edges=sum(int(r['C_F_valid_edge_count']) for r in tt),
                    outputs_with_valid_CF_edge=sum(int(r['C_F_valid_edge_count'])>0 for r in tt)))
    invariant_review=[]
    for rid in sorted(request_parent):
        rr=[r for r in review if r['base_request_id']==rid]
        invariant_review.append(dict(base_request_id=rid,rows=len(rr),C_adoption_all_stages_variants_identical=len({r['C_adoption'] for r in rr})==1))
    resource=[]
    for file in sorted((C.OUT/'resources').glob('*.json')):
        r=read_json(file);v=[x.strip() for x in r['gpu']['stdout'].strip().splitlines()[0].split(',')]
        memory={x.split(':')[0]:int(x.split(':')[1].split()[0]) for x in r['memory'].splitlines() if x.startswith(('MemTotal:','MemAvailable:'))}
        resource.append(dict(path=str(file),phase='D0' if file.name.startswith('D0') else 'generator',
            epoch_ns=int(file.stem.rsplit('_',1)[1]),GPU=v[0],total_MiB=float(v[1]),whole_device_used_MiB=float(v[2]),free_MiB=float(v[3]),
            utilization_percent=float(v[4]),CPU_logical=r['cpu_logical'],RAM_available_MiB=memory['MemAvailable']/1024,
            process_VRAM_peak='NA_WDDM_NOT_REPORTED',process_text=r['processes']['stdout']))
    rawindex=read_json(C.OUT/'evaluation/raw_index.json');postindex=read_json(C.OUT/'evaluation/post_index.json')
    time_rows={
        'rollout':[read_json(r['result_path']) for r in rawindex['rows']],
        'probe':[read_json(p) for p in sorted((C.OUT/'probes').glob('*/*/*_return.json'))],
        'D0':[read_json(r['proposal_receipt']) for r in postindex['rows']]}
    timings={}
    for name,rr in time_rows.items():
        alloc='cuda_peak_allocated_bytes' if name=='rollout' else 'peak_allocated_bytes'
        reserve='cuda_peak_reserved_bytes' if name=='rollout' else 'peak_reserved_bytes'
        timings[name]=dict(records=len(rr),saved_seconds=summary(r['seconds'] for r in rr),sum_saved_seconds=sum(r['seconds'] for r in rr),
            max_torch_peak_allocated_bytes=max(r[alloc] for r in rr),max_torch_peak_reserved_bytes=max(r[reserve] for r in rr),
            max_torch_peak_allocated_MiB=max(r[alloc] for r in rr)/1048576,max_torch_peak_reserved_MiB=max(r[reserve] for r in rr)/1048576)
    gen_complete=read_json(C.OUT/'execution/generator_complete.json');d0_complete=read_json(C.OUT/'execution/D0_complete.json')
    resource_by_phase={}
    for phase in ('generator','D0'):
        rr=[r for r in resource if r['phase']==phase]
        resource_by_phase[phase]=dict(snapshots=len(rr),max_sampled_whole_device_used_MiB=max(r['whole_device_used_MiB'] for r in rr),
            min_free_MiB=min(r['free_MiB'] for r in rr),min_RAM_available_MiB=min(r['RAM_available_MiB'] for r in rr),
            resource_snapshot_span_seconds=(max(r['epoch_ns'] for r in rr)-min(r['epoch_ns'] for r in rr))/1e9,
            process_VRAM_peak='NA; WDDM per-process used_memory is N/A',
            whole_device_scope='Sampled whole-device maximum, not continuous peak or exact worker attribution; includes unrelated desktop/context allocations.')
    result=dict(status='COMPLETE',method='Pure CPU aggregation of existing saved tables/metadata; no geometry/model rescoring',
        reviewer_blind_to_metrics_before_labels=review_receipt['metric_rank_not_read_before_labels'],review_type=review_receipt['review_type'],
        probe_stats=probe_stats,curve_stats=curve_stats,trajectory_stats_all51=temporal,
        individual_trajectory_monotonicity=trajmono,
        spatial_bins=spatial,spatial_bin_definition={
            'grouping':'Saved baseline free-face centroid relative to original C min/max X; no semantic part labels.',
            'aggregation':'Within each curve/bin equal-face means or RMS; mean two valid noises within parent; equal mean over valid parents.',
            'delta_centroid_vx_RMS':'RMS of existing face-mean delta vx, not RMS of individual corner delta vx.',
            'outward_sign_fraction':'Left: face-mean delta vx<0. Right: >0. Exact signs descriptive only; middle NA; no success cutoff.',
            'missing_bin':'No faces -> NA and excluded with valid curve/parent counts retained.'},
        visual_observations=visual,visual_stability=invariant_review,existing_quality_contrast=quality_summary,
        interpretation={
            'main':'EXP67 responds to these C width changes numerically and in aggregate free X width, while semantic adoption and exact interfaces remain unresolved/poor on this panel.',
            'excluded_in_this_panel':[
                'The .95/1.05 C perturbation is completely swallowed before the first backbone block: all64 actual embedding pairs differ.',
                'Free predictions completely ignore this width change: all64 fixed-state velocity pairs and all16 terminal RAW free pairs differ.',
                'The generator shows no sign-consistent endpoint width dependence: all16 RAW and all16 POST robust-width endpoint deltas are positive.',
                'POST alone causes all observed adoption failures: the same clear failures already appear in frozen RAW visual observations.'],
            'not_excluded_or_not_proven':[
                'Nonzero response, positive width dependence, or a large paired-slot displacement does not prove appropriate C adoption, nearby interface layout, or attachment.',
                'RAW/POST all96 valid CF edge counts are zero; this diagnoses absent exact legal shared edges, not whether every open boundary was required to attach.',
                'Height/control of other attributes is untested; no statement about arbitrary C or all chairs follows from width-only8parent2noise evidence.',
                'Smaller SEAT_HALF width slopes than OTHER are descriptive; parent identities/K/region distributions differ, so this is not an isolated causal half/full comparison.',
                'Multiaxis endpoint movement can include trajectory sensitivity and reorganization; same-slot correspondence is not semantic surface correspondence.',
                'No evidence selects a new loss/module/gain, establishes a threshold for weak response, or requires width slope=1.',
                'High UNCERTAIN visual counts limit semantic conclusions; they must not be recoded as success or failure.'],
            'timing_pattern':'Mean fixed-state relative response declines from t=.1 to .6 and rises at .9. Independent-rollout displacement accumulates; probe response and trajectory divergence measure different objects.',
            'axis_pattern':'Fixed-state velocity change is often X-dominant, while endpoint paired-slot displacements are distributed over X/Y/Z; no automatic claim of correct height response.'},
        resources=dict(GPU_workers=1,CPU_logical=24,training_updates=0,backwards=0,phase_resources=resource_by_phase,
            per_operation_timings=timings,generator_loop_wall_seconds=gen_complete['seconds'],D0_phase_wall_seconds=d0_complete['seconds'],
            timer_scope={
                'generator_loop':'Starts after generator load/resource snapshot; includes sampling/probes, CPU snapshots, validation and artifact IO; excludes model load.',
                'D0_phase':'Starts before first per-output loop; includes D0 load, CPU GEO/export, validation and IO.',
                'rollout_seconds':'Public sampling timer includes CPU trajectory and passive velocity copies; excludes final raw export. Not GPU-kernel-only time.',
                'probe_seconds':'Includes observation, CPU copies, NPZ serialization/hash; not GPU-kernel-only time.',
                'D0_proposal_seconds':'Includes one prediction plus saved proposal serialization before receipt timestamp; not GPU-kernel-only time.',
                'no_double_count':'Per-operation sums are nested inside their phase walls and must not be added to them. Torch peaks include resident model; physical process peak remains NA.'}),
        source_hashes=SOURCES,script_sha256=C.sha(Path(__file__)),new_model_calls=0,new_geometry_scores=0,review_modified=False)
    C.csvwrite(C.OUT/'audit/independent_probe_spatial_curve_bins.csv',spatial_curve)
    C.csvwrite(C.OUT/'audit/independent_probe_spatial_bins.csv',spatial)
    result['spatial_csv_hashes']={name:C.sha(C.OUT/'audit'/name) for name in ('independent_probe_spatial_curve_bins.csv','independent_probe_spatial_bins.csv')}
    C.write(C.OUT/'audit/independent_findings.json',result)
    print(json.dumps(dict(status='COMPLETE',output=str(C.OUT/'audit/independent_findings.json'),
        sha256=C.sha(C.OUT/'audit/independent_findings.json'),spatial_curve_rows=len(spatial_curve),spatial_bins=len(spatial),
        phase_resources=resource_by_phase),indent=2))
if __name__=='__main__':main()
