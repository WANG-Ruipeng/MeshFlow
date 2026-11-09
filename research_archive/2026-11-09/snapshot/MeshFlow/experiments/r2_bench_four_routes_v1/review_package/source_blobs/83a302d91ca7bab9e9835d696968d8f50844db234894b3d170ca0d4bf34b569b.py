"""Read-only aggregation of original metric definitions and selection ledgers."""
from . import common as C
from .run import LABEL
import argparse,csv
import numpy as np

def csvread(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def run():
    done=C.OUT/'results/complete.json'
    if done.exists():r=C.read(done);C.verify(r['files']);return r
    reg=C.read(C.OUT/'registration.json');C.verify(C.read(C.OUT/'completions/geometry_scores.json')['files'])
    from r2.pilots.bench_adaptation_strength_v1.collect_v2 import decorate,aggregate,numeric
    from r2.pilots.bench_seat_coverage_v1.report_tables import CORE
    from r2.pilots.bench_group_weight_match_v1 import evaluation as E
    metrics=C.read(E.MANIFEST)['metric_fields'];rows=[];inputs={};choices=[];facerows=[];boundaries=[];edgefates=[];pairfates=[];ledgerrows=[];areas=[];runtime_rows=[]
    recovery_path=C.OUT/'dispatch/preserved_worker_recovery_v1/complete.json'
    if recovery_path.with_name('registration.json').exists() and not recovery_path.exists():raise RuntimeError('Wait for scheduling recovery receipt')
    recovery=C.read(recovery_path) if recovery_path.exists() else None
    slow_request=C.read(recovery_path.with_name('registration.json'))['slow_request'] if recovery else None
    for task in reg['tasks']:
        q=task['request'];rid=q['request_id'];dest=C.OUT/'outputs'/rid
        meta={k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','report_group','eval_group')}
        selection=C.read(dest/'selection_summary.json')
        request_receipt=C.read(dest/'complete.json')
        cascade_receipt=C.read(C.OUT/'cpu'/LABEL/rid/'complete.json')
        base_face_count=len(C.arrays(dest/'post.npz')['faces'])
        for phase in cascade_receipt['stages']:
            sm=phase['summary']
            paused=bool(recovery and rid==slow_request)
            pause=recovery['resumes']['slow'] if paused else {}
            lower=float(pause.get('pause_lower_bound_seconds',0));upper=float(pause.get('pause_upper_bound_seconds',0))
            phase_paused=paused and phase['stage']==recovery['slow_phase_at_stop']
            runtime_rows.append(dict(meta,stage=phase['stage'],stage_seconds=phase['seconds'],
                requested_N=q['N'],known_K=q['K'],requested_free_faces=int(q['N'])-int(q['K']),
                selected_RAW_faces=selection['selected_RAW_faces'],BASE_POST_total_faces=base_face_count,
                stage_output_total_faces=len(C.arrays(C.OUT/'cpu'/LABEL/rid/phase['stage']/'post.npz')['faces']),
                request_total_seconds=request_receipt['seconds'],cascade_total_seconds=cascade_receipt['seconds'],
                request_explicit_pause_lower_seconds=lower,request_explicit_pause_upper_seconds=upper,
                phase_explicit_pause_lower_seconds=lower if phase_paused else 0,
                phase_explicit_pause_upper_seconds=upper if phase_paused else 0,
                stage_wall_excluding_explicit_pause_lower=max(0,phase['seconds']-(upper if phase_paused else 0)),
                stage_wall_excluding_explicit_pause_upper=max(0,phase['seconds']-(lower if phase_paused else 0)),
                candidates=sm.get('candidates',sm.get('registered_structural_candidates')),
                accepted=sm.get('accepted'),candidate_full_geometry_scans=sm.get('candidate_full_geometry_scans',0),
                new_face_pair_scans=sm.get('new_face_pair_scans',0),location_attempts=sm.get('location_attempts'),
                worker_peak_RSS_KiB=request_receipt['peak_RSS_KiB'],
                timing_scope='Original receipt wall includes explicit suspension if any; subtraction bounds are wall excluding that suspension, not CPU time. RSS is process-lifetime peak.'))
        boundary_rows=C.read(dest/'mixed_boundary.json')
        free_count=selection['N']-selection['K']
        choices.append(dict(meta,**selection,
            seed_free_face_fraction=selection['seed_faces']/free_count,
            selected_RAW_free_face_fraction=selection['selected_RAW_faces']/free_count,
            all_free_selected_RAW=selection['selected_RAW_faces']==free_count,
            selected_RAW_source_area_fraction=selection['selected_raw_area']/selection['free_raw_area'] if selection['free_raw_area'] else None,
            RAW_D0_boundary_clusters=sum(bool(r['D0_flat_corners']) for r in boundary_rows),
            RAW_C_boundary_clusters=sum(bool(r['C_flat_corners']) for r in boundary_rows),
            RAW_D0_noncoincident_clusters=sum(bool(r['D0_flat_corners']) and not r['all_members_same_position'] for r in boundary_rows)))
        facerows.extend(dict(meta,**r) for r in csvread(dest/'per_face_selection.csv'))
        boundaries.extend(dict(meta,**r,
            boundary_kind='RAW_D0_AND_C' if r['D0_flat_corners'] and r['C_flat_corners'] else 'RAW_D0' if r['D0_flat_corners'] else 'RAW_C',
            split_interpretation='Coordinate disagreement within a prior RAW-GEO cluster is descriptive, not proof of a newly torn legal edge') for r in boundary_rows)
        raw=C.arrays(task['paths']['raw'])['raw'];k=q['K'];N=q['N'];ra=.5*np.linalg.norm(np.cross(raw[:,1].astype(float)-raw[:,0],raw[:,2].astype(float)-raw[:,0]),axis=1)
        for label,stages in (('M',('RAW','BASE_POST','CLEAN_FINAL_POST')),(LABEL,('BASE_POST','CLEAN_FINAL_POST'))):
            for stage in stages:
                p=C.resolve(task['paths'][stage+'_row']) if label=='M' else C.OUT/'cpu'/LABEL/rid/stage/'scores/row.json'
                scorep=p.with_name('score.json');src=C.read(p);score=C.read(scorep);inputs[str(p)]=C.sha(p);inputs[str(scorep)]=C.sha(scorep)
                row=decorate(src,label,q,stage);row.update(meta,seam_applicability=score['seam_to_bulk']['status'],result_origin='REUSED_M_READ_ONLY' if label=='M' else 'NEW_CPU_HYBRID')
                if any(row[m]!=src[m] for m in metrics):raise AssertionError('Original metric altered')
                rows.append(row)
                if stage!='RAW':
                    path=task['paths']['original_base' if stage=='BASE_POST' else 'original_final'] if label=='M' else (
                        dest/'post.npz' if stage=='BASE_POST' else C.OUT/'cpu'/LABEL/rid/'CLEAN_FINAL_POST/post.npz')
                    mesh=C.arrays(path);ids=mesh.get('parent_source_raw_slot',mesh.get('source_face_ids'));retained={int(i) for i in ids if k<=i<N};deleted=sorted(set(range(k,N))-retained)
                    tri=mesh['output'].astype(float);actual=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
                    areas.append(dict(meta,label=label,stage=stage,raw_free_area=float(ra[k:].sum()),actual_free_area=float(actual[k:].sum()),
                        original_free_faces=N-k,retained_original_free_faces=len(retained),deleted_original_free_faces=len(deleted),
                        retained_RAW_reference_area=float(ra[sorted(retained)].sum()),deleted_RAW_reference_area=float(ra[deleted].sum()),
                        added_bridge_faces=int((ids<0).sum()),actual_area_over_RAW=float(actual[k:].sum()/ra[k:].sum()) if ra[k:].sum() else None,
                        deleted_RAW_area_fraction=float(ra[deleted].sum()/ra[k:].sum()) if ra[k:].sum() else None))
        for stage in ('BASE_POST','CLEAN_FINAL_POST'):
            ledger=C.read(dest/('ledger_'+stage+'.json'));ledgerrows.append(dict(meta,stage=stage,**ledger['summary']))
            edgefates.extend(dict(meta,stage=stage,**r) for r in ledger['edge_fates'])
            pairfates.extend(dict(meta,stage=stage,**r) for r in ledger['illegal_pair_fates'])
    if len(rows)!=230 or sum(r['label']==LABEL for r in rows)!=92:raise AssertionError('Expected 138 M reference + 92 new')
    condition=aggregate(rows,('label','stage','report_group','parent_id','condition_id'),metrics)
    parent=aggregate(condition,('label','stage','report_group','parent_id'),metrics)
    macro=aggregate(parent,('label','stage','report_group'),metrics)
    by={(r['label'],r['stage'],r['request_id']):r for r in rows};pairs=[]
    for task in reg['tasks']:
        q=task['request']
        for stage in ('BASE_POST','CLEAN_FINAL_POST'):
            a,b=by['M',stage,q['request_id']],by[LABEL,stage,q['request_id']]
            rr={k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','report_group','eval_group')};rr.update(stage=stage,reference='M',candidate=LABEL)
            for key in metrics:
                rr[key+'__reference']=a[key];rr[key+'__candidate']=b[key];rr[key+'__delta']=b[key]-a[key] if numeric(a[key]) and numeric(b[key]) else None
            pairs.append(rr)
    deltakeys=[m+'__delta' for m in metrics]
    pc=aggregate(pairs,('stage','report_group','parent_id','condition_id'),deltakeys);pp=aggregate(pc,('stage','report_group','parent_id'),deltakeys)
    tables=dict(per_output=rows,per_condition=condition,per_parent=parent,parent_macro=macro,paired_per_output=pairs,paired_per_parent=pp,
        selection_per_output=choices,selection_per_face=facerows,mixed_boundary_clusters=boundaries,connectivity_per_output=ledgerrows,
        original_edge_fates=edgefates,original_illegal_pair_fates=pairfates,area_deletion_per_output=areas,phase_runtime=runtime_rows)
    for name,values in tables.items():C.csvwrite(C.OUT/'results'/(name+'.csv'),values)
    C.write(C.OUT/'results/metric_rows.json',rows)
    C.csvwrite(C.OUT/'results/core_parent_macro.csv',[{k:r.get(k) for k in ('label','stage','report_group',*CORE)} for r in macro])
    result=dict(status='COMPLETE',rows=len(rows),new_scores=92,reference_rows=138,metric_fields=metrics,
        interpretation='Parent macro uses noise->condition->parent. I null/support states preserved. Boundary splits and original-slot ledgers descriptive; BRIDGE identity not paired across backends.',
        source_rows=inputs,model_calls=0,scoring_calls=0,
        files={str(p):C.sha(p) for p in (C.OUT/'results').iterdir() if p.is_file()})
    C.write(done,result);return result

if __name__=='__main__':print(run())
