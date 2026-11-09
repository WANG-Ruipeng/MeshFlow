"""Read completed evidence; never run a model, detector, selector or renderer.

Actual RAW4 and POST3 observations remain 230 cells. Mapping the already viewed
winner RAW to SELECTED is a declared lookup, not an additional visual rating.
"""
from __future__ import annotations
import argparse, collections, io, json, statistics, zipfile
from .common import *
from r2.pilots.bench_full_seat_curriculum_v1 import summarize_results as S

DEST=OUT/'results/delivery_v1'
GROUPS=('OLD','FOCUS','DEV','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT','NEW_WIDE_TRAIN')
LABELS=('FIXED_N0','FIXED_N1','SELECTED')
CORE=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
      'free_minimum_angle_lt5_fraction','free_near_degenerate_fraction','C_F_has_valid_edge',
      'C_F_valid_edge_total_length','hinge_diag_legal_opposed_edge_C_reached_free_area_fraction',
      'actual_free_faces','free_total_surface_area','deleted_free_faces',
      'hinge_diag_isolated_1face_edge_components','nonmanifold_edges')


def complete_visual_rows():
    protocol=read(OUT/'visual_review/protocol.json');verify(protocol['protected'])
    assignments=read(OUT/'visual_review/assignments.json')
    if assignments['protocol_sha256']!=sha(OUT/'visual_review/protocol.json'):raise ValueError('Changed assignments')
    expected={(u['page_id'],u['label']):u for u in protocol['expected_units']}
    rows=[];sources={}
    for owner,assignment in assignments['assignments'].items():
        path=OUT/'visual_review/shards'/(owner+'.json');payload=read(path)
        if payload['status']!='COMPLETE_ASSIGNED_SHARD' or payload['reviewer']!=owner:raise ValueError('Incomplete assigned shard')
        if payload['protocol_sha256']!=sha(OUT/'visual_review/protocol.json'):raise ValueError('Shard protocol changed')
        if payload['assignment_sha256']!=sha(OUT/'visual_review/assignments.json'):raise ValueError('Shard assignments changed')
        if payload['actual_pages']!=assignment['pages'] or payload['actual_observations']!=assignment['observations']:raise ValueError('Shard coverage')
        sources[str(path)]=sha(path);rows.extend(payload['rows'])
    keys=[(r['page_id'],r['label']) for r in rows]
    if len(rows)!=230 or len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('All230 actual observations required')
    for row in rows:
        want=expected[row['page_id'],row['label']]
        if any(row[k]!=want[k] for k in want):raise ValueError('Visual condition identity changed')
        if row['reviewer_kind']!='AI_PROXY' or row['blind'] is not False or row['actually_viewed'] is not True or row['views_and_sections_checked'] is not True:raise ValueError('Missing actual-view attestation')
        png=OUT/'gallery'/(row['page_id']+'.png');receipt=png.with_suffix('.json')
        if row['image_sha256']!=sha(png) or row['image_receipt_sha256']!=sha(receipt):raise ValueError('Viewed image changed')
        for axis,allowed in protocol['axes'].items():
            if row[axis] not in allowed:raise ValueError('Unrecognized axis')
    return rows,sources


def prerequisites():
    # The report owner completes actual visual records before reading metrics.
    complete_visual_rows()
    result=read(OUT/'results/completion.json');verify(result['files']);verify(result['sources'])
    if (result['status'],result['logical_rows'],result['all_RAW_candidate_rows'],result['numeric_fields'])!=('COMPLETE',207,92,177):raise ValueError('Fixed numeric universe differs')
    close=read(OUT/'audits/execution_closeout.json')
    if close['status']!='COMPLETE' or not close['all_new_RAW_precede_any_new_D0']:raise ValueError('Execution must close first')
    return result


def paired_conditions(reference,candidate,metric,policy):
    by={r['condition_id']:r for r in candidate};accepted=[];av=[];bv=[];dv=[];reasons=collections.Counter()
    if len(by)!=len(candidate) or set(by)!={r['condition_id'] for r in reference}:raise ValueError('Condition pairing differs')
    for a in reference:
        b=by[a['condition_id']]
        if any(a[k]!=b[k] for k in ('parent_id','report_group','stage')):raise ValueError('Condition metadata differs')
        x,xr=S.eligible(a,metric,policy);y,yr=S.eligible(b,metric,policy)
        if xr or yr:reasons['reference='+(xr or 'VALID')+';candidate='+(yr or 'VALID')]+=1;continue
        accepted.append(a);av.append((a,x));bv.append((a,y));dv.append((a,y-x))
    aa,bb,dd=S.parent_values(av),S.parent_values(bv),S.parent_values(dv)
    return dict(metric=metric,eligibility=policy,pairing='same_C_different_registered_noise_allowed',
        delta_definition='SELECTED_minus_fixed_reference',reference_mean_on_common_support=statistics.fmean(aa.values()) if aa else None,
        candidate_mean_on_common_support=statistics.fmean(bb.values()) if bb else None,mean_delta=statistics.fmean(dd.values()) if dd else None,
        parent_decrease=sum(d<0 for d in dd.values()),parent_equal=sum(d==0 for d in dd.values()),parent_increase=sum(d>0 for d in dd.values()),
        parent_deltas=dd,reference_parent_values=aa,candidate_parent_values=bb,**S.support(reference,accepted,reasons))


def deletion_tables(rows):
    import numpy as np
    from .gallery import cell
    qs={q['request_id']:q for q in candidates()};output=[];fates=[];cache={}
    def areas(tri):return .5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    for row in rows:
        if row['stage']=='RAW':continue
        q=qs[row['request_id']];key=(q['request_id'],row['stage'])
        if key not in cache:
            raw=arrays(raw_dir(q)/'raw.npz');pre=arrays(post_dir(q)/'pre_geo.npz')['output']
            geo=read(post_dir(q)/'geometry.json');mesh=arrays(cell(q,row['label'],row['stage'])['path'])
            k,n=int(q['K']),int(q['N']);ra,pa=areas(raw['raw']),areas(pre)
            if len(ra)!=n or len(pa)!=n:raise ValueError('Deletion slot domain differs')
            byid={int(r['source_face_id']):r for r in geo['face_records']}
            if set(byid)!=set(range(n)):raise ValueError('GEO source slot ledger missing')
            ids=mesh.get('parent_source_raw_slot',mesh.get('source_face_ids'))
            if ids is None or len(ids)!=len(mesh['faces']) or any(int(i)>=n for i in ids):raise ValueError('Output slot provenance')
            retained={int(i) for i in ids if k<=i<n};deleted=sorted(set(range(k,n))-retained)
            if row['stage']=='BASE_POST' and retained!={i for i,r in byid.items() if i>=k and r['kept']}:raise ValueError('GEO receipt retention mismatch')
            raw_area=float(ra[k:].sum());final_area=float(areas(mesh['output'])[k:].sum())
            stats=dict(raw_free_faces=n-k,retained_original_free_faces=len(retained),deleted_original_free_faces=len(deleted),
                added_bridge_faces=int((ids<0).sum()),raw_free_area=raw_area,pre_geo_free_area=float(pa[k:].sum()),
                deleted_raw_reference_area=float(ra[deleted].sum()),deleted_pre_geo_reference_area=float(pa[deleted].sum()),
                actual_final_free_area=final_area,final_over_own_raw_area=final_area/raw_area if raw_area>0 else None,
                deleted_raw_area_fraction=float(ra[deleted].sum()/raw_area) if raw_area>0 else None,
                original_slot_retention_fraction=len(retained)/(n-k),identity_domain='generated_RAW_slot_not_GT')
            slotrows=[dict(raw_slot=i,retained=i in retained,raw_area=float(ra[i]),pre_geo_area=float(pa[i]),
                GEO_kept=bool(byid[i]['kept']),GEO_reason=byid[i]['reason']) for i in range(k,n)]
            cache[key]=stats,slotrows
        stats,slotrows=cache[key];meta={k:row[k] for k in ('label','stage','request_id','condition_id','parent_id','report_group')}
        output.append(dict(**meta,**stats));fates.extend(dict(**meta,**r) for r in slotrows)
    return output,fates


def numeric():
    prerequisites();rows=read(OUT/'results/metric_rows.json');fields=read(MANIFEST)['metric_fields']
    if len(rows)!=207 or len({(r['label'],r['stage'],r['condition_id']) for r in rows})!=207:raise ValueError('Logical comparison cells')
    macro=[];primary=[];paired=[];paired_primary=[]
    for stage in STAGES:
        for group in ('ALL',*GROUPS):
            by={l:[r for r in rows if r['stage']==stage and r['label']==l and (group=='ALL' or r['report_group']==group)] for l in LABELS}
            if not by['SELECTED']:continue
            for label,part in by.items():
                meta=dict(label=label,stage=stage,report_group=group)
                macro.extend(dict(**meta,**S.aggregate(part,m,'ORIGINAL_FINITE_ONLY')) for m in fields)
                primary.extend(dict(**meta,**S.aggregate(part,m,'I_PASS_ONLY' if m in S.PASS_ONLY_I else 'ORIGINAL_FINITE_ONLY')) for m in CORE)
            for ref in LABELS[:2]:
                meta=dict(reference=ref,candidate='SELECTED',stage=stage,report_group=group)
                paired.extend(dict(**meta,**paired_conditions(by[ref],by['SELECTED'],m,'ORIGINAL_FINITE_ONLY')) for m in fields)
                paired_primary.extend(dict(**meta,**paired_conditions(by[ref],by['SELECTED'],m,'I_PASS_ONLY' if m in S.PASS_ONLY_I else 'ORIGINAL_FINITE_ONLY')) for m in CORE)
    area,fates=deletion_tables(rows);area_macro=[]
    for stage in STAGES[1:]:
        for group in ('ALL',*GROUPS):
            for label in LABELS:
                part=[r for r in area if r['stage']==stage and r['label']==label and (group=='ALL' or r['report_group']==group)]
                if not part:continue
                rec=dict(label=label,stage=stage,report_group=group,conditions=len(part),parents=len({r['parent_id'] for r in part}))
                for m in ('raw_free_area','actual_final_free_area','deleted_original_free_faces','deleted_raw_reference_area','final_over_own_raw_area','deleted_raw_area_fraction','original_slot_retention_fraction'):
                    v=S.parent_values([(r,r[m]) for r in part if r[m] is not None]);rec[m+'_parent_macro']=statistics.fmean(v.values()) if v else None
                denom=sum(r['raw_free_area'] for r in part);rec['pooled_final_over_raw_area']=sum(r['actual_final_free_area'] for r in part)/denom if denom>0 else None
                area_macro.append(rec)
    tables=dict(full177_parent_macro=macro,primary_parent_macro=primary,paired177_parent_macro=paired,
        paired_primary_parent_macro=paired_primary,deletion_area_per_output=area,deletion_fate_per_slot=fates,area_parent_macro=area_macro)
    for name,data in tables.items():csvwrite(DEST/(name+'.csv'),data)
    result=dict(status='COMPLETE',numeric_fields=177,logical_rows=207,all_RAW_candidates=92,
        comparisons=['SELECTED-FIXED_N0','SELECTED-FIXED_N1'],pairing='condition_not_same_noise',model_calls=0,scoring_calls=0,
        area_note='FINAL/ownRAW area ratio includes displacement/deletion/bridges; it is not retained GT surface coverage. Deleted generated slots use RAW and pre-GEO coordinates separately.',
        files={str(DEST/(name+'.csv')):sha(DEST/(name+'.csv')) for name in tables},code_sha256=sha(__file__))
    immutable(DEST/'numeric_complete.json',result);return result


def costs():
    prerequisites()
    raw=read(OUT/'evaluation'/LABEL/'raw_complete_full.json')
    post=read(OUT/'evaluation'/LABEL/'post_complete_full.json')
    close=read(OUT/'audits/execution_closeout.json')
    rawcpu=read(OUT/'evaluation/completions/raw.json');postcpu=read(OUT/'evaluation/completions/post.json')
    paths=[OUT/'evaluation'/LABEL/'raw_complete_full.json',OUT/'evaluation'/LABEL/'post_complete_full.json',
           OUT/'audits/execution_closeout.json',OUT/'evaluation/completions/raw.json',OUT/'evaluation/completions/post.json']
    result=dict(status='COMPLETE',generator='frozen_NATURAL_HIGH_B_global22500',training_updates=0,
        current_new_RAW=46,current_generator_forwards=2300,cached_RAW=46,total_RAW_candidates=92,
        current_new_D0=close['new_D0_forwards'],cached_fixed_reference_D0=46,reused_selected_D0=close['reused_winner_D0'],
        selected_conditions=23,failed_conditions=close['failed_conditions'],
        current_new_score_cells=46+2*close['new_D0_forwards'],cached_score_cells=138,
        unique_saved_score_cells=138+46+2*close['new_D0_forwards'],logical_comparison_score_cells=207,
        cold_budget_interpretation=dict(fixed_single_noise_RAW=23,fixed_single_noise_generator_forwards=1150,
            bestof4_RAW=92,bestof4_generator_forwards=4600,selected_single_D0=23,
            note='Conceptual from-scratch budget only; not additional executed calls. Marginal reuse cost differs.'),
        actual_sampling_receipt={k:v for k,v in raw.items() if k not in ('files','protected','code')},
        actual_D0_receipt={k:v for k,v in post.items() if k not in ('files','protected','code')},
        actual_RAW_CPU_seconds=rawcpu['seconds'],actual_POST_CPU_seconds=postcpu['seconds'],
        actual_RAW_CPU_workers=rawcpu['workers'],actual_POST_CPU_workers=postcpu['workers'],
        timing_note='Sampling/D0/CPU overlapped other authorized work; elapsed times are not isolated speed benchmarks.',
        sources={str(p):sha(p) for p in paths})
    immutable(DEST/'costs.json',result);return result


def visual():
    rows,sources=complete_visual_rows();winners={r['condition_id']:r['request_id'] for r in read(OUT/'selection/winners.json')}
    raw={(r['condition_id'],r['request_id']):r for r in rows if r['stage']=='RAW'};logical=[]
    for r in rows:
        if r['stage']!='RAW':logical.append(dict(r,observation_origin='ACTUAL_POST_CELL'))
    for cid in sorted(winners):
        part=[r for (c,_),r in raw.items() if c==cid]
        for label,r in [('FIXED_N0',next(r for r in part if r['noise_index']==0)),('FIXED_N1',next(r for r in part if r['noise_index']==1)),('SELECTED',raw[cid,winners[cid]])]:
            logical.append(dict(r,label=label,source_actual_label=r['label'],observation_origin='LOOKUP_FROM_ACTUALLY_VIEWED_RAW_CANDIDATE'))
    if len(logical)!=207:raise ValueError('Visual comparison universe')
    axes=read(OUT/'visual_review/protocol.json')['axes'];counts=[];transitions=[]
    def joint(r):return r['overall_shape']=='REASONABLE' and r['C_use']=='PLAUSIBLE' and r['support_retention']=='PRESENT'
    by={(r['condition_id'],r['stage'],r['label']):r for r in logical}
    for stage in STAGES:
        for group in ('ALL',*GROUPS):
            for label in LABELS:
                part=[r for r in logical if r['stage']==stage and r['label']==label and (group=='ALL' or r['report_group']==group)]
                if not part:continue
                vals=S.parent_values([(r,float(joint(r))) for r in part])
                rec=dict(stage=stage,report_group=group,label=label,conditions=len(part),parents=len(vals),three_axis_joint_count=sum(joint(r) for r in part),
                    three_axis_joint_parent_macro=statistics.fmean(vals.values()),uncertain_any_axis=sum(any(r[a]=='UNCERTAIN' for a in axes) for r in part))
                for axis,values in axes.items():
                    for v in values:rec[axis+'__'+v]=sum(r[axis]==v for r in part)
                counts.append(rec)
        for cid in winners:
            b=by[cid,stage,'SELECTED']
            for label in LABELS[:2]:
                a=by[cid,stage,label]
                t=dict(condition_id=cid,parent_id=b['parent_id'],report_group=b['report_group'],stage=stage,reference=label,candidate='SELECTED',
                    reference_request_id=a['request_id'],selected_request_id=b['request_id'],reference_joint=joint(a),selected_joint=joint(b),
                    joint_transition='GAIN' if joint(b)>joint(a) else 'LOSS' if joint(b)<joint(a) else 'UNCHANGED')
                for axis in axes:t[axis+'__reference']=a[axis];t[axis+'__selected']=b[axis]
                transitions.append(t)
    raw_capacity=[]
    for cid,rid in winners.items():
        part=[r for (c,_),r in raw.items() if c==cid];selected=raw[cid,rid]
        raw_capacity.append(dict(condition_id=cid,parent_id=selected['parent_id'],report_group=selected['report_group'],
            candidate_joint_count=sum(joint(r) for r in part),any_candidate_joint=any(joint(r) for r in part),selected_joint=joint(selected),
            selected_request_id=rid,selector_missed_available_joint=any(joint(r) for r in part) and not joint(selected),
            interpretation='Retrospective visual description only; not an oracle selector or tuning signal'))
    tables=dict(actual_observations=rows,logical_comparison_observations=logical,visual_counts=counts,visual_transitions=transitions,RAW_candidate_capacity=raw_capacity)
    for name,data in tables.items():csvwrite(DEST/(name+'.csv'),data)
    result=dict(status='COMPLETE',actual_pages=69,actual_observations=230,all_RAW_actual_observations=92,logical_comparison_rows=207,
        model_calls=0,scoring_calls=0,new_ratings=0,reviewer_kind='AI_PROXY',blind=False,historical_label_pooling=False,
        three_axis_note='Descriptive REASONABLE and PLAUSIBLE and PRESENT intersection; not a new formal metric and does not certify no fragments/folds/intersections.',
        sources=sources,files={str(DEST/(name+'.csv')):sha(DEST/(name+'.csv')) for name in tables},code_sha256=sha(__file__))
    immutable(DEST/'visual_complete.json',result);return result


def package():
    prerequisites();verify(read(DEST/'numeric_complete.json')['files']);verify(read(DEST/'visual_complete.json')['files'])
    report=OUT/'REPORT.md'
    if not report.exists():raise ValueError('Final branch report required')
    sources=dict(read(OUT/'registration.json')['code'])
    verify(sources)
    frozen={resolve(p):h for p,h in sources.items()}
    for p in CODE.glob('*.py'):
        h=sha(p)
        if p in frozen and frozen[p]!=h:raise ValueError('Frozen source changed before archive')
        if p not in frozen:sources[str(p)]=h
    verify(sources)
    archive={str(resolve(p).relative_to(ROOT)):h for p,h in sources.items()}
    immutable(OUT/'archive/source_hashes.json',dict(status='FROZEN_SOURCE_SNAPSHOT',files=archive,model_calls=0))
    files={report,OUT/'registration.json',OUT/'archive/source_hashes.json',OUT/'audits/execution_closeout.json'}
    for folder in ('data','selection','results','visual_review','audits'):
        files.update(p for p in (OUT/folder).rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.md','.html'))
    files.update(resolve(p) for p in sources)
    files.update(p for p in (OUT/'gallery').iterdir() if p.is_file() and p.suffix in ('.json','.html'))
    def build(include_images):
        payload=io.BytesIO()
        chosen=files|set((OUT/'gallery').glob('*.png')) if include_images else files
        with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p in sorted(chosen):z.write(p,str(p.relative_to(ROOT)))
        return payload.getvalue(),chosen
    raw,chosen=build(True);images=True
    if len(raw)>=10_000_000:raw,chosen=build(False);images=False
    if len(raw)>=10_000_000:raise ValueError('Evidence-only package exceeds10MB; no file silently omitted')
    dest=OUT/'BESTOF4_REVIEW_SMALL.zip'
    if dest.exists():raise ValueError('Package already exists; explicit review required')
    with dest.open('xb') as f:f.write(raw)
    result=dict(status='COMPLETE',archive_path=str(dest),bytes=len(raw),sha256=sha(dest),included_files=len(chosen),
        all69_gallery_PNG_included=images,local_full_gallery=str(OUT/'gallery'),model_weights_meshes_noise_arrays_included=False,
        report_sha256=sha(report),source_snapshot_sha256=sha(OUT/'archive/source_hashes.json'))
    immutable(OUT/'archive/package_receipt.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('numeric','visual','costs','package'));a=p.parse_args()
    r={'numeric':numeric,'visual':visual,'costs':costs,'package':package}[a.action]()
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','sources')},ensure_ascii=False,indent=2))
    if 'torch' in sys.modules:raise RuntimeError('Delivery imported torch')


if __name__=='__main__':main()
