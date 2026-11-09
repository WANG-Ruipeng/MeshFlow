"""Descriptive paired aggregation of already computed fixed-panel metrics.

No model, mesh operation, image review, re-scoring, significance test or scientific
conclusion. Original metric columns stay intact. Boundary-domain summaries use a
separate FULL_BOUNDARY prefix. Six studied parents / four fixed noises are not
independent replication trials.
"""
from collections import defaultdict
import csv, hashlib, io, json, math
import numpy as np
from . import common as C

VERSION='full_contour_readout_paired_comparison_v1'
MAIN=('CONTOUR_S0','CONTOUR_S1000','CONTOUR_S3000','CONTOUR_S6000')
REFERENCES=('FULL_S0000','FULL_S1000','FULL_S3000','FULL_S6000')
STEPS=dict(zip(MAIN,(0,1000,3000,6000)));STEPS.update(zip(REFERENCES,(0,1000,3000,6000)))
STAGES=('RAW','POST')
EV=C.OUT/'evaluation'
EXCLUDE={'request_id','parent_id','condition_id','panel','candidate','region_type','tier','N','K','ratio','base_request_id','base_condition_id','variant','alpha','pair_id','split','condition_type','label','stage','local_step','checkpoint_role','score_origin','score_path','score_sha256','score_present','boundary_present','boundary_receipt_path','boundary_summary_path','stratum','missing_reason'}


def number(value):
    if value is None or value=='':return None
    if isinstance(value,bool):return float(value)
    if value=='True':return 1.
    if value=='False':return 0.
    try:x=float(value)
    except (ValueError,TypeError):return None
    return x if math.isfinite(x) else None


def csv_read(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def csv_write(path,rows,fields=('status',)):
    rows=list(rows)
    if rows:return C.csvwrite(path,rows)
    stream=io.StringIO(newline='');csv.DictWriter(stream,fieldnames=list(fields)).writeheader()
    return C._write_bytes(path,stream.getvalue().encode('utf-8-sig'))


def _is_metric(key,rows):
    if key in EXCLUDE or key.lower().endswith('status') or key.endswith('_path') or key.endswith('_sha256'):return False
    values=[r[key] for r in rows if key in r and r[key] not in ('',None)]
    if values:return all(number(v) is not None for v in values)
    return key.startswith(('CF_','FF_','ConflictCov_C','C_F_','A_VMR__','PMR__','DESC__','FULL_BOUNDARY__','free_','whole_','boundary_distance_','opposite_winding_','reached_free_','EXACT_COORD_','RAW_SOUP_'))


def load_inputs(reg):
    requests={q['request_id']:q for q in reg['requests']};expected={(rid,stage) for rid in requests for stage in STAGES}
    if len(requests)!=24 or len({q['parent_id'] for q in requests.values()})!=6:raise ValueError('Fixed six-parent24request panel required')
    source_hashes={str(C.OUT/'registration.json'):C.sha(C.OUT/'registration.json')};available={};availability=[]
    for label in MAIN+REFERENCES:
        path=EV/'tables'/label/'per_output.csv';rows=csv_read(path) if path.exists() else []
        if path.exists():source_hashes[str(path)]=C.sha(path)
        score_rows={}
        for row in rows:
            key=(row['request_id'],row['stage'])
            if key not in expected or key in score_rows or row['label']!=label:raise ValueError('Unexpected/duplicate score table identity')
            q=requests[key[0]]
            for name in ('parent_id','condition_id','panel','split','condition_type'):
                if row[name]!=str(q[name]):raise ValueError('Table identity differs: '+name)
            for name in ('candidate','K','N'):
                if int(row[name])!=int(q[name]):raise ValueError('Table numerical identity differs: '+name)
            score_rows[key]=row
        receipt_path=EV/'boundary_completion'/f'{label}.json';receipts={}
        if receipt_path.exists():
            block=C.read(receipt_path);source_hashes[str(receipt_path)]=C.sha(receipt_path)
            if block['status']!='COMPLETE' or block['label']!=label:raise ValueError('Invalid boundary completion')
            for r in block['receipts']:
                key=(r['request_id'],r['stage'])
                if key in receipts:raise ValueError('Repeated boundary completion cell')
                receipts[key]=(r,str(receipt_path))
        else:
            # Partial work is read only through already committed per-cell receipts.
            for rid,stage in sorted(expected):
                p=EV/'boundary'/label/rid/stage/'completion.json'
                if p.exists():
                    receipts[(rid,stage)]=(C.read(p),str(p));source_hashes[str(p)]=C.sha(p)
        if not set(receipts).issubset(expected):raise ValueError('Boundary outside registered requests')
        for rid,stage in sorted(expected):
            q=requests[rid];key=(rid,stage);row=dict(score_rows.get(key,{}))
            for name in ('request_id','parent_id','condition_id','panel','candidate','split','condition_type','K','N'):row[name]=q[name]
            row.update(label=label,stage=stage,local_step=STEPS.get(label),checkpoint_role='LEARNING_CURVE' if label in MAIN else 'FROZEN_CONTEXT_REFERENCE',
                score_present=key in score_rows,boundary_present=False,stratum=q['split'])
            missing=[]
            if key not in score_rows:missing.append('SCORE_TABLE_NOT_AVAILABLE')
            if key in receipts:
                rec,rpath=receipts[key];identity=rec['identity']
                if rec['status']!='COMPLETE' or rec['label']!=label or identity['label']!=label or identity['request_id']!=rid or identity['stage']!=stage:raise ValueError('Boundary receipt identity mismatch')
                if identity['registration_sha256']!=source_hashes[str(C.OUT/'registration.json')]:raise ValueError('Boundary registration hash changed')
                if identity['source_sha256']!=q['condition_source_sha256']:raise ValueError('Boundary source C differs')
                summary_path=rec['summary_path'];actual=C.sha(summary_path)
                if rec['files'].get(summary_path)!=actual:raise ValueError('Boundary summary digest differs')
                if row.get('score_sha256') and identity['score_sha256']!=row['score_sha256']:raise ValueError('Boundary versus score identity differs')
                source_hashes[str(summary_path)]=actual;summary=C.read(summary_path)
                if float(summary['L'])!=C.L or int(summary['source_C_faces'])!=int(q['K']):raise ValueError('Boundary scale/denominator differs')
                for field in ('source_boundary_edges','source_boundary_vertices','source_boundary_length','edge_midpoint_queries','boundary_vertex_queries','sampling_max_step_L','free_faces','observed_F_edges','observed_F_open_edges','observed_F_vertices','observed_F_components','isolated_normal_hits','coplanar_line_intervals','normal_layer_groups'):
                    row['FULL_BOUNDARY__'+field]=summary.get(field)
                for domain in ('boundary_distances','boundary_vertex_distances'):
                    for geometric_set,stats in summary[domain].items():
                        for metric,value in stats.items():row['FULL_BOUNDARY__'+domain+'__'+geometric_set+'__'+metric]=value
                row.update(boundary_present=True,boundary_receipt_path=rpath,boundary_summary_path=summary_path)
            else:missing.append('BOUNDARY_COMPLETION_NOT_AVAILABLE')
            row['missing_reason']=';'.join(missing)
            available[(rid,label,stage)]=row
            availability.append({k:row[k] for k in ('request_id','parent_id','candidate','stratum','label','local_step','stage','checkpoint_role','score_present','boundary_present','missing_reason')})
    union=list(dict.fromkeys(k for r in available.values() for k in r));metrics=[k for k in union if _is_metric(k,list(available.values()))]
    return available,availability,metrics,source_hashes


def make_pairs(reg,available,metrics,labels):
    rows=[]
    for label in labels:
        for q in reg['requests']:
            rid=q['request_id']
            for stage in STAGES:
                a=available[rid,'CONTOUR_S0',stage];b=available[rid,label,stage]
                row=dict(request_id=rid,parent_id=q['parent_id'],condition_id=q['condition_id'],candidate=q['candidate'],stratum=q['split'],
                    label=label,local_step=STEPS.get(label),stage=stage,baseline='CONTOUR_S0',K=q['K'],N=q['N'],
                    baseline_score_present=a['score_present'],current_score_present=b['score_present'],
                    baseline_boundary_present=a['boundary_present'],current_boundary_present=b['boundary_present'],
                    status='COMPLETE' if all((a['score_present'],b['score_present'],a['boundary_present'],b['boundary_present'])) else 'PARTIAL')
                for metric in metrics:
                    av,bv=number(a.get(metric)),number(b.get(metric))
                    row[metric+'__step0']=av;row[metric+'__current']=bv;row[metric+'__delta_current_minus_step0']=bv-av if av is not None and bv is not None else None
                rows.append(row)
    return rows


def parent_summary(reg,available,metrics,labels,pairs=None):
    parent_ids=list(dict.fromkeys(q['parent_id'] for q in reg['requests']));pindex={}
    if pairs is not None:pindex={(p['request_id'],p['label'],p['stage']):p for p in pairs}
    result=[]
    for label in labels:
        for uid in parent_ids:
            qs=sorted((q for q in reg['requests'] if q['parent_id']==uid),key=lambda q:q['candidate'])
            if [q['candidate'] for q in qs]!=[0,1,2,3]:raise ValueError('Expected four frozen noises per parent')
            for stage in STAGES:
                for metric in metrics:
                    field=metric+'__delta_current_minus_step0' if pairs is not None else metric
                    samples=[(pindex[q['request_id'],label,stage] if pairs is not None else available[q['request_id'],label,stage]) for q in qs]
                    vals=[number(s.get(field)) for s in samples];valid=[v for v in vals if v is not None]
                    score_count=sum(bool(s['current_score_present'] and s['baseline_score_present']) if pairs is not None else bool(s['score_present']) for s in samples)
                    boundary_count=sum(bool(s['current_boundary_present'] and s['baseline_boundary_present']) if pairs is not None else bool(s['boundary_present']) for s in samples)
                    inputs_complete=boundary_count==4 if metric.startswith('FULL_BOUNDARY__') else score_count==4
                    row=dict(label=label,local_step=STEPS.get(label),parent_id=uid,stratum=qs[0]['split'],stage=stage,metric=metric,
                        quantity='paired_current_minus_step0' if pairs is not None else 'absolute',n0=vals[0],n1=vals[1],n2=vals[2],n3=vals[3],
                        noise_macro_mean=float(np.mean(valid)) if valid else None,noise_min=min(valid) if valid else None,noise_max=max(valid) if valid else None,
                        valid_noise_values=len(valid),expected_noise_values=4,score_requests_present=score_count,boundary_requests_present=boundary_count,
                        status='PARTIAL_INPUTS' if not inputs_complete else 'COMPLETE' if len(valid)==4 else 'NA_SOME_METRIC_VALUES',
                        aggregation='Equal weight per available fixed noise; missing metric values stay NA and valid counts are explicit.')
                    result.append(row)
    return result


def strata_summary(parent_rows,labels):
    grouped=defaultdict(list)
    for row in parent_rows:grouped[(row['label'],row['stage'],row['metric'])].append(row)
    result=[]
    for (label,stage,metric),rows in grouped.items():
        if label not in labels:continue
        for stratum in ('TRAIN','DEV','ALL'):
            selected=[r for r in rows if stratum=='ALL' or r['stratum']==stratum]
            valid=[float(r['noise_macro_mean']) for r in selected if r['noise_macro_mean'] is not None]
            complete=sum(r['status']=='COMPLETE' for r in selected)
            result.append(dict(label=label,local_step=STEPS.get(label),stage=stage,stratum=stratum,metric=metric,quantity=selected[0]['quantity'],
                parent_macro_mean=float(np.mean(valid)) if valid else None,parent_min=min(valid) if valid else None,parent_max=max(valid) if valid else None,
                valid_parent_means=len(valid),expected_parents=len(selected),complete_four_noise_parents=complete,
                valid_noise_values=sum(r['valid_noise_values'] for r in selected),expected_noise_values=4*len(selected),
                status='COMPLETE' if complete==len(selected) else 'PARTIAL_OR_NA',
                aggregation='Equal weight per parent mean, each from available fixed-noise values; partial coverage is not a completed-panel estimate.',
                scope='Six studied parents, four previously fixed noises each; TRAIN and DEV separate; no independence, repeatability or significance claim.'))
    return result


def load_reviews(reg,sources):
    from r2.pilots.full_seat_learning_curve_v1 import review_summary as R
    requests={q['request_id']:q for q in reg['requests']};images=[];audit=[];available={}
    baseline=C.ROOT/'experiments/r2_full_seat_learning_curve_v1/evaluation/review/AI_PROXY_reviews.csv'
    current=EV/'review/AI_PROXY_reviews.csv'
    for path,labels in ((baseline,REFERENCES),(current,MAIN[1:])):
        vals,_,checks,details=R.capture_reviews(path,labels,requests,sources,images,False)
        available.update(vals);audit.extend(checks)
    # The initial candidate reuses source bytes and labels; explicitly inherited.
    for q in reg['requests']:
        for stage in STAGES:
            old=available.get(('FULL_S0000',q['request_id'],stage))
            if old:available['CONTOUR_S0',q['request_id'],stage]=dict(old,label='CONTOUR_S0',review_provenance='INHERITED_IDENTICAL_FULL_S0000_GEOMETRY')
    return available,audit


def matched_budget_tables(reg,available,metrics,reviews,review_audit):
    rows=[];joint=[];pairs=[];numeric=[]
    def judgment(label,q):
        rid=q['request_id'];raw=reviews.get((label,rid,'RAW'));post=reviews.get((label,rid,'POST'))
        score=available[rid,label,'POST'];edge=number(score.get('C_F_valid_edge_count'))
        adoption=raw.get('C_adoption') if raw else None
        success=None if edge is None or adoption is None else bool(adoption=='YES' and edge>0)
        if adoption=='UNCERTAIN':state='UNCERTAIN_ADOPTION' if edge is not None and edge>0 else 'UNCERTAIN_ADOPTION_NO_LEGAL_INTERFACE'
        elif adoption is None or edge is None:state='NOT_EVALUATED'
        elif success:state='YES_ADOPTION_AND_LEGAL_INTERFACE'
        else:state='DOES_NOT_MEET_EXPLICIT_JOINT_CRITERION'
        return dict(label=label,local_step=STEPS[label],request_id=rid,parent_id=q['parent_id'],candidate=q['candidate'],stratum=q['split'],
            RAW_C_adoption=adoption,POST_legal_edge_count=edge,joint_explicit_success=success,joint_state=state,
            RAW_whole_body_complete=raw.get('whole_body_complete') if raw else None,
            POST_whole_body_complete=post.get('whole_body_complete') if post else None,
            RAW_fragments_or_folded_layers=raw.get('fragments_or_folded_layers') if raw else None,
            POST_fragments_or_folded_layers=post.get('fragments_or_folded_layers') if post else None,
            POST_legal_edge_total_length=number(score.get('C_F_valid_edge_total_length')),
            RAW_notes=raw.get('notes') if raw else None,POST_notes=post.get('notes') if post else None,
            review_provenance='INHERITED_IDENTICAL_FULL_S0000_GEOMETRY' if label=='CONTOUR_S0' else 'ACTUAL_SAVED_AI_PROXY',
            status='COMPLETE' if edge is not None and raw and post else 'PARTIAL')
    for label in MAIN+REFERENCES:
        for q in reg['requests']:joint.append(judgment(label,q))
    ji={(r['label'],r['request_id']):r for r in joint}
    for candidate,base in zip(MAIN,REFERENCES):
        for q in reg['requests']:
            rid=q['request_id'];a=ji[base,rid];b=ji[candidate,rid]
            av,bv=a['joint_explicit_success'],b['joint_explicit_success']
            change='NA' if av is None or bv is None else 'RETAINED' if av and bv else 'LOST' if av else 'GAINED' if bv else 'NEITHER'
            pairs.append(dict(request_id=rid,parent_id=q['parent_id'],candidate=q['candidate'],stratum=q['split'],
                label=candidate,local_step=STEPS[candidate],baseline=base,baseline_joint=av,current_joint=bv,joint_transition=change,
                baseline_adoption=a['RAW_C_adoption'],current_adoption=b['RAW_C_adoption'],
                baseline_legal_edges=a['POST_legal_edge_count'],current_legal_edges=b['POST_legal_edge_count'],
                baseline_body=a['RAW_whole_body_complete'],current_body=b['RAW_whole_body_complete'],
                baseline_notes=a['RAW_notes'],current_notes=b['RAW_notes'],
                status='COMPLETE' if a['status']==b['status']=='COMPLETE' else 'PARTIAL'))
            for stage in STAGES:
                old=available[rid,base,stage];new=available[rid,candidate,stage]
                row=dict(request_id=rid,parent_id=q['parent_id'],candidate=q['candidate'],stratum=q['split'],stage=stage,
                    label=candidate,local_step=STEPS[candidate],baseline=base)
                for metric in metrics:
                    x,y=number(old.get(metric)),number(new.get(metric))
                    row[metric+'__baseline']=x;row[metric+'__candidate']=y;row[metric+'__delta']=y-x if x is not None and y is not None else None
                numeric.append(row)
    summaries=[]
    for label in MAIN+REFERENCES:
        for stratum in ('TRAIN','DEV','ALL'):
            selected=[x for x in joint if x['label']==label and (stratum=='ALL' or x['stratum']==stratum)]
            valid=[x for x in selected if x['joint_explicit_success'] is not None]
            summaries.append(dict(label=label,local_step=STEPS[label],stratum=stratum,expected=len(selected),evaluated=len(valid),
                explicit_joint_success=sum(x['joint_explicit_success'] for x in valid),
                adoption_YES=sum(x['RAW_C_adoption']=='YES' for x in selected),adoption_NO=sum(x['RAW_C_adoption']=='NO' for x in selected),
                adoption_UNCERTAIN=sum(x['RAW_C_adoption']=='UNCERTAIN' for x in selected),
                legal_interface_outputs=sum(x['POST_legal_edge_count']>0 for x in selected if x['POST_legal_edge_count'] is not None),
                status='COMPLETE' if all(x['status']=='COMPLETE' for x in selected) else 'PARTIAL'))
    parent_pairs=[];strata_pairs=[]
    for label in MAIN:
        selected=[r for r in numeric if r['label']==label]
        for stage in STAGES:
            for uid in dict.fromkeys(q['parent_id'] for q in reg['requests']):
                samples=sorted((r for r in selected if r['stage']==stage and r['parent_id']==uid),key=lambda x:x['candidate'])
                for metric in metrics:
                    values=[r[metric+'__delta'] for r in samples];valid=[v for v in values if v is not None]
                    parent_pairs.append(dict(label=label,baseline=samples[0]['baseline'],local_step=STEPS[label],stage=stage,
                        parent_id=uid,stratum=samples[0]['stratum'],metric=metric,n0=values[0],n1=values[1],n2=values[2],n3=values[3],
                        delta_noise_mean=float(np.mean(valid)) if valid else None,delta_noise_median=float(np.median(valid)) if valid else None,
                        delta_min=min(valid) if valid else None,delta_max=max(valid) if valid else None,
                        negative_delta_noises=sum(v<0 for v in valid),positive_delta_noises=sum(v>0 for v in valid),
                        valid_noise_count=len(valid),expected_noise_count=4,status='COMPLETE' if len(valid)==4 else 'PARTIAL_OR_NA'))
    for label in MAIN:
        for stage in STAGES:
            for stratum in ('TRAIN','DEV','ALL'):
                for metric in metrics:
                    selected=[r for r in parent_pairs if r['label']==label and r['stage']==stage and r['metric']==metric and (stratum=='ALL' or r['stratum']==stratum)]
                    values=[r['delta_noise_mean'] for r in selected if r['delta_noise_mean'] is not None]
                    strata_pairs.append(dict(label=label,local_step=STEPS[label],stage=stage,stratum=stratum,metric=metric,
                        paired_parent_macro_delta=float(np.mean(values)) if values else None,
                        paired_parent_median_delta=float(np.median(values)) if values else None,
                        paired_parent_min_delta=min(values) if values else None,paired_parent_max_delta=max(values) if values else None,
                        negative_delta_parents=sum(v<0 for v in values),positive_delta_parents=sum(v>0 for v in values),
                        valid_parent_count=len(values),expected_parent_count=len(selected),
                        status='COMPLETE' if all(r['status']=='COMPLETE' for r in selected) else 'PARTIAL_OR_NA'))
    return {'joint_per_request.csv':joint,'joint_strata.csv':summaries,'joint_matched_budget_gains_losses.csv':pairs,
        'final_joint_vs_FULL_S6000.csv':[x for x in pairs if x['label']=='CONTOUR_S6000'],
        'matched_budget_per_output.csv':numeric,'final_vs_FULL_S6000.csv':[x for x in numeric if x['label']=='CONTOUR_S6000'],
        'actual_review_validation.csv':review_audit,'matched_budget_per_parent.csv':parent_pairs,'matched_budget_strata.csv':strata_pairs,
        'final_vs_FULL_S6000_parent.csv':[r for r in parent_pairs if r['label']=='CONTOUR_S6000'],
        'final_vs_FULL_S6000_strata.csv':[r for r in strata_pairs if r['label']=='CONTOUR_S6000']}


def run():
    registration_path=C.OUT/'registration.json'
    if not registration_path.exists():return dict(status='NA_REGISTRATION_NOT_FROZEN',model_calls=0)
    reg=C.read(registration_path)
    if reg.get('status')!='FROZEN':raise ValueError('Frozen evaluation registration required')
    available,availability,metrics,sources=load_inputs(reg)
    sources[str(C.resolve(__file__))]=C.sha(__file__)
    reviews,review_audit=load_reviews(reg,sources)
    tag=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:14]
    out=EV/'comparison'/('snapshot_'+tag);final=out/'summary.json'
    if final.exists():
        previous=C.read(final)
        for path,h in previous['output_hashes'].items():assert C.sha(path)==h,path
        C.write(EV/'comparison/latest.json',dict(path=str(final),sha256=C.sha(final),status=previous['status']),replace=True)
        return previous
    out.mkdir(parents=True,exist_ok=False)
    all_rows=list(available.values());mainrows=[r for r in all_rows if r['label'] in MAIN];refrows=[r for r in all_rows if r['label'] in REFERENCES]
    pairs=make_pairs(reg,available,metrics,MAIN[1:]);refpairs=make_pairs(reg,available,metrics,REFERENCES)
    parents=parent_summary(reg,available,metrics,MAIN);refparents=parent_summary(reg,available,metrics,REFERENCES)
    paired_parents=parent_summary(reg,available,metrics,MAIN[1:],pairs);refpaired_parents=parent_summary(reg,available,metrics,REFERENCES,refpairs)
    strata=strata_summary(parents,MAIN);refstrata=strata_summary(refparents,REFERENCES)
    paired_strata=strata_summary(paired_parents,MAIN[1:]);refpaired_strata=strata_summary(refpaired_parents,REFERENCES)
    files={'availability.csv':availability,'main_per_output.csv':mainrows,'reference_per_output.csv':refrows,
        'main_paired_to_step0.csv':pairs,'final_vs_step0.csv':[p for p in pairs if p['label']=='CONTOUR_S6000'],
        'main_per_parent_four_noises.csv':parents,'main_strata.csv':strata,
        'main_paired_parent_four_noises.csv':paired_parents,'main_paired_strata.csv':paired_strata,
        'final_vs_step0_parent.csv':[p for p in paired_parents if p['label']=='CONTOUR_S6000'],
        'final_vs_step0_strata.csv':[p for p in paired_strata if p['label']=='CONTOUR_S6000'],
        'reference_per_parent_four_noises.csv':refparents,'reference_strata.csv':refstrata,
        'reference_vs_step0.csv':refpairs,'reference_paired_parent_four_noises.csv':refpaired_parents,'reference_paired_strata.csv':refpaired_strata,
        'metric_inventory.csv':[dict(metric=m,origin='Full source boundary attribution summary' if m.startswith('FULL_BOUNDARY__') else 'Inherited original per_output metric/denominator',definition_changed=False) for m in metrics]}
    files.update(matched_budget_tables(reg,available,metrics,reviews,review_audit))
    for name,rows in files.items():csv_write(out/name,rows)
    per_label=[]
    for label in MAIN+REFERENCES:
        rows=[r for r in availability if r['label']==label]
        per_label.append(dict(label=label,role='LEARNING_CURVE' if label in MAIN else 'FROZEN_CONTEXT_REFERENCE',local_step=STEPS.get(label),
            score_cells=sum(r['score_present'] for r in rows),boundary_cells=sum(r['boundary_present'] for r in rows),expected_stage_cells=48,
            status='COMPLETE' if all(r['score_present'] and r['boundary_present'] for r in rows) else 'PARTIAL'))
    numerical_complete=all(r['status']=='COMPLETE' for r in per_label)
    joint_complete=all(r['status']=='COMPLETE' for r in files['joint_per_request.csv'])
    complete=numerical_complete and joint_complete
    summary=dict(status='COMPLETE' if complete else 'PARTIAL',version=VERSION,out=str(out),registered_requests=24,registered_parents=6,fixed_noises_per_parent=4,
        main_labels=list(MAIN),reference_labels=list(REFERENCES),main_endpoint='CONTOUR_S6000',baseline='CONTOUR_S0',strata=['TRAIN','DEV','ALL'],
        numerical_complete=numerical_complete,joint_and_review_complete=joint_complete,
        main_comparator='FULL_S6000',success_rule='RAW adoption YES AND POST C_F_valid_edge_count > 0 on the SAME request; uncertain retained separately; no threshold tuning',
        per_label=per_label,metric_columns=len(metrics),rows_per_table={name:len(rows) for name,rows in files.items()},
        source_hashes=sources,output_hashes={str(p):C.sha(p) for p in out.iterdir() if p.is_file()},
        aggregation='Paired request deltas first; equal-noise parent means; equal-parent TRAIN/DEV/ALL macro means. Every valid count and missing endpoint is retained.',
        missing_data='Future/missing tables or completion receipts produce PARTIAL and NA, never zeros, interpolated results or a best-checkpoint replacement.',
        interpretation='Descriptive evidence tables only. No image review, generated conclusions, significance tests or independent-repeatability claim. Four noises are within six studied parents and were seen in previous experiments.',
        model_calls=0,new_quality_scans=0,new_geometry_analyses=0,new_image_reviews=0)
    C.write(final,summary);C.write(EV/'comparison/latest.json',dict(path=str(final),sha256=C.sha(final),status=summary['status']),replace=True)
    return summary


if __name__=='__main__':
    r=run();print(json.dumps({k:r.get(k) for k in ('status','out','registered_requests','metric_columns','per_label','model_calls')},sort_keys=True))