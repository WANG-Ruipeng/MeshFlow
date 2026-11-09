"""Read-only Bench condition/exposure audit. No models, geometry edits or scores.

All statistics join frozen manifests and actually executed exposure logs. Source
semantic extents remain AI_PROXY descriptions, not new segmentation ground truth.
Run with Python 3; only the standard library is required.
"""
from pathlib import Path
import csv, json, hashlib, collections, statistics, math, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCES = {}

def read(rel):
    p = ROOT / rel
    data = p.read_bytes()
    SOURCES[str(p.relative_to(ROOT))] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return json.loads(data)

def lines(rel):
    p = ROOT / rel
    data = p.read_bytes()
    SOURCES[str(p.relative_to(ROOT))] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return [json.loads(s) for s in data.decode('utf-8-sig').splitlines() if s.strip()]

def number_stats(vals):
    vals = sorted(float(v) for v in vals if v is not None)
    if not vals: return {'n': 0}
    def q(f):
        a=(len(vals)-1)*f; i=int(a); j=min(i+1,len(vals)-1)
        return vals[i]*(j-a)+vals[j]*(a-i) if j != i else vals[i]
    return {'n':len(vals),'min':vals[0],'p10':q(.1),'median':statistics.median(vals),'p90':q(.9),'max':vals[-1],'mean':statistics.mean(vals)}

def tally(rows):
    return {'conditions':len(rows),'parents':len({r['parent_id'] for r in rows}),
            'K':number_stats(r['K'] for r in rows),'N':number_stats(r['N'] for r in rows),
            'area_fraction':number_stats(r['area_fraction'] for r in rows),
            'face_fraction':number_stats(r['face_fraction'] for r in rows),
            'source_edge_components':number_stats(r['source_edge_components'] for r in rows),
            'multicomponent_conditions':sum(r['source_edge_components']>1 for r in rows),
            'closed_conditions':sum(r['boundary_edges']==0 for r in rows),
            'no_owner_conditions':sum(r['source_owner_count']==0 for r in rows),
            'source_owners':sum(r['source_owner_count'] for r in rows),
            'source_aliases':sum(r['source_alias_count'] for r in rows),
            'source_null_unsupported':sum(r['source_null_unsupported_count'] for r in rows),
            'nonplanar_conditions':sum(bool(r['region_normal_not_single_plane']) for r in rows)}

def write_csv(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def main():
    started=time.time()
    m=read('experiments/r2_bench_clean_multipart_seam_v1/data/manifest.json')
    original=read('experiments/r2_bench_parts_convergence_v1/data/manifest.json')
    ev=read('experiments/r2_bench_state_schedule_v1/eval_manifest.json')
    schedule=read('experiments/r2_bench_clean_multipart_seam_v1/data/schedule.json')
    rows=m['rows']; byid={r['condition_id']:r for r in rows}
    active={r['condition_id'] for r in m['train']}
    excluded={r['condition_id']:r['reason'] for r in m['excluded_train']}
    selected=collections.Counter(r['condition_id'] for r in ev['requests'])
    arm_paths={
        'A':'experiments/r2_bench_adaptation_strength_v1/training/A_COMMON_LR1E5/exposure.jsonl',
        'PERT':'experiments/r2_bench_state_schedule_v1/training/PERTURBED_STATE/exposure.jsonl',
        'ASYNC':'experiments/r2_bench_state_schedule_v1/training/ASYNC_LEAD_SEAM/exposure.jsonl',
        'COMMON':'experiments/r2_bench_parts_convergence_v1/training/COMMON/exposure.jsonl'}
    logs={k:lines(p) for k,p in arm_paths.items()}
    stream_keys=['condition_id','parent_id','part_type','noise_seed','epsilon_seed','time_seed','permutation_seed','scale_xyz']
    def identity(r): return tuple(json.dumps(r.get(k),sort_keys=True) for k in stream_keys)
    identities={k:[identity(r) for r in logs[k]] for k in ['A','PERT','ASYNC']}
    planned=[identity(r) for s in schedule['steps'] for r in s['samples']]
    checks={'all_rows_unique':len(byid)==len(rows),'three_arm_streams_match':identities['A']==identities['PERT']==identities['ASYNC'],
            'A_stream_matches_schedule':identities['A']==planned,
            'selected_ids_known':set(selected)<=set(byid),
            'TRAIN_DEV_parent_disjoint':not ({r['parent_id'] for r in m['train']} & {r['parent_id'] for r in m['dev']}),
            'current_rows_identical_to_original':m['rows']==original['rows']}
    counts={k:collections.Counter(r['condition_id'] for r in logs[k]) for k in logs}
    # COMMON replay may repeat a step/micro. Distinguish logged computations from
    # unique scheduled sample identity rather than counting replay as new data.
    common_unique={}
    for r in logs['COMMON']:
        common_unique[(r['step'],r['micro'])]=r
    counts['COMMON_UNIQUE']=collections.Counter(r['condition_id'] for r in common_unique.values())
    per=[]
    for r in rows:
        box=r['bbox_normalized']; span=[box[1][j]-box[0][j] for j in range(3)]
        out={k:r.get(k) for k in ['condition_id','parent_id','group','official_split','part_type','proposed_part_type','extent_description','N','K','face_fraction','area_fraction','source_edge_components','boundary_edges','region_normal_not_single_plane','source_owner_count','source_alias_count','source_null_unsupported_count','seam_applicability','adapter_semantics','semantic_authority']}
        out.update(active_train=r['condition_id'] in active,exclusion_reason=excluded.get(r['condition_id'],''),evaluation_requests=selected[r['condition_id']],
                   original_id_part=r['condition_id'].split('__')[1],
                   id_vs_actual_part_mismatch=r.get('proposed_part_type')!=r['part_type'],
                   extent_authority='EXISTING_NONBLIND_AI_PROXY_NOT_NEW_HUMAN_SEGMENTATION',
                   original_review_notes=r['review'].get('notes',''),
                   span_x=span[0],span_y=span[1],span_z=span[2],
                   center_x=(box[0][0]+box[1][0])/2,center_y=(box[0][1]+box[1][1])/2,center_z=(box[0][2]+box[1][2])/2,
                   span_aspect_max_second=sorted(span)[2]/max(sorted(span)[1],1e-12),
                   construction_method=r.get('construction_evidence',{}).get('method',''),
                   source_geometry_unchanged=r['geometry_unchanged'])
        for arm,c in counts.items(): out[arm+'_exposure']=c[r['condition_id']]
        per.append(out)
    write_csv(HERE/'per_condition.csv',per)
    groups={'ORIGINAL_TRAIN':m['train_original'],'ACTIVE_TRAIN':m['train'],'DEV_ALL':m['dev'],
            'EVAL_TRAIN':[byid[c] for c in selected if byid[c]['group']=='TRAIN'],
            'EVAL_DEV':[byid[c] for c in selected if byid[c]['group']=='DEV']}
    part_rows=[]
    for name,rr in groups.items():
        for part in sorted({r['part_type'] for r in rr}):
            pr=[r for r in rr if r['part_type']==part]; ids={r['condition_id'] for r in pr}
            s=tally(pr)
            part_rows.append(dict(pool=name,part_type=part,conditions=s['conditions'],parents=s['parents'],
                K_median=s['K']['median'],area_fraction_median=s['area_fraction']['median'],multicomponent=s['multicomponent_conditions'],
                closed=s['closed_conditions'],owners=s['source_owners'],aliases=s['source_aliases'],
                A_exposure=sum(counts['A'][c] for c in ids),COMMON_unique_exposure=sum(counts['COMMON_UNIQUE'][c] for c in ids)))
    write_csv(HERE/'per_part.csv',part_rows)
    parent_rows=[]
    for p in sorted({r['parent_id'] for r in rows}):
        pr=[r for r in per if r['parent_id']==p]; act=[r for r in pr if r['active_train']]
        parent_rows.append(dict(parent_id=p,group=pr[0]['group'],original_conditions=len(pr),active_conditions=len(act),
            original_parts=';'.join(sorted({r['part_type'] for r in pr})),active_parts=';'.join(sorted({r['part_type'] for r in act})),
            A_exposure=sum(r['A_exposure'] for r in pr),COMMON_unique_exposure=sum(r['COMMON_UNIQUE_exposure'] for r in pr)))
    write_csv(HERE/'per_parent.csv',parent_rows)
    # Marginal geometry support. Bbox descriptors cannot certify semantic support.
    dev_rows=[]
    active_rows=[r for r in per if r['active_train']]
    dims=['K','area_fraction','source_edge_components','span_x','span_y','span_z','center_y','span_aspect_max_second']
    for r in per:
        if r['group']!='DEV':continue
        same=[v for v in active_rows if v['part_type']==r['part_type']]
        d={k:r[k] for k in ['condition_id','parent_id','part_type','extent_description','K','area_fraction','source_edge_components','evaluation_requests']}
        d.update(train_same_part_conditions=len(same),train_same_part_parents=len({v['parent_id'] for v in same}),train_same_part_exposures=sum(v['A_exposure'] for v in same),
                 training_closed_conditions=sum(v['boundary_edges']==0 for v in same),DEV_closed_C=r['boundary_edges']==0)
        outside=[]
        for key in dims:
            d[key]=r[key]
            vals=[float(v[key]) for v in same]
            d[key+'_train_min']=min(vals) if vals else None;d[key+'_train_max']=max(vals) if vals else None
            if vals and (float(r[key])<min(vals) or float(r[key])>max(vals)): outside.append(key)
        d['outside_same_part_unaugmented_marginal_ranges']=';'.join(outside)
        d['qualification']='MARGINAL_DESCRIPTOR_SCREEN_ONLY; xyz augmentation may cover small span differences; not proof of semantic OOD'
        dev_rows.append(d)
    write_csv(HERE/'dev_coverage.csv',dev_rows)
    summary={'schema':'BENCH_DATA_COVERAGE_EXISTING_MANIFEST_LOG_AUDIT_V1','checks':checks,
        'pools':{k:tally(v) for k,v in groups.items()},
        'exclusions':dict(collections.Counter(excluded.values())),
        'exclusions_by_part':{p:dict(collections.Counter(excluded.get(r['condition_id'],'RETAINED') for r in m['train_original'] if r['part_type']==p)) for p in sorted({r['part_type'] for r in m['train_original']})},
        'actual_logs':{k:{'logged_samples':len(v),'unique_conditions':len(counts[k]),'parents':len({r['parent_id'] for r in v}),'part_exposures':dict(collections.Counter(r['part_type'] for r in v))} for k,v in logs.items()},
        'COMMON_unique_scheduled_samples':len(common_unique),'COMMON_logged_repeated_sample_units':len(logs['COMMON'])-len(common_unique),
        'A_per_condition_exposure':number_stats(counts['A'][c] for c in active),
        'A_per_parent_exposure':number_stats(r['A_exposure'] for r in parent_rows if r['active_conditions']),
        'A_multicomponent_exposure':sum(r['A_exposure'] for r in per if r['source_edge_components']>1),
        'A_nonplanar_exposure':sum(r['A_exposure'] for r in per if r['region_normal_not_single_plane']),
        'excluded_COMMON_unique_exposures':sum(counts['COMMON_UNIQUE'][c] for c in excluded),
        'semantic_id_mismatches':[r['condition_id'] for r in per if r['id_vs_actual_part_mismatch']],
        'source_labels_inference':False,'new_model_calls':0,'new_training_updates':0,'new_geometry_scores':0,
        'limitations':['Source metadata, source semantic review and executed exposure logs only; no new rendering or semantic review.',
            'part_type is coarse. Complete solid leg, leg side, end frame, slat and broad seat are not distinguished by one model input label.',
            'Area fraction denominator includes source duplicate faces.',
            'Descriptor support screens are not topology-aware or semantic OOD certification.',
            'Counts of shape parents do not audit near-duplicate design families.'],
        'seconds':time.time()-started}
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    SOURCES[str(Path(__file__).relative_to(ROOT))]={'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'bytes':Path(__file__).stat().st_size}
    (HERE/'source_hashes.json').write_text(json.dumps(SOURCES,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['checks','exclusions','A_per_condition_exposure','A_per_parent_exposure','A_multicomponent_exposure','excluded_COMMON_unique_exposures','seconds']},indent=2))

if __name__=='__main__':main()
