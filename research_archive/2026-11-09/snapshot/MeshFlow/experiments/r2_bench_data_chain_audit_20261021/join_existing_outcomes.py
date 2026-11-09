"""Descriptive associations only; reuses frozen scores and AI_PROXY labels."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, statistics, time

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_bench_data_chain_audit_20261021'

def readcsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def dump(name,value):
    p=OUT/name
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)

def writecsv(name,rows):
    with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    start=time.monotonic()
    inputs=[OUT/'geometry/per_condition.csv',OUT/'geometry/per_parent.csv',
        ROOT/'experiments/r2_bench_state_schedule_v1/results/per_output.csv',
        ROOT/'experiments/r2_bench_state_schedule_v1/visual_review/per_cell.csv',
        ROOT/'experiments/r2_bench_clean_multipart_seam_v1/data/schedule.json',
        ROOT/'experiments/r2_bench_clean_multipart_seam_v1/data/manifest.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    geom={r['condition_id']:r for r in readcsv(inputs[0])}
    parents={r['parent_id']:r for r in readcsv(inputs[1])}
    metrics={(r['request_id'],r['label'],r['stage']):r for r in readcsv(inputs[2])}
    reviews=readcsv(inputs[3])
    manifest=json.loads(inputs[5].read_text())
    metadata={r['condition_id']:r for r in manifest['rows']}
    rows=[]
    for v in reviews:
        g=geom[v['condition_id']];p=parents[v['parent_id']];c=metadata[v['condition_id']]
        m=metrics[(v['request_id'],v['model'],v['stage'])]
        row={k:v[k] for k in ['request_id','condition_id','parent_id','split','model','stage','overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']}
        row.update(part_type=c['part_type'],extent_description=c['extent_description'],N=int(g['N']),K=int(g['K']),
            active_train_condition=g['active_current_pool']=='True' and v['split']=='TRAIN',
            closed_C=c['boundary_edges']==0,multicomponent_C=int(g['C_source_components'])>1,
            source_M_duplicate_excess=int(p['source_ID_excess_faces']),source_M_zero_area=int(p['exact_zero_area_faces_model64']),
            source_Y_duplicate_excess=int(g['Y_source_ID_excess_faces']),source_Y_zero_area=int(g['Y_exact_zero_area_model64_faces']),
            source_CY_exact_overlap_pairs=int(g['C_Y_exact_geometry_overlap_pairs']),
            source_CY_exact_overlap_C_area_fraction=float(g['C_exactly_duplicated_in_Y_area_fraction']))
        for k in ['FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction','ConflictCov_C','actual_free_faces','deleted_free_faces','free_only_reached_bulk_area_fraction','legal_seam_bulk_edge_count']:
            try:row[k]=float(m[k])
            except (ValueError,KeyError):row[k]=None
        row['label_provenance']='EXISTING_NONBLIND_AI_PROXY'
        rows.append(row)
    assert len(rows)==432 and len({(r['request_id'],r['model'],r['stage']) for r in rows})==432
    writecsv('existing_outcomes_join.csv',rows)
    schedule=json.loads(inputs[4].read_text())
    samples=[s for step in schedule['steps'] for s in step['samples']]
    ex=Counter(s['condition_id'] for s in samples)
    expo=[]
    for cid,n in sorted(ex.items()):
        g=geom[cid];p=parents[g['parent_id']]
        expo.append(dict(condition_id=cid,parent_id=g['parent_id'],part_type=g['part_type'],exposures=n,
            C_Y_overlap_pairs=int(g['C_Y_exact_geometry_overlap_pairs']),
            Y_duplicate_excess=int(g['Y_source_ID_excess_faces']),Y_zero_area=int(g['Y_exact_zero_area_model64_faces']),
            M_duplicate_excess=int(p['source_ID_excess_faces'])))
    writecsv('quality_exposure.csv',expo)
    exposure_summary={'instances':sum(ex.values()),'conditions':len(ex),
        'C_Y_exact_overlap_instances':sum(r['exposures'] for r in expo if r['C_Y_overlap_pairs']>0),
        'Y_duplicate_instances':sum(r['exposures'] for r in expo if r['Y_duplicate_excess']>0),
        'Y_zero_area_instances':sum(r['exposures'] for r in expo if r['Y_zero_area']>0)}
    summary=[]
    features={'M_duplicate':lambda r:r['source_M_duplicate_excess']>0,
        'M_duplicate_or_zeroarea':lambda r:r['source_M_duplicate_excess']>0 or r['source_M_zero_area']>0,
        'CY_exact_overlap':lambda r:r['source_CY_exact_overlap_pairs']>0,
        'C_closed':lambda r:r['closed_C'],'C_multicomponent':lambda r:r['multicomponent_C']}
    for model in sorted({r['model'] for r in rows}):
        for split in ['TRAIN','DEV']:
            group=[r for r in rows if r['model']==model and r['split']==split and r['stage']=='CLEAN_FINAL_POST']
            for name,fun in features.items():
                for present in [False,True]:
                    sub=[r for r in group if fun(r)==present]
                    summary.append(dict(model=model,split=split,feature=name,present=present,outputs=len(sub),
                        conditions=len({r['condition_id'] for r in sub}),parents=len({r['parent_id'] for r in sub}),
                        plausible_C=sum(r['C_use']=='PLAUSIBLE' for r in sub),
                        replaced_C=sum(r['C_use']=='REPLACED_OR_IGNORED' for r in sub),
                        uncertain_C=sum(r['C_use']=='UNCERTAIN' for r in sub),
                        reasonable_shape=sum(r['overall_shape']=='REASONABLE' for r in sub),
                        layering_yes=sum(r['layering_or_fold']=='YES' for r in sub),
                        FF_mean=statistics.mean(r['FF_illegal_affected_free_fraction'] for r in sub) if sub else None,
                        I_mean=statistics.mean(r['ConflictCov_C'] for r in sub) if sub else None))
    writecsv('descriptive_associations.csv',summary)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in inputs)
    dump('join_receipt.json',dict(status='COMPLETE',joined_existing_rows=len(rows),exposure=exposure_summary,
        source_hashes=hashes,source_unchanged=True,wall_seconds=time.monotonic()-start,
        new_scores=0,new_generation=0,model_calls=0,GPU_calls=0,
        interpretation='Exploratory associations on repeatedly used 12-parent development panel. Outputs within parent are not independent. Source defects do not establish causality; source shape matching is not success.'))
    print(json.dumps(exposure_summary))
    print(json.dumps([r for r in summary if r['model']=='A_S1500' and r['split']=='DEV'],ensure_ascii=False))

if __name__=='__main__':main()
