"""Read existing risk audit only. Does not change datasets or delete files."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
EXP=BASE/'experiments/r2_bench_coverage_expansion_v1'
AUD=BASE/'experiments/r2_bench_data_chain_audit_20261021/geometry'

def j(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rs):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)

def main():
    paths=[EXP/'data/manifest.json',EXP/'data/excluded_new.json',EXP/'data/condition_exposure.csv',
           AUD/'per_condition.csv',AUD/'per_parent.csv',AUD/'C_Y_exact_overlap_pairs.csv',AUD/'summary.json',
           BASE/'experiments/r2_bench_parts_v1/data/excluded.csv']
    manifest=j(paths[0]); train=manifest['train']; trainid={r['condition_id']:r for r in train}
    assert len(train)==321
    audit={r['condition_id']:r for r in csvrows(paths[3])}
    exposure={r['condition_id']:r for r in csvrows(paths[2])}
    parent_audit={r['parent_id']:r for r in csvrows(paths[4])}
    conditions=[]
    for r in train:
        cid=r['condition_id']; a=audit.get(cid)
        flags=[]
        if a and int(a['C_Y_exact_geometry_overlap_pairs']):flags.append('EXACT_WHOLE_C_Y_OVERLAP_REBUILD_TARGET')
        if a and int(a['Y_exact_zero_area_model64_faces']):flags.append('EXACT_ZERO_AREA_VALID_Y_REBUILD_TARGET')
        if a and int(a['Y_source_ID_excess_faces']):flags.append('Y_ID_DUPLICATE_REVIEW_NOT_PARENT_DELETION')
        if not a:flags.append('FULL_C_Y_AND_Y_DEGENERACY_AUDIT_NA')
        conditions.append(dict(condition_id=cid,parent_id=r['parent_id'],part_type=r['part_type'],
            origin=exposure[cid]['origin'],actual_completed_exposures=int(exposure[cid]['planned_exposures']),
            case_path=r['case_path'],case_sha256=r['case_sha256'],geometry_status=r.get('geometry_status'),
            semantic_decision=r.get('review',{}).get('decision'),
            audit_status='AUDITED_EXACT_GEOMETRY' if a else 'NA_NEW_PARENT_NO_MATCHING_EXACT_GEOMETRY_AUDIT',
            C_Y_exact_geometry_overlap_pairs=int(a['C_Y_exact_geometry_overlap_pairs']) if a else None,
            C_Y_ID_overlap_opposite_winding_pairs=int(a['C_Y_ID_overlap_opposite_winding_pairs']) if a else None,
            Y_exact_zero_area_model64_faces=int(a['Y_exact_zero_area_model64_faces']) if a else None,
            Y_source_ID_excess_faces=int(a['Y_source_ID_excess_faces']) if a else None,
            Y_source_ID_opposite_winding_pairs=int(a['Y_source_ID_opposite_winding_pairs']) if a else None,
            source_duplicate_excess_metadata=r.get('source_target_risks',{}).get('source_exact_duplicate_ID_faces'),
            proposed_action='REBUILD_IN_NEW_VERSION_BEFORE_REUSE' if any('REBUILD' in x for x in flags) else 'REVIEW_OR_KEEP_NOT_DELETE',
            risk_flags=';'.join(flags)))
    covered=[r for r in conditions if r['audit_status']=='AUDITED_EXACT_GEOMETRY']
    unknown=[r for r in conditions if r['audit_status']!='AUDITED_EXACT_GEOMETRY']
    assert len(covered)==277 and len(unknown)==44
    def summarize(rs):
        out=dict(conditions=len(rs),parents=len({r['parent_id'] for r in rs}),exposures=sum(r['actual_completed_exposures'] for r in rs))
        for key in ['C_Y_exact_geometry_overlap_pairs','Y_exact_zero_area_model64_faces','Y_source_ID_excess_faces']:
            present=[r for r in rs if r[key] is not None and r[key]>0]
            out[key]=dict(affected_conditions=len(present),affected_parents=len({r['parent_id'] for r in present}),condition_repeated_sum=sum(r[key] for r in present),exposures=sum(r['actual_completed_exposures'] for r in present))
        union=[r for r in rs if r['proposed_action']=='REBUILD_IN_NEW_VERSION_BEFORE_REUSE']
        out['overlap_or_zeroY_union']=dict(conditions=len(union),parents=len({r['parent_id'] for r in union}),exposures=sum(r['actual_completed_exposures'] for r in union))
        return out
    old_excluded=csvrows(paths[7]);new_excluded=j(paths[1])['rows']
    excluded=[]
    for source,rs in [('ORIGINAL_PARTS',old_excluded),('EXPANSION_NEW',new_excluded)]:
        for r in rs:
            cid=r['condition_id'];assert cid not in trainid
            reasons=r.get('exclusion_reasons',[])
            if isinstance(reasons,str):reasons=json.loads(reasons)
            excluded.append(dict(condition_id=cid,parent_id=r['parent_id'],group=r['group'],origin=source,
                exclusion_reasons=reasons,geometry_status=r.get('geometry_status'),in_current_training=False,
                proposed_action='KEEP_QUARANTINED_EVIDENCE_NO_PHYSICAL_DELETE'))
    pairs=[r for r in csvrows(paths[5]) if r['condition_id'] in trainid]
    oldparents={r['parent_id'] for r in covered}
    p=[parent_audit[k] for k in oldparents]
    summary=dict(schema='BENCH_DATA_RISK_PROPOSAL_READ_ONLY_V1',status='PROPOSAL_ONLY_NO_RETIREMENT_EXECUTED',
        current_train_conditions=321,current_train_parents=79,exact_audit_covered=summarize(covered),
        exact_audit_NA=dict(conditions=len(unknown),parents=len({r['parent_id'] for r in unknown}),condition_ids=[r['condition_id'] for r in unknown]),
        old64_parent_geometry=dict(parents=64,parents_with_exact_zero_area=sum(int(r['exact_zero_area_faces_model64'])>0 for r in p),
            exact_zero_area_faces=sum(int(r['exact_zero_area_faces_model64']) for r in p),
            parents_with_ID_duplicates=sum(int(r['source_ID_excess_faces'])>0 for r in p),
            ID_duplicate_excess=sum(int(r['source_ID_excess_faces']) for r in p),
            ID_duplicate_opposite_winding_pairs=sum(int(r['source_ID_opposite_winding_pairs']) for r in p)),
        by_origin={o:summarize([r for r in covered if r['origin']==o]) for o in sorted({r['origin'] for r in covered})},
        current_C_geometry_status=dict(Counter(r['geometry_status'] for r in conditions)),
        current_semantic_decision=dict(Counter(r['semantic_decision'] for r in conditions)),
        already_excluded_summary={source:dict(conditions=sum(r['origin']==source for r in excluded),
            training_conditions=sum(r['origin']==source and r['group']=='TRAIN' for r in excluded),
            dev_conditions=sum(r['origin']==source and r['group']=='DEV' for r in excluded),
            reason_counts=dict(Counter(reason for r in excluded if r['origin']==source for reason in r['exclusion_reasons']))) for source in ('ORIGINAL_PARTS','EXPANSION_NEW')},
        already_excluded=excluded,conditions=conditions,
        action_policy=[
            'Do not filter training using model-generated good/bad cases or DEV outcome.',
            'Current frozen 321-condition manifest and all historical evidence stay unchanged. Candidate decisions apply only to a separately versioned future pool.',
            'Already excluded C geometry failures/semantic uncertainty remain quarantined; no need to physically delete them or reject other conditions from the same parent.',
            'For exact C/Y whole-face overlap, inspect complementary target partition and intended representation; rebuild Y/face identities/owner labels/masks/N together rather than deleting original C or source M.',
            'Exact zero-area faces in valid Y are a concrete degenerate-target issue; omit/repair only in a new version with N/OT/owner/identity checks. A whole parent need not be discarded.',
            'Opposite-winding duplicate Y faces are not by themselves proof of invalid author intent; keep separate from definite zero-area risk. Do not delete all affected parents.',
            'New 15 parents/44 C have local C/source-boundary checks and some source duplicate metadata, but matching whole C/Y exact overlap and zero-Y audit is NA; do not mark them clean.',
            'No original files deleted. Physical cleanup requires a separately authorized artifact inventory; data-risk flags do not imply large disk savings.'],
        limitations=['No new geometry score, model forward, training or mesh repair. Existing CSV/JSON joins only.',
            'Exact overlap means entire coincident triangles; not partial coplanar overlap or all finite intersections.',
            'Condition-summed defect counts repeat shared parent faces; independent parent counts and parent-level totals are separate.',
            'No full Y-Y selfintersection audit exists for current79. Zero/NA must not be conflated.'],
        source_hashes={str(f.relative_to(BASE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        ledger=dict(new_geometry_score_calls=0,model_calls=0,GPU_calls=0,training_updates=0,source_modifications=0,deleted_files=0))
    (OUT/'risk_proposal.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    writecsv(OUT/'risk_conditions.csv',conditions)
    writecsv(OUT/'known_C_Y_overlap_pairs.csv',pairs)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['conditions','already_excluded','source_hashes','action_policy','limitations']},ensure_ascii=False))

if __name__=='__main__':main()
