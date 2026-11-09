"""Recover CSV/summary export from completed corner geometry; never rerun it."""
import csv, hashlib, json, time
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_pool47_couple_v1/corner_rule'
DEST=OUT/'export_recovery'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
def write(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def csvwrite(p,rr,fields):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def run():
    r=read(OUT/'registration.json');geo=read(OUT/'geometry_complete.json');assert geo['conditions']==50
    for p,h in {**r['protected'],**r['code']}.items():assert sha(p)==h,p
    assert (OUT/'failure.json').exists() and not DEST.exists()
    DEST.mkdir()
    source_hashes={str(p):sha(p) for p in OUT.rglob('*') if p.is_file()}
    rr=csvread(OUT/'per_condition.csv');pp=csvread(OUT/'proposal_diagnostic.csv');assert len(rr)==50
    fields=list(pp[0]);false=[x for x in pp if x['selected']=='True' and x['exact_source_alias_face_match']=='False']
    truth=[x for x in pp if x['exact_source_alias_face_match']=='True'];miss=[x for x in truth if x['selected']!='True']
    assert len({(x['parent_id'],x['source_alias_face_id']) for x in truth})==sum(int(x['source_alias_faces']) for x in rr)
    csvwrite(DEST/'false_proposals.csv',false,fields);csvwrite(DEST/'missed_source_alias_faces.csv',miss,fields)
    csvwrite(DEST/'source_alias_candidate_fates.csv',truth,fields)
    totals=[]
    for split in ('TRAIN','DEV'):
        group=[x for x in rr if x['split']==split];gg=[x for x in geo['results'] if x['split']==split]
        s={k:sum(int(x[k]) for x in group) for k in ('source_alias_faces','geometric_feasible','selected_proposals','matched_alias_faces','true_alias_edges','predicted_alias_edges','correct_alias_edges','false_proposals','missed_alias_faces')}
        for name,num,den in [('proposal_precision','matched_alias_faces','selected_proposals'),('alias_face_recall','matched_alias_faces','source_alias_faces'),('alias_edge_precision','correct_alias_edges','predicted_alias_edges'),('alias_edge_recall','correct_alias_edges','true_alias_edges')]:s[name]=s[num]/s[den] if s[den] else None
        s.update(split=split,conditions=len(group),unique_candidates=sum(x['unique_candidates'] for x in gg),generated_paths=sum(x['generated_paths'] for x in gg),
            ambiguous_corners=sum(x.get('ambiguous_corners',0) for x in gg),blocked_input_conditions=sum(x['status']!='COMPLETE' for x in gg),
            source_alias_faces_enumerated=len([x for x in truth if x['split']==split]))
        totals.append(s)
    summary=dict(groups=totals,rejections=dict(Counter(x['rejection'] for x in pp if x['rejection'])),
        source_alias_rejections=dict(Counter(x['rejection'] for x in miss)),
        source_alias_min_angle_range=[min(float(x['min_angle']) for x in truth),max(float(x['min_angle']) for x in truth)],
        interpretation='All9 source alias faces enumerated, all rejected by frozen5degree shape gate; selected geometric proposals are not source alias classifier success. No thresholds changed.')
    write(DEST/'diagnostic_summary.json',summary)
    for p,h in source_hashes.items():assert sha(p)==h,p
    completion=dict(status='COMPLETE_WITH_EXPORT_ONLY_RECOVERY',unix=time.time(),conditions=50,groups=totals,
        original_failed_attempt='failure.json',original_geometry='geometry_complete.json',original_source_and_results_untouched=True,
        geometry_reruns=0,extra_source_scoring=0,model_calls=0,GPU_calls=0,protected_hashes_verified=len(r['protected']),code_hashes_verified=len(r['code']),
        recovery_script_sha256=sha(__file__),preserved_input_hashes=source_hashes,
        recovered_files={str(p):sha(p) for p in DEST.iterdir() if p.is_file()})
    write(DEST/'completion.json',completion)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':run()
