"""Count the predeclared three-axis intersection of actual observations; no new ratings."""
from pathlib import Path
import collections, csv, hashlib, json, statistics

OUT = Path(__file__).resolve().parent
SRC = OUT/'visual_review/summary_v1/actual_observations.csv'
DEST = OUT/'visual_review/intersection_v1'

def write(name, rows):
    with (DEST/name).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    if (DEST/'completion.json').exists():
        done=json.loads((DEST/'completion.json').read_text('utf-8'))
        for name,digest in done['files'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
        print('COMPLETE: saved intersection verified'); return
    with SRC.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==270
    for r in rows:
        r['intersection'] = (r['overall_shape']=='REASONABLE' and r['C_use']=='PLAUSIBLE' and r['support_retention']=='PRESENT')
    groups=('OLD','FOCUS','DEV','HOLDOUT'); stages=('RAW','BASE_POST','CLEAN_FINAL_POST'); labels=('P0','A','B')
    summary=[]; per=[]; paired=[]; transitions=[]
    by={(r['label'],r['stage'],r['request_id']):r for r in rows}
    for g in groups:
        for s in stages:
            for l in labels:
                rr=[r for r in rows if (r['report_group'],r['stage'],r['label'])==(g,s,l)]
                parents=collections.defaultdict(list)
                for r in rr:
                    parents[r['parent_id']].append(int(r['intersection']))
                    per.append({k:r[k] for k in ('label','stage','report_group','parent_id','request_id','overall_shape','C_use','support_retention','intersection')})
                summary.append(dict(report_group=g,stage=s,label=l,yes=sum(r['intersection'] for r in rr),requests=len(rr),parents=len(parents),parent_equal_fraction=statistics.fmean(statistics.fmean(v) for v in parents.values())))
            for ref,cand in (('P0','A'),('P0','B'),('A','B')):
                rr=[r for r in rows if (r['report_group'],r['stage'],r['label'])==(g,s,cand)]
                counts=collections.Counter()
                for r in rr:
                    old=by[ref,s,r['request_id']]; key=('YES' if old['intersection'] else 'NO')+'_TO_'+('YES' if r['intersection'] else 'NO')
                    counts[key]+=1
                    transitions.append(dict(reference=ref,candidate=cand,report_group=g,stage=s,parent_id=r['parent_id'],request_id=r['request_id'],reference_intersection=old['intersection'],candidate_intersection=r['intersection'],transition=key))
                paired.append(dict(reference=ref,candidate=cand,report_group=g,stage=s,requests=len(rr),kept=counts['YES_TO_YES'],gained=counts['NO_TO_YES'],lost=counts['YES_TO_NO'],neither=counts['NO_TO_NO']))
    DEST.mkdir(parents=True,exist_ok=True)
    for name,rr in (('per_output.csv',per),('parent_macro_counts.csv',summary),('paired_counts.csv',paired),('paired_transitions.csv',transitions)):write(name,rr)
    drift=OUT/'visual_review/summary_v1/historical_P0_label_drift.csv'
    with drift.open(encoding='utf-8-sig',newline='') as f: dd=list(csv.DictReader(f))
    changed=[r for r in dd if r['changed']=='True']
    result=dict(status='COMPLETE',source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),definition='overall_shape == REASONABLE AND C_use == PLAUSIBLE AND support_retention == PRESENT',
        reviewer_kind='AI_PROXY',blind=False,new_ratings=0,model_calls=0,scoring_calls=0,
        caution='This descriptive intersection is not a clean/watertight/contact certification; gaps and layering remain separate axes.',
        historical_P0_drift=dict(axis_labels=len(dd),changed=len(changed),by_axis=dict(collections.Counter(r['axis'] for r in changed))),
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.iterdir() if p.is_file()})
    (DEST/'completion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'counts':[r for r in summary if r['stage']!='BASE_POST'],'paired_final':[r for r in paired if r['stage']=='CLEAN_FINAL_POST'],'drift':result['historical_P0_drift']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
