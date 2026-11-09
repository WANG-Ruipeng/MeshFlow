"""Independent stdlib recomputation from stored per-output rows; no scoring."""
from pathlib import Path
import collections, csv, hashlib, json, math, statistics

OUT=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def finite(x): return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)
def mean(xs): return statistics.fmean(xs) if xs else None
def agree(a,b):
    if b=='': assert a is None
    else: assert a is not None and math.isclose(a,float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)
def macro(rows,m):
    # This panel has exactly one condition and two noises per parent.
    by=collections.defaultdict(list)
    for r in rows:
        if finite(r.get(m)): by[r['parent_id']].append(r[m])
    return mean([mean(v) for v in by.values()]),len(by),sum(map(len,by.values()))
def main():
    source=OUT/'results/metric_rows.json'; rows=read(source)
    fields=read(OUT/'results/completion.json')['metric_fields']
    assert len(rows)==270 and len(fields)==177
    by={(r['label'],r['stage'],r['request_id']):r for r in rows}
    assert len(by)==270
    groups=collections.defaultdict(list)
    for r in rows: groups[r['label'],r['stage'],r['report_group']].append(r)
    totals=collections.Counter()
    for r in csvread(OUT/'results/analysis_v1/full177_parent_macro.csv'):
        val,npar,nreq=macro(groups[r['label'],r['stage'],r['report_group']],r['metric'])
        agree(val,r['mean']); assert npar==int(r['valid_parents']) and nreq==int(r['valid_requests'])
        totals['macro_cells']+=1
    for r in csvread(OUT/'results/analysis_v1/paired177_parent_macro.csv'):
        m=r['metric']; delta=[]
        for a in groups[r['reference'],r['stage'],r['report_group']]:
            b=by[r['candidate'],r['stage'],a['request_id']]
            if finite(a[m]) and finite(b[m]): delta.append(dict(parent_id=a['parent_id'],value=b[m]-a[m]))
        val,npar,nreq=macro(delta,'value'); agree(val,r['mean_delta'])
        assert npar==int(r['valid_parents']) and nreq==int(r['valid_requests'])
        totals['paired_macro_cells']+=1
    for r in csvread(OUT/'results/paired_per_output.csv'):
        a=by[r['reference'],r['stage'],r['request_id']];b=by[r['candidate'],r['stage'],r['request_id']]
        for m in fields:
            agree(b[m]-a[m] if finite(a[m]) and finite(b[m]) else None,r[m+'__delta'])
            totals['per_output_delta_cells']+=1
    for r in csvread(OUT/'results/analysis_v1/deletion_area_per_output.csv'):
        assert int(r['retained_original_free_faces'])+int(r['deleted_original_free_faces'])==int(r['raw_free_faces'])
        a,b,c=(float(r[k]) for k in ('retained_raw_reference_area','deleted_raw_reference_area','raw_free_area'))
        assert math.isclose(a+b,c,rel_tol=1e-6,abs_tol=1e-5)
        totals['deletion_balances']+=1
    result=dict(status='PASS',checks=dict(totals),all_I_status_counts=dict(collections.Counter(r['I_status'] for r in rows)),
        C_preserved_exact_counts=dict(collections.Counter(str(r['C_preserved_exact']) for r in rows)),
        model_calls=0,scoring_calls=0,method='Independent Python stdlib from frozen per-output evidence; parent-equal means, common-support deltas, deletion balances.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target=OUT/'audits/independent_numeric_readback.json'
    if target.exists(): assert read(target)==result
    else: target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
