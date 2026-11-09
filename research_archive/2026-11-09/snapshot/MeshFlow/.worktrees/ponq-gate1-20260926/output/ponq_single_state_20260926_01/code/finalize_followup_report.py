"""Attach separately performed numerical/integrity audits to the derived report."""
from pathlib import Path
import csv, hashlib, json, shutil

root=Path(__file__).resolve().parents[3]
parent=root/'output'
out=parent/'ponq_gate1_followup_20260926_01'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for folder, names in [
    ('ponq_gate1_cut_diagnostic_20260926_01',['NUMERICAL_DIAGNOSIS.json','NETWORKX_SOURCE_INSPECTION.json']),
    ('ponq_gate1_integrity_20260926_01',['VERIFICATION.json','SAMPLING_COVERAGE_DIAGNOSTIC.json'])]:
    for name in names:shutil.copy2(parent/folder/name,out/name)
rows=read(out/'LEDGER_92.json');diag=read(out/'NUMERICAL_DIAGNOSIS.json')
di={(r['study'],r['object'],r['state']):r for r in diag}
for row in rows:
    d=di.get((row['study'],row['object'],row['state']))
    row['actual_graph_independently_reconstructed']=d is not None
    row['actual_graph_nonfinite_capacity_count']=d['aggregated_graph_capacities']['nonfinite'] if d else None
    row['actual_graph_negative_capacity_count']=d['aggregated_graph_capacities']['negative'] if d else None
    if row['nonfinite_cut_weights'] or row['negative_cut_weights']:assert d is not None
assert len(rows)==92 and len(di)==7
dump(out/'LEDGER_92.json',rows)
with (out/'LEDGER_92.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
meta=read(out/'RESULT_NUMBERS.json')
meta['actual_graph_diagnostic']=dict(cases=7,nan_capacity_cases=2,negative_capacity_cases=0,
    nan_cases=[dict(study=r['study'],object=r['object'],state=r['state'],count=r['aggregated_graph_capacities']['nonfinite']) for r in diag if r['aggregated_graph_capacities']['nonfinite']],
    unaffected_case_policy='Other 85 cases had finite nonnegative candidate scores and were not separately reconstructed.')
meta['integrity_verification_sha256']=sha(out/'VERIFICATION.json')
meta['numeric_diagnostic_sha256']=sha(out/'NUMERICAL_DIAGNOSIS.json')
folders=[p for p in parent.glob('ponq_gate1_*') if p.is_dir()]
sizes={p.name:sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in folders}
meta['storage_bytes_before_final_manifest']=sizes
dump(out/'RESULT_NUMBERS.json',meta)
shutil.copy2(__file__,out/Path(__file__).name)
dump(out/'MANIFEST.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']))
manifest=read(out/'MANIFEST.json')
assert all(sha(out/r['path'])==r['sha256'] for r in manifest['files'])
print(json.dumps(dict(rows=92,diagnosed_cases=len(di),manifest_files=len(manifest['files']),storage_bytes=sizes),indent=2))
