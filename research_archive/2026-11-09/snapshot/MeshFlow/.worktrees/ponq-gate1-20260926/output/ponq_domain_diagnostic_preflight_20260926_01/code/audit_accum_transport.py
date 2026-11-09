"""Exact output auditing with reused, hash-bound same-position references."""
from common import *
from exact_audit import selftest,exact_mesh
selftest()
count=0
for p in sorted((ROOT/'runs').glob('*/mesh.obj')):
    v,f=load_mesh(p);r=exact_mesh(v,f)
    r.update(source_sha256=sha(p),method='integer dyadic orientation + exact rational intersection; no epsilon predicates; legal shared indexed simplices excluded')
    dump(p.parent/'exact_audit.json',r);count+=1
    print('EXACT_END',p.parent.name,r['exact_intersection_pairs'],r['exact_degenerate_faces'],flush=True)
refs=0
for p in sorted((ROOT/'inputs').glob('*/reference.obj')):
    r=json.loads((p.parent/'reference_exact_audit.json').read_text());assert r['source_sha256']==sha(p);refs+=1
print('OUTPUTS_AUDITED',count,'REFERENCE_AUDITS_REUSED',refs,flush=True)
