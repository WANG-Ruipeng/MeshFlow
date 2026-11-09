"""Audit new outputs exactly; preserve copied hash-bound reference audits."""
from common import *
from exact_audit import selftest,exact_mesh
selftest()
for p in sorted((ROOT/'runs').glob('*/mesh.obj')):
    v,f=load_mesh(p);r=exact_mesh(v,f)
    r.update(source_sha256=sha(p),method='integer dyadic orientation + exact rational intersection; no epsilon predicates; legal shared indexed simplices excluded')
    dump(p.parent/'exact_audit.json',r)
    print('EXACT_END',p.parent.name,r['exact_intersection_pairs'],r['exact_degenerate_faces'],flush=True)
for p in sorted((ROOT/'inputs').glob('*/reference.obj')):
    r=json.loads((p.parent/'reference_exact_audit.json').read_text());assert r['source_sha256']==sha(p)
print('48_NEW_OUTPUTS_AUDITED_24_REFERENCE_AUDITS_REUSED',flush=True)