"""High-face source supply gate, before any topology-changing simplification."""
from .supply import *
from pathlib import Path

def inventory_high():
    dest=OUT/'high_source';receipt=dest/'manifest.json'
    if receipt.exists():return b.read(receipt)
    supply=b.read(OUT/'supply/manifest.json')
    rows=[r for r in supply['held'] if r['reason']=='VALID_UNIQUE_N_OUTSIDE_CURRENT_NO_SIMPLIFICATION' and r['N_unique']>256]
    accepted=[];held=[]
    with tarfile.open(ARCHIVE) as tar:
        for row in rows:
            uid=row['parent_id'];data=tar.extractfile(row['source_archive_member']).read()
            assert hashlib.sha256(data).hexdigest()==row['source_archive_member_sha256']
            with np.load(io.BytesIO(data),allow_pickle=False) as z:a=preprocess({k:z[k].copy() for k in z.files})
            f=a['source_faces'];v=a['model_vertices'];good=(b.area2(v,f)>0)&(b.area2(a['source_vertices'],f)>0)
            groups={}
            for i in np.flatnonzero(good):groups.setdefault(tuple(sorted(tuple(map(float,x)) for x in v[f[i]])),[]).append(int(i))
            lineage=list(groups.values());clean=np.asarray([f[g[0]] for g in lineage],np.int64)
            oriented,orient=b.orient(clean)
            if orient['nonmanifold_edges'] or orient['parity_conflicts']:
                held.append(dict(parent_id=uid,reason='HIGH_SOURCE_NONMANIFOLD_OR_NONORIENTABLE',N=len(clean),nonmanifold=len(orient['nonmanifold_edges'])));continue
            d=dest/'sources'/uid;d.mkdir(parents=True,exist_ok=True)
            b.npz(d/'original.npz',a);ca=b.canonical(a,oriented);b.npz(d/'clean.npz',ca)
            b.save(d/'source_lineage.json',dict(new_to_old_face_ids=lineage,orientation=orient,removed_zero=np.flatnonzero(~good).tolist()))
            accepted.append(dict(row,source_case_path=str(d/'original.npz'),source_case_sha256=b.sha(d/'original.npz'),clean_case_path=str(d/'clean.npz'),clean_case_sha256=b.sha(d/'clean.npz'),N_clean=len(clean),source_lineage_path=str(d/'source_lineage.json')))
    m=dict(rows=accepted,held=held,input_count=len(rows),resources=b.resources(),policy='Only manifold orientable cleaned high-N source proposals proceed; source and final finite-triangle checks remain mandatory. No automatic semantic acceptance.')
    b.save(receipt,m);event('HIGH_SOURCE_TOPOLOGY_GATE',input=len(rows),remaining=len(accepted),held=len(held))
    print(json.dumps(dict(input=len(rows),remaining=len(accepted),held=len(held))))
    return m

def audit_one(row):
    dest=Path(row['clean_case_path']).parent;receipt=dest/'finite_geometry.json'
    if receipt.exists():q=b.read(receipt)
    else:
        assert b.sha(row['clean_case_path'])==row['clean_case_sha256']
        q=b.full_geometry(b.load(row['clean_case_path']));b.save(receipt,q)
    ok=not q['illegal_pairs'] and not q['unresolved_pairs'] and not q['topology']['same_winding_double_edges']
    print(json.dumps(dict(parent=row['parent_id'][:6],high_source_finite_pass=ok,illegal=len(q['illegal_pairs']),unknown=len(q['unresolved_pairs']))),flush=True)
    return dict(row,source_finite_pass=ok,source_geometry_path=str(receipt),illegal_pairs=len(q['illegal_pairs']),unresolved_pairs=len(q['unresolved_pairs']))

def audit():
    m=inventory_high();event('HIGH_SOURCE_FINITE_START',workers=4,resources=b.resources());start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,initializer=configure) as pool:rows=list(pool.map(audit_one,m['rows']))
    b.save(OUT/'high_source/audited.json',dict(rows=rows))
    event('HIGH_SOURCE_FINITE_COMPLETE',passed=sum(r['source_finite_pass'] for r in rows),total=len(rows),wall_seconds=time.monotonic()-start)
    print(json.dumps(dict(passed=sum(r['source_finite_pass'] for r in rows),total=len(rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','audit'],default='inventory',nargs='?');args=p.parse_args();configure()
    audit() if args.stage=='audit' else inventory_high()
