"""Read archived real Bench TRAIN sources, build new low-face target proposals."""
from __future__ import annotations
from .run import b, ROOT, OLD, OUT, VERSION, configure, event
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv, hashlib, io, json, tarfile, time
import numpy as np
from r2.pilots.bench_control_transfer_v1.data_prepare import preprocess, geometry_hash
from r2.data_tools.bench_parts_v1 import construct as C

INVENTORY=ROOT/'experiments/r2_bench_data_chain_audit_20261021/geometry/archive_supply/per_parent.csv'
ARCHIVE=ROOT/'downloaded_data/archives/shapenet-cls.tar.gz'

def inventory():
    dest=OUT/'supply';receipt=dest/'manifest.json'
    if receipt.exists():return b.read(receipt)
    prior=b.read(OUT/'candidate_manifest.json');used={p['parent_id'] for p in prior['parents']}
    dev={r['parent_id'] for r in prior['dev']}
    records=list(csv.DictReader(INVENTORY.open(encoding='utf-8-sig')))
    groups={}
    for p in prior['parents']:
        groups[geometry_hash(b.load(p['source_case_path']))]=p['parent_id']
    rows=[];held=[]
    with tarfile.open(ARCHIVE) as tar:
        split={}
        for s in ('train','test'):
            with np.load(io.BytesIO(tar.extractfile('shapenet-02828884/split/'+s+'.npz').read()),allow_pickle=True) as z:
                split.update({str(v['uid']):s for v in z['npz_list']})
        # Protect exact geometry identity of DEV even if a source uses another UID.
        for uid in sorted(dev):
            r=next(x for x in records if x['parent_id']==uid)
            data=tar.extractfile(r['archive_member']).read()
            assert hashlib.sha256(data).hexdigest()==r['member_sha256']
            with np.load(io.BytesIO(data),allow_pickle=False) as z:
                a=preprocess({k:z[k].copy() for k in z.files})
            groups[geometry_hash(a)]=uid
        for r in sorted(records,key=lambda x:x['parent_id']):
            uid=r['parent_id']
            if uid in used:continue
            base=dict(parent_id=uid,official_split=split.get(uid),source_archive_member=r['archive_member'],source_archive_member_sha256=r['member_sha256'])
            if uid in dev or split.get(uid)!='train':held.append(dict(base,reason='DEV_OR_OFFICIAL_TEST'));continue
            data=tar.extractfile(r['archive_member']).read()
            assert hashlib.sha256(data).hexdigest()==r['member_sha256']
            with np.load(io.BytesIO(data),allow_pickle=False) as z:a=preprocess({k:z[k].copy() for k in z.files})
            gh=geometry_hash(a)
            if gh in groups:held.append(dict(base,reason='EXACT_SOURCE_DUPLICATE',representative=groups[gh]));continue
            f=a['source_faces'];v=a['model_vertices']
            good=(b.area2(v,f)>0)&(b.area2(a['source_vertices'],f)>0)
            unique={tuple(sorted(tuple(map(float,x)) for x in tri)) for tri in v[f[good]]}
            n=len(unique)
            if not n or n>256:
                held.append(dict(base,reason='VALID_UNIQUE_N_OUTSIDE_CURRENT_NO_SIMPLIFICATION',N_original=len(f),N_unique=n));continue
            groups[gh]=uid
            path=dest/'sources'/(uid+'.npz');b.npz(path,a)
            rows.append(dict(base,source_case_path=str(path),source_case_sha256=b.sha(path),
                production_rank=80+len(rows),has_reviewed_back_candidate=False,was_pending_parent=False,
                exact_source_geometry_hash=gh,N_original=len(f),N_unique=n))
    m=dict(schema=VERSION,rows=rows,held=held,resources=b.resources(),
        protected_hashes={str(p):b.sha(p) for p in (INVENTORY,ARCHIVE)},
        policy='All remaining official TRAIN sources with 1..256 nonzero unique geometric faces; refine below128 using sealed v2 surface-preserving rule; no simplification/repair of positive-area intersections.',
        near_duplicate_status='Exact geometric duplicates excluded; broader shape-family near duplicates UNKNOWN')
    b.save(receipt,m);event('ARCHIVE_SUPPLY_FROZEN',selected=len(rows),held=dict(Counter(r['reason'] for r in held)))
    print(json.dumps(dict(selected=len(rows),held=dict(Counter(r['reason'] for r in held)))))
    return m

def condition_parent(pr):
    uid=pr['parent_id'];dest=OUT/'conditions'/uid;receipt=dest/'conditions.json'
    if receipt.exists():
        r=b.read(receipt)
        for p,h in r['artifact_hashes'].items():assert b.sha(p)==h
        return r['rows']
    dest.mkdir(parents=True,exist_ok=True);a=b.load(pr['target_case_path']);aud=b.Q.SourceAuditor(a)
    found,missing=C.propose(a)
    proposals=[]
    for kind,i,ids,evidence,uncertainty in found:
        proposals.append((kind,str(i+1),ids,evidence,uncertainty))
    # Only complete proposed semantic regions are combined; no random face sets.
    seat=next((x for x in proposals if x[0]=='SEAT_SURFACE'),None)
    for other,kind in [('BACK_SURFACE','SEAT_BACK'),('LEG_SUPPORT','SEAT_SUPPORT')]:
        part=next((x for x in proposals if x[0]==other),None)
        if seat and part:
            ids=sorted(set(seat[2])|set(part[2]))
            if len(ids)<len(a['source_faces']):proposals.append((kind,'1',ids,dict(method='union_whole_proposed_parts'),['GEOMETRIC_PROPOSAL_REQUIRES_ACTUAL_REVIEW']))
    results=[]
    for kind,i,ids,evidence,uncertainty in proposals:
        cid=uid+'__'+kind+'_'+i+'__V2E';d=dest/cid;d.mkdir(exist_ok=True)
        if not 2<=len(ids)<len(a['source_faces']):continue
        pair=b.partition(a,ids);q,sup=aud.candidate(np.asarray(ids,np.int64),dict(candidate_id=cid,parent_id=uid,group='TRAIN',proposed_part_type=kind))
        case=dict(a,**pair);case.update(sup)
        case['source_third_in_C']=np.asarray([bool(ok and tr[2] in set(case['C_source_vertex_ids'].tolist())) for ok,tr in zip(case['source_supported_labels'],case['edge_source_triple'])])
        b.npz(d/'case.npz',case);b.npz(d/'paired.npz',pair);b.npz(d/'supervision_only.npz',sup)
        inp={k:case[k] for k in ('C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations','N','K','target_version_id')}
        b.npz(d/'input.npz',inp);b.obj(d/'C.obj',case['C_vertices'],case['C_faces_original'])
        b.obj(d/'Y.obj',a['model_vertices'],a['source_faces'][case['free_ids']]);b.save(d/'quality.json',q)
        n,k=int(case['N']),int(case['K']);bc=len(case['edge_records'])
        row=dict(condition_id=cid,parent_id=uid,group='TRAIN',official_split='train',target_version_id=VERSION,
            template=kind,part_type=kind,extent_description='Whole geometric part proposal; actual image review pending',
            target_face_ids=ids,source_face_ids=ids,construction_evidence=evidence,uncertainty=uncertainty,
            case_path=str(d/'case.npz'),case_sha256=b.sha(d/'case.npz'),input_path=str(d/'input.npz'),input_sha256=b.sha(d/'input.npz'),
            paired_path=str(d/'paired.npz'),paired_sha256=b.sha(d/'paired.npz'),supervision_file=str(d/'supervision_only.npz'),supervision_sha256=b.sha(d/'supervision_only.npz'),
            quality_file=str(d/'quality.json'),quality_sha256=b.sha(d/'quality.json'),N=n,K=k,boundary_edges=bc,
            source_owner_count=int(sup['occupied_labels'].sum()),source_alias_count=int(sup['source_alias_labels'].sum()),
            adapter_semantics='CLOSED_NATIVE_NO_SEAM' if not bc else ('OPEN_WITH_SEAM_OWNER' if sup['occupied_labels'].any() else 'OPEN_NATIVE_NO_OWNER'),
            geometry_status=q['geometry_status'],adapter_compatible=bool(128<=n<=256 and bc<=n-k and q['geometry_status']=='PASS_PENDING_SEMANTIC_REVIEW'),
            parent_target_status=pr['status'],review_status='NOT_REVIEWED',eligibility='CANDIDATE_ONLY',
            model_input_policy='C and N only; target/owner/part not inference features')
        results.append(row)
    b.save(receipt,dict(rows=results,missing=missing,artifact_hashes={str(p):b.sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and p!=receipt}))
    return results

def build():
    configure();s=inventory();start=time.monotonic()
    event('SUPPLY_TARGETS_START',workers=4,resources=b.resources())
    with ProcessPoolExecutor(max_workers=4,initializer=configure) as pool:parents=list(pool.map(b.target_one,s['rows']))
    passed=[r for r in parents if not r['reasons']]
    with ProcessPoolExecutor(max_workers=4,initializer=configure) as pool:rr=list(pool.map(condition_parent,passed))
    old=b.read(OUT/'candidate_manifest.json')
    old['parents']+=parents;old['conditions']+=[r for rows in rr for r in rows]
    old['protected_source_hashes'].update(s['protected_hashes'])
    b.save(OUT/'expanded_candidate_manifest.json',old)
    summary=dict(new_source_candidates=len(parents),new_source_geometry_pass=len(passed),
        geometry_pass_total=sum(not p['reasons'] for p in old['parents']),new_conditions=sum(map(len,rr)),
        wall_seconds=time.monotonic()-start)
    b.save(OUT/'supply/summary.json',summary);event('SUPPLY_COMPLETE',**summary);print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','build']);a=p.parse_args();configure()
    inventory() if a.stage=='inventory' else build()
