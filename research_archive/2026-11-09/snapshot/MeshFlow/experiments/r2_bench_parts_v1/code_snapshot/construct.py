"""Source-face-preserving Bench part *candidates*, CPU only.

Geometry proposes regions; only a separate actual image review can assign a
semantic part label. No source repair, welding, remeshing, model, OT or RNG.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, time
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'experiments/r2_bench_control_transfer_v1/data/manifest.json'
OUT = ROOT / 'experiments/r2_bench_parts_v1/candidates'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n'
    if path.exists():
        if path.read_text('utf-8') != text: raise FileExistsError(str(path))
    else: path.write_text(text, encoding='utf-8')

def load_case(row):
    path = Path(row['case_path'])
    if not path.exists():
        # Native Windows and WSL use the same source file, never a copied mesh.
        if str(path).startswith('/mnt/e/MeshFlow/'):
            path = ROOT / str(path)[len('/mnt/e/MeshFlow/'):]
    with np.load(path, allow_pickle=False) as z: a = {k:z[k].copy() for k in z.files}
    return path, a

def book(f, ids=None):
    e = collections.defaultdict(list)
    for i in range(len(f)) if ids is None else ids:
        for j in range(3):
            u, v = map(int, (f[i,j], f[i,(j+1)%3]))
            e[tuple(sorted((u,v)))].append((int(i),u,v))
    return e

def comps(f, ids):
    remaining = set(map(int,ids)); adj = {i:set() for i in remaining}
    for entries in book(f, remaining).values():
        faces = {x[0] for x in entries}
        for i in faces: adj[i].update(faces-{i})
    out=[]
    while remaining:
        pending=[min(remaining)]; c=[]
        while pending:
            i=pending.pop()
            if i not in remaining: continue
            remaining.remove(i);c.append(i);pending.extend(sorted(adj[i]&remaining,reverse=True))
        out.append(sorted(c))
    return sorted(out,key=lambda x:(-len(x),x))

def geom(a):
    v=a['normalized_vertices'].astype(np.float64); f=a['source_faces']; t=v[f]
    cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); area=np.linalg.norm(cross,axis=1)/2
    n=cross/np.maximum(2*area[:,None],1e-30)
    return v,f,t,area,n,t.mean(1)

def planar_regions(a):
    v,f,t,area,n,cent=geom(a);adj=[set() for _ in f]
    for entries in book(f).values():
        for i,u,w in entries:
            for j,x,y in entries:
                if i>=j or (u,w)!=(y,x) or min(area[i],area[j])<=1e-12:continue
                if np.dot(n[i],n[j])>.998 and abs(np.dot(cent[j]-cent[i],n[i]))<1e-4:
                    adj[i].add(j);adj[j].add(i)
    rem=set(range(len(f)));rs=[]
    while rem:
        q=[min(rem)];ids=[]
        while q:
            i=q.pop()
            if i not in rem:continue
            rem.remove(i);ids.append(i);q.extend(sorted(adj[i]&rem,reverse=True))
        ids=sorted(ids)
        if len(ids)<2 or area[ids].min()<=1e-12:continue
        pts=t[ids].reshape(-1,3);nn=(n[ids]*area[ids,None]).sum(0);nn/=np.linalg.norm(nn)
        rs.append(dict(ids=ids,area=float(area[ids].sum()),normal=nn,
                       center=np.average(cent[ids],axis=0,weights=area[ids]),
                       low=pts.min(0),high=pts.max(0),span=np.ptp(pts,axis=0)))
    return rs

def topology(a, ids):
    f=a['source_faces'];entries=book(f,ids)
    return dict(source_edge_components=len(comps(f,ids)),
        boundary_edges=sum(len(x)==1 for x in entries.values()),
        nonmanifold_edges=sum(len(x)>2 for x in entries.values()),
        same_direction_edges=sum(len(x)==2 and x[0][1:]==x[1][1:] for x in entries.values()),
        duplicate_ID_faces=len(ids)-len(np.unique(np.sort(f[ids],axis=1),axis=0)))

def basic_eligible(a,ids):
    if not 2<=len(ids)<len(a['source_faces']):return False
    q=topology(a,ids);_,_,_,area,_,_=geom(a)
    return (q['source_edge_components']==1 and q['nonmanifold_edges']==0
            and q['same_direction_edges']==0 and q['duplicate_ID_faces']==0
            and bool((area[ids]>1e-12).all()))

def propose(a):
    v,f,t,area,n,cent=geom(a);height=float(np.ptp(v[:,1]));floor=float(v[:,1].min())
    rs=planar_regions(a); results=[];blocked=[]
    # Broad upward-facing source patches are proposals, not semantic labels.
    seats=[r for r in rs if abs(r['normal'][1])>.85 and r['span'][[0,2]].min()>.025
           and r['center'][1]>floor+.22*height and r['area']>.004
           and basic_eligible(a,r['ids'])]
    if not seats:
        blocked.append(dict(proposed_part_type='SEAT_SURFACE',reason='NO_QUALIFIED_COHERENT_HORIZONTAL_PATCH'))
        return results,blocked
    largest=max(r['area'] for r in seats)
    broad=[r for r in seats if r['area']>=.55*largest]
    # Choosing the highest of similarly broad surfaces avoids underside copies.
    seat=max(broad,key=lambda r:(r['center'][1],r['normal'][1],r['area'],-min(r['ids'])))
    sy=float(seat['center'][1]);sheight=sy-floor
    results.append(('SEAT_SURFACE',0,seat['ids'],dict(method='broad_horizontal_top_candidate',
        reference_height=sy,normal=seat['normal'].tolist(),similar_area_fraction=.55,
        qualification='original face-connected planar surface; top-most among >=55% largest horizontal area'),
        ['NO_SOURCE_SEMANTIC_LABEL','MAY_BE_SINGLE_SLAT_OR_BOARD_NOT_WHOLE_SEAT']))
    # A leg/support may bend and is not required to be planar. Whole source
    # triangles wholly below the seat are grouped on exact source-edge IDs.
    below=np.flatnonzero((t[:,:,1].max(1)<=sy+.025*max(sheight,.01)) & (area>1e-12))
    supports=[]
    for ids in comps(f,below):
        pts=t[ids].reshape(-1,3);span=np.ptp(pts,axis=0);lo=pts.min(0);hi=pts.max(0)
        if span[1]<.45*sheight or lo[1]>floor+.22*sheight or hi[1]<floor+.65*sheight:continue
        if len(ids)>max(96,int(len(f)*.65)):continue
        if basic_eligible(a,ids):
            supports.append(dict(ids=ids,area=float(area[ids].sum()),span=span,low=lo,high=hi,
                                 center=pts.mean(0),method='whole_source_below_seat_component'))
    # Where the full component is topologically ambiguous (e.g. two-sided
    # source sheets), propose an original coherent leg SIDE surface instead.
    # This selects a region; it does not deduplicate or modify the target.
    for r in rs:
        if r['high'][1]>sy+.025*max(sheight,.01) or r['low'][1]>floor+.22*sheight:continue
        if r['span'][1]<.45*sheight or r['high'][1]<floor+.65*sheight:continue
        if abs(r['normal'][1])>.65 or r['area']<.001:continue
        if not basic_eligible(a,r['ids']):continue
        if any(set(r['ids']).issubset(x['ids']) for x in supports):continue
        supports.append(dict(r,method='coherent_original_support_side_surface'))
    chosen=[]
    supports.sort(key=lambda r:(r['method']!='whole_source_below_seat_component',-r['span'][1],-r['area'],r['ids']))
    for r in supports:
        # Reject opposite views of the same support; prioritize distinct ends.
        horizontal_center=r['center'][[0,2]]
        if any(np.linalg.norm(horizontal_center+x['center'][[0,2]]*-1)<.12*max(np.ptp(v[:,[0,2]],axis=0))
               and np.linalg.norm(r['span'][[0,2]]-x['span'][[0,2]])<.10 for x in chosen):continue
        if any(set(r['ids'])&set(x['ids']) for x in chosen):continue
        chosen.append(r)
        if len(chosen)==2:break
    for i,r in enumerate(chosen):
        results.append(('LEG_SUPPORT',i,r['ids'],dict(method=r['method'],seat_reference_height=sy,
            floor_height=floor,vertical_span=float(r['span'][1]),span_normalized=r['span'].tolist(),
            qualification='source-edge-connected whole faces; reaches lower support zone'),
            ['NO_SOURCE_SEMANTIC_LABEL','MAY_BE_LEG_SIDE_OR_END_FRAME_NOT_COMPLETE_SOLID_LEG']))
    if not chosen:blocked.append(dict(proposed_part_type='LEG_SUPPORT',reason='NO_SEPARABLE_CONNECTED_SUPPORT_REGION_WITH_VALID_SOURCE_TOPOLOGY'))
    # Back proposals must extend above the selected horizontal region and
    # have a broad horizontal axis. No back is invented for backless benches.
    backs=[]
    for r in rs:
        if abs(r['normal'][1])>.65 or r['span'][1]<.14*height:continue
        if r['center'][1]<sy+.09*height or r['low'][1]<sy-.12*height:continue
        if max(r['span'][0],r['span'][2])<.20*max(np.ptp(v[:,[0,2]],axis=0)):continue
        if r['area']<.004 or not basic_eligible(a,r['ids']):continue
        backs.append(r)
    if backs:
        r=max(backs,key=lambda r:(r['area'],r['center'][1],-min(r['ids'])))
        results.append(('BACK_SURFACE',0,r['ids'],dict(method='broad_upper_vertical_source_patch',
            seat_reference_height=sy,normal=r['normal'].tolist(),
            qualification='broad coherent vertical/tilted surface predominantly above seat reference'),
            ['NO_SOURCE_SEMANTIC_LABEL','FRONT_VS_REAR_FACE_NOT_CERTIFIED','MAY_BE_ARM_OR_END_PANEL_PENDING_IMAGE_REVIEW']))
    else:blocked.append(dict(proposed_part_type='BACK_SURFACE',reason='NO_QUALIFIED_UPPER_BACK_CANDIDATE_OR_BACKLESS_OBJECT'))
    return results,blocked

def partition(a,ids):
    ids=np.asarray(sorted(ids),np.int64);f=a['source_faces'];free=np.setdiff1d(np.arange(len(f)),ids)
    vids=np.unique(f[ids]);lookup={int(v):i for i,v in enumerate(vids)}
    orig=np.asarray([[lookup[int(v)] for v in face] for face in f[ids]],np.int64)
    canonical=np.asarray([[lookup[int(v)] for v in face] for face in a['source_face_vertex_ids'][ids]],np.int64)
    c=a['full_target'][ids].copy();y=a['full_target'][free].copy();recon=np.empty_like(a['full_target']);recon[ids]=c;recon[free]=y
    assert recon.tobytes()==a['full_target'].tobytes()
    result=dict(C=c,Y=y,known_ids=ids,free_ids=free,source_face_ids=ids,free_source_face_ids=free,
        C_vertices=a['model_vertices'][vids],C_faces_original=orig,C_faces_model=canonical,
        C_source_vertex_ids=vids,C_source_corner_permutations=a['source_corner_permutations'][ids],
        C_source_faces_original=f[ids],C_pre_ot=a['pre_ot_full'][ids],Y_pre_ot=a['pre_ot_full'][free],
        N=np.asarray(len(f),np.int64),K=np.asarray(len(ids),np.int64))
    assert result['C_vertices'][canonical].tobytes()==c.tobytes()
    return result

def supplement_3461():
    """Explicit source-reviewed exception; does not change the frozen 221."""
    old=json.loads(OLD.read_text('utf-8'))
    source=next(r for r in old['train'] if r['parent_id'].startswith('3461cf51'))
    path,a=load_case(source);v,f,t,area,normal,cent=geom(a)
    out=OUT/'supplement_3461';rows=[]
    # A genuine narrow source slat failed the broad-patch minimum width. The
    # two rectangular vertical side surfaces belong to distinct source legs.
    chosen=[('SEAT_SURFACE',1,[210,211]),('LEG_SUPPORT',1,[0,4]),('LEG_SUPPORT',2,[16,20])]
    for kind,i,ids in chosen:
        assert basic_eligible(a,ids)
        cid=source['parent_id']+'__'+kind+'_'+str(i)
        dest=out/'cases'/(cid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        arrays=partition(a,ids)
        if not dest.exists():
            with dest.open('xb') as h:np.savez_compressed(h,**arrays)
        else:
            with np.load(dest,allow_pickle=False) as z:assert all(z[k].tobytes()==x.tobytes() for k,x in arrays.items())
        pts=t[ids].reshape(-1,3)
        rows.append(dict(candidate_id=cid,parent_id=source['parent_id'],group=source['group'],official_split=source['official_split'],
            source_case_path=str(path),source_case_sha256=source['case_sha256'],source_face_ids=ids,
            proposed_part_type=kind,part_index=i,construction_evidence=dict(method='source_image_and_exact_face_probe_supplement',
                source_image='experiments/r2_bench_control_transfer_v1/data/gallery/TRAIN_00.png',
                reason='Original slat width .02084 below automatic .025 minimum; explicit original rectangular face pair. Legs are side surfaces only.'),
            uncertainty=['CANDIDATE_REQUIRES_NEW_C_HIGHLIGHT_REVIEW','NOT_WHOLE_SEAT_OR_CLOSED_LEG'],
            status='PENDING_VISUAL_AND_GEOMETRY_REVIEW',semantic_authority='GEOMETRIC_PROPOSAL_ONLY',labels_fed_to_model=False,
            case_path=str(dest),case_sha256=sha(dest),N=len(f),K=len(ids),face_fraction=len(ids)/len(f),
            area=float(area[ids].sum()),area_fraction=float(area[ids].sum()/area.sum()),
            area_units='official_normalized_object_units_squared',area_denominator='sum_all_original_source_faces_including_duplicates',
            bbox_normalized=[pts.min(0).tolist(),pts.max(0).tolist()],span_normalized=np.ptp(pts,axis=0).tolist(),
            region_normal_not_single_plane=False,geometry_unchanged=True,source_partition_bitwise=True,**topology(a,ids)))
    write(out/'manifest.json',dict(schema='bench_part_candidates_supplement_v1',rows=rows,
         source_manifest_sha256=sha(OLD),reason='Recover narrow coherent source slat and two leg-side surfaces missed by broad-patch gate.',
         original_221_manifest_unchanged=True,model_calls=0,training_updates=0,
         blocked=[dict(parent_id=source['parent_id'],proposed_part_type='BACK_SURFACE',status='NA',reason='BACKLESS_BENCH_SOURCE_IMAGE')]))
    print(json.dumps({'supplement_candidates':len(rows),'manifest':str(out/'manifest.json')}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--probe',action='store_true');p.add_argument('--supplement-3461',action='store_true');args=p.parse_args()
    if args.supplement_3461:supplement_3461();return
    started=time.time();old=json.loads(OLD.read_text('utf-8'));rows=[];blocked=[];summary=[]
    for source in old['train']+old['dev']:
        sourcepath,a=load_case(source);assert sha(sourcepath)==source['case_sha256']
        proposals,missing=propose(a);v,f,t,area,n,cent=geom(a)
        summary.append(dict(parent_id=source['parent_id'],group=source['group'],N=len(f),
            planar_patches=len(planar_regions(a)),proposals=[dict(kind=x[0],K=len(x[2]),face_ids=x[2],evidence=x[3]) for x in proposals]))
        for m in missing:blocked.append(dict(parent_id=source['parent_id'],group=source['group'],status='BLOCKED_CANDIDATE_CONSTRUCTION',**m))
        if args.probe:continue
        for kind,i,ids,evidence,uncertainty in proposals:
            cid=source['parent_id']+'__'+kind+'_'+str(i+1);dest=args.out/'cases'/(cid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            arrays=partition(a,ids)
            if dest.exists():
                with np.load(dest,allow_pickle=False) as z:
                    assert set(z.files)==set(arrays) and all(z[k].tobytes()==x.tobytes() for k,x in arrays.items())
            else:
                with dest.open('xb') as h:np.savez_compressed(h,**arrays)
            pts=t[ids].reshape(-1,3);topo=topology(a,ids)
            rows.append(dict(candidate_id=cid,parent_id=source['parent_id'],group=source['group'],
                official_split=source['official_split'],source_case_path=str(sourcepath),source_case_sha256=source['case_sha256'],
                source_face_ids=ids,proposed_part_type=kind,part_index=i+1,
                construction_evidence=evidence,uncertainty=uncertainty,status='PENDING_VISUAL_AND_GEOMETRY_REVIEW',
                semantic_authority='GEOMETRIC_PROPOSAL_ONLY',labels_fed_to_model=False,case_path=str(dest),case_sha256=sha(dest),
                N=len(f),K=len(ids),face_fraction=len(ids)/len(f),area=float(area[ids].sum()),
                area_fraction=float(area[ids].sum()/area.sum()),area_units='official_normalized_object_units_squared',
                area_denominator='sum_all_original_source_faces_including_duplicates',
                bbox_normalized=[pts.min(0).tolist(),pts.max(0).tolist()],span_normalized=np.ptp(pts,axis=0).tolist(),
                region_normal_not_single_plane=bool(np.linalg.norm(np.std(n[ids],axis=0))>.1),
                geometry_unchanged=True,source_partition_bitwise=True,**topo))
    if args.probe:
        write(ROOT/'experiments/r2_bench_parts_v1/probe/region_proposals.json',dict(rows=summary,blocked=blocked));print(json.dumps({'parents':len(summary),'proposals':dict(collections.Counter(p['kind'] for s in summary for p in s['proposals'])),'blocked':len(blocked)},indent=2));return
    write(args.out/'manifest.json',dict(schema='bench_part_candidates_v1',source_manifest=str(OLD),source_manifest_sha256=sha(OLD),
        rows=rows,blocked=blocked,counts=dict(collections.Counter(r['proposed_part_type'] for r in rows)),
        status='CANDIDATES_NOT_TRAINING_DATA_UNTIL_REVIEW',parent_count=72,source_train_parents=64,source_dev_parents=8,
        source_modified=False,model_calls=0,training_updates=0,new_noise_arrays=0,OT_calls=0,
        selection_policy='Whole original source faces only; geometric proposals require actual semantic image review. No source repair/dedup/weld/remesh.'))
    write(args.out/'execution.json',dict(wall_seconds=time.time()-started,candidates=len(rows),blocked=len(blocked),model_calls=0,training_updates=0,gpu_calls=0))
    print(json.dumps({'candidates':len(rows),'types':dict(collections.Counter(r['proposed_part_type'] for r in rows)),'blocked':len(blocked)},indent=2))

if __name__=='__main__':main()
