"""Read-only, versioned exact seam-tip point-contact graph diagnostic.

No coordinates are moved. Point attachment is deliberately never classified
as legal edge attachment. Reuse the frozen detector's contacts/edge ledger.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import collections,csv,json,sys,time
import numpy as np
from .register import OUT,SOURCE,LABELS,read,write,sha,arrays

def bfs(seeds,adj):
    seen=set(seeds);todo=list(seen)
    while todo:
        i=todo.pop()
        for j in adj.get(i,set())-seen:seen.add(j);todo.append(j)
    return seen
def writecsv(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run():
    dest=OUT/'point_contact';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completion.json').exists():
        r=read(dest/'completion.json')
        for p,h in r['outputs'].items():assert sha(p)==h
        print('Verified completed point-contact diagnostic');return
    assert bfs({1},{1:{2},2:{1},3:{4}})=={1,2}
    start=time.perf_counter();rows=[];contacts_out=[];components=[];protected={}
    manifest=read(SOURCE/'evaluation_inputs/manifest.json')
    for label in LABELS:
        for q in manifest['requests']:
            rid=q['request_id'];folder=SOURCE/'evaluation'/label/rid/'post';p=folder/'post.npz'
            scores=SOURCE/'evaluation/scores'/label/rid/'POST'
            used=(p,scores/'valid_edges.csv',scores/'pairs.json',scores/'seam_to_bulk.json')
            protected.update({str(f):sha(f) for f in used})
            z=arrays(p);k=int(z['known_faces']);faces=z['faces'];tri=z['output'].astype(float)
            areas=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
            seam=set(np.flatnonzero(z['retained_seam_mask']).tolist());bulk=set(range(k,len(faces)))-seam
            bulk_area=float(areas[sorted(bulk)].sum());cvertices=set(faces[:k].ravel().tolist())
            pairs=read(scores/'pairs.json');cached=read(scores/'seam_to_bulk.json')
            ff_class={tuple(sorted(int(i)+k for i in x['pair'])):x['classification'] for x in pairs['FF']['contacts']}
            edges=list(csv.DictReader((scores/'valid_edges.csv').open(encoding='utf-8-sig')))
            adj=collections.defaultdict(set);badj=collections.defaultdict(set);seeds=set();legalsb=set()
            for e in edges:
                if e['valid_edge']!='True':continue
                i,j=map(int,json.loads(e['incident_face_ids']))
                if min(i,j)<k:
                    if max(i,j) in seam:seeds.add(max(i,j))
                else:
                    adj[i].add(j);adj[j].add(i)
                    if i in bulk and j in bulk:badj[i].add(j);badj[j].add(i)
                    if (i in seam)!=(j in seam):legalsb.add(tuple(sorted((i,j))))
            oldseen=bfs(seeds,adj);oldbulk=oldseen&bulk;reachedseam=oldseen&seam
            baseline=float(areas[sorted(oldbulk)].sum())/bulk_area if bulk_area>0 else None
            applicable=int(q['boundary_edges'])>0
            if applicable and bulk_area>0:assert abs(baseline-cached['free_only_reached_bulk_area_fraction'])<1e-10,(rid,baseline,cached)
            vertexbulk=collections.defaultdict(set)
            for i in bulk:
                for vid in faces[i]:vertexbulk[int(vid)].add(i)
            point_seeds=set();point_any=set();tip_at_c=0;point_pairs=0;invalid_points=0
            for sf in sorted(seam):
                idx=np.flatnonzero(z['source_corner_ids'][sf]==2)
                assert len(idx)==1
                vid=int(faces[sf,idx[0]]);is_c=vid in cvertices
                tip_at_c+=int(is_c)
                for bf in sorted(vertexbulk[vid]):
                    shared=set(faces[sf].tolist())&set(faces[bf].tolist())
                    classification=ff_class.get(tuple(sorted((sf,bf))),'UNKNOWN_NOT_IN_CONTACT_CACHE')
                    onlypoint=len(shared)==1 and not is_c
                    verified=onlypoint and classification=='shared_vertex_contact'
                    anchored=sf in reachedseam
                    contacts_out.append(dict(label=label,request_id=rid,seam_face=sf,bulk_face=bf,seam_raw_slot=int(z['source_face_ids'][sf]),
                        bulk_raw_slot=int(z['source_face_ids'][bf]),tip_vertex=vid,tip_is_C_vertex=is_c,shared_vertex_count=len(shared),
                        has_legal_pair_edge=tuple(sorted((sf,bf))) in legalsb,cached_classification=classification,
                        seam_connected_to_C_by_legal_edges=anchored,accepted_point_only_diagnostic=verified and anchored))
                    if verified:
                        point_any.add(bf)
                        if anchored:point_seeds.add(bf);point_pairs+=1
                    elif onlypoint and anchored:invalid_points+=1
            # One point jump from C-connected seam, then legal bulk/bulk edges.
            pointbulk=bfs(point_seeds,badj)&bulk;onlyextra=pointbulk-oldbulk
            anybulk=bfs(point_any,badj)&bulk
            numerator=float(areas[sorted(pointbulk)].sum());extra=float(areas[sorted(onlyextra)].sum())
            for seed in sorted(pointbulk):
                components.append(dict(label=label,request_id=rid,output_face_id=seed,raw_slot_id=int(z['source_face_ids'][seed]),
                    area=float(areas[seed]),already_edge_reachable=seed in oldbulk,direct_tip_contact=seed in point_seeds))
            rows.append(dict(label=label,request_id=rid,parent_id=q['parent_id'],condition_id=q['condition_id'],split=q['split'],part_family=q['part_family'],
                status='PASS' if applicable and bulk_area>0 else 'NA_CLOSED_C' if not applicable else 'NA_EMPTY_BULK',
                seam_faces=len(seam),C_connected_seam_faces=len(reachedseam),tip_at_C_vertex_count=tip_at_c,retained_bulk_faces=len(bulk),bulk_area=bulk_area,
                original_legal_edge_reach_fraction=baseline if applicable else None,
                verified_point_pairs=point_pairs,point_candidates_illegal_or_unknown=invalid_points,
                directly_tip_touching_bulk_faces=len(point_seeds),point_component_bulk_faces=len(pointbulk),point_component_bulk_area=numerator,
                point_component_bulk_fraction=numerator/bulk_area if applicable and bulk_area else None,
                extra_point_only_bulk_fraction=extra/bulk_area if applicable and bulk_area else None,
                edge_or_point_bulk_fraction=(extra+float(areas[sorted(oldbulk)].sum()))/bulk_area if applicable and bulk_area else None,
                any_seam_tip_component_fraction=float(areas[sorted(anybulk)].sum())/bulk_area if applicable and bulk_area else None))
    writecsv(dest/'per_output.csv',rows)
    if contacts_out:writecsv(dest/'tip_bulk_contacts.csv',contacts_out)
    if components:writecsv(dest/'point_bulk_faces.csv',components)
    summaries=[]
    fields=('original_legal_edge_reach_fraction','point_component_bulk_fraction','extra_point_only_bulk_fraction','edge_or_point_bulk_fraction','any_seam_tip_component_fraction')
    for label in LABELS:
        for split in ('TRAIN','DEV','ALL'):
            rr=[r for r in rows if r['label']==label and (split=='ALL' or r['split']==split)]
            row=dict(label=label,split=split,outputs=len(rr),applicable=sum(r['status']=='PASS' for r in rr),
                outputs_with_verified_tip_point=sum(r['verified_point_pairs']>0 for r in rr),
                edge_or_point_ge30=sum(r['edge_or_point_bulk_fraction'] is not None and r['edge_or_point_bulk_fraction']>=.3 for r in rr),
                point_pairs=sum(r['verified_point_pairs'] for r in rr))
            for field in fields:
                parent=[]
                for pid in sorted({r['parent_id'] for r in rr}):
                    cr=[r for r in rr if r['parent_id']==pid];cv=[]
                    for cid in sorted({r['condition_id'] for r in cr}):
                        vv=[r[field] for r in cr if r['condition_id']==cid and r[field] is not None]
                        if vv:cv.append(sum(vv)/len(vv))
                    if cv:parent.append(sum(cv)/len(cv))
                row[field]=sum(parent)/len(parent) if parent else None
            summaries.append(row)
    write(dest/'summary.json',summaries)
    for p,h in protected.items():assert sha(p)==h,p
    assert 'torch' not in sys.modules
    write(dest/'completion.json',dict(status='COMPLETE',version='exact_tip_one_point_jump_then_legal_bulk_edges_v1',rows=96,
        outputs={str(p):sha(p) for p in dest.iterdir() if p.is_file()},protected=protected,seconds=time.perf_counter()-start,
        model_calls=0,GPU_calls=0,new_geometry=0,new_queries=0,denominator='All retained bulk area. Closed-C NA. Parent-equal mean of condition/noise means.',
        caution='Point contact is a diagnostic, not a physical connection certificate or a replacement for the original legal-edge metric. A component can contain other illegal faces.',code_sha256=sha(__file__)))
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':run()
