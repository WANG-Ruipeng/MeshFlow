"""Versioned diagnostics on saved full indexed meshes, never section pieces.

Original pair classification is reused. Legal-edge and unfiltered-edge graphs
are both retained: failure of legality is not silently called physical tearing.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,collections,csv,json,time,resource
import numpy as np
from .evaluation import C,OUT,HIST,ARMS,requests,mesh_path,score_path

VERSION='hinge_fragment_topology_full_indexed_v2'

def truth(v):
    if isinstance(v,bool):return v
    if v in ('True','true',1):return True
    if v in ('False','false',0,'',None):return False
    raise ValueError('Invalid boolean '+str(v))

def edges_file(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        r['incident_face_ids']=json.loads(r['incident_face_ids'])
        r['valid_edge']=truth(r['valid_edge']);r['winding_opposite']=truth(r['winding_opposite'])
    return rows

def components(n,links,K,areas):
    adj=[set() for _ in range(n)]
    for i,j in links:adj[i].add(j);adj[j].add(i)
    ids=np.full(n,-1,np.int64);records=[]
    for start in range(n):
        if ids[start]>=0:continue
        ci=len(records);todo=[start];ids[start]=ci;seen=[]
        while todo:
            i=todo.pop();seen.append(i)
            for j in sorted(adj[i]):
                if ids[j]<0:ids[j]=ci;todo.append(j)
        seen.sort();records.append(dict(component_id=ci,face_indices=seen,face_count=len(seen),
            has_C=any(i<K for i in seen),area_model2=float(areas[seen].sum())))
    return ids,records,adj

def measure(mesh,K,edges,L):
    tri=np.asarray(mesh['output'],np.float64);faces=np.asarray(mesh['faces'],np.int64)
    vertices=np.asarray(mesh['vertices']);raw=np.asarray(mesh['raw_slot_ids'],np.int64)
    seam=np.asarray(mesh['raw_seam_slot_mask'],bool)[raw];n=len(tri)
    assert np.array_equal(vertices[faces],mesh['output']) and len(raw)==n
    areas=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    by_v=collections.defaultdict(set);by_e=collections.defaultdict(set)
    for i,f in enumerate(faces):
        for v in f:by_v[int(v)].add(i)
        for k in range(3):by_e[tuple(sorted((int(f[k]),int(f[(k+1)%3]))))].add(i)
    def all_pairs(groups):
        result=[]
        for group in groups:
            aa=sorted(group)
            result.extend((aa[i],aa[j]) for i in range(len(aa)) for j in range(i+1,len(aa)))
        return result
    legal=[tuple(map(int,e['incident_face_ids'])) for e in edges if truth(e['valid_edge'])]
    opposed=[tuple(map(int,e['incident_face_ids'])) for e in edges if truth(e['valid_edge']) and truth(e['winding_opposite'])]
    graphs={}
    for name,links in [('vertex',all_pairs(by_v.values())),('edge_unfiltered',all_pairs(by_e.values())),
                       ('edge_legal',legal),('edge_legal_opposed',opposed)]:
        graphs[name]=components(n,links,K,areas)
    masks={name:np.asarray([comp[int(cid)]['has_C'] for cid in ids],bool)
           for name,(ids,comp,adj) in graphs.items()}
    free=np.arange(n)>=K;bulk=free&~seam;den=float(areas[free].sum())
    vertex_only=free&masks['vertex']&~masks['edge_legal']
    unfiltered_only=free&masks['vertex']&~masks['edge_unfiltered']
    legal_single=[r for r in graphs['edge_legal'][1] if r['face_count']==1 and r['face_indices'][0]>=K]
    edge_single=[r for r in graphs['edge_unfiltered'][1] if r['face_count']==1 and r['face_indices'][0]>=K]
    metrics=dict(hinge_diag_isolated_1face_edge_components=len(legal_single),
        hinge_diag_vertex_attached_only_free_area_fraction=float(areas[vertex_only].sum())/den if den else None,
        hinge_diag_bulk_only_valid_shared_edge_count=sum(bool(bulk[i] and bulk[j]) for i,j in legal),
        hinge_diag_isolated_1face_unfiltered_edge_components=len(edge_single),
        hinge_diag_vertex_attached_only_unfiltered_free_area_fraction=float(areas[unfiltered_only].sum())/den if den else None,
        hinge_diag_open_edge_count=sum(len(v)==1 for v in by_e.values()),
        hinge_diag_free_area_denominator=den,
        hinge_diag_vertex_attached_only_free_area_numerator=float(areas[vertex_only].sum()),
        hinge_diag_fully_C_disconnected_free_area_fraction=float(areas[free&~masks['vertex']].sum())/den if den else None,
        hinge_diag_legal_edge_C_reached_free_area_fraction=float(areas[free&masks['edge_legal']].sum())/den if den else None,
        hinge_diag_legal_opposed_edge_C_reached_free_area_fraction=float(areas[free&masks['edge_legal_opposed']].sum())/den if den else None)
    rows=[]
    for i,t in enumerate(tri):
        lens=np.linalg.norm(t-np.roll(t,-1,axis=0),axis=1);angles=[]
        for k in range(3):
            a=t[(k+1)%3]-t[k];b=t[(k+2)%3]-t[k];dd=np.linalg.norm(a)*np.linalg.norm(b)
            angles.append(float(np.degrees(np.arccos(np.clip(np.dot(a,b)/dd,-1,1)))) if dd>0 else 0.)
        row=dict(post_face_index=i,source_raw_slot=int(raw[i]),role='C' if i<K else 'SEAM' if seam[i] else 'BULK',
            area_model2=float(areas[i]),area_over_L2=float(areas[i]/L**2),minimum_angle_degrees=min(angles),
            vertex_ids=faces[i].tolist(),vertex_attached_only_legal=bool(vertex_only[i]),
            vertex_attached_only_unfiltered=bool(unfiltered_only[i]))
        for name,(ids,records,adj) in graphs.items():
            rr=records[int(ids[i])]
            row.update({name+'_'+key:rr[key] for key in ('component_id','face_count','has_C','area_model2')})
            row[name+'_neighbor_raw_slots']=sorted(int(raw[j]) for j in adj[i])
        rows.append(row)
    return metrics,rows

def registry(labels,include_hinge):
    records=[]
    for label in labels:
        for q in requests():
            for stage in ('POST','BOUND015'):
                p=score_path(label,q['request_id'],stage)
                records.append(dict(label=label,stage=stage,request_id=q['request_id'],mesh_path=str(mesh_path(label,q['request_id'],stage)),
                    score_path=str(p),edges_path=str(p.parent/'valid_edges.csv')))
    if include_hinge:
        reg=C.read(OUT/'hinge/registry.json')
        records.extend(reg if isinstance(reg,list) else reg.get('records',reg.get('cells',[])))
    return records

def run(labels,include_hinge=False):
    start=time.perf_counter();qs={q['request_id']:q for q in requests()};records=registry(labels,include_hinge)
    name='final_v2' if include_hinge else '_'.join(labels)+'_v2';dest=OUT/'topology'/name
    if (dest/'completion.json').exists():
        old=C.read(dest/'completion.json')
        for p,h in old['files'].items():assert C.sha(p)==h
        return old
    C.resources(dest/'resources.json');rows=[];faces=[];protected={}
    for r in records:
        q=qs[r['request_id']];p=C.resolve(r['mesh_path']);ep=C.resolve(r['edges_path'])
        sp=C.resolve(r.get('score_path',r.get('scorepath')))
        for f in (p,ep,sp):protected[str(f)]=C.sha(f)
        score=C.read(sp)
        receipt=score if 'files' in score else C.read(sp.parent/'complete.json')
        for f,h in receipt['files'].items():assert C.sha(f)==h
        m,rr=measure(C.arrays(p),int(q['K']),edges_file(ep),C.L)
        common=dict(label=r['label'],stage=r['stage'],**{k:q[k] for k in ('request_id','parent_id','condition_id','noise_index','split','K','N')})
        rows.append(dict(**common,**m,mesh_path=str(p),definition_version=VERSION))
        faces.extend(dict(**common,**f) for f in rr)
    paths={}
    for name,rr in [('per_output',rows),('per_face_connectivity',faces)]:
        p=dest/(name+'.csv');C.csvwrite(p,rr);paths[str(p)]=C.sha(p)
    for p,h in protected.items():assert C.sha(p)==h
    result=dict(status='COMPLETE',version=VERSION,code_sha256=C.sha(__file__),cells=len(rows),face_records=len(faces),seconds=time.perf_counter()-start,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,files=paths,protected=protected,
        definition={'legal_graph':'Original valid_edge records (shared_edge_contact plus original legality), winding not added silently; opposed graph separate.',
                    'unfiltered_graph':'All actual indexed edges, including nonmanifold/illegal, for separating geometric split from legality split.',
                    'vertex_graph':'All actual shared vertex IDs, not proximity.',
                    'single_face':'Free faces alone in a legal edge component; C singleton excluded. No size threshold.',
                    'area_fraction':'All retained free face area denominator; C-connected in actual vertex graph but not legal edge graph numerator; NA if zero.',
                    'bulk_edges':'Legal exact-index two-face edges with both faces bulk; count depends on retained face budget, not semantic quality.'},
        model_calls=0,new_quality_scans=0,visual='PENDING_ACTUAL_REVIEW; these graphs do not prove floating or unsuitable semantic parts')
    C.write(dest/'completion.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels',nargs='+',default=['PLAIN_S2000',*ARMS]);p.add_argument('--include-hinge',action='store_true');a=p.parse_args()
    r=run(a.labels,a.include_hinge);print(json.dumps({k:v for k,v in r.items() if k not in ('protected','files')}),flush=True)
