"""CPU scalar geometry and descriptive topology; never repair mesh."""
from collections import defaultdict
import numpy as np
from .pair_audit import pair_scan,PROPER,COPLANAR
from .shape import shape_quality

def scan_geometry(C,F,L):
    cf=pair_scan(C,F,L,independent=True);ff=pair_scan(F,F,L,self_pairs=True,independent=True)
    m={}
    for name,scan in (('CF',cf),('FF',ff)):
        for k,s in [('proper_pair_count','proper_pair_count'),('coplanar_pair_count','coplanar_positive_area_pairs'),
                    ('proper_affected_free_fraction','proper_affected_face_fraction'),('unresolved_or_degenerate_pair_count','unresolved_or_degenerate_pairs')]:
            m[name+'_'+k]=scan[s]
    illegal={i for row in ff['contacts'] if row['classification'] in (PROPER,COPLANAR) for i in row['pair']}
    m.update(FF_illegal_pair_count=ff['proper_pair_count']+ff['coplanar_positive_area_pairs'],
             FF_illegal_affected_free_fraction=len(illegal)/len(F))
    q=shape_quality(F,L)
    m.update(free_minimum_angle_lt5_fraction=q['min_angle_lt5_fraction'],free_angle_p05_degrees=q['angle_p05_degrees'],
        free_q_gt10_fraction=q['q_actual_gt10_fraction'],free_q_mean=q['q_actual']['mean'],
        free_near_degenerate_count=q['near_degenerate_area_count'],free_near_degenerate_fraction=q['near_degenerate_area_count']/len(F),
        free_exact_zero_area_count=q['exact_zero_area_count'],free_exact_zero_area_fraction=q['exact_zero_area_count']/len(F),
        free_total_surface_area=q['total_area'])
    return m,cf,ff

def topology(tri,K):
    """RAW triangle soup and exact coordinate-index diagnostic, no tolerance merge."""
    t=np.ascontiguousarray(tri,dtype=np.float32);N=len(t)
    _,inverse=np.unique(t.reshape(-1,3).view(np.uint32).reshape(-1,3),axis=0,return_inverse=True)
    faces=inverse.reshape(-1,3);edges=defaultdict(list)
    for i,f in enumerate(faces):
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):edges[tuple(sorted((int(a),int(b))))].append((i,int(a)<int(b)))
    parent=list(range(N))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for incidence in edges.values():
        for j,_ in incidence[1:]:parent[find(j)]=find(incidence[0][0])
    keys=[tuple(sorted(tuple(map(float,p)) for p in f)) for f in t]
    free_keys=keys[K:]
    return dict(RAW_SOUP_face_components=N,RAW_SOUP_CF_shared_edges=0,
        EXACT_COORD_vertices=int(inverse.max()+1) if N else 0,EXACT_COORD_edges=len(edges),
        EXACT_COORD_boundary_edges=sum(len(v)==1 for v in edges.values()),
        EXACT_COORD_nonmanifold_edges=sum(len(v)>2 for v in edges.values()),
        EXACT_COORD_same_direction_two_face_edges=sum(len(v)==2 and v[0][1]==v[1][1] for v in edges.values()),
        EXACT_COORD_face_components=len({find(i) for i in range(N)}),
        EXACT_COORD_CF_shared_edges=sum(any(i<K for i,_ in v) and any(i>=K for i,_ in v) for v in edges.values()),
        EXACT_COORD_closed_two_faces_per_edge=bool(edges) and all(len(v)==2 for v in edges.values()),
        free_exact_duplicate_excess_fraction=(len(free_keys)-len(set(free_keys)))/len(free_keys) if free_keys else None)
