"""Read-only topological bound for one already running ALPHA stage; no scorer."""
from . import common as C
from collections import Counter,defaultdict
from itertools import combinations
import numpy as np

def run():
    rid='af264c02f69566cdf4fb4dee5181bee__COMPLETE_SLAT_SEAT_UPPER_SURFACE_GROUP__HIGH_JOINT_V1__n0'
    folder=C.OUT/'cpu/HYBRID'/rid
    path=folder/'HINGE_OPP015/post.npz'
    mesh=C.arrays(path);faces=mesh['faces'];vertices=mesh['vertices'];known=int(mesh['known_faces'])
    inc=defaultdict(list)
    for i,face in enumerate(faces):
        for a,b in ((0,1),(1,2),(2,0)):
            u,w=int(face[a]),int(face[b]);inc[tuple(sorted((u,w)))].append((i,u,w))
    by_hinge=defaultdict(list)
    for occurrences in inc.values():
        if len(occurrences)!=1:continue
        i,u,w=occurrences[0]
        if i<known:continue
        by_hinge[u].append((i,w,1));by_hinge[w].append((i,u,-1))
    loose=sum(len(v)*(len(v)-1)//2 for v in by_hinge.values())
    filtered=0;cap=.015*C.L
    for h,half in by_hinge.items():
        for (i,u,d),(j,w,dd) in combinations(half,2):
            if i==j or u==w or d==dd:continue
            if np.linalg.norm(vertices[u].astype(float)-vertices[w])>cap+1e-12:continue
            filtered+=1
    report={'status':'READ_ONLY_BOUND_NOT_LIVE_PROGRESS','utc':C.utc(),'request_id':rid,
        'input_HINGE_post':str(path),'input_sha256':C.sha(path),'faces':len(faces),'C_faces':known,
        'open_free_halfedges':sum(map(len,by_hinge.values())),
        'unfiltered_vertex_pair_bound':loose,'same_face_endpoint_winding_gap_prefiltered_candidate_upper_bound':filtered,
        'maximum_distinct_proposals_per_candidate':6,
        'maximum_location_attempts_upper_bound':6*filtered,'maximum_new_full_geometry_scans_upper_bound':6*filtered,
        'bound_scope':'ALPHA initial queue only. Extra fixed-C/cluster constraints reduce this bound; proposals are ORIGINAL plus five fixed alphas, FP32-deduplicated, and acceptance stops each candidate. No queue expansion occurs.',
        'current_candidate_rank':'UNKNOWN','completed_ALPHA_attempts':'UNKNOWN','time_remaining':'UNKNOWN',
        'reason':'ALPHA keeps candidate records and scan counters in memory; CPU producer writes them only after decode returns. No live per-candidate progress artifact exists.',
        'model_calls':0,'scoring_calls':0,'candidate_decode_calls':0,'changed_scientific_files':False}
    C.write(C.OUT/'audits/af264_n0_progress_readonly.json',report)
    print(report)

if __name__=='__main__':run()
