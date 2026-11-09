"""Why a fixed coordinate-grid fallback is not yet justified for source945."""
import json
import numpy as np
from .common import OUT,read,load,save,sha,resolve
from r2.data_tools.bench_parts_v1 import quality as Q

def run():
    p=next(p for p in read(OUT/'PROTOCOL.json')['parents'] if p['parent_id'].startswith('945214'))
    src=resolve(p['source']['clean_case_path']);a=load(src)
    v=a['model_vertices'][np.unique(a['source_faces'])].astype(float)
    bound=float(.0025*Q.L/np.sqrt(3));axes=[]
    for axis in range(3):
        levels=np.unique(v[:,axis]);gaps=np.diff(levels);close=np.flatnonzero(gaps<bound)
        axes.append(dict(axis='XYZ'[axis],levels=levels.tolist(),minimum_gap=float(gaps.min()),
            proposed_group_diameter_limit_model=bound,
            close_adjacent_pairs=[dict(low=float(levels[i]),high=float(levels[i+1]),gap=float(gaps[i])) for i in close]))
    result=dict(schema='BENCH945_COORDINATE_GRID_READONLY_ASSESSMENT_V1',parent_id=p['parent_id'],source_N=len(a['source_faces']),
        required_face_reduction_for_256=len(a['source_faces'])-256,axis_facts=axes,
        status='PROPOSAL_NOT_EXECUTED_NO_SUPPORTED_STRUCTURE_SAFE_RECIPE',
        reason='Only one close Y-level pair passes the per-axis diameter rule. Nearness does not establish redundant tessellation; it may encode thin structure or an assembly gap. No demonstrated route from448 to256 preserving the one component and8 open-boundary loops.',
        decision='Keep source/first exact attempt. Do not merge these levels merely to lower face count. This is not an impossibility claim for all part-based reconstructions.',
        protected={str(src):sha(src),str(OUT/'PROTOCOL.json'):sha(OUT/'PROTOCOL.json'),__file__:sha(__file__)},
        final_target_written=False,geometry_proposals=0,model_calls=0,source_mutations=0)
    save(OUT/'part945_recipe_assessment.json',result);return result

if __name__=='__main__':
    r=run();print(json.dumps(dict(status=r['status'],axes=r['axis_facts']),ensure_ascii=False))
