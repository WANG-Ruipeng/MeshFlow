"""Audit the exact target after float32 chart encode/decode; not learned output."""
import numpy as np
import traceback
from common import ROOT,dump,save_mesh,mesh_audit
from run_clay import strict_links,sorted_simplex
from exact_audit import exact_mesh
from safe_extract import extract
from fm_state import decode_state

def main():
    with np.load(ROOT/'data.npz') as z:data={k:z[k] for k in z.files}
    mean,scale=data['mean'],data['scale'];rows=[]
    dest=ROOT/'precision_oracle';dest.mkdir()
    for name,state in zip(data['object_names'],data['states']):
        name=str(name)
        try:
            normalized=((state-mean)/scale).astype(np.float32)
            physical=(normalized*scale.astype(np.float32)+mean.astype(np.float32)).astype(np.float64)
            params=decode_state(physical,normalize_normals=False)
            v,f,ids,trace,audit=extract(params);save_mesh(dest/(name+'.obj'),v,f)
            row=dict(object=name,**mesh_audit(v,f),**strict_links(f),**exact_mesh(v,f))
            row['valid']=bool(row['closed_oriented_2manifold'] and row['strict_all_vertex_links_simple_cycles'] and
                [row['beta'+str(k)] for k in range(3)]==[1,0,1] and row['exact_intersection_pairs']==0 and row['exact_degenerate_faces']==0 and row['duplicate_faces']==0)
            with np.load(ROOT/'targets'/name/'mesh.npz') as original:
                row['same_input_id_surface']=np.array_equal(sorted_simplex(ids[f]),sorted_simplex(original['input_vertex_ids'][original['faces']]))
            row['state_max_absolute_change']=float(np.max(abs(physical-state)))
            rows.append(row)
        except Exception as exc:
            row=dict(object=name,valid=False,error=repr(exc),traceback=traceback.format_exc());rows.append(row)
        print('FLOAT32_ORACLE',name,row['valid'],row.get('same_input_id_surface'),flush=True)
    dump(ROOT/'FLOAT32_ORACLE.json',dict(count=24,valid_count=sum(r['valid'] for r in rows),rows=rows,
        scope='Exact training targets after standardized float32 chart conversion. Numerical representation diagnostic; not a trained model.'))
if __name__=='__main__':main()