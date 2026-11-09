import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import numpy as np
from .geometry import projection,shape

def main():
    from r2.pilots.bench_user_v2_training_v1 import data as D
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays
    from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
    qs=read(OUT/'evaluation/score_registrations/COMMON.json')['requests'];rows=[];tests=[]
    for q in qs[::2]:
        case=arrays(q['case_path']);core=arrays(q['inference_npz']);source=arrays(q['source_npz']);dest=OUT/'source_reference'/q['condition_id']
        if (dest/'complete.json').exists():rows.append(read(dest/'complete.json')['row']);continue
        if 'fixed_sample_npz' in q:s=arrays(q['fixed_sample_npz'])
        else:
            z=arrays(q['noise_npz'])['z'];s=D.sample_for_probe(case,z,np.zeros_like(z),.5,2026102201)
            s={k:np.asarray(v) for k,v in s.items()};s['shared_vertex_ids']=case['source_face_vertex_ids'][s['source_face_ids'][:,None],s['source_corner_permutations']]
        inv=np.argsort(s['face_permutation']);lay=D.old.inference_layout(core['C_vertices'],core['C_faces_original'],core['C_faces_model'],q['N'],case['occupied_labels'].astype(bool))
        assert np.array_equal(s['anchor_ids'][inv],lay['anchor_ids'])
        mesh=raw_arrays(s['x1'][inv],lay,np.asarray(case['occupied_labels'],np.float32),source)
        ids=s['shared_vertex_ids'][inv];ids=ids[np.arange(len(ids))[:,None],mesh['source_corner_ids']]
        unique,first,inverse=np.unique(ids.reshape(-1),return_index=True,return_inverse=True)
        vertices=mesh['output'].reshape(-1,3)[first];faces=inverse.reshape(-1,3)
        assert np.allclose(vertices[faces],mesh['output'],atol=1e-6)
        mesh.update(vertices=vertices,faces=faces,index_domain=np.asarray('EXACT_SOURCE_VERTEX_ID_NO_CLUSTERING'),identity_domain=np.asarray('SOURCE_REFERENCE_NOT_GENERATION'),
            actual_source_face_ids=s['source_face_ids'][inv],actual_source_vertex_ids=unique)
        array_file(dest/'source.npz',mesh)
        if (dest/'scores/complete.json').exists():
            receipt=read(dest/'scores/complete.json');verify(receipt['files']);row=receipt['row']
        else:row=CPU.score_current(q,mesh,dest/'scores','SOURCE_REFERENCE',True)['row']
        rows.append(row);save(dest/'complete.json',dict(status='COMPLETE',row=row,source_mesh_sha256=sha(dest/'source.npz'),source_use='topology/reference only, not unique-completion success'))
        if 'fixed_sample_npz' in q:
            for t in (.2,.4,.6,.8,.94):
                x=(1-t)*s['x0']+t*s['x1'];v=s['u'];v[s['anchor_ids']>=0]=0
                analytic=projection(x,v,s);step=1e-4;nxt=projection(x+step*v,v,s)
                numerical=(nxt['all_split_mse']-analytic['all_split_mse'])/step
                tests.append(dict(condition_id=q['condition_id'],t=t,analytic=analytic['all_split_derivative'],finite_difference=numerical,
                    abs_difference=abs(numerical-analytic['all_split_derivative']),target_split_mse=projection(s['x1'],np.zeros_like(v),s)['all_split_mse']))
        print('SOURCE',q['condition_id'],flush=True)
    csvwrite(OUT/'source_reference/metrics.csv',rows);csvwrite(OUT/'source_reference/projection_validation.csv',tests)
    save(OUT/'source_reference/complete.json',dict(status='COMPLETE',conditions=len(rows),model_calls=0))
if __name__=='__main__':main()
