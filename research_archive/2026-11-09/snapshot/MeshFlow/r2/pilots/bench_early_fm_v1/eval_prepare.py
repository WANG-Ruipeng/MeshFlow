"""Freeze unchanged inference/export/scoring adapters before final sampling."""
from .common import *
import numpy as np

def main():
    done=OUT/'evaluation/registration.json'
    if done.exists():verify(read(done)['files']);return
    reg=registration();qs=reg['requests'];rows=[];files={}
    from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
    from r2.pilots.chair_back_transfer_v1.evaluate import source_hashes
    for q in qs:
        src=Path(q['baseline']['USER']['raw']).parent/'source_condition.npz'
        source=arrays(src);core=arrays(q['inference_npz']);lay=arrays(q['layout_npz'])
        assert set(source)=={'source_winding_schema','source_winding_provenance','source_vertex_id_status',
            'source_face_id_status','C','C_original','C_source_vertex_ids','C_original_source_vertex_ids',
            'C_source_corner_permutations','C_original_source_face_ids'},'C-only metadata contract'
        assert np.array_equal(source['C_original'],core['C_vertices'][core['C_faces_original']])
        fixed=lay['anchor_ids']>=0;x=np.zeros((q['N'],3,3),np.float32)
        x[q['K']:]=arrays(q['noise_npz'])['z'];x[fixed]=lay['C_vertices'][lay['anchor_ids'][fixed]]
        for name in ('COMMON','USER'):
            old=arrays(q['baseline'][name]['trajectory'])['states'][0]
            assert np.array_equal(x,old),'Frozen start differs'
        rows.append(dict(request_id=q['request_id'],source_C_metadata=str(src),source_sha256=sha(src),
            both_original_initial_states_exact=True,source_Y_fields_absent=True))
        files[str(src)]=sha(src)
    code=source_hashes()
    for p in (Path(CPU.__file__),Path(CPU.H.__file__),Path(CPU.alpha.__file__),Path(CPU.holes.__file__),
              Path(CPU.M.__file__),Path(__file__).with_name('gpu.py'),Path(__file__).with_name('evaluate.py')):
        code[str(p)]=sha(p)
    save(done,dict(status='FROZEN_CPU_PASS',rows=rows,files=files,source_hashes=code,
        training_registration_sha256=sha(OUT/'training_registration.json'),generator_calls=800,D0_calls=16,
        new_head_calls=0,new_noise=0,queries_unchanged=True,source_Y_read_in_inference=False,
        stages=['RAW','BASE_POST','CLEAN_FINAL_POST'],scorer='Original unchanged fixed-query full score',
        pipeline='Euler50 -> single official Bench D0 -> restore C/seam -> GEO_C/source winding -> HINGE_OPP015/ALPHA/HOLE'))
    print('EVALUATION_CPU_GATE_PASS16')

if __name__=='__main__':main()
