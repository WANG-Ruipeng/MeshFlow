"""Read-only CPU checks of registered inputs, mask contracts and GPU imports.
No head/generator/denoiser construction, forward, backward, score or CUDA call.
"""
import ast, os
from pathlib import Path
import numpy as np
from .common import OUT, sha
from . import panel as P

def run():
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(encoding='utf8'))
    check=P.validate_assets()
    previous=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']='0'
    from . import sample,post_d0
    from r2.pilots.bench_control_transfer_v1 import sample as S,train as T
    from r2.pilots.bench_parts_convergence_v1 import data as D,postprocess as legacy
    from r2.pilots.seam_slot_c200_v1 import generate as G,postprocess as geometry
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0' and not torch.cuda.is_initialized()
    seen=set();rows=[]
    for q in P.requests():
        if q['condition_id'] in seen:continue
        seen.add(q['condition_id']);core,source=sample.source_assets(q)
        edges,_=D.edge_features(core['C_vertices'],core['C_faces_original'],q['N'])
        assert len(edges)==q['boundary_edges']>0
        lay=D.inference_layout(**core,occupied=np.zeros(len(edges),bool))
        assert not lay['occupied'].any() and len(lay['seam_slots'])==0
        assert np.all(lay['anchor_ids'][q['K']:] == -1)
        assert np.array_equal(lay['anchor_ids'][:q['K']],core['C_faces_model'])
        assert np.array_equal(lay['bulk_slots'],np.arange(q['K'],q['N']))
        assert lay['free_coordinate_count']==9*(q['N']-q['K'])
        # Validate the zero-seam decode schema with the actual saved noise, without model generation.
        x=np.zeros((q['N'],3,3),np.float32);x[:q['K']]=core['C_vertices'][core['C_faces_model']]
        x[q['K']:]=P.arrays(q['noise_npz'])['z']
        mesh=G.raw_arrays(x,lay,np.full(len(edges),-1,np.float32),source)
        assert not mesh['raw_seam_slot_mask'].any() and not mesh['retained_seam_mask'].any()
        assert np.array_equal(mesh['C'],source['C_original'])
        assert mesh['raw'].shape==(q['N'],3,3)
        rows.append(dict(condition_id=q['condition_id'],N=q['N'],K=q['K'],boundary_edges=len(edges),native_free_scalars=lay['free_coordinate_count'],native_head_calls=0,native_seam_slots=0,source_restoration='PASS'))
    assert len(rows)==12 and not torch.cuda.is_initialized()
    if previous is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
    else:os.environ['CUDA_VISIBLE_DEVICES']=previous
    result=dict(status='PASS',panel_checks=check,layout_checks=rows,CUDA_initialized=False,
        sampling_GPU_imports_preserve_CUDA_visibility=True,model_constructions=0,model_forwards=0,
        GPU_calls=0,scoring_calls=0,synthetic_outputs_saved=0,
        checked_code={str(p):sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ('sample.py','post_d0.py','evaluate.py','collect.py','gallery.py','eval_runtime.py','panel.py','evaluation_cpu_checks.py')})
    P.P.immutable(OUT/'evaluation_cpu_gate.json',result);return result

if __name__=='__main__':
    r=run();print({k:v for k,v in r.items() if k not in ('checked_code','panel_checks','layout_checks')})
