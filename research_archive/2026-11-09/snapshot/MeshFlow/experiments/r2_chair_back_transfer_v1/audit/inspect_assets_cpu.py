"""CPU-only field/shape/input preflight; no model construction or forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'r2/src')]
import numpy as np,torch
from r2.pilots.chair_back_transfer_v1 import worker as W
from meshflow_control.seam.slot_data import inference_layout,edge_features
from meshflow_control.source_winding import validate_source
start=time.perf_counter();gen=torch.load(W.GEN,map_location='cpu',mmap=True,weights_only=True)
head=torch.load(W.HEAD,map_location='cpu',mmap=True,weights_only=True)
assert (gen['global_step'],gen['local_step'],gen['EMA'])==(15000,2000,False)
assert len(gen['generator_state'])==194
assert all(v.dtype==torch.float32 for v in gen['generator_state'].values())
assert head['threshold_logit']==0 and head['head_config']=={'features_dim':24,'hidden_dim':32,'seed':20261016}
assert sum(v.numel() for v in head['head_state'].values())==833
rows=W.G.read(W.OUT/'eval_manifest.json')['requests'];seen=set();checks=[]
for q in rows:
    bundle,source=W.assets(q)
    if q['condition_id'] in seen:continue
    seen.add(q['condition_id']);frozen=W.G.load_npz(q['source_npz'])
    assert set(source)==set(frozen) and all(W.G.same(source[k],frozen[k]) for k in source)
    e,features=edge_features(bundle['C_vertices'],bundle['C_faces_original'],int(bundle['total_N']))
    assert features.shape==(len(e),24) and len(e)>0
    for active in (False,True):
        value=inference_layout(**bundle,occupied=np.full(len(e),active,bool))
        buf=io.BytesIO();np.savez_compressed(buf,**value);buf.seek(0)
        with np.load(buf,allow_pickle=False) as a:
            assert set(a.files)==set(value)
            for k,v in value.items():assert W.G.same(a[k],v)
        validate_source(source,bundle['C_vertices'][bundle['C_faces_model']])
    checks.append(dict(condition_id=q['condition_id'],N=int(bundle['total_N']),K=len(bundle['C_faces_model']),B=len(e),zero_active_and_all_active_npz_roundtrip=True))
result=dict(status='PASS',generator_keys=list(gen),generator_tensors=194,model_config=gen['model_config'],head_keys=list(head),head_config=head['head_config'],head_parameters=833,requests_checked=len(rows),conditions_checked=len(checks),checks=checks,source_metadata_exact=True,CUDA_initialized=torch.cuda.is_initialized(),model_constructions=0,model_forwards=0,D0_loads=0,seconds=time.perf_counter()-start)
assert not result['CUDA_initialized']
with (Path(__file__).parent/'asset_preflight_receipt.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
