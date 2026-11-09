"""One-time local G0 export for cleanup; no forward, training or sampling."""
import copy,json,time
from pathlib import Path
from argparse import Namespace
import torch
from native_t1.artifacts import file_sha256,atomic_json,state_sha256
from native_t1.geometry_checkpoint import SCHEMA,validate_geometry_checkpoint,load_geometry_checkpoint
from native_t1.portable_checkpoint import NATIVE_META,config_sha256
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.data import load_archive
from native_t1.train_cli import make_stream
torch.set_num_threads(4)
repo=Path('/mnt/e/MeshFlow')
run=repo/'native_t1/runs/geometry_rep_alignment_20260930_205100'
out=repo/'output/tgeo_loss_cleanup_20260930_222130'
out.mkdir(parents=True,exist_ok=True)
r=json.loads((run/'results.json').read_text())
source=Path(r['arms']['G0']['checkpoints']['500']['path'])
expected=r['arms']['G0']['checkpoints']['500']
assert file_sha256(source)==expected['sha256']
p=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
assert p['arm']=='G0' and p['aux_projector_state'] is None and p['aux_scale_head_state'] is None
assert p['cumulative_updates']==2500 and p['additional_updates']==500
assert p['generator_state_sha256']=='6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392'
assert not p['optimizer_step_in_progress'] and not p['batch_in_progress']
destination=repo/'native_t1/checkpoints/tgeo_g0_cumulative2500.pt'
# Existing export is revalidated, never overwritten.
destination.parent.mkdir(parents=True,exist_ok=True)
payload=dict(schema=SCHEMA,native_meta=dict(NATIVE_META),context=copy.deepcopy(p['context']),
 model=p['generator_state'],optimizer={},stream_state=copy.deepcopy(p['stream_state']),
 stream_seed=1010,stream_start_batch_index=500,stream_start_sample_index=4000,
 input_sha256=r['assets']['inputs']['sha256'],base_identity=copy.deepcopy(r['arms']['G0']['load']),
 config_sha256=p['config_sha256'],base_cumulative_updates=2000,completed_updates=500,cumulative_updates=2500,
 loss='masked_free_fm',num_faces=112,train_K=[4,8,12],optimizer_reset_at_start=True,
 optimizer_step_in_progress=False,batch_in_progress=False,model_state_sha256=p['generator_state_sha256'],
 export=dict(kind='generator_and_input_rng_for_fresh_optimizer',source_path=str(source),
 source_schema=p['schema'],source_file_sha256=expected['sha256'],source_arm='G0',
 source_state_sha256=p['generator_state_sha256'],optimizer_state_present=False,
 optimizer_policy='Intentionally stripped. Public continuation always constructs fresh AdamW; exact optimizer resume is unavailable.',
 auxiliary_modules_present=False,tensor_arithmetic_performed=False,training_updates_added=0))
validate_geometry_checkpoint(payload,config_sha256(DEFAULT_CONFIG))
if not destination.exists():
 torch.save(payload,destination)
else:
 existing=torch.load(destination,map_location='cpu',weights_only=True,mmap=True)
 validate_geometry_checkpoint(existing,config_sha256(DEFAULT_CONFIG))
 assert existing['export']==payload['export'] and existing['model_state_sha256']==payload['model_state_sha256']
 del existing
del payload,p
model,audit=load_geometry_checkpoint(destination,device='cpu')
assert audit['state_sha256']==expected['state_sha256']
assert audit['parameters']==130401155 and audit['strict']
assert not torch.cuda.is_initialized()
archive=load_archive(r['assets']['inputs']['path'],r['assets']['inputs']['sha256'])
stream,mode=make_stream(Namespace(continue_stream=True,init_t1=None,init_geo=destination,seed=None),archive)
assert stream.state_dict()==r['arms']['G0']['final_stream_state']
result=dict(status='PASS',source=expected,portable_checkpoint=dict(path=str(destination),bytes=destination.stat().st_size,
 sha256=file_sha256(destination),state_sha256=audit['state_sha256']),strict_public_loader=audit,
 public_training_input_rng_restore=mode,stream_batch=stream.batch_index,stream_sample=stream.sample_index,
 model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,CUDA_initialized=torch.cuda.is_initialized(),
 optimizer_state_stripped=True,source_results_sha256=file_sha256(run/'results.json'))
atomic_json(out/'baseline_export.json',result)
print(json.dumps(result,indent=2),flush=True)
