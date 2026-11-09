
import json,subprocess,sys,inspect
from pathlib import Path
import torch
from native_t1.artifacts import file_sha256,atomic_json
from native_t1.geometry_checkpoint import load_geometry_checkpoint
from native_t1.model import NativeInpaintingModel
out=Path('output/tgeo_loss_cleanup_20260930_222130')
done=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','native_t1/tests','-v'],capture_output=True,text=True)
(out/'tests.log').write_text(done.stdout+'\n'+done.stderr,encoding='utf-8')
if done.returncode: raise RuntimeError(done.stdout+done.stderr)
assert 'Ran 74 tests' in done.stderr and '\nOK\n' in done.stderr
torch.set_num_threads(4)
model,audit=load_geometry_checkpoint('native_t1/checkpoints/tgeo_g0_cumulative2500.pt',device='cpu')
assert audit['state_sha256']=='6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392'
assert 'return_feature_layer' not in inspect.signature(NativeInpaintingModel.forward).parameters
assert not torch.cuda.is_initialized()
status=subprocess.check_output(['git','-c','core.autocrlf=true','status','--short'],text=True)
assert not status.strip(),status
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert head=='5ef35b421ad00ccd3a97867ce0e33e5f77f65b1c'
head_model=subprocess.check_output(['git','show','HEAD:native_t1/model.py'])
assert Path('native_t1/model.py').read_bytes()==head_model
assert not list(Path('native_t1').glob('representation_*.py'))
assert not Path('native_t1/geometry_representation.py').exists()
manifest=json.loads((out/'cleanup_manifest.json').read_text(encoding='utf-8-sig'))
assert all(t['deleted'] and not Path(t['path'].replace('E:\\MeshFlow','/mnt/e/MeshFlow').replace('\\','/')).exists() for t in manifest['targets'])
result=dict(status='PASS',tests=74,failed=0,test_log=str(out/'tests.log'),test_log_sha256=file_sha256(out/'tests.log'),
 strict_public_Geo_load_after_cleanup=audit,git_head=head,git_status=status,model_source_equals_HEAD=True,
 feature_return_extension_removed=True,retired_source_outside_package=True,
 official_model_forwards=0,official_model_backwards=0,official_optimizer_updates=0,OT_calls=0,CUDA_initialized=False,
 note='Unit tests use tiny fixtures/mocks; full real G0 only strictly loaded on CPU, never executed.')
atomic_json(out/'verification.json',result)
manifest['status']='PASS_CLEANUP_AND_VERIFICATION'
manifest['final_verification']=dict(path=str(out/'verification.json'),sha256=file_sha256(out/'verification.json'),tests=74,failed=0)
atomic_json(out/'cleanup_manifest.json',manifest)
print(json.dumps({'status':'PASS','tests':74,'git_status':status,'portable_G0':audit['state_sha256'],'CUDA_initialized':False}))
