"""CPU-only lossless Edge-only endpoint compaction before authorized cleanup."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,gc,time
from pathlib import Path
import torch
from native_t1.artifacts import file_sha256,state_sha256
from experiments.mf_h1_local0.edge_factorial_pilot.checkpoint import load,validate

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'native_t1/maintenance/baseline_cleanup_20261001_163503'
SOURCE=ROOT/'native_t1/runs/edge_factorial_20261001_123000/Edge_only/additional_step500_cumulative_step3000.pt'
DEST=ROOT/'native_t1/checkpoints/ablations/edge_only_cumulative3000.pt'
EXPECTED_FILE='fcef320b370311c5893f5a0f6e4ea30d81b68e19ceda305116e9c2deb6e357ae'
EXPECTED_STATE='b9ed4bfd5916d9731daec5afd9732b8bdfff5dbd81749dfda4c7ec3e12f74c7a'

def main():
    tick=time.perf_counter();torch.set_num_threads(4)
    if DEST.exists() or (OUT/'checkpoint_compaction.json').exists():raise FileExistsError('No migration overwrite')
    assert file_sha256(SOURCE)==EXPECTED_FILE
    old=torch.load(SOURCE,map_location='cpu',mmap=True,weights_only=True);validate(old)
    assert old['model_state_sha256']==EXPECTED_STATE and old['cumulative_updates']==3000
    payload={k:v for k,v in old.items() if k!='optimizer'}
    payload.update(optimizer={},optimizer_state_removed=True,
        cleanup_provenance=dict(original_path=str(SOURCE),original_sha256=EXPECTED_FILE,
          operation='Exact learned tensors and input RNG; optimizer momentum removed; no model execution'))
    validate(payload)
    temp=DEST.with_suffix('.pt.partial')
    if temp.exists():raise FileExistsError(temp)
    torch.save(payload,temp);temp.replace(DEST)
    model,audit=load(DEST,device='cpu')
    assert audit['strict'] and not audit['missing_keys'] and not audit['unexpected_keys']
    assert state_sha256(model)==EXPECTED_STATE
    current=model.state_dict();assert current.keys()==old['model'].keys()
    assert all(torch.equal(current[k].view(torch.uint8),old['model'][k].view(torch.uint8)) for k in current)
    restored=torch.load(DEST,map_location='cpu',mmap=True,weights_only=True)
    assert restored['stream_start']==old['stream_start'] and restored['stream_state']==old['stream_state']
    assert restored['optimizer']=={} and restored['optimizer_state_removed'] is True
    assert file_sha256(SOURCE)==EXPECTED_FILE and not torch.cuda.is_initialized()
    record=dict(status='PASS',source=dict(path=str(SOURCE),bytes=SOURCE.stat().st_size,sha256=EXPECTED_FILE),
        destination=dict(path=str(DEST),bytes=DEST.stat().st_size,sha256=file_sha256(DEST)),
        model_state_sha256=EXPECTED_STATE,every_tensor_bitwise_equal=True,input_stream_start_and_end_identical=True,
        schema_unchanged=payload['schema'],optimizer_momentum_removed=True,strict_CPU_load=audit,
        model_forwards=0,model_backwards=0,optimizer_updates=0,generations=0,CUDA_initialized=False,
        seconds=time.perf_counter()-tick)
    (OUT/'checkpoint_compaction.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
