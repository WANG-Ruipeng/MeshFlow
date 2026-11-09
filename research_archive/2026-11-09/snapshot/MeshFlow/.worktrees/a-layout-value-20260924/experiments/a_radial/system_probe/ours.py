"""CUDA resume gate using the unmodified handoff ShapeAE/FM trainer."""
from argparse import ArgumentParser,Namespace
from pathlib import Path
import json,time
import numpy as np
import torch
from .. import train_ours
from ..io_utils import file_hash,new_dir,write_json
from .official import configure,exact_same

class DeliberateCheckpointStop(Exception): pass

def gate(data,out):
    data=Path(data).resolve();out=new_dir(out)
    require_files=['radial_train.npz','DATASET.json']
    hashes={n:file_hash(data/n) for n in require_files}
    source=Path(train_ours.__file__);source_sha=file_hash(source)
    configure(10)
    assert torch.cuda.is_available()
    with np.load(data/'radial_train.npz',allow_pickle=False) as z:
        assert z['logr'].shape==(16,2050) and len(set(z['uids'].tolist()))==16
        assert np.isfinite(z['logr']).all()
    reports={};ae=None;started=time.perf_counter()
    for stage in ('ae','fm'):
        args=Namespace(stage=stage,data=str(data),out=str(out/stage/'whole'),ae=ae,resume=None,
            seed=10,device='cuda',threads=1,steps=4,batch=32 if stage=='ae' else 128,
            lr=6e-4 if stage=='ae' else 1e-3,checkpoint_every=2,log_every=1)
        train_ours.run(args)
        full_path=Path(args.out)/'step00000004.pt'
        args.out=str(out/stage/'first')
        original=train_ours.atomic_torch
        def stop(path,state):
            original(path,state)
            if state['step']==2:raise DeliberateCheckpointStop()
        train_ours.atomic_torch=stop
        try:
            train_ours.run(args)
            raise AssertionError('Deliberate stop did not execute')
        except DeliberateCheckpointStop:
            write_json(Path(args.out)/'EXPECTED_STOP.json',{'step':2,'saved_full_state':True})
        finally:train_ours.atomic_torch=original
        args.resume=str(Path(args.out)/'step00000002.pt');args.out=str(out/stage/'resumed')
        train_ours.run(args)
        a=torch.load(full_path,map_location='cpu',weights_only=True)
        b=torch.load(Path(args.out)/'step00000004.pt',map_location='cpu',weights_only=True)
        exact_same(a,b,stage)
        assert a['step']==4 and a['scheduler']['last_epoch']==4
        assert all(float(v['step'])==4 for v in a['optimizer']['state'].values())
        model=train_ours.ShapeAE() if stage=='ae' else train_ours.ShapeFM()
        model.load_state_dict(a['model'],strict=True)
        reports[stage]={'status':'PASS','complete_checkpoint_bitwise_equal':True,
            'continuous_updates':4,'split_updates':[2,2],'actual_optimizer_updates':8,
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'continuous_checkpoint_sha256':file_hash(full_path),'resume_checkpoint_sha256':file_hash(Path(args.out)/'step00000004.pt')}
        if stage=='ae':ae=str(full_path)
    assert file_hash(source)==source_sha
    assert hashes=={n:file_hash(data/n) for n in require_files}
    result={'status':'PASS','scope':'CUDA correctness only; not scientific fitting qualification',
        'original_trainer_sha256':source_sha,'data_hashes':hashes,'stages':reports,
        'actual_optimizer_updates':16,'device':torch.cuda.get_device_name(0),
        'full_checkpoint_keys_compared':['model','optimizer','scheduler','config','step','rng','batch_rng','draw_queue'],
        'source_and_data_unchanged':True,'elapsed_seconds':time.perf_counter()-started,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32}
    write_json(out/'GATE.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();gate(a.data,a.out)