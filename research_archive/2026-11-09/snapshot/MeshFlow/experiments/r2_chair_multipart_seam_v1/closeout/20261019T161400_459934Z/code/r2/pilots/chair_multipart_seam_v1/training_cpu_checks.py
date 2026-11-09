"""Pure tensor/record/IO fixtures; no generator construction, GPU or training."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
from . import training as T

def main():
    # 1 fully known face, 1 fixed-edge seam tip, 1 bulk, 1 padded face.
    sample=dict(valid_mask=np.array([1,1,1,0],bool),anchor_ids=np.array([[0,1,2],[0,1,-1],[-1,-1,-1],[-1,-1,-1]]))
    error=np.arange(36,dtype=np.float32).reshape(4,3,3)
    value=T.stat(error,sample)
    assert value['all_free_scalar_count']==12
    assert value['seam_tip_scalar_count']==3 and value['bulk_scalar_count']==9
    assert value['all_free_velocity_sse']==float(error[1,2].sum()+error[2].sum())
    assert value['seam_tip_velocity_sse']+value['bulk_velocity_sse']==value['all_free_velocity_sse']
    row={'condition_id':'test','part_type':'BACK'};context={'condition_id':'test','local_step':1}
    merged={**row,**context,**value};assert merged['condition_id']=='test'
    class Ledger:
        pending={};failed={}
        attempted=successful={'train_forward':8,'train_backward':8,'optimizer_update':1,'head_forward':8,'head_backward':8,'head_optimizer_update':1}
        rows=[{'event':'CALL_BEGIN','kind':'optimizer_update','context':{'global_step':14001}}]
    T.require_cursor(Ledger(),1,1)
    failed=False
    try:T.require_cursor(Ledger(),0,0)
    except RuntimeError:failed=True
    assert failed
    with tempfile.TemporaryDirectory(dir=T.DEFAULT_ROOT/'training_audit') as td:
        p=Path(td);T.install_io(p);T.write(p/'a.json',{'step':0});T.write(p/'a.json',{'step':1});assert T.read(p/'a.json')=={'step':1}
    result=dict(status='PASS',SSE_count_partition=True,valid_padding_excluded=True,duplicate_record_keys_safe=True,
                exact_cursor_and_stale_rejection=True,atomic_metadata_replace=True,generator_constructions=0,generator_forwards=0,GPU_calls=0)
    T._IO=None;T.write(T.DEFAULT_ROOT/'training_audit/cpu_trainer_checks.json',result)
    print(result)
if __name__=='__main__':main()
