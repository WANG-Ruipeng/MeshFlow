"""CPU-only final-selection, source pairing, failure-ledger and five-row gates."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from ..io_utils import file_hash,array_hash,write_json
from ..registry import load_attempts
from . import sample,finish
from .test_train import state_fixture
from .train import FORMAL_SCOPE,PROBE_SCOPE


def final_fixture():
    state=state_fixture(step=50000);state['rng']={'torch_cuda':[torch.zeros(1,dtype=torch.uint8)]}
    state['sampler_rng']=torch.zeros(2,dtype=torch.uint8)
    identity={k:v for k,v in state['identity'].items() if k!='seed'}
    done={'status':'FIXED_BUDGET_COMPLETE_UNQUALIFIED','scope':FORMAL_SCOPE,'step':50000,'seed':10,
        'primary_weight':'ema','protocol_sha256':identity['protocol_sha256'],'parent_exposures':[50000]*16}
    return state,identity,done


def test_only_formal_final50000_ema_and_same_seed_are_accepted():
    state,identity,done=final_fixture();sample.validate_final(state,done,identity,state['config'],10)
    for key,value in [('step',100),('is_predeclared_final',False),('scope',PROBE_SCOPE)]:
        with pytest.raises(AssertionError):sample.validate_final({**state,key:value},done,identity,state['config'],10)
    for key,value in [('primary_weight','model'),('seed',11),('step',49999),('protocol_sha256','another_run')]:
        with pytest.raises(AssertionError):sample.validate_final(state,{**done,key:value},identity,state['config'],10)
    assert not torch.cuda.is_initialized()


def test_incomplete_counters_rng_seed_identity_and_bytes_are_rejected():
    state,identity,done=final_fixture()
    wrong={**state,'ema_updates':49999}
    with pytest.raises(AssertionError,match='counters'):sample.validate_final(wrong,done,identity,state['config'],10)
    wrong={**state,'rng':{'torch_cuda':[]}}
    with pytest.raises(AssertionError,match='RNG'):sample.validate_final(wrong,done,identity,state['config'],10)
    wrong={**state,'identity':{**state['identity'],'seed':11}}
    with pytest.raises(AssertionError,match='resume_identity'):sample.validate_final(wrong,done,identity,state['config'],10)
    wrong={**state,'ema':{k:v.clone() for k,v in state['ema'].items()}};wrong['ema']['layer.weight'][0,0]+=1
    with pytest.raises(AssertionError,match='identity mismatch'):sample.validate_final(wrong,done,identity,state['config'],10)


def test_final_checkpoint_completion_sha_and_formal_directory_are_bound(tmp_path,monkeypatch):
    state,identity,done=final_fixture();identity.update(dataset_sha256='dataset',source_manifest_sha256='bound_source')
    state['identity']={**identity,'seed':10};path=tmp_path/'training/seed_10/invocations/0001/step00050000.pt'
    path.parent.mkdir(parents=True);torch.save(state,path)
    record={'path':str(path),'step':50000,'sha256':file_hash(path),'bytes':path.stat().st_size}
    write_json(path.parent/'checkpoints/step00050000.json',record)
    write_json(path.parent/'probes/step00050000.json',{'step':50000,'not_a_checkpoint_selection_criterion':True})
    done.update(primary_checkpoint=str(path),primary_checkpoint_sha256=file_hash(path))
    complete=tmp_path/'training/seed_10/TRAINING_COMPLETE.json';write_json(complete,done)
    cfg=deepcopy(state['config']);cfg['sample']={'sampling_method':'euler'}
    # State config intentionally remains the same object expected by this synthetic contract.
    cfg=state['config'];cfg['sample']={'sampling_method':'euler'}
    torch.save(state,path);done['primary_checkpoint_sha256']=file_hash(path);record.update(sha256=file_hash(path),bytes=path.stat().st_size)
    write_json(complete,done,replace=True);write_json(path.parent/'checkpoints/step00050000.json',record,replace=True)
    monkeypatch.setattr(sample,'contract',lambda run:({},cfg,identity,{'seeds':[10,11]},'sampling_sha'))
    result=sample.final_identity(tmp_path,10)
    assert result['weight']=='ema' and result['step']==50000 and result['ema_sha256']!=result['model_sha256']
    write_json(complete,{**done,'primary_checkpoint':str(tmp_path/'training/seed_11/invocations/0001/step00050000.pt')},replace=True)
    with pytest.raises(AssertionError,match='escapes'):sample.final_identity(tmp_path,10)
    write_json(complete,done,replace=True)
    with path.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(AssertionError,match='SHA/size'):sample.final_identity(tmp_path,10)


def generation_fixture(tmp_path,wrong_noise=False,post_full=False,fill_holes=False,duplicates=False):
    folder=tmp_path/'seed_10';generated=folder/'generated';generated.mkdir(parents=True)
    identity={'scope':sample.SCOPE,'seed':10,'step':50000,'weight':'ema','checkpoint_sha256':'fixed50000',
        'dataset_manifest_sha256':'dataset','sample_config':{'sampling_method':'euler','num_sampling_steps':50,'cfg_scale':2.,'requested_faces':512},
        'noise_seed':93610,'sampling_protocol_sha256':'sampling_protocol'}
    generator=torch.Generator(device='cpu').manual_seed(93610);noise=torch.randn(64,800,9,generator=generator).numpy()
    tokens=np.arange(512*9,dtype=np.float32).reshape(512,9)/1000
    vertices=tokens.reshape(-1,3)*(.3762*.5);faces=np.arange(1536,dtype=np.int64).reshape(512,3)
    raw=generated/'raw/00000.npz';raw.parent.mkdir();np.savez(raw,vertices=vertices,faces=faces,tokens=tokens,noise=noise[1] if wrong_noise else noise[0])
    rr=sample.base_row(identity,'raw','00000',noise[0]);rr.update(status='ok',mesh_path='raw/00000.npz',mesh_sha256=file_hash(raw),
        raw_tensor_sha256=array_hash(tokens),actual_faces=512,time_grid_points=50,observed_ode_rhs_calls=49,CFG_batch_factor=2)
    failed=sample.base_row(identity,'raw','00001',noise[1]);failed['error']='synthetic_sampler_failure_no_retry'
    write_json(generated/'raw_MANIFEST.json',{'row':'meshflow_raw','requested_samples':64,'records':[rr,rr.copy()] if duplicates else [rr,failed]})
    qp=generated/'public_quantized/00000.obj';qp.parent.mkdir();qp.write_text('synthetic immutable quantized intermediate\n')
    post=generated/'post/00000.npz';post.parent.mkdir();np.savez(post,vertices=vertices.astype(np.float64),faces=faces)
    pr=sample.base_row(identity,'post','00000',noise[0]);pr.update(status='ok',mesh_path='post/00000.npz',mesh_sha256=file_hash(post),
        source_raw_tensor_sha256=array_hash(tokens),quantized_export_sha256=file_hash(qp),actual_faces=512,paper_full_qualified=post_full)
    pr['post_config']['fill_holes']=fill_holes
    write_json(generated/'post_MANIFEST.json',{'row':'meshflow_public_post','requested_samples':64,'records':[pr]})
    return folder,identity


def test_interrupted_sampler_preserves64_slots_and_original_bytes_without_retries(tmp_path):
    folder,identity=generation_fixture(tmp_path);before=sample.inventory(folder/'generated')
    result=sample.finalize_view(folder,identity,17)
    assert result['successful']=={'raw':1,'post':1} and result['failed']=={'raw':63,'post':63}
    assert result['sampler_returncode']==17 and result['no_resampling'] and not result['paper_full_qualified']
    assert sample.inventory(folder/'generated')==before
    for arm in sample.ARMS:
        m=load_attempts(folder/'evaluation_view'/f'{arm}_MANIFEST.json')
        assert len(m['records'])==m['requested_samples']==64 and [r['sample_id'] for r in m['records']]==sample.slots()
    raw=load_attempts(folder/'evaluation_view/raw_MANIFEST.json')['records'][0]
    assert raw['observed_ode_rhs_calls']==49 and raw['time_grid_points']==50
    sample.audit_completed(folder,identity)
    with pytest.raises(FileExistsError):sample.finalize_view(folder,identity,0)


@pytest.mark.parametrize('option',['wrong_noise','duplicates','post_full','fill_holes'])
def test_mismatched_noise_duplicates_or_false_post_scope_become_explicit_failures(tmp_path,option):
    folder,identity=generation_fixture(tmp_path,**{option:True})
    result=sample.finalize_view(folder,identity,0)
    assert result['failed']['post']==64
    assert result['successful']['raw']==(0 if option in ('wrong_noise','duplicates') else 1)
    manifest=sample.read(folder/'evaluation_view/post_MANIFEST.json')
    assert all(row['paper_full_qualified'] is False and row['post_config']['fill_holes'] is False for row in manifest['records'])
    assert len(result['all_slots'])==128


def test_stale_view_or_source_and_automatic_resampling_are_rejected(tmp_path,monkeypatch):
    folder,identity=generation_fixture(tmp_path/'posttraining');sample.finalize_view(folder,identity,0)
    monkeypatch.setattr(sample,'final_identity',lambda run,seed:identity)
    monkeypatch.setattr(sample.subprocess,'Popen',lambda *a,**k:pytest.fail('Existing attempt must never be sampled again'))
    assert sample.run_seed(tmp_path,10)['successful']['raw']==1
    (folder/'generated/new_unregistered_file').write_text('changed')
    with pytest.raises(AssertionError,match='source_outputs'):sample.audit_completed(folder,identity)
    with pytest.raises(AssertionError,match='source_outputs'):sample.run_seed(tmp_path,10)


def test_original_sampler_invocation_uses_ema_checkpoint_registered_noise_and_no_hole_filling(tmp_path,monkeypatch):
    identity={'checkpoint':'registered_final50000.pt','noise_seed':93610};out=tmp_path/'posttraining/seed_10/generated'
    write_json(out.parent/'INVOCATION.json',{'identity':identity});write_json(tmp_path/'RUN_PROTOCOL.json',{'repository':'registered_repo'})
    calls=[];old_argv=list(sys.argv)
    monkeypatch.setattr(sample,'final_identity',lambda run,seed:identity)
    monkeypatch.setattr(sample,'configure',lambda seed:calls.append(('configure',seed)))
    monkeypatch.setattr(sample.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(sample.official_sample,'main',lambda:calls.append(('original_main',list(sys.argv))))
    sample.child(tmp_path,10,out)
    args=calls[1][1];assert calls[0]==('configure',10) and args[args.index('--checkpoint')+1]=='registered_final50000.pt'
    assert args[args.index('--samples')+1]=='64' and args[args.index('--noise-seed')+1]=='93610'
    assert '--public-post' in args and '--fill-holes' not in args and sys.argv==old_argv
    assert not torch.cuda.is_initialized()


def table_fixture(path):
    names=['meshflow_raw','meshflow_public_post','ours_fixed','ours_a','ours_qem']
    rows=[{'row':name,'attempts':64,'full_publication_ranking_qualified':False,'not_heldout':True,
        'paper_full_post':False if name=='meshflow_public_post' else None} for name in names]
    per=[{'row':name,'sample_id':sid,'status':'failed'} for name in names for sid in sample.slots()]
    write_json(path/'MATRIX.json',rows);write_json(path/'PER_SAMPLE.json',per);return names,rows,per


def test_five_row_table_preserves320_slots_even_if_every_output_failed(tmp_path):
    names,rows,per=table_fixture(tmp_path);assert finish.check_table(tmp_path,names)==rows
    rows[0]['attempts']=63;write_json(tmp_path/'MATRIX.json',rows,replace=True)
    with pytest.raises(AssertionError,match='denominator'):finish.check_table(tmp_path,names)
    rows[0]['attempts']=64;rows[1]['paper_full_post']=True;write_json(tmp_path/'MATRIX.json',rows,replace=True)
    with pytest.raises(AssertionError,match='paper full'):finish.check_table(tmp_path,names)
    rows[1]['paper_full_post']=False;write_json(tmp_path/'MATRIX.json',rows,replace=True)
    per[-1]['sample_id']=per[-2]['sample_id'];write_json(tmp_path/'PER_SAMPLE.json',per,replace=True)
    with pytest.raises(AssertionError,match='lost fixed slots'):finish.check_table(tmp_path,names)
