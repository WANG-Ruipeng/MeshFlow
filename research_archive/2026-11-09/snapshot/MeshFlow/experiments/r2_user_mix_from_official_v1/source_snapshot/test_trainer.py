"""CPU contract checks; no model construction/forward, CUDA, OT, or updates."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import trainer as t


def samples(step=1):
    rows=[]
    for i in range(8):
        n=128+i
        kind='USER' if i<6 else 'OLD'
        ratio=None if kind=='USER' else .2 if i==6 else .4
        k=1+i if ratio is None else round(n*ratio)
        clean=i<3 or i==6
        a=np.zeros((n,9),np.float32)
        rows.append(dict(object_id=str(i),y=np.asarray(n,np.int64),K=k,
            source_kind=kind,arm=t.ARM,step=step,global_step=step,
            alpha=1. if clean else .95,ratio=k/n if ratio is None else ratio,
            lambda_value=.25,free_coordinate_count=9*(n-k),
            known_mask=np.arange(n)<k,valid_mask=np.ones(n,bool),
            xt=a.copy(),u=a.copy(),context=a.copy()))
    return rows


def fixture_payload(step=6000):
    """Small CPU tensors isolate checkpoint envelope validation from the model."""
    g={'fixture_tensor':torch.tensor([1.,2.],dtype=torch.float32)}
    opt={'state':{},'param_groups':[]}
    rng={'fixture_rng':torch.tensor([1,2],dtype=torch.uint8)}
    stream={'batch_index':step,'namespace':'test-only'}
    return dict(schema=t.SCHEMA,experiment=t.EXPERIMENT,arm=t.ARM,
        global_step=step,update_in_progress=False,protocol=t.protocol(),
        protocol_sha256=t.json_hash(t.protocol()),model_config=copy.deepcopy(t.MODEL_CONFIG),
        generator_state=g,generator_state_sha256=t.state_hash(g),
        optimizer=opt,optimizer_state_sha256=t.base.tree_hash(opt),
        rng=rng,rng_sha256=t.base.tree_hash(rng),
        stream_state=stream,stream_state_sha256=t.json_hash(stream))


class MixedInputTests(unittest.TestCase):
    def test_variable_small_k_and_whole_batch_denominator(self):
        rows=samples()
        expected=9*sum(int(r['y'])-r['K'] for r in rows)
        self.assertEqual(t.validate_samples(rows,1),expected)
        self.assertNotEqual(rows[0]['ratio'],.2)
        user_coordinates=sum(r['free_coordinate_count'] for r in rows if r['source_kind']=='USER')
        self.assertGreater(user_coordinates/expected,.75)

    def test_wrong_old_k_is_not_accepted_as_a_ratio_label(self):
        rows=samples();r=rows[6];r['K']+=1
        r['known_mask']=np.arange(int(r['y']))<r['K']
        r['free_coordinate_count']=9*(int(r['y'])-r['K'])
        with self.assertRaises(ValueError):t.validate_samples(rows,1)

    def test_ratio_mix_not_silently_changed(self):
        rows=samples();rows[5]['source_kind']='OLD'
        with self.assertRaises(ValueError):t.validate_samples(rows,1)

    def test_category_clean_balance_not_only_batch_balance(self):
        rows=samples();rows[0]['alpha']=.95;rows[7]['alpha']=1.
        self.assertEqual(sum(r['alpha']==1. for r in rows),4)
        with self.assertRaises(ValueError):t.validate_samples(rows,1)

    def test_old_continuation_clock_cannot_enter_new_step0_training(self):
        rows=samples();rows[0]['global_step']=6001
        with self.assertRaises(ValueError):t.validate_samples(rows,1)

    def test_duplicate_parent_rejected(self):
        rows=samples();rows[1]['object_id']=rows[0]['object_id']
        with self.assertRaises(ValueError):t.validate_samples(rows,1)

    def test_c_signed_zero_mutation_rejected(self):
        rows=samples();rows[0]['xt'][0,0]=-0.
        with self.assertRaises(ValueError):t.validate_samples(rows,1)

    def test_nonfinite_and_corrupt_denominator_rejected(self):
        for mode in ('nan','denominator','hybrid'):
            with self.subTest(mode=mode):
                rows=samples()
                if mode=='nan':rows[0]['u'][0,0]=np.nan
                elif mode=='denominator':rows[0]['free_coordinate_count']+=9
                else:rows[0]['lambda_value']=0.
                with self.assertRaises(ValueError):t.validate_samples(rows,1)


class LedgerTests(unittest.TestCase):
    def ledger(self,path,limits=None):
        return t.Ledger(path,limits or t.LIMITS,t.ARM,'cpu_test',synchronize=False)

    def complete_update(self,ledger,step):
        for kind in ('train_forward','train_backward'):
            for index in range(8):ledger.call(kind,lambda:None,{'step':step,'microbatch':index})
        ledger.call('optimizer_update',lambda:None,{'step':step})

    def test_fresh_and_completed_cursor_with_persistent_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'calls.jsonl';ledger=self.ledger(p)
            ledger.require_exact_cursor(0)
            self.complete_update(ledger,1)
            reloaded=self.ledger(p);reloaded.require_exact_cursor(1)
            with self.assertRaises(ValueError):reloaded.require_exact_cursor(0)
            with self.assertRaises(ValueError):reloaded.require_exact_cursor(2)

    def test_pending_forward_prevents_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'calls.jsonl'
            p.write_text(json.dumps(dict(event='CALL_BEGIN',call_id=1,kind='train_forward',
                arm=t.ARM,attempt_id='abrupt_exit',context={'step':1}))+'\n',encoding='utf8')
            ledger=self.ledger(p)
            with self.assertRaises(ValueError):ledger.require_exact_cursor(0)
            with self.assertRaises(RuntimeError):ledger.call('train_forward',lambda:None,{'step':1})

    def test_failed_attempt_consumes_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'calls.jsonl';ledger=self.ledger(p,{'train_forward':1})
            def failed():raise FloatingPointError('deliberate CPU negative control')
            with self.assertRaises(FloatingPointError):ledger.call('train_forward',failed,{'step':1})
            ledger=self.ledger(p,{'train_forward':1})
            self.assertEqual(ledger.attempted['train_forward'],1)
            self.assertEqual(ledger.failed['train_forward'],1)
            with self.assertRaises(RuntimeError):ledger.call('train_forward',lambda:None,{'step':1})

    def test_optimizer_skip_or_repeated_step_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=self.ledger(Path(tmp)/'calls.jsonl');self.complete_update(ledger,2)
            with self.assertRaises(ValueError):ledger.require_exact_cursor(1)


class CheckpointTests(unittest.TestCase):
    def test_hashes_detect_generator_optimizer_rng_and_stream_changes(self):
        # Only the architecture tensor-key check is replaced, not envelope/hash logic.
        with mock.patch.object(t,'_check_generator',return_value=None):
            p=fixture_payload();self.assertIs(t.validate_payload(p),p)
            for field in ('generator','optimizer','rng','stream'):
                with self.subTest(field=field):
                    q=copy.deepcopy(p)
                    if field=='generator':q['generator_state']['fixture_tensor'][0]+=1
                    elif field=='optimizer':q['optimizer']['state']['unexpected']={}
                    elif field=='rng':q['rng']['fixture_rng'][0]+=1
                    else:q['stream_state']['namespace']='changed'
                    with self.assertRaises(ValueError):t.validate_payload(q)

    def test_wrong_endpoint_range_and_stream_cursor_rejected(self):
        with mock.patch.object(t,'_check_generator',return_value=None):
            for mode in ('negative','over_budget','stream_ahead','update_in_progress','wrong_protocol'):
                with self.subTest(mode=mode):
                    p=fixture_payload()
                    if mode=='negative':p['global_step']=-1
                    elif mode=='over_budget':p['global_step']=6001
                    elif mode=='stream_ahead':p['stream_state']['batch_index']=6001
                    elif mode=='update_in_progress':p['update_in_progress']=True
                    else:p['protocol']['USER_per_batch']=2
                    with self.assertRaises(ValueError):t.validate_payload(p)

    def test_fixed_final_export_rejects_intermediate_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(t,'_check_generator',return_value=None):
            full=Path(tmp)/'mid.pt';torch.save(fixture_payload(500),full)
            with self.assertRaises(ValueError):t.export_generator(full,Path(tmp)/'export.pt')
            self.assertFalse((Path(tmp)/'export.pt').exists())

    def test_checkpoint_write_refuses_existing_final_or_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'end.pt';p=fixture_payload()
            receipt=t.save(path,p)
            self.assertEqual(receipt['sha256'],t.digest(path))
            with self.assertRaises(FileExistsError):t.save(path,p)
            path.with_name(path.name+'.partial').write_bytes(b'unfinished')
            with self.assertRaises(FileExistsError):t.save(path,p,replace=True)


class InitializationTests(unittest.TestCase):
    def test_official_wrong_hash_rejected_before_model_constructed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'not_official.pt';path.write_bytes(b'not a checkpoint')
            with mock.patch.object(t.base,'build_model',side_effect=AssertionError('model construction forbidden')):
                with self.assertRaises(ValueError):t.recipe.initialize_official(path,device='cpu')

    def test_fresh_adamw_and_step_mismatch_without_any_update(self):
        class TinyParameters:
            training=False
            readout_mode='none'
            def __init__(self):self.p=torch.nn.Parameter(torch.ones(2,dtype=torch.float32))
            def parameters(self):return iter([self.p])
            def named_parameters(self):return iter([('cpu_contract_parameter',self.p)])
        model=TinyParameters();opt=t.base.make_optimizer(model,None)
        t.base.validate_optimizer(model,None,opt,'COMMON',0)
        self.assertEqual(len(opt.state),0)
        with self.assertRaises(ValueError):t.base.validate_optimizer(model,None,opt,'COMMON',1)
        opt.state[model.p]=dict(step=torch.tensor(6000.),exp_avg=torch.zeros_like(model.p),exp_avg_sq=torch.ones_like(model.p))
        t.base.validate_optimizer(model,None,opt,'COMMON',6000)
        with self.assertRaises(ValueError):t.base.validate_optimizer(model,None,opt,'COMMON',0)

    def test_no_cuda_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
