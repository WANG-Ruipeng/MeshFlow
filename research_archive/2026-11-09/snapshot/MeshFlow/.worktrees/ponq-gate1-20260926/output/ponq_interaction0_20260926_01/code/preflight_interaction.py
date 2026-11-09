"""Short synthetic/resume checks only; never saves a trained experimental model."""
import argparse,copy,hashlib,json,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from fm_model import DeepSetsVelocity,total_velocity
from interaction_model import InteractionVelocity,ARMS
from train_fm_pilot import BalancedSchedule,coupled_batch,sha
from train_interaction import initialize_branch,optimizer_state_fingerprint

FORWARD_ATOL=3e-5
FORWARD_RTOL=3e-5


def batch_hash(batch):
    noise,times,target,ids=batch
    return hashlib.sha256(noise.numpy().tobytes()+times.numpy().tobytes()+target.numpy().tobytes()+np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()


def original_data_stream(parent):
    saved=parent['rng']['balanced_schedule'];schedule=BalancedSchedule(saved['count'],0)
    schedule.order=copy.deepcopy(saved['order']);schedule.cursor=saved['cursor'];schedule.cycles=saved['cycles']
    schedule.rng.bit_generator.state=copy.deepcopy(saved['numpy_generator_state'])
    generator=torch.Generator(device='cpu');generator.set_state(parent['rng']['batch_generator_cpu'].cpu())
    return schedule,generator


def main(args):
    started=time.perf_counter();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    with np.load(args.data,allow_pickle=False) as z:states=z['states'].copy();mean=z['mean'].copy();scale=z['scale'].copy()
    devices=['cpu'] if args.device=='cpu' else ['cpu',args.device]
    checks=[];details={}
    def check(name,condition,**record):
        checks.append(dict(name=name,passed=bool(condition),**record))
        if not condition:raise AssertionError(name)
    try:
        check('frozen_data_hash',sha(args.data)==parent['protocol']['data_sha256'])
        if args.device!='cpu':check('CUDA_available',torch.cuda.is_available())
        if torch.cuda.is_available():torch.cuda.init()
        cpu_before=torch.get_rng_state().clone();cuda_before=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
        models=[InteractionVelocity(a) for a in ARMS]
        check('module_construction_does_not_consume_global_CPU_RNG',torch.equal(cpu_before,torch.get_rng_state()))
        check('module_construction_does_not_consume_global_CUDA_RNG',all(torch.equal(a,b) for a,b in zip(cuda_before,torch.cuda.get_rng_state_all())) if cuda_before else True)
        counts={m.arm:m.parameter_counts() for m in models};details['parameter_counts']=counts
        check('frozen_added_parameter_counts',[counts[a]['added'] for a in ARMS]==[0,66176,66304])
        check('original_parameter_names_preserved',all([n for n,p in m.named_parameters() if not n.startswith('interaction.')]==list(parent['model_state_dict']) for m in models))
        rng=torch.Generator().manual_seed(91031);z=torch.randn((2,64,22),generator=rng);t=torch.tensor([.23,.81]);perm=torch.randperm(64,generator=rng)
        active_states={};cpu_outputs={};next_hashes={}
        for arm in ARMS:
            model,opt,schedule,generator,audit=initialize_branch(parent,arm,'cpu')
            check(arm+'_old_and_new_optimizer_mapping',audit['passed'])
            schedule2,generator2=original_data_stream(parent)
            hashes=[]
            for _ in range(3):
                a=coupled_batch(states,mean,scale,schedule,generator);b=coupled_batch(states,mean,scale,schedule2,generator2)
                hashes.append(batch_hash(a));check(arm+'_continuation_batch_'+str(len(hashes)),batch_hash(a)==batch_hash(b))
            next_hashes[arm]=hashes
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(91032)
                if arm=='B1':torch.nn.init.normal_(model.interaction.linear2.weight,std=.015);torch.nn.init.normal_(model.interaction.linear2.bias,std=.015)
                if arm=='B2':torch.nn.init.normal_(model.interaction.attention.out_proj.weight,std=.015);torch.nn.init.normal_(model.interaction.attention.out_proj.bias,std=.015)
            active_states[arm]=copy.deepcopy(model.state_dict())
        check('three_arms_continue_identical_batch_streams',next_hashes['B0']==next_hashes['B1']==next_hashes['B2']);details['first_three_continued_batch_sha256']=next_hashes
        for device in devices:
            with torch.random.fork_rng(devices=[]):baseline=DeepSetsVelocity(128).to(device)
            baseline.load_state_dict(parent['model_state_dict']);baseline.eval()
            zz=z.to(device);tt=t.to(device);pp=perm.to(device)
            with torch.no_grad():base=baseline(zz,tt)
            for arm in ARMS:
                model,_,_,_,audit=initialize_branch(parent,arm,device);model.eval()
                with torch.no_grad():value,hidden=model.forward_with_hidden(zz,tt)
                check(arm+'_'+device+'_initial_function_exact',torch.equal(value,base),max_abs=float((value-base).abs().max()))
                model.load_state_dict(active_states[arm])
                with torch.no_grad():value,hidden=model.forward_with_hidden(zz,tt);reordered,hperm=model.forward_with_hidden(zz[:,pp],tt)
                check(arm+'_'+device+'_active_block_permutation_equivariance',torch.allclose(reordered,value[:,pp],atol=FORWARD_ATOL,rtol=FORWARD_RTOL)
                    and torch.allclose(hperm,hidden[:,pp],atol=FORWARD_ATOL,rtol=FORWARD_RTOL),
                    max_velocity_abs=float((reordered-value[:,pp]).abs().max()),max_hidden_abs=float((hperm-hidden[:,pp]).abs().max()))
                if device=='cpu':cpu_outputs[arm]=(value.cpu(),hidden.cpu())
                else:
                    check(arm+'_CPU_GPU_active_block_consistency',torch.allclose(value.cpu(),cpu_outputs[arm][0],atol=FORWARD_ATOL,rtol=FORWARD_RTOL)
                        and torch.allclose(hidden.cpu(),cpu_outputs[arm][1],atol=FORWARD_ATOL,rtol=FORWARD_RTOL),
                        max_velocity_abs=float((value.cpu()-cpu_outputs[arm][0]).abs().max()),max_hidden_abs=float((hidden.cpu()-cpu_outputs[arm][1]).abs().max()))
                model.train();value,hidden=model.forward_with_hidden(zz,tt)
                gradient_norms=[]
                for lo,hi in [(0,3),(3,6),(6,22)]:
                    gradient=torch.autograd.grad(value[...,lo:hi].square().mean(),hidden,retain_graph=True)[0]
                    gradient_norms.append(float(gradient.norm()))
                check(arm+'_'+device+'_returned_hidden_is_actual_decoder_graph',all(np.isfinite(x) and x>0 for x in gradient_norms),gradient_norms=gradient_norms)
            # Independent old-constructor/load_state_dict continuation vs new B0.
            fresh,opt,schedule,generator,_=initialize_branch(parent,'B0',device)
            with torch.random.fork_rng(devices=[]):old=DeepSetsVelocity(128).to(device)
            old.load_state_dict(parent['model_state_dict']);old_opt=torch.optim.AdamW(old.parameters(),lr=3e-4,weight_decay=1e-4)
            old_opt.load_state_dict(copy.deepcopy(parent['optimizer_state_dict']))
            old_schedule,old_generator=original_data_stream(parent)
            a=coupled_batch(states,mean,scale,schedule,generator);b=coupled_batch(states,mean,scale,old_schedule,old_generator)
            check('B0_'+device+'_old_continuation_same_batch',batch_hash(a)==batch_hash(b))
            step_losses=[]
            for m,o,batch in [(old,old_opt,b),(fresh,opt,a)]:
                noise,times,target,_=batch;noise=noise.to(device);times=times.to(device);target=target.to(device)
                m.train();o.zero_grad(set_to_none=True)
                zz=(1-times)*noise+times*target;pred,_=total_velocity(m,zz,times,torch.tensor(mean,dtype=torch.float32,device=device),torch.tensor(scale,dtype=torch.float32,device=device),'direct')
                loss=(pred-(target-noise)).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step();step_losses.append(float(loss.detach()))
            same_weights=all(torch.equal(v,fresh.state_dict()[k]) for k,v in old.state_dict().items())
            old_params=dict(old.named_parameters());fresh_params=dict(fresh.named_parameters())
            same_optimizer=all(optimizer_state_fingerprint(old_opt.state[p])==optimizer_state_fingerprint(opt.state[fresh_params[name]]) for name,p in old_params.items())
            check('B0_'+device+'_one_step_old_continuation_exact',same_weights and same_optimizer and step_losses[0]==step_losses[1],
                weights_exact=same_weights,optimizer_exact=same_optimizer,losses=step_losses)
        passed=True
    except Exception as exc:
        passed=False;details['error']=dict(exception=repr(exc),traceback=traceback.format_exc())
    source_files=[Path(__file__),Path(__file__).with_name('interaction_model.py'),Path(__file__).with_name('train_interaction.py'),Path(__file__).with_name('fm_model.py'),Path(__file__).with_name('train_fm_pilot.py')]
    record=dict(passed=passed,checks=checks,details=details,check_count=len(checks),devices=devices,seconds=time.perf_counter()-started,
        tolerances=dict(forward_atol=FORWARD_ATOL,forward_rtol=FORWARD_RTOL,initial_same_function='bitwise exact on each device',B0_one_update='bitwise exact weights and optimizer state on each device'),
        checkpoint_sha256=sha(args.checkpoint),data_sha256=sha(args.data),
        source_files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in source_files],
        scope='Short initialization, active-block permutation, CPU/GPU, data-stream and one-update recovery checks. Synthetic nonzero residual weights and one-step models are discarded; no formal experiment training.')
    (out/'RESULT.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2,allow_nan=False),flush=True)
    return passed


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['checkpoint','data','out']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    raise SystemExit(0 if main(args) else 1)