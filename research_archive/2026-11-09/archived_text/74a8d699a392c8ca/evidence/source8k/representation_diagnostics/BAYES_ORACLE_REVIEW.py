"""Reproduce a CPU-only empirical16 Bayes-velocity review on the fixed4096 probe.

Run from the repository with:
python -B -m experiments.a_radial.system_probe.bayes_review --run RUN
Use --out-dir NEW_DIRECTORY for an independent replay without overwriting results.
No optimization, sampling replacement, or model selection is performed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from experiments.a_radial.models import ShapeAE,ShapeFM
from experiments.a_radial.io_utils import file_hash,array_hash,write_json


@torch.no_grad()
def conditional_velocity(xt,t,targets):
    """Uniform empirical target, independent standard-normal source, linear path."""
    x=xt.double();time=t.double();centres=targets.double()
    if x.ndim!=2 or centres.ndim!=2 or x.shape[1]!=centres.shape[1] or time.shape!=(len(x),):raise ValueError('Oracle shapes')
    if not bool(((time>=0)&(time<1)).all()):raise ValueError('Oracle requires 0<=t<1')
    logw=-(x[:,None,:]-time[:,None,None]*centres[None,:,:]).square().sum(-1)/(2*(1-time[:,None]).square())
    weights=torch.softmax(logw,dim=1)
    answer=(weights@centres-x)/(1-time[:,None])
    if not bool(torch.isfinite(answer).all()):raise FloatingPointError('Nonfinite conditional velocity')
    return answer


@torch.no_grad()
def make_probe(ae_state,fm_state,y,seed):
    """Identical tensors to ours_diagnose.py; preserve process CPU random state."""
    with torch.random.fork_rng(devices=[]):
        ae=ShapeAE().eval();ae.load_state_dict(ae_state)
        fm=ShapeFM().eval();fm.load_state_dict(fm_state)
        recon,codes=ae(y.cpu());targets=fm.standardize(codes)
    gen=torch.Generator(device='cpu').manual_seed(243025+seed)
    ids=torch.arange(4096)%16;x1=targets[ids];eps=torch.randn(x1.shape,generator=gen)
    t=torch.rand(4096,generator=gen);xt=(1-t[:,None])*eps+t[:,None]*x1;truth=x1-eps
    oracle=conditional_velocity(xt,t,targets)
    return {'ids':ids,'targets':targets,'eps':eps,'t':t,'xt':xt,'truth':truth,'oracle':oracle}


def probe_hash(probe):
    return array_hash(*(probe[name].numpy() for name in ['ids','targets','eps','t','xt','truth','oracle']))


@torch.no_grad()
def measure(model,probe):
    if next(model.parameters()).device.type!='cpu':raise ValueError('The diagnostic is explicitly CPU inference')
    was=model.training;model.eval()
    try:prediction=model(probe['xt'],probe['t'])
    finally:model.train(was)
    pair=(prediction-probe['truth']).square().mean(-1)
    oracle_pair=(probe['oracle']-probe['truth'].double()).square().mean(-1)
    error=(prediction.double()-probe['oracle']).square().mean(-1)
    bins=[]
    for low,high in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        selected=(probe['t']>=low)&(probe['t']<high)
        bins.append({'low':float(low),'high':float(high),'count':int(selected.sum()),
            'network_pair_MSE':float(pair[selected].mean()),'oracle_pair_MSE':float(oracle_pair[selected].mean()),
            'network_vs_conditional_mean_MSE':float(error[selected].mean())})
    return {'probe_sha256':probe_hash(probe),'probe_count':len(pair),'per_parent_count':256,
        'network_pair_MSE':float(pair.mean()),'oracle_pair_MSE':float(oracle_pair.mean()),
        'network_vs_conditional_mean_MSE':float(error.mean()),'t_bins':bins,
        'oracle_numerics':'float64 posterior softmax on the original float32 diagnostic tensors; no clipping or endpoint extension',
        'finite_sample_pair_loss_is_not_an_exact_Pythagorean_decomposition':True}


def run(run,out=None):
    run=Path(run).resolve();out=run/'representation_diagnostics' if out is None else Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    assert not torch.cuda.is_initialized(),'Read-only CPU diagnostic'
    p=json.loads((run/'RUN_PROTOCOL.json').read_text());done=json.loads((run/'TRAINING_COMPLETE.json').read_text())
    original=json.loads((run/'representation_diagnostics/RESULT.json').read_text())
    root=Path(__file__).resolve().parents[1]
    assert all(file_hash(root/name)==h for name,h in p['training_source_sha256'].items())
    assert all(file_hash(Path(p['data'])/name)==h for name,h in p['data_hashes'].items())
    assert all(file_hash(run/row['final_checkpoint'])==row['final_sha256'] for row in done['stages'])
    with np.load(Path(p['data'])/'radial_train.npz',allow_pickle=False) as z:y=torch.from_numpy(z['logr']);uids=z['uids'].tolist()
    report={'scope':'POSTHOC_CLOSED16_TRAIN_SUPPORT_REVIEW_NO_SELECTION','source_protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),
        'original_representation_RESULT_sha256':file_hash(run/'representation_diagnostics/RESULT.json'),
        'review_source_sha256':file_hash(__file__),'data_hashes':p['data_hashes'],'parent_uids':uids,
        'uniform_empirical_prior':1/16,'probe_seed_rule':'243025+training_seed','probe_count_per_seed':4096,
        'oracle_formula':'w_j=softmax(-||x-t*z_j||^2/(2*(1-t)^2)); v_star=(sum_j w_j*z_j-x)/(1-t)',
        'interpretation':'Exact conditional-mean formula for these16 training latents and independent Gaussian coupling; empirical pair-MSE includes irreducible conditional variance. This is not held-out category quality.',
        'training_updates':0,'generated_samples_changed':False,'model_selection_performed':False,'seeds':{}}
    for seed in p['seeds']:
        sp=run/f'seed{seed}';ap=sp/'ae/step00003200.pt';fp=sp/'fm/step00008000.pt'
        ac=torch.load(ap,map_location='cpu',weights_only=True);fc=torch.load(fp,map_location='cpu',weights_only=True)
        assert ac['step']==3200 and fc['step']==8000 and fc['config']['ae_sha256']==file_hash(ap)
        probe=make_probe(ac['model'],fc['model'],y,seed)
        with torch.random.fork_rng(devices=[]):model=ShapeFM().eval();model.load_state_dict(fc['model'])
        record=measure(model,probe)
        assert abs(record['network_pair_MSE']-original['seeds'][str(seed)]['fixed_velocity_probe_final_MSE'])<1e-7
        centered=probe['targets'].double()-probe['targets'].double().mean(0)
        record.update(ae_sha256=file_hash(ap),fm_sha256=file_hash(fp),
            standardized_latent_centered_singular_values=torch.linalg.svdvals(centered).tolist(),
            original_recorded_final_MSE=original['seeds'][str(seed)]['fixed_velocity_probe_final_MSE'])
        report['seeds'][str(seed)]=record
    write_json(out/'BAYES_ORACLE_REVIEW.json',report)
    destination=out/'BAYES_ORACLE_REVIEW.py'
    with destination.open('xb') as f:f.write(Path(__file__).read_bytes())
    print(json.dumps({seed:{key:row[key] for key in ['network_pair_MSE','oracle_pair_MSE','network_vs_conditional_mean_MSE']} for seed,row in report['seeds'].items()},indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out-dir')
    a=p.parse_args();run(a.run,a.out_dir)


if __name__=='__main__':main()
