"""CPU representation diagnostics; generated production samples stay immutable."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from ..models import ShapeAE,ShapeFM
from ..io_utils import new_dir,write_json,file_hash,save_npz,array_hash


def main(run):
    run=Path(run).resolve();p=json.loads((run/'RUN_PROTOCOL.json').read_text());out=new_dir(run/'representation_diagnostics')
    torch.set_num_threads(1)
    data=Path(p['data']);assert all(file_hash(data/n)==h for n,h in p['data_hashes'].items())
    with np.load(data/'radial_train.npz',allow_pickle=False) as f:y=torch.tensor(f['logr']);uids=f['uids'].tolist()
    result={'scope':'train-support diagnostics, not held-out quality or converged distribution qualification',
        'predeclared_final_checkpoints':True,'PCA_train16_already_interpolates':True,
        'velocity_probe_count_per_seed':4096,'probe_seed_base':243025,
        'auxiliary_CPU_Euler128':'same production epsilon; no production sample replacement or checkpoint selection',
        'timing':'diagnostic elapsed time is not inference timing','seeds':{}}
    for seed in p['seeds']:
        sp=run/f'seed{seed}';ac=torch.load(sp/'ae/step00003200.pt',map_location='cpu',weights_only=True)
        fc=torch.load(sp/'fm/step00008000.pt',map_location='cpu',weights_only=True)
        assert ac['step']==3200 and fc['step']==8000 and fc['config']['ae_sha256']==file_hash(sp/'ae/step00003200.pt')
        ae=ShapeAE().eval();ae.load_state_dict(ac['model']);fm=ShapeFM().eval();fm.load_state_dict(fc['model'])
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state());initial=ShapeFM().eval()
        initial.mu.copy_(fm.mu);initial.sd.copy_(fm.sd)
        samples=json.loads((sp/'samples/SAMPLES.json').read_text());assert samples['requested']==64 and len(samples['records'])==64
        assert samples['ae_sha256']==file_hash(sp/'ae/step00003200.pt') and samples['fm_sha256']==file_hash(sp/'fm/step00008000.pt')
        noise=[];nodes=[];zs=[];recorded=[]
        for r in samples['records']:
            if r['status']!='ok':recorded.append({'sample_id':r['sample_id'],'status':'upstream_failure'});continue
            with np.load(sp/'samples'/r['path'],allow_pickle=False) as z:
                assert array_hash(z['nodes'])==r['nodes_hash']
                noise.append(z['noise'].copy());nodes.append(z['nodes'].copy());zs.append(z['z_standardized'].copy())
            recorded.append({'sample_id':r['sample_id'],'status':'ok'})
        with torch.no_grad():
            recon,codes=ae(y);targets=fm.standardize(codes)
            generator=torch.Generator().manual_seed(243025+seed)
            ids=torch.arange(4096)%16;x1=targets[ids];eps=torch.randn(x1.shape,generator=generator)
            t=torch.rand(4096,generator=generator);xt=(1-t[:,None])*eps+t[:,None]*x1;truth=x1-eps
            trained_loss=(fm(xt,t)-truth).square().mean(-1);initial_loss=(initial(xt,t)-truth).square().mean(-1)
            bins=[]
            for low,high in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
                sel=(t>=low)&(t<high);bins.append({'low':float(low),'high':float(high),'count':int(sel.sum()),
                    'trained_mse':float(trained_loss[sel].mean()),'initial_mse':float(initial_loss[sel].mean())})
            row={'AE_mean_logradius_MSE':float((recon-y).square().mean()),'AE_per_parent_logradius_MSE':(recon-y).square().mean(-1).tolist(),
                'PCA_initial':json.loads((run/f'PCA_INITIAL_{seed}.json').read_text()),
                'latent_training_std':codes.std(0,unbiased=False).tolist(),'FM_sd':fm.sd.tolist(),
                'floor_dimensions':int((codes.std(0,unbiased=False)<1e-3).sum()),
                'fixed_velocity_probe_initial_MSE':float(initial_loss.mean()),'fixed_velocity_probe_final_MSE':float(trained_loss.mean()),
                'velocity_probe_t_bins':bins,'records':recorded}
            if len(noise)==64:
                noise=torch.from_numpy(np.stack(noise));stored_nodes=np.stack(nodes);stored_zs=np.stack(zs)
                z32=fm.sample_standardized(noise,32);z128=fm.sample_standardized(noise,128)
                n32=ae.decode_nodes(fm.unstandardize(z32));n128=ae.decode_nodes(fm.unstandardize(z128))
                row['CPU32_vs_production_GPU32_max_abs_nodes']=float(np.abs(n32.numpy()-stored_nodes).max())
                row['CPU32_vs_production_GPU32_max_abs_standardized_z']=float(np.abs(z32.numpy()-stored_zs).max())
                row['same_noise_CPU32_vs_CPU128_node_RMS']=float((n32-n128).square().mean().sqrt())
                for name,n in [('production32',torch.from_numpy(stored_nodes)),('auxiliary_CPU128',n128)]:
                    distances=(n[:,None,:]-y[None,:,:]).square().mean(-1).sqrt();nearest=distances.argmin(-1)
                    minimum=distances.min(-1).values
                    row[name]={'nearest_training_logradius_RMS_mean':float(minimum.mean()),
                        'nearest_training_logradius_RMS_median':float(minimum.median()),
                        'nearest_parent_counts':{uid:int((nearest==i).sum()) for i,uid in enumerate(uids)},
                        'nearest_parent_coverage':int(len(nearest.unique()))/16,
                        'generated_logradius_node_std_mean':float(n.std(0,unbiased=False).mean()),
                        'train_logradius_node_std_mean':float(y.std(0,unbiased=False).mean())}
                save_npz(out/f'auxiliary_cpu_seed{seed}.npz',noise=noise.numpy(),z_standardized32=z32.numpy(),z_standardized128=z128.numpy(),nodes32=n32.numpy(),nodes128=n128.numpy())
            result['seeds'][str(seed)]=row
    result['protocol_sha256']=file_hash(run/'RUN_PROTOCOL.json');result['source_sha256']=file_hash(__file__)
    write_json(out/'RESULT.json',result)
    print(json.dumps({k:{a:v[a] for a in ['AE_mean_logradius_MSE','fixed_velocity_probe_initial_MSE','fixed_velocity_probe_final_MSE','production32','auxiliary_CPU128']} for k,v in result['seeds'].items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();main(a.run)