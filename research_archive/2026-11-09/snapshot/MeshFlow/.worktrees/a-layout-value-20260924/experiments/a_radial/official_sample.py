"""GPU official-model sampling; Raw tokens and public post identity explicitly separate.
This adapter NEVER calls a public demo fallback 'paper full post'. No training here.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from .verify_repo import verify
from .official_train import build_model
from .io_utils import new_dir,write_json,save_npz,file_hash,array_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--samples',type=int,default=64);p.add_argument('--noise-seed',type=int,default=93610);p.add_argument('--public-post',action='store_true');p.add_argument('--fill-holes',action='store_true');a=p.parse_args()
    verify(a.repo)
    checkpoint_hash=file_hash(a.checkpoint) # stream once, not per sample
    if not torch.cuda.is_available():raise RuntimeError('Official CFG contains CUDA autocast; use verified local CUDA, no concealed CPU fallback')
    from flow_matching import create_transport
    from inference import _make_sample_fn
    from datasets.mesh_dataset import save_mesh
    from .legacy.common import save_obj
    import trimesh
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);cfg=ck['config'];model=build_model(cfg).cuda();model.load_state_dict(ck['ema']);model.eval().requires_grad_(False)
    transport=create_transport(cfg['transport']['path_type'],cfg['transport']['prediction'],use_lognorm=cfg['transport']['use_lognorm'],use_jit=cfg['transport']['use_jit'])
    fn=_make_sample_fn(transport,cfg['sample'],cfg['sample']['timestep_shift'])
    if cfg['sample']['requested_faces']!=512:raise ValueError('Frozen budget512 required')
    out=new_dir(a.out);gen=torch.Generator().manual_seed(a.noise_seed);noise=torch.randn(a.samples,800,9,generator=gen);rawrows=[];postrows=[]
    post_fn=None;post_error=None
    if a.public_post:
        try:
            import gradio_demo
            if a.fill_holes and not gradio_demo.PYMESHLAB_AVAILABLE:raise RuntimeError('Hole filling requested but PyMeshLab unavailable; do not substitute trimesh-only fill')
            post_fn=gradio_demo._post_process_mesh
        except Exception as e:post_error=repr(e)
    for i in range(a.samples):
        sid=f'{i:05d}';source_hash=array_hash(noise[i].numpy());base={'sample_id':sid,'checkpoint_sha256':checkpoint_hash,'dataset_manifest_sha256':ck['identity']['dataset_sha256'],'requested_faces':512,'noise_sha256':source_hash,'config':cfg['sample'],'reference_scope':'closed_set16','fit_qualification':'MUST_ATTACH_EXTERNAL_QUALIFICATION; completion alone is not fit'}
        postbase={**base,'row':'meshflow_public_post','post_level':'PUBLIC_CODE_NOT_PAPER_FULL','paper_full_qualified':False,'post_config':{'merge_duplicates':True,'merge_tolerance':0.,'fill_holes':a.fill_holes,'max_hole_edges':200}}
        try:
            eps=noise[i:i+1].cuda();y=torch.tensor([512],device='cuda');mask=(torch.arange(800,device='cuda')[None]<512);calls=[0]
            def model_fn(x,t,**kw):calls[0]+=1;return model.forward_with_cfg(x,t,**kw)
            torch.cuda.synchronize();t0=time.perf_counter()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                trajectory=fn(torch.cat([eps,eps],0),model_fn,y=y,mask=mask,cfg_scale=cfg['sample']['cfg_scale'])
                tokens=trajectory[-1][0,:512].float().cpu().numpy()
            torch.cuda.synchronize();secs=time.perf_counter()-t0
            if not np.isfinite(tokens).all():raise FloatingPointError('nonfinite tokens')
            tensorpath=out/'raw'/f'{sid}.npz';v=tokens.reshape(-1,3)*(.3762*.5);f=np.arange(1536,dtype=np.int64).reshape(512,3)
            save_npz(tensorpath,vertices=v,faces=f,tokens=tokens,noise=noise[i].numpy());save_obj(tensorpath.with_suffix('.obj'),v,f)
            rawrows.append({**base,'row':'meshflow_raw','status':'ok','mesh_path':str(tensorpath.relative_to(out)),'mesh_sha256':file_hash(tensorpath),'raw_tensor_sha256':array_hash(tokens),'raw_space':'tokens *0.3762/2; no clipping/quantization/welding','actual_faces':512,'seconds_noise_to_mesh':secs,'time_grid_points':cfg['sample']['num_sampling_steps'],'observed_ode_rhs_calls':calls[0],'CFG_batch_factor':2})
            if post_fn is None:raise RuntimeError(post_error or 'Public post not requested; full paper denoiser unavailable')
            # This branch reproduces public exporter quantization before public cleanup; retain that intermediate.
            qpath=out/'public_quantized'/f'{sid}.obj';qpath.parent.mkdir(exist_ok=True)
            pt=time.perf_counter();save_mesh(tokens,str(qpath),clean=False,max_val=1/.3762)
            quantized=trimesh.load(qpath,force='mesh',process=False)
            work,stats=post_fn(quantized.copy(),merge_duplicates=True,merge_tolerance=0.,fill_holes=a.fill_holes,max_hole_edges=200)
            # Public saved-coordinate space is token units. Convert to common canonical geometry here.
            pv=np.asarray(work.vertices,dtype=float)*(.3762*.5);pf=np.asarray(work.faces,dtype=np.int64)
            pp=out/'post'/f'{sid}.npz';save_npz(pp,vertices=pv,faces=pf);save_obj(pp.with_suffix('.obj'),pv,pf)
            postrows.append({**postbase,'status':'ok','mesh_path':str(pp.relative_to(out)),'mesh_sha256':file_hash(pp),'source_raw_tensor_sha256':array_hash(tokens),'quantized_export_sha256':file_hash(qpath),'actual_faces':len(pf),'seconds_noise_to_mesh':secs+time.perf_counter()-pt,'public_stats':stats})
        except Exception as e:
            if len(rawrows)<=i:rawrows.append({**base,'row':'meshflow_raw','status':'failed','error':repr(e)})
            postrows.append({**postbase,'status':'failed','error':repr(e)})
        write_json(out/'raw_MANIFEST.json',{'row':'meshflow_raw','requested_samples':a.samples,'records':rawrows},replace=(out/'raw_MANIFEST.json').exists())
        write_json(out/'post_MANIFEST.json',{'row':'meshflow_public_post','requested_samples':a.samples,'records':postrows,'paper_full_post':'NOT_IMPLEMENTED_WITHOUT_VERIFIED_DENOISER_AND_RECIPE'},replace=(out/'post_MANIFEST.json').exists())
    write_json(out/'COMPLETE.json',{'raw_attempts':len(rawrows),'post_attempts':len(postrows),'full_official_post_qualified':False,'sampling_is_not_convergence_proof':True})
if __name__=='__main__':main()
