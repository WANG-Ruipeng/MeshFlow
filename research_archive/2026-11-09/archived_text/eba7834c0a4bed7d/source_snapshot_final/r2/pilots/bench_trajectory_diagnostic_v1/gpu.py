from .common import *
import argparse,copy,shutil,traceback
import numpy as np
from .geometry import roles,projection,shape

def layouts():
    reg=manifest();qs=read(OUT/'evaluation_inputs/manifest.json')['requests'];dest=OUT/'layouts'
    if (dest/'complete.json').exists():verify(read(dest/'complete.json')['files']);return
    from r2.pilots.bench_parts_convergence_v1 import data as D
    import torch
    from meshflow_control import runtime
    from meshflow_control.seam.occupancy import OccupancyHead
    runtime.configure_stable_runtime();save(dest/'resources.json',resources())
    p=torch.load(reg['head_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    head=OccupancyHead(24).eval().requires_grad_(False);head.load_state_dict(p['head_state'],strict=True);head.cuda()
    records=[];files={}
    for q in qs[::2]:
        core=arrays(q['inference_npz']);cid=q['condition_id'];target=dest/cid/'layout.npz'
        if target.exists():a=arrays(target)
        elif q['split']=='DEV':
            a=arrays(q['old_layout_npz']);array_file(target,a)
        else:
            edges,features=D.edge_features(core['C_vertices'],core['C_faces_original'],int(core['N']))
            assert len(edges)>0
            ledger=dest/cid/'calls.jsonl'
            if ledger.exists():raise RuntimeError('Unfinished head attempt, no automatic repeat')
            event(ledger,stage='HEAD_STARTED')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                logits=head(torch.from_numpy(features).cuda(),torch.ones(len(features),dtype=torch.bool,device='cuda')).float().cpu().numpy()
            event(ledger,stage='HEAD_COMPLETE')
            layout=D.inference_layout(core['C_vertices'],core['C_faces_original'],core['C_faces_model'],int(core['N']),logits>=0)
            a={**layout,'occupancy_logits':logits};array_file(target,a)
        case=arrays(q['case_path']);actual=case['occupied_labels'].astype(bool)
        records.append(dict(condition_id=cid,split=q['split'],edges=len(a['occupied']),predicted=int(a['occupied'].sum()),source_owner=int(actual.sum()),
            false_negative=int((actual&~a['occupied']).sum()),false_positive=int((~actual&a['occupied']).sum()),same=bool(np.array_equal(actual,a['occupied']))))
        files[str(target)]=sha(target)
    csvwrite(dest/'comparison.csv',records);save(dest/'complete.json',dict(status='COMPLETE',files=files,head_forwards=4,records=records))

class Runner:
    def __init__(self,label):
        import torch
        from meshflow_control import runtime
        from meshflow_control.config import MODEL_CONFIG
        from meshflow_control.models.native import build_model
        self.torch=torch;self.label=label;self.reg=manifest();self.out=OUT/'execution'/label
        self.out.mkdir(parents=True,exist_ok=True);self.ledger=self.out/'calls.jsonl';self.attempt=self.out/'attempts'/str(time.time_ns())
        self.prior_calls=sum(json.loads(line).get('stage')=='RETURNED' for line in self.ledger.read_text().splitlines()) if self.ledger.exists() else 0
        runtime.configure_stable_runtime();torch.set_num_threads(1)
        save(self.attempt/'resources_before.json',resources())
        from r2.pilots.seam_slot_c200_v1.generate import dispatch_guard
        dispatch_guard(resources(),3500)
        p=torch.load(self.reg['checkpoints'][label],map_location='cpu',weights_only=True,mmap=True)
        self.model=build_model(copy.deepcopy(p.get('model_config',MODEL_CONFIG)),readout_mode='none',trainable=False).eval()
        self.model.load_state_dict(p['generator_state'],strict=True);self.model.cuda().set_trainable(False)
        self.calls=0;self.activations={};self.capture=False;self.hooks=[]
        if label=='COMMON':
            for name,module in self.model.named_modules():
                if isinstance(module,torch.nn.Linear) and (any(f'backbone.layers.{i}.' in name for i in (0,5,11)) or name.startswith('context_encoder')):
                    if name.endswith(('qkv','proj','fc1','fc2')) or name.startswith('context_encoder'):
                        self.hooks.append(module.register_forward_pre_hook(self.hook(name)))
            save(self.out/'captured_modules.json',[n for n,m in self.model.named_modules() if isinstance(m,torch.nn.Linear)])
        del p;torch.cuda.reset_peak_memory_stats();save(self.attempt/'resources_loaded.json',resources())
    def hook(self,name):
        def inner(module,args):
            if self.capture:
                h=args[0].detach().float().cpu().numpy();flat=h.reshape(-1,h.shape[-1]);step=max(1,len(flat)//64)
                self.activations[name]=flat[::step][:64].copy()
        return inner
    def forward(self,x,t,s,kind,rid,extra=None):
        torch=self.torch
        from meshflow_control.models.native import native_execution
        from meshflow_control.seam.model_adapter import seam_forward
        n=len(x);xt=torch.from_numpy(np.asarray(x,np.float32).reshape(1,n,9)).cuda()
        known=torch.from_numpy(s['known_mask'][None].astype(bool)).cuda();valid=torch.from_numpy(s['valid_mask'][None].astype(bool)).cuda()
        fixed=torch.from_numpy((s['anchor_ids']>=0)[None]).cuda();tt=torch.tensor([t],dtype=torch.float32,device='cuda');y=torch.tensor([n],dtype=torch.int64,device='cuda')
        event(self.ledger,stage='STARTED',kind=kind,request_id=rid,t=float(t),call=self.prior_calls+self.calls+1)
        with torch.no_grad(),native_execution(self.model,coordinates=xt):v=seam_forward(self.model,xt,tt,y,valid,known,fixed)
        torch.cuda.synchronize();self.calls+=1
        assert v.dtype==torch.bfloat16 and bool(torch.isfinite(v).all())
        out=v.float().cpu().numpy().reshape(n,3,3);out[s['anchor_ids']>=0]=0
        event(self.ledger,stage='RETURNED',kind=kind,request_id=rid,t=float(t),call=self.prior_calls+self.calls)
        return out
    def rollout(self,x,s,start,kind,rid):
        path=[x.copy()];vel=[];fixed=s['anchor_ids']>=0;original=x[fixed].copy()
        try:
            for step in range(round(start*50),50):
                v=self.forward(x,step/50,s,kind,rid);vel.append(v);x=x+np.float32(.02)*v;x[fixed]=original;path.append(x.copy())
        except BaseException:
            array_file(self.attempt/'partial_trajectories'/f'{kind}_{rid}_start{start}.npz',dict(states=np.stack(path),velocities=np.stack(vel) if vel else np.empty((0,*x.shape),np.float32)))
            raise
        assert all(np.array_equal(p[fixed],original) for p in path)
        return np.stack(path),np.stack(vel)
    def run(self):
        reg=self.reg;qs=read(OUT/'evaluation_inputs/manifest.json')['requests']
        from r2.pilots.seam_slot_c200_v1.generate import raw_arrays
        from r2.pilots.bench_parts_convergence_v1 import data as D
        receipts=[];start=time.perf_counter()
        for q in qs:
            rid=q['request_id'];lay=arrays(OUT/'layouts'/q['condition_id']/'layout.npz');lay['N']=int(lay['N']);lay['K']=int(lay['K'])
            dest=OUT/'evaluation'/self.label/rid/'raw';fixed=lay['anchor_ids']>=0
            if (dest/'receipt.json').exists():receipts.append(dest/'receipt.json');continue
            if dest.exists():raise RuntimeError('Unfinished rollout: no automatic replay '+rid)
            save(dest/'attempt.json',dict(request_id=rid,generator_budget=50))
            z=arrays(q['noise_npz'])['z'];x=np.zeros((q['N'],3,3),np.float32);x[q['K']:]=z;x[fixed]=lay['C_vertices'][lay['anchor_ids'][fixed]]
            trajectory,v=self.rollout(x,lay,0,'full',rid);source=arrays(q['source_npz'])
            array_file(dest/'raw.npz',raw_arrays(trajectory[-1],lay,lay['occupancy_logits'],source));array_file(dest/'trajectory.npz',dict(states=trajectory,velocities=v,times=np.arange(51)/50))
            array_file(dest/'source_condition.npz',source)
            save(dest/'receipt.json',dict(status='COMPLETE',label=self.label,arm=self.label,request_id=rid,generator_forwards=50,raw_sha256=sha(dest/'raw.npz'),files={str(p):sha(p) for p in dest.glob('*.npz')}))
            receipts.append(dest/'receipt.json');print(self.label,'full',len(receipts),rid,flush=True)
            if len(receipts) in (1,2,8,16):save(self.attempt/f'resources_full{len(receipts)}.json',resources())
        save(OUT/'evaluation'/self.label/'raw_complete_full.json',dict(status='COMPLETE',outputs=16,files={str(p):sha(p) for p in receipts}))
        # Fixed train states, local trajectories, and source-layout diagnostic share frozen OT identities.
        rows=[];continuation=[]
        for q in [q for q in qs if q['split']=='CURRENT_TRAIN']:
            rid=q['request_id'];s=arrays(q['fixed_sample_npz']);dest=self.out/'fixed'/rid
            if (dest/'complete.json').exists():
                r=read(dest/'complete.json');rows.extend(r['rows']);continuation.extend(r['continuation']);continue
            if dest.exists():raise RuntimeError('Unfinished diagnostic, no repeat '+rid)
            save(dest/'attempt.json',dict(request_id=rid))
            for t in reg['times']:
                x=(np.float32(1-t)*s['x0']+np.float32(t)*s['x1']).astype(np.float32);x[s['anchor_ids']>=0]=s['x1'][s['anchor_ids']>=0]
                self.capture=self.label=='COMMON';self.activations={};v=self.forward(x,t,s,'fixed',rid);self.capture=False
                if self.activations:array_file(dest/f'activations_t{t}.npz',self.activations)
                row=dict(label=self.label,request_id=rid,part=q['part_type'],t=t,**projection(x,v,s))
                for name,mask in roles(s).items():row[name+'_velocity_mse']=float(((v-s['u'])[mask]**2).mean()) if mask.any() else None
                truth=projection(x,s['u'],s);row.update({'truth_'+k:a for k,a in truth.items() if 'derivative' in k})
                row.update({'one_step_'+k:a for k,a in shape(x+(1-t)*v,s).items()});rows.append(row)
                array_file(dest/f'prediction_t{t}.npz',dict(x=x,v=v))
            local=[]
            if self.label=='USER':
                for t in reg['continuations']:
                    x=(np.float32(1-t)*s['x0']+np.float32(t)*s['x1']).astype(np.float32);x[s['anchor_ids']>=0]=s['x1'][s['anchor_ids']>=0]
                    path,v=self.rollout(x,s,t,'continuation',rid);array_file(dest/f'continuation_t{t}.npz',dict(states=path,velocities=v,times=np.arange(round(t*50),51)/50))
                    for j,p in enumerate(path):
                        row=dict(request_id=rid,start_t=t,t=t+j/50,**shape(p,s))
                        if j<len(v):row.update(projection(p,v[j],s))
                        local.append(row)
                case=arrays(q['case_path']);core=arrays(q['inference_npz']);prod=arrays(OUT/'layouts'/q['condition_id']/'layout.npz')
                if not np.array_equal(prod['occupied'],case['occupied_labels']):
                    lay=D.inference_layout(core['C_vertices'],core['C_faces_original'],core['C_faces_model'],q['N'],case['occupied_labels'].astype(bool))
                    x=np.zeros((q['N'],3,3),np.float32);x[q['K']:]=arrays(q['noise_npz'])['z'];fixed=lay['anchor_ids']>=0;x[fixed]=lay['C_vertices'][lay['anchor_ids'][fixed]]
                    path,v=self.rollout(x,lay,0,'oracle',rid);array_file(dest/'oracle_trajectory.npz',dict(states=path,velocities=v))
                    array_file(dest/'oracle_raw.npz',raw_arrays(path[-1],lay,np.asarray(case['occupied_labels'],np.float32),arrays(q['source_npz'])))
                    save(dest/'oracle_layout.json',dict(status='DIAGNOSTIC_SOURCE_OWNER_NOT_DEPLOYABLE',changed_anchor_corners=int(np.count_nonzero((prod['anchor_ids']>=0)!=(lay['anchor_ids']>=0))),slot_pool='identical addressable Gaussian array; masks differ, not every free variable paired'))
                else:save(dest/'oracle_layout.json',dict(status='SKIPPED_IDENTICAL_LAYOUT',generator_calls=0))
            continuation.extend(local);save(dest/'complete.json',dict(rows=[r for r in rows if r['request_id']==rid],continuation=local))
        csvwrite(self.out/'fixed_metrics.csv',rows);csvwrite(self.out/'continuation_metrics.csv',continuation)
        response=[]
        for q in qs[::2]:
            rid=q['request_id'];dest=self.out/'response'/rid
            if (dest/'complete.json').exists():response.extend(read(dest/'complete.json')['rows']);continue
            if dest.exists():raise RuntimeError('Unfinished perturbation, no repeat')
            save(dest/'attempt.json',dict(request_id=rid))
            lay=arrays(OUT/'layouts'/q['condition_id']/'layout.npz');path=arrays(OUT/'evaluation/COMMON'/rid/'raw/trajectory.npz')['states'];local=[]
            for t in reg['response_times']:
                vv=[];cc=[];quant=[]
                for scale in reg['response_scale']:
                    s={k:v.copy() for k,v in lay.items()};s['C_vertices'][:,0]*=np.float32(scale);x=path[round(t*50)].copy();fixed=s['anchor_ids']>=0;x[fixed]=s['C_vertices'][s['anchor_ids'][fixed]]
                    c=x[:q['K']].copy();cc.append(c);quant.append(self.torch.from_numpy(c).to(self.torch.bfloat16).float().numpy())
                    vv.append(self.forward(x,t,s,'response',rid))
                den=float(np.sqrt(np.mean((cc[1]-cc[0])**2)));diff=vv[1]-vv[0];row=dict(label=self.label,request_id=rid,split=q['split'],part=q['part_type'],t=t,C_delta_rms=den,BF16_C_delta_rms=float(np.sqrt(np.mean((quant[1]-quant[0])**2))))
                for name,mask in roles(lay).items():row[name+'_response']=float(np.sqrt(np.mean(diff[mask]**2))/den) if mask.any() and den else None
                array_file(dest/f'response_t{t}.npz',dict(vminus=vv[0],vplus=vv[1],Cminus=cc[0],Cplus=cc[1]));local.append(row)
            response.extend(local);save(dest/'complete.json',dict(rows=local))
        csvwrite(self.out/'response_metrics.csv',response)
        save(self.out/'complete.json',dict(status='COMPLETE',new_generator_calls=self.calls,seconds=time.perf_counter()-start,peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),peak_reserved_bytes=self.torch.cuda.max_memory_reserved(),peak_scope='all inference phases after model load',resources=resources()))
        print(self.label,'COMPLETE',self.calls,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['layouts','COMMON','USER']);a=p.parse_args()
    if a.action=='layouts':layouts();return
    if (OUT/'execution'/a.action/'complete.json').exists():print('Already complete');return
    try:Runner(a.action).run()
    except BaseException as exc:
        event(OUT/'engineering_events.jsonl',kind='GPU_EXECUTION_FAILURE',label=a.action,error=repr(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
