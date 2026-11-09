"""Fixed-budget CPU-only C-only owner heads. Never imports a generator."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse, csv, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_owner_layout_v1'
OLD=ROOT/'experiments/r2_seam_abc_v1'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def csvwrite(p,rr):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def features(inp):
    from meshflow_control.seam.slot_data import edge_features
    edges,x=edge_features(inp['C_vertices'],inp['C_faces_original'],int(inp['total_N']))
    e=len(x)
    x=np.c_[x,np.arange(e,dtype=np.float32)/max(1,e-1),np.full(e,e/32,dtype=np.float32)].astype(np.float32)
    return edges,x

def prepare():
    if (OUT/'registration.json').exists():return read(OUT/'registration.json')
    reg=read(OLD/'B_DATA_REGISTRATION.json');records=reg['records'];dev=reg['DEV_records']
    train=[];protected={str(OLD/'B_DATA_REGISTRATION.json'):sha(OLD/'B_DATA_REGISTRATION.json')}
    for r in records:
        d=arrays(r['inference_npz']);y=arrays(r['labels_npz']);edges,x=features(d)
        assert np.array_equal(x[:,:24],y['edge_features'])
        roles=np.where(y['occupied_labels'],1,np.where(y['source_alias_labels'],2,0)).astype(np.int64)
        assert np.array_equal(y['source_supported_labels'],roles!=0)
        train.append((x,roles))
    flat=np.concatenate([x for x,y in train]);mu=flat.mean(0);sd=np.maximum(flat.std(0),1e-4)
    for r in records+dev:
        for k in ('inference_npz','labels_npz'):protected[r[k]]=sha(r[k])
    nmax=max(r['edges'] for r in records+dev)
    x=np.zeros((47,nmax,26),np.float32);mask=np.zeros((47,nmax),bool);roles=np.full((47,nmax),-1,np.int64)
    for i,(f,y) in enumerate(train):x[i,:len(f)]=(f-mu)/sd;mask[i,:len(f)]=True;roles[i,:len(f)]=y
    assert np.sum(roles==1)==360 and np.sum(roles==0)==34 and np.sum(roles==2)==6
    data=OUT/'data';data.mkdir(exist_ok=True)
    np.savez_compressed(data/'train.npz',x=x,mask=mask,roles=roles,mean=mu,std=sd)
    protected[str(data/'train.npz')]=sha(data/'train.npz')
    v=dict(status='FROZEN',records=records,DEV_records=dev,protected=protected,steps=2000,
           lr=.001,weight_decay=.0001,seed=2026101701,protocol_sha256=sha(OUT/'PROTOCOL.md'),
           code_sha256=sha(__file__),candidate='CONTEXT',threshold=.5,head_arms=['MLP','CONTEXT'],
           gpu_calls=0,generator_training_updates=0)
    write(OUT/'registration.json',v);print(json.dumps({'stage':'PREPARED','TRAIN':47,'DEV':3,'edges':400}))

def build(arm):
    import torch
    from torch import nn
    class Block(nn.Module):
        def __init__(self):
            super().__init__();self.n1=nn.LayerNorm(32);self.n2=nn.LayerNorm(32)
            self.qkv=nn.Linear(32,96);self.proj=nn.Linear(32,32)
            self.ff=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,32))
        def forward(self,x,mask):
            b,e,_=x.shape
            q,k,v=self.qkv(self.n1(x)).reshape(b,e,3,4,8).permute(2,0,3,1,4).unbind(0)
            score=(q@k.transpose(-1,-2))/math.sqrt(8)
            score=score.masked_fill(~mask[:,None,None,:],float('-inf'))
            y=(score.float().softmax(-1).to(v.dtype)@v).transpose(1,2).reshape(b,e,32)
            x=x+self.proj(y);return x+self.ff(self.n2(x))
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.proj=nn.Linear(26,32)
            self.blocks=nn.ModuleList([Block(),Block()]) if arm=='CONTEXT' else nn.ModuleList([])
            self.out=nn.Linear(32,1)
            for m in self.modules():
                if isinstance(m,nn.Linear):nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
        def forward(self,x,mask):
            x=torch.nn.functional.silu(self.proj(x))
            for block in self.blocks:x=block(x,mask)
            return self.out(x).squeeze(-1).float()
    return Model()

def setup():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.manual_seed(2026101701)
    return torch

def contract():
    r=read(OUT/'registration.json')
    assert sha(__file__)==r['code_sha256'] and sha(OUT/'PROTOCOL.md')==r['protocol_sha256']
    for p,h in r['protected'].items():assert sha(p)==h,p
    return r

def train(arm):
    r=contract();dest=OUT/'heads'/arm;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():print(json.dumps(read(dest/'complete.json')));return
    if (dest/'started.json').exists():raise RuntimeError('Unresolved head attempt; no silent retry')
    torch=setup();d=arrays(OUT/'data/train.npz')
    x=torch.from_numpy(d['x']);mask=torch.from_numpy(d['mask']);y=torch.from_numpy((d['roles']==1).astype(np.float32))
    model=build(arm);opt=torch.optim.AdamW(model.parameters(),lr=r['lr'],weight_decay=r['weight_decay'])
    write(dest/'started.json',dict(arm=arm,unix=time.time(),steps=2000,device='cpu',params=sum(p.numel() for p in model.parameters())))
    start=time.perf_counter();logs=[]
    try:
        for step in range(1,2001):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cpu',dtype=torch.bfloat16):z=model(x,mask)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')*mask).sum(1)/mask.sum(1)
            loss=loss.mean();assert bool(torch.isfinite(loss))
            loss.backward();opt.step()
            if step==1 or step%50==0:
                rec=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-start)
                logs.append(rec);print(json.dumps(dict(arm=arm,**rec)),flush=True)
        torch.save(dict(arm=arm,step=2000,state_dict=model.state_dict(),optimizer=opt.state_dict(),
                        normalization_mean=d['mean'],normalization_std=d['std'],threshold=.5,
                        feature_schema='C_edge26_rank_context_v1',registration_sha256=sha(OUT/'registration.json')),
                   dest/'step2000.pt')
        csvwrite(dest/'learning_curve.csv',logs)
        write(dest/'complete.json',dict(status='COMPLETE',arm=arm,updates=2000,head_forwards=2000,
              head_backwards=2000,parent_exposures=94000,parameters=sum(p.numel() for p in model.parameters()),
              seconds=time.perf_counter()-start,device='CPU',precision='BF16 forward / FP32 parameters, optimizer and BCE',
              generator_forwards=0,D0_forwards=0,GPU_calls=0,checkpoint_sha256=sha(dest/'step2000.pt')))
    except BaseException as e:
        write(dest/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),step=step,no_retry=True));raise

def metrics(z,roles):
    p=1/(1+np.exp(-np.clip(z,-60,60)));y=roles==1;pred=z>=0
    active=roles>=0
    z,p,y,pred,roles=z[active],p[active],y[active],pred[active],roles[active]
    result=dict(edges=len(z),owner=int(y.sum()),Null=int((roles==0).sum()),alias=int((roles==2).sum()),
                TP=int((pred&y).sum()),FN=int((~pred&y).sum()),FP=int((pred&~y).sum()),TN=int((~pred&~y).sum()),
                owner_recall=float(pred[y].mean()) if y.any() else None,
                null_specificity=float((~pred[roles==0]).mean()) if (roles==0).any() else None,
                alias_specificity=float((~pred[roles==2]).mean()) if (roles==2).any() else None,
                brier=float(np.mean((p-y)**2)),nll=float(np.mean(np.logaddexp(0,z)-y*z)))
    bins=np.minimum((p*10).astype(int),9);ece=0
    for i in range(10):
        m=bins==i
        if m.any():ece+=m.mean()*abs(p[m].mean()-y[m].mean())
    result['ECE10']=float(ece);return result

def evaluate():
    r=contract();torch=setup();d=arrays(OUT/'data/train.npz');dest=OUT/'evaluation_head';dest.mkdir(exist_ok=True)
    if (dest/'started.json').exists():raise RuntimeError('Terminal evaluation already attempted; no silent replay')
    write(dest/'started.json',dict(unix=time.time(),terminal_head_batches=4,model_selection='CONTEXT_FIXED'))
    # DEV features are from C-only; source labels are read below solely for final scoring.
    xs=[]
    for q in r['DEV_records']:
        _,f=features(arrays(q['inference_npz']));xs.append((f-d['mean'])/d['std'])
    dx=np.zeros((3,d['x'].shape[1],26),np.float32);dm=np.zeros(dx.shape[:2],bool)
    for i,f in enumerate(xs):dx[i,:len(f)]=f;dm[i,:len(f)]=True
    allrows=[];parent=[];summaries=[];predictions={}
    for arm in r['head_arms']:
        done=read(OUT/'heads'/arm/'complete.json');assert done['updates']==2000
        ck=OUT/'heads'/arm/'step2000.pt';assert sha(ck)==done['checkpoint_sha256']
        model=build(arm);model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state_dict']);model.eval()
        for split,records,x,mask in [('TRAIN',r['records'],d['x'],d['mask']),('DEV',r['DEV_records'],dx,dm)]:
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                z=model(torch.from_numpy(x),torch.from_numpy(mask)).float().numpy()
            rr=np.full(mask.shape,-1,np.int64)
            for i,q in enumerate(records):
                labels=arrays(q['labels_npz']);roles=np.where(labels['occupied_labels'],1,np.where(labels['source_alias_labels'],2,0))
                e=len(roles);rr[i,:e]=roles
                parent.append(dict(arm=arm,split=split,parent_id=q['parent_id'],**metrics(z[i,:e],roles)))
                for j in range(e):
                    allrows.append(dict(arm=arm,split=split,parent_id=q['parent_id'],condition_id=q['condition_id'],edge_id=j,
                         role=['NULL','OWNER','ALIAS'][roles[j]],logit=float(z[i,j]),probability=float(1/(1+np.exp(-float(z[i,j])))),predicted_owner=bool(z[i,j]>=0)))
                if arm=='CONTEXT':
                    p=OUT/'head_predictions'/q['parent_id'][:8];p.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(p/'prediction.npz',occupied=z[i,:e]>=0,occupancy_logits=z[i,:e].astype(np.float32))
                    receipt=dict(condition_id=q['condition_id'],C_bundle_sha256=sha(q['inference_npz']),head_sha256=sha(ck),threshold=.5,
                                 source_labels_used_in_forward=False,features='C_only26',head_batch_calls=1)
                    write(p/'receipt.json',receipt)
                    predictions[q['condition_id']]=dict(prediction_npz=str(p/'prediction.npz'),receipt=str(p/'receipt.json'))
            summaries.append(dict(arm=arm,split=split,**metrics(z,rr)))
    csvwrite(dest/'per_edge.csv',allrows);csvwrite(dest/'per_parent.csv',parent);csvwrite(dest/'summary.csv',summaries)
    c=[p for p in parent if p['arm']=='CONTEXT'];tr=next(s for s in summaries if s['arm']=='CONTEXT' and s['split']=='TRAIN')
    de=next(s for s in summaries if s['arm']=='CONTEXT' and s['split']=='DEV')
    familiar=[p for p in c if p['parent_id'].startswith(('73b369','3f7417','85b169','8c9b20'))]
    gates=dict(TRAIN_owner_ge90=tr['owner_recall']>=.9,TRAIN_trueNull_ge80=tr['null_specificity']>=.8,
               TRAIN_alias_ge50=tr['alias_specificity']>=.5,
               familiar_each_owner_ge90=all(p['owner_recall']>=.9 for p in familiar),
               familiar_all_trueNull_inactive=all(p['null_specificity'] in (None,1.) for p in familiar),
               DEV_each_owner_ge90=all(p['owner_recall']>=.9 for p in c if p['split']=='DEV'),
               DEV_trueNull_inactive=de['null_specificity']==1.,DEV_alias_atleast2of3=de['alias_specificity']>=2/3)
    passed=all(gates.values())
    write(OUT/'head_predictions/manifest.json',predictions)
    write(OUT/'gate.json',dict(status='PASS' if passed else 'FAIL',candidate='CONTEXT',gates=gates,
          generation_allowed=passed,training_updates=4000,terminal_head_batch_calls=4,
          checkpoint=str(OUT/'heads/CONTEXT/step2000.pt'),checkpoint_sha256=sha(OUT/'heads/CONTEXT/step2000.pt'),
          prediction_manifest_sha256=sha(OUT/'head_predictions/manifest.json'),
          failure_action='STOP_NO_REFIT_NO_THRESHOLD_TUNING_NO_GENERATION'))
    write(dest/'complete.json',dict(status='COMPLETE',head_batch_forwards=4,per_edge_rows=len(allrows),per_parent_rows=len(parent),gate=passed))
    print(json.dumps(read(OUT/'gate.json'),indent=2));print(json.dumps(summaries,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate']);p.add_argument('--arm',choices=['MLP','CONTEXT']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='train':train(a.arm)
    else:evaluate()
if __name__=='__main__':main()
