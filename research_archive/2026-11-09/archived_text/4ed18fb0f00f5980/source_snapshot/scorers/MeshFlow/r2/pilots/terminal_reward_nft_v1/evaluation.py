"""Terminal NFT pilot: unchanged D0/geometry scoring and explicit reward ledger.

No model import at module load. CPU score functions require a model-free
process; only explicit postprocess_one(checkpoint=...) may call official D0.
No binding, RBF, denoiser iterations, learned metric or automatic image labels.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_terminal_reward_nft_v1'
VERSION='r2_terminal_reward_nft_original_evaluation_v1'
L=float(np.sqrt(3.)*2/.3762)
COMPONENTS=('length','has_interface','CF','FF','I','degenerate','deleted')
POSITIVE=('length','has_interface')
NEGATIVE=('CF','FF','I','degenerate','deleted')


def resolve(p):
    s=str(p).replace('\\','/')
    if os.name=='nt' and s.startswith('/mnt/'):s=s[5].upper()+':/'+s[7:]
    elif os.name!='nt' and len(s)>2 and s[1]==':':s='/mnt/'+s[0].lower()+s[2:]
    p=Path(s)
    return p if p.is_absolute() else ROOT/p


def sha(p):
    h=hashlib.sha256()
    with resolve(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(resolve(p).read_text(encoding='utf-8-sig'))

def arrays(p):
    with np.load(resolve(p),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,Path):return str(value)
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write(p,value):
    p=resolve(p); data=json.dumps(clean(value),ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    if p.exists():
        if p.read_text(encoding='utf8')!=data:raise ValueError('Refusing changed receipt overwrite: '+str(p))
        return
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(data,encoding='utf8');tmp.replace(p)


def verify(files):
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('Input/source changed: '+str(p))


def package():
    source=str(ROOT/'r2/src')
    if source not in sys.path:sys.path.insert(0,source)


def cpu_metrics():
    if 'torch' in sys.modules:raise RuntimeError('CPU scoring must run in a separate model-free process')
    from r2.tools.partial_edge_binding_v1 import metrics
    return metrics


def definition_hashes():
    m=cpu_metrics();result=m.definition_hashes()
    for p in (Path(__file__),ROOT/'r2/src/meshflow_control/source_winding.py',ROOT/'r2/src/meshflow_control/winding_solver.py'):
        result[str(p)]=sha(p)
    return result


def postprocess_one(raw_path,condition_path,K,out_dir,*,checkpoint=None,proposal_path=None,proposal_receipt=None,resume=False):
    """Reuse maintained exactly-one official D0 + C restore + GEO_C + winding.

    For efficient resident D0 workers, generate one official proposal and its
    authenticated receipt externally, then provide proposal_path/receipt here.
    This adapter never loops D0 or draws noise. CPU scorers run separately.
    """
    package()
    from meshflow_control import denoise
    if not Path(denoise.__file__).resolve().is_relative_to((ROOT/'r2/src').resolve()):
        raise RuntimeError('Expected maintained repository denoise package')
    if proposal_path is not None:
        upstream=read(proposal_receipt)
        if upstream.get('model_forwards')!=1 or upstream.get('status')!='PASS':
            raise ValueError('Cached proposal must certify exactly one official D0 forward')
    else:upstream=None
    result=denoise.run(input=resolve(raw_path),known_faces=int(K),out=resolve(out_dir),
        condition=resolve(condition_path),input_key='raw',checkpoint=resolve(checkpoint) if checkpoint else None,
        proposal=resolve(proposal_path) if proposal_path else None,
        proposal_receipt=resolve(proposal_receipt) if proposal_receipt else None,
        coordination='none',winding='source',resume=resume)
    dest=resolve(out_dir)
    return dict(status=result['status'],post_path=str(dest/'post.npz'),post_sha256=sha(dest/'post.npz'),
        canonical_path=str(dest/'post_canonical.npz'),pre_geo_path=str(dest/'denoised.npz'),
        denoise_receipt=str(dest/'denoise.json'),denoise_receipt_sha256=sha(dest/'denoise.json'),
        upstream_model_forwards=1,model_forwards_this_invocation=result['model_forwards_this_invocation'],
        cached_upstream_receipt=clean(upstream),post_contract='SINGLE_OFFICIAL_D0_GEO_C_SOURCE_WINDING',
        coordination='none',result=result)


def source_metadata(condition_path,canonical_C):
    package()
    from meshflow_control.source_winding import validate_source
    source=validate_source(arrays(condition_path),canonical_C)
    if not source['has_real_source_vertex_ids']:raise ValueError('Real source C vertex IDs required')
    return source


def raw_source_index(raw,source):
    """Original FULL exact uint32 coordinate index, source C, fixed-C winding."""
    package()
    from meshflow_control.winding_solver import orient_fixed_C
    from meshflow_control.source_winding import _C_vertex_ID_topology
    raw=np.asarray(raw);K=len(source['C']);n=len(raw)
    if raw.dtype!=np.float32 or raw.shape!=(n,3,3) or not np.isfinite(raw).all():raise ValueError('Expected finite FP32 RAW')
    flat=np.ascontiguousarray(raw).reshape(-1,3)
    _,first,inverse=np.unique(flat.view(np.uint32).reshape(-1,3),axis=0,return_index=True,return_inverse=True)
    vertices=flat[first];faces=inverse.reshape(n,3)
    corners=np.tile(np.arange(3,dtype=np.int64),(n,1))
    faces[:K]=np.take_along_axis(faces[:K],source['P_inverse'],axis=1)
    corners[:K]=np.take_along_axis(corners[:K],source['P_inverse'],axis=1)
    orientation=orient_fixed_C(vertices,faces,K);faces=orientation['faces']
    corners[orientation['flips']]=corners[orientation['flips']][:,[0,2,1]]
    topology=_C_vertex_ID_topology(source['C_original_source_vertex_ids'],faces[:K],True)
    if topology['status']=='CHANGED_BY_GEO_C':topology['status']='CHANGED_BY_EXACT_COORDINATE_INDEX'
    topology['domain']='RAW_EXACT_UINT32_COORDINATE_INDEX_NO_TOLERANCE'
    result=dict(vertices=vertices,faces=faces,output=vertices[faces],source_face_ids=np.arange(n,dtype=np.int64),
        source_corner_ids=corners,known_faces=np.asarray(K),input_faces=np.asarray(n),C=vertices[faces[:K]],
        free_face_flips=orientation['flips'])
    if result['C'].tobytes()!=source['C_original'].tobytes():raise ValueError('RAW C source restoration failed')
    return result,orientation,topology


def score_one(request,raw_path,post_path,condition_path,query_descriptor,out_dir,*,label=None,registration_sha256=None,stages=('RAW','POST')):
    """Score a single request's RAW/POST using unchanged numerical definitions.

    request requires request_id,N,K,condition_id; query_descriptor is the frozen
    original 2048-query descriptor. Stage outputs include separate topology,
    finite-face contacts, shape records, vectors, connectivity and denominators.
    No source parent style or GT reconstruction is read/scored.
    """
    M=cpu_metrics();K=int(request['K']);N=int(request['N']);root=resolve(out_dir)
    rawfile=arrays(raw_path);raw=np.asarray(rawfile['raw']).reshape(N,3,3)
    source=source_metadata(condition_path,raw[:K]);post=arrays(post_path)
    if post['output'][:K].tobytes()!=source['C_original'].tobytes():raise ValueError('POST C differs')
    if query_descriptor['condition_id']!=request['condition_id']:raise ValueError('Frozen query/request identity differs')
    if abs(float(query_descriptor['identity']['L0'])-L)>1e-12:raise ValueError('Fixed reference length differs')
    frozen={str(resolve(raw_path)):sha(raw_path),str(resolve(post_path)):sha(post_path),
        str(resolve(condition_path)):sha(condition_path),query_descriptor['path']:query_descriptor['sha256']}
    verify(frozen);defs=definition_hashes();results={}
    for stage in stages:
        if stage not in ('RAW','POST'):raise ValueError('Only RAW and POST')
        dest=root/stage;done=dest/'score.json'
        identity=dict(version=VERSION,request_id=request['request_id'],condition_id=request['condition_id'],
            label=label or request.get('label'),stage=stage,registration_sha256=registration_sha256,
            input_hashes=frozen,definition_hashes=defs,N=N,K=K)
        if done.exists():
            old=read(done)
            if old['identity']!=identity:raise ValueError('Cached score identity changed')
            verify(old['files']);results[stage]=old;continue
        if dest.exists():raise RuntimeError('Unfinished score needs explicit engineering recovery: '+str(dest))
        write(dest/'started.json',identity);tick=time.perf_counter()
        if stage=='RAW':
            mesh,orientation,topology=raw_source_index(raw,source)
            np.savez_compressed(dest/'mesh_source_direction.npz',**mesh)
            write(dest/'orientation.json',{k:v for k,v in orientation.items() if not isinstance(v,np.ndarray)})
        else:
            mesh=post
            package();from meshflow_control.source_winding import _C_vertex_ID_topology
            topology=_C_vertex_ID_topology(source['C_original_source_vertex_ids'],mesh['faces'][:K],True)
        score=M.score_arrays(mesh['vertices'],mesh['faces'],K,raw[:K],source['C_source_vertex_ids'],query_descriptor,N,mesh['source_face_ids'])
        vectors=score.pop('vectors');np.savez_compressed(dest/'vectors.npz',**vectors)
        score['topology']['domain']='SOURCE_C_RESTORED_FREE_ORIENTED_'+('EXACT_COORDINATE_RAW_INDEX' if stage=='RAW' else 'ACTUAL_POST_INDEX')
        ledger=M.connectivity_ledger(mesh['vertices'],mesh['faces'],K,score['pairs'],mesh['source_face_ids'],mesh['source_corner_ids'])
        write(dest/'connectivity.json',ledger);write(dest/'pairs.json',score['pairs']);write(dest/'shape_per_face.json',score['shape'])
        score.update(version=VERSION,identity=identity,stage=stage,request_id=request['request_id'],label=identity['label'],N=N,K=K,
            source_C_connectivity_status=topology['status'],source_C_connectivity_audit=topology,
            scoring_geometry_path=str(dest/'mesh_source_direction.npz') if stage=='RAW' else str(resolve(post_path)),
            reward_status='NOT_ASSIGNED_REQUIRES_ACTUAL_REVIEW',new_queries=0,model_calls=0,seconds=time.perf_counter()-tick)
        score['files']={str(p):sha(p) for p in dest.iterdir() if p.is_file()}
        write(done,score);results[stage]=clean(score)
    verify(frozen)
    return dict(status='COMPLETE',version=VERSION,request_id=request['request_id'],label=label,stages=results,
        score_paths={s:str(root/s/'score.json') for s in stages},model_calls=0,new_queries=0)


def reward_geometry(post_score):
    """Return unweighted, bounded terminal quantities; no semantic conclusion."""
    m=post_score['metrics'];g=post_score['topology'];k=int(post_score['K']);n=int(post_score['N'])
    perimeter=float(post_score['boundary_audit']['boundary_length_model'])
    if perimeter<=0:raise ValueError('Reward boundary is empty')
    edges=int(g['C_F_valid_edge_count']);length=float(g['C_F_valid_edge_total_length'])
    actual=int(m['actual_free_faces']);original=n-k
    if original<=0:raise ValueError('No original free-face denominator')
    def finite01(value,name):
        if value is None or not math.isfinite(float(value)):raise ValueError('Unscorable reward component '+name)
        value=float(value)
        if value < -1e-10 or value > 1+1e-10:raise ValueError('Reward fraction out of range '+name)
        return min(1.,max(0.,value))
    components=dict(length=min(1.,max(0.,length/perimeter)),has_interface=float(edges>0),
        CF=finite01(m['CF_illegal_affected_free_fraction'],'CF'),FF=finite01(m['FF_illegal_affected_free_fraction'],'FF'),
        I=finite01(m['ConflictCov_C'],'I'),
        degenerate=finite01(float(m['free_near_degenerate_count'])/actual if actual else None,'degenerate'),
        deleted=finite01(float(m['deleted_free_faces'])/original,'deleted'))
    return dict(status='PASS',components=components,legal_CF_edges=edges,legal_CF_length=length,
        C_boundary_perimeter=perimeter,free_denominator=actual,original_free_denominator=original,
        score_status=post_score['status'],I_status=m.get('ConflictCov_C_status'),C_preserved_exact=post_score['C_preserved_exact'],
        stage='POST',definition='CF graph/legal length unchanged; length ratio clipped at1; all original source C open edges are descriptive reward domain, not a claim all must seal',
        no_binding=True)


def geometry_review_reward(geometry,review):
    """Locked pilot terminal reward; actual RAW/POST review is mandatory.

    R = gate * (.25 + .25*has_CF + clipped_CF_length_ratio)
        * (1-CF)*(1-FF)*(1-I)*(1-near_degenerate)*retained_free**.25.
    Coefficients are preregistered pilot choices, not optimized weights.
    """
    if 'metrics' in geometry:geometry=reward_geometry(geometry)
    if geometry.get('status')!='PASS':raise ValueError('Geometry failed; do not hide as low reward')
    if geometry.get('C_preserved_exact') is False:raise ValueError('C hard-preservation failed')
    c={k:float(geometry['components'][k]) for k in COMPONENTS}
    if not all(math.isfinite(v) for v in c.values()):raise ValueError('Nonfinite reward geometry')
    c={k:min(1.,max(0.,v)) for k,v in c.items()}
    retained=min(1.,max(0.,float(geometry['free_denominator'])/float(geometry['original_free_denominator'])))
    quality_factors=dict(CF_survival=1-c['CF'],FF_survival=1-c['FF'],I_survival=1-c['I'],
        nondegenerate_fraction=1-c['degenerate'],retained_free_fraction=retained,retained_free_quarter_power=retained**.25)
    quality=quality_factors['CF_survival']*quality_factors['FF_survival']*quality_factors['I_survival']*quality_factors['nondegenerate_fraction']*quality_factors['retained_free_quarter_power']
    task_terms=dict(base=.25,has_valid_CF=.25*c['has_interface'],length=c['length'])
    task=sum(task_terms.values());reasons=[]
    keys=('RAW_adoption','POST_adoption','POST_body_complete')
    labels={k:review.get(k,'UNKNOWN') for k in keys}
    if any(v not in ('YES','NO','UNKNOWN','UNCERTAIN','NA') for v in labels.values()):raise ValueError('Invalid explicit review label')
    for k,v in labels.items():
        if v not in ('YES','NO'):reasons.append(k+'_ABSTAIN')
    if review.get('review_status')!='REVIEWED':reasons.append('NO_ACTUAL_REVIEW')
    if review.get('reviewer_kind') not in ('AI_PROXY','HUMAN') or not review.get('reviewer'):reasons.append('NO_REVIEWER_IDENTITY')
    if not (review.get('image_sha256') or review.get('reviewed_image_sha256') or review.get('evidence_sha256')):reasons.append('NO_VIEWED_IMAGE_IDENTITY')
    if geometry.get('I_status') not in (None,'PASS'):reasons.append('I_NOT_SCORABLE')
    gate=None if reasons else int(all(v=='YES' for v in labels.values()))
    return dict(version='terminal_NFT_locked_reward_v1',status='ABSTAIN' if reasons else 'SCORABLE',reasons=reasons,
        reward=None if reasons else float(gate*task*quality),review_gate=gate,review_labels=labels,
        components=c,quality_factors=quality_factors,quality=quality,task_terms=task_terms,task=task,
        legal_CF_edges=geometry['legal_CF_edges'],legal_CF_length=geometry['legal_CF_length'],
        C_boundary_perimeter=geometry['C_boundary_perimeter'],free_denominator=geometry['free_denominator'],
        original_free_denominator=geometry['original_free_denominator'],
        reviewer_kind=review.get('reviewer_kind'),reviewer=review.get('reviewer'),
        formula='gate*(.25+.25*has_valid_CF+clip(CF_length/C_perimeter,0,1))*(1-CF)*(1-FF)*(1-I)*(1-near_degenerate)*(retained_free**.25)',
        deletion_policy='ABSOLUTE_RETAINED_FREE_POST/(N-K); not improvement over baseline',
        coefficients='PREREGISTERED_PILOT_CHOICES_NOT_OPTIMIZED',automatic_semantic_labels=False)


def group_rewards(rows,*,group_key='condition_id',expected_group_size=8):
    """Pure per-C standardization, abstentions never enter preference training.

    rows contain condition_id/request_id and geometry_review_reward fields.
    The full eight-request group is required; known reviews alone set mean/std.
    """
    grouped=collections.defaultdict(list)
    for i,row in enumerate(rows):
        if group_key not in row:raise ValueError('Missing frozen group identity')
        grouped[row[group_key]].append((i,row))
    result=[None]*len(rows);groups=[]
    for key,members in sorted(grouped.items(),key=lambda x:str(x[0])):
        if expected_group_size is not None and len(members)!=expected_group_size:raise ValueError('Incomplete frozen group: '+str(key))
        if len({r['request_id'] for _,r in members})!=len(members):raise ValueError('Repeated request in reward group')
        known=[float(r['reward']) for _,r in members if r.get('status')=='SCORABLE' and r.get('reward') is not None]
        if not all(math.isfinite(v) for v in known):raise ValueError('Nonfinite known reward')
        mean=float(np.mean(known)) if known else None;std=float(np.std(known,ddof=0)) if known else None
        usable=len(known)>=2 and std>=1e-8
        group=dict(group_id=key,count=len(members),known=len(known),mean=mean,std=std,
            usable=usable,ddof=0,epsilon=1e-8,advantage_clip=[-2.,2.],neutral_r=.5,
            reason='USABLE' if usable else 'INSUFFICIENT_KNOWN_OR_STD_LT_1E-8')
        groups.append(group)
        for i,row in members:
            known_here=row.get('status')=='SCORABLE' and row.get('reward') is not None
            preference_known=bool(usable and known_here)
            advantage=(float(row['reward'])-mean)/(std+1e-8) if preference_known else None
            clipped=float(np.clip(advantage,-2.,2.)) if preference_known else 0.
            result[i]=dict(row,preference_known=preference_known,r_NFT=.5+clipped/4.,
                advantage=advantage,advantage_clipped=clipped,group_mean=mean,group_std=std,group_known=len(known),
                preference_status='KNOWN' if preference_known else 'NEUTRAL_REFERENCE_ONLY',reference_term_retained=True)
    return dict(status='COMPLETE',version='terminal_NFT_group_rewards_v1',groups=groups,rows=result,
        formula='clip((R-mean)/(std+1e-8),-2,2)/4+.5; unknown/insufficient/constant groups neutral with preference_known=False')

def postprocess_batch(jobs,checkpoint,ledger_dir,*,resource_snapshot_path,proposal_only=False):
    """Explicit GPU-worker entrypoint: one resident official D0 for all jobs.

    jobs contain raw_path, condition_path, K, out_dir and request_id; optional
    proposal_dir defaults to out_dir's sibling NAME_official_D0. This function
    is never invoked by import, score or render. Root must record resource
    admission before launching. Each new authorized job has exactly one
    predict_proposal call; saved proposal+receipt precede CPU geometry.
    """
    package();ledger=resolve(ledger_dir);snapshot=resolve(resource_snapshot_path)
    if not snapshot.is_file():raise ValueError('Resource-admission snapshot required before D0 dispatch')
    if len({str(resolve(j['out_dir'])) for j in jobs})!=len(jobs):raise ValueError('D0 jobs must own separate output directories')
    checkpoint=resolve(checkpoint)
    from meshflow_control.denoise import OFFICIAL_SHA
    inputs={str(resolve(j['raw_path'])):sha(j['raw_path']) for j in jobs}
    inputs.update({str(resolve(j['condition_path'])):sha(j['condition_path']) for j in jobs})
    definition=dict(version=VERSION,checkpoint_path=str(checkpoint),checkpoint_sha256=OFFICIAL_SHA,
        resource_snapshot=str(snapshot),resource_snapshot_sha256=sha(snapshot),jobs=[clean(j) for j in jobs],
        input_hashes=inputs,D0_per_new_job=1,coordination='none',GPU_workers=1,proposal_only=proposal_only)
    write(ledger/'registration.json',definition)
    work=[];receipts=[]
    for j in jobs:
        j=dict(j);dest=resolve(j['out_dir']);proposal=resolve(j.get('proposal_dir',dest.parent/(dest.name+'_official_D0')))
        j['proposal_dir']=str(proposal)
        if (proposal/'receipt.json').exists():
            pr=read(proposal/'receipt.json')
            verify({str(proposal/'proposal.npz'):pr['output_sha256']})
            if pr.get('model_forwards')!=1 or pr.get('raw_sha256')!=sha(j['raw_path']) or pr.get('checkpoint_sha256')!=OFFICIAL_SHA:raise ValueError('Saved resident D0 identity differs')
        elif proposal.exists():
            raise RuntimeError('Incomplete D0 attempt; no silent retry: '+str(proposal))
        else:work.append(j)
    model=None;new_forwards=0;load_seconds=0.;tick=time.perf_counter()
    if work:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        from meshflow_control.denoise_model import load_denoiser,predict_proposal
        import torch
        t=time.perf_counter();model,model_identity=load_denoiser(checkpoint)
        load_seconds=time.perf_counter()-t
        write(ledger/'resident_model.json',dict(identity=model_identity,load_seconds=load_seconds,model_resident_instances=1))
    try:
        for original in jobs:
            j=dict(original);dest=resolve(j['out_dir']);proposal=resolve(j.get('proposal_dir',dest.parent/(dest.name+'_official_D0')))
            if not (proposal/'receipt.json').exists():
                # This is an already authorized real sample, never a probe.
                raw=arrays(j['raw_path'])['raw'].reshape(-1,3,3)
                source_metadata(j['condition_path'],raw[:int(j['K'])])
                write(proposal/'started.json',dict(request_id=j['request_id'],raw_sha256=sha(j['raw_path']),
                    checkpoint_sha256=OFFICIAL_SHA,event='D0_FORWARD_STARTED',automatic_retry=False))
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();ft=time.perf_counter()
                output=predict_proposal(model,raw)
                torch.cuda.synchronize();seconds=time.perf_counter()-ft
                if output.dtype!=np.float32 or output.shape!=raw.shape or not np.isfinite(output).all():raise ValueError('Invalid official D0 proposal')
                pp=proposal/'proposal.npz'
                with pp.open('wb') as f:np.savez_compressed(f,official_unconstrained_proposal=output)
                receipt=dict(status='PASS',request_id=j['request_id'],raw_sha256=sha(j['raw_path']),checkpoint_sha256=OFFICIAL_SHA,
                    output_sha256=sha(pp),model_forwards=1,source='OFFICIAL_D0_RESIDENT',seconds=seconds,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_scope='Reset immediately before this actual single predict_proposal; resident D0 retained; not full process VRAM')
                write(proposal/'receipt.json',receipt);new_forwards+=1
            if proposal_only:
                result=dict(status='PROPOSAL_COMPLETE',raw_path=j['raw_path'],condition_path=j['condition_path'],
                    K=j['K'],out_dir=str(dest),proposal_path=str(proposal/'proposal.npz'),proposal_receipt=str(proposal/'receipt.json'),
                    proposal_sha256=sha(proposal/'proposal.npz'),model_forwards=1,post_status='NOT_RUN_CPU_STAGE_FOLLOWS')
            else:
                result=postprocess_one(j['raw_path'],j['condition_path'],j['K'],dest,proposal_path=proposal/'proposal.npz',
                    proposal_receipt=proposal/'receipt.json',resume=dest.exists())
            result.update(request_id=j['request_id'],label=j.get('label'),upstream_proposal_path=str(proposal/'proposal.npz'),
                upstream_receipt_path=str(proposal/'receipt.json'))
            receipts.append(result)
            print(json.dumps(dict(stage='OFFICIAL_D0_GEO_C_SOURCE',completed=len(receipts),registered=len(jobs),new_D0_forwards=new_forwards)),flush=True)
        verify(inputs)
        result=dict(status='COMPLETE',jobs=len(jobs),receipts=receipts,new_D0_forwards=new_forwards,
            official_D0_forward_calls_in_all_jobs=len(jobs),resident_models_loaded=int(bool(work)),GPU_workers=int(bool(work)),
            load_seconds=load_seconds,seconds=time.perf_counter()-tick,training_updates=0,new_noise=0,coordination='none')
        write(ledger/'completion.json',result);return result
    except BaseException as error:
        failure=ledger/('failure_'+str(time.time_ns())+'.json')
        write(failure,dict(status='STOPPED',error=repr(error),new_D0_forwards_completed=new_forwards,
            receipts_completed=len(receipts),automatic_retry=False))
        raise
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()
