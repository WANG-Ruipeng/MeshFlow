"""Fixed-budget, zero-training GPU guidance and protected-weld experiment."""
from .stable_runtime import configure_stable_runtime
import argparse,copy,gc,hashlib,json,platform,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
import yaml
from .guided_native_geometry_run import (PROJECT,REPO,SOURCE,CHECKPOINT,CONFIG,EXPECTED_CHECKPOINT_SHA,
    EXPECTED_STATE_SHA,prepare_assets,record,verify_records,assert_frozen)
from .official_wrapper import state_sha256
from .learning_data import array_hash as archive_hash
from .gpu_guidance_sampling import PrevalidatedNative,sample_gpu,masks,integrate,clip_displacement
DEV=(9201,9202,9203,9204);CONF=tuple(range(9401,9409));MODES=('G0','GA','GR','GAR')
AUTH=Path('/mnt/c/Users/warpwang/.codex/attachments/039ebd4f-1c76-413b-8807-4e49320fd3b7/Pasted text.txt')


def save(path,value):
    tmp=Path(path).with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def protection(out):
    paths=[p for p in (PROJECT/'outputs').rglob('*') if p.is_file() and not p.is_relative_to(out)]
    paths.extend(p for p in PROJECT.glob('*.py') if not p.name.startswith('gpu_guidance_'))
    paths.extend((REPO/'models').rglob('*.py'));paths.extend([CONFIG,Path('/mnt/c/Users/warpwang/Downloads/last.pt')])
    return {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(paths))}


def protocol():
    return dict(schema='gpu_guidance_protected_weld_v1',authorization=record(AUTH),checkpoint_sha256=EXPECTED_CHECKPOINT_SHA,
        N=112,K=12,N_free=100,development_seeds=list(DEV),confirmation_seeds=list(CONF),
        seed_prior_scan=dict(scope='Project existing py/md/json',numeric_matches='9408 file byte sizes and9402 broadphase count; not seeds',seed_or_noise_context_matches=0),
        geometry=dict(normalization='Y=current free Xt/L; Cnorm=C/L for calculations only',dtype='FP32 CUDA, autocast disabled',
            q_formula='sum squared edge lengths/(4*sqrt(3)*area)',q_cut=10.,area_floor=1e-10,q_min=1e-12,
            actual_near_degenerate_area_normalized=1e-12,distance_denominator_floor=1e-30,
            EA='mean(.5*relu(log(q/10))**2)',ER='mean(exp(-minimum finite triangle distance squared/.005**2))',
            sigma=.005,sample_barycentric='3 corners,3 edge midpoints,centroid; all1/7 differentiable',
            GAR='EA+ER; sum separately computed component gradients; one combined clipping budget',
            lambda_before_k40=0.,lambda_k40_through49=1.,dt=.02,steps=50,rho_fraction=.001,
            FP32_clip_comparison_relative_tolerance=1e-6,no_backtracking=True,no_collision_acceptance=True),
        numerical_probe_tolerances=dict(FP64_finite_difference_step=1e-6,FP64_atol=2e-7,FP64_rtol=2e-4,
            FP32_energy_atol=2e-6,FP32_energy_rtol=2e-5,FP32_gradient_atol=2e-5,FP32_gradient_rtol=2e-3,
            actual_guided_repeat='Bitwise entire51-state equality required'),
        native_execution=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',deterministic=True,warn_only=False,TF32=False,SDPA='math',
            model_autocast='BF16',parameters_C_noise_states='FP32',frozen_eval=True,
            prevalidated_wrapper='Identical native arithmetic; immutable y/mask value checks before timing, identity/version guards during loop; original source unchanged',
            original_vs_prevalidated_same_input_equivalence='Bitwise required',G0_replays='P0/P1 seed9201 entire51 states bitwise'),
        warmup=dict(extra_model_equivalence_forwards=2,extra_model_warmup_forwards=2,
            extra_geometry_warmup='2 invocations each GA/GR/GAR on archived P0k40;6 energy forwards and8 coordinate backwards',
            repeated_trajectory='Formal GAR/P0/9201 reused first, exactly1 additional complete repeat maximum'),
        budgets=dict(new_full_generations=59,development=24,baseline_replay=2,guided_repeat=1,confirmation=32,
            full_generation_model_forwards_max=2950,extra_model_forwards=4,total_model_forwards_max=2954,
            model_backwards=0,model_parameter_updates=0,optimizer_updates=0),
        weld=dict(epsilon_fraction=.001,protected_coordinates='Only exact FP32 bit patterns share index; distinct C never tolerance-merged',
            anchor_tie='Nearest distance then lexicographic protected coordinate; no arbitrary face-ID anchor choice',
            free_order='Coordinate lexicographic, then unoriented incident triangle geometry, ordinal only for geometrically identical corners',
            free_clustering='Nearest fixed representative within epsilon, lexicographic ties; representative never moves; rejected point becomes own representative',
            local_guards=['No protected-coordinate conflict','No repeated face indices','No new area<=1e-12L squared','Normal dot originalRaw >0'],
            global_collision_guard=False,GT_inputs=False,no_hole_filling=True,no_face_deletion=True,no_edge_splitting=True),
        audit=dict(static_C_C='Each parent66 unique pairs once, then hash-checked static reuse',
            C_free='All1200 pairs per endpoint',free_free='All4950 unique pairs per endpoint',
            topology='Exact indexed edges,link,orientation,components,duplicates,Tjunction scope explicitly recorded',
            CCD='NOT_RUN',scale_scope='Known object full bbox metadata; unknown-full-scale deployment NOT_RUN'),
        selection=dict(raw_only=True,raw_R12_parent_mean_max_ratio=1.05,no_added_degenerate_or_nonfinite=True,C_preserved=True,
            option_a='Both parents CF affected-face fraction decreases>=25% and mean FF illegal pairs does not increase',
            option_b='Both parents free min-angle<5deg face fraction decreases>=.02 and mean CF affected-face fraction and FF illegal pairs do not increase',
            tie_order=['Macro CF affected-face fraction','Macro FF illegal pairs','Macro R12 symmetric RMS','Measured full-generation wall time'],
            at_most_one=True,save_selection_before_confirmation=True),
        output_scope='One report/results/code.diff,protocol.json/selection.json,fixed fixtures,raw/W output and fixedseed9201 figures',
        confirmed_absences=['No training','No context guidance','No boundary embedding','No backbone backward','No GT sampling gradient','No Phase3'])


def initialize(out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);p=protocol();save(out/'protocol.json',p)
    counts=dict(model_forward_attempts=0,model_forward_returns=0,model_backward_calls=0,optimizer_updates=0,model_parameter_updates=0,
        forward_by_category={},generation_attempts=0,complete_generations=0,generation_attempts_by_category={},completed_by_category={},euler_updates=0,
        active_forward_category='NOT_RUNNING')
    result=dict(status='INITIALIZING',protocol_path=str(out/'protocol.json'),protocol_sha256=record(out/'protocol.json')['sha256'],
        protocol=p,counts=counts,geometry_ledger={},evaluation_ledger={},scores=[],generated=[],weld=[],binding=[],raw_registry=[],
        probes={},gates={},promotion=dict(status='NOT_RUN'),confirmation=dict(status='NOT_RUN'),
        protection=dict(metadata_before=protection(out),scope='Historical metadata plus separately byte-hashed used critical inputs'),
        initial_git_status='?? experiments/',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        resource_snapshot=dict(GPU='NVIDIA GeForce RTX5080',driver='591.86',total_MiB=16303,used_MiB=4096,utilization_pct=1))
    save(out/'results.json',result)
    bank,result['assets']=prepare_assets()
    result['status']='ASSETS_PASS';save(out/'results.json',result)
    print(json.dumps(dict(event='assets_pass',checkpoint=result['assets']['checkpoint']['sha256'],protocol=result['protocol_sha256'])),flush=True)
    return result



def serial(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().item() if v.numel()==1 else v.detach().cpu().tolist()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    return v


def checkpoint_result(r,out,status=None):
    if status:r['status']=status
    save(out/'results.json',r)


def event(name,**kw):print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)


def formal_probes(r,bank,out):
    from . import gpu_guidance_energies as energy
    from . import gpu_guidance_protected_weld as weld
    from . import gpu_guidance_evaluation as evaluation
    configure_stable_runtime();r['geometry_ledger']=energy.initial_energy_ledger()
    ledger=r['geometry_ledger']
    r['implementation_development_counts']=dict(energy_fixture_runs=1,energy_forwards=38,geometry_backwards=48,
        CPU_energy_forwards=24,CUDA_energy_forwards=14,CPU_geometry_backwards=24,CUDA_geometry_backwards=24,
        independent_numpy_energy_reference_evaluations=57,primary_pair_tests=7,independent_pair_tests=7,
        direct_finite_triangle_probe=1,evaluation_helper_runs=1,evaluation_fixture_families=9,
        W_fixture_runs=2,W_calls=24,topology_audits=8,direct_protected_guards=2,Raw_exact_index=2,temporary_OBJ_roundtrips=2,
        W_candidates=90,W_accepted=84,W_rejected=6,failed_executions=0,model_forwards=0,model_backwards=0,optimizer_updates=0,
        scope='Development checks before frozen formal fixtures, separately disclosed; not formal samples')
    report,arrays=energy.run_fixed_probes(ledger=ledger,gpu=True);r['probes']['energy_fixed']=report
    wr,wa=weld.run_cpu_fixtures(output=out/'fixtures');r['probes']['weld_fixed']=wr
    arrays.update({'W_'+k:v for k,v in wa.items()})
    r['probes']['evaluation_helpers']=evaluation.run_cpu_tests()
    r['probes']['GT_compatibility']={}
    for parent in (0,1):
        case=bank['cases_by_key'][parent,12];L=case['bbox_diagonal'];C=case['constraints'];free=case['free_target'].reshape(100,3,3)
        snap=energy.diagnostic_snapshot(torch.from_numpy(free.copy()).cuda()/L,torch.from_numpy(C.copy()).cuda()/L,ledger=ledger)
        t,g=snap['terms'],snap['gradients'];gr=g['g_R'].cpu().numpy()
        shared=(free.reshape(-1,1,3)==C.reshape(1,-1,3)).all(-1).any(-1).reshape(100,3)
        sharedgrad=np.linalg.norm(gr[shared],axis=-1)
        r['probes']['GT_compatibility'][f'P{parent}']=dict(E_A=serial(t['E_A']),E_R=serial(t['E_R']),gradient_A_norm=serial(g['norm_A']),
            gradient_R_norm=serial(g['norm_R']),q_above_cut_count=serial(t['q_above_cut_count']),area_floor_count=serial(t['area_floor_count']),
            gradient_cosine=serial(g['cosine']),shared_vertex_count=int(shared.sum()),
            shared_vertex_gradient_R_RMS=float(np.sqrt(np.mean(sharedgrad**2))) if sharedgrad.size else None,
            shared_vertex_gradient_R_max=float(sharedgrad.max()) if sharedgrad.size else None,
            all_free_vertex_gradient_R_max=float(np.linalg.norm(gr,axis=-1).max()),
            gradient_finite=serial(g['gradient_finite']),scope='GT C/free compatibility diagnostic only, never fed into sampling or W')
        arrays[f'GT_P{parent}_g_A']=g['g_A'].cpu().numpy();arrays[f'GT_P{parent}_g_R']=gr
        arrays[f'GT_P{parent}_shared_C_vertex_mask']=shared
    np.savez_compressed(out/'fixed_probes.npz',**arrays)
    r['probes']['arrays']=record(out/'fixed_probes.npz')
    with np.load(bank['archived_paths'][0,9201]) as f:state=f['path'][40].copy()
    case=bank['cases_by_key'][0,12];L=case['bbox_diagonal']
    Y=(torch.from_numpy(state[12:].reshape(100,3,3)).cuda()/L).detach().requires_grad_(True)
    Cn=torch.from_numpy(case['constraints'].copy()).cuda()/L
    terms=energy.energy_terms(Y,Cn,mode='GAR',ledger=ledger)
    pending=[terms['E_selected'].grad_fn];seen=set();leaves=[]
    while pending:
        node=pending.pop()
        if node is None or node in seen:continue
        seen.add(node)
        if hasattr(node,'variable'):leaves.append(node.variable)
        pending.extend(n[0] for n in node.next_functions)
    if len(leaves)!=1 or leaves[0] is not Y:raise AssertionError('Unexpected geometry graph leaf')
    grad=energy.gradients_from_terms(terms,Y,mode='GAR',ledger=ledger)
    if not bool(grad['gradient_finite']):raise AssertionError('Nonfinite graph gate')
    r['gates']['geometry_graph']=dict(status='PASS',leaf_count=1,only_leaf='Detached current free Y',C_requires_grad=Cn.requires_grad,model_graph=False)
    r['probes']['formal_ledger_after_probes']=dict(ledger)
    r['new_core_sources']={str(p):record(p) for p in PROJECT.glob('gpu_guidance_*.py') if p.name!='gpu_guidance_report.py'}
    checkpoint_result(r,out,'PROBES_PASS');event('formal_probes_pass',GT=r['probes']['GT_compatibility'],model_forwards=r['counts']['model_forward_attempts'])


def load_model(r):
    from models.equidit import DiT
    from models.utils import get_embedder
    c=yaml.safe_load(CONFIG.read_text());cm=dict(c['model']);cm.pop('model_type')
    required=dict(version=3,hidden_dim=768,num_layers=12,num_heads=12,max_length=800,face_bin=20,use_qknorm=True,
        use_rmsnorm=True,use_coord_encoding=True,use_dit_like_pe=False,face_cond=True)
    if any(cm.get(k)!=v for k,v in required.items()) or c['transport']['prediction']!='velocity':raise RuntimeError('Architecture mismatch')
    p=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    if (p['arm'],p['completed_updates'],p['schema'])!=('Native_correct',500,'native_connected_patch_v1'):raise RuntimeError('Wrong checkpoint')
    with torch.device('meta'):model=PrevalidatedNative(DiT(**cm),trainable=False)
    model.backbone.x_embedder.embed_fn=get_embedder(cm['pe_freq'],input_dims=3)[0]
    keys=model.load_state_dict(p['model'],strict=True,assign=True);model.set_trainable(False).eval();del p
    sha=state_sha256(model)
    if sha!=EXPECTED_STATE_SHA:raise RuntimeError('State SHA mismatch')
    model.cuda();assert_frozen(model)
    counts=r['counts']
    def before(module,args):
        if counts['model_forward_attempts']>=2954:raise RuntimeError('Hard model forward cap2954')
        assert_frozen(module);counts['model_forward_attempts']+=1
        category=counts['active_forward_category'];counts['forward_by_category'][category]=counts['forward_by_category'].get(category,0)+1
    def after(module,args,value):
        if value.requires_grad:raise RuntimeError('Backbone graph created')
        counts['model_forward_returns']+=1
    model.register_forward_pre_hook(before);model.register_forward_hook(after)
    r['gates']['strict_load']=dict(status='PASS',strict=True,missing_keys=keys.missing_keys,unexpected_keys=keys.unexpected_keys,
        state_sha256=sha,parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameter_count=0)
    r['gates']['initial_state_hash']=sha
    return model


def model_gates_and_warmup(r,bank,out,model):
    from .native_inpainting_model import NativeInpaintingModel,native_execution
    from .gpu_guidance_energies import energy_terms,gradients_from_terms
    counts=r['counts'];case=bank['cases_by_key'][0,12];L=case['bbox_diagonal']
    with np.load(bank['archived_paths'][0,9201]) as f:state=torch.from_numpy(f['path'][40:41].copy()).cuda()
    y,valid,known=masks(state.device);model.bind_contract(y,valid,known);t=torch.full((1,),.8,device='cuda')
    with torch.no_grad(),native_execution(model,coordinates=state) as facts:
        counts['active_forward_category']='original_forward_equivalence'
        counts['model_forward_attempts']+=1;counts['forward_by_category']['original_forward_equivalence']=1
        original=NativeInpaintingModel.forward(model,state,t,y,valid,known)
        if original.requires_grad:raise RuntimeError('Original graph')
        counts['model_forward_returns']+=1
        counts['active_forward_category']='prevalidated_forward_equivalence';actual=model(state,t,y,valid,known)
        if not torch.equal(original,actual):raise RuntimeError('Prevalidated wrapper arithmetic differs')
        r['gates']['original_forward_equivalence']=dict(status='PASS',bitwise=True,extra_model_forwards=2)
        counts['active_forward_category']='model_warmup'
        for _ in range(2):model(state,t,y,valid,known)
        r['gates']['runtime']=facts
    cn=torch.from_numpy(case['constraints'].copy()).cuda()/L
    before=dict(r['geometry_ledger'])
    for mode in ('GA','GR','GAR'):
        for _ in range(2):
            Y=(state[0,12:].reshape(100,3,3)/L).detach().requires_grad_(True)
            terms=energy_terms(Y,cn,mode=mode,ledger=r['geometry_ledger'])
            gradients_from_terms(terms,Y,mode=mode,ledger=r['geometry_ledger'])
    torch.cuda.synchronize()
    r['warmup']=dict(extra_model_forwards=2,geometry_ledger_delta={k:v-before.get(k,0) for k,v in r['geometry_ledger'].items()})
    r['gates']['G0_replays']=[]
    for parent in (0,1):
        case=bank['cases_by_key'][parent,12]
        row=sample_gpu(model,bank['noises'][9201],case['constraints'],case['bbox_diagonal'],'G0',counts,out,
            f'replay_G0_P{parent}_seed9201','baseline_replay',r['geometry_ledger'])
        with np.load(row['raw_path']) as a,np.load(bank['archived_paths'][parent,9201]) as b:equal=np.array_equal(a['path'],b['path'])
        row.update(parent=parent,seed=9201,all51_bitwise_equal_to_archive=bool(equal));r['gates']['G0_replays'].append(row)
        checkpoint_result(r,out)
        if not equal:raise RuntimeError('G0 historical full trajectory replay failed')
        event('G0_replay_pass',parent=parent,nfe=50,seconds=row['costs']['complete_generation_wall_seconds'])


def generate_development(r,bank,out,model):
    for parent in (0,1):
        for seed in DEV:r['raw_registry'].append(dict(phase='development',model='G0',parent=parent,seed=seed,
            raw_path=str(bank['archived_paths'][parent,seed]),archived_reuse=True))
    # Fixed order: GAR first permits repeat gate before all other formal outputs.
    r['development_execution_order']=['GAR','GA','GR']
    for mode in ('GAR','GA','GR'):
        for parent in (0,1):
            case=bank['cases_by_key'][parent,12]
            for seed in DEV:
                row=sample_gpu(model,bank['noises'][seed],case['constraints'],case['bbox_diagonal'],mode,r['counts'],out,
                    f'development_{mode}_P{parent}_seed{seed}','development',r['geometry_ledger'])
                row.update(phase='development',model=mode,parent=parent,seed=seed)
                r['generated'].append(row);r['raw_registry'].append({k:row[k] for k in ('phase','model','parent','seed','raw_path')})
                with np.load(row['raw_path']) as a,np.load(bank['archived_paths'][parent,seed]) as b:offequal=np.array_equal(a['path'][:41],b['path'][:41])
                row['off_first41_states_bitwise']=bool(offequal)
                checkpoint_result(r,out,'DEVELOPMENT_RUNNING')
                if not offequal:raise RuntimeError('Guidance-off prefix differs from G0')
                event('development_output',model=mode,parent=parent,seed=seed,seconds=row['costs']['complete_generation_wall_seconds'],model_forwards=r['counts']['model_forward_attempts'])
                if mode=='GAR' and parent==0 and seed==9201:
                    repeat=sample_gpu(model,bank['noises'][seed],case['constraints'],case['bbox_diagonal'],mode,r['counts'],out,
                        'repeat_GAR_P0_seed9201','guided_repeat',r['geometry_ledger'])
                    with np.load(row['raw_path']) as a,np.load(repeat['raw_path']) as b:
                        equal=np.array_equal(a['path'],b['path']);diag=np.array_equal(a['geometry_diagnostics'],b['geometry_diagnostics'])
                    repeat.update(all51_bitwise_equal=bool(equal),geometry_diagnostics_bitwise_equal=bool(diag))
                    r['gates']['guided_repeat']=repeat;checkpoint_result(r,out)
                    if not equal or not diag:raise RuntimeError('Actual GPU guided trajectory repeat failed')
                    event('guided_repeat_pass',model_forwards=r['counts']['model_forward_attempts'])
    r['raw_before_W_development']=dict(status='PASS',all32_ready=True,records=[record(x['raw_path']) for x in r['raw_registry'] if x['phase']=='development'])
    checkpoint_result(r,out,'DEVELOPMENT_RAW_COMPLETE')


def endpoint_energy(output,C,L,ledger):
    from .gpu_guidance_energies import energy_terms
    Y=(torch.from_numpy(np.asarray(output,np.float32)[12:].reshape(100,3,3).copy()).cuda()/L).detach().requires_grad_(True)
    cn=torch.from_numpy(C.copy()).cuda()/L
    t=energy_terms(Y,cn,mode='GAR',ledger=ledger)
    return dict(E_A=serial(t['E_A']),E_R=serial(t['E_R']),q_above_cut_count=serial(t['q_above_cut_count']),
        area_floor_count=serial(t['area_floor_count']),geometry_backwards=0,scope='Offline endpoint values only; no correction or selection beyond registered geometric metrics')


def score_phase(r,bank,out,phase):
    from . import gpu_guidance_evaluation as ev
    from . import gpu_guidance_protected_weld as w
    ledger=r['evaluation_ledger']
    if 'context_audits' not in r:
        r['context_audits']={str(p):ev.audit_context(bank['cases_by_key'][p,12]['constraints'],bank['cases_by_key'][p,12]['bbox_diagonal'],ledger) for p in (0,1)}
    bank['context_audits']={int(p):v for p,v in r['context_audits'].items()}
    registry=[q for q in r['raw_registry'] if q['phase']==phase]
    if any(not Path(q['raw_path']).is_file() for q in registry):raise RuntimeError('Raw-before-W gate failed')
    (out/'meshes').mkdir(exist_ok=True);(out/'welded').mkdir(exist_ok=True)
    for src in registry:
        parent,seed,mode=src['parent'],src['seed'],src['model'];case=bank['cases_by_key'][parent,12];C=case['constraints'];L=case['bbox_diagonal']
        with np.load(src['raw_path']) as f:raw=f['output'].copy()
        key=f'{phase}_{mode}_P{parent}_seed{seed}'
        for variant in ('Raw','W'):
            tick=time.perf_counter()
            indexed=w.raw_exact_index(raw,C) if variant=='Raw' else w.protected_weld(raw,C,L)
            operator_seconds=time.perf_counter()-tick
            ledger['raw_index_calls' if variant=='Raw' else 'W_operator_calls']=ledger.get('raw_index_calls' if variant=='Raw' else 'W_operator_calls',0)+1
            output=indexed['output'];tick=time.perf_counter()
            topology=w.audit_indexed_topology(indexed['vertices'],indexed['faces'],L,C=C)
            topo_seconds=time.perf_counter()-tick;ledger['indexed_topology_audits']=ledger.get('indexed_topology_audits',0)+1
            export=w.export_protected_obj(out/'meshes'/f'{key}_{variant}.obj',indexed,C)
            ledger['OBJ_FP32_roundtrip_checks']=ledger.get('OBJ_FP32_roundtrip_checks',0)+1
            path=src['raw_path']
            if variant=='W':
                path=str(out/'welded'/f'{key}_W.npz')
                np.savez_compressed(path,output=output,vertices=indexed['vertices'],faces=indexed['faces'])
                r['weld'].append(dict(phase=phase,model=mode,parent=parent,seed=seed,welded_path=path,audit=indexed['audit'],
                    indexed_topology=topology,operator_seconds=operator_seconds,index_export_seconds=export['export_seconds'],topology_audit_seconds=topo_seconds,
                    export=export,file=record(path)))
            metric=ev.score_variant(output,bank,parent,ledger,topology=dict(indexing=indexed['audit'],topology=topology))
            metric['gpu_endpoint_energy']=endpoint_energy(output,C,L,r['geometry_ledger'])
            row=dict(phase=phase,model=mode,parent=parent,seed=seed,K=12,variant=variant,metrics=metric,output_path=path,raw_path=src['raw_path'],
                indexing_seconds=operator_seconds if variant=='Raw' else 0,topology_audit_seconds=topo_seconds,export=export)
            r['scores'].append(row)
            if mode=='G0' and phase=='development' and variant=='Raw':
                old=bank['source_scores'][parent,seed]
                oldr=old['remainder']['own']['symmetric_rms_bbox_pct'];newr=metric['remainder']['own']['symmetric_rms_bbox_pct']
                if abs(newr-oldr)>1e-11:raise RuntimeError('G0 historical metric mismatch')
                row['archived_R12_reconciliation']=dict(status='PASS',old=oldr,new=newr)
            checkpoint_result(r,out,f'{phase.upper()}_SCORING')
            event('endpoint_scored',phase=phase,model=mode,parent=parent,seed=seed,variant=variant,
                R12=metric['remainder']['own']['symmetric_rms_bbox_pct'],CF=metric['guided_geometry_audit']['C_free']['proper_affected_face_fraction'],
                closure=metric['gpu_guidance_evaluation']['COMBINATORIAL_CLOSURE_STATUS'])
    rows=[q for q in r['scores'] if q['phase']==phase];seeds=DEV if phase=='development' else CONF
    for mode in sorted({q['model'] for q in rows}):
        for variant in ('Raw','W'):
            for seed in seeds:
                mm=[next(q for q in rows if (q['model'],q['variant'],q['parent'],q['seed'])==(mode,variant,p,seed))['metrics'] for p in (0,1)]
                r['binding'].append(dict(phase=phase,model=mode,variant=variant,seed=seed,metrics=ev.cross_binding(*mm)))
    r.setdefault('comparisons',[])
    for row in rows:
        match=lambda mode,variant:next(q for q in rows if (q['model'],q['variant'],q['parent'],q['seed'])==(mode,variant,row['parent'],row['seed']))
        if row['variant']=='W':
            r['comparisons'].append(dict(phase=phase,model=row['model'],parent=row['parent'],seed=row['seed'],kind='W_vs_own_Raw',
                metrics=ev.compare_variants(match(row['model'],'Raw')['metrics'],row['metrics'])))
        if row['model']!='G0':
            r['comparisons'].append(dict(phase=phase,model=row['model'],parent=row['parent'],seed=row['seed'],kind=f"{row['variant']}_vs_G0_same_layer",
                metrics=ev.compare_variants(match('G0',row['variant'])['metrics'],row['metrics'])))
    checkpoint_result(r,out,f'{phase.upper()}_SCORED')


def selection(r,out):
    from .guided_native_geometry_run import metric_flags
    rows=[q for q in r['scores'] if q['phase']=='development' and q['variant']=='Raw']
    def statistics(mode,parent):
        subset=[q for q in rows if q['model']==mode and q['parent']==parent]
        if len(subset)!=4:raise AssertionError('Four fixed dev noises required')
        flags=[metric_flags(q['metrics']) for q in subset]
        return dict(R12=float(np.mean([f['R12_symmetric_rms'] for f in flags])),CF=float(np.mean([f['proper_face_fraction'] for f in flags])),
            FF=float(np.mean([f['free_free_illegal'] for f in flags])),
            narrow=float(np.mean([q['metrics']['gpu_guidance_evaluation']['free_shape_quality']['min_angle_lt5_fraction'] for q in subset])),
            all_C=all(f['C_exact'] for f in flags),degenerate_by_seed={str(q['seed']):f['degenerate'] for q,f in zip(subset,flags)})
    baselines={str(p):statistics('G0',p) for p in (0,1)};decisions={}
    for mode in ('GA','GR','GAR'):
        stats={str(p):statistics(mode,p) for p in (0,1)};checks={}
        for p in (0,1):
            a,b=stats[str(p)],baselines[str(p)]
            checks[str(p)]=dict(R12_within5pct=a['R12']<=1.05*b['R12'],C_exact=a['all_C'],
                no_added_degeneracy=all(a['degenerate_by_seed'][str(s)]<=b['degenerate_by_seed'][str(s)] for s in DEV),
                option_A=a['CF']<=.75*b['CF'] and a['FF']<=b['FF'],
                option_B=a['narrow']<=b['narrow']-.02 and a['CF']<=b['CF'] and a['FF']<=b['FF'])
        hard=all(all(c[k] for k in ('R12_within5pct','C_exact','no_added_degeneracy')) for c in checks.values())
        a=all(c['option_A'] for c in checks.values());b=all(c['option_B'] for c in checks.values())
        times=[q['costs']['complete_generation_wall_seconds'] for q in r['generated'] if q['model']==mode and q['phase']=='development']
        decisions[mode]=dict(pass_gate=bool(hard and (a or b)),hard_gates_pass=hard,option_A_allparents=a,option_B_allparents=b,parents=stats,checks=checks,
            ranking_key=[float(np.mean([x['CF'] for x in stats.values()])),float(np.mean([x['FF'] for x in stats.values()])),
                float(np.mean([x['R12'] for x in stats.values()])),float(np.mean(times))])
    eligible=[m for m,v in decisions.items() if v['pass_gate']]
    chosen=min(eligible,key=lambda m:decisions[m]['ranking_key']) if eligible else None
    result=dict(status='SELECTED' if chosen else 'NO_QUALIFIED_CANDIDATE',candidate=chosen,eligible=eligible,baseline_parents=baselines,
        decisions=decisions,raw_only=True,selection_saved_before_any_new_noise=True,confirmation_seeds=list(CONF),
        no_nonfinite='All51-state finite runtime assertions passed',selected_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(out/'selection.json',result);r['promotion']=dict(result,selection_file=record(out/'selection.json'));checkpoint_result(r,out,'SELECTION_SAVED')
    event('selection',candidate=chosen,decisions={k:v['pass_gate'] for k,v in decisions.items()})
    return chosen


def confirmation(r,bank,out,model,chosen):
    if chosen is None:
        r['confirmation']=dict(status='NOT_RUN',reason='No Raw candidate passed preregistered two-parent gate',model_forwards=0,new_noise_generated=False)
        return
    r['confirmation']=dict(status='RUNNING',candidate=chosen,seeds=list(CONF),selection_sha256=r['promotion']['selection_file']['sha256'])
    inputs={}
    for seed in CONF:
        gen=torch.Generator(device='cpu').manual_seed(seed);noise=torch.randn((1,112,9),generator=gen,dtype=torch.float32).numpy()
        inputs[f'noise_seed{seed}']=noise;bank['noises'][seed]=noise
    np.savez_compressed(out/'confirmation_independent_Gaussian.npz',**inputs)
    r['confirmation']['noise_file']=record(out/'confirmation_independent_Gaussian.npz')
    for mode in ('G0',chosen):
        for parent in (0,1):
            case=bank['cases_by_key'][parent,12]
            for seed in CONF:
                row=sample_gpu(model,bank['noises'][seed],case['constraints'],case['bbox_diagonal'],mode,r['counts'],out,
                    f'confirmation_{mode}_P{parent}_seed{seed}','confirmation',r['geometry_ledger'])
                row.update(phase='confirmation',model=mode,parent=parent,seed=seed)
                r['generated'].append(row);r['raw_registry'].append({k:row[k] for k in ('phase','model','parent','seed','raw_path')})
                checkpoint_result(r,out,'CONFIRMATION_RUNNING');event('confirmation_output',model=mode,parent=parent,seed=seed)
    r['raw_before_W_confirmation']=dict(status='PASS',all32_ready=True,records=[record(x['raw_path']) for x in r['raw_registry'] if x['phase']=='confirmation'])
    checkpoint_result(r,out,'CONFIRMATION_RAW_COMPLETE');score_phase(r,bank,out,'confirmation')
    r['confirmation']['status']='PASS_EXECUTION';r['confirmation']['scope']='New independent noises on the same two TRAIN parents, not new-parent generalization'


def finalize(r,out,model):
    assert_frozen(model);sha=state_sha256(model);r['gates']['final_state_hash']=sha
    if sha!=EXPECTED_STATE_SHA:raise AssertionError('Frozen state changed')
    r['protection']['critical_files_after']=verify_records(r['assets']['critical_files'])
    after=protection(out);before=r['protection']['metadata_before']
    changed=[p for p,v in before.items() if after.get(p)!=v]
    r['protection']['historical_metadata_after']=dict(status='PASS' if not changed else 'FAIL',files=len(before),changed=changed)
    r['protection']['new_core_source_check']=verify_records(r['new_core_sources'])
    if changed or any(r['protection'][k]['status']!='PASS' for k in ('critical_files_after','new_core_source_check')):raise RuntimeError('Immutable file check failed')
    counts=r['counts'];counts['active_forward_category']='FINISHED'
    if counts['model_forward_attempts']!=counts['model_forward_returns'] or counts['generation_attempts']!=counts['complete_generations']:raise RuntimeError('Unaccounted invocation')
    if counts['model_forward_attempts']!=50*counts['complete_generations']+4:raise RuntimeError('Unexpected extra model call')
    r['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());checkpoint_result(r,out,'PASS_EXECUTION')
    event('experiment_complete',counts=counts,promotion=r['promotion']['status'])


def execute(out,stage):
    if stage=='init':return initialize(out)
    r=json.loads((out/'results.json').read_text());model=None
    try:
        if record(out/'protocol.json')['sha256']!=r['protocol_sha256']:raise RuntimeError('Frozen protocol modified')
        configure_stable_runtime();bank,_=prepare_assets()
        if stage=='probes':
            if r['status']!='ASSETS_PASS':raise RuntimeError('Probes stage already executed or wrong state')
            formal_probes(r,bank,out);return
        if stage=='run':
            if r['status']!='PROBES_PASS':raise RuntimeError('Formal probes must pass before model run')
            if verify_records(r['new_core_sources'])['status']!='PASS':raise RuntimeError('Core sources changed after probes')
            model=load_model(r);model_gates_and_warmup(r,bank,out,model);generate_development(r,bank,out,model)
            score_phase(r,bank,out,'development');chosen=selection(r,out);confirmation(r,bank,out,model,chosen);finalize(r,out,model)
        elif stage=='report':
            if r['status']!='PASS_EXECUTION':raise RuntimeError('Cannot claim completed experiment')
            from .gpu_guidance_report import deliver_report
            r['delivery']=deliver_report(r,out,bank);checkpoint_result(r,out)
    except BaseException as e:
        r['failure']=dict(error=repr(e),traceback=traceback.format_exc(),stage=stage)
        checkpoint_result(r,out,'STOP_IMPLEMENTATION_OR_NUMERICAL_FAILURE');event('stopped',error=repr(e),counts=r['counts']);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['init','probes','run','report'],required=True)
    args=parser.parse_args();execute(args.out.resolve(),args.stage)
