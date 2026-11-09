"""CPU-only frozen input preparation; no sampling, model, Y or source M read."""
import collections, json
import numpy as np
from common import *

ATTACHMENT = Path('/mnt/c/Users/warpwang/.codex/attachments/587341ea-536d-4752-be9e-b1dda90cb348/Pasted text.txt')
VARIANTS = (('m095', np.float32(.95)), ('base', np.float32(1)), ('p105', np.float32(1.05)))

def affine(a, alpha):
    result = a.copy()
    if alpha != np.float32(1): result.reshape(-1, 3)[:, 0] *= alpha
    return result

def verify_file(path, expected):
    actual = sha(path)
    if actual != expected: raise ValueError('Inherited artifact hash mismatch: '+str(path))
    return actual

def choose(panel):
    candidates = []
    for row in panel['conditions']:
        verify_file(row['condition_source'], row['condition_source_sha256'])
        C = arrays(row['condition_source'])['C']
        width = float(np.ptp(C.reshape(-1, 3)[:, 0].astype(np.float64)))
        candidates.append(dict(condition_id=row['condition_id'], parent_id=row['parent_id'],
            condition_type=row['condition_type'], N=row['N'], K=row['K'], C_x_span=width,
            width_eligible=width >= .01*L, rejection_reason='' if width >= .01*L else 'C_X_SPAN_LT_0.01_L',
            source_path=row['condition_source'], source_sha256=row['condition_source_sha256']))
    selected = []; used = set(); steps = []
    for kind, count in [('SEAT_HALF',4), ('SEAT_FULL',2), ('BACK_HALF',1), ('BACK_FULL',1)]:
        pool = sorted([r for r in candidates if r['condition_type']==kind and r['width_eligible'] and r['parent_id'] not in used],
            key=lambda r:(r['K'],r['parent_id'],r['condition_id']))
        if len({r['parent_id'] for r in pool}) != len(pool):
            raise ValueError('Duplicate same-type parent candidates require an explicit selection rule')
        if len(pool)<count: raise ValueError('Insufficient qualified unused parents for '+kind)
        n = len(pool)
        ranks = [0, round((n-1)/3), round(2*(n-1)/3), n-1] if kind=='SEAT_HALF' else ([0,n-1] if kind=='SEAT_FULL' else [(n-1)//2])
        if len(set(ranks)) != count: raise ValueError('Rank selection collapsed for '+kind)
        picks = [pool[i] for i in ranks]
        for rank,r in zip(ranks,picks):
            selected.append(dict(r, selection_order=len(selected), selection_pool_size=n, selection_rank_zero_based=rank))
            used.add(r['parent_id'])
        steps.append(dict(condition_type=kind, sorted_unused_eligible_ids=[r['condition_id'] for r in pool],
            selected_ranks_zero_based=ranks, selected_ids=[r['condition_id'] for r in picks]))
    ids = {r['condition_id'] for r in selected}
    for r in candidates:
        r['selected'] = r['condition_id'] in ids
        if r['width_eligible'] and not r['selected']: r['rejection_reason']='NOT_SELECTED_BY_FIXED_RANK_AND_DISTINCT_PARENT_RULE'
    if len(selected)!=8 or len(used)!=8: raise AssertionError('Panel must have eight distinct parents')
    return selected, candidates, steps

def transformed_source(source, alpha):
    value = {k:v.copy() for k,v in source.items()}
    for key in ('C','C_original'):
        if key in value: value[key] = affine(value[key], alpha)
    # Raw source-unit coordinates remain provenance of the original asset, not
    # a new unnormalized input. The actual transformed input is C_original.
    return value

def transform_queries(oldq, C0, C, alpha, ctl, seams):
    result = {k:v.copy() for k,v in oldq.items()}
    if alpha == np.float32(1):
        return result, dict(status='BASE_ALL_ARRAYS_BYTE_IDENTICAL', weight_rule='ORIGINAL_1_OVER_2048',
            weight_denominator=float(result['C_weights'].sum()), new_random_draws=0, affine_barycentric_roundoff_max=0.)
    normal, valid, area = ctl.normals(C,L)
    _,_,area0 = ctl.normals(C0,L)
    faces = result['C_point_faces']
    if np.any(area0<=0) or np.any(area<=0): raise ValueError('Cannot importance-transform degenerate C')
    result['C_points'] = affine(oldq['C_points'], alpha)
    importance = area[faces] / area0[faces]
    unnormalized = oldq['C_weights'] * importance
    result['C_weights'] = unnormalized / unnormalized.sum()
    result['C_point_normals'] = normal[faces]
    result['C_point_normal_valid'] = valid[faces]
    result['C_open_edges'], facts = ctl.open_edges(C)
    result['C_border_distance'] = (np.sqrt(seams.nearest_segments(result['C_points'], result['C_open_edges'])[0])
        if len(result['C_open_edges']) else np.full(len(faces), np.inf))
    bary_rebuilt = np.einsum('ni,nij->nj',result['C_barycentric'],C.astype(np.float64)[faces])
    audit = dict(status='AFFINE_FROZEN_POINTS_IMPORTANCE_REWEIGHTED',
        weight_rule='w_alpha_i = (w0_i * area_alpha[face_i]/area0[face_i]) / sum_j(w0_j * area_alpha[face_j]/area0[face_j])',
        original_sampler='area-proportional face draw with original equal 1/2048 weights',
        importance_ratio_min=float(importance.min()), importance_ratio_max=float(importance.max()),
        sampled_importance_denominator=float(unnormalized.sum()), geometric_area_ratio=float(area.sum()/area0.sum()),
        weight_denominator=float(result['C_weights'].sum()), new_random_draws=0,
        point_rule='old FP64 C_points X multiplied by the same FP32 alpha; face and barycentric arrays unchanged',
        normal_rule='original normals(new FP32 C,L)', open_edge_rule='original exact C-coordinate incidence',
        border_rule='original nearest_segments finite-segment squared distance then sqrt',
        affine_barycentric_roundoff_max=float(np.max(np.abs(result['C_points']-bary_rebuilt))),
        open_edge_count=int(len(result['C_open_edges'])))
    return result, audit

def prepare():
    dest = OUT/'registration.json'
    if dest.exists():
        reg = read(dest)
        if reg['status']!='FROZEN': raise ValueError('Unexpected existing registration')
        for row in reg['requests']:
            verify_file(row['input_path'],row['input_sha256']);verify_file(row['condition_source'],row['condition_source_sha256'])
        for row in reg['queries']: verify_file(row['path'],row['sha256'])
        for path,h in reg['preparation_source_hashes'].items(): verify_file(path,h)
        return reg
    if (OUT/'raw').exists() or (OUT/'post').exists(): raise ValueError('Cannot register after scientific output directory exists')
    event('PREPARE_BEGIN',model_calls=0,training_updates=0)
    resources(OUT/'resources_prepare.json')
    panelpath=OLD/'data/panel_manifest.json'; oldregpath=OLD/'evaluation/registration.json'
    panel=read(panelpath); oldreg=read(oldregpath)
    if panel['status']!='FROZEN' or oldreg['status']!='FROZEN': raise ValueError('Unfrozen original panel')
    if len(panel['conditions'])!=64 or len({r['parent_id'] for r in panel['conditions']})!=32: raise ValueError('Wrong inherited DEV panel')
    selected,candidates,steps=choose(panel)
    write(OUT/'selection.json',dict(status='FROZEN',L=L,x_span_threshold=.01*L,
        order=['SEAT_HALF','SEAT_FULL','BACK_HALF','BACK_FULL'],halfseat_ranks='0, Python round((n-1)/3), Python round(2*(n-1)/3), n-1',
        fullseat_ranks='0,n-1 after excluding used parents',back_rank='floor((n-1)/2), lower median when even',
        sort_key=['K','parent_id','condition_id'],source=panelpath,source_sha256=sha(panelpath),candidates=candidates,steps=steps,selected=selected,
        output_metrics_read=False,generated_images_read=False,selection_uses_source_Y=False))
    csvwrite(OUT/'selection_candidates.csv',candidates);csvwrite(OUT/'selected_conditions.csv',selected)
    package()
    from meshflow_control.source_winding import validate_source
    legacy=module('_width_original_eval',ROOT/'r2/pilots/user_condition_finetune_v1/evaluation.py')
    maths=legacy.cpu_math();ctl=maths['control'];seams=maths['seams']
    inherited_queries={q['condition_id']:q for q in oldreg['queries'] if q['panel']=='NEW32'}
    conditions=[];requests=[];queries=[];source_receipts=[]
    original_rows={r['condition_id']:r for r in panel['conditions']}
    for chosen in selected:
        cid=chosen['condition_id']; row=original_rows[cid]; source=arrays(row['condition_source']);C0=source['C']
        original_validation=validate_source(source,C0)
        oldrequests=sorted([r for r in oldreg['requests'] if r['panel']=='NEW32' and r['condition_id']==cid],key=lambda r:r['candidate'])
        if [r['candidate'] for r in oldrequests]!=[0,1]: raise ValueError('Need original two noise identities')
        qr=inherited_queries[cid];verify_file(qr['path'],qr['sha256']);oldq=arrays(qr['path'])
        if not np.all(oldq['C_weights']==1/2048): raise ValueError('Unexpected inherited sampling-weight definition')
        for variant,alpha in VARIANTS:
            vid=cid+'__'+variant;src=transformed_source(source,alpha);C=src['C'];sv=validate_source(src,C)
            for key in ('C_source_corner_permutations','C_source_vertex_ids','C_original_source_vertex_ids','source_face_ids','free_source_ids'):
                if key in source and not same(source[key],src[key]): raise ValueError('Source identity/order changed')
            if not same(C[...,1:],C0[...,1:]): raise ValueError('Changed Y/Z coordinate')
            sp=OUT/'sources'/(vid+'.npz');frozen_npz(sp,src)
            qa,audit=transform_queries(oldq,C0,C,alpha,ctl,seams)
            qp=OUT/'queries'/(vid+'.npz');frozen_npz(qp,qa)
            queryrow=dict(condition_id=vid,base_condition_id=cid,parent_id=row['parent_id'],variant=variant,alpha=float(alpha),
                panel='WIDTH8',path=str(qp),sha256=sha(qp),query_count=len(qa['C_points']),weight_denominator=float(qa['C_weights'].sum()),
                array_sha256={k:array_sha(v) for k,v in qa.items()},identity=dict(condition_id=vid,C_seed=qr['identity']['C_seed'],C_sha256=array_sha(C),L0=L),
                inherited_query_path=qr['path'],inherited_query_sha256=qr['sha256'],transform_audit=audit)
            queries.append(queryrow)
            conditions.append(dict(condition_id=vid,base_condition_id=cid,parent_id=row['parent_id'],group_uid=row.get('group_uid'),
                condition_type=row['condition_type'],region_type=row['region_type'],tier=row['tier'],N=row['N'],K=row['K'],
                ratio=row['K']/row['N'],variant=variant,alpha=float(alpha),C_x_span=float(np.ptp(C.reshape(-1,3)[:,0].astype(np.float64))),
                base_C_x_span=chosen['C_x_span'],condition_source=str(sp),condition_source_sha256=sha(sp),C_sha256=array_sha(C),
                source_face_ids=row['source_face_ids'],original_source_path=row['condition_source'],original_source_sha256=row['condition_source_sha256'],
                source_quality_at_base=row.get('source_quality','UNKNOWN'),transformed_source_Y_quality='NOT_APPLICABLE_NO_Y_TRANSFORM_OR_EVALUATION',
                source_winding_validation=sv['summary'],selection_order=chosen['selection_order']))
            for oldr in oldrequests:
                verify_file(oldr['input_path'],oldr['input_sha256']);b=arrays(oldr['input_path'])
                if not same(b['C'],C0) or oldr['Gaussian_sha256']!=array_sha(b['z']): raise ValueError('Inherited C/noise identity failed')
                rid='WIDTH8__'+vid+'__n'+str(oldr['candidate']);ip=OUT/'inputs'/(rid+'.npz')
                frozen_npz(ip,dict(C=C,z=b['z'].copy(),N=b['N'].copy(),K=b['K'].copy()))
                check=arrays(ip)
                if not same(check['z'],b['z']) or not same(check['C'],C): raise ValueError('New input identity failure')
                requests.append(dict(request_id=rid,base_request_id=oldr['request_id'],base_condition_id=cid,condition_id=vid,parent_id=row['parent_id'],
                    group_uid=row.get('group_uid'),panel='WIDTH8',candidate=oldr['candidate'],region_type=row['region_type'],tier=row['tier'],condition_type=row['condition_type'],
                    N=row['N'],K=row['K'],ratio=row['K']/row['N'],variant=variant,alpha=float(alpha),input_path=str(ip),input_sha256=sha(ip),
                    C_sha256=array_sha(C),Gaussian_sha256=array_sha(b['z']),noise_seed=oldr['noise_seed'],noise_provenance='BYTE_COPY_OF_FROZEN_NEW32_ARRAY_NO_REDRAW',
                    condition_source=str(sp),condition_source_sha256=sha(sp),source_quality='NOT_REEVALUATED_TRANSFORMED_C_ONLY',
                    original_input_path=oldr['input_path'],original_input_sha256=oldr['input_sha256'],original_condition_source=row['condition_source'],
                    original_condition_source_sha256=row['condition_source_sha256'],selection_order=chosen['selection_order']))
            source_receipts.append(dict(condition_id=vid,source_validation=sv['summary'],query_audit=audit,
                C_x_only=True,source_indices_and_corner_permutations_unchanged=True,
                original_raw_coordinates='PRESERVED_SOURCE_PROVENANCE; transformed model-domain original input is C_original'))
    if len(requests)!=48 or len(queries)!=24 or len(conditions)!=24: raise AssertionError('Wrong preparation count')
    bindingpath=OLD/'evaluation/checkpoints/USER_EXPANDED_67.json';checkpoint=read(bindingpath)
    cp=resolve(checkpoint['path']);cp_receipt=read(cp.with_suffix('.json'))
    if checkpoint['global_step']!=6000 or cp_receipt['sha256']!=checkpoint['sha256'] or not cp.is_file(): raise ValueError('Wrong EXP67 fixed endpoint')
    checkpoint.update(bytes=cp.stat().st_size,identity_status='INHERITED_COMPLETED_RECEIPTS; hash and state rechecked by actual loader',binding_path=str(bindingpath),binding_sha256=sha(bindingpath))
    d0identity=OLD/'evaluation/D0/identity.json';d0=read(d0identity);dp=resolve(d0['checkpoint_path'])
    if not dp.is_file() or dp.stat().st_size!=d0['checkpoint_bytes']: raise ValueError('D0 missing or size changed')
    sources={str(p):sha(p) for p in [CODE/'common.py',CODE/'prepare.py',ROOT/'r2/src/meshflow_control/data/dataset.py',
        ROOT/'r2/src/meshflow_control/sampling.py',ROOT/'r2/src/meshflow_control/source_winding.py',
        ROOT/'r2/pilots/user_condition_finetune_v1/evaluation.py',Path('/mnt/e/MeshFlow-Control/src/meshflow_control/evaluation/control.py'),
        Path('/mnt/e/MeshFlow-Control/src/meshflow_control/evaluation/seams.py')]}
    reg=dict(version=VERSION,experiment='R2_CONDITION_WIDTH_RESPONSE_V1',status='FROZEN',created_utc=utc(),labels=['EXP67_FROZEN'],
        requests=requests,conditions=conditions,queries=queries,checkpoint=checkpoint,
        D0=dict(path=str(dp),sha256=d0['checkpoint_sha256'],bytes=dp.stat().st_size,receipt_path=str(d0identity),receipt_sha256=sha(d0identity)),
        L=L,variants=[dict(variant=v,alpha=float(a)) for v,a in VARIANTS],
        definition=dict(transform='copy FP32 C then C[...,0] *= np.float32(alpha); model origin x=0; no recenter/renormalization',
            axis=0,origin=[0.,0.,0.],source_M_or_Y_input=False,source_M_or_Y_read=False,N_K_fixed=True,source_face_corner_order_fixed=True,
            Gaussian='original array bitwise unchanged for all three variants',sampler='existing free-only Euler50, full51-state CPU trajectory',
            probe_times=[.1,.3,.6,.9],probe_trajectory_indices=[5,15,30,45],probe_state='base free trajectory fixed; only known C replaced by m095/p105',
            post='single official D0 -> canonical C restoration -> GEO_C -> source C restoration -> fixed-C free orientation',
            stage_order='ALL48_RAW_THEN_SINGLE_D0_POST',microbatch=1,GPU_workers=1,training_updates=0,backwards=0,
            new_Gaussian_draws=0,new_C_query_draws=0,no_unique_GT_ranking=True,no_adaptive_panel_or_noise_changes=True,
            cache_policy='48 newly authorized rollouts; historical baseline trajectories not silently substituted',
            scope='width only; no height/general-conditioning certification; no model/loss/default change'),
        budget=dict(rollouts=48,generator_sampling_forwards=2400,fixed_state_probe_forwards=128,generator_forwards_max=2528,D0_forwards_max=48,
            RAW=48,POST=48,training_updates=0,backwards=0),
        preparation_source_hashes=sources,source_documents={str(p):sha(p) for p in [panelpath,oldregpath,bindingpath,d0identity,ATTACHMENT]},
        selection_path=str(OUT/'selection.json'),selection_sha256=sha(OUT/'selection.json'),model_calls_during_preparation=0)
    reg['interpretation_policy']=dict(
        exact_zero='report exact array equality and zero deltas; no invented weak-effect threshold',
        velocity_effect='RMS(v_plus-v_minus) / mean(RMS(v_plus),RMS(v_minus)); zero denominator marked NA',
        endpoint_effect='RMS per-corner displacement / L and relative to RMS C_plus-C_minus displacement',
        width_effect='X-coordinate quantile95-minus-quantile5 width response slope is descriptive, not required to equal one',
        time='early small response is not by itself failure; interpret all times with final RAW/POST visual review',
        embedding_zero='if actual known embedding is unchanged, record UNRESOLVED; no added perturbation or extra forward',
        additional_FP32_noise_floor_test=False,weak_numeric_threshold=None,
        conclusion='condition response is not adoption or legal attachment certification; width cannot certify height response')
    reg['d0_checkpoint']=reg['D0'].copy()
    write(OUT/'input_transform_receipts.json',dict(status='PASS',rows=source_receipts,models_loaded=0,model_forwards=0,new_random_draws=0))
    csvwrite(OUT/'requests.csv',requests);csvwrite(OUT/'conditions.csv',conditions)
    write(dest,reg)
    event('PREPARE_COMPLETE',requests=48,condition_variants=24,parents=8,model_calls=0,new_random_draws=0,registration_sha256=sha(dest))
    return reg

if __name__=='__main__':
    r=prepare();print(json.dumps(dict(status=r['status'],requests=len(r['requests']),conditions=len(r['conditions']),parents=len({x['parent_id'] for x in r['requests']}),registration=str(OUT/'registration.json')),ensure_ascii=False))
