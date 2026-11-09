"""Fixed, bounded CPU R/S correction of FP32 free triangle soups.

Inputs are current free triangles, exact context C and declared scale metadata L.
No target geometry, model, optimizer, topology change, or geometry autograd.
Every proposal is built in FP64 and all acceptance tests use its FP32 writeback.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import math
import time
import numpy as np
from .native_patch_seams import (extract_boundary_segments, nearest_segments,
    point_to_triangle_surface_squared, _sample_boundary, triangle_soup_topology)

PARAMETERS = dict(eta_bbox=1e-4,delta_bbox=5e-3,rho_bbox=1e-3,beta=.25,
    module_FP32_candidate_budget=64,R_candidates_per_free_face=4,
    backtracking=[1.,.5,.25,.125],axis_relative_zero_tolerance=1e-12,
    axis_duplicate_cosine_tolerance=1e-12,area_epsilon_bbox_squared=1e-12,
    intersection_epsilon_bbox=1e-9,independent_near_degenerate_area_bbox_squared=1e-11,boundary_seed=9919,boundary_target_samples=512,
    coverage_MSE_roundoff_tolerance_bbox_squared=1e-15,
    proxy_progress_tolerance_bbox=1e-12,
    radius_roundoff_tolerance_bbox=0.,
    displacement_radius_definition='Maximum individual vertex distance from the same original Xhat; R and S share it',
    CONTINUOUS_COLLISION_SAFETY='NOT_RUN')
ILLEGAL = frozenset(('proper_transverse_penetration','coplanar_positive_area_overlap'))


def _hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((str(a.dtype),a.shape)).encode()+a.tobytes()).hexdigest()


def _triangles(a):
    a=np.asarray(a)
    if a.ndim==2 and a.shape[1]==9:return a.reshape(-1,3,3)
    if a.ndim==3 and a.shape[1:]==(3,3):return a
    raise ValueError('Expected [N,9] or [N,3,3] triangles')


def _normals_and_areas(a):
    a=np.asarray(a,dtype=np.float64)
    n=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
    return n,np.linalg.norm(n,axis=1)*.5


def prepare_boundary(C,L):
    """Static cache depends only on exact C, L and fixed boundary seed9919."""
    C=_triangles(C)
    if C.dtype!=np.float32 or not np.isfinite(C).all() or not math.isfinite(L) or L<=0:
        raise ValueError('Finite FP32 C and positive L required')
    segments,topology=extract_boundary_segments(C)
    samples,identities,weights,counts=_sample_boundary(segments,9919,512)
    for array in (segments,samples,identities,weights):array.setflags(write=False)
    return dict(C_sha256=_hash(C),L=float(L),segments=segments,points=samples,weights=weights,
        segment_ids=identities,audit=dict(**topology,seed=9919,requested_samples=512,
        actual_samples=len(samples),sample_counts_per_segment=counts,
        sample_points_sha256=_hash(samples),sample_weights_sha256=_hash(weights),
        source='Exact C incidence-one edges only; no GT free adjacency',
        topology_closed='NOT_CERTIFIED'))


def _verified(A,B,L,counts):
    from .guided_native_geometry_geometry import verified_pair
    counts['verified_pair_calls']+=1
    counts['full_pair_tests']+=1
    counts['primary_pair_classifications']+=1
    counts['independent_pair_classifications']+=1
    result=verified_pair(A,B,L)
    if not result.get('verified',False) or not result.get('resolved',False):return 'UNRESOLVED'
    classification=result['classification']
    if result.get('illegal') is None:return 'UNRESOLVED'
    if bool(result['illegal'])!=(classification in ILLEGAL):return 'UNRESOLVED'
    return classification


def _pair_map(C,F,L,counts,affected=None):
    """AABB only rejects clearly separated pairs; all possible contacts verified.

    Affected maps replace exactly those incident pairs in the full current map.
    Reusing unchanged pairs within this one call is algebraically the same as
    testing the complete current C/free and free/free sets after every proposal.
    No dynamic map is retained across corrector calls/Euler steps.
    """
    selected=set(range(len(F))) if affected is None else set(map(int,affected))
    eps=1e-9*L;cmin=C.min(1);cmax=C.max(1);fmin=F.min(1);fmax=F.max(1)
    _,ca=_normals_and_areas(C);_,fa=_normals_and_areas(F)
    cd=ca<=1e-11*L*L;fd=fa<=1e-11*L*L
    cf={};ff={}
    for fi in sorted(selected):
        counts['AABB_pair_checks']+=len(C)
        possible=np.flatnonzero((np.all(cmax>=fmin[fi]-eps,axis=1)&np.all(fmax[fi]>=cmin-eps,axis=1))|cd|fd[fi])
        counts['broadphase_pruned']+=len(C)-len(possible)
        for ci in possible:
            kind=_verified(C[ci],F[fi],L,counts)
            if kind!='disjoint':cf[(int(ci),fi)]=kind
    for fi in range(len(F)):
        others=np.arange(fi+1,len(F),dtype=int)
        if fi not in selected:others=others[np.array([q in selected for q in others],dtype=bool)]
        counts['AABB_pair_checks']+=len(others)
        if not len(others):continue
        possible=others[(np.all(fmax[others]>=fmin[fi]-eps,axis=1)&np.all(fmax[fi]>=fmin[others]-eps,axis=1))|fd[others]|fd[fi]]
        counts['broadphase_pruned']+=len(others)-len(possible)
        for fj in possible:
            kind=_verified(F[fi],F[fj],L,counts)
            if kind!='disjoint':ff[(fi,int(fj))]=kind
    return dict(cf=cf,ff=ff)


def _sets(maps,section):
    return {pair for pair,kind in maps[section].items() if kind in ILLEGAL}


def _typed_illegal_sets(maps,section):
    return {(pair,kind) for pair,kind in maps[section].items() if kind in ILLEGAL}


def _summary(maps):
    result={}
    for name in ('cf','ff'):
        counts=Counter(maps[name].values())
        result[name]=dict(classification_counts=dict(counts),
            proper_pairs=[list(k) for k,v in sorted(maps[name].items()) if v=='proper_transverse_penetration'],
            coplanar_overlap_pairs=[list(k) for k,v in sorted(maps[name].items()) if v=='coplanar_positive_area_overlap'],
            unresolved_pairs=[list(k) for k,v in sorted(maps[name].items()) if v=='UNRESOLVED'],
            illegal_pair_count=len(_sets(maps,name)))
    return result


def _merge_maps(before,updates,affected):
    affected=set(affected)
    cf={key:kind for key,kind in before['cf'].items() if key[1] not in affected}
    ff={key:kind for key,kind in before['ff'].items() if not affected.intersection(key)}
    cf.update(updates['cf']);ff.update(updates['ff'])
    return dict(cf=cf,ff=ff)


def _gate(current,proposal,C,Xhat,L,rho,maps,affected,kind,counts,
          coverage_before=None,coverage_after=None):
    """Hard tests of the actual candidate FP32 array, never just FP64 scratch."""
    reasons=[]
    if proposal.dtype!=np.float32 or proposal.shape!=current.shape or not np.isfinite(proposal).all():
        return False,['NONFINITE_OR_SHAPE_OR_DTYPE'],None
    delta=proposal.astype(np.float64)-Xhat.astype(np.float64)
    maximum=float(np.linalg.norm(delta,axis=-1).max(initial=0))
    if maximum>rho:reasons.append('TOTAL_RHO_EXCEEDED_AFTER_FP32')
    n0,a0=_normals_and_areas(current[list(affected)]);n1,a1=_normals_and_areas(proposal[list(affected)])
    if np.any(a1<=1e-12*L*L):reasons.append('DEGENERATE_OR_UNRESOLVED_AFFECTED_FACE')
    if np.any(a0<=1e-12*L*L):reasons.append('PREEXISTING_DEGENERATE_AFFECTED_FACE')
    if kind=='S':
        if np.any(a1<a0*.5):reasons.append('S_AREA_BELOW_HALF')
        if np.any(np.einsum('ij,ij->i',n0,n1)<=0):reasons.append('S_NORMAL_FLIP')
        if coverage_before is None or coverage_after is None:reasons.append('S_NO_COMPLETE_BOUNDARY_COVERAGE')
        elif coverage_after>coverage_before+1e-15*L*L:reasons.append('S_COMPLETE_GAMMA_COVERAGE_WORSE')
    if reasons:return False,reasons,None
    updates=_pair_map(C,proposal,L,counts,affected)
    if any(v=='UNRESOLVED' for group in updates.values() for v in group.values()):
        reasons.append('UNRESOLVED_AFFECTED_PAIR')
    merged=_merge_maps(maps,updates,affected)
    for section in ('cf','ff'):
        for label in sorted(ILLEGAL):
            old_pairs={p for p,v in maps[section].items() if v==label}
            new_pairs={p for p,v in merged[section].items() if v==label}
            if new_pairs-old_pairs:reasons.append('NEW_'+section.upper()+'_'+label.upper()+'_PAIR')
    return not reasons,reasons,merged


def _ray_fraction(current,Xhat,direction,rho):
    """One shared scalar keeps each proposed vertex in its Xhat rho ball."""
    offset=np.asarray(current,dtype=np.float64)-np.asarray(Xhat,dtype=np.float64)
    step=np.asarray(direction,dtype=np.float64)
    aa=np.einsum('ij,ij->i',step,step);bb=2*np.einsum('ij,ij->i',offset,step)
    cc=np.einsum('ij,ij->i',offset,offset)-rho*rho
    active=aa>0
    if not np.any(active):return 0.
    discriminant=np.maximum(bb[active]**2-4*aa[active]*cc[active],0)
    root=(-bb[active]+np.sqrt(discriminant))/(2*aa[active])
    return float(max(0.,min(1.,float(root.min()))))


def _separation_candidates(F,C,contacts,L,eta,counts):
    entries=[]
    fe=np.roll(F,-1,axis=0)-F
    fn=np.cross(fe[0],fe[1]);fnorm=np.linalg.norm(fn)
    for ci in sorted(contacts):
        tri=C[ci].astype(np.float64);ce=np.roll(tri,-1,axis=0)-tri
        cn=np.cross(ce[0],ce[1]);cnorm=np.linalg.norm(cn)
        axes=[]
        def add(vector,reference):
            norm=float(np.linalg.norm(vector))
            if norm<=1e-12*reference:
                counts['near_zero_candidate_axes_skipped']+=1;return
            axis=np.asarray(vector,dtype=np.float64)/norm
            dominant=int(np.argmax(np.abs(axis)))
            if axis[dominant]<0:axis=-axis
            if any(abs(float(axis@old))>=1-1e-12 for old in axes):
                counts['duplicate_candidate_axes_skipped']+=1;return
            axes.append(axis)
        add(fn,L*L);add(cn,L*L)
        for a in fe:
            for b in ce:add(np.cross(a,b),L*L)
        if fnorm>0 and cnorm>0:
            coplanar=(np.max(np.abs((tri-F[0])@(fn/fnorm)))<=1e-9*L and
                      np.max(np.abs((F-tri[0])@(cn/cnorm)))<=1e-9*L)
            if coplanar:
                for edge in fe:add(np.cross(fn/fnorm,edge),L)
                for edge in ce:add(np.cross(cn/cnorm,edge),L)
        for axis_id,axis in enumerate(axes):
            fp=F@axis;cp=tri@axis
            for sign,q in ((1,max(0.,float(cp.max()+eta-fp.min()))),(-1,min(0.,float(cp.min()-eta-fp.max())))):
                if q==0:continue
                entries.append(dict(ci=int(ci),axis=axis,axis_id=axis_id,sign=sign,q=q,
                    magnitude=abs(q),signature=(int(ci),tuple(np.round(axis,15)),sign)))
    entries.sort(key=lambda r:(r['magnitude'],r['ci'],r['axis_id'],0 if r['sign']==1 else 1))
    counts['candidate_axes_signed_generated']+=len(entries)
    return entries


def _proxy(F,C,axis,sign,eta):
    a=np.asarray(F,dtype=np.float64)@axis;b=np.asarray(C,dtype=np.float64)@axis
    return max(0.,float(b.max()+eta-a.min())) if sign==1 else max(0.,float(a.max()+eta-b.min()))


def _coverage_matrix(points,free,L):
    if not len(points) or not len(free):return None
    return np.column_stack([point_to_triangle_surface_squared(points,triangle[None],L) for triangle in free])


def _coverage_mse(matrix,weights):
    return None if matrix is None else float(weights@matrix.min(1))


def _correct(free,C,L,mode,boundary=None,*,fixture_rho=None,fixture_label=None):
    start=time.perf_counter();shape=np.asarray(free).shape
    F=_triangles(free);C=_triangles(C)
    if mode not in ('R','S','RS','off'):raise ValueError('Only preregistered R/S/RS/off modes')
    if F.dtype!=np.float32 or C.dtype!=np.float32 or not np.isfinite(F).all() or not np.isfinite(C).all():
        raise ValueError('Current free and exact C must be finite FP32 arrays')
    if not math.isfinite(L) or L<=0:raise ValueError('Positive scale metadata L required')
    if fixture_rho is not None and not fixture_label:raise ValueError('Rho override is for named isolated CPU fixtures only')
    Xhat=F.copy();current=F.copy();C_before=C.copy();fhash=_hash(F);chash=_hash(C)
    L=float(L);rho=1e-3*L if fixture_rho is None else float(fixture_rho);eta=1e-4*L
    counts=Counter();modules=[];accepted_path_lengths=np.zeros((len(F),3),dtype=np.float64)
    if boundary is None:boundary=prepare_boundary(C,L)
    if boundary['C_sha256']!=chash or boundary['L']!=L:raise ValueError('Static boundary cache belongs to different C/L')
    maps=_pair_map(C,current,L,counts);initial_maps=maps
    coverage=_coverage_matrix(boundary['points'],current,L)
    counts['boundary_point_triangle_distance_evaluations']+=len(boundary['points'])*len(current)
    coverage_initial=_coverage_mse(coverage,boundary['weights'])
    def run_module(kind):
        nonlocal current,maps,coverage
        module_start=time.perf_counter();before=_summary(maps);before_cov=_coverage_mse(coverage,boundary['weights'])
        calls=accepted=truncated=0;rejections=Counter();records=[];face_candidates=Counter();eligible_count=0
        if kind=='R':
            face_order=sorted({pair[1] for pair in _sets(maps,'cf')});eligible_count=len(face_order)
        else:
            distances,_,_=nearest_segments(current.reshape(-1,3),boundary['segments'])
            face_order=[] if distances is None else np.flatnonzero((distances.reshape(-1,3)<(5e-3*L)**2).any(1)).tolist()
            eligible_count=0 if distances is None else int((distances<(5e-3*L)**2).sum())
        for fi in face_order:
            tried=set()
            while calls<64:
                if kind=='R':
                    contacts=sorted(ci for ci,f in _sets(maps,'cf') if f==fi)
                    if not contacts or face_candidates[fi]>=4:break
                    candidates=_separation_candidates(current[fi].astype(np.float64),C,contacts,L,eta,counts)
                    choices=[entry for entry in candidates if entry['signature'] not in tried]
                    if not choices:break
                    selected=choices[0];tried.add(selected['signature']);face_candidates[fi]+=1
                    direction=np.broadcast_to(selected['q']*selected['axis'],(3,3)).copy()
                    proxy_before=_proxy(current[fi],C[selected['ci']],selected['axis'],selected['sign'],eta)
                    details=dict(context_face=selected['ci'],axis=selected['axis'].tolist(),sign=selected['sign'],
                        required_translation=selected['magnitude'],proxy_before=proxy_before)
                else:
                    if face_candidates[fi]:break
                    d2,nearest,_=nearest_segments(current[fi],boundary['segments'])
                    take=d2<(5e-3*L)**2
                    direction=np.where(take[:,None],.25*(nearest-current[fi].astype(np.float64)),0.)
                    face_candidates[fi]+=1
                    details=dict(near_corners=np.flatnonzero(take).tolist(),local_distance_RMS_before=float(np.sqrt(d2[take].mean())) if take.any() else None)
                fraction=_ray_fraction(current[fi],Xhat[fi],direction,rho)
                was_truncated=fraction<1
                if was_truncated:truncated+=1
                if fraction<=0 or not np.any(direction):
                    records.append(dict(free_face=int(fi),status='NO_NONZERO_FEASIBLE_DISPLACEMENT',radius_fraction=fraction,**details))
                    break
                candidate_accepted=False
                for scale in PARAMETERS['backtracking']:
                    if calls>=64:break
                    scratch=current[fi].astype(np.float64)+fraction*scale*direction
                    proposal=current.copy();proposal[fi]=scratch.astype(np.float32)
                    calls+=1;counts['FP32_candidate_checks']+=1
                    counts['C_bitwise_candidate_checks']+=1
                    if not np.array_equal(C,C_before):raise AssertionError('C changed during proposal')
                    if np.array_equal(proposal[fi],current[fi]):
                        rejections['NO_FP32_CHANGE']+=1;counts['rejected_candidates']+=1
                        records.append(dict(free_face=int(fi),backtrack=scale,status='REJECTED',reasons=['NO_FP32_CHANGE'],**details));continue
                    proposed_coverage=coverage.copy() if coverage is not None else None
                    if proposed_coverage is not None:
                        proposed_coverage[:,fi]=point_to_triangle_surface_squared(boundary['points'],proposal[fi:fi+1],L)
                        counts['boundary_point_triangle_distance_evaluations']+=len(boundary['points'])
                    old_cov=_coverage_mse(coverage,boundary['weights']);new_cov=_coverage_mse(proposed_coverage,boundary['weights'])
                    ok,reasons,newmaps=_gate(current,proposal,C,Xhat,L,rho,maps,[fi],kind,counts,old_cov,new_cov)
                    proxy_after=None
                    if kind=='R':
                        proxy_after=_proxy(proposal[fi],C[selected['ci']],selected['axis'],selected['sign'],eta)
                        if not proxy_after<proxy_before-1e-12*L:
                            ok=False;reasons.append('NO_STRICT_SEPARATION_PROXY_PROGRESS')
                    else:
                        after_d2,_,_=nearest_segments(proposal[fi],boundary['segments'])
                        if not float(after_d2[take].sum())<float(d2[take].sum()):
                            ok=False;reasons.append('NO_LOCAL_NEAR_DISTANCE_PROGRESS')
                    record=dict(free_face=int(fi),backtrack=scale,radius_fraction=fraction,truncated=was_truncated,
                        status='ACCEPTED' if ok else 'REJECTED',reasons=reasons,
                        complete_Gamma_MSE_before=old_cov,complete_Gamma_MSE_after=new_cov,
                        max_total_displacement=float(np.linalg.norm(proposal.astype(np.float64)-Xhat,axis=-1).max()),**details)
                    if kind=='R':
                        fp=proposal[fi].astype(np.float64)@selected['axis'];cp=C[selected['ci']].astype(np.float64)@selected['axis']
                        clearance=float(fp.min()-cp.max()) if selected['sign']==1 else float(cp.min()-fp.max())
                        record.update(proxy_after=proxy_after,actual_signed_projection_clearance=clearance,
                            projection_intervals_strictly_separated=clearance>0.,
                            projection_requested_eta_achieved=clearance>=eta,
                            actual_illegal_contact_resolved=bool(newmaps is not None and (selected['ci'],fi) not in _sets(newmaps,'cf')))
                    records.append(record)
                    if ok:
                        accepted_path_lengths[:]+=np.linalg.norm(proposal.astype(np.float64)-current.astype(np.float64),axis=-1)
                        counts['accepted_candidates']+=1;accepted+=1;current=proposal;maps=newmaps;coverage=proposed_coverage;candidate_accepted=True;break
                    counts['rejected_candidates']+=1;rejections.update(reasons)
                if kind=='S':break
                # Accepted R proposals regenerate contacts/axes from the current
                # actually written state before trying another pair/direction.
                if calls>=64:break
            if calls>=64:break
        after=_summary(maps)
        unprocessed=sorted(set(face_order)-set(face_candidates))
        unresolved=after['cf']['illegal_pair_count']
        modules.append(dict(module=kind,FP32_candidate_checks=calls,accepted=accepted,rejected=calls-accepted,
            rejected_reasons=dict(rejections),radius_truncations=truncated,budget_exhausted=calls>=64,
            eligible_faces=len(face_order),eligible_near_vertex_slots=eligible_count if kind=='S' else None,
            candidate_directions_per_free_face={str(k):int(v) for k,v in sorted(face_candidates.items())},
            unprocessed_free_faces=unprocessed,unprocessed_illegal_pairs=[list(p) for p in sorted(_sets(maps,'cf')) if p[1] in unprocessed],
            unresolved_collision_pairs=unresolved,status='UNRESOLVED_COLLISION' if unresolved else 'NO_REMAINING_C_FREE_ILLEGAL_PAIR',
            before=before,after=after,boundary_MSE_before=before_cov,boundary_MSE_after=_coverage_mse(coverage,boundary['weights']),
            records=records,seconds=time.perf_counter()-module_start))
    if mode in ('R','RS'):run_module('R')
    if mode in ('S','RS'):run_module('S')
    if _hash(F)!=fhash or _hash(C)!=chash or not np.array_equal(C,C_before):raise AssertionError('Input or C mutated')
    final_delta=current.astype(np.float64)-Xhat.astype(np.float64)
    displacement=np.linalg.norm(final_delta,axis=-1)
    if float(displacement.max(initial=0))>rho:raise AssertionError('Final shared radius violated')
    _,initial_area=_normals_and_areas(Xhat);_,final_area=_normals_and_areas(current)
    if np.any((final_area<=1e-12*L*L)&(initial_area>1e-12*L*L)):
        raise AssertionError('Accepted sequence introduced a new degenerate face')
    if _typed_illegal_sets(maps,'cf')-_typed_illegal_sets(initial_maps,'cf') or _typed_illegal_sets(maps,'ff')-_typed_illegal_sets(initial_maps,'ff'):
        raise AssertionError('Accepted sequence introduced a new illegal pair')
    audit=dict(status='PASS_EXECUTION',mode=mode,parameters=dict(PARAMETERS),bbox_diagonal=L,
        fixture_only_override=None if fixture_rho is None else dict(label=fixture_label,rho=fixture_rho,excluded_from_experiment=True),
        before=_summary(initial_maps),after=_summary(maps),modules=modules,counts=dict(counts),
        boundary=boundary['audit'],boundary_MSE_before=coverage_initial,boundary_MSE_after=_coverage_mse(coverage,boundary['weights']),
        displacement_RMS=float(np.sqrt(np.mean(displacement**2))),displacement_max=float(displacement.max(initial=0)),
        sum_vertex_displacement_lengths=float(displacement.sum()),
        sum_cumulative_vertex_motion_lengths=float(accepted_path_lengths.sum()),
        cumulative_vertex_motion_RMS=float(np.sqrt(np.mean(accepted_path_lengths**2))),
        cumulative_vertex_motion_max=float(accepted_path_lengths.max(initial=0)),
        cumulative_motion_definition='Sum of actual FP32 accepted proposal-to-current vertex lengths within this invocation; separate from final net displacement',
        C_bitwise_unchanged=True,free_face_count_before=len(F),free_face_count_after=len(current),
        input_free_sha256=fhash,output_free_sha256=_hash(current),C_sha256=chash,inputs_unchanged=True,
        FP64_working_copy=True,acceptance_on_actual_FP32=True,
        no_new_C_free_illegal_pairs=True,no_new_free_free_illegal_pairs=True,no_new_degenerate_faces=True,
        all_finite=True,degenerate_faces_before=int((initial_area<=1e-12*L*L).sum()),degenerate_faces_after=int((final_area<=1e-12*L*L).sum()),
        illegal_pair_policy='Compare proper and coplanar pair identity sets separately, primary and independently verified; unresolved affected proposals rejected',
        static_check_only=True,CONTINUOUS_COLLISION_SAFETY='NOT_RUN',topology_welding='NOT_RUN',
        GT_inputs=False,model_forwards=0,model_backwards=0,optimizer_updates=0,geometry_autograd_calls=0,
        seconds=time.perf_counter()-start)
    return current.reshape(shape),audit


def correct_geometry(free,C,L,mode='R',boundary=None):
    """Production entry: fixed parameters; no rho/eta/beta tuning arguments."""
    return _correct(free,C,L,mode,boundary)


def run_cpu_fixtures():
    """Analytic, input-only corrector gates; never call a model or optimizer."""
    cases=[];saved={};L=math.sqrt(2)
    C=np.array([[[0,0,0],[1,0,0],[0,1,0]]],dtype=np.float32)
    for name,z in (('wide_internal_puncture',(1.,-1.,1.)),('thin_internal_puncture',(.02,-.02,.02))):
        F=np.array([[[.2,.2,z[0]],[.6,.2,z[1]],[.2,.6,z[2]]]],dtype=np.float32)
        out,audit=_correct(F,C,L,'R',fixture_rho=2*L,fixture_label=name+'_isolated_large_radius')
        assert audit['before']['cf']['proper_pairs']==[[0,0]] and not audit['after']['cf']['illegal_pair_count'],audit
        _,before_area=_normals_and_areas(F);_,after_area=_normals_and_areas(out)
        np.testing.assert_allclose(before_area,after_area,rtol=2e-6,atol=1e-10)
        cases.append(dict(name=name,status='PASS',audit=audit,area_before=before_area.tolist(),area_after=after_area.tolist()))
        saved[name+'_before']=F;saved[name+'_after']=out
    F=np.array([[[.2,.2,1],[.6,.2,-1],[.2,.6,1]]],dtype=np.float32)
    out,audit=correct_geometry(F,C,L,'R')
    assert audit['after']['cf']['illegal_pair_count']==1 and audit['modules'][0]['status']=='UNRESOLVED_COLLISION'
    assert audit['displacement_max']<=1e-3*L
    cases.append(dict(name='production_radius_reports_unresolved_puncture',status='PASS',audit=audit))
    saved['rho_limited_before']=F;saved['rho_limited_after']=out
    near=(C+np.array([0,0,.001*L],dtype=np.float32))
    out,audit=correct_geometry(near,C,L,'S')
    assert audit['boundary_MSE_after']<audit['boundary_MSE_before'] and audit['modules'][0]['accepted']>=1,audit
    cases.append(dict(name='S_feasible_near_boundary_improves_full_Gamma',status='PASS',audit=audit))
    saved['S_feasible_before']=near;saved['S_feasible_after']=out
    # Test the S acceptance gate on the mandated complete-collapse proposal.
    # The fixed beta=.25 update is not replaced by full projection in production.
    prior=np.array([[[.2,.001,0],[.5,.002,0],[.8,.001,0]]],np.float32)
    collapsed=prior.copy();collapsed[0,:,1]=0
    counts=Counter();maps=_pair_map(C,prior,L,counts)
    counts['fixture_direct_FP32_gate_checks']+=1
    ok,reasons,_=_gate(prior,collapsed,C,prior,L,.01*L,maps,[0],'S',counts,1.,0.)
    assert not ok and 'DEGENERATE_OR_UNRESOLVED_AFFECTED_FACE' in reasons
    cases.append(dict(name='S_acceptance_rejects_three_vertices_collapsed_to_one_line',status='PASS',
        fixture_only_direct_proposal=True,production_beta_unchanged=.25,rejection_reasons=reasons,counts=dict(counts)))
    saved['S_degenerate_before']=prior;saved['S_degenerate_rejected_proposal']=collapsed
    transition_records=[]
    piercing=np.array([[[.2,.2,.001],[.6,.2,-.001],[.2,.6,.001]]],np.float32)
    overlapping=piercing.copy();overlapping[:,:,2]=0.
    for section in ('cf','ff'):
        for old,new,destination in ((piercing,overlapping,'COPLANAR_POSITIVE_AREA_OVERLAP'),(overlapping,piercing,'PROPER_TRANSVERSE_PENETRATION')):
            known=C if section=='cf' else C+np.array([0,0,10],np.float32)
            before=old if section=='cf' else np.concatenate([C,old])
            proposed=new if section=='cf' else np.concatenate([C,new])
            affected=[0] if section=='cf' else [1]
            counter=Counter();initial=_pair_map(known,before,L,counter)
            counter['fixture_direct_FP32_gate_checks']+=1
            ok,reasons,_=_gate(before,proposed,known,before,L,.01*L,initial,affected,'R',counter)
            required='NEW_'+section.upper()+'_'+destination+'_PAIR'
            assert not ok and required in reasons,(required,reasons)
            transition_records.append(dict(section=section,expected_rejection=required,reasons=reasons,counts=dict(counter)))
    cases.append(dict(name='proper_overlap_type_swap_rejected_for_same_CF_and_FF_pair',status='PASS',
        fixture_only_direct_proposals=True,transitions=transition_records))
    free=np.array([[[0,0,0],[.5,0,0],[.25,-.5,0]],[[.5,0,0],[1,0,0],[.75,-.5,0]]],np.float32)
    out,audit=correct_geometry(free,C,L,'S');assert np.array_equal(out,free)
    sample=np.array([[.1,0,0],[.4,0,0],[.6,0,0],[.9,0,0]])
    assert np.all(point_to_triangle_surface_squared(sample,free,L)==0)
    topology=triangle_soup_topology(np.concatenate([C,free]),L)
    assert topology['exact_edge_connected_face_components']==3
    cases.append(dict(name='T_junction_zero_local_distance_is_not_topology_connection',status='PASS',
        selected_AB_samples_distance_squared=[0.,0.,0.,0.],topology=topology,audit=audit))
    saved['T_junction_C']=C;saved['T_junction_free']=free
    invariance_audits=[]
    for mode in ('off','R','S','RS'):
        out,audit=correct_geometry(near,C,L,mode)
        invariance_audits.append(audit)
        assert audit['displacement_max']<=1e-3*L
        if mode=='off':assert np.array_equal(out,near)
    cases.append(dict(name='RS_shared_Xhat_radius_and_off_exact',status='PASS',audits=invariance_audits))
    saved['C']=C
    total_counts=Counter()
    for row in cases:
        if 'audit' in row:total_counts.update(row['audit']['counts'])
        if 'audits' in row:
            for item in row['audits']:total_counts.update(item['counts'])
        if 'counts' in row:total_counts.update(row['counts'])
        if 'transitions' in row:
            for item in row['transitions']:total_counts.update(item['counts'])
    return dict(status='PASS',tests=cases,fixture_count=len(cases),parameters=dict(PARAMETERS),
        counts=dict(total_counts),corrector_invocations=9,direct_acceptance_fixture_checks=5,
        model_forwards=0,model_backwards=0,optimizer_updates=0,geometry_autograd_calls=0),saved
