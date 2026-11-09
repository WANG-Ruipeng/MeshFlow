"""Eight fixed CPU configurations, no model/optimizer/GPU or real-data scoring."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
from . import project


def run(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists():
        raise FileExistsError('Preserve prior CPU test results; use an explicit new attempt directory')
    started = time.perf_counter(); calls=[]; checks=[]; saved={}
    def check(name, value):
        checks.append(dict(name=name, passed=bool(value)))
        if not value: raise AssertionError(name)
    def call(name,C,F,remaining=None):
        cb=C.tobytes(); fb=F.tobytes()
        value, log=project.project_local_attachment(C,F,remaining)
        calls.append(dict(call=len(calls)+1,fixture=name,log=log))
        check(name+'_C_unchanged_call'+str(len(calls)),C.tobytes()==cb)
        check(name+'_input_F_unchanged_call'+str(len(calls)),F.tobytes()==fb)
        check(name+'_output_FP32_call'+str(len(calls)),value.dtype==np.float32 and value.shape==F.shape)
        check(name+'_budget_call'+str(len(calls)),log['budget_pass'])
        if log['accepted_count']:
            check(name+'_residual_zero_call'+str(len(calls)),log['maximum_constraint_residual']==0)
            check(name+'_no_added_events_call'+str(len(calls)),not log['geometry_events']['added'])
        saved[name+'_C']=C.copy(); saved[name+'_F']=F.copy(); saved[name+'_out']=value.copy()
        return value, log
    C=np.array([[[0,0,0],[1,0,0],[.5,1,0]]],np.float32)
    F=np.array([[[0,-.005,0],[1,-.005,0],[.5,-1,0]]],np.float32)
    fixtures=[]
    try:
        value,log=call('1_coplanar_gap',C,F)
        check('1_one_accepted',log['accepted_count']==1)
        check('1_exact_shared_edge',np.array_equal(value[0,:2],C[0,:2]))
        check('1_only_endpoints_moved',np.array_equal(value[0,2],F[0,2]))
        fixtures.append(dict(id=1,name='small_translated_legal_coplanar_seam',status='PASS'))

        F2=np.array([[[0,-.005,0],[1,-.005,0],[.5,-.005,1]]],np.float32)
        value,log=call('2_right_angle',C,F2)
        check('2_one_accepted',log['accepted_count']==1)
        check('2_exact_shared_edge',np.array_equal(value[0,:2],C[0,:2]))
        fixtures.append(dict(id=2,name='small_offset_90_degree_seam',status='PASS'))

        F3=F.copy(); F3[0,:2,1]=-.1
        value,log=call('3_outside_budget',C,F3)
        check('3_no_candidate_bitwise',log['status']=='NO_CANDIDATE' and value.tobytes()==F3.tobytes())
        # Same first configuration with zero remaining allowance, not a new geometry.
        value,log=call('1_coplanar_gap',C,F,np.zeros((1,3)))
        check('3_remaining_budget_rejects',log['accepted_count']==0 and value.tobytes()==F.tobytes())
        check('3_remaining_budget_reason',log['rejection_counts'].get('CORNER_BUDGET_EXCEEDED',0)>0)
        fixtures.append(dict(id=3,name='outside_displacement_budget',status='PASS'))

        C4=np.array([[[0,0,0],[1,0,0],[.5,0,.8]],[[.008,0,0],[.008,1,0],[.008,.5,.8]]],np.float32)
        p=[.004,-.004,0]
        F4=np.array([[p,[1,-.004,0],[.5,-1,0]],[p,[.004,1,0],[-1,.5,0]]],np.float32)
        value,log=call('4_group_target_conflict',C4,F4)
        check('4_conflict_rejected',log['rejection_counts'].get('GROUP_MULTIPLE_TARGETS',0)>=2)
        check('4_no_first_winner',log['accepted_count']==0 and value.tobytes()==F4.tobytes())
        fixtures.append(dict(id=4,name='same_exact_group_two_different_anchors',status='PASS'))

        F5=np.array([[[0,-.005,.005],[1,-.005,.005],[.5,-1,.005]],[[.2,-.2,.002],[.8,-.2,.002],[.5,-.7,.002]]],np.float32)
        value,log=call('5_new_intersection',C,F5)
        check('5_new_event_rejected',log['rejection_counts'].get('NEW_ILLEGAL_EVENT',0)>0)
        check('5_no_half_step',log['accepted_count']==0 and value.tobytes()==F5.tobytes())
        trial_checks=[x['acceptance_check'] for x in log['candidates'] if 'acceptance_check' in x]
        check('5_new_specific_FF_proper',any(any(e[0]=='FF' and e[-1]=='proper_transverse_penetration' for e in x['added_events']) for x in trial_checks))
        fixtures.append(dict(id=5,name='new_free_free_proper_intersection_rejected',status='PASS'))

        F6=F.copy(); F6[0,:2]=C[0,:2]
        value,log=call('6_already_aligned',C,F6)
        check('6_already_aligned',log['status']=='ALREADY_ALIGNED')
        check('6_bitwise_unchanged',value.tobytes()==F6.tobytes())
        fixtures.append(dict(id=6,name='exactly_joined_edge_unchanged',status='PASS'))

        F7=np.concatenate([F,np.array([[F[0,0],[-.5,-.4,.3],[-.2,-.8,.6]]],np.float32)])
        value7,log7=call('7_exact_group',C,F7)
        check('7_accepted',log7['accepted_count']==1)
        check('7_all_equal_corners_move',np.array_equal(value7[0,0],C[0,0]) and np.array_equal(value7[1,0],C[0,0]))
        check('7_third_corners_unmodified',np.array_equal(value7[:,2],F7[:,2]))
        fixtures.append(dict(id=7,name='move_all_existing_equal_corners_without_tearing',status='PASS'))

        C8=np.concatenate([C,np.array([[[3,0,0],[4,0,0],[3.5,1,0]]],np.float32)])
        F8=np.concatenate([F7,np.array([[[5,0,0],[6,0,0],[5.5,-1,0]]],np.float32)])
        base,logbase=call('8_permutations',C8,F8)
        # One preregistered face permutation and per-face corner reversal/cycle.
        cf=np.array([1,0]); ff=np.array([2,0,1]); corner=np.array([[2,0,1],[1,0,2],[2,1,0]])
        cp=C8[cf][:,[2,0,1]]; fp=np.stack([F8[ff[i],corner[i]] for i in range(3)])
        permuted,logperm=call('8_permutations',cp,fp)
        undo=np.empty_like(F8)
        for i in range(3): undo[ff[i],corner[i]]=permuted[i]
        check('8_geometry_output_equivariant',np.array_equal(undo,base))
        check('8_geometric_correspondences_equal',sorted(x['candidate_id'] for x in logbase['accepted_constraints'])==sorted(x['candidate_id'] for x in logperm['accepted_constraints']))
        fixtures.append(dict(id=8,name='face_and_corner_permutation',status='PASS'))

        check('public_signature',list(inspect.signature(project.project_local_attachment).parameters)==['C','F','remaining_budget','fixed_config'])
        tree=ast.parse(inspect.getsource(project))
        banned={'open','load','loadtxt','read_text','read_bytes','read_csv','read_json','torch','forward','backward'}
        called={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        check('no_external_asset_or_model_read',not(called & banned))
        # Counterfactual score fixture exists only outside the public call.
        external=out/'unused_external_GT_score.json'
        external.write_text(json.dumps(dict(GT='counterfactual_A',Gamma=0,score=0)))
        with patch('builtins.open',side_effect=AssertionError('Project attempted external IO')),patch.object(np,'load',side_effect=AssertionError('Project attempted array IO')):
            va,la=call('1_coplanar_gap',C,F)
        external.write_text(json.dumps(dict(GT='counterfactual_B',Gamma=999,score=999)))
        with patch('builtins.open',side_effect=AssertionError('Project attempted external IO')),patch.object(np,'load',side_effect=AssertionError('Project attempted array IO')):
            vb,lb=call('1_coplanar_gap',C,F)
        check('external_GT_score_replacement_invariant',va.tobytes()==vb.tobytes() and [x['candidate_id'] for x in la['accepted_constraints']]==[x['candidate_id'] for x in lb['accepted_constraints']])
        check('maximum_eight_configurations',len(fixtures)==8)
        status='PASS'; error=None
    except Exception as exc:
        status='FAIL'; error=repr(exc)
    np.savez_compressed(out/'fixtures.npz',**saved)
    source_files={str(Path(m.__file__).resolve()):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in [project]}
    source_files[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ledger=Counter()
    for call_row in calls: ledger.update(call_row['log']['classifier_ledger'])
    report=dict(status=status,error=error,configuration_limit=8,configurations_completed=len(fixtures),fixtures=fixtures,project_calls=len(calls),calls=calls,checks=checks,passed_checks=sum(x['passed'] for x in checks),classifier_ledger=dict(ledger),source_hashes=source_files,fixed_config=dict(project.FIXED_CONFIG),seconds=time.perf_counter()-started,model_forwards=0,model_backward=0,optimizer_updates=0,geometry_configuration_scope='Eight explicitly fixed analytic configurations. Repeated calls are budget/IO/equivariance checks, not extra geometry families.')
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','error','configurations_completed','project_calls','passed_checks','seconds','classifier_ledger']},indent=2))
    if status!='PASS': raise SystemExit(1)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True)
    run(parser.parse_args().out)
