"""CPU-only tests for timing scopes, fixed slots, replay binding and failure retention."""
import copy
from pathlib import Path
import numpy as np
import pytest
from ..io_utils import array_hash,file_hash
from .ours_timing import (STAGES,cases,case_id,warm_schedule,pipeline,replay_identity,
    save_record,summarize,failed_record,verify_bound_inputs,load_arrays)


class Clock:
    def __init__(self):self.value=0.
    def __call__(self):self.value+=.01;return self.value


class FakeRuntime:
    def __init__(self,invalid=False,fail=None):self.events=[];self.invalid=invalid;self.fail=fail
    def called(self,name):
        self.events.append(name)
        if name==self.fail:raise RuntimeError('deliberate '+name)
    def transfer(self,noise):self.called('H2D');return noise[None].copy()
    def fm32(self,eps):self.called('FM32');return eps+2
    def inverse(self,zs):self.called('inverse_once');return zs*2+3
    def decode(self,z):self.called('AE');return np.full((1,2050),z[0,0],np.float32)
    def host(self,eps,zs,z,nodes):
        self.called('D2H');return {k:a[0].copy() for k,a in [('noise',eps),('z_standardized',zs),('z',z),('nodes',nodes)]}
    def export(self,nodes,arm):
        self.called('export_'+arm)
        # The produced mesh depends on the freshly computed field, not saved nodes.
        return {'vertices':np.eye(3,dtype=np.float64)*nodes[0],'faces':np.array([[0,1,2]],np.int32)}, {'arm':arm}
    def quality(self,mesh):self.called('quality');return {'min_angle':60}
    def topology(self,mesh):self.called('topology');return {'closed_oriented_g0':not self.invalid}
    def exact(self,mesh):self.called('exact');return {'embedded_endpoint_pass':not self.invalid}


def expected():
    return {'noise':np.ones(64,np.float32),'z_standardized':np.full(64,3,np.float32),
        'z':np.full(64,9,np.float32),'nodes':np.full(2050,9,np.float32)}


def run_fake(invalid=False,fail=None):
    rt=FakeRuntime(invalid,fail);syncs=[]
    row,mesh,arrays=pipeline(rt,expected(),'a',sync=lambda:syncs.append('sync'),clock=Clock())
    return rt,row,mesh,arrays,syncs


def test_registered_slots_and_warm_order_no_selection():
    base=cases();assert len(base)==24 and len({case_id(x) for x in base})==24
    assert {x['sample_id'] for x in base}=={'00000','00016','00032','00048'}
    schedule=warm_schedule();assert len(schedule)==96
    assert [x['phase'] for x in schedule[:24]]==['warmup']*24
    assert [{k:x[k] for k in ('seed','sample_id','arm')} for x in schedule[48:72]]==list(reversed(base))
    for case in base:
        chosen=[x for x in schedule if case_id(x)==case_id(case)]
        assert [(x['phase'],x['repeat']) for x in chosen]==[('warmup',0),('measured',1),('measured',2),('measured',3)]


def test_pipeline_measures_transfers_once_inverse_and_real_export():
    rt,row,mesh,arrays,syncs=run_fake()
    assert rt.events==['H2D','FM32','inverse_once','AE','D2H','export_a','quality','topology','exact']
    assert tuple(row['stage_seconds'])==STAGES and len(syncs)==20
    assert row['runtime_status']=='ok' and row['mesh_total_seconds']<row['checked_total_seconds']
    assert row['actual_faces']==1 and row['budget512_match'] is False
    assert np.array_equal(mesh['vertices'],np.eye(3)*9)
    assert replay_identity(expected(),arrays,mesh,mesh)['exact_production_replay']


def test_geometry_failure_is_not_runtime_failure_and_is_retained():
    _,row,mesh,arrays,_=run_fake(invalid=True)
    assert row['runtime_status']=='ok' and row['geometry_valid'] is False
    assert row['geometry_and_budget_valid'] is False
    assert row['checked_total_seconds']>0 and mesh is not None
    assert replay_identity(expected(),arrays,mesh,mesh)['exact_production_replay']


def test_runtime_failure_retains_partial_stage_without_repair():
    rt,row,mesh,arrays,_=run_fake(fail='FM32')
    assert row['runtime_status']=='failed' and list(row['stage_seconds'])==['noise_H2D','FM32']
    assert rt.events==['H2D','FM32'] and mesh is None and arrays is None
    assert 'checked_total_seconds' not in row and 'seconds_until_runtime_failure' in row


def test_exact_replay_detects_latent_nodes_faces_and_radial_changes():
    _,row,mesh,arrays,_=run_fake()
    for name in arrays:
        changed={k:v.copy() for k,v in arrays.items()};changed[name][0]+=1
        check=replay_identity(expected(),changed,mesh,mesh)
        assert not check['exact_production_replay'] and check['arrays'][name]['max_abs_error']==1
    changed={k:v.copy() for k,v in mesh.items()};changed['faces'][0]=[0,2,1]
    assert not replay_identity(expected(),arrays,mesh,changed)['exact_production_replay']
    other_dtype={**mesh,'faces':mesh['faces'].astype(np.int64)}
    assert not replay_identity(expected(),arrays,mesh,other_dtype)['exact_production_replay']
    assert replay_identity(expected(),arrays,mesh,other_dtype)['canonical_geometry_hash']==replay_identity(expected(),arrays,mesh,mesh)['canonical_geometry_hash']
    radial={**mesh,'u':np.eye(3),'r':np.ones(3),'alpha':np.ones(32)}
    changed={**radial,'alpha':np.zeros(32)}
    assert not replay_identity(expected(),arrays,radial,changed)['exact_production_replay']


def test_artifact_writes_are_separate_and_typed_hash_is_explicit(tmp_path):
    _,row,mesh,arrays,_=run_fake(invalid=True);before=copy.deepcopy(row['stage_seconds'])
    saved=save_record(tmp_path/'attempt',row,mesh,arrays)
    assert saved['stage_seconds']==before and saved['artifact_writing_outside_pipeline']
    assert saved['mesh_file_sha256']==file_hash(tmp_path/'attempt/mesh.npz')
    assert saved['mesh_typed_hash']==array_hash(mesh['vertices'],mesh['faces'])
    canonical=array_hash(mesh['vertices'],mesh['faces'].astype(np.int64))
    assert canonical!=saved['mesh_typed_hash']
    assert replay_identity(expected(),arrays,mesh,mesh)['canonical_geometry_hash']==canonical
    with pytest.raises(FileExistsError):save_record(tmp_path/'attempt',row,mesh,arrays)


def completed_rows():
    rows=[];parents=[]
    for case in cases():
        rows.append({**case,'mode':'cold','phase':'measured','repeat':1})
        parents.append({**case,'parent_process_wall_seconds':3.,'returncode':0})
    for case in warm_schedule():rows.append({**case,'mode':'warm'})
    for row in rows:
        row.update(runtime_status='ok',replay={'exact_production_replay':True},geometry_valid=True,budget512_match=True,
            mesh_total_seconds=1.,checked_total_seconds=2.,stage_seconds={name:.1 for name in STAGES})
    return rows,parents


def test_summary_keeps_fixed_denominators_and_all_failure_kinds():
    rows,parents=completed_rows();rows[0]['geometry_valid']=False
    rows[1]['replay']['exact_production_replay']=False
    failed=rows[2];case={k:failed[k] for k in ('seed','sample_id','arm')}
    rows[2]=failed_record(case,'cold','measured',1,'deliberate')
    parents[0]['returncode']=1
    summary=summarize(rows,parents)
    assert summary['cold_process_failure_count']==1 and not summary['all_cold_processes_completed_normally']
    with pytest.raises(ValueError,match='Cold parent process ledger'):
        summarize(rows,parents[:-1])
    assert summary['runtime_failures']==1 and summary['geometry_invalid_count_including_repeats']==1
    assert summary['all_recorded_outputs_exactly_match_production'] is False
    qem=next(x for x in summary['groups'] if x['seed']=='pooled' and x['mode']=='cold' and x['arm']=='qem')
    assert qem['planned_records']==8 and qem['runtime_success_count']==7
    assert qem['times_seconds']['mesh_total_seconds']['n']==7 and qem['cold_parent_wall_seconds']['n']==8
    assert len(rows)==120
    rows[-1]=copy.deepcopy(rows[-2])
    with pytest.raises(ValueError,match='duplicated or substituted'):summarize(rows,parents)


def test_bound_input_and_sample_array_tamper_rejected(tmp_path):
    path=tmp_path/'sample.npz';arrays=expected();np.savez(path,**arrays)
    digest=file_hash(path);plan={'bound_files_sha256':{str(path):digest}}
    record={'path':str(path),'file_sha256':digest,'array_hash':array_hash(*arrays.values()),'nodes_hash':array_hash(arrays['nodes'])}
    verify_bound_inputs(plan);assert np.array_equal(load_arrays(record)['nodes'],arrays['nodes'])
    arrays['nodes'][0]+=1;np.savez(path,**arrays)
    with pytest.raises(ValueError,match='Bound timing input changed'):verify_bound_inputs(plan)
    with pytest.raises(ValueError,match='Production sample file changed'):load_arrays(record)
    record['file_sha256']=file_hash(path)
    with pytest.raises(ValueError,match='arrays changed'):load_arrays(record)
