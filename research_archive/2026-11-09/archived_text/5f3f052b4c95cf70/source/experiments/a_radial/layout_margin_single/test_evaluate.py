import numpy as np
import pytest
from . import evaluate as e
from .prepare import job_table,UIDS
from ..io_utils import save_npz


def protocol():return {'jobs':job_table('file','model')}


def test_complete_fixed_denominators():
    assert len(e.slots(protocol(),'diagnostic'))==24
    assert len(e.slots(protocol(),'final'))==20
    assert {u for j,u in e.slots(protocol(),'diagnostic')}=={0,5000,20000}


def test_missing_failed_and_unqualified_stay_in_accounting():
    attempts=[{'uid':UIDS[0],'endpoint':'old_final','status':'success','qualified_original':{'qualified':False}},
              {'uid':UIDS[0],'endpoint':'margin_final','status':'error','error':'intentional'}]
    result=e.summarize_slots(protocol(),attempts,'final')
    assert result['planned_slots']==20 and result['recorded_slots']==2 and result['success_count']==1
    assert len(result['errors'])==1 and len(result['unattempted'])==18 and len(result['missing_success_slots'])==19
    assert not result['rows'][0]['qualified_original']['qualified']
    with pytest.raises(ValueError):e.summarize_slots(protocol(),attempts+[attempts[0]],'final')


def test_observation_paths_do_not_accept_best_checkpoint(tmp_path):
    assert e.checkpoint_path(tmp_path,'job',0).name=='initial.pt'
    assert e.checkpoint_path(tmp_path,'job',5000).name=='step_05000.pt'
    assert e.checkpoint_path(tmp_path,'job',20000).name=='final.pt'
    with pytest.raises(ValueError):e.checkpoint_path(tmp_path,'job',19999)


def test_mesh_readback_rejects_stale_radius_and_keeps_dtype_semantics(tmp_path):
    class Field:
        def query(self,logr,u):return np.ones(len(u),dtype=np.float64)
    faces=np.array([[0,1,2]],np.int32);u=np.eye(3,dtype=np.float64)
    data={'faces':faces,'logr':np.zeros(1)}
    mesh={'vertices':u.copy(),'faces':faces.astype(np.int64),'u':u,'r':np.ones(3)}
    path=tmp_path/'good.npz';save_npz(path,**mesh)
    result=e.check_raw_readback(path,mesh,Field(),data)
    assert result['mesh_hash']!=result['stored_array_hash'] and result['stored_faces_dtype']=='int64'
    stale={**mesh,'r':np.full(3,2.)};bad=tmp_path/'bad.npz';save_npz(bad,**stale)
    with pytest.raises(ValueError,match='radii stale'):e.check_raw_readback(bad,stale,Field(),data)


def test_zero_teacher_keeps_absolute_error_but_no_ratio():
    base=np.eye(3,dtype=np.float64);actual=base.copy();actual[0]=[.999,.01,0]
    result=e.endpoint_probe(actual,base,base,np.array([[0,1,2]],np.int32),1.)
    assert result['endpoint_mse_over_zero_motion'] is None
    assert result['zero_motion_has_zero_teacher_error'] is True and result['endpoint_direction_mse_normalized']>0


def test_partial_attempt_is_recorded_not_retried(tmp_path):
    folder=tmp_path/'slot';folder.mkdir();(folder/'directions.npz').write_bytes(b'partial')
    assert not e.begin_attempt(folder,{'job':'registered','update':0})
    result=e.read(folder/'RESULT.json')
    assert result['status']=='error' and 'no automatic retry' in result['error']
    assert (folder/'directions.npz').read_bytes()==b'partial'


def test_existing_attempt_cannot_change_identity(tmp_path):
    folder=tmp_path/'slot'
    assert e.begin_attempt(folder,{'job':'registered','update':0})
    e.write_json(folder/'RESULT.json',{'job':'registered','update':0,'status':'error'})
    with pytest.raises(ValueError):e.begin_attempt(folder,{'job':'registered','update':5000})
