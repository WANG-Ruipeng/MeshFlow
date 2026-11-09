"""Bounded CPU tests for paired recipes, no model/forward/GPU construction."""
import copy
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
from meshflow_control.data import recipe_factorial as recipe
from meshflow_control.data import schedule40
from meshflow_control.data.io import atomic_json,digest,read_json,save_npz


def manifest_fixture():
    parents=['p%02d'%i for i in range(32)];tasks=[]
    for i,uid in enumerate(parents[:26]):
        for j in range(8 if i<25 else 3):
            tasks.append(dict(uid=uid,task_id=f'{uid}_train40_{j}',role='train',split='train',status='READY',
                N=128,K=51,ratio=.4,source_face_ids=list(range(51)),free_source_ids=list(range(51,128))))
    for j in range(250):
        uid=parents[j%32]
        tasks.append(dict(uid=uid,task_id=f'{uid}_train20_{j}',role='train',split='train',status='READY',
            N=128,K=26,ratio=.2,source_face_ids=list(range(26)),free_source_ids=list(range(26,128))))
    return dict(schema='meshflow_control_training_data_v1',splits={'train':parents},
                parents=[dict(uid=uid,N=128) for uid in parents],tasks=tasks)


@pytest.fixture
def prepared(tmp_path):
    data=tmp_path/'data';data.mkdir();manifest=manifest_fixture()
    for i,row in enumerate(manifest['parents']):
        full=np.random.RandomState(i).normal(size=(128,3,3)).astype(np.float32)
        p=data/(row['uid']+'.npz')
        save_npz(p,full_target=full.reshape(128,9),pre_ot_full=full*np.float32(.5),
                 model_vertices=full.reshape(-1,3),source_face_vertex_ids=np.arange(384,dtype=np.int64).reshape(128,3))
        row.update(npz=p.name,sha256=digest(p))
    atomic_json(data/'train_manifest.json',manifest)
    old=tmp_path/'old'/'stream';new=tmp_path/'new'/'streams'
    schedule40.prepare_training_plan(data,old,total_updates=6)
    schedule40.precompute(data,old,workers=2)
    recipe.prepare_recipe_plan(data,old,new)
    return data,old,new


def test_exact_mix_and_same_multiset_curricula_across_parent_block_boundaries():
    manifest=manifest_fixture();old=schedule40.make_plan(manifest,36)
    new=recipe.make_plan(manifest,old)
    assert new==recipe.make_plan(copy.deepcopy(manifest),copy.deepcopy(old))
    assert new['eligible20_task_count']==208 and new['eligible40_task_count']==203
    assert new['ratio_counts']=={'0.2':144,'0.4':144}
    preserved=('uid','N','alpha','gaussian_seed','epsilon_seed','permutation_seed','t_seed')
    for a,b in zip(old['records'],new['mixed_records']):
        assert all(a[key]==b[key] for key in preserved)
        if b['ratio']==.4:assert a['task_id']==b['task_id']
    for start in range(0,288,8):
        batch=new['mixed_records'][start:start+8]
        assert Counter((r['ratio'],r['alpha']==1) for r in batch)=={(.2,True):2,(.2,False):2,(.4,True):2,(.4,False):2}
    for name,order in new['curriculum_orders'].items():
        assert sorted(order)==list(range(288))
        assert order[192:]==list(range(192,288))
        expected=(.2,.4) if name=='C20_40_MIX' else (.4,.2)
        for batch_index in range(36):
            selected=[new['mixed_records'][i] for i in order[8*batch_index:8*batch_index+8]]
            assert len({r['uid'] for r in selected})==8 and sum(r['alpha']==1 for r in selected)==4
            if batch_index<24:assert {r['ratio'] for r in selected}=={expected[batch_index//12]}
    # The final MIX tail is identical, not independently reselected or reshuffled.
    assert Counter(new['curriculum_orders']['C20_40_MIX'])==Counter(new['curriculum_orders']['C40_20_MIX'])


def test_missing20_support_blocks_without_parent_substitution():
    manifest=manifest_fixture();old=schedule40.make_plan(manifest,6)
    manifest['tasks']=[r for r in manifest['tasks'] if not(r['ratio']==.2 and r['uid']=='p00')]
    with pytest.raises(ValueError,match='eight existing20'):recipe.make_plan(manifest,old)


def test_lambda_boundaries_and_recipe_mismatch():
    assert recipe.lambda_for_step('R1',1)==.25
    assert recipe.lambda_for_step('R2_MIX_H_ALL',4000)==.25
    assert recipe.lambda_for_step('R3',4000)==0 and recipe.lambda_for_step('R3',4001)==.25
    assert recipe.lambda_for_step('R4',6000)==.25
    assert recipe.lambda_for_step('C20_40_MIX',4000,'late')==0
    with pytest.raises(ValueError,match='mismatch'):recipe.lambda_for_step('R3',1,'all')


def test_new20_cache_missing_fails_without_writing_old_or_new(prepared):
    data,old,new=prepared
    before={str(p.relative_to(old)):(digest(p),p.stat().st_mtime_ns) for p in old.rglob('*') if p.is_file()}
    stream=recipe.RecipeFactorialStream(data,new,'R2')
    record=next(r for r in stream.records if r['ratio']==.2)
    with pytest.raises(RuntimeError,match='must be prepared'):stream.sample(record)
    assert not (new/'new20_ot_cache').exists()
    assert before=={str(p.relative_to(old)):(digest(p),p.stat().st_mtime_ns) for p in old.rglob('*') if p.is_file()}


def test_precompute_only_new20_and_actual_pairing_resume_pure_ot(prepared):
    data,old,new=prepared
    before={str(p.relative_to(old)):(digest(p),p.stat().st_mtime_ns) for p in old.rglob('*') if p.is_file()}
    receipt=recipe.precompute(data,new,workers=2)
    assert receipt['actual_new20_OT_returns']==24 and receipt['old40_maps_read']==48
    assert len(list((new/'new20_ot_cache').glob('*.npz')))==24
    assert before=={str(p.relative_to(old)):(digest(p),p.stat().st_mtime_ns) for p in old.rglob('*') if p.is_file()}
    original=schedule40.Schedule40Stream(data,old)
    first=recipe.RecipeFactorialStream(data,new,'R2')
    late=recipe.RecipeFactorialStream(data,new,'R4')
    r3=recipe.RecipeFactorialStream(data,new,'R3')
    hybrid_batch=first.next_effective_batch();pure_batch=late.next_effective_batch();old_batch=original.next_effective_batch()
    pure40=r3.next_effective_batch()
    for h,p in zip(hybrid_batch,pure_batch):
        assert h['sample_index']==p['sample_index'] and h['task_id']==p['task_id']
        np.testing.assert_array_equal(h['x1'],p['x1'])
        np.testing.assert_array_equal(h['ot_face_map'],p['ot_face_map'])
        assert h['lambda_value']==.25 and p['lambda_value']==0
        assert h['shared_input_sha256']!=p['shared_input_sha256']
        assert h['free_coordinate_count']==9*(int(h['y'])-h['K'])
    for p,o in zip(pure40,old_batch):
        np.testing.assert_array_equal(p['x1'],o['x1']);np.testing.assert_array_equal(p['t'],o['t'])
        expected=np.zeros((128,3,3),np.float32);expected[51:]=o['free_gaussian_before_OT']
        expected=expected[o['face_permutation']].reshape(128,9)
        np.testing.assert_array_equal(p['x0'],expected)
        np.testing.assert_array_equal(p['u'],p['x1']-expected)
        np.testing.assert_array_equal(p['xt'][p['known_mask']],p['context'][p['known_mask']])
    for h,o in zip(hybrid_batch,old_batch):
        if h['ratio']==.4:assert all(h[k]==o[k] for k in ('shared_input_sha256','input_bytehash','label_bytehash','ot_cache_key'))
    state=first.state_dict();resumed=recipe.RecipeFactorialStream(data,new,'R2_MIX_H_ALL');resumed.load_state_dict(state)
    assert [s['shared_input_sha256'] for s in first.next_effective_batch()]==[s['shared_input_sha256'] for s in resumed.next_effective_batch()]
    assert first.last_audit==resumed.last_audit
    with pytest.raises(ValueError,match='identity'):late.load_state_dict(state)
    with pytest.raises(ValueError,match='cursor'):resumed.load_state_dict(dict(state,batch_index=7,consumed_samples=56))
    paired=read_json(first.paired_path)['records']
    assert len(paired)==48 and paired[0]['shared_input_sha256']==hybrid_batch[0]['shared_input_sha256']
    again=recipe.precompute(data,new,workers=1)
    assert again['actual_new20_OT_returns']==0 and again['new20_cache_hits']==24


def test_curriculum_runtime_reuses_same_samples_and_tail(prepared):
    data,old,new=prepared;recipe.precompute(data,new,workers=2)
    mixed=recipe.RecipeFactorialStream(data,new,'R2')
    forward=recipe.RecipeFactorialStream(data,new,'C20_40_MIX',hybrid_schedule='all')
    reverse=recipe.RecipeFactorialStream(data,new,'C40_20_MIX',hybrid_schedule='all')
    hashes={s['origin_sample_index']:s['shared_input_sha256'] for step in range(1,7) for s in mixed.samples(step)}
    for stream in (forward,reverse):
        seen=[]
        for step in range(1,7):
            samples=stream.samples(step)
            assert len({s['object_id'] for s in samples})==8
            for sample in samples:
                assert sample['shared_input_sha256']==hashes[sample['origin_sample_index']]
                assert sample['step']==step;seen.append(sample['sample_index'])
        assert sorted(seen)==list(range(48))
        for step in (5,6):assert [s['shared_input_sha256'] for s in stream.samples(step)]==[s['shared_input_sha256'] for s in mixed.samples(step)]

def test_namespace_cannot_masquerade_as_independent_training_repeat():
    manifest=manifest_fixture();old=schedule40.make_plan(manifest,6)
    with pytest.raises(ValueError,match='Independent confirmation BLOCKED'):
        recipe.make_plan(manifest,old,namespace='independent_confirmation')


def test_tampered_valid20_task_and_course_order_are_rejected(prepared):
    data,old,new=prepared
    path=new/'recipe_plan.json';saved=read_json(path)
    tampered=copy.deepcopy(saved)
    record=next(r for r in tampered['mixed_records'] if r['ratio']==.2)
    record['task_id']=next(t for t in tampered['eligible20_tasks'][record['uid']] if t!=record['task_id'])
    atomic_json(path,tampered)
    with pytest.raises(ValueError,match='deterministic registered construction'):
        recipe.RecipeFactorialStream(data,new,'R2')
    tampered=copy.deepcopy(saved)
    order=tampered['curriculum_orders']['C20_40_MIX'];order[0],order[1]=order[1],order[0]
    atomic_json(path,tampered)
    with pytest.raises(ValueError,match='deterministic registered construction'):
        recipe.RecipeFactorialStream(data,new,'C20_40_MIX')


def test_registered_course_tables_cover_each_actual_origin_and_lambda(prepared):
    data,old,new=prepared;recipe.precompute(data,new,workers=2)
    assert read_json(new/'precompute_progress.json')['status']=='COMPLETE'
    for course in ('C20_40_MIX','C40_20_MIX'):
        for policy in ('all','late'):
            stream=recipe.RecipeFactorialStream(data,new,course,hybrid_schedule=policy)
            rows=read_json(stream.paired_path)['records']
            assert [r['sample_index'] for r in rows]==[r['sample_index'] for r in stream.records]
            registered={r['sample_index']:r for r in rows}
            for step in range(1,7):
                for sample in stream.samples(step):
                    assert recipe.paired_record(sample)==registered[sample['sample_index']]
