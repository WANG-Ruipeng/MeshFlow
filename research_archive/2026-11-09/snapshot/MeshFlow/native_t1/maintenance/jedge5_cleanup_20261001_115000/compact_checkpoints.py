"""One-time, non-destructive CPU export. Original files are never modified here."""
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from native_t1 import geometry_checkpoint as checkpoint
from native_t1.artifacts import atomic_json, file_sha256, state_sha256
from native_t1.objectives import objective_spec

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT/'native_t1/maintenance/jedge5_cleanup_20261001_115000'
SCREEN = ROOT/'experiments/mf_h1_local0/outputs/loss_screen_20260930_232200'
PILOT = ROOT/'native_t1/runs/l3_l5_interface_pilot_20261001_104600'
G0 = ROOT/'native_t1/checkpoints/tgeo_g0_cumulative2500.pt'
G0_SHA = '3a66659e86586f54d2ec30ed3d4a18275ab69ba0f8cba2866ef67b340c481c0e'
G0_STATE = '6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392'
JOBS = [(PILOT,'JEdge5','edge5','jedge5_cumulative3000.pt'),
        (SCREEN,'L0','fm','ablations/fm_cumulative3000.pt'),
        (SCREEN,'L5','surface','ablations/surface_cumulative3000.pt')]


def main():
    torch.set_num_threads(4)
    if (OUT/'checkpoint_migration.json').exists():
        raise FileExistsError('Migration already registered')
    assert file_sha256(G0) == G0_SHA
    report = dict(status='RUNNING', checkpoint_exports=[], model_forwards=0,
                  model_backwards=0, optimizer_updates=0, generations=0,
                  original_files_modified=False, cuda_initialized=False)
    atomic_json(OUT/'checkpoint_migration.json',report)
    try:
        for source_root,arm,recipe,relative in JOBS:
            source_record=json.loads((source_root/'results.json').read_text())['arms'][arm]['checkpoints']['500']
            source=source_root/arm/'additional_step500_cumulative_step3000.pt'
            destination=ROOT/'native_t1/checkpoints'/relative
            if destination.exists() or destination.with_suffix('.pt.partial').exists():
                raise FileExistsError(destination)
            assert source.stat().st_size == source_record['bytes']
            assert file_sha256(source) == source_record['sha256']
            old=torch.load(source,map_location='cpu',mmap=True,weights_only=True)
            assert old['arm']==arm and old['additional_updates']==500 and old['cumulative_updates']==3000
            assert old['base_cumulative_updates']==2500 and old['initial_checkpoint_sha256']==G0_SHA
            assert old['initial_state_sha256']==G0_STATE
            assert old['optimizer_reset_at_start'] is True
            assert old['batch_in_progress'] is False and old['optimizer_step_in_progress'] is False
            assert old['stream_start']['batch_index']==1000 and old['stream_start']['sample_index']==8000
            assert old['stream_state']['batch_index']==1500 and old['stream_state']['sample_index']==12000
            assert old['stream_state']['seed']==1010 and old['stream_state']['arm']=='Native_correct'
            expected=source_record['state_sha256']
            assert state_sha256(SimpleNamespace(state_dict=lambda:old['model']))==expected==old['model_state_sha256']
            spec=objective_spec(recipe)
            if recipe=='edge5':
                assert old['lambdas']['D_edge']==spec['lambda_edge'] and old['lambdas']['L5']==spec['lambda_surface']
                assert old['gate']=='FP32_t>=0.5' and old['ramp']=='min(additional_step/50,1)'
            elif recipe=='surface':
                # Historical screen uses one scalar coefficient per branch.
                assert old['lambda_geometry']==spec['lambda_surface']
                assert old['gate']=='FP32_t>=0.5' and old['ramp']=='min(additional_step/50,1)'
            payload=dict(schema=checkpoint.SCHEMA if recipe=='fm' else checkpoint.OBJECTIVE_SCHEMA,
                native_meta=old['native_meta'],context=old['context'],model=old['model'],optimizer={},
                stream_state=old['stream_state'],stream_seed=1010,stream_start_batch_index=1000,
                stream_start_sample_index=8000,input_sha256=old['input_sha256'],
                base_identity=dict(cumulative_updates=2500,checkpoint_sha256=G0_SHA,state_sha256=G0_STATE),
                config_sha256=old['config_sha256'],base_cumulative_updates=2500,completed_updates=500,
                cumulative_updates=3000,loss='masked_free_fm' if recipe=='fm' else 'masked_free_fm_plus_geometry',
                num_faces=112,train_K=[4,8,12],optimizer_reset_at_start=True,
                optimizer_step_in_progress=False,batch_in_progress=False,model_state_sha256=expected,
                optimizer_state_removed=True,provenance=dict(original_path=str(source.relative_to(ROOT)),
                    original_file_sha256=source_record['sha256'],original_schema=old['schema'],arm=arm,
                    protocol_sha256=old['protocol_sha256'],operation='lossless_model_and_input_rng_export_no_updates'))
            if recipe!='fm':
                payload['training_objective']=spec
            checkpoint.validate_geometry_checkpoint(payload)
            assert all(bool(torch.isfinite(v).all()) for v in payload['model'].values())
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix('.pt.partial')
            torch.save(payload,temporary)
            temporary.rename(destination)
            loaded,audit=checkpoint.load_geometry_checkpoint(destination,device='cpu',expected_recipe=recipe)
            state=loaded.state_dict()
            assert state.keys()==old['model'].keys()
            assert all(torch.equal(state[k].view(torch.uint8),old['model'][k].view(torch.uint8)) for k in state)
            assert audit['state_sha256']==expected
            restored=torch.load(destination,map_location='cpu',mmap=True,weights_only=True)
            assert restored['stream_state']==old['stream_state']
            assert restored['optimizer']=={} and restored['optimizer_state_removed'] is True
            row=dict(arm=arm,recipe=recipe,source=str(source.relative_to(ROOT)),source_bytes=source.stat().st_size,
                source_sha256=source_record['sha256'],destination=str(destination.relative_to(ROOT)),
                bytes=destination.stat().st_size,sha256=file_sha256(destination),state_sha256=expected,
                every_tensor_bitwise_equal=True,input_rng_identical=True,strict_load=audit)
            report['checkpoint_exports'].append(row)
            atomic_json(OUT/'checkpoint_migration.json',report)
            print(json.dumps({k:row[k] for k in ('arm','destination','bytes','sha256','state_sha256')}),flush=True)
            del loaded,state,restored,payload,old
            gc.collect()
        report['cuda_initialized']=torch.cuda.is_initialized()
        assert report['cuda_initialized'] is False
        report['status']='PASS'
    except BaseException as error:
        report.update(status='STOPPED',error=repr(error))
        raise
    finally:
        atomic_json(OUT/'checkpoint_migration.json',report)


if __name__=='__main__':
    main()
