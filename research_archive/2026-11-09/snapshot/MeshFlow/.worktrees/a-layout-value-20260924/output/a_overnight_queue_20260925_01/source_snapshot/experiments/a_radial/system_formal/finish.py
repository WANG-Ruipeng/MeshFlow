"""Post-budget finish command: immutable official sampling and explicit five-row tables.

Run only after run_all has completed both50000-step seeds. A completed budget or
completed evaluation does not confer convergence/generalization qualification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
from ..io_utils import write_json,file_hash,new_dir
from ..system_probe.official import require,exact_same
from ..system_probe import ours_metrics as ours
from . import sample


def ready_completion(run,seed,protocol_sha):
    done=sample.read(Path(run)/'training'/f'seed_{seed}'/'TRAINING_COMPLETE.json')
    require(done['status']=='FIXED_BUDGET_COMPLETE_UNQUALIFIED' and done['seed']==seed
        and done['step']==50000 and done['primary_weight']=='ema' and done['protocol_sha256']==protocol_sha,
        'Both registered50000 EMA training completions must exist before finish')
    return done


def ours_for_table(run,sp,seed,tail=False):
    if not tail:
        registration=sample.register_ours(sp['primary_ours']['run'],sp['dataset_manifest_sha256'])
        exact_same(registration,sp['primary_ours'],'main_original8k_registration')
        source=Path(registration['run']);inputs=ours.seed_inputs(source,seed,sample.read(source/'RUN_PROTOCOL.json'))
        return inputs['export_dir'],{'role':'PRIMARY_ORIGINAL8K','registration':registration,
            'sample_manifest_sha256':inputs['sample_manifest_sha256'],'outlet_manifest_sha256':inputs['outlet_manifest_sha256']}
    registration=sp['optional_tail_ours'];require(registration is not None,'No optional64k comparison registered')
    from ..fm_tail import metrics as tail_metrics
    source=Path(registration['run'])
    require(file_hash(source/'RUN_PROTOCOL.json')==registration['training_protocol_sha256']
        and file_hash(source/'ARTIFACT_PROTOCOL.json')==registration['artifact_protocol_sha256'],'Optional tail protocol changed')
    inputs=tail_metrics.seed_inputs(source,seed)
    require(inputs['primary'] and inputs['identity']['FM_step']==64000 and inputs['sample_manifest']['fm_steps']==32,'Optional comparison is not fixed64k Euler32')
    require(inputs['sample_manifest']['dataset_manifest_sha256']==sp['dataset_manifest_sha256'],'Optional comparison data differ')
    require(len(inputs['sample_records'])==64 and all(len(x)==64 for x in inputs['outlet_records'].values()),'Optional comparison dropped planned slots')
    for sid in sample.slots():
        record=inputs['sample_records'][sid]
        if record['status']=='ok':ours.load_sample(inputs,sid)
        for arm in ours.ARMS:
            row=ours.verify_outlet_binding(inputs,arm,sid,record)
            if row['status']=='ok':require(file_hash(sample.inside(inputs['export_dir'],row['mesh_path']))==row['mesh_sha256'],'Optional tail mesh changed')
    return inputs['export_dir'],{'role':'SEPARATE_ADDITIONAL64K_NOT_PRIMARY_REPLACEMENT','registration':registration,
        'generation_identity':inputs['identity'],'sample_manifest_sha256':inputs['sample_manifest_sha256'],
        'outlet_manifest_sha256':inputs['outlet_manifest_sha256'],'auxiliary_Euler128_excluded':True}


def check_table(path,expected_rows):
    rows=sample.read(Path(path)/'MATRIX.json')
    require([row['row'] for row in rows]==expected_rows,'Five-row order/name changed')
    require(all(row['attempts']==64 and row['full_publication_ranking_qualified'] is False and row['not_heldout'] is True for row in rows),
        'Fixed64 denominator or closed-set qualification changed')
    post=next(row for row in rows if row['row']=='meshflow_public_post')
    require(post['paper_full_post'] is False,'Public post falsely labeled paper full')
    per=sample.read(Path(path)/'PER_SAMPLE.json')
    require(len(per)==320 and len({(row['row'],row['sample_id']) for row in per})==320,'Per-sample table lost fixed slots')
    for name in expected_rows:require({r['sample_id'] for r in per if r['row']==name}==set(sample.slots()),'Five-row sample IDs changed')
    return rows


def evaluate_one(run,sp,seed,official_identity,tail=False):
    run=Path(run);official=run/'posttraining'/f'seed_{seed}'
    completed=sample.audit_completed(official,official_identity)
    ours_dir,ours_identity=ours_for_table(run,sp,seed,tail)
    role='five_row_tail64000' if tail else 'five_row_main'
    root=run/'posttraining'/role;root.mkdir(parents=True,exist_ok=True)
    target=root/f'seed_{seed}';record_path=root/f'seed_{seed}_INVOCATION.json'
    require(not target.exists() and not record_path.exists(),'Never overwrite or automatically rerun an evaluated table')
    command=[sys.executable,'-B','-m','experiments.a_radial.evaluate','--data',sp['data'],
        '--ours',str(ours_dir),'--meshflow',str(official/'evaluation_view'),'--out',str(target),'--exact','--points','2048']
    result=subprocess.run(command,cwd=Path(__file__).resolve().parents[3],capture_output=True,text=True)
    record={'seed':seed,'comparison_role':role,'command':command,'returncode':result.returncode,
        'stdout':result.stdout,'stderr':result.stderr,'ours_identity':ours_identity,'official_identity':official_identity,
        'sampling_complete_sha256':file_hash(official/'SAMPLING_COMPLETE.json'),
        'official_evaluation_inventory_sha256':completed['evaluation_inventory_sha256'],
        'sampling_protocol_sha256':file_hash(run/'SAMPLING_PROTOCOL.json'),
        'original_evaluate_sha256':file_hash(Path(__file__).resolve().parents[1]/'evaluate.py'),
        'main8k_reference_never_replaced':True,'all_attempts_preserved':True,'paper_full_post_qualified':False,
        'fixed_budget_not_convergence_or_external_ranking_proof':True}
    if result.returncode==0:
        try:
            record['five_rows']=check_table(target,sp['evaluation']['rows']);record['table_contract_status']='PASS'
        except Exception as exc:record.update(table_contract_status='FAIL',table_contract_error=repr(exc))
    else:record['table_contract_status']='PROCESS_FAILED'
    record['output_inventory_sha256']=sample.inventory(target);write_json(record_path,record)
    require(record['table_contract_status']=='PASS','Original evaluator or fixed five-row contract failed; outputs retained')
    return {'seed':seed,'comparison_role':role,'path':str(target),'invocation_sha256':file_hash(record_path),
        'five_rows':record['five_rows'],'paper_full_post_qualified':False,'source_outputs_unchanged':True}


def run(run):
    run=Path(run).resolve();p,cfg,identity,sp,sha=sample.contract(run,check_ours=True)
    for seed in sp['seeds']:ready_completion(run,seed,identity['protocol_sha256'])
    root=run/'posttraining';root.mkdir(parents=True,exist_ok=True)
    marker=root/'FINISH_STARTED.json';require(not marker.exists(),'Finish already attempted; inspect saved records, never silently rerun samples')
    write_json(marker,{'sampling_protocol_sha256':sha,'training_protocol_sha256':identity['protocol_sha256'],
        'source_sha256':file_hash(__file__),'main_ours_FM_step':8000,'official_step':50000,'official_weight':'ema',
        'samples_per_seed':64,'seeds':[10,11],'no_automatic_retry_or_checkpoint_selection':True})
    primary=[];optional=[];sample_records=[]
    try:
        for seed in sp['seeds']:
            print(json.dumps({'event':'official_fixed_EMA_sampling','seed':seed,'requested':64,'step':50000}),flush=True)
            sample_result=sample.run_seed(run,seed);sample_records.append({'seed':seed,
                'successful':sample_result['successful'],'failed':sample_result['failed'],
                'sampler_returncode':sample_result['sampler_returncode'],
                'sampling_complete_sha256':file_hash(root/f'seed_{seed}'/'SAMPLING_COMPLETE.json')})
            official_identity=sample_result['identity']
            primary.append(evaluate_one(run,sp,seed,official_identity,False))
            if sp['optional_tail_ours'] is not None:
                try:optional.append(evaluate_one(run,sp,seed,official_identity,True))
                except Exception as exc:
                    record={'seed':seed,'status':'UNAVAILABLE_OR_FAILED','error':repr(exc),'traceback':traceback.format_exc(),
                        'does_not_replace_or_invalidate_registered_main8k_table':True,'no_automatic_retry':True}
                    optional.append(record);write_json(root/f'OPTIONAL_TAIL_SEED{seed}_FAILED.json',record)
        sample.contract(run,check_ours=True)
        completed={'status':'FIXED_BUDGET_SAMPLING_AND_MAIN_TABLE_DONE_UNQUALIFIED','sampling_protocol_sha256':sha,
            'training_protocol_sha256':identity['protocol_sha256'],'sampling':sample_records,'primary_original8k_tables':primary,
            'separate_additional64k_tables':optional,'primary_samples_per_seed_per_row':64,'primary_rows_per_seed':5,
            'primary_planned_metric_slots':640,'paper_full_post_qualified':False,
            'sampling_failures_are_not_replaced':True,'fixed_budget_does_not_prove_convergence':True,
            'closed_training16_only_not_generalization':True,'external_publication_ranking_qualified':False}
        write_json(root/'FINISH_COMPLETE.json',completed)
        return completed
    except BaseException as exc:
        write_json(root/'FINISH_FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),
            'completed_sampling':sample_records,'completed_primary_tables':primary,'optional_tables':optional,
            'all_existing_outputs_retained':True,'no_automatic_sampling_retry':True});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);args=parser.parse_args();run(args.run)
