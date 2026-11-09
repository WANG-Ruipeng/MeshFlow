"""Final ledger, protected hashes and actual new-code diff. No scientific calls."""
from pathlib import Path
import difflib,json,time
from .registration import ROOT,OUT,check_protected
from experiments.mf_h1_local0.jedge5_diagnosis.common import read_json,write_json,file_sha256

def finalize():
    analysis=read_json(OUT/'analysis/summary.json');assert analysis['status']=='PASS'
    local=read_json(OUT/'raw/local_rows.json');suffix=read_json(OUT/'raw/suffix_rows.json')
    assert len(local['rows'])==144 and len(local['supplemental_rows'])==12 and len(suffix['rows'])==36
    workers=[read_json(p) for p in sorted((OUT/'workers').glob('*.json'))]
    passed=[w for w in workers if w['status']=='PASS'];failed=[w for w in workers if w['status']!='PASS']
    assert len(passed)==12 and len(failed)==1 and failed[0]['counts']['model_forward_attempts']==1
    totals={k:sum(w['counts'][k] for w in workers) for k in workers[0]['counts']}
    assert totals['model_forward_attempts']==totals['model_forward_returns']==2797
    assert totals['suffix_returns']==36 and totals['euler_updates']==1764 and totals['optimizer_updates']==0 and totals['model_backward_calls']==0
    assert all(r['next_state_bitwise'] for r in local['rows']+local['supplemental_rows'])
    assert all(r['audit']['known_all_states_bitwise'] for r in suffix['rows'])
    protected=check_protected();assert protected['status']=='PASS'
    scores={m:read_json(OUT/f'scores/{m}/results.json') for m in ('B','J')}
    scoring_reg=read_json(OUT/'scores/registration.json')
    checks=[]
    def visit(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value and isinstance(value['path'],str):
                path=Path(value['path']);
                checks.append(dict(path=str(path),unchanged=path.is_file() and file_sha256(path)==value['sha256']))
            for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    visit(scoring_reg);assert all(c['unchanged'] for c in checks)
    write_json(OUT/'registration/scoring_protected_after.json',dict(status='PASS',checks=checks))
    own=ROOT/'experiments/mf_h1_local0/jedge5_local_error_B'
    paths=sorted(own.rglob('*.py'));parts=[]
    for path in paths:
        rel=path.relative_to(ROOT).as_posix();parts.extend(difflib.unified_diff([],path.read_text(encoding='utf-8-sig').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+rel))
    (OUT/'code.diff').write_text(''.join(parts),encoding='utf-8')
    write_json(OUT/'registration/executed_source_identity.json',dict(files=[dict(path=str(p),sha256=file_sha256(p)) for p in paths],public_source_protection=protected,public_scoring_protection_checks=len(checks)))
    phases={}
    for stage in ('local','suffix'):
        selected=[w for w in workers if w['stage']==stage]
        phases[stage]=dict(attempted_workers=len(selected),successful_workers=sum(w['status']=='PASS' for w in selected),lock_wait_seconds=sum(w.get('lock_wait_seconds',0) for w in selected),lock_owned_work_seconds=sum(w.get('lock_owned_work_seconds',0) for w in selected),measured_model_cuda_event_seconds=sum(w.get('network_cuda_event_seconds',0) for w in selected),forward_attempts=sum(w['counts']['model_forward_attempts'] for w in selected),peak_allocated_bytes=max(w.get('peak_cuda_allocated',0) for w in selected),peak_reserved_bytes=max(w.get('peak_cuda_reserved',0) for w in selected))
    result=dict(status='COMPLETE_WITH_RECORDED_IMPLEMENTATION_CORRECTION',window='B',training_updates=0,OT_calls=0,model_backward_calls=0,new_full_generations=0,missing_trajectory_replays=0,local_primary_forwards=864,t08_dependency_forwards=24,directional_forwards=144,suffix_forwards=1764,implementation_assertion_forward=1,total_model_forward_attempts=2797,total_model_forward_returns=2797,hard_limit=4500,counts=totals,phases=phases,cpu_analysis_timing=read_json(OUT/'registration/analysis_figure_timing.json'),cpu_resource_clarification=read_json(OUT/'registration/cpu_resource_clarification.json'),timing_note='Measured successful-network CUDA-event time excludes failed initial1F; its wall time is included. Lock-owned wall includes loading, CPU statistics, hashes, transfers. Shared-machine time is not isolated throughput.',resources=[{k:w.get(k) for k in ('stage','case_id','attempt','resources_before_lock','resources_after_lock','resources_models_loaded','resources_after_work','peak_cuda_allocated','peak_cuda_reserved','memory_peak_scope')} for w in workers],failure_records=failed,preflight_incidents=read_json(OUT/'registration/preflight_incidents.json'),gates=dict(primary_next_state_bitwise=144,supplemental_next_state_bitwise=12,all_suffix_C_bitwise=True,all_weights_after_sha_pass=True,all_protected_pass=True),scoring={m:{k:v for k,v in data.items() if k not in ('baseline_rows','suffix_rows')} for m,data in scores.items()},analysis=analysis,artifacts=dict(report='B_LOCAL_ERROR_REPORT.md',local_csv='local_defect_per_state.csv',propagation_csv='amplification_per_case.csv',code_diff='code.diff',raw='raw/',figures='figures/',tests='registration/'),limits=analysis['limits'])
    write_json(OUT/'results.json',result)
    print(json.dumps(dict(status=result['status'],forward_attempts=2797,new_full_generations=0,suffixes=36,protected_files=protected['files'])))
if __name__=='__main__':finalize()

