"""Finalize completed read-only analyses and create a small cloud evidence supplement."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import re
import subprocess
import zipfile

ROOT=Path('/mnt/e/MeshFlow')
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'native_t1/runs/chair_domain_control_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run():
    before=read(OUT/'source_hashes_before.json')
    after={rel:{'bytes':(ROOT/rel).stat().st_size,'sha256':sha(ROOT/rel)} for rel in before}
    assert after==before, 'Protected source changed'
    write('source_hashes_after.json',after)
    reported=read(SOURCE/'reporting_status.json')['artifacts']
    for record in reported:
        assert sha(SOURCE/record['path'])==record['sha256'],record['path']
    codes=read(SOURCE/'registration/code_hashes.json')
    for rel,expected in codes.items(): assert sha(ROOT/rel)==expected,rel
    stages={}
    for name in ['width_analysis','matrix_audit','coverage_audit','teacher_audit']:
        audit=read(OUT/name/'audit.json')
        assert audit['status']=='PASS',(name,audit.get('status'))
        stages[name]={'status':'DONE','audit_path':name+'/audit.json','audit_sha256':sha(OUT/name/'audit.json')}
    matrix=read(OUT/'matrix_audit/audit.json')['findings']
    assert matrix['source_matrix_cells']==648 and matrix['complete_matrices']==72
    assert matrix['changed_advantage_signs']==117
    width=read(OUT/'width_analysis/audit.json')
    assert width['raw_files_verified']==216 and width['max_raw_scalar_abs_difference']==0
    with (OUT/'width_analysis/width_per_curve.csv').open() as f:
        curves=list(csv.DictReader(f))
    validation=[r for r in curves if r['phase_step']=='3000' and r['split']=='validation']
    assert len(validation)==24 and all(float(r['baseline_ratio'])<.9 for r in validation)
    metrics_equal={}
    for name in ['A_context_per_case.csv','B_width_geometry_per_case.csv']:
        with (SOURCE/name).open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        valid=[r for r in rows if r['status']=='PASS']
        assert all(float(r['FF_proper_affected_free_fraction'])==float(r['FF_illegal_affected_free_fraction']) for r in valid)
        metrics_equal[name]={'valid_rows':len(valid),'FF_proper_equals_illegal_affected_fraction':True}
    links=[]
    for name in ['FOLLOWUP_ANALYSIS_REPORT.md','ERRATA_MATRIX_DIRECTION.md']:
        f=OUT/name; content=f.read_text()
        assert '\ufffd' not in content
        for link in re.findall(r'\]\(([^)]+)\)',content):
            if '://' in link: continue
            assert (f.parent/link.split('#')[0]).is_file(),link
            links.append(link)
    ledger={'schema':'chair_existing_evidence_followup_execution_v1','status':'DONE',
            'scope':'CPU aggregation of existing results, independent raw width arithmetic and code review',
            'new_optimizer_updates':0,'new_model_forwards':0,'new_model_backwards':0,'new_generations':0,
            'new_teacher_calls':0,'new_OT_calls':0,'new_geometry_scorer_calls':0,'new_point_triangle_distance_calls':0,
            'GPU_workers':0,'CPU_worker_limit':4,'per_worker_threads':1,
            'actual_peak_concurrent_CPU_processes':'NA: short independent jobs; no continuous CPU process peak sampling',
            'scientific_script_executions':[
                {'stage':'width','runs':2,'successes':2,'failures':0,'raw_width_recomputations_per_run':216,
                 'actual_total_raw_width_recomputations':432,'unique_raw_outputs':216,
                 'reason_for_second_run':'Add exact cross-alpha position aggregation and shared plot axes; no change to experiment'},
                {'stage':'matrix','runs':1,'successes':1,'failures':0,'existing_matrix_cells':648},
                {'stage':'coverage','runs':1,'successes':1,'failures':0,'identity_and_numeric_checks':9920},
                {'stage':'teacher','runs':2,'successes':2,'failures':0,'rows_per_run':288,
                 'reason_for_second_run':'Clarify mean-vs-parent interpretation and teacher/generation condition differences'}],
            'display_only_scatter_rerenders':1,
            'additional_readonly_work':'Schema/code/resource reads, CSV numeric readbacks, reviewer audits, plot inspection, hashing and packaging. These are not model or geometry scoring calls.',
            'recorded_non_scientific_failures':[
                {'stage':'coverage_tool_orchestration','count':1,'reason':'JavaScript string syntax error before process start; no analysis execution or file write'},
                {'stage':'teacher_text_read','count':1,'reason':'UTF-8 text search mismatch via shell; no scientific analysis failure'},
                {'stage':'width_independent_reviewer','count':1,'reason':'Read-only numeric checks printed, then local Path variable overwritten and audit print failed; subsequent read completed'}],
            'model_budget_and_original_ledger_unchanged':True,'source_files_before_after_equal':len(before),
            'original_reported_artifacts_verified':len(reported),'original_frozen_code_files_verified':len(codes),
            'FF_metric_alias_checks':metrics_equal,'report_links_checked':len(links)}
    write('EXECUTION_LEDGER.json',ledger)
    resources={}
    for key,args in [('gpu',['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
                     ('gpu_compute_processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'])]:
        result=subprocess.run(args,capture_output=True,text=True)
        resources[key]={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    resources['meminfo']=Path('/proc/meminfo').read_text()
    write('resources_after.json',resources)
    completion={'status':'DONE','finished_utc':datetime.now(timezone.utc).isoformat(),'stages':stages,
                'read_only_source_integrity':'PASS','original_reported_artifacts_verified':len(reported),
                'original_frozen_code_files_verified':len(codes),'new_model_calls':0,'new_training_updates':0,
                'new_generations':0,'new_geometry_scoring_calls':0,
                'raw_width_unique_outputs_rechecked':216,'raw_width_values_per_unique_output':18,
                'all_registered_matrix_cells_aggregated':648,'case_level_width_decomposition_rows':216,
                'existing_teacher_rows':288,'existing_A_requests':192,'existing_A_valid_requests':174,
                'existing_A_NA_requests':18,'known_metric_issue':'Old correct-column fields used fixed-output rows; corrected column outputs are separate; old files unchanged',
                'scope_limit':'No causal claim, new data or noise, model evaluation, training extension, loss or solver change'}
    write('RUN_COMPLETE.json',completion)
    selected=['00_README.md','FOLLOWUP_ANALYSIS_REPORT.md','ERRATA_MATRIX_DIRECTION.md',
              'EXECUTION_LEDGER.json','RUN_COMPLETE.json',
              'width_analysis/WIDTH_DECOMPOSITION_REPORT.md','width_analysis/width_per_curve.csv',
              'width_analysis/width_decomposition_per_case.csv','width_analysis/width_start_end_per_parent.csv',
              'width_analysis/noise_response.png',
              'matrix_audit/MATRIX_DIRECTION_AUDIT.md','matrix_audit/fixed_target_beta_per_case.csv',
              'matrix_audit/axis_summary_parent_equal.csv',
              'coverage_audit/COVERAGE_AUDIT_REPORT.md','coverage_audit/comparison_parent.csv','coverage_audit/tradeoff_summary.csv',
              'teacher_audit/TEACHER_AUDIT_REPORT.md','teacher_audit/teacher_endpoint_changes.csv']
    files={name:OUT/name for name in selected}
    for name in ['B_width_response.csv','B_cross_condition_scores.csv','teacher_per_case.csv']:
        files['source/'+name]=SOURCE/name
    for name in ['width_analytics.py','reporting.py']:
        files['source_code/'+name]=ROOT/'experiments/mf_h1_local0/chair_domain_control_v1'/name
    manifest={'scope':'Small results supplement; raw/weights/full logs and most intermediate tables omitted',
              'files':[{'path':name,'bytes':path.stat().st_size,'sha256':sha(path),
                        'source_repository_path':path.relative_to(ROOT).as_posix()} for name,path in sorted(files.items())]}
    write('FILES_SHA256.json',manifest)
    files['FILES_SHA256.json']=OUT/'FILES_SHA256.json'
    final=ROOT/'deliverables/chair_followup_analysis_brief_v1.zip'
    partial=final.with_suffix('.zip.partial')
    assert not final.exists() and not partial.exists(),'Do not overwrite an existing delivery'
    prefix='CHAIR_FOLLOWUP_ANALYSIS/'
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(files.items()):z.write(path,prefix+name)
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)
        assert set(z.namelist())=={prefix+name for name in files}
        assert json.loads(z.read(prefix+'FILES_SHA256.json'))==manifest
        for record in manifest['files']:
            payload=z.read(prefix+record['path'])
            assert len(payload)==record['bytes'] and hashlib.sha256(payload).hexdigest()==record['sha256']
    partial.rename(final)
    checksum=sha(final)
    final.with_suffix('.zip.sha256').write_text(checksum+'  '+final.name+'\n',encoding='ascii')
    result={'status':'PASS','zip':str(final),'files':len(files),'bytes':final.stat().st_size,
            'zip_sha256':checksum,'all_member_SHA256_and_CRC_verified':True,'source_integrity':'PASS'}
    write('PACKAGE_VERIFICATION.json',result)
    # Local complete analysis manifest also covers scripts and detailed tables omitted from the small ZIP.
    local={str(f.relative_to(OUT)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(OUT.rglob('*'))
           if f.is_file() and f.name!='ANALYSIS_ARTIFACTS.json'}
    write('ANALYSIS_ARTIFACTS.json',{'status':'PASS','files':local,'manifest_self_excluded':True})
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
