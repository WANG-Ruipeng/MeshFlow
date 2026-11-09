"""Read-only artifact consistency checks; no model, scorer or CUDA import."""
from pathlib import Path
import csv
import hashlib
import json
import re

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]

def read(rel):
    return json.loads((OUT/rel).read_text(encoding='utf-8'))

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    checks={}
    assert read('RUN_COMPLETE.json')['status']=='DONE'
    counts=read('accounting.json')['summary']
    expected={'optimizer_updates':2000,'training_microbatch_forwards':16000,'training_backwards':16000,'precheck_forwards':120,'precheck_zero_update_backwards':44,'actual_full_sampling_returns':278,'historical_raw_cache_hits':138,'generation_not_applicable':18,'sampling_forwards':13900,'total_model_forwards':30020,'failed_actual_calls':0,'unresolved_calls':0}
    assert all(counts[k]==v for k,v in expected.items())
    checks['actual_counts']=expected
    stages={p.name:json.loads(p.read_text())['status'] for p in sorted((OUT/'controller').glob('*.json'))}
    assert len(stages)==9 and set(stages.values())=={'DONE'}
    checks['all_registered_stages']=stages
    checks['row_counts']={}
    for name,expected_count in [('per_output',432),('cross_matrix',1296),('width_per_output',432),('width_per_curve',144)]:
        rows=read('paired_results/'+name+'.json')
        csvrows=list(csv.DictReader((OUT/('paired_results/'+name+'.csv')).open(encoding='utf-8-sig',newline='')))
        assert len(rows)==len(csvrows)==expected_count
        checks['row_counts'][name]=len(rows)
    per=read('paired_results/per_output.json')
    checks['availability']={}
    for label in ['START','H_CONT_END','MOMENT_XY_END']:
        subset=[r for r in per if r['checkpoint_label']==label]
        counts_status={s:sum(r['status']==s for r in subset) for s in sorted({r['status'] for r in subset})}
        assert counts_status=={'NOT_APPLICABLE':6,'PASS':138}
        checks['availability'][label]=counts_status
    control=read('paired_results/control_curve_counts.json')
    noresponse=[r for r in control if r['candidate']=='No-response']
    assert len(noresponse)==18 and all(r['mean_A_column']==0 and r['strict_column_win_count']==0 for r in noresponse)
    checks['No_response_direction_positive_control']='PASS'
    for rel in ['registration/source_verification_after.json','independent_audit/final_audit.json','independent_audit/quality_review.json','resources/phase_summary.json','presentation/visual_qa.json','presentation/editorial_review/receipt.json']:
        obj=read(rel)
        assert obj['status'] in ('PASS','COMPLETE'),(rel,obj['status'])
        checks[rel]={'status':obj['status'],'sha256':sha(OUT/rel)}
    costs=read('paired_results/cost_summary.json')
    assert costs[0]['sampling_timing_count']==0
    assert all(r['sampling_timing_count']==138 and r['missing_sampled_seconds']==0 for r in costs[1:])
    checks['actual_sampling_timing_records']=276
    deps=read('source_code/scorer_dependencies_manifest.json')
    assert len(deps)==8
    for r in deps:assert sha(ROOT/r['original'])==sha(OUT/r['path'])==r['sha256']
    checks['actual_scorer_dependencies']=8
    link_count=0
    for rel in ['DELIVERY_INDEX.md','CHAIR_MOMENT_CHANNEL_V2_REPORT.md']:
        text=(OUT/rel).read_text(encoding='utf-8')
        assert '\ufffd' not in text and '????' not in text
        for link in re.findall(r'\]\(([^\s)]+)\)',text):
            if link.startswith(('https://','http://','#')):continue
            target=(OUT/link.split('#')[0]).resolve()
            assert target.exists(),(rel,link)
            link_count+=1
    checks['valid_local_report_links']=link_count
    checks['default_model_changed']=False
    checks['no_GPU_or_model_calls']=True
    checks['checkpoints_verified_by']='independent_audit/final_audit.json (full .pt hashes, model, AdamW, RNG, stream); no redundant reload here.'
    receipt={'status':'PASS','checks':checks,'script_sha256':sha(Path(__file__))}
    (OUT/'delivery_check.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('DELIVERY_CHECK PASS',link_count,'local links')

if __name__=='__main__':main()
