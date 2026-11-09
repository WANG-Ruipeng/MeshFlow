"""Close only the completed registered experiment; no model execution."""
import ast,csv,json,datetime
from pathlib import Path
from . import common as C

def main():
    O=C.OUT
    audit=C.read(O/'independent_audit/final_audit.json')
    account=C.read(O/'call_accounting.json')
    review=C.read(O/'evaluation/gallery/review_completion.json')
    summary=C.read(O/'summary.json')
    assert audit['status']=='PASS' and not audit['errors']
    assert account['generator_rollouts']==144 and account['generator_returned']==7200 and account['D0_returned']==144
    assert account['training_updates']==account['backwards']==account['unresolved_model_calls']==0
    assert review['rows']==288 and review['style_rows']==24
    assert len(summary['per_condition'])==48 and len(summary['per_panel'])==12
    assert (O/'REPORT.md').is_file()
    for p in C.CODE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    important=[O/'REPORT.md',O/'README_RESUME.md',O/'summary.json',O/'registration.json',O/'call_accounting.json',O/'independent_audit/final_audit.json',O/'independent_audit/final_audit.md',O/'evaluation/gallery/review_completion.json',O/'evaluation/per_output.csv',O/'results/per_condition.csv',O/'results/paired_decisions.csv']
    result=dict(status='COMPLETE',experiment='R2_SEAT_NOISE_TRANSFER_V1',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        counts=dict(RAW=144,POST=144,scored_stages=288,request_pages=48,contact_sheets=8,actual_AI_PROXY_reviews=288,style_reviews=24,generator_forwards=7200,D0_forwards=144,training_updates=0,backwards=0),
        decision='Local POST attachment gain repeats on familiar C; no net adoption gain; DEV transfer weak and representation-confounded. No default promotion.',
        audit='PASS',scientific_work_remaining=0,default_changed=False,automatic_training=False,commit_or_push=False,
        important_hashes={str(p):C.sha(p) for p in important},review_package='Packaging is a separate verified export; package receipt supplies ZIP hash and size.')
    C.resources(O/'resources_completion.json')
    C.write(O/'COMPLETE.json',result)
    C.event('EXPERIMENT_COMPLETE',RAW=144,POST=144,scored_stages=288,training_updates=0,default_changed=False)
    print(json.dumps({k:v for k,v in result.items() if k!='important_hashes'},ensure_ascii=False))
if __name__=='__main__':main()