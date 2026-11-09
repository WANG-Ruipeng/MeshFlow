"""Seal this completed fixed experiment; no model calls or default changes."""
import datetime,json,ast
from . import common as C

def main():
    O=C.OUT
    audit=C.read(O/'independent_audit/final_audit.json')
    account=C.read(O/'call_accounting.json');review=C.read(O/'evaluation/gallery/review_completion.json');summary=C.read(O/'summary.json')
    assert audit['status']=='PASS'
    assert account['status']=='PASS'
    assert account['budget']['optimizer_updates']==200 and account['budget']['training_forwards']==1600 and account['budget']['training_backwards']==1600
    assert account['budget']['new_generator_forwards']==4800 and account['budget']['new_D0_forwards']==96
    assert review['rows']==384 and review['style_rows']==32
    assert len(summary['per_condition'])==64 and len(summary['per_panel'])==16
    assert (O/'REPORT.md').is_file()
    for p in C.CODE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    important=[O/'REPORT.md',O/'README_RESUME.md',O/'summary.json',O/'registration.json',O/'evaluation/registration.json',O/'call_accounting.json',O/'independent_audit/final_audit.json',O/'evaluation/gallery/review_completion.json',O/'evaluation/per_output.csv',O/'results/per_condition.csv',O/'results/paired_decisions.csv',O/'training/CREF_C200/status.json']
    result=dict(status='COMPLETE',experiment='R2_C_REFERENCE_FRAME_V1',completed_utc=C.utc(),
      counts=dict(new_RAW=96,new_POST=96,cached_RAW=96,cached_POST=96,new_scored_stages=192,cached_scored_stages=192,request_pages=48,contact_sheets=8,actual_AI_PROXY_reviews=384,style_reviews=32,generator_forwards=4800,D0_forwards=96,training_updates=200,training_forwards=1600,backwards=1600),
      decision='See REPORT.md for scientific decision and paired evidence; no automatic promotion.',scientific_work_remaining=0,default_changed=False,commit_or_push=False,
      important_hashes={str(p):C.sha(p) for p in important})
    C.resources(O/'resources_completion.json');C.write(O/'COMPLETE.json',result);C.event('EXPERIMENT_COMPLETE',**result['counts'],default_changed=False)
    print(json.dumps({k:v for k,v in result.items() if k!='important_hashes'}))
if __name__=='__main__':main()