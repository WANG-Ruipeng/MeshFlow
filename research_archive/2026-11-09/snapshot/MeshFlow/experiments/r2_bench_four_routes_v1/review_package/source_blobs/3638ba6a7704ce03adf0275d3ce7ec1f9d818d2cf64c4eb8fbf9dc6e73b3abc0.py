"""Read-only actual evaluation accounting, independent of frozen science code."""
import json
from . import common as C
from . import evaluation as E
from .closeout import require,verify

def main():
    raw=C.read(C.OUT/'evaluation/EXPOSURE/raw_complete_full.json')
    post=C.read(C.OUT/'evaluation/EXPOSURE/post_complete_full.json')
    require(raw['status']=='COMPLETE' and raw['total_new_generator_forwards']==2300
            and raw['total_new_head_forwards']==0 and raw['outputs']==46,'RAW budget differs')
    require(post['status']=='COMPLETE' and post['new_D0_forwards']==post['new_GEO_calls']==46,'D0 budget differs')
    E.verify_raw('EXPOSURE'); verify(raw['files']); verify(post['files'])
    score=C.read(C.OUT/'results/completion.json'); verify(score['files']); verify(score['source_rows'])
    require((score['rows'],score['new_score_rows'],score['reused_score_rows'])==(276,138,138),'Score rows differ')
    require(len(score['metric_fields'])==177,'Full score fields differ')
    verify(C.read(C.OUT/'audits/reference_cache_audit.json')['files'])
    visual=C.read(C.OUT/'visual_review/summary_v1/completion.json')
    verify(visual['files']); verify(visual['sources'])
    require(visual['status']=='COMPLETE' and visual['actual_observations']==276,'Actual visual coverage incomplete')
    paths=[C.OUT/'evaluation/EXPOSURE/raw_complete_full.json',C.OUT/'evaluation/EXPOSURE/post_complete_full.json',
           C.OUT/'results/completion.json',C.OUT/'visual_review/summary_v1/completion.json']
    result=dict(status='PASS',new_generator_forwards=2300,new_head_forwards=0,new_D0_forwards=46,
                new_RAW=46,new_GEO_calls=46,new_score_rows=138,reused_score_rows=138,all_fields=177,
                actual_visual_observations=276,model_calls=0,scoring_calls=0,
                inputs={str(p):C.sha(p) for p in paths},code_sha256=C.sha(__file__))
    C.write_new(C.OUT/'closeout/evaluation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
