"""Observed CLEAN 73/n1 tail, separately from scorer-derived quantities."""
from .evaluation import C,OUT

def run():
    rid='73b369b9_n1';arm='CLEAN_EDGE_COH'
    images=[OUT/'gallery/POST'/(rid+'.png'),OUT/'sections'/arm/(rid+'.png')]
    scores={s:C.read(OUT/'evaluation/scores'/arm/rid/s/'score.json') for s in ('RAW','POST')}
    result=dict(reviewer='AI_PROXY',blind=False,request_id=rid,label=arm,
        status='ACTUALLY_VIEWED_TWO_REGISTERED_IMAGES',
        image_hashes={str(p):C.sha(p) for p in images},
        visual_observation='CLEAN改为侧板支撑形体；同位置front截面有紧贴蓝C下方的灰色薄层，side中段也有近乎平行的下层线。RAW阶段已经出现这种近层结构，D0/恢复/POST后仍可见。背板与支撑的整体外观可辨认，但外观规整不等于冲突覆盖低。',
        localization_limit='两条有限截面和两个外观角度不能把全部冲突查询点逐一定位到面，因此不由图片单独断言47.31%的覆盖全部来自某一张面，也不把款式变化本身判为失败。',
        scorer_observations={s:{k:scores[s]['metrics'][k] for k in ('ConflictCov_C','ConflictCov_C_dup','ConflictCov_C_cross','FF_illegal_affected_free_fraction')} for s in scores},
        interpretation='高I坏尾已在RAW存在；POST仍有47.31%覆盖，dup分量45.12%。这是实际评分的材料代价，不能用较薄/规整截面或较低FF掩盖。上述分量并非相互排斥，不能直接相加。',
        new_model_calls=0,new_score_calls=0)
    p=OUT/'visual/CLEAN_73n1_quality_tail_actual_review.json';C.write(p,result);return p

if __name__=='__main__':print(run(),flush=True)
