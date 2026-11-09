"""Four actually viewed BOUND015 cases selected for material metric costs."""
from .evaluation import C,OUT

NOTES={
'2d08a64e_n0':'CLEAN外观仍是低背/弯曲下壳；front左侧在C与下层之间仍有明显空隙，side下壳呈多段折线并有近层。不能把41.23%的共边可达解释为整圈接缝已经封好。SMOOTH的side多条斜线交叠也仍清楚。',
'73b369b9_n1':'CLEAN薄层仍紧贴C下方，整体形体保持；SMOOTH front下方有贴近蓝C的灰色平行线与局部短斜线。图中没有整体崩坏，但不能凭这两条中央截面定位每一对新增FF。',
'85b16941_n1':'三模型外部板背/细腿仍完整；SMOOTH front两侧及side右侧仍有间隔，底框较规整。没有从图中观察到与新增9对FF一一对应的九处新破洞，指标与图审分别保留。',
'610b1621_n1':'SMOOTH主体仍可辨为扶手椅，front/side下层框架与蓝C保持明显高度间隔，side后背附近多条线靠近。CLEAN仍是过宽C加窄小主体。不能由外观规整否认新增FF，也不能把截面短线直接叫悬片。',
}

def run():
    rows=[]
    for rid,note in NOTES.items():
        p=OUT/'gallery/BOUND015'/(rid+'.png');receipt=C.read(p.with_suffix('.json'))
        assert C.sha(p)==receipt['image_sha256']
        rows.append(dict(request_id=rid,stage='BOUND015',reviewer='AI_PROXY',blind=False,
                         observed=True,image_path=str(p),image_sha256=C.sha(p),observation=note))
    C.write(OUT/'visual/BOUND015_four_tail_cases_actual_review.json',dict(
        status='ACTUALLY_VIEWED_FOUR_REGISTERED_PAGES',rows=rows,
        scope='Four explicitly selected cost cases only; not a complete 28-request BOUND visual review',
        new_model_calls=0,new_score_calls=0))
    return len(rows)

if __name__=='__main__':print(run(),flush=True)
