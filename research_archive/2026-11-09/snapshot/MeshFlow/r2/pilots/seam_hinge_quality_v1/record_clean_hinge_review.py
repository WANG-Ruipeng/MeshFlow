"""Actual observations of 28 CLEAN POST/015/025 comparison PNGs."""
from .evaluation import C,OUT,requests

NOTES={
'73b369b9_n0':'两档front底线中央小缝不再明显，side中间断点也较连续；side右内角间隔仍在。整体悬臂形体保持，2.5档未显示明确额外视觉收益。',
'73b369b9_n1':'两档与原POST外观和截面近似；C下方紧贴灰层、side端部间隔仍在。没有凭图看到高I坏尾被修复。',
'73b369b9_n2':'两档均保留靠背顶部大斜开口、side内层两端间隔；front盒形轮廓近似，没有明确整体改善。',
'73b369b9_n3':'两档与原POST形体近似；front右下与side左内角间隔保留，未见明显新增大破损，也未见明确视觉升级。',
'3f741759_n0':'两档均保留front两侧较大缺口及背部斜细片；side盒状底线和上方短线近似。没有明显修好主要缺口。',
'3f741759_n1':'三列外观与截面基本相同，front右端未连接到C的间隔保持；两档未显示清晰视觉差别。',
'3f741759_n2':'两档均保留front和side座下宽缺口，以及靠背顶部白缝；HINGE没有修复本轮CLEAN的明显局部表面损失。',
'3f741759_n3':'两档只见很小局部轮廓变化，front右上间隔仍在；外观主体保持，无明确质量升级。',
'85b16941_n0':'两档front右上到C的小间隔较接近闭合，但底部宽缺口仍在；side外框与整体板背细腿保持。局部改善不能覆盖原底部缺面。',
'85b16941_n1':'两档side左上转角比原POST更连贯；front左侧中段间隔仍在，外观相近。2.5档未见额外可辨收益。',
'85b16941_n2':'两档仍保留front左侧和side左上间隔，背顶小白口亦在；没有明确视觉变化。',
'85b16941_n3':'两档均保留front底线两端断点，side完整盒形轮廓近似，外观无明显变化。',
'8c9b2023_n0':'两档front底部少量很小断点有所合拢，但左侧大间隔及右侧断点保持；side盒形轮廓和横档背外观近似。',
'8c9b2023_n1':'两档front右下断点与side左边缺失仍在，上方方框短线也保持；没有明确视觉收益。',
'8c9b2023_n2':'两档front多个底部断点依旧，side盒形轮廓完整但与原POST近似；背横档薄斜条未消失。',
'8c9b2023_n3':'两档没有补回front底部中央大段缺失和side几乎整段底线缺失；外观整体仍是一把椅子，但截面退化持续。',
'2d08a64e_n0':'两档side下壳左部小断口较连贯，但front左侧大间隔、side右端缺口与多层弯曲线仍在。外观低背形体保持；不能据此称整圈接好。',
'2d08a64e_n1':'两档外观与原POST近似，front左下长段缺失与side C/下层框架分离仍在；未见明显改善。',
'2d08a64e_n2':'两档side下壳V形最低点附近局部线段变得较连续，front双层底线与side多层长斜线仍在；没有整体变干净的证据。',
'2d08a64e_n3':'两档均保留底座斜开口和C下楔形空隙；front/side在固定窗口内仍不闭合，外观没有明显改变。',
'610b1621_n0':'两档厚扶手板/高背形体及front/side长线近似，蓝C前缘外伸保持；未见清晰接续改善。',
'610b1621_n1':'两档仍为过宽蓝C加小背板、窄下框；front/side下方矩形与C明显分开，未见改变。',
'610b1621_n2':'两档保持竖条背、窄主体和宽C；front/side下方框与两侧斜片间距持续，没有明显修复。',
'610b1621_n3':'两档保持圆孔背板和四腿；两截面上内侧短L线与主体间隔仍在，没有明显改善。',
'608b46b0_n0':'两档均保留背底白色三角缺口与front/side座下间隔；外观和主要断口基本不变。',
'608b46b0_n1':'两档保留背顶孔缝、front左下间隔；side长底线近似，未观察到明显额外改善。',
'608b46b0_n2':'两档箱式形体、内部短斜线及两截面层次基本不变；未见明显新增破洞或明显收益。',
'608b46b0_n3':'两档都保留指定front下层断口；side中央下层线的间隔在绑定后更明显，不能把这两档称为所有截面均无损。没有据此声称网格新裂开，仍需看完整索引图；腿部和背部外观大体保持。',
}

def run():
    assert set(NOTES)=={q['request_id'] for q in requests()}
    rows=[];files={}
    for q in requests():
        rid=q['request_id'];p=OUT/'gallery/HINGE_CLEAN_EDGE_COH'/(rid+'.png')
        receipt=C.read(p.with_suffix('.json'));assert C.sha(p)==receipt['image_sha256'];files[str(p)]=C.sha(p)
        for variant in ('HINGE015','HINGE025'):
            rows.append(dict(label='CLEAN_EDGE_COH',request_id=rid,split=q['split'],variant=variant,
                comparator='ORIGINAL_POST',reviewer='AI_PROXY',blind=False,actually_viewed=True,
                image_path=str(p),image_sha256=C.sha(p),observation=NOTES[rid],
                inference_limit='Visual observations from two exteriors and fixed finite sections; no independent human rating, no certification of legal edges or absence of collisions.'))
    p=OUT/'visual/CLEAN_EDGE_COH_HINGE_actual_review.csv';C.csvwrite(p,rows)
    result=dict(status='ACTUAL_REVIEW_COMPLETE',label='CLEAN_EDGE_COH',reviewer='AI_PROXY',blind=False,
        actual_pages=28,variant_observations=56,requests=[q['request_id'] for q in requests()],
        files={str(p):C.sha(p)},image_hashes=files,model_calls=0,new_scoring=0,
        summary='A few small section gaps close, but wide gaps in 3f/n2 and 8c/n3 and DEV C/bulk separation remain. Guard success is not visual no-harm; 608/n3 side section gap remains/conspicuously opens. No claim of a uniform mesh quality upgrade.')
    C.write(OUT/'visual/CLEAN_EDGE_COH_HINGE_review_completion.json',result);return result

if __name__=='__main__':print(run()['status'],flush=True)
