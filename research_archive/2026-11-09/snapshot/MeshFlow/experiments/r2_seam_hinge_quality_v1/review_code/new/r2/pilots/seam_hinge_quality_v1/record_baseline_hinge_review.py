"""Manual AI_PROXY observations recorded AFTER actual viewing of all28 PNGs.

Observations below were supplied from images, never generated from metric rows.
"""
from .evaluation import C,OUT,requests

NOTES={
'608b46b0_n0':('座下front水平截线中间开口、side下层水平线断点在三列都保留；两外观视角未见明显变化。','UNCHANGED_VISIBLE'),
'608b46b0_n1':('front座下左侧缺口和side座下几段水平短线仍在；外部椅腿/背板保持，未见悬片消失。','UNCHANGED_VISIBLE'),
'608b46b0_n2':('箱式/扶手形外壳保持；front两侧内侧短斜线仍可见，side蓝C与左右上翘轮廓位置无明显改变。','UNCHANGED_VISIBLE'),
'608b46b0_n3':('front蓝C下左部的短线与长线间隙仍在，side下层水平线断口仍在；外观未见明显修复。','UNCHANGED_VISIBLE'),
'610b1621_n0':('宽扶手箱式外观保持，front两侧内层竖线及side背部附近双线保持；本尺度未见明确连接修复。','UNCHANGED_VISIBLE'),
'610b1621_n1':('框架椅外观保持；side靠背/座面附近的斜线及前端短折角仍在；front侧端分离竖线仍在。','UNCHANGED_VISIBLE'),
'610b1621_n2':('细腿与小背板保持；front/side蓝C下方较远的U形/托盘状截线仍存在，未因HINGE靠近C。','UNCHANGED_VISIBLE'),
'610b1621_n3':('带洞靠背与四腿外观保持；front座下左右短L线、side中间两条短折线仍在。','UNCHANGED_VISIBLE'),
'2d08a64e_n0':('外部背板/滑橇框架保持；side蓝C下方孤立短线和下层截线仍可见。025改变front下层一段横线为斜线、side下层线局部倾斜；不能据此确认更好。','LOCAL_SHAPE_CHANGE_025'),
'2d08a64e_n1':('板式背和悬臂底架保持；side蓝C下的微小孤立截线与下层框架断点仍可见，未见明显改动。','UNCHANGED_VISIBLE'),
'2d08a64e_n2':('扶手框架与蓝座面保持；side座下长水平线和靠背附近短截线位置大体不变，未见明显新增大破损。','UNCHANGED_VISIBLE'),
'2d08a64e_n3':('悬臂式下部大结构和背板保持；side蓝C下前端斜线/下支架仍分开显示。图片不足以把款式开口判为缺损。','UNCHANGED_VISIBLE'),
'3f741759_n0':('背部竖条和四腿保持；front座下左侧截面缺口仍在，side上方短横线保留；不能把短横线直接叫漂浮。','UNCHANGED_VISIBLE'),
'3f741759_n1':('外观保持；front右侧竖线与蓝C间空隙仍在，side底线的小阶梯保持。','UNCHANGED_VISIBLE'),
'3f741759_n2':('背部斜条/薄条仍在；side上方短横线在三列保留，front座下左侧缺口仍在。','UNCHANGED_VISIBLE'),
'3f741759_n3':('外观保持；front右端缺口及side下层阶梯附近短断点仍在。','UNCHANGED_VISIBLE'),
'73b369b9_n0':('板式悬臂外观保持；front右侧座下缺口、side前后座下截线断开仍在，未见明显消除。','UNCHANGED_VISIBLE'),
'73b369b9_n1':('外观保持；side前沿与下部短线局部角度略变但缺口仍可见，front座下轮廓近似不变。','MINOR_LOCAL_SHAPE_CHANGE'),
'73b369b9_n2':('背板上部斜向开口仍在；front下层中间细缺口及side下层断点保留，未见大尺度质量改变。','UNCHANGED_VISIBLE'),
'73b369b9_n3':('多边背板与悬臂底座保持；front/side下层轮廓右端小间隔仍可见。','UNCHANGED_VISIBLE'),
'85b16941_n0':('四条细腿和背板保持；front/side座下轮廓连续程度本来较好，HINGE未见明显视觉增量。','UNCHANGED_VISIBLE'),
'85b16941_n1':('外观保持；front左侧竖向小缺口仍在，side轮廓近似不变；未见新的大片裂口。','UNCHANGED_VISIBLE'),
'85b16941_n2':('背板/细腿保持；front右上角微小间隔、side左上接近C处的间隔未见明确消除。','UNCHANGED_VISIBLE'),
'85b16941_n3':('背板顶端小洞仍在；front下层右侧断点及side上方背板截线保留。','UNCHANGED_VISIBLE'),
'8c9b2023_n0':('外观保持；front座下底线右侧细间隙在两个HINGE图中看起来闭合；side右侧大开口仍在。','LOCAL_SECTION_GAP_CLOSED'),
'8c9b2023_n1':('梯状靠背/四腿保持；side上方U形背条截线和座下左侧开口仍在，无明显外观变化。','UNCHANGED_VISIBLE'),
'8c9b2023_n2':('front座下左侧断点仍在；025的side上方背条矩形截线底部变为轻微折线，外部轮廓未见大改变。','LOCAL_SHAPE_CHANGE_025'),
'8c9b2023_n3':('front座下底线右侧间隙在015仍可见、025看起来明显缩小/闭合；外观座下侧缘白缝及side大开口仍在。','LOCAL_SECTION_GAP_CLOSED_025'),
}

def main():
    qq={q['request_id']:q for q in requests()};assert set(NOTES)==set(qq)
    pages=[];variants=[]
    for rid,(note,change) in NOTES.items():
        p=OUT/'gallery/HINGE_PLAIN_S2000'/(rid+'.png');receipt=C.read(p.with_suffix('.json'));assert C.sha(p)==receipt['image_sha256']
        common=dict(request_id=rid,split=qq[rid]['split'],parent_id=qq[rid]['parent_id'],noise_index=qq[rid]['noise_index'],
            reviewer='AI_PROXY_NONBLIND',reviewer_agent='/root/eval_sections',actually_viewed=True,image_path=str(p),image_sha256=C.sha(p),
            viewed_panels='POST/HINGE015/HINGE025: two exterior views + exact front/side central finite sections',
            observation=note,uncertainty='Finite sections miss off-plane connections; small local differences below this display scale may be invisible. No manifold/semantic guarantee from image alone.')
        pages.append(dict(**common,visible_change=change,new_severe_visual_damage='NOT_OBSERVED_IN_THESE_VIEWS'))
        for variant in ('HINGE015','HINGE025'):
            local=change
            if change.endswith('_025') and variant=='HINGE015':local='UNCHANGED_VISIBLE'
            variants.append(dict(**common,label='PLAIN_S2000_'+variant,baseline='PLAIN_S2000_POST',
                visible_change=local,new_severe_visual_damage='NOT_OBSERVED_IN_THESE_VIEWS',
                overall_shape='PRESERVED_IN_DISPLAYED_VIEWS',broad_fragment_removal='NOT_OBSERVED',
                local_section_closure='VISIBLE' if local.startswith('LOCAL_SECTION_GAP_CLOSED') else 'NOT_CONFIRMED'))
    dest=OUT/'visual';C.csvwrite(dest/'hinge_baseline_pages.csv',pages);C.csvwrite(dest/'hinge_baseline_variants.csv',variants)
    C.write(dest/'hinge_baseline_completion.json',dict(status='ACTUAL_REVIEW_COMPLETE',viewed_pages=28,variant_reviews=56,
        reviewer='AI_PROXY_NONBLIND',no_metric_based_labels=True,files={str(dest/n):C.sha(dest/n) for n in ('hinge_baseline_pages.csv','hinge_baseline_variants.csv')},
        finding='Most views show little perceptible change. Local front-section gap closure is visible for8cn0 both thresholds and8cn3 mainly025, not broad fragment disappearance.025alsochanges2dn0underside slope and8cn2backbar sectionshape. No severe new exterior damage observed atthisscale.',
        limitation='No claim of zero hidden damage or universal joint success; exact topology and original quality scores remain separate.'))
    print('ACTUAL_REVIEW_COMPLETE28_PAGES56_VARIANTS',flush=True)

if __name__=='__main__':main()
