"""Transcription of root's actual image viewing; never automated image ratings."""
from . import run as S
C,OUT=S.C,S.OUT
NOTES={
'2d08a64e_n0':'五列均保留同一有腿椅身与背板。ALPHA两列底部轮廓有细微移动；侧截面C下方的分离短线仍在，未见大范围消缝。',
'2d08a64e_n1':'五列保持板式底座/背板外观；前截面下层长线及侧截面折线仍分离。本显示尺度无法确认局部合并的语义收益。',
'2d08a64e_n2':'框架与扶手整体外形一致。HOLE两列未见新增大挡板，但小桥在当前尺度下不可明确辨认；C周围原开口轮廓仍在。',
'2d08a64e_n3':'折叠板式底座与尖顶背板的整体形态不变，截面未显示明显新破损或大范围闭合；小范围差异不确定。',
'610b1621_n0':'五列箱式扶手椅近似相同；侧截面左侧竖直分离线、前侧开口保留。未看到本轮带来主体重组。',
'610b1621_n1':'四腿扶手外观一致；侧截面右上短孤立线和左下斜线仍可见，本轮未明显移除。',
'610b1621_n2':'C横向突出于生成座框的外观保留；ALPHA两列局部下层线略有移动，前截面的间隔与碎短线仍在。未见新增大裂口。',
'610b1621_n3':'带圆孔背板的造型保留；侧截面C上方短线和下层分离轮廓仍可见。孔洞形状并未被本轮填掉。',
'608b46b0_n0':'五列低背四腿外观一致，C边缘外伸仍明显。前截面C下方长条中间断口及侧截面下层间隙保留。',
'608b46b0_n1':'局部合并未造成明显新大面或椅身改变；前/侧截面的下层分离线仍在。不能凭当前全景认证微小拓扑接续。',
'608b46b0_n2':'箱式扶手形态保留。HOLE两列在侧截面右上出现新增短斜线/小三角轮廓，需局部图判断用途，不能自动称为视觉改善。',
'608b46b0_n3':'原先担心的座面下方分离短线/长线在各列均仍可见；ALPHA只见小位置变化，整体没有明显新崩坏，也没有普遍消除悬片。',
}

def main():
    rows=[]
    for q in S.O.OLD.requests():
        if q['split']!='DEV':continue
        rid=q['request_id'];p=OUT/'gallery'/f'{rid}.png'
        rows.append(dict(request_id=rid,split='DEV',reviewer='root',review_type='AI_PROXY_NONBLIND',
            actual_viewed=True,image_path=str(p),image_sha256=C.sha(p),stages='BASELINE/MIDPOINT_REPLAY/ALPHA/HOLE/ALPHA_HOLE',
            views='two exterior views and original fixed front/side sections',visible_new_major_damage='NOT_OBSERVED_AT_DISPLAY_SCALE',
            universal_gap_repair='NOT_OBSERVED',local_semantic_benefit='UNCERTAIN',notes=NOTES[rid],
            limitation='Display-scale finite slices; no topology or absence-of-intersection certification; no unique source GT ranking.'))
    C.csvwrite(OUT/'review_dev.csv',rows)
    return rows

if __name__=='__main__':main()
