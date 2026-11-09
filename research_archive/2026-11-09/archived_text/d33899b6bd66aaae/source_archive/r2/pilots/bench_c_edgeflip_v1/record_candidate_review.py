"""Record the root review after viewing the four f495 comparison pages."""
from .common import *

def main():
    rid='f495bd67af8d9f0cadd1cf65e8dc4de__SEAT_SURFACE_1'
    records=[]
    for noise in (0,1):
        for stage in ('RAW','CLEAN_FINAL_POST'):
            path=OUT/'candidate_evaluation/gallery'/f'{stage}__ORIGINAL__{rid}__n{noise}'/'comparison.png'
            assert path.is_file()
            for model in ('EXPOSURE','PLAIN200','FLIP200'):
                if noise==0:
                    text='两视图均保留宽端板支撑与长条主体。front/side截面显示灰色水平顶层和下折层位于蓝C上方；蓝C没有成为可确认的顶层座面。三模型关系基本相同。'
                    adoption='IMPLAUSIBLE'
                    if stage!='RAW':text+=' 后处理后顶层/端板交界仍有可见开口，右端及内层可见短线；仅凭截面短线不能认证独立浮片。两候选没有清楚修好该开口。'
                else:
                    text='三模型均有长条板条座体和两端宽支撑。side截面有密集细长条截线跨过蓝C高度，C被灰色条带覆盖，合理采用关系不确定；不是以唯一来源款式判失败。'
                    adoption='UNCERTAIN'
                    if stage!='RAW':text+=' 后处理后板条截线更短且部分倾斜/不闭合，但图上不能区分真实独立碎片与有限切面的开口。未看到FLIP200相对PLAIN200明确减少碎裂。'
                records.append(dict(cell_id=f'{model}__ORIGINAL__{rid}__n{noise}',stage=stage,label=model,
                    representation='ORIGINAL',request_id=f'{rid}__n{noise}',actually_viewed=True,
                    source_image=str(path),source_image_sha256=sha(path),reviewer='AI_PROXY',review_agent='/root',blind=False,
                    adoption=adoption,body='REASONABLE',support='PRESENT',fragment_evidence='UNCERTAIN' if stage!='RAW' else 'NOT_CLEAR',
                    observation=text,comparison='无明确FLIP200独有的碎面修复或采用提升。',
                    scope='实际双视图及同位置front/side截面；非人类认证；截面端点不自动认作拓扑孤岛'))
    write(OUT/'visual_review/candidates_f495.json',dict(status='COMPLETE',records=records,reviewed_pages=4,reviewed_cells=12))

if __name__=='__main__':main()
