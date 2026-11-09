"""Transcribe root's actual 24 DEV comparison-page observations; no inferred labels."""
from r2.pilots.seam_abc_v1 import eval_common as C
OUT=C.ROOT/'experiments/r2_seam_followup_v1'

NOTES={
'2d08a64e_n0':'三列均保留躺椅形状、靠背与细框支撑。新布局仍有蓝C下方的灰层和前缘间隔；没有看见整段接缝封闭。POST细杆很薄，不能从小图认证无局部损伤。',
'2d08a64e_n1':'三列均为高背、板式支撑形状。新布局与旧B外观接近；POST正面截面底部横线较连续，但侧剖面蓝C与下层仍分开，不将其记为接入主体。',
'2d08a64e_n2':'LONG下半部明显散裂成横条/板片；旧B和新布局都为较连贯的躺椅细框，因此新布局相对LONG较整齐，这一收益旧B已有。新布局POST细框比旧B更细，完整保留程度记不确定；C下方的层间距仍在。',
'2d08a64e_n3':'拱顶靠背及弯折底座的大形均保留。RAW中LONG背板斜缝较大，新布局较小；旧B斜缝位置不同且新布局中央缝更显眼。POST三列背板均较封合，底座与座下大间隔仍存在。',
'610b1621_n0':'LONG是较高背扶手椅，旧B/新布局是低背箱式沙发形，款式变化本身不判坏。三列有主体和支撑，新布局与旧B近似；靠背向蓝C内侧伸入，不能认证用途或接口改善。',
'610b1621_n1':'三列均有四腿和短背，蓝板明显伸出较窄主体。新布局保留旧B的大形，但背侧新增/加重一张斜向大片，旧B该处较平。POST仍有蓝板下方独立灰层，未见全面接合。',
'610b1621_n2':'四腿和竖条靠背都在。新布局相对LONG及旧B新增背侧大斜三角片，另一视角靠背条间也更杂；RAW/POST均可见，记折片代价。不能把该片伸向C当成合法用途改善。',
'610b1621_n3':'三列均保留圆孔靠背和粗腿；圆孔按可解释款式处理，不标为裂口。新布局没有明显宏观修复或主体损失，蓝C周边与灰色内部构件关系仍不能仅凭图片确认。',
'608b46b0_n0':'三列椅背和四腿均保留。新布局减少旧B蓝C边上的部分显眼灰条，但背顶细碎孔隙仍有；剖面灰层仍位于C下方，不能认证真实接续。',
'608b46b0_n1':'旧B背顶有明显不规则大孔，新布局将其缩小到与LONG相近的细碎孔隙，RAW/POST均有可见改善。主体仍在，蓝座面用途关系近似；没有看到能认证的整圈接缝修复。',
'608b46b0_n2':'新布局与LONG均为高侧板箱形扶手椅，旧B更矮；款式不作唯一来源排名。主要板件均在，新布局与LONG很接近，蓝C周边的遮挡使局部接口判断不确定。',
'608b46b0_n3':'三列均保留背板/四腿，座面附近有长灰条或细小穿出片，背顶有细孔。新布局外观更接近LONG，未见明确消除这些缺陷；局部接续仍不确定。',
}

def main():
    rows=[]
    for request,note in NOTES.items():
        for stage in ('RAW','POST'):
            pid=request+'_'+stage;p=OUT/'gallery/layout'/(pid+'.png')
            assert p.exists()
            for base in ('LONG','B_POOL47'):
                whole='Y';use='Y';crack='SAME';fragment='SAME';join='UNCERTAIN'
                if request.startswith('610'):use='U'
                if request=='608b46b0_n2':use='U'
                if request=='2d08a64e_n2':
                    if base=='LONG':fragment='IMPROVED'
                    elif stage=='POST':whole='U'
                if request=='2d08a64e_n3' and stage=='RAW':crack='IMPROVED' if base=='LONG' else 'WORSE'
                if request=='610b1621_n1' and base=='B_POOL47':fragment='WORSE'
                if request=='610b1621_n2':fragment='WORSE'
                if request=='608b46b0_n1' and base=='B_POOL47':crack='IMPROVED'
                rows.append(dict(page_id=pid,request_id=request,stage=stage,split='DEV',baseline=base,candidate='B_LONG_LAYOUT',
                    reviewer='AI_PROXY_NONBLIND_ROOT',review_status='ACTUALLY_VIEWED',whole_body_preserved_vs_base=whole,
                    C_use_preserved_vs_base=use,visible_crack_change=crack,fragment_change=fragment,visible_join_change=join,
                    notes=note,image_path=str(p),sha=C.sha(p),
                    scope='Actual two depth views and finite front/side sections. Relative preservation is not absolute usability; images do not certify index connectivity.'))
    assert len(rows)==48
    C.csvwrite(OUT/'reviews/layout_dev.csv',rows)
    C.write(OUT/'reviews/layout_dev_receipt.json',dict(status='COMPLETE',actually_viewed_pages=24,paired_rows=48,
        review_type='NONBLIND_AI_PROXY',labels_from_metrics=False,new_model_calls=0,csv_sha256=C.sha(OUT/'reviews/layout_dev.csv')))

if __name__=='__main__':main()
