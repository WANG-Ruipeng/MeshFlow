"""Record the root review after actual viewing of all eight 3f pages; no inferred scores."""
import csv
from . import eval_common as C

# Each tuple was entered from the displayed page after inspection of its four columns.
# Fields: noise, stage, model, fragments/layers, cracks, concrete observation, change vs LONG.
OBS = [
(0,'RAW','LONG','NO','NO','蓝色座面、圆弧靠背、竖条和四腿均可辨；侧裙可见，座下剖线略断续。','REFERENCE'),
(0,'RAW','A_COUPLE','NO','UNCERTAIN','主体与座面齐全；座下前剖和侧剖的底线缺段比LONG更长，整体外视未见主体缺失。','外观相近，底部剖线的断段增加，不能直接判为外壳破损。'),
(0,'RAW','B_POOL47','NO','UNCERTAIN','蓝C处在座面；靠背与腿完整，座箱剖面更深，前侧蓝灰交界有细白边。','座下盒状轮廓较LONG更深；不以来源款式一致为要求。'),
(0,'RAW','C_SHARED_TIP','NO','YES','主体仍完整，第一视角左侧座裙有白色开口，另一视角前缘也有细缝；底部剖线明显缺段。','座裙局部开口较LONG更明显。'),
(0,'POST','LONG','NO','NO','蓝色座面、背条和四腿保留；侧剖盒状下边连续，未见明显散片。','REFERENCE'),
(0,'POST','A_COUPLE','NO','UNCERTAIN','主体和C用途保留；底部前剖与侧剖仍有较大缺段，外视未见整块支撑消失。','与LONG外形相近，底部断段仍更多。'),
(0,'POST','B_POOL47','NO','UNCERTAIN','完整椅身与蓝座面保留；座箱更深、底线平直，蓝灰交界细节较小。','座箱尺寸改变，没有清楚的整体视觉胜出。'),
(0,'POST','C_SHARED_TIP','NO','YES','侧裙左上缘的白色开口仍可见；座面、靠背和四腿没有整块缺失。','D0后局部开口仍比LONG明显。'),
(1,'RAW','LONG','YES','UNCERTAIN','座面和主体齐全；一条内侧腿末端有尖薄突出片，背条略不齐。','REFERENCE'),
(1,'RAW','A_COUPLE','NO','UNCERTAIN','蓝C仍作座面；四腿和靠背齐全，没有LONG内侧腿末端同样明显的尖片。','腿末端较平整，座裙细缝仍难确认。'),
(1,'RAW','B_POOL47','NO','UNCERTAIN','座箱、靠背和四腿完整；背条略有不齐，座面前缘细白线可见。','与LONG总体接近，座箱略深，未见同样明显的腿末尖片。'),
(1,'RAW','C_SHARED_TIP','YES','YES','靠背弧板有三角白洞，内部竖条倾斜并呈尖薄形；蓝色座面和腿仍存在。','靠背的破口和尖薄竖条比LONG明显。'),
(1,'POST','LONG','NO','NO','主体与C座面用途清楚，内腿末端的尖薄突出不再明显；剖面底线连续。','REFERENCE'),
(1,'POST','A_COUPLE','NO','NO','完整靠背、座箱和四腿保留；两剖面形态接近LONG，未见明显额外薄片。','与LONG非常接近，无明确整体胜出。'),
(1,'POST','B_POOL47','NO','UNCERTAIN','椅身完整、C位置合理；座箱底线较低，前缘蓝灰接近但细节有限。','主要是座箱轮廓改变，未见明显支撑缺失。'),
(1,'POST','C_SHARED_TIP','YES','YES','弧形靠背的三角白洞仍在，内部有尖斜竖条；主体、座面和四腿仍可辨。','靠背局部破损持续，D0未消除。'),
(2,'RAW','LONG','YES','UNCERTAIN','蓝C作为座面，主体完整；靠背中间数根窄条扭斜，侧剖底部有短碎线。','REFERENCE'),
(2,'RAW','A_COUPLE','NO','UNCERTAIN','靠背竖条较直且规整，座面和四腿保留；底部侧剖仍有断段。','背条比LONG平整，座下是否有细裂口仍不确定。'),
(2,'RAW','B_POOL47','UNCERTAIN','UNCERTAIN','主体完整，背条比LONG直；左侧裙蓝灰交界有细白线，底部有少量短段。','背条更规整，侧裙小缝没有明确消失。'),
(2,'RAW','C_SHARED_TIP','YES','YES','弧形靠背出现多条白色裂口，内部竖条尖斜、错位；蓝座面及支撑保留。','靠背裂口和尖条明显多于LONG。'),
(2,'POST','LONG','NO','UNCERTAIN','D0后背条更直，主体和蓝C保留；座箱底线大体连续，但局部端部关系不明。','REFERENCE'),
(2,'POST','A_COUPLE','NO','NO','靠背、座面和四腿齐全，剖面下缘连续，未见明显额外薄片。','与LONG接近，底部轮廓更整齐。'),
(2,'POST','B_POOL47','NO','UNCERTAIN','主体保留，背条平直；侧裙左角细白线与局部空隙难分辨。','总体与LONG接近，座下外形有所变化。'),
(2,'POST','C_SHARED_TIP','YES','YES','靠背右上弧板有明显三角缺口，中间竖条仍斜尖；蓝C和腿保持完整轮廓。','靠背破损没有随D0消失，明显弱于LONG的完整背条。'),
(3,'RAW','LONG','NO','UNCERTAIN','座面、圆弧靠背和四腿齐全；底部侧剖存在小断段，外侧看不到大块缺失。','REFERENCE'),
(3,'RAW','A_COUPLE','NO','UNCERTAIN','C座面用途与主体保留；背条平直，侧剖底线仍有间隔。','与LONG非常接近，未见明确整体视觉变化。'),
(3,'RAW','B_POOL47','NO','UNCERTAIN','主体完整；座箱略深，内侧腿末端可见开放截口，是否有缺面不确定。','总体与LONG接近，局部腿端与座箱尺寸有变化。'),
(3,'RAW','C_SHARED_TIP','UNCERTAIN','YES','蓝座面、背条和四腿仍在，圆弧背板呈多条放射状白缝与开口。','靠背弧板比LONG更破碎，是否另有折层不能据图确定。'),
(3,'POST','LONG','NO','UNCERTAIN','主体和C保留，靠背规整；侧剖底部小间断仍存在。','REFERENCE'),
(3,'POST','A_COUPLE','NO','UNCERTAIN','主体完整、蓝C用途合理；整体与LONG近似，座下剖线的小断段仍在。','未见明确整体胜出或重大退化。'),
(3,'POST','B_POOL47','NO','UNCERTAIN','靠背、座面和四腿保留，内侧腿端可见局部开放截口；不据此认定支撑失效。','与LONG总体近似，底部细节有所改变。'),
(3,'POST','C_SHARED_TIP','UNCERTAIN','YES','弧形靠背上多条窄白裂口仍可辨，C及整体椅身保留。','靠背弧板裂口比LONG明显，D0后仍存在。'),
]

def main():
    root=C.OUT/'evaluation';dest=root/'reviews/3f.csv';dest.parent.mkdir(exist_ok=True)
    if dest.exists():raise RuntimeError('Actual review already recorded')
    fields=C.read(root/'review_schema.json')['csv_columns']
    requests={q['request_id']:q for q in C.read(C.OUT/'eval_manifest.json')['requests']}
    rows=[]
    for noise,stage,model,fragment,crack,observation,change in OBS:
        rid=f'3f741759_n{noise}';q=requests[rid];image=root/'gallery'/f'{rid}_{stage}.png'
        rows.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=noise,split='TRAIN',stage=stage,model=model,
            reviewer='AI_PROXY_ROOT',review_kind='AI_PROXY',review_status='ACTUALLY_VIEWED',blinding='NONBLIND_MODEL_LABELS_VISIBLE',
            C_adoption='YES',whole_body_complete='YES',support_missing='NO',fragments_or_folded_layers=fragment,
            visible_cracks=crack,visible_join='YES',visible_observations=observation,change_vs_LONG=change,
            uncertainty='蓝C可遮住近共面重层；中心剖面缺段不必然是外壳裂缝。可见接近不认证实际索引连接、无交或水密，细小缝隙受分辨率限制。',
            image_path=str(image.relative_to(C.ROOT)),image_sha256=C.sha(image)))
    assert len(rows)==32 and len({(r['request_id'],r['stage'],r['model']) for r in rows})==32
    with dest.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    notes='''# 3f实际图审（AI_PROXY_ROOT，非盲）

实际打开并检查n0至n3的RAW/POST八页，每页LONG/A/B/C四列，合计32行。未用几何评分填写视觉标签。

- 四组均保留了可辨认的椅身、支撑和蓝色座面用途；此处的YES不认证完整合法接缝。
- A的n2 RAW背条比LONG更规整，n1腿末尖片也不明显；整体形态与LONG接近。n0底部剖线缺段更长，保留其是否为裂口的不确定性。
- B整体与LONG接近，座箱常更深；少量侧裙细白线、腿端截口不够清楚，未将其强判成支撑缺失。
- C的n0座裙开口，以及n1/n2/n3的靠背白洞、裂口或尖斜背条，是更清楚的局部退化；POST没有全部消除。
- 没有把较大剖面底部空隙自动解释为错误，也没有据蓝C遮挡下看不见冲突而宣称低I。
'''
    (root/'reviews/3f.md').write_text(notes,encoding='utf-8')
    print('ACTUAL_REVIEW_SAVED',len(rows))

if __name__=='__main__':main()
