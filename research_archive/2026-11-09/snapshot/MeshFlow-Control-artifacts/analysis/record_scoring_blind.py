import json, hashlib, datetime
from pathlib import Path
OUT=Path('/mnt/e/MeshFlow-Control-artifacts/analysis')
ROOT=Path('/mnt/e/MeshFlow-Control-artifacts')
OUT.mkdir(exist_ok=True)
uids={'c':'c1b8c2f0c4f3cc43aff7c08b060f5ed6','f':'f59627a077dcc2566c63b8b48495261a'}
labels=['START','A500','B500','C500','A1000','B1000','C1000']
notes={
('c',0):[
'整体为清楚的直背四腿椅；上背橙色斜片与蓝色C重叠遮挡，侧面座面上方有一条细水平突片；线框繁杂。',
'直背四腿椅完整；START的水平突片不明显，上背仍有斜折片，C下方形成开口。',
'直背四腿椅完整；C下方开口较大，上背仍有斜片，未见主体缺失。',
'整体椅形完整；上背同类斜折片仍在，与A/B的明显结构差别很小。',
'直背四腿椅仍清楚；上背折片保留，未见新大面积主体坍塌。',
'整体完整，背部开口较清楚；上背局部折片保留。',
'整体完整，与A/B1000轮廓近似；仍有上背局部折片，没有清楚独有的整体改善。'],
('c',1):[
'清楚的椅子，但侧面座背交界有向上尖三角，座面下有长悬垂三角片，腿角不齐。',
'主体仍完整；START的大尖片和大悬垂片减小，仍有细片及不规则渐尖的下支撑。',
'主体完整；座背侧面较START简洁，下方仍有细长渐尖三角。',
'主体完整；START大尖片明显减弱，仍有细水平片和不齐下支撑。',
'椅形保留；下背有三角开口，侧面座面上方有两条向上细三角鳍片。',
'椅形保留；下背三角开口和侧面两条向上鳍片仍明显，未见相对A1000的大改善。',
'椅形保留；下背开口及侧面上翘尖片与A/B1000同类，局部缺陷没有消除。'],
('c',4):[
'高脚凳式椅形完整，长腿和下横撑清楚；腿尖偏细且不齐，可能有细游离线。',
'高脚凳主体完整；座面见小三角白隙，腿尖略不齐。',
'高脚凳主体完整；座面小楔形白隙及腿尖局部不齐仍在。',
'高脚凳主体完整；座面相对均匀，但不能从图确认没有细缝。',
'主体完整，但下横撑区域有明显长斜三角薄帆/细片，多视角可见。',
'主体完整，下横撑长薄帆较A1000不显著；局部仍有细渐尖支撑。',
'主体完整，下横撑区域比A1000简洁；背座交界仍略不齐。'],
('c',5):[
'四腿直背椅清楚；左上背有小角片/条片，座面边缘不齐，没有明显大碎裂。',
'四腿椅清楚；背座角略规整，上背条片仍在。',
'四腿椅清楚；与A500接近，上背条片和局部座边不齐仍在。',
'四腿椅清楚；与A/B500接近，未见大主体缺失。',
'四腿椅清楚；主要是小比例/三角化变化，无明显新大碎裂。',
'四腿椅清楚；上背局部角片仍在，没有可明确识别的整体改善或大退化。',
'四腿椅清楚；与A/B1000非常接近，未见明显新大缺陷。'],
('f',0):[
'高背四腿椅清楚；橙色座面遮挡不少蓝C，侧面似有叠层座面，是否精确重复无法由像素确定。',
'条栅背椅清楚，C较可见；侧面背部向前有细小尖片，横向规则开口可属于款式。',
'条栅背椅清楚；侧面仍有细片，C从座面区域可见，未见主体坍塌。',
'条栅背椅清楚；与A/B500接近，背座侧面仍有局部细片。',
'变为圆拱高背扶手椅；款式变化本身不判错，但座/扶手处有明显长三角薄片和多条上翘鳍片。',
'圆拱扶手椅仍可辨识；薄片较A1000部分减少，但扶手处仍有突兀三角面。',
'圆拱扶手椅可辨识；扶手/座周仍有三角薄片，整体变化与A/B同向，不能单归因C分支。'],
('f',1):[
'梯背扶手椅清楚；侧面座/扶手后有上翘三角鳍片，座周横杆较杂。',
'椅形保留，上翘大片减小；下侧框有锯齿状悬垂三角片。',
'椅形保留；腿间/座下侧面出现明显大三角填充片，支撑较粗且不规则。',
'椅形保留；下侧框有数条悬垂尖三角，比B500大填充片弱但缺陷仍在。',
'椅形保留；侧面座下有长细三角刀片，背部横条折叠不齐并有小向前尖片。',
'椅形保留；座下长细悬垂片和背部折条仍在，与A1000相近。',
'椅形仍可辨识，但背横条更密集交错/填充且参差，扶手附近多条上翘片，侧面多条悬丝；比A/B1000局部更紊乱。'],
('f',4):[
'圆拱背椅清楚；背中大斜三角及一侧大开口，座下有细游离线；开口本身也可能是款式。',
'变为实背圆拱扶手椅；腿间有细垂线/薄片，扶手座周层叠角面。',
'圆拱扶手椅完整；腿间细垂片及扶手层叠角面仍在，接近A500。',
'圆拱扶手椅完整；同类腿间细垂片和座周角面仍在。',
'圆拱扶手椅完整；背/扶手折面密，侧面扶手上有向前上翘的细片。',
'圆拱扶手椅完整；侧面比A1000略简洁，但大角形扶手面仍在。',
'变为简洁圆拱无扶手椅；主体相对A/B1000更清楚，蓝C较可见；侧面仍有极细悬线，去掉扶手本身属于款式差异。'],
('f',5):[
'圆拱扶手椅可辨；背中大斜三角遮挡并伴大缺口，侧面向前尖片及座下长薄片明显，腿边有小碎片。',
'圆拱实背椅完整，C座面清楚；侧面从背部伸出悬空楔形条片，另有极细座下垂片。',
'圆拱实背椅完整；侧面仍有悬空楔片及细垂片，接近A500。',
'圆拱实背椅完整；侧面悬空条片仍有，整体接近A/B500。',
'圆拱实背椅完整；500阶段向前悬空楔片已不明显，拱顶仍有不规则孔/折片。',
'圆拱实背椅完整；侧面较500简洁，拱顶仍有不齐开口，未见大主体缺失。',
'圆拱实背椅完整；侧面较500简洁，拱顶仍有不齐开口，与A/B1000差别小。']
}
rows=[]
for (short,n),observations in notes.items():
    uid=uids[short]; p=ROOT/'gallery'/'blind'/f'{uid}_C1__n{n}.png'
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    for label,note in zip(labels,observations):
        c='蓝C位于上背，橙色主体与其位置相容；局部遮挡/邻接不能证明精确连接。' if short=='c' else '蓝C位于座面，生成主体围绕该位置；遮挡和角面邻近不能证明精确连接或排除重复覆盖。'
        style='直背椅或高脚凳的比例/开口差异不自动判为破碎。' if short=='c' else '有无扶手、圆拱/条栅/实背及粗细腿是款式差异；尖片、悬片与不规则折面另记局部缺陷。'
        rows.append(dict(request_id=f'P1__{uid}_C1__n{n}',parent_id=uid,candidate=n,label=label,review_type='AI_VISUAL_REVIEW',C_compatibility=c,overall_recognizability_coherence='主体仍可辨识为椅子；具体协调性和局部损坏见观察。',local_defects=note,style_vs_fragmentation=style,certainty='确定：上述可见轮廓/薄片；可能：游离或重叠；无法判断：精确相交、拓扑连通及水密。',blind_page='E:/MeshFlow-Control-artifacts/gallery/blind/'+p.name,blind_sha256=sha,view_scope=['front_3q','back_3q','front','side','wire'],source_reference_seen=False,human_usability='NOT_RUN'))
doc=dict(schema='meshflow_control_ai_visual_review_v1',status='BLIND_COMPLETE_REFERENCE_NOT_VIEWED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_count=2,request_count=8,output_count=len(rows),order='All 8 blind pages were viewed before any reference page.',rows=rows)
assert len(rows)==56
p=OUT/'visual_review_scoring_blind.json';p.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'path':str(p),'rows':len(rows),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
