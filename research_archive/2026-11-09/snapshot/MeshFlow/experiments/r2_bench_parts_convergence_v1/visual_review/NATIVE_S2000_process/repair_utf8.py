"""Restore reviewed Chinese observations after a Windows stdin encoding loss."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
before = root / 'encoding_repair' / 'before'
before.mkdir(parents=True, exist_ok=False)
for name in ['review.jsonl', 'raw_post_comparison.jsonl', 'review_schema.json', 'REVIEW.md', 'receipt.json']:
    shutil.copyfile(root / name, before / name)
rows = [json.loads(s) for s in (root/'review.jsonl').read_text(encoding='utf-8').splitlines()]
pairs = [json.loads(s) for s in (root/'raw_post_comparison.jsonl').read_text(encoding='utf-8').splitlines()]
notes = [
('靠背板',
 '两个表面视角都能辨认长凳与蓝色靠背。蓝板后方还有近乎平行的灰色背板；下部自由面呈弯曲兜状宽片，左端支撑不规整。侧截面可见蓝灰靠背位置偏离。',
 '下部截面比RAW更平顺，但平行灰背板与蓝C的错层仍在，整体大形基本未变。图像不足以判断弯曲下片是否为独立分量。',
 '局部平顺；未看到输入错层得到消除。'),
('座板＋端支撑组合',
 '蓝色输入包含大块座板和一个曲面端支撑；灰色自由面另形成高于蓝座板的长条片状表面。前、侧截面均可见两层高度差；另一端灰支撑短且不规整。支撑位置合理，但座面是否被直接采用不明确。',
 '灰色长条更平整，同时仍悬在蓝座板上方；条带间开口与较短的灰端支撑仍可见。没有观察到整体协调修复。',
 '部分表面平整；额外座面层与支撑不对称仍在。'),
('完整矩形座面',
 '灰色生成体已形成有靠背、座面和端部支撑的长凳。蓝色整块座面位于灰座面上方，两个截面都显示持续间隔；更像额外悬置层，没有作为可见灰座面的延续。',
 '灰体略整理，但蓝座面与下方灰座面之间的间隔保持；没有明显用途转化。小截面短线不能单独认定为漂浮面。',
 '宏观结构与蓝灰座面错层基本不变。'),
('U形端支撑',
 '蓝U形架位于长凳一端，灰座面沿其内侧伸向另一端，作为端支撑的相对位置合理。蓝架上梁高于灰座面。灰体另一端腿以及蓝架附近有不规整薄片和细长边，不能从图像确认共享连接。',
 '灰座面明显更平整；另一端灰腿从RAW可见的片状腿变成近线状支撑，蓝架内侧细碎线仍可见。输入用途合理不等于生成腿完整或拓扑接合。',
 '座面平顺改善，同时对端支撑显著变薄。'),
('一条实心腿',
 '蓝色腿上端靠近生成座面下侧，支撑角色在位置上合理，但表面视图中大部分蓝腿被灰色宽支撑面遮住。灰体有宽面板、多层侧片和扇状薄片；整体像带靠背的凳体，表面不够规整。',
 '蓝腿与座面位置关系保持；宽支撑面与扇状侧片仍在，下方横向长间隙更清楚。不能仅凭遮挡判定蓝腿已被替代或发生精确重叠。',
 '主体变化小，复杂片状侧面没有明显改善。'),
('一条完整座面板条',
 '灰色生成体形成完整四腿板条凳；蓝色狭长C在座面下方、腿部中段，视觉上更像横向拉杆而非上方座面板条。上表面与C有明显高度差。模型只接收几何，没有部件名，这也可能是不同用途解释，不能作为唯一来源款式失败。',
 '板条间开口更清楚，但仍保留完整凳体；蓝C没有移到灰座面层。规则板条缝不能自动算裂口，有限截面短段也不能自动算悬片。',
 '未见蓝C座面用途改善；整体凳体仍可辨。'),
('开放矩形靠背框',
 '灰色体是完整高背长凳，蓝色矩形背框位于一整块灰背板前方。侧截面显示明显前后错层，蓝框不像灰背板的连续组成部分；但仍处于靠背区域。',
 '高灰背板与前方蓝框关系基本不变，没有视觉上的接入改进。表面视图没有明显大量小碎片，但不能据此认证无断片。',
 '几乎不变；背框前后错层持续。'),
('后座板条＋端支撑侧面',
 '蓝色长条在灰座面后缘附近，蓝楔形侧片位于一端支撑处，功能位置大致合理。侧截面中蓝短段与灰座面端点仍有间隔，蓝端片旁还有灰腿，无法仅凭投影确认连为一体。',
 '灰靠背和座面更平整，但自由腿及支撑明显变成很细的条或线，蓝楔形端片仍突出；连接处间隔没有明确消失。',
 '表面平顺伴随支撑变细，不能记为全面视觉改善。'),
]
assert len(rows)==16 and len(pairs)==8
for i, (part, raw, post, comparison) in enumerate(notes):
    for j, text in enumerate([raw, post]):
        r=rows[2*i+j]
        assert r['stage']==['RAW','POST'][j]
        r['part_description']=part
        r['notes']=text
    pairs[i]['part_description']=part
    pairs[i]['RAW_to_POST']=comparison
for name, data in [('review.jsonl',rows),('raw_post_comparison.jsonl',pairs)]:
    (root/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
md=['# NATIVE_S2000 过程图审（AI_PROXY）','',
 '已实际逐张查看固定 process 面板的8张图页；每页同时含 RAW、POST，两种表面视角与固定位置的前/侧有限截面，共形成16条逐阶段记录。非盲 AI_PROXY，未作人工图审，未使用数值评分生成标签。此处只记录第2000步过程现象，不评价最终模型，也不改变预注册训练、数据或噪声。','',
 '整体上已经可以辨认长凳形状，但“主体可辨认”和“C自然成为组成部分”仍需区分。此次较明确的错层包括：459f90的完整座面在另一层灰座面上方；20222的蓝背框在一整块灰背板前方。122a的单条座面C被放在生成座面下方，更像腿间横条。该条输入只有几何，没有语义用途标签，因此这是构造角色不匹配的观察，不以唯一来源款式给开放生成判失败。','',
 'D0/GEO_C 后部分座面与靠背变平顺，但9fd5腿架请求和20222组合请求的灰色支撑明显变薄。不能把这种变化笼统记成后处理改善，也不能从截面里的短线段直接宣布存在拓扑悬片。','',
 '|请求（均n0）|部位|RAW → POST 观察|','|---|---|---|']
for p,n in zip(pairs,notes):
    md.append('|'+p['request_id'].split('__')[0][:6]+' / '+p['request_id'].split('__')[1]+'|'+n[0]+'|'+n[3]+'|')
md += ['',
 '逐阶段细节见 `review.jsonl`。`C_visual_coherent_use` 是视觉判读，不等同于合法共边或接续指标；`construction_intended_role_visible` 只是条件构造角色诊断。完整闭合C也不因没有开放接口受到额外惩罚。本次所有请求的拓扑断片/精确连接状态均保持 UNKNOWN/NOT_ASSESSED_FROM_IMAGES。','',
 '没有读取来源完整网格进行款式排名；没有生成新样本、没有模型前向、没有重新评分。两张投影和有限截面可能遗漏离截面较远的缺陷，图像也不能认证无相交、无孔洞或所有部件已连通。正式结果仍须按预定后续checkpoint和完整面板评估。','',
 '写入工程记录：首次经Windows stdin传入的中文被默认ASCII替换，已保留原写入文件于 `encoding_repair/before/`。本次通过UTF-8源码恢复同一批实际图审文字，ASCII判读字段及样本身份未改变。','']
(root/'REVIEW.md').write_text('\n'.join(md),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
repair=dict(status='UTF8_REPAIRED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cause='PowerShell stdin to WSL default ASCII replaced Chinese characters',scope='Only newly authored visual review text; no original assets, scores or model calls changed',before={str(p):sha(p) for p in before.iterdir()},repair_code_sha256=sha(__file__))
(root/'encoding_repair/receipt.json').write_text(json.dumps(repair,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
r=json.loads((before/'receipt.json').read_text())
r['encoding_repair_receipt']=str(root/'encoding_repair/receipt.json')
r['files']={str(root/name):sha(root/name) for name in ['review.jsonl','raw_post_comparison.jsonl','review_schema.json','REVIEW.md','encoding_repair/receipt.json','repair_utf8.py']}
(root/'receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert all('?' not in x['notes'] and len(x['notes'])>20 for x in rows)
print(json.dumps(dict(status='PASS_UTF8',rows=len(rows),pairs=len(pairs),model_calls=0),ensure_ascii=True))
