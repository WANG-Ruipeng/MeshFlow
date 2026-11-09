"""Transcribe actual image observations made through view_image (not scoring).

All 12 named comparison JPGs were inspected at both saved views, all columns,
and both stages by the AI_PROXY reviewer before these labels were entered.
"""
from r2.pilots.target_cleanup_c200_v1 import common as C
import json

dest=C.OUT/'evaluation/gallery'
template=C.csvread(dest/'review_template.csv')
seen=['73b369b9','3f741759','85b16941']
rows=[]
for source in template:
    prefix=source['parent_id'][:8]
    if prefix not in seen:continue
    r=dict(source);noise=int(r['noise']);label=r['label'];stage=r['stage']
    assert C.sha(r['image_path'])==r['image_sha256']
    r.update(reviewer='AI_PROXY',review_status='REVIEWED',review_mode='NON_BLIND_AI_PROXY',
        current_continuation_exposure='NOT_IN_16_PARENT_POOL',C_usage='YES',body_complete='YES',
        visible_fragments='NO',visible_cracks='NO',visible_extra_surfaces='UNKNOWN',visible_join='UNKNOWN',
        observation_scope='Two opposite saved views; NO means not visible here, not certified absent. Exact connectivity not judged from pixels.',
        reviewed_images_json=json.dumps([dict(path=r['image_path'],sha256=r['image_sha256'])]),
        notes='')
    notes=[]
    if prefix=='73b369b9':
        if noise==0:
            notes.append('蓝C是宽座面，背板/底座形成可辨认的悬臂椅；背板下沿与座面、下方斜撑附近可见细白间隔。')
            r['visible_cracks']='YES'
            if label=='DEDUP_C200':notes.append('整体仍为同类悬臂轮廓，未见明显新增游离碎片；不能从贴近外观认证共边。')
        elif noise==1:
            notes.append('蓝C处于可用座面位置，折线顶背板与底座均在；三模型总体相近，面内对角线不是裂缝证据。')
            if stage=='RAW':r['visible_cracks']='UNKNOWN';notes.append('边缘细小间隙与三角网格线在该尺度不能完全区分。')
        elif noise==2:
            r.update(C_usage='NO',visible_extra_surfaces='YES')
            notes.append('三模型均有灰色上层座面，蓝C仅从较低侧缘露出；按当前座面输入用途，上层灰面仍在替代蓝C。扶手、背与支撑主体完整。')
            if label=='DEDUP_C200':
                r['visible_cracks']='YES'
                notes.append('前缘可见蓝色细带/灰色座板边缝，POST仍存在；不是从图像确认的拓扑断裂。')
        else:
            r['C_usage']='UNKNOWN';r['visible_extra_surfaces']='YES'
            if label=='FULL_S6000':
                notes.append('厚背/厚前挡板和条状座面并存，蓝C分段露出；座面附近存在多层条带，是否以蓝C为主要承坐面不确定。')
                if stage=='RAW':r['visible_cracks']='YES';notes.append('RAW条带间和背板边缘有明显空隙。')
            elif label=='ORIGINAL_C200':
                notes.append('形态变为带扶手框椅，蓝C处于框架内较低处，前横板/扶手在更高位置；C用途与局部接续需保留不确定。')
                r['visible_cracks']='YES'
            else:
                notes.append('变为高背板/箱形支撑，灰色斜座板跨在蓝C上方，蓝C前后缘外伸；额外层的用途不清。')
                r['visible_cracks']='YES'
    elif prefix=='3f741759':
        if noise in (0,1,3):
            notes.append('蓝C明确作为座面，拱形靠背、竖条和四腿主体均可辨认；不以来源款式一致作为优劣。')
        if noise==0 and label in ('ORIGINAL_C200','DEDUP_C200'):
            r['visible_cracks']='YES';notes.append('后侧视角座框侧裙有三角白缺口，POST仍可见；其余主体未缺失。')
        elif noise==1:
            if stage=='RAW':
                r['visible_cracks']='YES';notes.append('RAW靠背顶帽有裂口或局部腿面开缝；细长腿仍在。')
            elif label=='ORIGINAL_C200':
                r['visible_cracks']='YES';notes.append('POST靠背顶帽仍有小白裂口。')
            elif label=='DEDUP_C200':
                r['visible_cracks']='YES';notes.append('POST一侧腿端仍见开缝/缺边，顶帽较RAW完整。')
            else:notes.append('POST较RAW腿部更整齐，两个视角未见同等明显长裂口。')
        elif noise==2:
            notes.append('蓝C处在桶形围背椅的座面位置；整体可辨认，封闭外壳不是判定正确款式的要求。')
            if label=='FULL_S6000':
                if stage=='RAW':r['visible_cracks']='YES';notes.append('RAW上沿、外壳拼片有细小间隔。')
                else:notes.append('POST外壳较连贯，局部隐藏重层不可由外观排除。')
            elif label=='ORIGINAL_C200':
                r['visible_cracks']='YES';notes.append('后上沿有明显三角长缺口，正面左支撑出现细长缺口，POST仍未消失。')
            else:
                r.update(visible_cracks='YES',visible_fragments='UNKNOWN')
                notes.append('外壳有不规则竖向裂口和底缘缺口，RAW尤其显著，POST仍保留；条片是否独立断开无法仅凭图确认。清理目标未在此例带来完整外壳。')
        elif noise==3 and label=='DEDUP_C200' and stage=='RAW':
            r['visible_cracks']='YES';notes.append('RAW靠背顶帽有小三角缺口，POST该处较闭合。')
    else:
        notes.append('蓝C处于座面，椅背与支撑轮廓可辨认。')
        if noise==1:
            if label=='DEDUP_C200':
                r.update(body_complete='UNKNOWN',visible_cracks='YES',visible_fragments='UNKNOWN')
                notes.append('形态明显改为细框/空背，部分框杆端部错开，背横条在POST变宽；可能是开放框架款式，也可能部分面缺失，不据此直接判坍缩。')
            else:
                notes.append('粗柱/厚脚架款式，蓝座面外伸；部分腿有开放侧面，造型用途仍可辨认。')
                r['visible_cracks']='YES'
        elif noise==2 and label=='DEDUP_C200' and stage=='RAW':
            r['visible_cracks']='YES';notes.append('靠背上沿有小三角开缝，POST较整齐。')
        elif noise==3 and label in ('FULL_S6000','DEDUP_C200'):
            r['visible_cracks']='YES';notes.append('侧裙有可见开放缺口，POST仍留有局部空隙。')
        else:
            notes.append('三列整体相近，两个视角未见明显游离碎片；隐藏面交叠仍不作无缺陷认证。')
    r['notes']=''.join(notes)
    rows.append(r)
assert len(rows)==72 and len({r['image_path'] for r in rows})==12
C.csvwrite(dest/'actual_review_historical_train.csv',rows)
C.write(dest/'actual_review_historical_train_receipt.json',dict(status='COMPLETE',reviewer='AI_PROXY',review_mode='NON_BLIND',
    rows=72,actually_viewed_images=12,source='Direct view_image inspection in this agent session before transcription',
    image_hashes={r['image_path']:r['image_sha256'] for r in rows},review_sha256=C.sha(dest/'actual_review_historical_train.csv'),
    source_code_sha256=C.sha(__file__),model_calls=0,score_derived_visual_labels=False))
print(json.dumps(dict(status='COMPLETE',rows=72,images=12)))
