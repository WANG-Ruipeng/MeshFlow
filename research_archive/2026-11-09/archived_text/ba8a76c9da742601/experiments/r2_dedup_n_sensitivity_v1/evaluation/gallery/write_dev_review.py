"""Transcribe actual viewing of all twelve DEV four-column contact sheets."""
import csv,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
OLD=P.parents[2]/'r2_target_cleanup_c200_v1/evaluation/gallery/actual_review_dev.csv'
def readcsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
old={(r['request_id'],r['label'],r['stage']):r for r in readcsv(OLD)}
# low-N observations: C usage, cracks, body complete, RAW, POST.
obs={
('2d08a64e',0,'ORIGINAL_C200'):('YES','YES','YES','改成包围侧板、卷曲顶背及曲面底座，座下有多条不齐白缝和细片。','座下白色缺口、层状薄条仍在；C呈座面，但不能把换成曲面款式称为补裂。'),
('2d08a64e',0,'DEDUP_C200'):('YES','YES','YES','也改成包围侧板和卷曲顶背，背与座下表面有不齐开口。','背部与座下仍有不齐白色间隙，侧板完整不等于整体裂片改善。'),
('2d08a64e',1,'ORIGINAL_C200'):('YES','YES','YES','曲面椅体替代滑橇框；座侧有向外尖薄片，底座面带不齐横向开口。','尖薄片/底座开口仍见，蓝色C仍呈座面；不能认定较低N更规整。'),
('2d08a64e',1,'DEDUP_C200'):('YES','YES','YES','窄板底座改为曲面悬臂式，侧部有突出的薄条与开口。','局部面比RAW整块，但座侧突出薄片与底座不齐开口仍在。'),
('2d08a64e',2,'ORIGINAL_C200'):('YES','YES','YES','出现与n1近似的曲面椅体；座侧突出细片，底座表面裂缝样白带。','底座仍有显著白色开口，侧向突出细片仍在。'),
('2d08a64e',2,'DEDUP_C200'):('YES','YES','YES','曲面椅体、座侧尖片与底座不齐条带均可见。','底座/座侧开口未消失，不能把平滑曲背当作整体质量改善。'),
('2d08a64e',3,'ORIGINAL_C200'):('YES','UNKNOWN','YES','从厚外壳椅改为曲面椅；座侧有薄片，背部有条带。','主体曲面较连续，但侧向突出薄条仍见；款式改变主导，收益与代价混合。'),
('2d08a64e',3,'DEDUP_C200'):('YES','UNKNOWN','YES','板状悬臂底座变成曲面底座，座侧仍突出细片。','曲面体看起来连贯，但侧向薄条/细缝还在，不能单独记为修复。'),
('610b1621',0,'ORIGINAL_C200'):('NO','YES','YES','扶手椅改成较窄曲面椅；蓝色C大幅超出生成座面，灰色座前条与C分层。','C大幅悬伸及前缘分层仍在；整体椅体尚可辨，但输入采用比原N不可信。'),
('610b1621',0,'DEDUP_C200'):('NO','YES','YES','曲面椅体明显小于C，灰色前部板条与蓝面之间有大白隙。','白隙及C悬伸仍在；未见合理贴合完整输入。'),
('610b1621',1,'ORIGINAL_C200'):('UNCERTAIN','UNKNOWN','YES','变为细腿条带背；背部横条有白缝，C仍比座框宽。','背板较整块，但C仍外伸；原扶手壳和前裙减少主要属于款式变化。'),
('610b1621',1,'DEDUP_C200'):('UNCERTAIN','UNKNOWN','YES','同样变为细腿横带背，C与窄座框轮廓不完全相容。','背面局部平整，C外伸仍在；不能认定空壳减少就是几何改善。'),
('610b1621',2,'ORIGINAL_C200'):('NO','NO','YES','仍是过小椅架配大C，背部改为圆顶板。','四腿/圆顶板完整可辨，C相对椅架过大的问题未改善。'),
('610b1621',2,'DEDUP_C200'):('NO','UNKNOWN','YES','小椅架加两横条背，C仍多侧大幅悬伸。','腿/背型变化，C失配仍在；未见明确裂片或主体表面收益。'),
('610b1621',3,'ORIGINAL_C200'):('UNCERTAIN','UNKNOWN','YES','圆孔厚背改为条带薄背和细腿，C外伸。','薄背表面整块，但主体款式已变化；无法确认比原N减少裂片。'),
('610b1621',3,'DEDUP_C200'):('UNCERTAIN','YES','NO','RAW有四根可见细腿、薄背，C仍外伸；腿比原N显著细。','POST两视角腿部几乎退成断续细线，主要支承表面明显流失；不能将删面少或薄化当成改善。'),
('608b46b0',0,'ORIGINAL_C200'):('YES','YES','YES','实体圆顶背变为开窗背；前腿纵向白缝仍明显，座框仍偏于C内部。','前腿纵隙和C轮廓偏置仍在；背部开窗本身是款式变化，不算缺面或修复。'),
('608b46b0',0,'DEDUP_C200'):('YES','YES','YES','开窗背，前腿纵向白缝明显，C仍呈座面。','前腿纵隙仍大，背底横向条板也不齐；未见清晰补裂收益。'),
('608b46b0',1,'ORIGINAL_C200'):('YES','UNKNOWN','YES','扶手椅背与前裙改形，座前新增层状条板与侧面开口。','背部较整块，但前裙层次和侧角开口仍在；不是一致改善。'),
('608b46b0',1,'DEDUP_C200'):('YES','YES','YES','由低背整块方箱变成带头枕板的扶手椅，前部/小脚可见细缝。','前裙层状条与脚部小缺口仍见，比原N整块方箱表面更不齐。'),
('608b46b0',2,'ORIGINAL_C200'):('UNCERTAIN','YES','YES','从方箱椅改成高背窄椅，右侧板有大三角开口，C穿出侧缘。','侧板大开口及C侧向穿出仍在，未见原侧板问题得到可靠修复。'),
('608b46b0',2,'DEDUP_C200'):('YES','YES','YES','侧板/背仍齐，但侧面上沿出现突出的薄板尖翼。','薄板尖翼及侧角不齐仍在，原N较整块的表面未变得更规整。'),
('608b46b0',3,'ORIGINAL_C200'):('YES','UNKNOWN','YES','实体背改为开窗背，座框仍内缩，腿部有细缝。','开窗背和四腿可辨，腿部稍整块，但看不出超越原N的明确裂片收益。'),
('608b46b0',3,'DEDUP_C200'):('YES','UNKNOWN','YES','开窗背下横条呈倾斜/缺口形态；腿部局部细缝。','腿较整块，但背下横条保留不齐白色开口，收益与代价混合。'),
}
# Paired POST outcome; C role retained refers to plausible use, not C bitwise preservation.
judgments={
('2d08a64e',0):('WORSE','WORSE'),('2d08a64e',1):('WORSE','WORSE'),
('2d08a64e',2):('WORSE','WORSE'),('2d08a64e',3):('MIXED','MIXED'),
('610b1621',0):('WORSE','WORSE'),('610b1621',1):('MIXED','MIXED'),
('610b1621',2):('UNCHANGED','UNCHANGED'),('610b1621',3):('UNCHANGED','WORSE'),
('608b46b0',0):('UNCHANGED','UNCHANGED'),('608b46b0',1):('MIXED','WORSE'),
('608b46b0',2):('WORSE','WORSE'),('608b46b0',3):('UNCHANGED','MIXED'),
}
manifest=json.loads((P/'manifest.json').read_text(encoding='utf8'))
pages={x['request_id']:x for x in manifest['images']};viewed=[]
for page in pages.values():
    if (page['parent_id'][:8],int(page['candidate'])) not in judgments:continue
    p=P/Path(page['path']).name;assert hashlib.sha256(p.read_bytes()).hexdigest()==page['sha256']
    viewed.append(dict(path=page['path'],sha256=page['sha256'],actually_viewed=True,columns=4,RAW_POST_views_per_column=4))
rows=[];delta=[]
for r in readcsv(P/'review_template.csv'):
    key=(r['parent_id'][:8],int(r['noise']))
    if key not in judgments:continue
    source=r['source_arm'];low=r['label'].endswith('LOW_N')
    if not low:
        prior=old[r['request_id'],source,r['stage']]
        for field in ('C_usage','body_complete','visible_fragments','visible_cracks','visible_extra_surfaces','visible_join'):
            r[field]=prior[field]
        r['notes']='本轮四列图重新实际查看，原N几何与原记录一致。 '+prior['notes']
    else:
        use,cracks,body,raw,post=obs[key+(source,)]
        r.update(C_usage=use,body_complete=body,visible_fragments='UNKNOWN' if key[0]=='2d08a64e' else 'NO',
            visible_cracks=cracks,visible_extra_surfaces='UNKNOWN',visible_join='UNKNOWN',
            notes=(raw if r['stage']=='RAW' else post)+' 隐藏重叠及精确拓扑UNKNOWN；NO仅表示两视角未见，不认证无缺陷。')
        # RAW still has visible supports for the specific POST loss case.
        if key==('610b1621',3) and source=='DEDUP_C200' and r['stage']=='RAW':r['body_complete']='YES'
    r.update(reviewer='AI_PROXY',review_status='ACTUALLY_VIEWED',review_mode='NON_BLIND_AI_PROXY')
    rows.append(r)
    if low and r['stage']=='POST':
        use,cracks,body,raw,post=obs[key+(source,)]
        ci=0 if source=='ORIGINAL_C200' else 1
        retained='YES' if use=='YES' else 'UNKNOWN'
        if key==('610b1621',0):retained='NO'
        delta.append(dict(request_id=r['request_id'],parent_id=r['parent_id'],split=r['split'],source_arm=source,stage='POST',
            judgment=judgments[key][ci],C_role_retained=retained,
            new_major_body_loss='YES' if key==('610b1621',3) and source=='DEDUP_C200' else 'NO',
            raw_notes=raw,notes=post+' 与同权重原N比较，未以换款式或精确来源款式匹配排名。',
            image_path=r['image_path'],image_sha256=r['image_sha256']))
assert len(rows)==96 and len(delta)==24 and len(viewed)==12
def writecsv(path,rows):
    assert not path.exists(),'Never overwrite existing actual review'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
writecsv(P/'actual_review_dev.csv',rows);writecsv(P/'visual_delta_dev.csv',delta)
receipt=dict(status='COMPLETE',reviewer='AI_PROXY',review_mode='NON_BLIND_AI_PROXY',images_actually_viewed=12,per_cell_rows=96,POST_paired_rows=24,
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'actual_review_dev.csv',P/'visual_delta_dev.csv')},viewed=viewed,
    scientific_calls=0,new_rendering=0,score_derived_labels=False,
    limitation='Nonblind AI proxy, two fixed views. UNKNOWN differs from UNCERTAIN. C-usage is not historical strict adoption. All six parents are outside the 16-parent C200 continuation pool.')
(P/'actual_review_dev_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(status='COMPLETE',rows=96,delta=24,viewed=12)))
