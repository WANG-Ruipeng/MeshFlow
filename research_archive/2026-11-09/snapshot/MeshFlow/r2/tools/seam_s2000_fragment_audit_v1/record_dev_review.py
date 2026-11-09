"""Manual observations written after root actually viewed all 12 DEV panels."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_s2000_fragment_audit_v1'
# These are actual image observations, not labels derived from geometry metrics.
REVIEWS=[
('2d08a64e_n0','YES','UNCERTAIN','YES','UNRESOLVED',
 '靠背下沿白缝在D0及RESTORED已可见，POST更清楚；RAW该处密集重层/细线遮挡，不能确定同一开口的首次形成。',
 '靠背下沿有细条与开缝；红色也覆盖正常背框和腿框，不能把它们全部认作碎片。未确认细条完全脱离。'),
('2d08a64e_n1','YES','NO','UNCERTAIN','RAW',
 'RAW已有窄边和薄板边缘；D0座面前沿白线一度较明显，恢复与GEO后较整齐，未见新增大裂口。',
 '板式椅完整，细长面主要在板边；没有肉眼明确游离小片，细边线是否裂口不确定。'),
('2d08a64e_n2','YES','UNCERTAIN','YES','RAW',
 '扶手下三角片及不规则背下缘关系在RAW已有，D0/GEO整理后仍在；未见仅锚点恢复才产生该构件。',
 '存在扶手下较大三角撑/折片和背角不规则开口；该灰色大三角不属于极小面积面。红色腿框与扶手包含正常细结构。'),
('2d08a64e_n3','YES','NO','YES','UNRESOLVED',
 'RAW上背的部分缝隙经D0收窄；下座/底座长缝在D0及RESTORED可见，POST更清晰。不同部位方向不同，不能统一归因。',
 '座下有明确长缝，未见完全游离小片；诊断红色主要是底座和座下长条，不都属于异常面。'),
('608b46b0_n0','YES','NO','UNCERTAIN','RAW',
 '顶部密集小三角与不整齐边缘从RAW持续到POST；D0和恢复没有突然出现一组新碎片。',
 '背板有长而窄的三角带，顶部小缺口/孔的性质不确定；未看到主体外漂浮小片。'),
('608b46b0_n1','YES','NO','YES','RAW',
 '靠背顶端中部的小白开口在RAW可见，D0后仍在，POST没有完整封上。',
 '背顶局部小开口与斜向细三角带并存；四腿整体保留，没有明确游离小面。'),
('608b46b0_n2','YES','NO','UNCERTAIN','RAW',
 'RAW已有板边窄条和粗糙线；D0与GEO大体将其整理，未看到新大缺口。',
 '箱式扶手椅较连贯，红色主要为板边/背横条；局部细隙不能确定为真实裂口。'),
('608b46b0_n3','YES','NO','YES','RAW',
 '背顶中部白色孔隙在RAW已存在，D0后更清楚，GEO后仍保留；并非恢复锚点才第一次出现。',
 '顶缘小开口明确，背板长窄三角及座面上方细条可见；未证实漂浮碎片。'),
('610b1621_n0','NO','NO','NO','NONE',
 '四阶段均为较完整的箱式椅身；后处理将密集重线整理，未见明确新增裂口。',
 '主要是较大平面，底边少量红色极薄条未形成明显破损观感；没有可辨认游离片。'),
('610b1621_n1','YES','UNCERTAIN','UNCERTAIN','RAW',
 '扶手下三角片与背侧关系在RAW已存在；D0/GEO后仍在，无法仅凭这两视角确认三角片是否完全脱离。',
 '腿框/扶手红色是细构件；左扶手下较大灰色三角片和背角窄隙值得检查，但不自动叫极小碎面。'),
('610b1621_n2','YES','NO','NO','RAW',
 '细腿和背框薄边在RAW已在，D0/GEO减少粗糙重线，未见明显新增断片。',
 '细长面多数是正常腿边/背框，主体较完整，无明确可见裂口。'),
('610b1621_n3','YES','NO','UNCERTAIN','UNRESOLVED',
 '椭圆背孔从RAW到POST保持，应视为设计开孔；下背与座面邻域的不整齐关系跨阶段存在，但被C遮挡，首次损伤不明。',
 '背框长窄面明显，椭圆开孔不算裂口；背下缘局部细隙不确定，未见漂浮小片。'),
]
def main():
    rows=[]
    for rid,sliver,detached,crack,first,stage_note,post_note in REVIEWS:
        p=OUT/'gallery'/f'{rid}.png'
        rows.append(dict(request_id=rid,split='DEV',reviewer='AI_PROXY_NONBLIND',actually_viewed=True,
            image_path=str(p),image_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            post_sliver_visible=sliver,post_detached_piece_visible=detached,post_crack_visible=crack,
            first_visible_stage=first,stage_change_note=stage_note,post_note=post_note))
    p=OUT/'visual_dev.csv'
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'visual_dev_receipt.json').write_text(json.dumps(dict(status='ACTUALLY_REVIEWED',outputs=12,
        source='Root view_image: all12 full four-stage two-view panels, no metric-derived ratings',
        limitations='Sliver visibility includes legitimate slender members; NO visible detachment does not certify topological connection. New render/labels do not overwrite old review.'),indent=2),encoding='utf-8')
    print('DEV_REVIEW_COMPLETE',len(rows))
if __name__=='__main__':main()
