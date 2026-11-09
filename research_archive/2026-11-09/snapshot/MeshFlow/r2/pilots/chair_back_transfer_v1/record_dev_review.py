"""Transcribe the root reviewer's already viewed eight DEV galleries.

No model, rendering, or metric calls. Labels are nonblind AI_PROXY observations,
not inferred from legal-edge scores. Keep uncertain geometry judgments explicit.
"""
from __future__ import annotations
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_chair_back_transfer_v1'

# Each page was opened through view_image: four stages, two surface views and
# two finite-section views. These are observations, not machine classifications.
NOTES={
 'fb8c3a83_BACK_FULL_n0': dict(adoption='UNCERTAIN',reasonable='YES',fragment='UNCERTAIN',overlay='YES',hole='NA_NO_INPUT_HOLE',
  note='四腿、座面和靠背组成可辨认的完整椅子。灰色条栅/弧形背层遮住蓝色矩形C的大部分，蓝面在上沿与条带间可见。侧截面中蓝色平板位于前后灰背轮廓之间，间距持续存在；不能从这些视图确定是合理背板厚度还是替代重层，因此靠背用途采用保留不确定。',
  raw='RAW灰背边线稍零散。',post='D0/GEO后灰背边线较整齐，但双层关系仍在。',change='SAME_AT_VIEW_SCALE'),
 'fb8c3a83_BACK_FULL_n1': dict(adoption='UNCERTAIN',reasonable='UNCERTAIN',fragment='UNCERTAIN',overlay='YES',hole='NA_NO_INPUT_HOLE',
  note='可辨认完整椅身。生成的弧形灰背在高而平的蓝C前方，蓝C在顶部和上角突出；侧截面中两者有持续偏移。C处于靠背位置，但是否作为同一背板自然采用不明确。',
  raw='RAW已存在这一双背关系。',post='BASE到FINAL未见双背偏移被明确消除。',change='SAME_AT_VIEW_SCALE'),
 'abbe7279_BACK_FULL_n0': dict(adoption='YES',reasonable='YES',fragment='YES',overlay='YES',hole='NA_NO_INPUT_HOLE',
  note='蓝色拱顶面可辨认为靠背正面，灰色背侧、座面和四腿构成完整椅子。拱顶与蓝面周围有灰色薄片/尖片候选；侧截面中背后仍有平行灰层和下部间隔。可辨认用途不等于已无缝接入。',
  raw='RAW顶缘薄片较杂。',post='D0/GEO使顶缘稍整齐，后续CPU阶段本页尺度下变化不大。',change='SAME_AT_VIEW_SCALE'),
 'abbe7279_BACK_FULL_n1': dict(adoption='YES',reasonable='YES',fragment='YES',overlay='YES',hole='NA_NO_INPUT_HOLE',
  note='蓝色拱形板可作为靠背表面，背侧有灰色竖条/框架；座面和四腿完整。蓝背下缘附近有不规整小片，侧截面保留平行灰轮廓及间隔。',
  raw='RAW边线更松散。',post='BASE边线稍收拢，HINGE/FINAL没有改变整体外形判断。',change='SAME_AT_VIEW_SCALE'),
 'f199965d_BACK_FULL_n0': dict(adoption='YES',reasonable='YES',fragment='YES',overlay='YES',hole='NA_NO_INPUT_HOLE',
  note='宽拱顶与下部中央蓝面组成可辨认靠背，灰色背柱、座面和腿支撑完整。上沿有少量灰薄片，侧截面及蓝背下部仍见间隔；不能凭图片证明共享索引连接。',
  raw='RAW顶部灰片较突出。',post='D0/GEO后顶部略整齐，额外CPU处理变化较小。',change='SAME_AT_VIEW_SCALE'),
 'f199965d_BACK_FULL_n1': dict(adoption='YES',reasonable='YES',fragment='UNCERTAIN',overlay='YES',hole='NA_NO_INPUT_HOLE',
  note='蓝色宽拱背及下方中央板由灰色背柱框住，座面和四腿可辨认。BASE较RAW减少部分灰背侧间隔；后续处理在背侧左上缘/顶缘有小缺口收拢迹象，但仍保留外侧灰板与多层轮廓。',
  raw='RAW背侧边线和小缺口较明显。',post='HINGE/FINAL有局部小缺口改善，非整把网格无裂缝认证。',change='SLIGHT_LOCAL_IMPROVEMENT'),
 '9d81bc69_BACK_FULL_n0': dict(adoption='YES',reasonable='YES',fragment='YES',overlay='YES',hole='NO',
  note='蓝色拱形背框及中央开口可辨认，灰座面、腿和后侧拱顶构成完整椅子。RAW拱缘有小翻片，D0后更整齐；侧截面仍有灰背框偏移和局部间隔。',
  raw='RAW拱缘薄片候选较明显。',post='BASE到FINAL整体外观相近；不能从视觉变化小推断拓扑没变。',change='SAME_AT_VIEW_SCALE'),
 '9d81bc69_BACK_FULL_n1': dict(adoption='YES',reasonable='YES',fragment='UNCERTAIN',overlay='YES',hole='YES_INTENT_UNKNOWN',
  note='蓝拱框位于椅背，灰色矩形板填充了大部分中央开口，各阶段均存在。输入只给表面，未给禁止填充的空域，因此不把填洞自动判为失败。侧截面中座背附近仍见薄条/间隔。',
  raw='RAW已经填充开口。',post='后处理未改变填充这一形状选择。',change='SAME_AT_VIEW_SCALE'),
}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def csv_write(p,rows):
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    dest=OUT/'visual_review/dev';dest.mkdir(parents=True,exist_ok=True)
    rows=[];pairs=[];evidence={}
    qs=[q for q in read(OUT/'eval_manifest.json')['requests'] if q['split']=='DEV']
    assert len(qs)==len(NOTES)==8
    for q in qs:
        rid=q['request_id'];n=NOTES[rid];p=OUT/'gallery/S2000_BACK_FULL'/f'{rid}.png'
        evidence[str(p)]=digest(p)
        for stage in ('RAW','BASE_POST','HINGE_OPP015','FINAL_POST'):
            rows.append(dict(**{k:q[k] for k in ('request_id','condition_id','parent_id','split','noise_index')},
                stage=stage,reviewer='AI_PROXY',blind=False,image_path=str(p),image_sha256=digest(p),actual_viewed=True,
                back_role_adoption=n['adoption'],body_complete='YES',overall_reasonable=n['reasonable'],
                visible_fragment_or_sliver_suspect=n['fragment'],visible_gap_or_split='YES',local_gray_overlay_on_C=n['overlay'],
                input_opening_filled_or_blocked=n['hole'],exact_topological_connection='UNKNOWN',intersection_legal='UNKNOWN',watertight='UNKNOWN',
                observations=n['note']+' '+n['raw' if stage=='RAW' else 'post'],section_seen=True,opposing_views_seen=True))
        pairs.append(dict(request_id=rid,RAW_to_BASE_visual_change='SLIGHT_IMPROVEMENT' if not rid.startswith('fb8c') else 'SAME_AT_VIEW_SCALE',
            BASE_to_FINAL_visual_change=n['change'],BASE_to_FINAL_back_role='UNCHANGED',BASE_to_FINAL_obvious_new_body_damage='NO',
            note='所记变化限定于实际四阶段图片尺度；未以指标代替图审。'+n['post']))
    data=dict(schema='chair_back_transfer_actual_visual_review_v1',status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
        reviewer='AI_PROXY',blind=False,split='DEV',requests=8,stages_per_request=4,observations=rows,paired_observations=pairs,evidence=evidence,
        policy='Actual image review by root, nonblind. Back-role YES is visual function only, not exact topology or watertightness. Gray reverse sides are not automatically illegal duplicates. Source shape is not a unique required completion.')
    write(dest/'visual_review_dev.json',data);csv_write(dest/'visual_review_dev.csv',rows);csv_write(dest/'visual_review_dev_pairs.csv',pairs)
    familiar=read(OUT/'visual_review/familiar/visual_review_familiar.json')
    allrows=familiar['observations']+rows;allpairs=familiar['paired_observations']+pairs
    assert len(allrows)==64 and len({(r['request_id'],r['stage']) for r in allrows})==64
    allrows.sort(key=lambda r:(r['request_id'],r['stage']));allpairs.sort(key=lambda r:r['request_id'])
    csv_write(OUT/'visual_review/per_stage.csv',allrows);csv_write(OUT/'visual_review/paired.csv',allpairs)
    counts={}
    for split in ('FAMILIAR','DEV','ALL'):
        rr=[r for r in allrows if r['stage']=='FINAL_POST' and (split=='ALL' or r['split']==split)]
        counts[split]={key:{v:sum(r[key]==v for r in rr) for v in sorted({r[key] for r in rr})} for key in ('back_role_adoption','body_complete','overall_reasonable')}
    write(OUT/'visual_review/completion.json',dict(status='COMPLETE',requests=16,stages_reviewed=4,actual_review_rows=64,blind=False,reviewer='AI_PROXY',
        source_hashes={str(dest/'visual_review_dev.json'):digest(dest/'visual_review_dev.json'),str(OUT/'visual_review/familiar/visual_review_familiar.json'):digest(OUT/'visual_review/familiar/visual_review_familiar.json')},final_counts=counts))
    print(json.dumps(counts,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
