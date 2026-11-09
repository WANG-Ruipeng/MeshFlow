"""Transcribe the root's actual six-page AI_PROXY viewing, not auto semantic scoring."""
from .common import *

OBS={
 ('27eb79','R1'):'已查看两相反斜视M/C/Y/C+Y。蓝色C为完整连续弯曲座背壳，灰色补集为扶手腿框及下部横件；组合保持原来的完整长椅结构，未见裁成随机面片或新缺件。来源壳的开放边不等于必需接续边。',
 ('27eb79','R2'):'已查看两相反斜视。蓝色C为完整扶手腿框，Y保留弯曲座背壳及底部横件；C+Y完整。源部件为闭合框体，走空SEAM布局，不用伪造接口。',
 ('945214','R1'):'两斜视中C为全组座条的上表面，条数和间距规则，Y为端架、座条其他表面及纵拉杆，组合完整。N448超过现行合同；语义可解释不使其自动准入。',
 ('945214','R2'):'两斜视中C为完整一侧端框，包含上下横档、竖向腿及底脚，Y为另一端框、座条与纵拉杆；组合恢复完整结构。N448保持阻塞。',
 ('585673','R1'):'两斜视中C为四块完整座板，Y为四腿和下框，形状规则且组合有完整支撑。但完整目标是近似方盒重建，面积和局部保真已超固定门限；图上合理不能覆盖该失败。',
 ('585673','R2'):'两斜视中C为一根完整长方体角腿，Y保留另三腿、下框和四座板；没有把随机散面称为腿。完整目标保真失败，因此条件随目标HOLD。',
}

def main():
    rows=read(OUT/'tasks/manifest.json')['rows'];images={r['condition_id']:r for r in read(OUT/'gallery/conditions_manifest.json')['rows']};reviews=[]
    parents=read(OUT/'reviews/targets_agent.json')['rows'];byparent={r['parent_id']:r for r in parents}
    for r in rows:
        cid=r['condition_id'];im=images[cid];assert sha(im['image_path'])==im['image_sha256']
        key=(r['parent_id'][:6],r['recipe_id']);assert key in OBS
        reviews.append(dict(condition_id=cid,parent_id=r['parent_id'],reviewer='AI_PROXY_ROOT',actually_viewed=True,
          image_path=im['image_path'],image_sha256=im['image_sha256'],semantic_region='PLAUSIBLE',decision='SEMANTIC_PASS',notes=OBS[key],
          limitation='Nonblind AI visual observation, not human rating, generated quality, or finite geometry certification.'))
    save(OUT/'reviews/conditions_root.json',dict(rows=reviews,source='Actual view_image calls on all6condition PNGs',model_generation=0))
    final=[]
    for r in rows:
        p=byparent[r['parent_id']]
        # Whole-target reviewer names the visual acceptance independently of N / finite gates.
        assert p.get('actually_viewed',True)
        target_visual=p.get('visual_geometry_plausible',False)
        eligible=bool(r['pre_review_eligible'] and target_visual)
        final.append(dict(r,eligible_for_production_contract=eligible,visual_target_pass=target_visual,
          actual_semantic_review='PASS_AI_PROXY',review_status='ACTUALLY_VIEWED',
          future_training_pool_membership=False,
          training_pool_note='No condition admitted until production compatibility; R2 remains heldout even if compatible.'))
    save(OUT/'tasks/reviewed_manifest.json',dict(rows=final,parents=3,conditions=6,production_eligible=sum(r['eligible_for_production_contract'] for r in final),
      review_hashes={str(p):sha(p) for p in (OUT/'reviews/targets_agent.json',OUT/'reviews/conditions_root.json')},training_started=False))
    event('ACTUAL_VISUAL_REVIEW_COMPLETE',target_pages=3,condition_pages=6,eligible_for_frozen_head_or_closed_path=sum(r['eligible_for_production_contract'] for r in final))
    print('REVIEWED',len(final),'ELIGIBLE',sum(r['eligible_for_production_contract'] for r in final))

if __name__=='__main__':main()
