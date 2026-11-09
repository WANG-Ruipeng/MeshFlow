"""Record three actually viewed CPU diagnostic image pages; no new inference."""
from .register import OUT,write,sha

def main():
    observations=[
      ('f5d370e9fe5b7452dbbc9440457e303e__BACK_SURFACE_1__n1',
       'RAW双表面视角可见座面和数根细腿。PRE_GEO中细腿已近乎线状/断续，只剩微弱线段；POST几乎不可见，保留蓝背与灰座板。最明显的支撑变薄在D0后提议中已经出现，GEO又删除了坍缩面。固定C中央截面未穿过所有腿，不能仅凭其空白判断腿。',
       'VISIBLE_SUPPORT_LOSS_ALREADY_IN_PRE_GEO'),
      ('9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n0',
       'RAW腿是可辨认的有宽度支撑；PRE_GEO明显压成细线，POST保留断续线条。蓝色窄C和灰座层高度偏移在RAW就存在，后处理并未解决；不把全部问题都归GEO。',
       'VISIBLE_SUPPORT_THINNING_AND_PREEXISTING_LAYER_OFFSET'),
      ('1dd7409c95d59d0da192483aa282f8e5__SEAT_SURFACE_1__n0',
       '三阶段都保留箱式长椅轮廓与背板。PRE_GEO座部仍多层，POST蓝色座板更清楚，整体箱体未因高删面计数明显消失。高删面不自动等于主体被毁；图片不认证无交或精确连接。',
       'HIGH_DELETION_RETAINED_BOX_BODY_COUNTEREXAMPLE')]
    rows=[]
    for rid,note,label in observations:
        p=OUT/'gallery/RAW_PRE_GEO_POST'/f'{rid}.png'
        rows.append(dict(request_id=rid,checkpoint='SEAM_S21000',image=str(p),image_sha256=sha(p),
            actual_viewed=True,reviewer='AI_PROXY',blind=False,stages=['RAW','PRE_GEO','POST'],
            opposing_views_seen=True,fixed_front_and_side_sections_seen=True,observations=note,descriptive_label=label,
            exact_topology_from_images='UNKNOWN',all_deleted_faces_are_normal_wood='UNKNOWN'))
    write(OUT/'visual_review.json',dict(status='COMPLETE',pages=3,reviewer='AI_PROXY',blind=False,observations=rows,
        scope='Purposeful diagnostic examples, not full96 appearance ratings or a human study. No source-style reconstruction criterion.'))
    print('Recorded3actually_viewed_pages')
if __name__=='__main__':main()
