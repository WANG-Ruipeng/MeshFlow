"""Persist root's actual review of four emitted paired contact sheets."""
from r2.pilots.seam_slot_c200_v1 import common as C

OUT=C.ROOT/'experiments/r2_seam_geometry_guard_v1'
NOTES={
 '73b369b9':'C仍表现为座面，靠背和底部折板形状保留；n0/n1原有底部间隙与n2背部裂缝仍可见。容量版未见新增大块主体缺失，未把原有间隙视为已修复。',
 '3f741759':'四噪声的座面、带竖条靠背和支撑腿均保持可辨；容量版全局轮廓接近基线。n2重复覆盖被蓝色C及遮挡掩盖，图像不能证明I问题消失。',
 '85b16941':'四噪声中座面、背板与支撑腿保持可辨；n0保留较宽背板，n1/n2/n3保留较薄靠背和腿。局部边角三角划分可有细微改变，没有看到新增大块形状丢失。',
 '8c9b2023':'四噪声的座面、横档靠背和四腿轮廓均保持；容量版未见明显新增支撑缺失或主体崩坏。共边合法性和相交仍以数值账本为准。',
}

def run():
    g=C.read(OUT/'gallery/completion.json');rows=[]
    for r in g['rows']:
        assert C.sha(r['image'])==r['sha256']
        for rid in r['requests']:
            rows.append(dict(request_id=rid,parent_id=r['parent_id'],reviewer='root_AI_PROXY',review_status='ACTUALLY_VIEWED_NONBLIND',
                views='both fixed views of baseline and capacity; all four noise rows',image=r['image'],image_sha256=r['sha256'],
                obvious_new_global_shape_loss='NO_VISIBLE_IN_THESE_VIEWS',semantic_use_visibly_changed='NO_VISIBLE_CHANGE',
                precise_connection='UNKNOWN_FROM_IMAGES_USE_INDEX_GRAPH',collision_free='NOT_CERTIFIED',
                notes=NOTES[r['parent_id'][:8]]))
    C.csvwrite(OUT/'gallery/actual_visual_review.csv',rows)
    C.write(OUT/'gallery/review_receipt.json',dict(status='COMPLETE',paired_requests=16,mesh_cells=32,views_per_mesh=2,
        reviewer='root_AI_PROXY',blind=False,human_rating=False,semantic_relabel=False,registered_scope='postprocessing appearance preservation only',
        files={str(OUT/'gallery/actual_visual_review.csv'):C.sha(OUT/'gallery/actual_visual_review.csv')}))

if __name__=='__main__':run()
