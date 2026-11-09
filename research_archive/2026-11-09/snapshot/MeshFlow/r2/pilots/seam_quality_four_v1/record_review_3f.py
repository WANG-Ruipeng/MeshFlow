"""Transcribe root's actual eight-page visual inspection, not an image classifier."""
import csv
from . import eval_common as C


def main():
    root=C.OUT; schema=C.read(root/'evaluation/review_schema.json')
    manifest=C.read(root/'eval_manifest.json'); rows=[]
    for q in manifest['requests']:
        if not q['parent_id'].startswith('3f741759'): continue
        n=int(q['noise_index'])
        for stage in ('RAW','POST'):
            path=root/'evaluation/gallery'/(q['request_id']+'_'+stage+'.png')
            for model in schema['model_order']:
                old=model=='SEAM200'
                observation='蓝色C处于座面，圆弧靠背、竖向背条与四腿轮廓可辨；未见整块主体或支撑消失。'
                fragments='NO'
                if n==0:
                    observation+='正面中央剖面存在侧壁断段；外观整体近似，侧剖下层仍在。'
                    change='REFERENCE' if old else '整体轮廓近似；正面剖面右侧壁断段位置改变，没有明显整体视觉跃升。'
                elif n==1:
                    if old:
                        fragments='YES'
                        observation+='正面剖面有从C左端向内斜伸的灰线，侧剖有独立内层短线；整体视图该层大部分被座面遮挡。'
                    else:
                        observation+='正面与侧面剖面不再出现原来的内斜线/独立内层短线；侧裙与靠背依然可辨。'
                    change='REFERENCE' if old else '原内斜片和侧剖内层线在这些剖面中消失；主体保持，不能据此证明全表面无重层。'
                elif n==2:
                    if old and stage=='RAW':
                        fragments='YES'
                        observation+='两剖面中C下方可见近邻灰色额外线段，整体外观将其遮住。'
                    elif old:
                        fragments='UNCERTAIN'
                        observation+='C下近邻线段在POST剖面中难以分辨，可能被重合投影/遮挡；不能判定该层已消失。'
                    elif stage=='RAW':
                        fragments='UNCERTAIN' if model in ('LONG','ANTI_COVER') else 'NO'
                        observation+='C下近邻长线不再明显，前剖左侧有开口；'+('靠背中条较斜，是否折叠还是可接受设计不确定。' if model in ('LONG','ANTI_COVER') else '靠背条较直，侧剖底部有局部短线。')
                    else:
                        observation+='靠背条整理为较直的条带，前剖左侧开口仍在，外形与旧结果很接近。'
                    change='REFERENCE' if old else 'RAW中原C下近邻长线不明显；POST整体近似。LONG/ANTI的RAW背条比LOCAL更斜，但不据此判款式错误。'
                else:
                    observation+='各模型整体外形很接近；'+('侧剖下层较连续。' if old else '侧剖下层左部出现可见断段，是否为需封闭的壳体裂口仍不确定。')
                    change='REFERENCE' if old else '主体、蓝色座面用途保持；侧剖底部较旧结果多一处断段，没有明确整体优势。'
                rows.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=n,stage=stage,model=model,
                    reviewer='AI_PROXY_ROOT',review_kind='AI_PROXY',review_status='ACTUALLY_VIEWED',
                    blinding='NONBLIND_MODEL_LABELS_VISIBLE',C_adoption='YES',whole_body_complete='YES',support_missing='NO',
                    fragments_or_folded_layers=fragments,visible_cracks='UNCERTAIN',visible_join='UNCERTAIN',
                    visible_observations=observation,change_vs_SEAM200=change,
                    uncertainty='仅双整体视图与两个有限中央剖面；蓝色C会遮挡近共面层，剖面断段不自动等于非法破洞。图像不认证共享索引、隐藏穿插或整圈接续。',
                    image_path=str(path.relative_to(C.ROOT)),image_sha256=C.sha(path)))
    assert len(rows)==32
    dest=root/'evaluation/reviews';dest.mkdir(exist_ok=True)
    with (dest/'3f.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=schema['csv_columns']);w.writeheader();w.writerows(rows)
    with (dest/'3f.md').open('x',encoding='utf-8') as f:
        f.write('# 3f actual AI_PROXY review\n\nRoot opened all eight PNGs, RAW/POST n0–n3, and inspected all four model columns before writing 32 rows. Nonblind; no score-derived visual labels.\n\nAll displayed models retain a recognizable chair and use blue C as its seat. On n1 the old inward gray section and internal layer line are absent in the new models. On n2 RAW the old near-seat internal line is no longer obvious, while LONG/ANTI back slats are more oblique than LOCAL; POST largely regularizes them. Whole views otherwise look very similar. New n3 side sections contain a lower-surface break that remains ambiguous as a design opening versus crack. No clear loss of a complete support or body is visible.\n\nNear-coplanar coverage under C, exact topological joins and hidden collisions cannot be certified from these views. In particular the visually similar n2 POST is not a proof that its conflict tail is absent.\n')


if __name__=='__main__':main()
