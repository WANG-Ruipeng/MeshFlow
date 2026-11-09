"""Serialize the root agent's actual seven-image spot review (no inference)."""
from .common import OUT, sha, write_new

OBSERVATIONS = {
 '04': '完整L型两翼座面可见；C-only覆盖完整坐面，两翼与靠背/腿目标相容。保留。',
 '05': 'C覆盖整套平行座板上表面；板间真实空隙并非随机漏片。语义完整，保留。',
 '12': '完整弧形座面横跨两端框架，保留曲率；不是仅挑一条木条。保留。',
 '14': '三段座位与两处隆起分隔构成完整连续坐面，支撑来自原Y。保留。',
 '21': '两三角形覆盖两端框之间完整顶面；不含弯曲端架合理。保留。',
 '22': '蓝C仅为全座面的前沿长条，灰Y仍包含大部坐面；本轮完整座面范围排除，不删除原件。',
 '24': '蓝C覆盖完整弧形长条座面，灰Y为其支撑及侧面；窄长不是局部条带。保留。',
}

def run():
    rows=[]
    for prefix, note in OBSERVATIONS.items():
        matches=list((OUT/'data/inspection_v2').glob(prefix+'_*.png'))
        assert len(matches)==1
        path=matches[0]
        rows.append(dict(image=str(path),image_sha256=sha(path),actually_viewed=True,
                         observation=note,decision='EXCLUDE_PARTIAL' if prefix=='22' else 'ACCEPT_SEMANTIC_FULL'))
    value=dict(status='PASS',reviewer='root AI_PROXY',blind=False,
               actual_images_viewed=len(rows),reviews=rows,
               scope='Root spot review of7 complex/partial candidates. Full25 actual reviews belong to data reviewer; not claiming root viewed all25.',
               limitation='Visual semantic scope and C/Y relation, not certification of every intersection or perfect target topology.',
               model_forwards=0,training_updates=0)
    write_new(OUT/'data/root_input_review.json',value)
    return value

if __name__=='__main__': print(run()['status'])
