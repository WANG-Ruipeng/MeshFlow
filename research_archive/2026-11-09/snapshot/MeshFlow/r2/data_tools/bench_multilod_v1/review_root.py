"""Persist actual /root image observations made after view_image, not auto grading."""
from . import common as c


def main():
    observations={
        'c97683ce5dcea25ad1d789f3b2120d0': ('engineering_v2/', '三视图中长座面、两端弯曲支撑及底部横板均保留；减面主要改变灰色自由面的划分。未见新增开口或支撑缺失，蓝色矩形C形状与边界一致。'),
        'ef207b78de8277ec42fc59644ef58e22': ('engineering_v2/', '两端高侧板、下部支脚和横向承托的轮廓在原版与候选间保持；蓝色座面未变。侧板三角划分明显减少，但这些视角未见新的孔洞或细支撑消失。'),
        '4cd23aae912b7de019fb4103277a6b93': ('', '箱式Bench的外壳、上缘框和蓝色C均保留。候选减少了箱体内部三角划分；前后斜视及俯视未见明显外形损伤或缺面。'),
    }
    for parent,(prefix,note) in observations.items():
        p=c.OUT/prefix/'gallery'/(parent+'.png')
        c.write(c.OUT/'visual_review'/(parent+'.json'),dict(parent_id=parent,status='PASS',
            reviewer='/root',method='AI_PROXY',actually_viewed=True,nonblind=True,
            viewed_images=[dict(path=str(p),sha256=c.sha(p))],observations=note,
            rejection_reason=None,limitation='Actual image inspection of three fixed views; not human certification or exhaustive hidden-surface proof.'))


if __name__=='__main__':main()
