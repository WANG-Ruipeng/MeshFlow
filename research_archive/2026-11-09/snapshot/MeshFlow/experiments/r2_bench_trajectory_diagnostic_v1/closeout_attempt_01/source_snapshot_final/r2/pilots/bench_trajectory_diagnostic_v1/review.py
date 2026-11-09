"""Persist actual non-blind AI_PROXY image observations, not metric-derived labels.

All main, oracle and continuation pages were opened and inspected in this session.
Only the eight explicitly listed five-stage pages receive stage-specific reviews.
"""
from .common import *

# Values were transcribed after actual two-camera and two-section inspection.
# Each entry: COMMON adoption, USER adoption, COMMON note, USER note, final cost.
OBS={
('35948e',0):('NO','NO','灰座板位于蓝色座面上方，C单独悬在座下。','仍重新生成更高的灰座板，C未成为实际座面。','高度错层在RAW已经存在，FINAL没有修好。'),
('35948e',1):('YES','YES','蓝C处于座面，靠背和两侧支撑围绕其组织。','蓝C用作座面，座背与两端支撑关系合理。','局部缝隙和表面变薄仍在；主体关系保持。'),
('943dde',0):('PARTIAL','YES','蓝C大致处于背部，灰主体的座背位置及覆盖与C不完全协调。','蓝C作为外背板，前方座面、内侧背条和端架组织更协调。','背条间有缝隙；不能将所有内侧灰背条都当成非法重复。'),
('943dde',1):('PARTIAL','PARTIAL','蓝色背板位于生成座面下方，更像侧裙。未输入用途标签，不能仅因不是靠背判定所有完成无效。','C仍处于座下侧裙位置，没有复现n0的靠背组织。','USER的灰座面和支撑在D0后明显变细、缺失；COMMON主体相对完整。'),
('bf504b',0):('NO','YES','较小灰长椅在蓝L形支撑内部/旁侧，蓝支撑大幅外露。','灰主体沿蓝L形侧支撑的尺度与位置形成座面及端架。','USER主体保留，局部细条和缝隙仍可见。'),
('bf504b',1):('NO','YES','蓝L形支撑与较小灰长椅位置不相容，未承担合理支撑用途。','蓝L形支撑与新生成的较大主体对应。','USER宏观支撑关系保持，不能由外观认证无自交。'),
('838c48',0):('PARTIAL','YES','C上表面部分采用，但蓝腿与灰支撑错位，灰条横过顶面。','组合C的上表面和腿部被同时利用，主体组织更连贯。','USER最终结构保持，局部侧面纹理/薄片不能忽略。'),
('838c48',1):('PARTIAL','YES','顶部C部分采用，蓝腿仍落在另一些灰支撑内部，侧壳倾斜。','顶部和对应腿部关系合理，围绕组合C完成主体。','USER后处理没有明显毁掉宏观关系。'),
('5dda4e',0):('NO','NO','灰座面在C下方，蓝座板上浮。','增加靠背但灰座面仍远低于C，蓝C不能自然成为座面。','USER支脚出现明显三角缺口；后处理未恢复座面用途。'),
('5dda4e',1):('NO','NO','灰座面在C上方，C没有被采用为座面。','灰座面低于C，C位于高处；错层方向改变但未解决。','USER腿/支撑变细，部分只剩短条。'),
('9d5608',0):('UNCERTAIN','PARTIAL','薄C位于靠背上沿附近，局部是否真正参与结构不能仅凭图确认。','蓝色薄顶条与灰背板上沿之间可见间隔。','USER细腿变薄；薄顶条语义有歧义，不按来源款式唯一判定。'),
('9d5608',1):('PARTIAL','PARTIAL','灰双层箱形主体在C附近，C不明显成为预登记靠背上沿。','蓝顶条位于灰靠背上方，仍有可见间隔。','没有足够视觉证据升级为明确采用；不把不同款式本身判坏。'),
('2d1e5',0):('NO','NO','灰箱形主体包住蓝L形支撑，C没有成为主体的外侧支撑。','同样出现包住C的灰箱壳，C处于主体内部。','箱形款式本身允许；问题是蓝支撑与主体的空间关系。'),
('2d1e5',1):('NO','NO','较高灰靠背及灰座面与蓝L支撑错位。','蓝L形C仍在灰主体上部/内部，未合理参与支撑。','后处理没有修复宏观错位。'),
('f495bd',0):('PARTIAL','PARTIAL','蓝座板上方/下方存在另层灰座板，蓝中腿没有被合理利用。','RAW座面较靠近C，但仍有重层，蓝中腿未与灰主体协调。','USER灰座面和中部支撑在D0后明显变薄/丢失，最终主要剩C与端片。'),
('f495bd',1):('PARTIAL','PARTIAL','灰条跨过蓝上表面，蓝中腿未清楚接入主体。','顶部有部分采用，但中央蓝腿和非对称灰支撑未协调。','两模型均变薄；USER仍有非对称附加支撑。'),
}

def note(q):
    return next(v for (prefix,n),v in OBS.items() if q['parent_id'].startswith(prefix) and int(q['noise_index'])==n)

def main():
    requests=read(OUT/'evaluation_inputs/manifest.json')['requests'];d=OUT/'visual_review';rr=[]
    for q in requests:
        rid=q['request_id'];a,b,c,u,cost=note(q);im=OUT/'gallery_full/main'/f'{rid}.png'
        assert im.exists()
        for label,adoption,explanation in [('COMMON',a,c),('USER',b,u)]:
            for stage in ('RAW','CLEAN_FINAL_POST'):
                compromised=label=='USER' and stage=='CLEAN_FINAL_POST' and (rid.startswith('943dde') and int(q['noise_index'])==1 or rid.startswith('f495bd') and int(q['noise_index'])==0)
                rr.append(dict(request_id=rid,parent_id=q['parent_id'],part=q['part_type'],split=q['split'],noise_index=q['noise_index'],label=label,stage=stage,
                    reviewer='AI_PROXY',blind=False,review_session_date='2026-10-22',C_adoption=adoption,
                    body_completeness='COMPROMISED' if compromised else 'BROADLY_PRESENT_WITH_LOCAL_DEFECTS',
                    observation=explanation,stage_note=cost if stage!='RAW' else '用途与位置按RAW单独评阅；不由POST合法边倒推。',
                    exact_connectivity='NOT_CERTIFIED_BY_IMAGE',image=str(im.relative_to(OUT)),image_sha256=sha(im)))
    assert len(rr)==64
    csvwrite(d/'main.csv',rr)
    oracle=[];continuation=[]
    for q in requests:
        if q['split']!='CURRENT_TRAIN':continue
        rid=q['request_id'];adoption=note(q)[1];im=OUT/'gallery_full/oracle'/f'{rid}.png'
        exact=rid.startswith('943dde')
        oracle.append(dict(request_id=rid,reviewer='AI_PROXY',blind=False,production_adoption=adoption,source_layout_adoption=adoption,
            source_layout_same=exact,observation='布局相同，复用同一RAW；不计新增生成。' if exact else '增加/改变橙色seam和局部侧面，没有新增宏观明确采用请求；来源布局不是可部署结果。',
            image=str(im.relative_to(OUT)),image_sha256=sha(im)))
        im=OUT/'gallery_full/continuation'/f'{rid}.png'
        for t in (.5,.8,.94):
            continuation.append(dict(request_id=rid,start_t=t,reviewer='AI_PROXY',blind=False,
                start_observation={.5:'已有目标信息，但角点仍明显噪声分裂。',.8:'主体已经可辨，仍有噪声扭曲。',.94:'主体与C关系已接近来源，细条仍有扰动。'}[t],
                end_observation='座背/支撑/组合的宏观关系维持或恢复；局部条纹、缝隙仍有，不认证无相交或拓扑完整。',
                new_macro_collapse_observed=False,source_informed=True,generation_success='NA_SOURCE_INFORMED',
                image=str(im.relative_to(OUT)),image_sha256=sha(im)))
    csvwrite(d/'source_layout.csv',oracle);csvwrite(d/'continuation.csv',continuation)
    stage_notes={
        '35948e':'RAW已出现灰座面高于蓝C。D0移动C但未消除错层，恢复C使间隔重新显露；不能归因于GEO单一步骤。',
        '943dde':'n1的C处于座下。COMMON仍保持较完整灰主体；USER的灰座面/细腿在D0提议阶段已明显变薄，PRE/GEO没有恢复。',
        '5dda4e':'RAW已经错层。USER在D0提议中支脚出现缺口/变薄，后续保留；COMMON同样未解决C用途。',
        'f495bd':'RAW已有重层和中腿关系问题。USER的座板/中部灰支撑在D0提议阶段明显收缩；恢复C后蓝表面重新出现，不能误算为灰bulk恢复。'
    }
    stages=[]
    for q in requests:
        key=next((k for k in stage_notes if q['parent_id'].startswith(k)),None)
        if key is None or int(q['noise_index'])!=(1 if key=='943dde' else 0):continue
        for label in LABELS:
            im=OUT/'gallery_full'/label/(q['request_id']+'.png')
            stages.append(dict(request_id=q['request_id'],label=label,reviewer='AI_PROXY',blind=False,
                stages='RAW|D0_PROPOSAL|PRE_GEO|GEO|FINAL',observation=stage_notes[key],
                selection='targeted stage attribution after full RAW/FINAL review; not a prevalence sample',
                image=str(im.relative_to(OUT)),image_sha256=sha(im)))
    assert len(stages)==8
    csvwrite(d/'targeted_stages.csv',stages)
    save(d/'complete.json',dict(status='COMPLETE',reviewer='AI_PROXY_NONBLIND',main_cells=64,main_pages=16,
        oracle_pages=8,continuation_pages=8,continuation_pairs=24,targeted_stage_pages=8,
        total_distinct_reviewed_pages=40,rendered_pages=64,
        caveat='C adoption is a conservative qualitative relation label, not exact shared-edge certification. Metadata part names are not generator inputs; alternative plausible uses and cap ambiguity stay PARTIAL/UNCERTAIN. No unique-source style score.',
        files={str(p):sha(p) for p in d.glob('*.csv')}))
    print('Recorded64 main cells,8 layout comparisons,24 continuation pairs,8 targeted stage pages; no model calls.')

if __name__=='__main__':main()
