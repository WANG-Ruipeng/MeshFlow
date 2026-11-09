"""Record the completed image review and failed data-readiness gate; no dispatch."""
from __future__ import annotations
import numpy as np
from . import common as C


def run():
    review_path=C.OUT/'visual_review/review_backend_data.json'
    review=C.read(review_path)
    records=review['records']
    expected={r['request_id'] for r in C.read(C.OUT/'data/manifest.json')['tasks']}
    assert len(records)==12 and {r['request_id'] for r in records}==expected
    for r in records:
        assert r['actually_viewed'] is True
        for image in r['image_sources']:
            assert C.sha(image['path'])==image['sha256']
    for name in ('independent_target_audit','independent_result_audit','independent_topology_audit'):
        assert C.read(C.OUT/f'audits/{name}.json')['status']=='PASS'
    a=C.read(C.OUT/'analysis/COMPLETE.json')
    assert a['requests_with_EDIT']==6 and a['readiness_gate_required_requests']==7
    root_observations=[]
    for rid,images,note in [
        ('91e169ea3ceb587beff42b9e13c388bc__SEAT_SURFACE_1__V2__n1',
         ['matched_control.png','comparison.png','patch_zoom_002.png','patch_zoom_012.png'],
         '两种同链FINAL在整物与中心截面上没有明显宏观差异；C仍是箱体座面上表面。局部放大里顶缘变化很小，不能单凭图片认证精确相交消失，也不能据数字把图审改为明显改善。拒绝提案与接受提案均已实际查看。'),
        ('838c48b41d513a2dd0992406ba4944f5__SEAT_SURFACE_1__V2__n0',
         ['matched_control.png'],
         '同链两FINAL均保留板面与粗支撑，整物形状和中心截面基本相同；没有可辨的新增主体损伤，也没有明确的局部视觉升级。'),
        ('2d1e5eeb255e79dcab53bf7fa22f3636__SEAT_SURFACE_1__n0',
         ['comparison.png'],
         'RAW到候选目标没有修正宏观层间关系；蓝C仍在灰色主座面下方。两侧D0系列也不能从该图证明输入已被正确采用。该样本不是本轮小位移顶缘规则的正例。')]:
        paths=[C.OUT/'gallery'/rid/x for x in images]
        root_observations.append(dict(request_id=rid,actually_viewed=True,
            image_sources=[dict(path=str(p),sha256=C.sha(p)) for p in paths],notes=note))
    root_path=C.OUT/'visual_review/review_root_supplement.json'
    C.write(root_path,dict(reviewer='AI_PROXY_root',blind=False,
        scope='Actual supplementary viewing of six current images in three requests, not a second full-panel review',
        records=root_observations,ratings_inferred_from_metrics=False))
    manual=C.OUT/'visual_review/manual_KEEP_annotations.json'
    assert manual.is_file(), 'The separate actual support annotation record must be completed first'
    annotations=C.read(manual)
    assert annotations['manual_KEEP_corners']==len(annotations['rows'])==12
    for r in annotations['rows']:
        p=C.arrays(C.OUT/'targets'/r['request_id']/'target.npz')
        i,j=r['raw_slot_id'],r['raw_canonical_corner_id']
        assert np.array_equal(p['raw'][i,j],np.asarray(r['raw_xyz_model'],dtype=np.float32))
        assert np.array_equal(p['target'][i,j],p['raw'][i,j])
        assert p['corner_state'][i,j]==0 and not p['loss_mask'][i,j]
        for image in r['evidence']:
            assert image['actually_viewed'] and C.sha(image['path'])==image['sha256']
    visfiles={str(p):C.sha(p) for p in sorted((C.OUT/'visual_review').rglob('*')) if p.is_file()}
    C.write(C.OUT/'visual_review/COMPLETE.json',dict(status='COMPLETE',requests=12,
        review_type='Nonblind AI_PROXY actual image review; no independent human claim',
        root_supplement_requests=3,root_supplement_images=6,files=visfiles,
        original_target_masks_unchanged=True,separate_manual_KEEP_corners=12,training_gate_relaxed=False))
    C.write(C.OUT/'decision.json',dict(status='NO_TRAINING_DATA_GATE_FAILED',
        headline='局部坐标修复能消除部分真实穿刺，但这版数据尚不足以启动后端训练。',
        explanation='固定12请求中，6份获得非空EDIT，低于事先冻结的至少7份门槛。完整处理链的保守数值资格为4/12；新增最终修复集中在91e两份输出，伴随未获得对照后处理新增的5条合法F/F连接。没有为了启动训练而扩大半径、替换样本或把UNKNOWN当监督。',
        visual_summary='已实际查看固定12请求的多视角、同位置截面、支撑截面、状态图和局部提案图。整体形状变化很小，当前图片不足以把微小编辑判为明确的视觉升级；2d的宏观错层未解决。a8/27的已有细支撑保留有价值，但应区分绕过D0的保形收益与本轮编辑增量。非盲AI_PROXY逐例记录与少量独立KEEP角点注释保留在visual_review；后者不覆盖原目标mask、不增加EDIT请求数。',
        requests_with_EDIT=6,numerically_qualified_requests=4,required_requests=7,
        qualification_note='Gate fails even under the optimistic upper bound counting all six nonempty EDIT requests; no post-hoc visual threshold is needed to stop training.',
        training_decision='没有启动训练、前向或新生成，没有新增权重。用户授权的是值得训练时继续；这版数据未通过已冻结的准备门槛，所以保留真实局部对与反例后收口。',
        next_priority='本轮应保留的正证据是：小幅、明确的真实C/F穿刺可以用同槽坐标修复，且未靠删面改善冲突。它尚不能支撑普遍的后端训练收益；安全通过的a8/838c修复多为原GEO已能做到，91e的额外收益又与下游连接竞争。因此暂不追加RELDEF或训练这版自动标签。若继续投入，应先取得直接改善最终残余缺陷、且保留同链连接的可信修复目标；对于2d式主体错层，仍应放在生成阶段的条件采用问题中处理。当前证据不支持把所有失败都转成删面/增面需求，也不构成D0或坐标修复的理论上限。',
        model_calls=0,training_updates=0,new_weights=0,
        evidence_files={str(C.OUT/'analysis/COMPLETE.json'):C.sha(C.OUT/'analysis/COMPLETE.json'),str(review_path):C.sha(review_path)}))
    print('ACTUAL_REVIEW_AND_NO_TRAINING_DECISION_RECORDED')


if __name__=='__main__':run()
