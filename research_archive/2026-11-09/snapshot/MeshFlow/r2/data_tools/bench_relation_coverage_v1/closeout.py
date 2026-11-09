"""Record actual AI_PROXY page observations and seal this CPU investigation."""
from .run import *

NOTES={
 'accepted_targets_all25':'已实看25件完整目标总览。含薄板长椅、箱体、L形、木条、曲面和装饰端架；不能称完全没有复杂来源。7b58a3与c97683外框及下层架十分接近，款式独立性需另核，未判为重复。',
 'train_named_assemblies':'三条PART_ASSEMBLY分别为座面加两墩侧面、顶面加两长墙、顶面加两纵向闭环框。与f495的中间两根短支撑组合不同；其他SEAT_SUPPORT也含组合，不能只按此标签统计全部组合覆盖。',
 'neighbors_35948e':'完整略倾斜座面。异父8967e6和c89d4c也提供相近完整座面关系。已有精确训练条件且有相似示范，不能把n0失败完全解释为覆盖缺失。',
 'neighbors_5dda4e':'输入是一条完整木条的上表面，而不是整块座面。最近筛出的d9de3b为前缘矩形带，整体来源是带背板长椅；另两例为完整板面。相似细条存在，但关系与高度不同。',
 'neighbors_943dde':'蓝色完整外背面近似平板；近邻91e169是底座长墙，其余为背板。这说明相近局部几何可属于不同合理用途，不证明标签可由C唯一确定。',
 'neighbors_9d5608':'蓝色是高背顶部水平盖面，非竖直靠背板。前三近邻是窄座面。用途标签与实际输入面方向需分开，不能要求唯一来源款式。',
 'neighbors_bf504b':'训练L形侧架竖向及横向跨度明显大于2d的小L侧面；筛出的相近数据多为U框或墩侧面。两份噪声成功只证明该熟悉大L条件的局部能力。',
 'neighbors_2d1e5e':'小型L形腿侧面加短上横杆，与最近墩端面/U框/扶手面在结构上不同。最近矩特征距离不构成语义等价。',
 'neighbors_838c48':'训练输入为完整异形顶面及两墩侧面，拓扑位置关系明确；近邻多为座面加端架或侧框。不能用熟悉组合成功证明所有组合泛化。',
 'neighbors_f495bd':'蓝色窄长顶面和中段两根短箱形支撑，来源另有两端框；最近输入多为一侧纵向框或座背组合。未发现这些近邻示范同样的中段支撑关系。',
 'candidate_9e429b':'完整三湾曲面座壳和两端墙可见，旧C却是被高度切断的端墙下半段或极细座面条。优先重提完整曲面/整端墙，不直接放行旧C。',
 'candidate_943dde':'目标有曲面座背过渡，候选SEAT却取平底面，SEAT_BACK也用了底面；仅外背及下部组件入池。应重提真实上表面/内曲面。',
 'candidate_df6fd3':'双面座围绕中央隔背，当前只保留箱体底座；双座面候选存在缺角小缝，质量记录报C非流形顶点。重提连续部位后再验收。',
 'candidate_8967e6':'座面加腿组合视觉有意义，但两个命名相同集合，原质量表报C非流形顶点。不能当成面数容量不足或直接放行。',
 'candidate_dbd069':'座面加四腿组合视觉有意义，但同样有非流形顶点门槛。优先核对选择造成的局部缺面/点接触，不凭整体外形认证拓扑。',
 'candidate_f35697':'来源本身座面大面积空缺，只有窄前后条，不能因端架漂亮就当完整可靠训练目标。保持暂存。',
}

def main():
    manifest=read(OUT/'gallery_manifest_v2.json');review=[]
    assert len(manifest['pages'])==len(NOTES)==16
    for p in manifest['pages']:
        name=Path(p['path']).stem;assert name in NOTES and sha(p['path'])==p['sha256']
        review.append(dict(page=name,path=p['path'],sha256=p['sha256'],reviewer='AI_PROXY_Codex',actually_viewed=True,
                           observation=NOTES[name],no_human_rating=True,source_completion_not_unique_GT=True))
    save(OUT/'visual_review.json',dict(records=review,scope='16 current pages actually viewed; 25-target overview +8 diagnostic nearest pages +6 existing candidate pages +3-assembly page'))
    rows=list(csv.DictReader((OUT/'condition_descriptors.csv').open(encoding='utf-8-sig')))
    facts={}
    for part in ('SEAT_SURFACE','BACK_SURFACE'):
        rr=[r for r in rows if r['split']=='TRAIN' and r['part_type']==part]
        facts[part]=dict(count=len(rr),max_plane_rms_L=max(float(r['C_plane_rms_L']) for r in rr),
            min_normal_abs_y_area=min(float(r['C_horizontal_normal_area']) for r in rr),
            max_normal_abs_y_area=max(float(r['C_horizontal_normal_area']) for r in rr),
            min_centroid_y=min(float(r['C_centroid_y']) for r in rr),max_centroid_y=max(float(r['C_centroid_y']) for r in rr))
    facts['multi_part_named_categories']=dict(conditions=27,samples=3840,fraction=.32,categories=['SEAT_BACK','SEAT_SUPPORT','PART_ASSEMBLY'])
    save(OUT/'derived_facts.json',facts)
    plan=read(OUT/'render_plan.json');ps={p['parent_id'][:6]:p for p in plan['parents']}
    candidates=[]
    definitions=[
      ('a8e987','P1','完整木条组及单根完整木条','目标已验收；当前仅1条支撑条件、480次曝光；补板条输入关系','EXISTING_TRAIN_PARENT'),
      ('7b58a3','P1','单根完整木条与完整端架','已有整组木条和纵框；补整根单木条，保留自然缝隙','EXISTING_TRAIN_PARENT'),
      ('9e429b','P1','完整曲面座壳或整块端墙','目标几何/外形已有通过记录，旧C均被截断或过细；重新提取，不能直接放行','ONE_NEW_SOURCE_CANDIDATE_NOT_READY'),
      ('943dde','P1','曲面座面上表面与内背过渡','已有C选到座底和外背；完整目标可用于重提真实曲面','EXISTING_TRAIN_PARENT'),
      ('df6fd3','P2','两个完整座面表面及组合','当前只有底座；两条候选因C_NONMANIFOLD_VERTICES拦截，需修条件选择','EXISTING_TRAIN_PARENT'),
      ('8967e6','P2','完整座面与支撑框组合','组合有意义，但C_NONMANIFOLD_VERTICES；不靠拆ID或豁免拓扑强放行','EXISTING_TRAIN_PARENT'),
      ('dbd069','P2','完整座面与支撑框组合','同上，单座/背已有；补的是新关系不是新父','EXISTING_TRAIN_PARENT'),
      ('35948e','P2','整条靠背顶盖或完整背板加顶盖','来源已有明确顶盖，可审核新输入；不要强求C必须复制来源款式','EXISTING_TRAIN_PARENT'),
      ('d9de3b','P2','完整背部顶条','已有前缘座面带和背正面，可审核完整顶条','EXISTING_TRAIN_PARENT'),
      ('f35697','HOLD','不直接补入','完整来源座面缺失，须先有可信完整目标；本轮未修','WHOLE_TARGET_REQUIRES_RECONSTRUCTION')]
    for pre,priority,part,reason,status in definitions:
        p=ps[pre];candidates.append(dict(parent_id=p['parent_id'],priority=priority,proposed_input=part,reason=reason,
             current_status=status,target_mesh_path=p['target_mesh_path'],target_mesh_sha256=sha(p['target_mesh_path']),
             new_C_created=False,admitted=False))
    csvout('candidate_actions.csv',candidates)
    sources={}
    for f in ('input_hashes.json','supplement_input_hashes.json'):
        for p,h in json.loads((OUT/f).read_text()).items():assert sha(p)==h,p;sources[p]=h
    for f in ('run.py','supplement.py','closeout.py','inspect_assets.py','resume.ps1'):
        p=Path(__file__).parent/f;sources[str(p)]=sha(p)
    for rel in ('r2/data_tools/bench_user_v2/render.py','r2/pilots/regular_patch_control_v1/depth_gallery.py','r2/pilots/regular_patch_control_v1/common.py',
                'r2/pilots/bench_user_v2_training_v1/data.py','native_t1/docs/control_metrics.md'):
        sources[str(ROOT/rel)]=sha(ROOT/rel)
    save(OUT/'source_hashes.json',sources)
    # All narrative claims are linked to immutable tables / reviewed images.
    deliver={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('ledger.jsonl','delivery_hashes.json','COMPLETE.json')}
    save(OUT/'delivery_hashes.json',deliver)
    save(OUT/'COMPLETE.json',dict(status='COMPLETE_CPU_DATA_INVESTIGATION',sources_verified=len(sources),deliverables=len(deliver),
          current_review_pages=16,TRAIN_conditions=81,DEV_conditions=39,original_training_pool_unchanged=True,
          generated_model_outputs=0,training_updates=0,model_calls=0,new_C=0,new_parents_admitted=0))
    event('CLOSEOUT_COMPLETE',resources=resources());print(json.dumps(dict(status='COMPLETE',sources=len(sources),deliverables=len(deliver))))

if __name__=='__main__':main()
