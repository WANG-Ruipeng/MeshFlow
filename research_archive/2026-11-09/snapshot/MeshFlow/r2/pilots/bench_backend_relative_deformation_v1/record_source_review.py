"""Record root's completed visual inspection; does not generate ratings from metrics."""
from .common import *
from datetime import datetime, timezone

NOTES={
 '27eb79ddf26fd7ac7898580cb9d16864':'原图四个细支撑与扶手完整；三种D0均明显收细/缺失支撑，候选GEO与上轮接近，不能称为完整恢复；侧截面扶手和座下细结构仍破碎。',
 '2d1e5eeb255e79dcab53bf7fa22f3636':'输入细腿及靠背厚度在D0后明显削弱；两适配候选比官方留下更多细腿痕迹，但仍细且不完整；侧截面背部缺口没有消除。',
 '35948e4d06a9b07e8ba8ff316aec8d3d':'厚端板、靠背及座下主体三种均基本保留；候选与上轮宏观相近，座下前/侧截面仍有局部断口。',
 '459f90aa2162b1f1d46c340938e2ff1c':'弧形端架的大体轮廓保留，座下层与C间隔及左侧接续断口仍可见；新候选没有明显超过上轮的宏观修复。',
 '4cd23aae912b7de019fb4103277a6b93':'箱式主体稳定，候选与上轮视觉接近；侧截面上部薄层和箱侧之间仍有小断裂，未看到大规模新破损。',
 '6b894c1c57ba921c98d10ca8df5ecbe7':'L形座背主体与多支撑保留；两适配较官方的细腿表达更实，候选与上轮近似，截面局部缝仍在。',
 '7b58a3e5d7f4dddcad1d789f3b2120d0':'两端板及下层框架保留；新候选与上轮均比官方保住更多圆弧/板条厚度，但上部转弯处截面仍稀疏有缺口。',
 '838c48b41d513a2dd0992406ba4944f5':'双粗支撑与异形台面三种均保留；候选与上轮整体近似，侧截面少量缝隙；仅此图不能认证无交。',
 '91e169ea3ceb587beff42b9e13c388bc':'箱体宏观完整；候选没有显著视觉增益，边缘少量错层/断点仍在，不从表面完整推断几何冲突已减少。',
 '943dde2754ddc7822e8ff3556a90169':'端板和背板宏观保持；原来的靠背/座面有限厚度在提议中变薄，侧截面背部断续，新旧适配都未完全恢复。',
 '9e429b7f77646b4ccc8f8e7f17e9d7b6':'三个座位的曲面和端架在各D0提议中均出现密集裂缝；恢复C修回蓝座面后灰端架仍破碎，候选没有解决此失败。',
 'a8e987a6a6486b962d6c8e51365a5a87':'官方明显丢细腿，两个适配候选的四腿更可见；新候选与上轮接近。侧截面木条由闭合薄矩形变为断续下沿，不能称厚度完全保住。',
 'df6fd36040a5719662701a18243b45d3':'双侧箱体及中央背板稳定，各阶段宏观差别小，边缘有轻微起伏和断口；新候选未显示显著结构升级。',
 'f495bd67af8d9f0cadd1cf65e8dc4de':'官方GEO几乎丢掉整排细支撑；新旧适配均留下部分腿架，但仍细且多段不完整，候选没有恢复到输入支撑厚度。',
 'fd355d06238853c7f0d9805ad6c2059c':'较粗的四腿、高背与扶手保持稳定；候选和上轮接近，背板三角接缝、座下侧截面断点仍可见。'
}

def run():
 path=OUT/'audits/source_visual_review.json'
 if path.exists():return read(path)
 rows=[]
 for parent,note in NOTES.items():
  p=OUT/'source_analysis'/parent/'identity.png'
  rows.append(dict(parent_id=parent,image_path=str(p),image_sha256=sha(p),actually_viewed=True,
     reviewer='root',reviewer_kind='AI_PROXY',blind=False,
     actual_view_source='functions.exec tools.view_image(detail=original), five batches of three full source pages in this conversation',
     note=note))
 r=dict(status='PASS_ACTUAL_VIEW_COMPLETE',pages=15,columns_per_page=7,all_actually_viewed=True,
  viewed_recorded_at_utc=datetime.now(timezone.utc).isoformat(),
  columns=['INPUT','OFFICIAL_SOURCE_PROPOSAL','OFFICIAL_SOURCE_GEO','PREVIOUS_SOURCE_PROPOSAL','PREVIOUS_SOURCE_GEO','CANDIDATE_SOURCE_PROPOSAL','CANDIDATE_SOURCE_GEO'],
  limitation='同实例identity来源恢复的非盲AI图审，非开放生成成功率/独立人评；两视图与原位置前/侧截面。观察笔记不由数值指标推断。',
  rows=rows)
 write(path,r);return r
if __name__=='__main__':print(run()['status'])
