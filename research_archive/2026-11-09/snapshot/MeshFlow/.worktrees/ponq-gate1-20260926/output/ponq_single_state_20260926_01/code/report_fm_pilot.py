"""Post-only closed-set report and unselected sample montage."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from common import load_mesh

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(root):
    root=Path(root).resolve();post=root/'post_analysis';post.mkdir(exist_ok=True)
    rows=read(root/'LEDGER.json');summary=read(root/'SUMMARY_FM.json');protocol=read(root/'RUN_PROTOCOL.json')
    arms=['direct','transport_residual'];numbers={};traces={}
    for arm in arms:
        rr=[r for r in rows if r['arm']==arm];result=read(root/'training'/arm/'RESULT.json')
        tr=[json.loads(line) for line in (root/'training'/arm/'TRAIN_TRACE.jsonl').read_text().splitlines()]
        traces[arm]=tr;faces=[r['faces'] for r in rr if r.get('faces') is not None]
        q=[r['geometry_vs_site_selected_target']['rms_pct_bboxdiag'] for r in rr if r.get('geometry_vs_site_selected_target')]
        numbers[arm]=dict(valid=sum(r['valid'] for r in rr),denominator=len(rr),completed_steps=result['completed_steps'],
            training_seconds=result['training_elapsed_seconds'],sampling_seconds=summary['by_arm'][arm]['timing']['sampling_32_seconds'],
            extraction_complete=sum(r.get('extraction_completed',False) for r in rr),failures=dict(Counter(r['status'] for r in rr)),
            last100_loss_mean=float(np.mean([r['loss'] for r in tr[-100:]])),
            last100_block_mse_mean={k:float(np.mean([r[k] for r in tr[-100:]])) for k in ['x_mse','n_mse','F_mse']},
            faces_min_median_max=[min(faces),float(np.median(faces)),max(faces)] if faces else None,
            surface_rms_to_site_selected_target_mean=float(np.mean(q)) if q else None,surface_metric_denominator=len(q),
            site_cd_mean=summary['by_arm'][arm]['nearest_target_site_cd_mean'],
            geometry_failures=[dict(sample_id=r['sample_id'],status=r['status'],nonmanifold_vertices=r.get('nonmanifold_vertices'),
                nonmanifold_edges=r.get('nonmanifold_edges'),beta=[r.get('beta'+str(k)) for k in range(3)],
                exact_intersection_pairs=r.get('exact_intersection_pairs'),error=r.get('error')) for r in rr if not r['valid']])
    keyed={(r['arm'],r['sample_id']):r for r in rows}
    paired=Counter()
    for i in range(32):paired[str((bool(keyed['direct',i]['valid']),bool(keyed['transport_residual',i]['valid'])))]+=1
    shared=len(traces['direct'])==len(traces['transport_residual']) and all(a['batch_sha256']==b['batch_sha256'] for a,b in zip(traces['direct'],traces['transport_residual']))
    result=dict(by_arm=numbers,paired_validity=dict(paired),training_batches_identical=shared,
        source_files=[dict(path=str(p),sha256=sha(p)) for p in [root/'LEDGER.json',root/'SUMMARY_FM.json',root/'RUN_PROTOCOL.json',Path(__file__)]])
    dump(post/'RESULT_NUMBERS.json',result)
    fig,axes=plt.subplots(2,8,figsize=(20,6),subplot_kw={'projection':'3d'})
    meshes={};limit=.5
    for arm in arms:
        for i in range(8):
            p=root/'samples'/f'{arm}_{i:03d}'/'mesh.obj'
            if p.exists():
                v,f=load_mesh(p);meshes[arm,i]=(v,f);limit=max(limit,float(np.max(abs(v))))
    for a,arm in enumerate(arms):
        for i in range(8):
            ax=axes[a,i];r=keyed[arm,i]
            if (arm,i) in meshes:
                v,f=meshes[arm,i];tris=v[f];normal=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-30)
                brightness=.45+.45*abs(normal@np.array([.3,.5,.8124]));base=np.array([.28,.59,.82]) if a==0 else np.array([.85,.48,.24])
                ax.add_collection3d(Poly3DCollection(tris,facecolors=brightness[:,None]*base[None],edgecolors=(0,0,0,.16),linewidths=.15))
            ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit);ax.set_zlim(-limit,limit);ax.set_box_aspect((1,1,1));ax.view_init(22,35);ax.set_axis_off()
            ax.set_title(f'{arm}\nID {i} | valid={r["valid"]} | F={r.get("faces","NA")}',fontsize=8)
    fig.suptitle('First 8 shared prior samples, no selection; 3 known parents / 24 target states; 3000 updates per arm',fontsize=12)
    fig.tight_layout();fig.savefig(post/'samples_first8.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,key in zip(axes,['x_mse','n_mse','F_mse']):
        for arm in arms:
            tr=traces[arm];points=[tr[i:i+100] for i in range(0,len(tr),100)]
            ax.plot([p[-1]['step'] for p in points],[np.mean([x[key] for x in p]) for p in points],label=arm)
        ax.set_title(key+' (100-step mean)');ax.set_xlabel('optimizer updates');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(post/'training_losses.png',dpi=150);plt.close(fig)
    lines=['# 累积传输之后的小型闭集 FM 对照','',
        '已从公共高斯先验实际训练和自由采样。目标只有3个已见父对象的24个相关状态；本轮不是独立泛化、完整生成质量或MeshFlow胜负结论。','',
        '|项目|直接联合状态 FM|传输＋可学习修正 FM|','|---|---:|---:|']
    fields=[('训练更新','completed_steps'),('合格网格 / 32','valid'),('训练实耗秒','training_seconds'),('32个采样积分实耗秒','sampling_seconds'),('最后100步平均loss','last100_loss_mean'),('面数最小/中位/最大','faces_min_median_max')]
    for label,key in fields:lines.append('|'+label+'|'+'|'.join(str(numbers[a][key]) for a in arms)+'|')
    lines+=['',f'逐步训练样本/噪声/时间/匹配哈希全部一致：{shared}。两臂同一初始参数、相同3000次更新，计算路径不同，不能称相同墙钟成本。',
        '',f'公共32噪声的有效性配对（direct, transport）：{dict(paired)}。所有失败保留在完整分母中。',
        '',f'编码目标回放24/24通过；float32目标回放{read(root/"FLOAT32_ORACLE.json")["valid_count"]}/24通过。原法向幅度保留，没有单位化、位置裁切或Q修复。',
        '', '![固定前8个样本](samples_first8.png)','', '![训练损失](training_losses.png)','',
        '两臂使用相同置换等变DeepSets、坐标Hungarian配对、标准化线性FM概率路径和总速度监督。传输项由当前预测位置速度估计局部速度梯度，再作用于完整中心quadric因子；修正头与传输项相加后接受监督。',
        '', '训练目标来自当前曲面的reference-assisted refit参数。采样前向不读取对象ID、目标状态、参考表面或解析变形；参考数据仅用于采样结束后的闭集最近邻诊断。',
        '', '几何距离先按256元素点集距离从24目标中选择一个对象状态，再测输出到所选目标网格的表面距离。它不是独立真值，也不是全部目标表面距离的最小值。RESULT_NUMBERS保存可评估分母，不能将成功子集距离解释成全部生成质量。',
        '', '只用一个训练种子和一个最终检查点。3,000步失败不能否定整个FM方向，成功也不能证明传输机制已经具有稳定收益。正式推进需要可归因的增量与独立对象验证。','']
    (post/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);main(ap.parse_args().root)