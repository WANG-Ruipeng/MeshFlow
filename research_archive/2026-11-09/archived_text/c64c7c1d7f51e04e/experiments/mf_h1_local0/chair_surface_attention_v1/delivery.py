"""Readable evidence index and small source snapshot for this gate-stopped run."""
import csv
import json
from pathlib import Path
import shutil
from .common import *


def finish(out=OUT):
    out=Path(out);gate=read_json(out/'gate_results.json');accounting=read_json(out/'accounting.json')
    numeric=read_json(out/'tests/relations/relations_gate_summary.json')
    own=read_json(out/'registration/implementation_hashes.json');protected=read_json(out/'registration/source_hashes_before.json')
    scorer_sources=read_json(CHAIR/'protocol.json')['source_scorer_identity']
    freeze_json(out/'registration/additional_historical_scorer_hashes.json',scorer_sources)
    protected={**protected,**scorer_sources}
    source_hashes={}
    for rel,expected in sorted({**{k:v for k,v in protected.items() if k.endswith('.py')},**own}.items()):
        src=ROOT/rel;assert digest(src)==expected
        dst=out/'source_code'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copyfile(src,dst)
        assert digest(dst)==expected;source_hashes[rel]=expected
    atomic_json(out/'source_code/source_hashes.json',source_hashes)
    # The plot reads retained differences only, never recomputes any quadrature.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (out/'tests/relations/relations_real_pair_errors.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.8),constrained_layout=True)
    fields=('mu_L2','covariance_Frobenius','compressed_mass_abs')
    labels=('Mean vector L2 difference','Covariance Frobenius difference','Compressed mass difference')
    for ax,field,label in zip(axes,fields,labels):
        for fixture in numeric['fixtures']:
            selected=[r for r in rows if r['fixture_id']==fixture['fixture_id'] and r['scope']=='ordinary_gate']
            ts=sorted({float(r['t']) for r in selected})
            maxima=[max(float(r[field]) for r in selected if float(r['t'])==t) for t in ts]
            ax.plot(ts,maxima,marker='o',label=fixture['fixture_id'])
        ax.axhline(.01,color='black',linestyle='--',label='Frozen gate 0.01')
        ax.set_yscale('log');ax.set_xlabel('Frozen path time t');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('192-node FP32 vs 12,288-node FP64: retained ordinary pairs\nDense-reference convergence fails one registered subset; these are differences, not certified integration errors',fontsize=10)
    (out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures/quadrature_gate.png',dpi=160);plt.close(fig)
    checkpoint_manifest=dict(source=read_json(out/'registration/start_identity.json'),endpoints=[dict(arm=a,status='NOT_RUN',
        reason='Frozen gate failure prohibits every formal training arm',surface_step=0,conditional_total_step=5000,
        fixed_endpoint=None) for a in ARMS],large_weights_copied=0)
    atomic_json(out/'checkpoint_manifest.json',checkpoint_manifest)
    diagnostic=read_json(out/'numerical_failure_analysis.json') if (out/'numerical_failure_analysis.json').exists() else None
    numeric_table=['|Fixture|N/K|mu L2 max|Covariance max|Mass max|Reference converged|',
                   '|---|---:|---:|---:|---:|---|']
    for f in numeric['fixtures']:
        e=f['ordinary'];numeric_table.append(f"|{f['fixture_id']}|{f['identity']['N']}/{f['K']}|{e['mu_L2']:.8f}|{e['covariance_Frobenius']:.8f}|{e['compressed_mass_abs']:.8f}|{f['reference_convergence_pass']}|")
    supplement=['','## 实际门禁失败证据','',
        '本次正式 optimizer 更新为0，三臂均未训练。192点与12288点的差异在四个真实训练任务均未达到冻结阈值；ordinary_40的参考子集也未达到0.001收敛要求。因此这些数值不是已认证的真实积分误差界。其余三个参考子集通过，不等于覆盖了全空间。','',
        *numeric_table,'','![按时间和任务的求积差异](figures/quadrature_gate.png)','',
        '逐query/known的8576行差异、96行参考收敛差异、原始冻结C和查询均已保留。任何求积点数、阈值、带宽、自然N/K均未调整。真实模型预检只用于接线与工程成本核验，不能覆盖或推翻这次准确度失败。','',
        'CPU隔离小模块测试有1次SGD测试更新和6次小模块反向，单独计账；它们不是Chair训练更新。关系单测另有2次autograd检查。正式训练、完整新采样、I/P交叉前向和真实SURFACE轨迹诊断均未执行。','',
        '下一步优先检验：固定窄带宽下，较长三角边上的离散节点是否不足以解析核权重在边/角附近的集中。先对本次冻结失败输入建立收敛的积分参考，再检验保持同一关系定义的数值实现能否同时满足准确度与成本门槛；本轮不执行该后续，不追加点数扫描或训练。有限表面与质心矩的质量比较当前均为NA，尚不能据此决定采用其中一个。','',
        '完整失败定位见 [NUMERICAL_FAILURE_ANALYSIS.md](NUMERICAL_FAILURE_ANALYSIS.md)；源码及原评分代码见 [source_code/source_hashes.json](source_code/source_hashes.json)。','']
    report=out/'CHAIR_FINITE_SURFACE_ATTENTION_REPORT.md'
    with report.open('a',encoding='utf8') as f:f.write('\n'.join(supplement))
    (out/'RUN_AND_RESUME.md').write_text('''# Run and resume

Run from WSL using the existing environment:

```bash
cd /mnt/e/MeshFlow
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.chair_surface_attention_v1.run_all --resume
```

The controller reuses durable completed stage results. An attempted failed or interrupted scientific call is never replayed automatically. The current numerical gate is IMPLEMENTATION_BLOCKED; resume verifies identities and rebuilds the report only, and cannot start any formal training arm. A numerical redesign requires a separately authorized protocol, not editing this frozen run.

The one GPU audit worker acquires `/tmp/wrp_meshflow_jedge5_diag_gpu0.lock` before CUDA initialization. No unrelated process is killed and the lock is never deleted. Real forward/backward calls are reserved in ledger JSONL before execution, including failed calls. CPU module tests have separate attempt records. Known old checkpoint/raw files are references only.

Formal training code is implemented but NOT_RUN. No new fixed endpoint or generated SURFACE trajectory exists. All three arms' training, quality comparisons and trajectory quadrature remain blocked by the frozen gate. START uses 138 independently verified old raw files and six original inapplicable requests, with no replay.
''',encoding='utf8')
    (out/'DELIVERY_INDEX.md').write_text('''# Evidence index

- `CHAIR_FINITE_SURFACE_ATTENTION_REPORT.md`: results, limitations and one next hypothesis.
- `protocol.json`, `protocol.sha256`, `registration/`: exact source identity, 576 requests, frozen training plan, real C/query NPZ fixtures, initial resources and source hashes.
- `tests/relations/`: synthetic tests, 8576 pair differences and 96 reference convergence rows.
- `tests/attention/`: literal/optimized algebra, gradients, nonzero permutation/padding and isolated one-update test.
- `precheck/`, `resources/`, `logs/precheck.log`: real model zero-exit, nonzero permutation, padding and measured complete model cost.
- `NUMERICAL_FAILURE_ANALYSIS.md`, `.json`: retained-error analysis, no quadrature rerun.
- `evaluation/START/`: verified historical raw references, per-output scoring, frozen per-point distances and same-GT angle references.
- `paired_results/`: 576 registered output rows and all comparisons; unrun arms remain blocked/NA, with full registered matrix slots preserved.
- `accounting.json`, `ledger/`, `controller/`, `stages/`: actual calls, failures, elapsed time and recovery status.
- `checkpoint_manifest.json`: exact old source and explicit absent new endpoints; no copied old large weights.
- `source_code/`, `code.diff`: actual new implementation and protected scoring source files with SHA256.
- `RUN_AND_RESUME.md`: single recoverable entry and permanent gate stop for this frozen run.

No additional handoff ZIP is made, as requested by this experiment.
''',encoding='utf8')
    atomic_json(out/'delivery_hashes.json',{p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='delivery_hashes.json' and not p.name.endswith('.lock')})
    return dict(source_files_copied=len(source_hashes),new_large_checkpoints=0,figure='figures/quadrature_gate.png')
