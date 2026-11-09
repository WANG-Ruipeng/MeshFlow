from pathlib import Path
root=Path('/mnt/e/MeshFlow')
p=root/'README.md'
body=p.read_text(encoding='utf-8')
line='> **R2 condition-training recipe:** Full-course C20/C40 mixing + HYBRID FM now has a separate [r2/](r2/README.md) source entry. See the [training directory map](r2/docs/training_structure.md) for the distinction between original unconditional training, legacy N112 training, START sampling and R2. Weights and frozen data remain external.\n\n'
if line not in body: p.write_text(line+body,encoding='utf-8')
p=root/'WORKSPACE.md'
body=p.read_text(encoding='utf-8-sig')
entry="""## 2026-10-09 R2 代码独立整理（本地，未 commit/push）

用户确定下一步研究配方为 R2 全程混合 C20/C40 + 全程 HYBRID，并选择现有 GitHub 仓库内独立的 r2/ 目录。使用说明见 [r2/README.md](r2/README.md)，三条训练/采样路径见 [训练结构](r2/docs/training_structure.md)。

实际 GitHub master 核查为 8961b7a41fc153b5b45f0c8903370c6bfd1cf04f，与本地 HEAD 一致；此前 R2 位于无提交、无 remote 的 E:/MeshFlow-Control。此次复制计算依赖并保留其源码字节，新增只公开 R2 的入口；旧项目及外部权重/数据/RAW 不修改。主实验和确认实验两套 R2 均保留，没有自动选定默认终点，native_t1 的 START 采样默认继续保留。

当前 R2 训练接口仍绑定历史 26 父条件池、6000 步、20%/40% 面数占比；新 C 构造、洗数据与追加训练尚未执行。此轮只做 CPU 测试、权重严格加载和冻结输入抽查，不代表重训或生成等价性已经验证。证据位于 native_t1/maintenance/r2_port_20261009/，公开验证说明见 r2/docs/validation.md。

"""
if '## 2026-10-09 R2 代码独立整理' not in body:
    first,rest=body.split('\n',1)
    p.write_text(first+'\n\n'+entry+rest.lstrip('\n'),encoding='utf-8')
print('Updated root README and local WORKSPACE.')
