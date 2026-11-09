"""Read-only CPU N-contract/coverage audit; never imports torch or a model.

python -B -m r2.data_tools.bench_three_family_v1.n_budget [--verify]
Completed runs verify their small outputs and protected text/JSON sources only.
This module owns only experiments/r2_bench_three_family_v1/n_budget/.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_three_family_v1/n_budget"
OLD = "experiments/r2_bench_deployment_task_data_v1"
TASKS = OLD + "/dataset/source_tasks_pending_adapter/manifest.json"
REUSED = "experiments/r2_bench_user_relation_rebuild_v1/proposed_merged_manifest.json"
VERSION = "BENCH_THREE_FAMILY_N_CONTRACT_COVERAGE_V1"

# Each fact is tied to a literal source line, not an executed module import.
CHECKS = [
    ("official_config_N", "configs/snet/base-120m-ot-v-bench.yaml", "max_length: 800"),
    ("official_config_bin", "configs/snet/base-120m-ot-v-bench.yaml", "face_bin: 20"),
    ("official_config_no_sequence_PE", "configs/snet/base-120m-ot-v-bench.yaml", "use_dit_like_pe: False"),
    ("local_model_config", "r2/src/meshflow_control/config.py", "hidden_dim=768, num_heads=12, max_length=800"),
    ("official_dataset_strict_upper", "datasets/mesh_dataset.py", "if num_faces > 20 and num_faces < max_face_length:"),
    ("official_dataset_default", "datasets/mesh_dataset.py", "max_face_length=800"),
    ("official_dynamic_padding", "datasets/mesh_dataset.py", "max_len = min(max_len, max_seq_length)"),
    ("official_training_padding", "train.py", "collate_fn=partial(collate_fn, max_seq_length=800)"),
    ("Native_hard_gate", "r2/src/meshflow_control/models/native.py", "((y < 128) | (y > 256))"),
    ("Native_count_bin", "r2/src/meshflow_control/models/native.py", "self.backbone.y_embedder(y // self.backbone.face_bin, False)"),
    ("SEAM_hard_gate", "r2/src/meshflow_control/seam/model_adapter.py", "((y < 128) | (y > 256))"),
    ("SEAM_layout_gate", "r2/src/meshflow_control/seam/slot_data.py", "if not 128 <= n <= 256 or not 0 < k < n or len(e) > n-k:"),
    ("head_feature_scaling", "r2/src/meshflow_control/seam/slot_data.py", "k/256., n/256."),
    ("Bench_layout_gate", "r2/pilots/bench_parts_convergence_v1/data.py", "if not 128 <= n <= 256 or not 0 < k < n or len(edges) > n - k:"),
    ("Bench_empty_boundary_branch", "r2/pilots/bench_parts_convergence_v1/data.py", "seam_slots=np.empty(0, np.int64), bulk_slots=np.arange(k, n, dtype=np.int64)"),
    ("relation_training_gate", "r2/pilots/bench_relation_training_v1/training.py", "if not 128 <= int(s[\"N\"]) <= 256:"),
    ("evaluation_query_gate", "r2/pilots/bench_control_transfer_v1/evaluate.py", "if not 128 <= int(q['N']) <= 256 or int(q['K']) < 2:"),
    ("sampling_dynamic_N", "r2/pilots/bench_control_transfer_v1/sample.py", "initial=np.zeros((n,3,3),np.float32)"),
    ("sampling_dispatch_guarded_adapter", "r2/pilots/bench_control_transfer_v1/sample.py", "v=seam_forward(model,state,t,y,valid,known,anchor)"),
    ("OT_dynamic_free_N", "r2/src/meshflow_control/training/coupling.py", "n = len(data)"),
    ("OT_six_dense_costs", "r2/src/meshflow_control/training/coupling.py", "costs = np.stack([cdist(data.reshape(n, -1)"),
    ("OT_Hungarian", "r2/src/meshflow_control/training/coupling.py", "row, col = linear_sum_assignment(np.min(costs, axis=0))"),
    ("D0_input_gate", "r2/src/meshflow_control/denoise_model.py", "not 1 <= len(raw) <= 800"),
    ("embedding_category_count", "r2/src/meshflow_control/models/backbone/equidit.py", "self.num_classes = max_length//face_bin"),
    ("embedding_allocation", "r2/src/meshflow_control/models/backbone/equidit.py", "nn.Embedding(num_classes + use_cfg_embedding, hidden_size)"),
    ("CFG_index", "r2/src/meshflow_control/models/backbone/equidit.py", "labels = torch.where(drop_ids, self.num_classes, labels)"),
    ("CFG_default_positive", "r2/src/meshflow_control/models/backbone/equidit.py", "class_dropout_prob=0.1"),
    ("conditional_sequence_PE", "r2/src/meshflow_control/models/backbone/equidit.py", "self.pos_embed = nn.Parameter(torch.randn(1, max_length, hidden_dim)"),
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if read(path) != value:
            raise ValueError("Immutable output differs: " + str(path))
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)


def facts():
    found = []
    for name, path, text in CHECKS:
        matches = [dict(line=i, text=line.strip()) for i, line in
                   enumerate((ROOT/path).read_text(encoding="utf-8").splitlines(), 1) if text in line]
        if not matches:
            raise ValueError("Audited code anchor missing: " + name)
        found.append(dict(fact=name, path=path, matches=matches))
    return found


def stats(rows):
    owners = [r for r in rows if int(r["source_owner_count"]) > 0]
    return dict(conditions=len(rows), parents=len({r["parent_id"] for r in rows}),
                visual_families=len({r["family_id"] for r in rows}),
                source_owner_positive_conditions=len(owners),
                source_owner_positive_parents=len({r["parent_id"] for r in owners}),
                source_owner_positive_visual_families=len({r["family_id"] for r in owners}),
                source_owner_edges=sum(int(r["source_owner_count"]) for r in rows),
                closed_C_conditions=sum(int(r["C_open_boundary_count"]) == 0 for r in rows),
                zero_owner_open_C_conditions=sum(int(r["source_owner_count"]) == 0 and
                                                int(r["C_open_boundary_count"]) > 0 for r in rows),
                N_range=[min(int(r["N"]) for r in rows), max(int(r["N"]) for r in rows)] if rows else None,
                K_range=[min(int(r["K"]) for r in rows), max(int(r["K"]) for r in rows)] if rows else None,
                N_within_current_128_256=sum(128 <= int(r["N"]) <= 256 for r in rows),
                model_ready_conditions=sum(bool(r.get("training_ready")) for r in rows))


def small_checks():
    # Pure integer logic tests the code-derived count bin edge, not checkpoint contents.
    assert 800 // 20 == 40 and 799 // 20 == 39
    assert 256 // 20 == 12 and 512 // 20 == 25
    assert 40 == 800 // 20  # CFG slot is num_classes, regular labels are 0..39.
    assert not (20 < 800 < 800) and 20 < 799 < 800
    assert stats([])["conditions"] == 0
    toy = dict(parent_id="p", family_id="f", source_owner_count=0, C_open_boundary_count=0,
               N=300, K=20, training_ready=False)
    result = stats([toy])
    assert result["closed_C_conditions"] == 1 and result["source_owner_positive_visual_families"] == 0
    return dict(integer_boundary_assertions="PASS", coverage_empty_and_closed_fixture="PASS",
                generator_constructions=0, model_forwards=0, checkpoint_loads=0)


def resources():
    import psutil
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        gpu = dict(returncode=q.returncode, stdout=q.stdout.strip(), stderr=q.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu = dict(status="UNAVAILABLE", reason=str(exc))
    return dict(cpu_logical=psutil.cpu_count(), available_RAM_bytes=psutil.virtual_memory().available,
                CPU_percent=psutil.cpu_percent(interval=.1), audit_workers=1, numeric_threads=1,
                GPU_workers=0, GPU_query=gpu,
                scope="Tiny sequential CPU JSON/source audit; nvidia-smi inspection is not GPU computation")


def build():
    source_paths = sorted({p for _, p, _ in CHECKS} | {TASKS, REUSED,
        OLD + "/split_audit/new_data_split.json", str(Path(__file__).relative_to(ROOT)).replace(os.sep, "/")})
    before = {p: sha(ROOT/p) for p in source_paths}
    evidence = facts()
    manifest = read(ROOT/TASKS)
    rows = manifest["rows"]
    accepted = [r for r in rows if r["final_source_task_status"] == "ACCEPT_SOURCE_TASK_PENDING_ADAPTER"]
    train = [r for r in accepted if r["effective_eval_group"] == "TRAIN_KNOWN_C"]
    assert all(r["recipe_id"] == "R1" for r in train)
    train_proposed = [r for r in rows if r["intended_eval_group"] == "TRAIN_KNOWN_C" and r["recipe_id"] == "R1"]
    old = read(ROOT/REUSED)["train"]
    bounds = [(128,256), (257,384), (385,512), (513,799), (800,800)]
    bands = []
    for lo, hi in bounds:
        subset = [r for r in train if lo <= int(r["N"]) <= hi]
        bands.append(dict(N_min=lo, N_max=hi, **stats(subset),
                          condition_ids=[r["condition_id"] for r in subset]))
    counts = dict(all_candidates=stats(rows), accepted_source_tasks=stats(accepted),
                  proposed_train_R1=stats(train_proposed), accepted_train_R1=stats(train),
                  reused_current_pool=dict(conditions=len(old), parents=len({r["parent_id"] for r in old})))
    per = [{k:r[k] for k in ("condition_id","parent_id","family_id","recipe_id","N","K",
                            "C_open_boundary_count","source_owner_count","source_alias_count",
                            "effective_eval_group","final_source_task_status","training_ready")}
           for r in rows]
    result = dict(schema=VERSION, status="COMPLETE_CPU_READ_ONLY", counts=counts,
        accepted_train_R1_N_bands=bands, per_condition=per, code_evidence=evidence,
        scope="Recompute counters from prior saved manifests only; no new geometry qualification, semantic review, layout prediction, or data admission.",
        family_boundary="Prior conservative visual-family IDs; not formal shape clustering or a guarantee of independent geometry.",
        source_owner_boundary="Positive source owner denotes source-supervised connectivity. It does not certify a compatible production C-only layout or generator learnability.",
        current_N_contract=[128,256], contract_changed=False,
        architecture=dict(config_max_length=800, face_bin=20, regular_count_classes=40,
                          regular_count_class_ids=[0,39], CFG_unconditional_class_id=40,
                          N800_class_id=40, official_dataset_strict_range=[21,799],
                          N800_is_regular_condition=False, sequence_position_embedding_enabled=False),
        unknowns=["Actual saved checkpoint behavior outside N128..256 is untested.",
                  "High-N generation, training convergence, D0 quality and resource peaks are unmeasured.",
                  "A larger N protocol requires independent validation of guards, occupancy feature domain, faithful layout, evaluation and resource budget; it is not authorized by this audit."],
        CPU_checks=small_checks(), calls=dict(generator_loads=0, occupancy_loads=0, generator_forwards=0,
            occupancy_forwards=0, D0_forwards=0, training_updates=0, new_data=0),
        accepted_train_R1_owner_positive_families=sorted({r["family_id"] for r in train if r["source_owner_count"]>0}))
    assert before == {p: sha(ROOT/p) for p in source_paths}
    return result, dict(schema=VERSION, sources=[dict(path=p, sha256=h, bytes=(ROOT/p).stat().st_size)
                       for p,h in before.items()], protected_inputs_unchanged=True)


def report(c):
    a, t, p = c["counts"]["accepted_source_tasks"], c["counts"]["accepted_train_R1"], c["counts"]["proposed_train_R1"]
    lines = ["# Bench 三家族小样：N合同与已有覆盖核对", "",
        "**N128–256是当前生产适配与实验合同，不是官方Bench主干的天然256面极限。本轮只读取源码和旧清单，没有改守卫或开训。**", "",
        "## 1. 现有数据实际支持什么", "",
        f"前轮30条候选中，源任务准入{a['conditions']}条；这些只是来源几何/语义准入，模型就绪仍为0。",
        f"真正拟用于新增训练的R1原提议{p['conditions']}条/{p['parents']}父；准入后为 **{t['conditions']}条/{t['parents']}父/{t['visual_families']}个保守视觉家族**。",
        f"其中source-owner>0的只有 **{t['source_owner_positive_conditions']}条/{t['source_owner_positive_parents']}父/{t['source_owner_positive_visual_families']}家族**，共{t['source_owner_edges']}条来源owner边；闭合C为{t['closed_C_conditions']}条，开放但零owner为{t['zero_owner_open_C_conditions']}条。",
        "source-owner是来源监督事实，不等于生产C-only预测已通过。R2同父预留及适配未见父没有计入训练。闭合C可走既有Bench空布局/bulk分支，不需要新SEAM网络，但并不直接增加接续监督覆盖。", "",
        "| 准入TRAIN R1的N档 | 条件 | 父 | 正owner家族 |", "|---|---:|---:|---:|"]
    for b in c["accepted_train_R1_N_bands"]:
        lines.append(f"| {b['N_min']}–{b['N_max']} | {b['conditions']} | {b['parents']} | {b['source_owner_positive_visual_families']} |")
    lines += ["", "家族计数复用已冻结的保守视觉分组，不把不同UID自动认证为独立结构。旧26父91C只是复用背景，没有算成新增。", "",
        "## 2. 实际代码边界", "",
        "- Native forward、SEAM adapter、边槽数据构造、RELATION训练器以及旧Bench评价query入口均显式限制128≤N≤256；当前高面任务不能直接送入这些入口。",
        "- 官方和当前主干配置均为max_length=800、face_bin=20、关闭序列位置embedding。官方Dataset筛的是20<N<800，padding按batch实际长度生成、上限800。",
        "- 40个正常面数类别索引为0..39；CFG无条件索引为40。N=800时800//20=40，会占用无条件槽。因此不能写‘官方正常支持N≤800’；799与800必须区分。",
        "- SEAM小占用头的特征仍用K/256和N/256。放宽N不只是删一条断言，会改变已训练头的特征范围。",
        "- bulk OT按实际free面数构建6个密集代价矩阵并作Hungarian匹配，没有256常量上限；代价矩阵随free面数平方增大，实际运行时间仍未知。",
        "- 采样分配按layout.N动态建数组，模型调用仍受Native/SEAM守卫；D0公共输入验证允许1..800，但这不证明高N去噪质量或显存已验证。",
        "- 当前源码不显示超过256必需新增可学习参数；这只是静态形状判断，未加载权重、未验证高N状态兼容或行为。", "",
        "具体路径、原行号和匹配文本完整保存于coverage.json/code_evidence；source_hashes.json锁定本轮读取版本。", "",
        "## 3. 对当前小样的含义", "",
        "维持当前128–256合同；三家族制作若能保住完整目标且通过C-only合同，就得到可实际推进的低面证据。若为了256面必须丢支撑或部件，应明确失败，不做破坏性填数。",
        "同时避免把当前范围失败说成模型架构不可能支持该结构。若以后决定扩大N，另立有界合同，独立核对小头、匹配、mask/padding、评价和资源；本核对不自动放行高N训练或推理。", "",
        "## 4. 恢复与调用账本", "",
        "`python -B -m r2.data_tools.bench_three_family_v1.n_budget --verify`", "",
        "已完成时默认执行同样的只读hash/计数验证。源清单与生产代码不改；1个CPU worker、0个GPU worker。生成器/占用头/D0加载与前向、训练更新、新数据生成均为0。", ""]
    return "\n".join(lines)


def verify():
    done = read(OUT/"COMPLETE.json")
    for name, expected in done["outputs"].items():
        if sha(OUT/name) != expected:
            raise ValueError("Output hash changed: " + name)
    hashes = read(OUT/"source_hashes.json")
    for row in hashes["sources"]:
        if sha(ROOT/row["path"]) != row["sha256"]:
            raise ValueError("Protected source changed: " + row["path"])
    fresh, _ = build()
    if fresh != read(OUT/"coverage.json"):
        raise ValueError("Counter/evidence recomputation differs")
    return dict(status="VERIFIED", protected_sources=len(hashes["sources"]), outputs=len(done["outputs"]),
                small_json_recomputation="PASS", model_calls=0,
                accepted_train_R1=fresh["counts"]["accepted_train_R1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify or (OUT/"COMPLETE.json").exists():
        print(json.dumps(verify(), ensure_ascii=False)); return
    tick = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    c, hashes = build()
    save(OUT/"coverage.json", c)
    save(OUT/"source_hashes.json", hashes)
    text = report(c)
    target = OUT/"REPORT.md"
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise ValueError("Immutable report differs")
    else:
        target.write_text(text, encoding="utf-8", newline="\n")
    resource_file = OUT/"resources.json"
    if not resource_file.exists():
        save(resource_file, resources())
    save(OUT/"COMPLETE.json", dict(schema=VERSION, status="COMPLETE_CPU_READ_ONLY",
        finished_UTC=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-tick,
        outputs={p:sha(OUT/p) for p in ("coverage.json","source_hashes.json","REPORT.md","resources.json")},
        counts=c["counts"], calls=c["calls"], scientific_contract_changed=False))
    print(json.dumps(verify(), ensure_ascii=False))


if __name__ == "__main__":
    main()
