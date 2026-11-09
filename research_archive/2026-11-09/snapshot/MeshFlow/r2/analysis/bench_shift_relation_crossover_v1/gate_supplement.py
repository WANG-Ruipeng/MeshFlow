"""Read-only gate/ledger verification and compact sealed-review summaries.

This never runs a gate, revises a visual label, renders an image or calls a
model. Only new supplement files are written after all existing hashes pass.
"""
from collections import Counter
import json

from . import blind_gate as B

A, C = B.A, B.C


def identity(row):
    return {k: row[k] for k in ("request_id", "parent_id", "condition_id", "part_type")}


def subset(rows):
    selected = sorted(rows, key=lambda r: r["request_id"])
    return dict(request_count=len(selected), parent_count=len({r["parent_id"] for r in selected}),
                parents=sorted({r["parent_id"] for r in selected}), requests=[identity(r) for r in selected])


def run(exp=A.DEFAULT):
    exp = C.resolve(exp).resolve()
    base = exp / "blind_review"
    gate_path, complete_path = base / "gate.json", exp / "COMPLETE.json"
    gate, done = C.read(gate_path), C.read(complete_path)
    assert gate["status"] == "FAIL" and done["status"] == "COMPLETE_GATE_STOP_BEFORE_GPU"
    A.verify_hashes(gate["sealed_input_hashes"])
    A.verify_hashes(gate["analysis_files"])
    assert gate["script_sha256"] == C.sha(B.__file__)
    assert done["gate_sha256"] == C.sha(gate_path)
    assert done["report_sha256"] == C.sha(exp / "REPORT.md")
    review_path = base / "review_records/review.json"
    decoded = B.validate_and_decode(exp, review_path)
    assert decoded["status"] == "VALIDATED" and len(decoded["rows"]) == 64
    assert gate["review_sha256"] == decoded["review_sha256"]
    pairs, _, differences, prior_hashes = B.tables(decoded["rows"])
    recomputed = B.gate_from_pairs(pairs)
    for key, value in recomputed.items():
        assert gate[key] == value, key
    assert gate["actual_review_pairs"] == done["actual_review_pairs"] == 32
    assert gate["actual_review_cells"] == done["actual_review_cells"] == 64
    budget_fields = ("new_training_updates", "new_generator_forwards", "new_D0_forwards",
                     "new_RAW", "new_POST", "new_score_rows")
    assert all(done[k] == 0 for k in budget_fields) and done["candidate_sampled"] is False
    assert gate["new_model_calls"] == gate["new_scoring_calls"] == 0
    no_files = [exp / "workflow/state.json", exp / "results/per_output.csv", exp / "gallery/completion.json"]
    assert not any(p.exists() for p in no_files)
    candidate = exp / "evaluation" / A.CANDIDATE
    forbidden = []
    cpu_receipts = []
    patterns = ("calls.jsonl", "trajectory.npz", "raw.npz", "receipt.json", "attempt.json", "failure.json")
    if candidate.exists():
        for child in candidate.iterdir():
            if child.name == "layouts" and child.is_dir():
                cpu_receipts.extend(child.rglob("receipt.json"))
                continue
            if child.name.startswith("sampling_shard_"):
                forbidden.append(str(child))
            elif child.is_file() and child.name in patterns:
                forbidden.append(str(child))
            elif child.is_dir():
                for name in patterns:
                    forbidden.extend(str(p) for p in child.rglob(name))
    assert not forbidden, forbidden
    axes = {}
    for label in B.MODELS:
        rows = [r for r in decoded["rows"] if r["label"] == label]
        assert len(rows) == 32
        axes[label] = {axis: {value: sum(r[axis] == value for r in rows) for value in allowed}
                       for axis, allowed in B.AXES.items()}
    ref = lambda p: p[B.REF + "__C_use"]
    shift = lambda p: p[B.SHIFT + "__C_use"]
    adoption = {
        "strict_replaced_to_positive": subset([p for p in pairs if ref(p) == "REPLACED_OR_IGNORED" and shift(p) == "PLAUSIBLE"]),
        "uncertain_to_positive": subset([p for p in pairs if ref(p) == "UNCERTAIN" and shift(p) == "PLAUSIBLE"]),
        "all_shift_unique_positive": subset([p for p in pairs if ref(p) != "PLAUSIBLE" and shift(p) == "PLAUSIBLE"]),
        "positive_lost": subset([p for p in pairs if ref(p) == "PLAUSIBLE" and shift(p) != "PLAUSIBLE"]),
        "positive_to_uncertain": subset([p for p in pairs if ref(p) == "PLAUSIBLE" and shift(p) == "UNCERTAIN"]),
        "positive_to_replaced": subset([p for p in pairs if ref(p) == "PLAUSIBLE" and shift(p) == "REPLACED_OR_IGNORED"]),
        "both_positive": subset([p for p in pairs if ref(p) == shift(p) == "PLAUSIBLE"]),
    }
    assert len(differences) == 320
    history = {}
    for label in B.MODELS:
        history[label] = {}
        for axis in B.AXES:
            rows = [r for r in differences if r["label"] == label and r["axis"] == axis]
            assert len(rows) == 32
            counts = Counter((r["prior_nonblind_value"], r["current_anonymous_value"]) for r in rows)
            history[label][axis] = dict(compared=32, changed=sum(r["changed"] for r in rows),
                transitions=[dict(previous=a, anonymous=b, count=n) for (a, b), n in sorted(counts.items())],
                prior_counts=dict(Counter(r["prior_nonblind_value"] for r in rows)),
                anonymous_counts=dict(Counter(r["current_anonymous_value"] for r in rows)))
    sources = {str(p): C.sha(p) for p in (gate_path, complete_path, exp / "REPORT.md", review_path, base / "review_seal.json")}
    sources.update(prior_hashes)
    value = dict(schema="BENCH_CROSSOVER_GATE_SUPPLEMENT_V1", status="VERIFIED_GATE_STOP_BEFORE_GPU",
        scope="Existing DEV32 FINAL anonymous ratings only; no rerating, no candidate generation",
        reviewer_kind="AI_PROXY", blind=True, records=64, pairs=32, parents=8,
        gate_status=gate["status"], gate_checks=gate["checks"], counts=gate["counts"],
        five_axis_counts=axes, adoption_transitions=adoption,
        historical_nonblind_differences=dict(compared=320, changed=sum(r["changed"] for r in differences),
                                            per_model_axis=history),
        zero_new_gpu_work_audit=dict(status="PASS", candidate_sampled=False,
            **{k: done[k] for k in budget_fields}, actual_candidate_artifacts=[],
            absent_paths=[str(p) for p in no_files], allowed_cpu_layout_receipts=len(cpu_receipts),
            scope="This experiment's sealed records and artifact audit; not a claim about unrelated GPU use"),
        input_hashes=sources, script_sha256=C.sha(__file__),
        original_gate_review_report_completion_modified=False,
        supplement_model_calls=0, supplement_scoring_calls=0)
    lines = ["# 匿名门判补充复核", "",
        "**门判 FAIL 正确，本轮在 GPU 派发前停止；SHIFT→RELATION 候选未采样，不能称为候选的负生成结果。** "
        "本补充只重算已封存的 32 个 DEV FINAL 配对、64 条 AI_PROXY 匿名记录，不重新看图、改标或改变门槛。", "",
        f"采用 PLAUSIBLE：RELATION {gate['counts'][B.REF]}/32，SHIFT {gate['counts'][B.SHIFT]}/32。总量门通过；"
        f"严格 REPLACED_OR_IGNORED→PLAUSIBLE 只有 {adoption['strict_replaced_to_positive']['request_count']} 个请求、"
        f"{adoption['strict_replaced_to_positive']['parent_count']} 个父，不满足至少 2 父。UNCERTAIN→PLAUSIBLE 不得替代严格改善。", "",
        "| 采用配对变化 | 请求数 | 不同父数 |", "| --- | ---: | ---: |"]
    names = {
        "strict_replaced_to_positive": "明确未采用→采用", "uncertain_to_positive": "不确定→采用",
        "all_shift_unique_positive": "SHIFT 独有采用（前两项合计）", "positive_lost": "原采用丢失",
        "positive_to_uncertain": "采用→不确定", "positive_to_replaced": "采用→明确未采用",
        "both_positive": "共同采用",
    }
    lines += [f"| {names[k]} | {v['request_count']} | {v['parent_count']} |" for k, v in adoption.items()]
    lines += ["", "各集合的精确请求、父 ID、condition_id 和实际 part_type 已写入 summary.json；父数不可跨集合相加。", "",
              "| 轴 / 枚举 | RELATION /32 | SHIFT /32 |", "| --- | ---: | ---: |"]
    for axis, allowed in B.AXES.items():
        for enum in allowed:
            lines.append(f"| {axis}: {enum} | {axes[B.REF][axis][enum]} | {axes[B.SHIFT][axis][enum]} |")
    lines += ["", "**匿名重评并不稳定复现旧非盲标签。** 下表比较同一组旧网格、同一 DEV FINAL 请求；"
              "变化来自评阅者、匿名呈现和比较上下文等因素，不能解释为模型或网格改变，也不能据此单独证明哪一次标签是真值。", "",
              "| 轴 | RELATION 标签变化 /32 | SHIFT 标签变化 /32 |", "| --- | ---: | ---: |"]
    for axis in B.AXES:
        lines.append(f"| {axis} | {history[B.REF][axis]['changed']} | {history[B.SHIFT][axis]['changed']} |")
    lines += ["", f"共 64×5={len(differences)} 项标签比较，{sum(r['changed'] for r in differences)} 项不同。"
              "逐枚举转移和旧/新计数见 summary.json；逐条证据仍使用已封存的 historical_label_differences.csv。"
              "当前配对与旧非盲成功率必须分别陈述，不能跨轮拼成统一成功率。该匿名评阅仍是 AI_PROXY，非独立人类盲测。", "",
              f"账本核对：新训练更新、生成器前向、D0 前向、RAW、POST、新评分均为 0。"
              f"无 workflow/state、采样 shard、真实调用/轨迹或候选评分/图集完成产物；"
              f"{len(cpu_receipts)} 份冻结 CPU layout receipt 属于准备缓存，不能算生成。"
              "已验证原 review、图片/映射、gate、REPORT 与 COMPLETE 哈希及身份一致。", "",
              "正确的有限结论是：SHIFT 在本次匿名采用总量上仍有增量，但严格跨父采用改善未达到预设继续门。"
              "因此不启动交叉采样；这既不否定所有 SHIFT 局部收益，也不回答尚未执行的交叉候选质量。"
              "本补充没有训练、重评分、生成、补评、改门、commit 或 push。", ""]
    dest = base / "gate_analysis"
    C.write_json(dest / "summary.json", value)
    C.write_bytes(dest / "SUPPLEMENT.md", "\n".join(lines).encode("utf-8"))
    # Verify all original evidence again after adding only the two new files.
    A.verify_hashes(sources)
    return dict(status=value["status"], counts=gate["counts"],
                adoption={k: {x: v[x] for x in ("request_count", "parent_count")} for k, v in adoption.items()},
                old_label_changes={l: {a: v["changed"] for a, v in aa.items()} for l, aa in history.items()},
                all_label_changes=sum(r["changed"] for r in differences),
                files={str(p): C.sha(p) for p in (dest / "summary.json", dest / "SUPPLEMENT.md")})


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
