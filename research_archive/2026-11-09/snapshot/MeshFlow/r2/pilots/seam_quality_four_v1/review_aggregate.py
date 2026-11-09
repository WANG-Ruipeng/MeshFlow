"""Validate and aggregate existing AI_PROXY records; no new evaluation or model calls."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experiments/r2_seam_quality_four_v1"
OUT = RUN / "evaluation"
FIELDS = [
    "C_adoption", "whole_body_complete", "support_missing",
    "fragments_or_folded_layers", "visible_cracks", "visible_join",
]
SOURCE_NAMES = ("3f", "73", "85", "8c")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def jbytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_once_or_verify(path: Path, contents: bytes) -> str:
    if path.exists():
        check(path.read_bytes() == contents, f"Refuse changed existing output: {path}")
        return "VERIFIED_EXISTING"
    with path.open("xb") as handle:
        handle.write(contents)
    return "WRITTEN_NEW"


def main() -> None:
    schema_path = OUT / "review_schema.json"
    manifest_path = RUN / "eval_manifest.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    columns = schema["csv_columns"]
    labels = schema["labels"]
    models = schema["model_order"]
    stages = schema["stages"]
    check(len(columns) == 20, "Expected frozen 20-field schema")
    check(labels == ["YES", "NO", "UNCERTAIN"], "Label contract changed")
    check(models == ["SEAM200", "LONG", "ANTI_COVER", "LOCAL_NOISE"], "Model order changed")
    check(stages == ["RAW", "POST"], "Stage contract changed")
    requests = {r["request_id"]: r for r in manifest["requests"]}
    check(len(requests) == len(manifest["requests"]) == 16, "Expected 16 unique frozen requests")
    check(len({r["parent_id"] for r in requests.values()}) == 4, "Expected 4 parents")
    check(all(r["split"] == "TRAIN" for r in requests.values()), "Unexpected split")
    expected = {(r, s, m) for r in requests for s in stages for m in models}
    protected = {relative(p): sha(p) for p in (schema_path, manifest_path, Path(__file__))}
    rows = []
    original_rows = {}
    provenance = []
    for name in SOURCE_NAMES:
        path = OUT / "reviews" / f"{name}.csv"
        note_path = OUT / "reviews" / f"{name}.md"
        protected[relative(path)] = sha(path)
        protected[relative(note_path)] = sha(note_path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            check(reader.fieldnames == columns, f"Schema/order mismatch: {path}")
            local_rows = list(reader)
        check(len(local_rows) == 32, f"Expected 32 source rows: {path}")
        check(len({r["parent_id"] for r in local_rows}) == 1, "Mixed parent in one source review")
        for row in local_rows:
            key = (row["request_id"], row["stage"], row["model"])
            check(key in expected, f"Unregistered identity: {key}")
            check(key not in original_rows, f"Repeated identity: {key}")
            req = requests[row["request_id"]]
            check(row["parent_id"] == req["parent_id"], f"Wrong parent: {key}")
            check(row["noise_index"] == str(req["noise_index"]), f"Wrong noise: {key}")
            check(row["review_status"] == "ACTUALLY_VIEWED", f"Not actually viewed: {key}")
            check(row["review_kind"] == "AI_PROXY", f"Not AI_PROXY: {key}")
            check(row["blinding"] == "NONBLIND_MODEL_LABELS_VISIBLE", f"Wrong blinding: {key}")
            check(row["reviewer"].startswith("AI_PROXY_"), f"Missing reviewer: {key}")
            check(all(row[f] in labels for f in FIELDS), f"Unexpected label: {key}")
            check(all(row[f] for f in ("visible_observations", "change_vs_SEAM200", "uncertainty")), f"Missing notes: {key}")
            image_path = OUT / "gallery" / f"{row['request_id']}_{row['stage']}.png"
            check(row["image_path"] == relative(image_path), f"Wrong image identity: {key}")
            image_hash = sha(image_path)
            check(row["image_sha256"] == image_hash, f"Image hash mismatch: {key}")
            protected[relative(image_path)] = image_hash
            render_path = image_path.with_suffix(".json")
            render = json.loads(render_path.read_text(encoding="utf-8-sig"))
            check(render["request_id"] == row["request_id"] and render["stage"] == row["stage"], "Render identity mismatch")
            check(render["image_sha256"] == image_hash, "Render receipt image hash mismatch")
            protected[relative(render_path)] = sha(render_path)
            original_rows[key] = dict(row)
            rows.append(row)
        provenance.append({"source": relative(path), "sha256": sha(path), "rows": len(local_rows),
                           "reviewer": sorted({r["reviewer"] for r in local_rows}),
                           "notes": relative(note_path), "notes_sha256": sha(note_path)})
    check(set(original_rows) == expected, "Missing frozen identities")
    order = {req_id: i for i, req_id in enumerate(requests)}
    rows.sort(key=lambda r: (order[r["request_id"]], stages.index(r["stage"]), models.index(r["model"])))
    page_groups = defaultdict(list)
    for row in rows:
        page_groups[row["image_path"]].append(row)
    check(len(page_groups) == 32, "Expected 32 actually-viewed page attestations")
    pages = []
    for image_path, page in page_groups.items():
        check(len(page) == 4 and {r["model"] for r in page} == set(models), "Incomplete page model columns")
        pages.append({"request_id": page[0]["request_id"], "stage": page[0]["stage"],
                      "image_path": image_path, "image_sha256": page[0]["image_sha256"],
                      "models_viewed": models, "row_count": 4,
                      "reviewers": sorted({r["reviewer"] for r in page}),
                      "status": "ALL_FOUR_ROWS_ATTEST_ACTUALLY_VIEWED",
                      "inherited_rows": 0})
    counts = []
    for stage in stages:
        for model in models:
            group = [r for r in rows if r["stage"] == stage and r["model"] == model]
            check(len(group) == 16, "Expected 16 outputs per model/stage")
            counts.append({"model": model, "stage": stage, "denominator": len(group),
                           "label_counts": {f: {v: Counter(r[f] for r in group)[v] for v in labels} for f in FIELDS}})
    flags = []
    uncertainties = []
    for row in rows:
        bad = [f for f in FIELDS if row[f] == ("YES" if f in ("support_missing", "fragments_or_folded_layers", "visible_cracks") else "NO")]
        uncertain = [f for f in FIELDS if row[f] == "UNCERTAIN"]
        if bad:
            flags.append({"flagged_fields": bad, "record": row})
        if uncertain:
            uncertainties.append({"uncertain_fields": uncertain, "record": row})
    summary = {
        "schema": "seam_quality_four_visual_review_aggregate_v1",
        "status": "COMPLETE_VALIDATED_EXISTING_REVIEWS",
        "scope": "Four seen TRAIN C, four fixed noises, four models, RAW and POST separately",
        "review_kind": "NONBLIND_AI_PROXY_MULTIPLE_REVIEWERS_BY_PARENT",
        "rows": len(rows), "requests": len(requests), "parents": 4, "pages": len(pages),
        "actual_view_attestation": "All 32 pages have four ACTUALLY_VIEWED records from their original named reviewer. Aggregator validates records and hashes, not an independent second viewing of all pages.",
        "new_visual_judgments": 0, "new_geometry_evaluations": 0, "model_calls": 0,
        "inherited_records": 0, "UNCERTAIN_relabelled": 0,
        "counts": counts, "page_ledger": pages, "sources": provenance,
        "explicit_unfavorable_labels": flags, "uncertainty_records": uncertainties,
        "flag_selection_only": "Original positive-property NO or defect-property YES; this selection is an index of source labels, not a new composite metric or failure criterion.",
        "excluded_evidence": ["antiflatten CPU paired review", "redirection CPU paired review", "historical SEAM_SLOT_C200 reviews", "numeric CF/FF/I/connectivity scores"],
        "limits": ["Not a blind human study or inter-rater agreement estimate.",
                   "One reviewer per parent; descriptive label frequencies can include reviewer calibration differences.",
                   "YES visible join is not indexed or legal connectivity.",
                   "Blue C occludes near-coplanar generated surfaces; apparently clean images do not certify FF/I.",
                   "Four seen TRAIN conditions do not establish new-parent generalization.",
                   "No unique-source style ranking; no new overall visual success score."],
    }
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    merged = buffer.getvalue().encode("utf-8")
    reread = list(csv.DictReader(io.StringIO(merged.decode("utf-8"))))
    check(reread == rows, "CSV round trip altered reviewed cells")
    check(all(original_rows[(r["request_id"], r["stage"], r["model"])] == r for r in reread), "Source cell changed")

    md = ["# 端点图审汇总：已有记录的只读核账", "",
          "完整合并四位原评阅者对四条已见 TRAIN C 的记录：16 请求 × 4 模型 × RAW/POST = 128 行，32 页每页四列全部登记为 ACTUALLY_VIEWED。这里核验原图审声明、身份与图像 hash；不是聚合程序重新看图。没有新增视觉判断、几何评分或模型调用。", "",
          "[逐行 CSV](visual_review.csv)；[完整 JSON、逐页账本与所有不确定记录](visual_review_summary.json)；[校验回执](visual_review_receipt.json)。", "",
          "以下每格为 **YES / NO / UNCERTAIN** 原始计数，每组分母 16。未将 UNCERTAIN 当作 NO；缺陷字段的 YES 是发现可见问题，正向字段的 YES 是观察到对应性质，不能把各列直接相加排名。", ""]
    for group_fields in (FIELDS[:3], FIELDS[3:]):
        md += ["| Stage | Model | " + " | ".join(group_fields) + " |",
               "| --- | --- | " + " | ".join(["---"] * len(group_fields)) + " |"]
        for item in counts:
            md.append("| " + item["stage"] + " | " + item["model"] + " | " + " | ".join(" / ".join(str(item["label_counts"][f][v]) for v in labels) for f in group_fields) + " |")
        md.append("")
    md += ["## 具体可见问题索引（原标签原文）", "",
           "下面只索引原表中正向性质为 NO 或缺陷性质为 YES 的记录，不建立新的复合失败定义。全部 UNCERTAIN 原文单独保存在 JSON 的 uncertainty_records，原始逐行结果没有删减。", "",
           "| 请求 | 阶段 / 模型 | 原不利标签 | 原评阅者具体观察 |",
           "| --- | --- | --- | --- |"]
    for item in flags:
        r = item["record"]
        escaped = r["visible_observations"].replace("|", "\\|").replace("\n", " ")
        md.append(f"| {r['request_id']} | {r['stage']} / {r['model']} | " + "; ".join(f"{f}={r[f]}" for f in item["flagged_fields"]) + f" | {escaped} |")
    if not flags:
        md.append("| 无显式不利标签 | — | — | 不代表没有隐藏缺陷，UNCERTAIN仍保留 |")
    md += ["", "## 各父图审文字结论", "",
           "以下概括沿用原评阅文字，不据评分改判。", "",
           "- **3f**：主体与 C 座面用途保留；n1 旧的内向灰色剖面线在新臂中不明显，n2 RAW 的近座层线也更少见。但 n3 新侧剖存在含义不明的下表面断段；相似 POST 外观不证明冲突尾部已消失。见[原记录](reviews/3f.md)。",
           "- **73**：n2 旧模型靠背的大三角开缝在新臂更连续；总体差别不大。座面下平行线可能是正常厚度或重层，保留不确定；没有 ANTI_COVER/LOCAL_NOISE 独有整体优势。见[原记录](reviews/73.md)。",
           "- **85**：n1 三个续训臂均出现后侧长片，n2 LOCAL 的侧裙开口最明显；n0 LONG/ANTI 的背板开口保留款式歧义。n3 四臂接近。见[原记录](reviews/85.md)。",
           "- **8c**：四种噪声均保留座面、梯形背条和支撑；新臂部分座箱剖面更长、更方，但背条斜线、局部小不齐与开口仍难解释为厚度还是折层。没有 ANTI_COVER/LOCAL_NOISE 相对 LONG 的明确整体视觉胜出。见[原记录](reviews/8c.md)。", "",
           "这不是统一人工标尺下的盲评。各父由不同 AI 评阅者负责，标签频次仅是已记录观察的描述统计，不估计评阅一致性。可见接近不认证合法拓扑；蓝色 C 可遮挡冲突层。原源款式不是唯一正确答案。", "",
           "## 全部 32 页查看账本", "",
           "每页四模型均由下列原评阅者登记实际查看；原 CPU antiflatten/redirection 图审、历史 C200 图审不计入本轮 128 行。", "",
           "| 请求 | 阶段 | 原评阅者 | 模型列数 | 页面 |", "| --- | --- | --- | --- | --- |"]
    for page in pages:
        filename = Path(page["image_path"]).name
        md.append(f"| {page['request_id']} | {page['stage']} | {', '.join(page['reviewers'])} | 4 | [{filename}](gallery/{filename}) |")
    markdown = ("\n".join(md) + "\n").encode("utf-8")
    contents = {"visual_review.csv": merged, "visual_review_summary.json": jbytes(summary), "visual_review_summary.md": markdown}
    for filename, value in contents.items():
        write_once_or_verify(OUT / filename, value)
    check(all(sha(ROOT / path) == value for path, value in protected.items()), "Protected input changed during aggregation")
    receipt = {
        "schema": "seam_quality_four_visual_review_validation_receipt_v1", "status": "PASS",
        "script": relative(Path(__file__)), "script_sha256": sha(Path(__file__)),
        "row_count": 128, "page_count": 32, "field_count": 20,
        "unique_registered_identities": 128, "actual_view_record_count": 128,
        "actual_view_attested_pages": 32, "independent_second_viewing_by_aggregator": False,
        "all_page_four_model_columns_complete": True, "all_image_hashes_match_source_and_render_receipt": True,
        "source_cell_round_trip_identical": True, "protected_originals_unchanged": True,
        "labels_reassigned": 0, "new_visual_judgments": 0, "new_geometry_scores": 0, "model_calls": 0,
        "excluded_old_CPU_review_rows": True, "protected_hashes": protected,
        "output_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
        "runtime": "stdlib Python only; no numpy, torch, neural modules, network or GPU call",
        "recovery": "Rerunning this script validates source hashes and exact deterministic output bytes; it refuses changed pre-existing outputs.",
    }
    action = write_once_or_verify(OUT / "visual_review_receipt.json", jbytes(receipt))
    print(json.dumps({"status": "PASS", "rows": 128, "pages": 32, "action": action, "explicit_flag_rows": len(flags), "uncertainty_rows": len(uncertainties)}))


if __name__ == "__main__":
    main()
