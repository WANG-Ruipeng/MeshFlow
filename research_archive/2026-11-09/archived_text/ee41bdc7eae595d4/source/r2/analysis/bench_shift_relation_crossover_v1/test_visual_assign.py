"""Synthetic CPU-only checks; never inspect the real anonymous review."""
import json
from pathlib import Path
import tempfile
import unittest

from . import visual_assign as V


def fixture(root):
    write = V.C.write_json
    qs = []
    pages = []
    for parent in range(12):
        for req in range(4):
            q = dict(parent_id=f"p{parent:02}", request_id=f"p{parent:02}_r{req}",
                     condition_id=f"p{parent:02}_c{req // 2}", noise_index=req % 2,
                     split="TRAIN" if parent < 4 else "DEV", part_type="SEAT_SURFACE")
            qs.append(q)
            for stage in V.A.STAGES:
                pages.append(dict(**q, sheet=stage, page_id=stage + "__" + q["request_id"],
                                  cells=[dict(model=l, stage=stage) for l in V.A.LABELS]))
    write(root / "eval_manifest.json", dict(requests=qs))
    write(root / "blind_review/gate.json", dict(status="PASS"))
    write(root / "gallery/manifest.json", dict(pages=pages))
    write(root / "gallery/definition.json", dict(manifest_sha256=V.C.sha(root / "gallery/manifest.json")))
    dh = V.C.sha(root / "gallery/definition.json")
    images = []
    for page in pages:
        p = root / "gallery" / (page["page_id"] + ".png")
        p.write_bytes(b"\x89PNG\r\n\x1a\nSYNTHETIC_NOT_FOR_REVIEW")
        images.append(dict(**page, path=str(p), image_sha256=V.C.sha(p), definition_sha256=dh))
    write(root / "gallery/completion.json", dict(status="RENDER_COMPLETE_PENDING_ACTUAL_REVIEW", sheets=144, images=images))
    return images


class VisualAssignTests(unittest.TestCase):
    def test_missing_gallery_waits_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(V.prepare(root)["status"], "WAITING")
            self.assertEqual(list(root.iterdir()), [])

    def test_complete_grouping_schema_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root)
            result = V.prepare(root)
            self.assertEqual(result["labels_written"], 0)
            a = V.C.read(root / "visual_review/assignments.json")
            self.assertEqual(a["labels"], list(V.A.LABELS))
            self.assertEqual(a["protocol_sha256"], V.C.sha(root / "visual_review/protocol.md"))
            seen = []
            for g, t in a["tasks"].items():
                self.assertEqual(t["record_file"], "review_" + g + ".json")
                self.assertEqual(len(t["parents"]), 4)
                self.assertEqual(len(t["pages"]), 48)
                self.assertEqual(t["expected_model_stage_rows"], 144)
                self.assertEqual(t["required_record_fields"], list(V.FIELDS))
                seen.extend(p["page_id"] for p in t["pages"])
            self.assertEqual(len(set(seen)), 144)
            self.assertEqual(V.prepare(root), result)
            self.assertEqual(sorted(p.name for p in (root / "visual_review").iterdir()),
                             ["assignments.json", "protocol.md"])

    def test_missing_png_waits_without_partial_assignment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); images = fixture(root)
            Path(images[-1]["path"]).unlink()
            self.assertEqual(V.prepare(root)["status"], "WAITING")
            self.assertFalse((root / "visual_review").exists())

    def test_changed_png_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); images = fixture(root)
            Path(images[0]["path"]).write_bytes(b"changed")
            with self.assertRaises(AssertionError): V.prepare(root)
            self.assertFalse((root / "visual_review").exists())

    def test_gate_failure_does_not_prepare(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root)
            (root / "blind_review/gate.json").write_text(json.dumps(dict(status="FAIL")))
            self.assertEqual(V.prepare(root)["status"], "WAITING")
            self.assertFalse((root / "visual_review").exists())


if __name__ == "__main__":
    unittest.main()
