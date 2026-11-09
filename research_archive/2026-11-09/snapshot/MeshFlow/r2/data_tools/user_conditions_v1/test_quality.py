"""Small CPU tests for the new partition adapter; no source data rescans."""
import json
from pathlib import Path
import tempfile
import unittest
import quality

class QualityTest(unittest.TestCase):
    def test_unknown_not_clean(self):
        rows = [{"pair": [0, 1], "classification": quality.PROPER},
                {"pair": [0, 2], "classification": quality.COPLANAR},
                {"pair": [1, 2], "classification": "UNRESOLVED"},
                {"pair": [2, 3], "classification": "shared_edge_contact"}]
        value = quality.summarize_pairs(rows, 6)
        self.assertEqual(value["category_counts"], dict(proper=1, coplanar_positive_overlap=1, UNKNOWN=1, legal_contact=1, disjoint=2))
        self.assertEqual(value["confirmed_illegal_pair_count"], 2)
        self.assertEqual(value["unknown_pair_count"], 1)
        self.assertEqual(value["status"], "PARTIAL_UNKNOWN")

    def test_partition_ids_and_denominators(self):
        rows = [{"pair": [0, 1], "original_face_ids": [10, 11], "classification": quality.PROPER},
                {"pair": [0, 2], "original_face_ids": [10, 12], "classification": "UNRESOLVED"},
                {"pair": [2, 3], "original_face_ids": [12, 13], "classification": quality.COPLANAR}]
        parent = dict(N=4, pair_contacts=rows, face_id_policy="test", whole_parent=quality.summarize_pairs(rows, 6),
            basic_geometry=dict(exact_duplicate_groups=[], nonfinite_source_face_ids=[], exact_zero_area_source_face_ids=[]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "parents/u.json"
            path.parent.mkdir()
            path.write_text(json.dumps(parent))
            groups = quality.partition_condition("u", [0, 1], folder)["partitions"]
            self.assertEqual([groups[g]["expected_pairs"] for g in ("CC", "CF", "FF")], [1, 4, 1])
            self.assertEqual(groups["CC"]["confirmed_illegal_pair_count"], 1)
            self.assertEqual(groups["CF"]["unknown_pair_count"], 1)
            self.assertEqual(groups["FF"]["problem_pairs"][0]["original_face_ids"], [12, 13])
            with self.assertRaises(ValueError): quality.partition_condition("u", [0, 0], folder)
            with self.assertRaises(ValueError): quality.partition_condition("u", [], folder)

if __name__ == "__main__": unittest.main()