"""Pure CPU schedule contract checks; no arrays, model, OT or training calls."""
from __future__ import annotations

import copy
import unittest

from .sampler import adapt_schedule, tag_exposure_pairs, PAIR_KEY, RANDOM_FIELDS


def fixture():
    parents = []
    for i, group in enumerate(("OLD", "OLD", "FOCUS", "HIGH")):
        pid, cid = f"parent{i}", f"physical_c{i}"
        n = 308 if group == "HIGH" else 200
        variants = [dict(variant_id="natural", kind="NATURAL", case_path=f"{pid}/natural.npz",
                         case_sha256=str(i)*64, target_version_id=f"{pid}_v0", N=n, K=4)]
        if i != 1:
            variants.append(dict(variant_id="low", kind="LOW", case_path=f"{pid}/low.npz",
                                 case_sha256=str(i+4)*64, target_version_id=f"{pid}_v1", N=160, K=4))
        parents.append(dict(parent_id=pid, condition_id=cid, replay_group=group, variants=variants))
    steps = []
    for step in range(1, 5):
        rows = []
        for i, p in enumerate(parents):
            for variant, augmentation in enumerate(("CLEAN", "X_WIDTH")):
                alpha = 1. if variant == 0 else .95 + step*.01
                r = dict(index=i, condition_index=i, parent_id=p["parent_id"], condition_id=p["condition_id"],
                         replay_group=p["replay_group"], source_group="SOURCE_"+p["replay_group"],
                         sample_index=len(rows), augmentation=augmentation, alpha=alpha,
                         scale_xyz=[alpha, 1., 1.])
                r.update({key: step*10000+i*100+variant*10+j for j, key in enumerate(RANDOM_FIELDS)})
                rows.append(r)
        steps.append(dict(stage_step=step, step=step, samples=rows))
    schedule = dict(schema="BENCH_FOUR_ROUTES_M_PLUS3000_STREAM_V1", steps=steps, updates=4)
    return schedule, dict(parents=parents)


class ParentFirstLODTests(unittest.TestCase):
    def test_pairing_exposure_rng_identity_and_variant_paths(self):
        source, manifest = fixture()
        old_source, old_manifest = copy.deepcopy(source), copy.deepcopy(manifest)
        tagged = tag_exposure_pairs(source)
        result, proof = adapt_schedule(tagged, manifest)
        self.assertEqual(source, old_source)
        self.assertEqual(manifest, old_manifest)
        self.assertEqual(result["parent_exposure"], {f"parent{i}": 8 for i in range(4)})
        self.assertEqual(result["group_exposure"], {"OLD": 16, "FOCUS": 8, "HIGH": 8})
        self.assertEqual(proof["instances"], 32)
        self.assertEqual(proof["logical_instances"], 16)
        self.assertFalse(proof["original_coefficients_reusable"])
        self.assertFalse(result["training_authorized"])
        self.assertEqual(result["trainer_integration_status"], "NOT_IMPLEMENTED")
        self.assertEqual(proof["random_draws"], 0)
        for step, before_entry, after_entry in zip(range(1, 5), tagged["steps"], result["steps"]):
            for before, after in zip(before_entry["samples"], after_entry["samples"]):
                for key, value in before.items():
                    self.assertEqual(after[key], value)
                low = step % 2 == 0 and before["parent_id"] != "parent1"
                kind = "LOW" if low else "NATURAL"
                suffix = kind.lower()
                self.assertEqual(after["variant_kind"], kind)
                self.assertEqual(after["case_path"], f"{before['parent_id']}/{suffix}.npz")
                self.assertEqual(after["N"], 160 if low else (308 if before["replay_group"] == "HIGH" else 200))
                self.assertEqual(after["parent_occurrence"], step)
            for start in (0, 2, 4, 6):
                a, b = after_entry["samples"][start:start+2]
                for key in (PAIR_KEY, "case_id", "case_sha256", "variant_id", "N", "K", "target_version_id"):
                    self.assertEqual(a[key], b[key])
        # HIGH remains HIGH even when its accepted LOW crosses below N=256.
        row = result["steps"][1]["samples"][6]
        self.assertEqual((row["replay_group"], row["N"]), ("HIGH", 160))

    def test_deterministic_repeat_and_no_double_adaptation(self):
        source, manifest = fixture()
        tagged = tag_exposure_pairs(source)
        first, proof = adapt_schedule(tagged, manifest)
        second, repeated_proof = adapt_schedule(tagged, manifest)
        self.assertEqual(first, second)
        self.assertEqual(proof, repeated_proof)
        with self.assertRaisesRegex(ValueError, "twice"):
            adapt_schedule(first, manifest)
        with self.assertRaisesRegex(ValueError, "overwritten"):
            tag_exposure_pairs(tagged)

    def test_unknown_schema_needs_explicit_pair_identity(self):
        source, manifest = fixture()
        source["schema"] = "UNKNOWN_CUSTOM_SCHEMA"
        with self.assertRaisesRegex(ValueError, "Unknown schedule"):
            tag_exposure_pairs(source)
        with self.assertRaisesRegex(ValueError, "Explicit logical_instance_id"):
            adapt_schedule(source, manifest)
        for entry in source["steps"]:
            for row in entry["samples"]:
                row[PAIR_KEY] = f"explicit-step{entry['stage_step']}-parent{row['parent_id']}"
        result, _ = adapt_schedule(source, manifest)
        self.assertEqual(result["source_schema"], "UNKNOWN_CUSTOM_SCHEMA")

    def test_malformed_pairs_and_missing_rng_rejected(self):
        source, manifest = fixture()
        for fault in ("parent", "augmentation", "rng", "pair_id"):
            tagged = tag_exposure_pairs(source)
            row = tagged["steps"][0]["samples"][1]
            if fault == "parent": row["parent_id"] = "parent1"
            elif fault == "augmentation": row["augmentation"] = "CLEAN"
            elif fault == "rng": del row["time_seed"]
            else: row[PAIR_KEY] += "-unpaired"
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                adapt_schedule(tagged, manifest)

    def test_manifest_contract_and_source_group_preserved(self):
        source, manifest = fixture()
        tagged = tag_exposure_pairs(source)
        for fault in ("group", "K", "SHA", "duplicate", "equalN"):
            broken = copy.deepcopy(manifest)
            p = broken["parents"][0]
            if fault == "group": p["replay_group"] = "HIGH"
            elif fault == "K": p["variants"][1]["K"] = 5
            elif fault == "SHA": p["variants"][1]["case_sha256"] = "z"*64
            elif fault == "duplicate": broken["parents"].append(copy.deepcopy(p))
            else: p["variants"][1]["N"] = p["variants"][0]["N"]
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                adapt_schedule(tagged, broken)


if __name__ == "__main__":
    unittest.main()
