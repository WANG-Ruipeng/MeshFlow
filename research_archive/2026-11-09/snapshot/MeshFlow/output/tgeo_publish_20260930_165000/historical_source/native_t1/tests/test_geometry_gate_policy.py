"""CPU-only tests for explicit evidence and unchanged hard gate semantics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from native_t1 import geometry_gates as gates
from native_t1.artifacts import file_sha256

COUNTS = dict(evidence_validation_calls=0, scalar_comparison_checks=0,
              model_forwards=0, backwards=0, optimizer_updates=0, complete_rollouts=0)


class GeometryGatePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        self.results = directory/"results.json"
        self.velocities = directory/"velocities.npz"
        self.identity = dict(checkpoint_sha256="1"*64, state_sha256="2"*64)
        self.inputs_sha = "3"*64
        self.model_sha = file_sha256(gates.REPO/"native_t1/model.py")
        self.evidence = dict(
            status="DIAGNOSIS_COMPLETED_EXPERIMENT_REMAINS_STOPPED",
            registration=dict(source_commit="2502d5f89c54706cc4700d3cdea4a12fd6792529",
                checkpoint_sha256=self.identity["checkpoint_sha256"], inputs_sha256=self.inputs_sha,
                source_pins={"native_t1/model.py":self.model_sha}),
            initialization=dict(checkpoint_sha256=self.identity["checkpoint_sha256"]),
            state_before="2"*64, state_after="2"*64, base_wrapper_state_sha256="2"*64,
            base_wrapper_state_after="2"*64, parameters_unchanged=True, source_unchanged=True,
            fixed_batch_hashes_match_failed_gate=True,
            current_vs_base_commit={key:dict(bitwise=True, finite=True) for key in ("reference","combined")},
            counts=dict(model_forward_attempts=9, optimizer_updates=0))
        x=np.arange(1008,dtype=np.float32).reshape(1,112,9)/1024
        self.arrays=dict(base_commit_reference=x.copy(),current_reference=x.copy(),
                         base_commit_combined=x.copy(),current_combined=x.copy())
        self.patch_results=mock.patch.object(gates,"BASELINE_EVIDENCE_RESULTS",self.results)
        self.patch_velocities=mock.patch.object(gates,"BASELINE_EVIDENCE_VELOCITIES",self.velocities)
        self.patch_results.start();self.addCleanup(self.patch_results.stop)
        self.patch_velocities.start();self.addCleanup(self.patch_velocities.stop)
        self.write_fixture()

    def write_fixture(self, *, proof_sha=None):
        np.savez_compressed(self.velocities,**self.arrays)
        self.evidence["velocity_proof"] = dict(path=str(self.velocities),
            sha256=file_sha256(self.velocities) if proof_sha is None else proof_sha)
        self.results.write_text(json.dumps(self.evidence),encoding="utf-8")
        self.contract=dict(gate_policy=gates.GATE_POLICY,baseline_regression_evidence=dict(
            results_path=str(self.results),results_sha256=file_sha256(self.results),
            velocities_path=str(self.velocities),velocities_sha256=file_sha256(self.velocities)))

    def validate(self, contract=None, identity=None, inputs_sha=None):
        COUNTS["evidence_validation_calls"] += 1
        return gates.load_baseline_regression_evidence(
            self.contract if contract is None else contract,
            self.identity if identity is None else identity,
            self.inputs_sha if inputs_sha is None else inputs_sha)

    def test_valid_explicit_evidence_returns_saved_exact_cpu_velocities(self):
        values,audit=self.validate()
        self.assertEqual(audit["status"],"VERIFIED")
        self.assertEqual(audit["policy"],gates.GATE_POLICY)
        self.assertFalse(audit["external_counts_included_in_this_gate_budget"])
        self.assertEqual(audit["existing_diagnostic_counts"]["model_forward_attempts"],9)
        for label in ("reference","combined"):
            self.assertEqual(values[label].device.type,"cpu")
            self.assertTrue(torch.equal(values[label],torch.from_numpy(self.arrays["base_commit_"+label])))

    def test_no_missing_implicit_or_alternative_evidence_policy(self):
        changes=({},dict(gate_policy="legacy"),dict(gate_policy=gates.GATE_POLICY),
                 dict(gate_policy=gates.GATE_POLICY,baseline_regression_evidence={}))
        for contract in changes:
            with self.subTest(contract=contract),self.assertRaises(ValueError):self.validate(contract)
        contract=copy.deepcopy(self.contract)
        contract["baseline_regression_evidence"]["results_path"]=str(self.results.with_name("other.json"))
        with self.assertRaises(ValueError):self.validate(contract)
        for key in ("results_sha256","velocities_path","velocities_sha256"):
            contract=copy.deepcopy(self.contract);del contract["baseline_regression_evidence"][key]
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(contract)

    def test_file_and_declared_velocity_proof_hashes_are_required(self):
        for key in ("results_sha256","velocities_sha256"):
            contract=copy.deepcopy(self.contract);contract["baseline_regression_evidence"][key]="0"*64
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(contract)
        self.write_fixture(proof_sha="0"*64)
        with self.assertRaises(ValueError):self.validate()

    def test_current_checkpoint_inputs_code_and_parameter_state_must_match(self):
        with self.assertRaises(ValueError):self.validate(identity=dict(self.identity,checkpoint_sha256="0"*64))
        with self.assertRaises(ValueError):self.validate(inputs_sha="0"*64)
        with self.assertRaises(ValueError):self.validate(identity=dict(self.identity,state_sha256="0"*64))
        self.evidence["registration"]["source_pins"]["native_t1/model.py"]="0"*64
        self.write_fixture()
        with self.assertRaises(ValueError):self.validate()

    def test_incomplete_or_nonexact_report_is_rejected_even_with_fresh_file_hash(self):
        original=copy.deepcopy(self.evidence)
        for key,value in (("status","RUNNING"),("parameters_unchanged",False),
                          ("source_unchanged",False),("fixed_batch_hashes_match_failed_gate",False),
                          ("state_after","0"*64)):
            with self.subTest(key=key):
                self.evidence=copy.deepcopy(original);self.evidence[key]=value;self.write_fixture()
                with self.assertRaises(ValueError):self.validate()
        for label in ("reference","combined"):
            self.evidence=copy.deepcopy(original)
            self.evidence["current_vs_base_commit"][label]["bitwise"]=False
            self.write_fixture()
            with self.subTest(label=label),self.assertRaises(ValueError):self.validate()

    def test_npz_actual_values_shape_dtype_and_finite_are_verified(self):
        original=copy.deepcopy(self.arrays)
        for kind in ("different","nonfinite","float64","wrong_shape","missing"):
            self.arrays=copy.deepcopy(original)
            if kind=="different":self.arrays["current_combined"][0,0,0]+=.125
            elif kind=="nonfinite":self.arrays["base_commit_reference"][0,0,0]=np.nan
            elif kind=="float64":self.arrays["current_reference"]=self.arrays["current_reference"].astype(np.float64)
            elif kind=="wrong_shape":self.arrays["current_reference"]=self.arrays["current_reference"][0]
            else:del self.arrays["base_commit_combined"]
            self.write_fixture()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.validate()

    def test_numerical_failure_remains_fail_and_only_verified_evidence_allows_recording(self):
        _,audit=self.validate()
        actual=torch.tensor([.0625],dtype=torch.float32);expected=torch.zeros_like(actual)
        COUNTS["scalar_comparison_checks"]+=1
        row=gates._permutation_diagnostic(actual,expected,evidence=audit)
        self.assertEqual(row["status"],"FAIL")
        self.assertFalse(row["allclose"])
        self.assertFalse(row["blocking"])
        self.assertEqual((row["atol"],row["rtol"]),(.015625,.03125))
        self.assertEqual(row["violating_elements"],1)
        for evidence in (None,{},dict(status="VERIFIED",policy="wrong")):
            COUNTS["scalar_comparison_checks"]+=1
            with self.assertRaises(ValueError):gates._permutation_diagnostic(actual,expected,evidence=evidence)
        COUNTS["scalar_comparison_checks"]+=1
        with self.assertRaises(RuntimeError):gates._permutation_diagnostic(actual*float("nan"),expected,evidence=audit)

    def test_default_strict_policy_never_exempts_original_threshold_failure(self):
        actual=torch.tensor([.0625],dtype=torch.float32);expected=torch.zeros_like(actual)
        COUNTS["scalar_comparison_checks"]+=1
        with self.assertRaises(RuntimeError):gates._permutation_check(actual,expected)
        COUNTS["scalar_comparison_checks"]+=1
        with self.assertRaises(RuntimeError):
            gates._permutation_check(actual,expected,policy=gates.STRICT_GATE_POLICY,
                evidence=dict(status="VERIFIED",policy=gates.GATE_POLICY,results_sha256="0"*64))
        COUNTS["scalar_comparison_checks"]+=1
        passed=gates._permutation_check(expected,expected)
        self.assertEqual(passed["status"],"PASS")
        self.assertTrue(passed["blocking"])
        self.assertEqual((passed["atol"],passed["rtol"]),(.015625,.03125))
        COUNTS["scalar_comparison_checks"]+=1
        with self.assertRaises(ValueError):
            gates._permutation_check(actual,expected,policy=gates.GATE_POLICY)

    def test_hard_bitwise_branch_inheritance_and_feature_thresholds_are_unchanged(self):
        a=torch.zeros(2);b=torch.tensor([1e-7,0.])
        COUNTS["scalar_comparison_checks"]+=1
        with self.assertRaises(RuntimeError):gates._comparison(a,b,bitwise=True)
        COUNTS["scalar_comparison_checks"]+=1
        with self.assertRaises(RuntimeError):gates._comparison(a,torch.ones(2),atol=2e-6,rtol=2e-5)
        COUNTS["scalar_comparison_checks"]+=1
        self.assertEqual(gates._comparison(a,a,bitwise=True)["status"],"PASS")


if __name__=="__main__":
    unittest.main()
