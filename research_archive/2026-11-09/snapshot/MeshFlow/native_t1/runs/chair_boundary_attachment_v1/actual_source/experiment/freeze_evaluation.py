"""Separate CPU evaluation source freeze; does not alter the training protocol."""
from .common import *
from experiments.mf_h1_local0.chair_control_usability_quick_v1 import scoring as quality
def run():
    files={str((CODE/n).relative_to(ROOT)):digest(CODE/n) for n in ('evaluation.py','aggregation.py','gallery.py','reporting.py','control_metrics.py','metric_tests.py','independent_audit.py')}
    files.update(quality.source_identity())
    files[str(Path(quality.__file__).relative_to(ROOT))]=digest(quality.__file__)
    freeze_json(OUT/'protocol/evaluation_sources_before.json',files)
    freeze_json(OUT/'protocol/evaluation_freeze_receipt.json',dict(status='PASS',files=len(files),
        cpu_method_tests='metric_tests_v1r1/metric_test_results.json',numeric_regressions='engineering/metric_revision002/regression_results.json',
        training_protocol_unchanged=True,model_calls=0,already_scored_START=76,new_endpoint_outputs_seen=0))
    print(dict(status='PASS',frozen_evaluation_files=len(files)))
if __name__=='__main__':run()
