"""Small CPU identity tests; no checkpoint IO, model construction, or CUDA."""
import copy
import unittest

from native_t1.context_geometry import ENCODER_METADATA
from native_t1.geometry_checkpoint import SCHEMA, validate_geometry_checkpoint
from native_t1.portable_checkpoint import NATIVE_META, validate_training_metadata

COUNTS = dict(geometry_metadata_validations=0, portable_metadata_validations=0,
              checkpoint_reads=0, model_constructions=0, model_forwards=0,
              backwards=0, optimizer_updates=0, OT_calls=0)


def valid_payload(mode="geo", completed=500):
    return dict(schema=SCHEMA, native_meta=copy.deepcopy(NATIVE_META),
        context=dict(copy.deepcopy(ENCODER_METADATA), context_mode=mode, additional_parameters=149888),
        model={}, optimizer={}, stream_state=dict(seed=1010, arm="Native_correct",
            batch_index=completed, sample_index=8*completed), stream_seed=1010,
        input_sha256="1"*64, config_sha256="2"*64, model_state_sha256="3"*64,
        base_identity=dict(cumulative_updates=1500, checkpoint_sha256="4"*64, state_sha256="5"*64),
        base_cumulative_updates=1500, completed_updates=completed, cumulative_updates=1500+completed,
        loss="masked_free_fm", num_faces=112, train_K=[4,8,12], optimizer_reset_at_start=True,
        optimizer_step_in_progress=False, batch_in_progress=False)


def validate(payload):
    COUNTS["geometry_metadata_validations"] += 1
    return validate_geometry_checkpoint(payload)


class GeometryCheckpointTests(unittest.TestCase):
    def reject(self, payload):
        with self.assertRaises(ValueError):
            validate(payload)

    def test_accept_registered_modes_and_fixed_save_steps(self):
        for mode in ("geo", "graph"):
            for step in (250, 500):
                with self.subTest(mode=mode, step=step):
                    payload = valid_payload(mode, step)
                    self.assertEqual(validate(payload), payload["context"])

    def test_reject_wrong_architecture_feature_and_precision(self):
        for key, value in (("schema", "native_t1_training_v1"), ("native_meta", {})):
            with self.subTest(key=key):
                payload=valid_payload(); payload[key]=value; self.reject(payload)
        changes = dict(context_mode="none", feature_dim=12, hidden_dim=256,
            graph_layers=2, output_dim=384, initialization_seed=10,
            branch_dtype="torch.bfloat16", injection="after_block12",
            graph="tolerance_weld", geometry="signed_normal", area_reject_le=1e-8)
        for key, value in changes.items():
            with self.subTest(key=key):
                payload=valid_payload();payload["context"][key]=value;self.reject(payload)

    def test_reject_stream_seed_arm_and_counters(self):
        payload=valid_payload();payload["stream_seed"]=10;self.reject(payload)
        for key,value in (("seed",10),("arm","Native_independent"),
                          ("batch_index",499),("sample_index",3999)):
            with self.subTest(key=key):
                payload=valid_payload();payload["stream_state"][key]=value;self.reject(payload)
        for key in ("optimizer","stream_state"):
            for replacement in (None, []):
                with self.subTest(key=key, replacement=replacement):
                    payload=valid_payload();payload[key]=replacement;self.reject(payload)

    def test_reject_progress_loss_and_incomplete_update(self):
        changes=dict(base_cumulative_updates=1000,completed_updates=501,cumulative_updates=2001,
                     loss="masked_free_fm_plus_FC",num_faces=100,train_K=[12],
                     optimizer_reset_at_start=False,optimizer_step_in_progress=True,batch_in_progress=True)
        for key,value in changes.items():
            with self.subTest(key=key):
                payload=valid_payload();payload[key]=value;self.reject(payload)
        payload=valid_payload();payload["base_identity"]["cumulative_updates"]=500;self.reject(payload)
        for flag in ("optimizer_step_in_progress", "batch_in_progress"):
            payload=valid_payload();del payload[flag];self.reject(payload)

    def test_reject_missing_or_invalid_hash_identity(self):
        for key in ("input_sha256","config_sha256","model_state_sha256"):
            for value in (None, "abc", "A"*64):
                with self.subTest(key=key, value=value):
                    payload=valid_payload();payload[key]=value;self.reject(payload)
            payload=valid_payload();del payload[key];self.reject(payload)
        for key in ("checkpoint_sha256", "state_sha256"):
            payload=valid_payload();del payload["base_identity"][key];self.reject(payload)
        payload=valid_payload();del payload["base_identity"];self.reject(payload)

    def test_original_portable_schema_does_not_accept_new_architecture(self):
        for mode in ("geo", "graph"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                COUNTS["portable_metadata_validations"] += 1
                validate_training_metadata(valid_payload(mode))


if __name__ == "__main__":
    unittest.main()
