"""Small synthetic accounting fixtures only; never call a network or scorer."""
import hashlib
import unittest

from .stream_audit import (paired_stream, first_identity_micros, FIRST_MICRO_METRICS,
                           TRANSLATION_DOMAIN, AXIS_BOUND)


def fixture():
    samples, new_samples, old_exposure, exposure = [], [], [], []
    for micro in range(8):
        row = dict(sample_index=micro, condition_id="c"+str(micro % 4), parent_id="p"+str(micro % 4),
                   augmentation="CLEAN" if micro < 4 else "ANISOTROPIC",
                   scale_xyz=[1.,1.,1.] if micro < 4 else [.95,1.05,1.01], noise_seed=10+micro)
        token = f'{TRANSLATION_DOMAIN}|1|{micro}|{row["condition_id"]}'
        seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little")
        translated = dict(row, translation_seed=seed,
                          translation_xyz=[0.,0.,0.] if micro < 4 else [.01,-.02,.03])
        runtime = dict(N=128,K=2,t=.3,local_step=1,global_step=12001,microbatch=micro)
        stats = {key:float(micro+1) for key in FIRST_MICRO_METRICS}
        samples.append(row)
        new_samples.append(translated)
        old_exposure.append(dict(row,**runtime,**stats))
        exposure.append(dict(translated,**runtime,**stats))
    return ({"steps":[dict(step=1,samples=new_samples)]},
            {"steps":[dict(step=1,samples=samples)]}, exposure, old_exposure)


class StreamContract(unittest.TestCase):
    def test_exact_old_inputs_and_translation_counts(self):
        got = paired_stream(*fixture(),expected_updates=1)
        self.assertEqual(got["status"],"PASS")
        self.assertEqual(got["zero_translation_samples"],4)
        self.assertEqual(got["shifted_samples"],4)
        self.assertEqual(got["axis_statistics_shifted_only"]["y"]["max"],-.02)

    def test_old_rng_change_is_not_allowed(self):
        args=list(fixture())
        args[0]["steps"][0]["samples"][4]["noise_seed"]+=1
        with self.assertRaisesRegex(ValueError,"Old scheduled input"):
            paired_stream(*args,expected_updates=1)

    def test_undeclared_new_input_key_rejected(self):
        args=list(fixture())
        args[0]["steps"][0]["samples"][0]["surprise"]=1
        with self.assertRaisesRegex(ValueError,"Unexpected scheduled keys"):
            paired_stream(*args,expected_updates=1)

    def test_actual_exposure_does_not_escape_schedule_check(self):
        args=list(fixture())
        args[2][5]["noise_seed"]+=1
        with self.assertRaisesRegex(ValueError,"actual input mismatch"):
            paired_stream(*args,expected_updates=1)

    def test_cluster_of_translations_cannot_exceed_axis_budget(self):
        args=list(fixture())
        bad=[AXIS_BOUND+.001,0.,0.]
        args[0]["steps"][0]["samples"][4]["translation_xyz"]=bad
        args[2][4]["translation_xyz"]=bad
        with self.assertRaisesRegex(ValueError,"outside frozen"):
            paired_stream(*args,expected_updates=1)

    def test_first_four_are_exact_and_failure_preserves_delta(self):
        _,_,new,old=fixture()
        got=first_identity_micros(new,old)
        self.assertEqual(got["comparisons"],40)
        self.assertEqual(got["status"],"PASS")
        new[2]["endpoint_sse"]+=.25
        failed=first_identity_micros(new,old)
        self.assertEqual(failed["status"],"FAIL")
        self.assertEqual(failed["mismatches"][0]["delta"],.25)
        self.assertFalse(failed["automatic_retry"])

    def test_shifted_half_not_claimed_equal(self):
        _,_,new,old=fixture()
        new[4]["endpoint_sse"]+=1234
        self.assertEqual(first_identity_micros(new,old)["status"],"PASS")


if __name__ == "__main__":
    unittest.main()
