"""CPU-only EMA/schema/history tests. No real model construction, OT or GPU."""
import copy, json, tempfile, time, unittest, zipfile
from pathlib import Path
import torch
from . import replay as rp


class TinyContext(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([2.,-1.]))
    def metadata(self):
        from native_t1.context_geometry import ENCODER_METADATA
        return dict(ENCODER_METADATA,context_mode="geo",additional_parameters=149888)

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=torch.nn.Linear(2,2,bias=False)
        self.role_embedding=torch.nn.Parameter(torch.tensor([1.,3.]))
        self.context_encoder=TinyContext();self.register_buffer("fixed",torch.tensor([4.]))
        self.register_buffer("integer",torch.tensor([2],dtype=torch.int64))
        self.register_buffer("not_persistent",torch.tensor([7.]),persistent=False)
    # This fixture is never forwarded.

class Stream:
    seed=1010
    def __init__(self,step):self.step=step
    def state_dict(self):return dict(seed=1010,batch_index=1000+self.step,sample_index=8000+8*self.step,arm="Native_correct",rng={})


def legacy_math():
    with zipfile.ZipFile(rp.ZIP) as z:
        def src(name):return z.read("experiments/mf_h1_local0/"+name).decode("utf-8")
        relation={"__name__":"legacy_relation_fixture"};exec(compile(src("loss_screen/relation_losses.py"),"<archived relation>","exec"),relation)
        edge={"__name__":"legacy_edge_fixture","prepare_relations":relation["prepare_relations"]}
        code=src("l3_l5_interface_pilot/edge_loss.py").replace("from ..loss_screen.relation_losses import prepare_relations","")
        exec(compile(code,"<archived edge>","exec"),edge)
        surface={"__name__":"legacy_surface_fixture"};exec(compile(src("loss_screen/distribution_losses.py"),"<archived surface>","exec"),surface)
    return edge,surface


class TestEMA(unittest.TestCase):
    def test_initial_independence_and_all_groups(self):
        torch.manual_seed(110);m=Tiny();before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
        e=rp.FP32EMA(m)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertEqual(set(e.parameters),set(dict(m.named_parameters())))
        for k,v in before.items():self.assertTrue(torch.equal(v,e.state_dict()[k]));self.assertNotEqual(v.data_ptr(),e.state_dict()[k].data_ptr())
        self.assertIn("not_persistent",e.buffers);self.assertNotIn("not_persistent",e.state_dict())

    def test_ema_formula_raw_and_rng_unchanged(self):
        m=Tiny();e=rp.FP32EMA(m);old={k:v.clone() for k,v in e.parameters.items()}
        with torch.no_grad():
            for p in m.parameters():p.add_(.125)
            m.fixed.fill_(9.);m.integer.fill_(4);m.not_persistent.fill_(11.)
        raw={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
        e.update(m,1)
        for n,p in m.named_parameters():
            expected=old[n].mul(rp.BETA).add(p.detach(),alpha=1-rp.BETA)
            self.assertTrue(torch.equal(expected,e.parameters[n]));self.assertIsNone(p.grad)
        for k,v in raw.items():self.assertTrue(torch.equal(v,m.state_dict()[k]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(e.buffers["integer"].item(),4)
        self.assertEqual(e.buffers["fixed"].item(),9.);self.assertEqual(e.buffers["not_persistent"].item(),11.)

    def test_half_life_and_update_guards(self):
        m=Tiny()
        with torch.no_grad():
            for p in m.parameters():p.fill_(1.)
        e=rp.FP32EMA(m)
        with torch.no_grad():
            for p in m.parameters():p.zero_()
        for i in range(1,501):e.update(m,i)
        for p in e.parameters.values():self.assertTrue(torch.allclose(p,torch.full_like(p,1/32),atol=2e-6,rtol=0))
        self.assertAlmostEqual(rp.BETA**500,1/32,places=14)
        with self.assertRaises(ValueError):e.update(m,500)
        with self.assertRaises(ValueError):e.update(m,502)

    def test_reject_non_FP32_or_frozen_parameter(self):
        with self.assertRaises(ValueError):rp.FP32EMA(Tiny().double())
        m=Tiny();m.role_embedding.requires_grad_(False)
        with self.assertRaises(ValueError):rp.FP32EMA(m)


class TestReplayContracts(unittest.TestCase):
    def test_gate_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(RuntimeError):rp.read_stage_gate(root)
            (root/"gates").mkdir();p=root/"gates/ABC_COMPLETE.json";p.write_text('{"status":"RUNNING"}')
            with self.assertRaises(RuntimeError):rp.read_stage_gate(root)
            p.write_text('{"status":"PASS"}');self.assertEqual(rp.read_stage_gate(root)["status"],"PASS")

    def test_history_exact_stream(self):
        start,rows,records=rp.load_history()
        self.assertEqual(rp.seal(start),rp.START_STREAM_SHA);self.assertEqual(rp.seal(rows),rp.PAIRED_STREAM_SHA)
        self.assertEqual(len(rows),500);self.assertEqual(set(records),set(rp.ARMS))

    def test_historical_function_identity(self):
        audit=rp.audit_historical_sources();self.assertEqual(audit["status"],"PASS")
        self.assertEqual(len(audit["checks"]),15)

    def test_checkpoint_tiny_roundtrip_and_tamper(self):
        from native_t1.checkpoint import DEFAULT_CONFIG
        m=Tiny();e=rp.FP32EMA(m)
        for step in range(1,251):e.update(m,step)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"EMA.pt"
            rec=rp.save_checkpoint(path,m,e,Stream(250),"R_J",250,"EMA",DEFAULT_CONFIG)
            saved=torch.load(path,map_location="cpu",weights_only=True)
            rp.validate_payload(saved,expected_arm="R_J",expected_kind="EMA")
            self.assertEqual(rec["state_sha256"],rp.hash_state(e.state_dict()))
            with self.assertRaises(ValueError):rp.validate_payload(saved,expected_arm="R_B")
            with self.assertRaises(ValueError):rp.validate_payload(saved,expected_kind="raw")
            corrupt=copy.deepcopy(saved);corrupt["model"]["role_embedding"][0]+=1
            with self.assertRaises(ValueError):rp.validate_payload(corrupt)
            corrupt=copy.deepcopy(saved);del corrupt["model"]["context_encoder.weight"]
            with self.assertRaises(ValueError):rp.validate_payload(corrupt)
            corrupt=copy.deepcopy(saved);corrupt["ema_protocol"]["half_life_updates"]=50
            with self.assertRaises(ValueError):rp.validate_payload(corrupt)

    def test_final_difference_reports_first_tensor(self):
        before={"a":torch.tensor([1.,2.]),"b":torch.tensor([3.])};after=copy.deepcopy(before)
        self.assertEqual(rp.first_state_difference(after,before)["status"],"BITWISE_EQUAL")
        after["b"][0]+=1;diff=rp.first_state_difference(after,before)
        self.assertEqual(diff["tensor"],"b");self.assertEqual(diff["max_abs"],1.)


class TestHistoricalLossParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.edge,cls.surface=legacy_math()

    def check_dtype(self,dtype):
        from native_t1.losses.edge import prepare_edge_target,edge_loss
        from native_t1.losses.surface import prepare_distribution_target,distribution_loss
        free=torch.tensor([[[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]]],dtype=dtype)
        known=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],dtype=dtype)
        label=dict(free_gt=free,known=known,free_ids=torch.tensor([[1,3,2]]),known_ids=torch.tensor([[0,1,2]]),bbox_L=2.)
        old_edge=self.edge["prepare_edge_target"](label);new_edge=prepare_edge_target(label)
        for k in ("free_faces","free_corners","GT_endpoints","GT_lengths"):self.assertTrue(torch.equal(old_edge[k],new_edge[k]))
        old_surface=self.surface["prepare_distribution_target"](free,2.);new_surface=prepare_distribution_target(free,2.)
        for name,old_fn,new_fn,ot,nt in (
            ("edge",lambda x,t:self.edge["edge_loss"](x,t),edge_loss,old_edge,new_edge),
            ("surface",lambda x,t:self.surface["distribution_loss"]("L5",x,t),lambda x,t:distribution_loss("L5",x,t),old_surface,new_surface)):
            p=(free+free.new_tensor([.11,-.03,.025])).requires_grad_(True);q=p.detach().clone().requires_grad_(True)
            a,b=old_fn(p,ot)["loss"],new_fn(q,nt)["loss"]
            ga,gb=torch.autograd.grad(a,p)[0],torch.autograd.grad(b,q)[0]
            self.assertTrue(torch.equal(a,b),name);self.assertTrue(torch.equal(ga,gb),name)
            self.assertTrue(torch.isfinite(ga).all())

    def test_FP32_value_and_gradient_bitwise(self):self.check_dtype(torch.float32)
    def test_FP64_value_and_gradient_bitwise(self):self.check_dtype(torch.float64)


def run_cpu_tests(record_path=None):
    import datetime
    start=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__,fromlist=["*"]))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record=dict(status="PASS" if result.wasSuccessful() else "FAIL",tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                seconds=time.perf_counter()-start,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                counts=dict(real_model_constructions=0,model_forward_calls=0,model_backward_calls=0,optimizer_updates=0,
                            CUDA_calls=0,OT_calls=0,full_generations=0,CPU_tiny_geometry_loss_evaluations=8,CPU_prediction_leaf_gradient_calls=8,
                            CPU_tiny_checkpoint_writes=1,CPU_tiny_checkpoint_reads=1),
                sources={str(Path(p).name):rp.file_sha256(p) for p in (rp.__file__,__file__)},
                note="EMA tensor-only loops do not call an optimizer or a neural forward; all loss gradients use one synthetic free triangle. History audit is read-only.")
    if record_path is not None:rp.atomic_json(Path(record_path),record)
    return record

if __name__=="__main__":
    record=run_cpu_tests(Path(__file__).with_name("replay_cpu_test_record.json"))
    if record["status"]!="PASS":raise SystemExit(1)
