"""Small CPU checkpoint/schema tests; no official weights, forward, or CUDA."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from native_t1 import geometry_checkpoint as gc
from native_t1.artifacts import file_sha256, state_sha256
from native_t1.context_geometry import ENCODER_METADATA
from native_t1.portable_checkpoint import NATIVE_META, OFFICIAL_CONFIG_SHA256, validate_training_metadata


class TinyBackbone(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.x_embedder = torch.nn.Linear(2, 2)

    def forward(self, *args, **kwargs):
        raise AssertionError("Checkpoint tests must not call a model")


class TinyGeo(torch.nn.Module):
    def __init__(self, mode="geo", *, seed=73):
        super().__init__()
        if mode != "geo":
            raise ValueError("Geo only")
        self.mode, self.seed = mode, seed
        self.vector = torch.nn.Parameter(torch.zeros(gc.ENCODER_PARAMETER_COUNT, device="cpu"))

    def metadata(self):
        return dict(ENCODER_METADATA, context_mode=self.mode, initialization_seed=self.seed,
                    additional_parameters=gc.ENCODER_PARAMETER_COUNT)

    def forward(self, *args, **kwargs):
        raise AssertionError("Checkpoint tests must not call the encoder")


class TinyNative(torch.nn.Module):
    def __init__(self, backbone, trainable=False):
        super().__init__()
        self.backbone = backbone
        self.role_embedding = torch.nn.Parameter(torch.zeros(2, 768))
        self.context_encoder = None
        self.set_trainable(trainable)

    def set_trainable(self, value):
        self.experiment_trainable = bool(value)
        self.requires_grad_(value)
        return self.eval()

    def forward(self, *args, **kwargs):
        raise AssertionError("Checkpoint tests must not call a model")


def valid_payload(*, legacy=False, model=None, base=13, completed=7, start=5, seed=42):
    if legacy:
        base, completed, start, seed = 1500, 500, 0, 1010
    context = dict(gc.LEGACY_ENCODER_METADATA if legacy else ENCODER_METADATA,
                   context_mode="geo", initialization_seed=1010 if legacy else 73,
                   additional_parameters=gc.ENCODER_PARAMETER_COUNT)
    state = {"weight": torch.ones(1)} if model is None else model.state_dict()
    return dict(schema=gc.LEGACY_SCHEMA if legacy else gc.SCHEMA,
        native_meta=copy.deepcopy(NATIVE_META), context=context, model=state, optimizer={},
        stream_state=dict(seed=seed, arm="Native_correct", batch_index=start+completed,
                          sample_index=8*(start+completed)), stream_seed=seed,
        stream_start_batch_index=start, stream_start_sample_index=8*start,
        input_sha256="1"*64, config_sha256=OFFICIAL_CONFIG_SHA256,
        model_state_sha256="3"*64 if model is None else state_sha256(model),
        base_identity=dict(cumulative_updates=base, checkpoint_sha256="4"*64, state_sha256="5"*64),
        base_cumulative_updates=base, completed_updates=completed, cumulative_updates=base+completed,
        loss="masked_free_fm", num_faces=112, train_K=[4,8,12], optimizer_reset_at_start=True,
        optimizer_step_in_progress=False, batch_in_progress=False)


class GeometryCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = TinyNative(TinyBackbone(), trainable=True)
        self.model.context_encoder = TinyGeo()
        self.model.set_trainable(True)

    def constructors(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(gc, "DiT", TinyBackbone))
        stack.enter_context(mock.patch.object(gc, "NativeInpaintingModel", TinyNative))
        stack.enter_context(mock.patch.object(gc, "ContextGeometryEncoder", TinyGeo))
        stack.enter_context(mock.patch.object(gc, "configure_stable_runtime", return_value={}))
        return stack

    def write(self, payload, name="fixture.pt"):
        path = self.root / name
        torch.save(payload, path)
        return path

    def reject(self, payload):
        with self.assertRaises(ValueError):
            gc.validate_geometry_checkpoint(payload)

    def test_portable_arbitrary_progress_seed_and_stream_offset(self):
        for base, completed, start, seed in ((0,1,0,4),(2000,17,300,19),(13,0,5,0)):
            with self.subTest(base=base, completed=completed):
                p=valid_payload(base=base,completed=completed,start=start,seed=seed)
                self.assertEqual(gc.validate_geometry_checkpoint(p),p["context"])
        p=valid_payload();p["context"]["initialization_seed"]=17
        self.assertEqual(gc.validate_geometry_checkpoint(p)["initialization_seed"],17)

    def test_legacy_accepts_only_exact_geo_definition_and_progress(self):
        for step in (250,500):
            p=valid_payload(legacy=True)
            p.update(completed_updates=step,cumulative_updates=1500+step)
            p["stream_state"].update(batch_index=step,sample_index=8*step)
            self.assertEqual(gc.validate_geometry_checkpoint(p),p["context"])
        for change in (dict(base_cumulative_updates=0),dict(completed_updates=17),dict(stream_seed=19)):
            p=valid_payload(legacy=True);p.update(change);self.reject(p)
        p=valid_payload(legacy=True);p["context"]["initialization_seed"]=17;self.reject(p)

    def test_graph_and_wrong_architecture_rejected(self):
        for legacy in (False,True):
            for key,value in (("context_mode","graph"),("feature_dim",12),("graph_layers",2),
                              ("additional_parameters",149888.0),("initialization_seed",True),
                              ("branch_dtype","torch.bfloat16"),("injection","after_final")):
                with self.subTest(legacy=legacy,key=key):
                    p=valid_payload(legacy=legacy);p["context"][key]=value;self.reject(p)
        p=valid_payload();p["native_meta"]={};self.reject(p)
        p=valid_payload();p["schema"]="native_t1_training_v1";self.reject(p)

    def test_integer_progress_and_training_contract_rejected(self):
        changes=dict(completed_updates=True,base_cumulative_updates=-1,cumulative_updates=21,
            stream_seed=False,stream_start_batch_index=4,stream_start_sample_index=39,
            loss="FM_plus_FC",num_faces=112.0,train_K=[12],optimizer_reset_at_start=False,
            optimizer_step_in_progress=True,batch_in_progress=True)
        for key,value in changes.items():
            with self.subTest(key=key):
                p=valid_payload();p[key]=value;self.reject(p)
        for key,value in (("seed",43),("arm","Native_independent"),("batch_index",True),
                          ("sample_index",95)):
            p=valid_payload();p["stream_state"][key]=value;self.reject(p)
        p=valid_payload();p["base_identity"]["cumulative_updates"]=14;self.reject(p)

    def test_hash_state_and_config_identity_rejected(self):
        for key in ("input_sha256","config_sha256","model_state_sha256"):
            p=valid_payload();p[key]="BAD";self.reject(p)
        for key in ("checkpoint_sha256","state_sha256"):
            p=valid_payload();del p["base_identity"][key];self.reject(p)
        for state in ({},{"weight":torch.ones(1,dtype=torch.float64)},{"weight":"tensor"}):
            p=valid_payload();p["model"]=state;self.reject(p)
        with self.assertRaisesRegex(ValueError,"Config identity"):
            gc.validate_geometry_checkpoint(valid_payload(),"0"*64)

    def test_pure_loader_metadata_rejects_both_geo_schemas(self):
        for legacy in (False,True):
            with self.assertRaises(ValueError):
                validate_training_metadata(valid_payload(legacy=legacy))

    def test_portable_strict_load_and_identity_aliases(self):
        p=valid_payload(model=self.model);path=self.write(p)
        before=torch.random.get_rng_state().clone()
        with self.constructors():
            loaded,audit=gc.load_geometry_checkpoint(path,device="cpu")
        self.assertEqual(audit["schema"],gc.SCHEMA)
        self.assertEqual(audit["sha256"],file_sha256(path))
        self.assertEqual(audit["checkpoint_sha256"],audit["sha256"])
        self.assertEqual(audit["state_sha256"],state_sha256(self.model))
        self.assertEqual(audit["cumulative_updates"],20)
        self.assertEqual(audit["context"]["initialization_seed"],73)
        self.assertTrue(audit["strict"])
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_explicit_legacy_geo_strict_load(self):
        p=valid_payload(legacy=True,model=self.model);path=self.write(p)
        with self.constructors():
            loaded,audit=gc.load_geometry_checkpoint(path,device="cpu")
        self.assertTrue(audit["legacy_schema"])
        self.assertEqual(audit["schema"],gc.LEGACY_SCHEMA)
        self.assertEqual(audit["cumulative_updates"],2000)
        self.assertEqual(audit["context"],p["context"])
        self.assertEqual(state_sha256(loaded),p["model_state_sha256"])

    def test_strict_keys_shapes_hash_and_finite_checks(self):
        for kind in ("missing","unexpected","shape","hash","nonfinite"):
            with self.subTest(kind=kind):
                p=valid_payload(model=self.model)
                p["model"]={k:v.clone() for k,v in p["model"].items()}
                key="backbone.x_embedder.weight"
                if kind=="missing":del p["model"][key]
                elif kind=="unexpected":p["model"]["extra"]=torch.zeros(1)
                elif kind=="shape":p["model"][key]=torch.zeros(3,3)
                elif kind=="hash":p["model_state_sha256"]="0"*64
                else:
                    p["model"][key][0,0]=float("nan")
                    p["model_state_sha256"]=state_sha256(SimpleNamespace(state_dict=lambda:p["model"]))
                path=self.write(p,kind+".pt")
                with self.constructors(),self.assertRaises((ValueError,RuntimeError)):
                    gc.load_geometry_checkpoint(path,device="cpu")

    def test_graph_rejected_before_any_constructor(self):
        for legacy in (False,True):
            p=valid_payload(legacy=legacy,model=self.model);p["context"]["context_mode"]="graph"
            path=self.write(p,str(legacy)+".pt")
            with mock.patch.object(gc,"DiT") as construct,self.assertRaisesRegex(ValueError,"Graph"):
                gc.load_geometry_checkpoint(path,device="cpu")
            construct.assert_not_called()

    def test_save_actual_progress_and_continued_stream(self):
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=0.)
        stream=SimpleNamespace(seed=42,state_dict=lambda:dict(seed=42,arm="Native_correct",batch_index=12,sample_index=96))
        identity=dict(cumulative_updates=13,checkpoint_sha256="4"*64,state_sha256="5"*64)
        path=self.root/"saved.pt"
        before=torch.random.get_rng_state().clone()
        rec=gc.save_geometry_checkpoint(path,self.model,optimizer,stream,SimpleNamespace(sha256="1"*64),
            config_path=gc.DEFAULT_CONFIG,base_identity=identity,completed=7)
        payload=torch.load(path,map_location="cpu",weights_only=True)
        self.assertEqual(rec["cumulative_updates"],20)
        self.assertEqual(payload["stream_start_batch_index"],5)
        self.assertEqual(payload["stream_start_sample_index"],40)
        self.assertEqual(payload["context"]["initialization_seed"],73)
        self.assertEqual(payload["model_state_sha256"],state_sha256(self.model))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        with self.constructors():
            loaded,audit=gc.load_geometry_checkpoint(path,device="cpu")
        self.assertEqual(audit["state_sha256"],state_sha256(loaded))
        with self.assertRaises(FileExistsError):
            gc.save_geometry_checkpoint(path,self.model,optimizer,stream,SimpleNamespace(sha256="1"*64),
                config_path=gc.DEFAULT_CONFIG,base_identity=identity,completed=7)

    def test_save_explicit_base_without_implicit_private_budget(self):
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=0.)
        stream=SimpleNamespace(seed=42,state_dict=lambda:dict(seed=42,arm="Native_correct",batch_index=1,sample_index=8))
        identity=dict(checkpoint_sha256="4"*64,state_sha256="5"*64)
        rec=gc.save_geometry_checkpoint(self.root/"from_official.pt",self.model,optimizer,stream,
            SimpleNamespace(sha256="1"*64),config_path=gc.DEFAULT_CONFIG,base_identity=identity,
            base_updates=0,completed=1)
        self.assertEqual(rec["cumulative_updates"],1)
        with self.assertRaisesRegex(ValueError,"Base checkpoint progress"):
            gc.save_geometry_checkpoint(self.root/"conflict.pt",self.model,optimizer,stream,
                SimpleNamespace(sha256="1"*64),config_path=gc.DEFAULT_CONFIG,
                base_identity=dict(identity,cumulative_updates=100),base_updates=0,completed=1)
        self.assertFalse((self.root/"conflict.pt").exists())

    def test_save_rejects_incomplete_optimizer_and_graph(self):
        stream=SimpleNamespace(seed=42,state_dict=lambda:dict(seed=42,arm="Native_correct",batch_index=1,sample_index=8))
        kwargs=dict(config_path=gc.DEFAULT_CONFIG,base_identity=dict(cumulative_updates=0,
            checkpoint_sha256="4"*64,state_sha256="5"*64),completed=1)
        optimizer=torch.optim.AdamW(self.model.backbone.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=0.)
        with self.assertRaisesRegex(ValueError,"every Geo model parameter"):
            gc.save_geometry_checkpoint(self.root/"bad.pt",self.model,optimizer,stream,SimpleNamespace(sha256="1"*64),**kwargs)
        self.model.context_encoder.mode="graph"
        with self.assertRaisesRegex(ValueError,"Geo encoder"):
            gc.save_geometry_checkpoint(self.root/"bad.pt",self.model,optimizer,stream,SimpleNamespace(sha256="1"*64),**kwargs)

    def test_canonical_config_allows_crlf_not_changed_contents(self):
        p=valid_payload(model=self.model);path=self.write(p)
        config=self.root/"chair.yaml"
        text=gc.DEFAULT_CONFIG.read_text(encoding="utf-8-sig")
        config.write_bytes(text.replace(chr(10),chr(13)+chr(10)).encode())
        with self.constructors():
            _,audit=gc.load_geometry_checkpoint(path,config,device="cpu")
        self.assertEqual(audit["config_sha256"],OFFICIAL_CONFIG_SHA256)
        config.write_text(text+"# changed",encoding="utf-8")
        with self.assertRaises(ValueError):
            gc.load_geometry_checkpoint(path,config,device="cpu")


if __name__ == "__main__":
    unittest.main()
