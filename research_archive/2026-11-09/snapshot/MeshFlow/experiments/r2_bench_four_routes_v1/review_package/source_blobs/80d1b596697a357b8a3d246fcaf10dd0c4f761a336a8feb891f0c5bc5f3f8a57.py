"""Bounded CPU preflight: four registered OT packings, zero neural forward.

The source live generator, matching AdamW and RNG are authenticated/restored on
CPU. Contract fixtures never perform model calls or optimizer updates.
"""
from __future__ import annotations
import ast
import copy
import gc
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from . import common as C, runtime as R, stream as Q, training as M
from meshflow_control.training.schedule40 import make_optimizer


def contract():
    return dict(updates=1500,source_global_step=21000,final_global_step=22500,
                full_steps=[1500],checkpoint_every=250)


def pool():
    return C.read(C.OUT/"data/manifest.json")["train"]


def fixture_update(ledger,reg,step):
    for mi in range(8):
        context=dict(arm=ledger.arm,**M.cursor(reg,step),microbatch=mi)
        for kind in ("train_forward","train_backward"):
            ledger.call(kind,lambda:None,context)
    ledger.call("optimizer_update",lambda:None,dict(arm=ledger.arm,**M.cursor(reg,step)))


class ContractTests(unittest.TestCase):
    def test_exact_budget_and_cursors(self):
        reg=contract()
        self.assertEqual((M.cursor(reg,0)["global_step"],M.cursor(reg,0)["native_total_step"]),(21000,9000))
        self.assertEqual((M.cursor(reg,1500)["global_step"],M.cursor(reg,1500)["native_total_step"]),(22500,10500))
        self.assertEqual(M.limits(reg)["train_forward"],12000)
        for value in (-1,1501):
            with self.assertRaises(ValueError):M.cursor(reg,value)
        with self.assertRaises(ValueError):M.stage_spec(dict(reg,updates=3000))

    def test_frozen_stream_and_mutations(self):
        rows=pool();a,b=[Q.make_schedule(rows,arm) for arm in Q.ARMS]
        self.assertEqual(Q.validate_pair(a,b,rows)["status"],"PASS")
        for key,value in (("replay_group","HIGH"),("noise_seed",-1),("alpha",1.2),("parent_id","UNKNOWN")):
            bad=copy.deepcopy(a);bad["steps"][0]["samples"][0][key]=value
            with self.assertRaises(ValueError):Q.validate(bad,rows)
        with self.assertRaises(ValueError):Q.make_schedule(rows,Q.ARMS[0],3000)
        rows.append(dict(rows[0],condition_id="SECOND_C_SAME_PARENT"))
        with self.assertRaises(ValueError):Q.groups_for(rows)

    def test_stream_preserves_global_rng(self):
        n=np.random.get_state();t=torch.get_rng_state().clone()
        Q.make_schedule(pool(),Q.ARMS[1])
        now=np.random.get_state()
        self.assertEqual(n[0],now[0]);self.assertTrue(np.array_equal(n[1],now[1]));self.assertEqual(n[2:],now[2:])
        self.assertTrue(torch.equal(t,torch.get_rng_state()))

    def test_exact_boundary_resume(self):
        reg=contract()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"synthetic_calls.jsonl"
            a=M.T.historical.CallLedger(path,M.limits(reg),Q.ARMS[0],"synthetic_first",synchronize=False)
            M.require_cursor(a,reg,0);fixture_update(a,reg,1);M.require_cursor(a,reg,1)
            b=M.T.historical.CallLedger(path,M.limits(reg),Q.ARMS[0],"synthetic_resume",synchronize=False)
            M.require_cursor(b,reg,1);fixture_update(b,reg,2);M.require_cursor(b,reg,2)
            with self.assertRaises(RuntimeError):M.require_cursor(b,reg,1)
            with self.assertRaises(ValueError):M.T.historical.CallLedger(path,M.limits(reg),Q.ARMS[1],"wrong",synchronize=False)

    def test_partial_failed_calls_forbid_replay(self):
        reg=contract()
        with tempfile.TemporaryDirectory() as temp:
            a=M.T.historical.CallLedger(Path(temp)/"synthetic_partial",M.limits(reg),Q.ARMS[0],"partial",synchronize=False)
            a.call("train_forward",lambda:None,dict(**M.cursor(reg,1),microbatch=0))
            with self.assertRaises(RuntimeError):M.require_cursor(a,reg,0)
            with self.assertRaises(RuntimeError):a.call("head_forward",lambda:None,{})
        with tempfile.TemporaryDirectory() as temp:
            a=M.T.historical.CallLedger(Path(temp)/"synthetic_failure",M.limits(reg),Q.ARMS[0],"failed",synchronize=False)
            def fail():raise RuntimeError("Synthetic ledger fixture; not a neural call")
            with self.assertRaises(RuntimeError):a.call("train_forward",fail,dict(**M.cursor(reg,1),microbatch=0))
            with self.assertRaises(RuntimeError):M.require_cursor(a,reg,0)

    def test_only_original_global_coordinate_objective(self):
        source=Path(M.__file__).read_text()
        calls=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)]
        losses=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=="corner_masked_fm_loss"]
        self.assertEqual(len(losses),1)
        self.assertIn("denominator=sum(int(s['free_coordinate_count']) for s in samples)",source)
        self.assertIn("sample_for_training(cases[r['condition_id']],r,'PLAIN')",source)


def main():
    dest=C.OUT/"training_cpu_gate.json"
    names=("training.py","stream.py","runtime.py","train_tests.py")
    hashes={str(Path(__file__).with_name(n)):C.sha(Path(__file__).with_name(n)) for n in names}
    if dest.exists():
        old=C.read(dest)
        if old["source_hashes"] != hashes:
            raise ValueError("Completed bounded CPU preflight changed; no silent repeat")
        if old["runtime_cpu_gate_sha256"] != C.sha(C.OUT/"runtime_cpu_gate.json"):
            raise ValueError("Referenced runtime gate changed")
        return old
    ledger=C.OUT/"cpu_ot_calls.jsonl"
    if ledger.exists():
        raise RuntimeError("Prior CPU OT attempts exist; no unbudgeted replay")
    if torch.cuda.is_initialized():raise RuntimeError("CPU gate must not own CUDA")
    torch.set_num_threads(2)
    before=C.resources()
    runtime_receipt=R.cpu_gate()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if C.sha(C.SOURCE) != C.SOURCE_IDENTITIES["source_checkpoint_sha256"]:
        raise ValueError("Source B checkpoint bytes changed")
    reg=dict(contract(),**C.SOURCE_IDENTITIES,source_checkpoint=str(C.SOURCE))
    payload=M._source_payload(reg)
    model,optimizer=M._build_restore(payload,21000,"cpu")
    if len(optimizer.param_groups)!=1 or len(optimizer.state)!=194:
        raise ValueError("Source B optimizer structure changed")
    if any(float(s["step"])!=21000 for s in optimizer.state.values()):
        raise ValueError("Inherited optimizer steps differ")
    process_rng=M.rng_state()
    cpu_source_rng=copy.deepcopy(payload["rng"]);cpu_source_rng["torch_cuda"]=[]
    M.restore_rng(cpu_source_rng)
    if M.T.base.tree_hash(M.rng_state()) != M.T.base.tree_hash(cpu_source_rng):
        raise ValueError("Source CPU RNG did not roundtrip")
    M.restore_rng(process_rng)
    full_load=dict(schema=payload["schema"],global_step=payload["global_step"],
        native_total_step=payload["native_total_step"],optimizer_states=len(optimizer.state),
        optimizer_step=21000,optimizer_groups=1,generator_strict_load=True,
        optimizer_tensor_and_parameter_order_exact=True,CPU_RNG_restore_exact=True,
        saved_CUDA_RNG_arrays_authenticated_not_restored=len(payload["rng"]["torch_cuda"]),
        new_parameters=0,neural_model_forwards=0)
    del model,optimizer,payload;gc.collect()
    manifest=C.read(C.OUT/"data/manifest.json")
    rows={r["condition_id"]:r for r in manifest["train"]}
    schedules={a:Q.make_schedule(manifest["train"],a) for a in Q.ARMS}
    old_slot=schedules["CONTROL_OLD"]["steps"][0]["samples"][0]
    high_slots={}
    for entry in schedules["NATURAL_HIGH"]["steps"]:
        for row in entry["samples"]:
            if row["replay_group"]=="HIGH" and row["augmentation"]=="CLEAN":
                high_slots.setdefault(row["parent_id"],row)
    if len(high_slots)!=2:raise ValueError("Exactly two HIGH CPU samples required")
    jobs=[("OLD_ORIGINAL",old_slot,M.D.sample_for_training),("OLD_NEW_RUNTIME",old_slot,R.sample_for_training)]
    jobs.extend(("HIGH_"+p,r,R.sample_for_training) for p,r in sorted(high_slots.items()))
    actual=[];first=None
    for index,(label,row,fn) in enumerate(jobs,1):
        source=rows[row["condition_id"]]
        case=M.D.load_case(source)
        M.append_event(ledger,dict(event="CPU_OT_BEGIN",index=index,label=label,condition_id=row["condition_id"],
            augmentation=row["augmentation"],registered_stream_row=row,budget=4))
        sample=fn(case,row,"PLAIN")
        audit=R.verify_sample(sample)
        if int(sample["production_OT_calls"])!=1:raise ValueError("Exactly one production OT per CPU packing")
        compared=False
        if index==1:first=sample
        if index==2:
            if set(sample)!=set(first) or any(not np.array_equal(sample[k],first[k]) for k in sample):
                raise ValueError("Original and new old-range Native packing are not exact")
            compared=True
        M.append_event(ledger,dict(event="CPU_OT_COMPLETE",index=index,label=label,N=int(sample["N"]),K=int(sample["K"]),OT_calls=1))
        actual.append(dict(label=label,condition_id=row["condition_id"],N=int(sample["N"]),K=int(sample["K"]),
            augmentation=row["augmentation"],noise_seed=row["noise_seed"],old_vs_new_exact=compared,
            source_case_sha256=source["case_sha256"],audit=audit,OT_calls=1))
    records=[json.loads(s) for s in ledger.read_text().splitlines()]
    if sum(r["event"]=="CPU_OT_BEGIN" for r in records)!=4 or sum(r["event"]=="CPU_OT_COMPLETE" for r in records)!=4:
        raise ValueError("CPU OT attempt/completion budget differs")
    if torch.cuda.is_initialized():raise RuntimeError("CPU preflight initialized CUDA")
    after=C.resources()
    report=dict(status="PASS",schema="bench_natural_n_training_cpu_gate_v1",tests_run=result.testsRun,
        source_identities=C.SOURCE_IDENTITIES,retained_checkpoint_tensor_validation=full_load,
        runtime_cpu_gate=str(C.OUT/"runtime_cpu_gate.json"),runtime_cpu_gate_sha256=C.sha(C.OUT/"runtime_cpu_gate.json"),
        source_hashes=hashes,actual_cpu_samples=actual,OT_calls=4,OT_attempts=4,
        model_forwards=0,model_backwards=0,optimizer_updates=0,CUDA_initialized=False,
        CPU_only=True,CPU_threads=2,resources_before=before,resources_after=after,
        manifest_sha256=C.sha(C.OUT/"data/manifest.json"),CPU_OT_ledger_sha256=C.sha(ledger),
        neural_model_constructions=1,synthetic_ledger_fixtures_not_scientific_calls=True)
    C.write_new(dest,report)
    print(json.dumps({k:report[k] for k in ("status","tests_run","OT_calls","model_forwards","optimizer_updates","CUDA_initialized")}))
    return report


if __name__=="__main__":
    main()

