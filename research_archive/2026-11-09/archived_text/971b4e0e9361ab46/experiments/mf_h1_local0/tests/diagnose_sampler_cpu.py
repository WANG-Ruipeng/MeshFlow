"""Read-only CPU diagnosis of sampler time/discretization semantics.

No official checkpoint/model, GPU calls, training or file modifications outside
this self-contained diagnostic report. A simple deterministic arithmetic vector
field tests the integrators; it is not a substitute MeshFlow model.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import torch
from torchdiffeq._impl.fixed_grid import Euler
from torchdiffeq._impl.solvers import FixedGridODESolver
from flow_matching import FlowMatching
from experiments.mf_h1_local0.sample import euler_sample


def error(a, b):
    return {"bitwise_equal": bool(torch.equal(a,b)),
            "max_abs": float((a.double()-b.double()).abs().max()),
            "mismatched_elements": int((a!=b).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    formal_grid = torch.tensor([k/50 for k in range(51)],dtype=torch.float32)
    official_grid = torch.linspace(0,1,51,dtype=torch.float32)
    official_dts = official_grid[1:]-official_grid[:-1]
    formal_dt = torch.full((50,),1/50,dtype=torch.float32)
    source_paths = {"FlowMatching": inspect.getfile(FlowMatching),
                    "Euler": inspect.getfile(Euler),
                    "FixedGridODESolver": inspect.getfile(FixedGridODESolver),
                    "euler_sample": inspect.getfile(euler_sample)}
    transport = FlowMatching(prediction="velocity")
    transport.method = "euler"
    transport.atol, transport.rtol = 1e-6, 1e-5
    seed = 20260928
    state = torch.randn((1,8,9),generator=torch.Generator().manual_seed(seed))
    y = torch.tensor([8])
    mask = torch.ones((1,8),dtype=torch.bool)
    calls = {"reference_raw":0,"reference_manual":0,"formal":0,"reference_integer_clock":0}
    current = ["formal"]
    def arithmetic_field(x,t,**kwargs):
        calls[current[0]] += 1
        return .3*x + torch.sin(17*t[:,None,None])
    direct, _ = euler_sample(arithmetic_field,state,y,mask)
    current[0] = "reference_raw"
    reference = transport.sample(state.clone(),arithmetic_field,num_steps=51,model_kwargs={"y":y,"mask":mask})
    current[0] = "reference_manual"
    manual = state.clone()
    manual_states = [manual]
    for start,stop in zip(official_grid[:-1],official_grid[1:]):
        time = torch.full((len(state),),float(start),dtype=state.dtype)
        dt = stop-start
        manual = manual + dt*arithmetic_field(manual,time,y=y,mask=mask)
        manual_states.append(manual)
    current[0] = "reference_integer_clock"
    model_times = []
    def integer_clock_model(x,s,**kwargs):
        integer = int(float(s[0]))
        assert bool((s == integer).all())
        time = torch.full((len(x),),integer/50,dtype=x.dtype)
        model_times.append(float(time[0]))
        return arithmetic_field(x,time,**kwargs)*(1/50)
    integer_reference = transport.sample(state.clone(),integer_clock_model,t0=0,t1=50,num_steps=51,
                                         model_kwargs={"y":y,"mask":mask})
    rows = [{"k":k,"formal_t_fp32":float(formal_grid[k]),"official_t_fp32":float(official_grid[k]),
             "node_abs_difference":float(abs(formal_grid[k].double()-official_grid[k].double())),
             "formal_dt_fp32":float(formal_dt[k]),"official_dt_fp32":float(official_dts[k]),
             "dt_abs_difference":float(abs(formal_dt[k].double()-official_dts[k].double()))}
            for k in range(50)]
    report = {"status":"DIAGNOSIS_COMPLETE_CPU_ONLY", "torch_version":torch.__version__,
        "inputs":{"seed":seed,"shape":[1,8,9],"dtype":"float32","device":"cpu",
                  "field":"0.3*x + sin(17*t)","field_is_not_official_model":True},
        "source_files":{key:{"path":file,"sha256":hashlib.sha256(Path(file).read_bytes()).hexdigest()}
                        for key,file in source_paths.items()},
        "source_evidence":{"Euler_step":inspect.getsource(Euler._step_func),
            "solver_integrate":inspect.getsource(FixedGridODESolver.integrate),
            "solver_interpolate":inspect.getsource(FixedGridODESolver._linear_interp),
            "official_sample":inspect.getsource(FlowMatching.sample)},
        "timegrid_comparison":error(formal_grid,official_grid),
        "dt_comparison":error(formal_dt,official_dts),"steps":rows,
        "raw_reference_vs_exact_source_semantics_manual_full_trajectory":error(reference,torch.stack(manual_states)),
        "raw_reference_vs_formal_final":error(reference[-1],direct),
        "integer_clock_reparameterized_reference_vs_formal_final":error(integer_reference[-1],direct),
        "integer_reference_model_physical_times":model_times,
        "model_times_equal_formal":bool(torch.equal(torch.tensor(model_times),formal_grid[:-1])),
        "arithmetic_field_call_counts":calls,
        "official_model_forwards":0,"official_complete_rollouts":0,"optimizer_updates":0,
        "conclusions":[
            "Installed torchdiffeq Euler uses adjacent FP32 node differences, not a constant FP32 dt.",
            "FP32 linspace nodes differ from individually rounded k/50; the grids are only equal as real-number specifications.",
            "Raw reference-vs-formal rollout discrepancy is not yet attributed solely to timegrid until real-model stepwise diagnosis is completed.",
            "A source-semantics manual loop must first match the unchanged original sampler with identical representable nodes and adjacent dt.",
            "An integer solver clock s=0..50 with dx/ds=(1/50)*v(x,s/50) yields identical physical model times and constant formal updates, without editing official sampler source.",
            "The integer-clock audit is a disclosed mathematical reparameterization, not the untouched raw [0,1] call. Preserve the original FAIL and report both separately.",
            "Formal sampling remains t_k=k/50 and constant dt=1/50. Do not increase thresholds or substitute adjacent-grid dt in training/evaluation."
        ]}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as out:
        json.dump(report,out,indent=2)
        out.write("\n")
    print(json.dumps({key:report[key] for key in ["timegrid_comparison","dt_comparison",
            "raw_reference_vs_exact_source_semantics_manual_full_trajectory",
            "raw_reference_vs_formal_final","integer_clock_reparameterized_reference_vs_formal_final",
            "arithmetic_field_call_counts"]}))

if __name__ == "__main__":
    main()
