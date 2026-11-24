# Feed the AOTAutograd fwd/bwd graphs to coremltools' torch.export frontend; report unsupported ops.
import sys, traceback, torch, coremltools as ct
sys.argv = ["x"]; exec(open("01_capture_aot_graphs.py").read().split("graphs = {}")[0])  # reuse model defs
from torch._dynamo.backends.common import aot_autograd
from torch._decomp import core_aten_decompositions
print("torch", torch.__version__, "coremltools", ct.__version__)
G = {}; REAL = {}
def mk(tag):
    def comp(gm, ex):
        G[tag] = (gm, [(tuple(e.shape), e.dtype) for e in ex])
        def run(*a):
            REAL.setdefault(tag, [t.detach().clone() if isinstance(t, torch.Tensor) else t for t in a]); return gm.forward(*a)
        return run
    return comp
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch._decomp import get_decompositions
aten = torch.ops.aten
DEC = {**core_aten_decompositions(), **get_decompositions([aten.silu_backward])}
import os, json, collections
def _plain_softmax(x, dim, dtype=None):
    return aten._softmax(x if dtype is None else x.to(dtype), dim, False)
DEC_PLAIN = {**DEC, aten._safe_softmax.default: _plain_softmax}
VARIANTS = {"math_sdpa": DEC, "math_sdpa_plain_softmax": DEC_PLAIN}
CENSUS = {}
for tag, dec in VARIANTS.items():
    if os.environ.get("VARIANT") and os.environ["VARIANT"] != tag: continue
    torch._dynamo.reset(); G.clear(); REAL.clear(); torch.manual_seed(0)
    m = Block(); x = torch.randn(1, SEQ, DIM, requires_grad=True)
    cm = torch.compile(m, backend=aot_autograd(fw_compiler=mk("fw"), bw_compiler=mk("bw"), decompositions=dec), dynamic=False)
    with sdpa_kernel(SDPBackend.MATH):
        out = cm(x); loss = out.pow(2).mean(); loss.backward()
    ref = Block(); ref.load_state_dict(m.state_dict()); xr = x.detach().clone().requires_grad_(); ref(xr).pow(2).mean().backward()
    print(tag, 'grad err', float((x.grad - xr.grad).abs().max()))
    for g in ("fw", "bw"):
        gm, specs = G[g]
        for n in list(gm.graph.nodes):
            if n.op == 'call_function' and n.target in (aten.alias.default, aten.detach.default):
                n.replace_all_uses_with(n.args[0]); gm.graph.erase_node(n)
        gm.recompile()
        open(f"{tag}_{g}_graph.py.txt", "w").write(gm.code)
        CENSUS.setdefault(tag, {})[g] = {"n_inputs": sum(1 for n in gm.graph.nodes if n.op == "placeholder"),
            "aten_ops": dict(collections.Counter(str(n.target) for n in gm.graph.nodes if n.op == "call_function").most_common())}
        json.dump({"torch": torch.__version__, **CENSUS}, open("aten_op_census.json", "w"), indent=1)
        args = tuple(t.contiguous() for t in REAL[g])
        class W(torch.nn.Module):
            def __init__(s): super().__init__(); s.gm = gm
            def forward(s, *a): return tuple(o for o in s.gm(*a) if isinstance(o, torch.Tensor))
        try:
            ep = torch.export.export(W(), args).run_decompositions({})
        except Exception as e:
            print(f"[{tag}/{g}] export FAILED: {type(e).__name__}: {str(e)[:300]}"); continue
        try:
            ml = ct.convert(ep, convert_to="milinternal", minimum_deployment_target=ct.target.iOS18)
            import numpy as np
            mm = ct.convert(ep, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16)
            mm.save(f"{tag}_{g}.mlpackage")
            ins = [i.name for i in mm.get_spec().description.input]; outs = [o.name for o in mm.get_spec().description.output]
            ref = W()(*args)
            np.savez(f"{tag}_{g}_io.npz", **{f"in:{n}": a.numpy() for n, a in zip(ins, args)}, **{f"out:{n}": r.detach().numpy() for n, r in zip(outs, ref)})
            ops = {}
            for f in ml.functions.values():
                for op in f.operations: ops[op.op_type] = ops.get(op.op_type, 0) + 1
            print(f"[{tag}/{g}] CONVERTED. MIL ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
        except Exception as e:
            msg = str(e).strip().splitlines(); print(f"[{tag}/{g}] convert FAILED: {type(e).__name__}: {' | '.join(msg[:3])[:400]}")
