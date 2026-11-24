# Capture AOTAutograd fwd/bwd ATen graphs for a Llama-style block via a dumping torch.compile backend.
import json, collections, torch, torch.nn as nn, torch.nn.functional as F
from torch._dynamo.backends.common import aot_autograd
from torch._decomp import core_aten_decompositions
print("torch", torch.__version__)
DIM, HID, H, KVH, HD, SEQ = 256, 688, 8, 4, 32, 64
class RMSNorm(nn.Module):
    def __init__(s, d): super().__init__(); s.w = nn.Parameter(torch.ones(d))
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * s.w
def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1*cos - x2*sin, x1*sin + x2*cos), -1).flatten(-2)
class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1, s.n2 = RMSNorm(DIM), RMSNorm(DIM)
        s.wq = nn.Linear(DIM, H*HD, bias=False); s.wk = nn.Linear(DIM, KVH*HD, bias=False)
        s.wv = nn.Linear(DIM, KVH*HD, bias=False); s.wo = nn.Linear(H*HD, DIM, bias=False)
        s.w1 = nn.Linear(DIM, HID, bias=False); s.w3 = nn.Linear(DIM, HID, bias=False); s.w2 = nn.Linear(HID, DIM, bias=False)
        t = torch.arange(SEQ)[:, None] * (10000 ** (-torch.arange(0, HD, 2) / HD))[None]
        s.register_buffer("cos", t.cos()); s.register_buffer("sin", t.sin())
    def forward(s, x):
        B = x.shape[0]; h = s.n1(x)
        q = s.wq(h).view(B, SEQ, H, HD).transpose(1, 2); k = s.wk(h).view(B, SEQ, KVH, HD).transpose(1, 2)
        v = s.wv(h).view(B, SEQ, KVH, HD).transpose(1, 2)
        q, k = rope(q, s.cos, s.sin), rope(k, s.cos, s.sin)
        k = k.repeat_interleave(H // KVH, 1); v = v.repeat_interleave(H // KVH, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.wo(a.transpose(1, 2).reshape(B, SEQ, H*HD))
        h = s.n2(x)
        return x + s.w2(F.silu(s.w1(h)) * s.w3(h))
graphs = {}
def mk(tag):
    def comp(gm, ex):
        graphs[tag] = gm
        return gm.forward
    return comp
def run(tag, decomps, loss_fn):
    torch._dynamo.reset(); graphs.clear()
    torch.manual_seed(0); m = Block(); x = torch.randn(1, SEQ, DIM, requires_grad=True)
    be = aot_autograd(fw_compiler=mk("fw"), bw_compiler=mk("bw"), decompositions=decomps)
    cm = torch.compile(m, backend=be, dynamic=False)
    out = loss_fn(cm(x)); out.backward()
    ref = Block(); ref.load_state_dict(m.state_dict()); xr = x.detach().clone().requires_grad_()
    loss_fn(ref(xr)).backward()
    res = {"max_grad_err": float((x.grad - xr.grad).abs().max())}
    for g in ("fw", "bw"):
        gm = graphs[g]; c = collections.Counter(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        res[g] = {"n_inputs": sum(1 for n in gm.graph.nodes if n.op == "placeholder"), "ops": dict(c.most_common())}
        open(f"{tag}_{g}.txt", "w").write(gm.code)
    return res
out = {"torch": torch.__version__,
       "default": run("default", None, lambda y: y.pow(2).mean()),
       "core_aten": run("core_aten", core_aten_decompositions(), lambda y: y.pow(2).mean())}
json.dump(out, open("census.json", "w"), indent=1)
for k in ("default", "core_aten"):
    print(f"== {k}: grad err {out[k]['max_grad_err']:.2e}")
    for g in ("fw", "bw"): print(f"  {g} inputs={out[k][g]['n_inputs']}", out[k][g]["ops"])
