import numpy as np, coremltools as ct, glob, os, sys
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
C, S, O = 64, 32, 48
# Dynamic-weight style: activations and W packed into ONE input along the last axis (repo trick)
@mb.program(input_specs=[mb.TensorSpec((1, C, 1, S + O), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog(x):
    a = mb.slice_by_size(x=x, begin=[0, 0, 0, 0], size=[1, C, 1, S])
    w = mb.slice_by_size(x=x, begin=[0, 0, 0, S], size=[1, C, 1, O])
    a2 = mb.transpose(x=mb.reshape(x=a, shape=[1, 1, C, S]), perm=[0, 1, 3, 2])
    w2 = mb.reshape(x=w, shape=[1, 1, C, O])
    y = mb.matmul(x=a2, y=w2)                                       # [1,1,S,O]
    ms = mb.reduce_mean(x=mb.mul(x=y, y=y), axes=[-1], keep_dims=True)
    y = mb.mul(x=y, y=mb.rsqrt(x=mb.add(x=ms, y=np.float16(1e-5))))  # RMSNorm
    y = mb.silu(x=y)
    ev = mb.slice_by_index(x=y, begin=[0, 0, 0, 0], end=[1, 1, S, O], stride=[1, 1, 1, 2])  # RoPE-style stride-2
    sm = mb.softmax(x=y, axis=-1)
    return sm, ev
mlm = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                 compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY)
mlm.save("probe.mlpackage")
cpath = ct.models.utils.compile_model("probe.mlpackage", destination_path="probe.mlmodelc")
print("compiled to", cpath); print(sorted(os.listdir(cpath)))
mil = open(os.path.join(cpath, "model.mil")).read() if os.path.exists(os.path.join(cpath, "model.mil")) else None
print("---- model.mil ----"); print(mil if mil else "NO model.mil")
x = (np.random.RandomState(0).randn(1, C, 1, S + O) * 0.3).astype(np.float16)
out = mlm.predict({"x": x})
a = x[0, :, 0, :S].astype(np.float32).T; W = x[0, :, 0, S:].astype(np.float32)
y = a @ W; y = y / np.sqrt((y * y).mean(-1, keepdims=True) + 1e-5); y = y / (1 + np.exp(-y))
e = np.exp(y - y.max(-1, keepdims=True)); sm = e / e.sum(-1, keepdims=True)
names = list(out.keys()); print("outputs", {k: v.shape for k, v in out.items()})
for k, v in out.items():
    ref = sm if v.shape[-1] == O else y[:, ::2]
    print(k, "max abs err vs numpy fp32:", float(np.abs(v.reshape(ref.shape) - ref).max()))
