# Run on a Mac host (Intel works): load fw/bw .mlpackage from step 03, predict on Core ML CPU (fp16),
# compare with torch fp32 reference outputs, and dump op census of the compiled model.mil.
import sys, re, collections, numpy as np, coremltools as ct
d = sys.argv[1] if len(sys.argv) > 1 else "."
tag = sys.argv[2] if len(sys.argv) > 2 else "math_sdpa_plain_softmax"
for g in ("fw", "bw"):
    io = np.load(f"{d}/{tag}_{g}_io.npz")
    m = ct.models.MLModel(f"{d}/{tag}_{g}.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
    ins = {k[3:] for k in io.files if k.startswith("in:")}
    out = m.predict({k: io["in:" + k] for k in ins})
    rows = []
    for k in io.files:
        if not k.startswith("out:"): continue
        r, o = io[k], out[k[4:]]
        passthru = any(io["in:" + i].shape == r.shape and np.array_equal(io["in:" + i], r) for i in ins)
        rows.append((k[4:], float(np.abs(o - r).max() / (np.abs(r).max() + 1e-12)), passthru))
    real = [x for x in rows if not x[2]]
    print(f"{g}: {len(real)} compared ({len(rows)-len(real)} input pass-throughs skipped); "
          f"worst rel err {max(x[1] for x in real):.2e}, median {np.median([x[1] for x in real]):.2e}")
    c = ct.models.utils.compile_model(f"{d}/{tag}_{g}.mlpackage", destination_path=f"{d}/{tag}_{g}.mlmodelc")
    mil = open(f"{c}/model.mil").read()
    print("   model.mil ops:", dict(collections.Counter(re.findall(r"= (\w+)\(", mil)).most_common()))
