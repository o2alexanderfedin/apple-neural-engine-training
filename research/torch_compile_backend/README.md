# Research: a `torch.compile` backend for the ANE

Status: research spike, 2026-11-23. Done on an **Intel Mac** (Xeon W-3245, macOS 26.2), which has no ANE.

Every claim below carries one tag:

- **[verified]**: run and checked on this machine.
- **[repo]**: taken from this repo's code or docs, not re-checked here.
- **[prior-art]**: from a summary of another project's paper or README, not checked against their code.
- **[code]**: read in another project's source or docs under `~/projects/apple-neural-engine/`. The file:line references are relative to that repo. Nothing was run.
- **[needs-AS]**: needs an Apple Silicon Mac to check.

**Scope of the spike:** one Llama-style transformer block, batch 1, dim 256, seq 64, with GQA 8/4 heads, interleaved RoPE, causal SDPA and SwiGLU. It does not cover the embedding, the LM head, cross-entropy, the optimizer, or larger shapes.

## Summary

The software path from PyTorch to MIL text works today with stock tools, for both the forward and the **backward** graph of the block:

```
torch.compile(backend=…)                    # Dynamo, fixed shapes
  → AOTAutograd (fw graph + bw graph)       # core_aten decomps + math SDPA + _safe_softmax→softmax
  → cleanup pass (strip alias/detach, contiguous inputs, drop None outputs)
  → torch.export → coremltools 9.0 → MIL program
  → Core ML compile → model.mil             # same header and dialect as the repo's hand-written kernels
```

On Core ML **CPU**, fp16 results match torch fp32 within 8e-4 (fw) and 5e-3 (bw) relative error.

A code review of 11 related repos (section 6) settles most of the hardware questions:
- **Multiple inputs work.** The "0x1d" limit was most likely a compile-cache collision.
- **The compile limit can be avoided** by compiling once with `ANECCompile`, saving the `.hwx`, and reloading it.
- **Ops:** on an M5, every op the converter emits runs on the ANE except `slice_update`, `cast` and (probably) strided `slice_by_index`. All three can be rewritten or avoided.
- **Biggest open risk: M1.** One M1 source reports that reshape, transpose, slice, concat, reduce, rsqrt and pow fail when weights are passed as runtime inputs. Another source says they work on M1.
- **ANEForge** (pure Python MIL emitter, numpy only, per-chip rewrites, on-ANE training) is a strong lowering target to evaluate before building on coremltools (section 7).

## Findings

### 1. Environment
- **[verified]** The newest PyTorch wheel for Intel macOS is **2.2.2**. Current PyTorch is **2.15.0** (wheels for arm64 macOS and Linux only). The capture and conversion half doesn't need macOS, so it runs in a Linux Docker container with torch 2.15. The Core ML compile and predict steps run on the Mac host with coremltools 9.0 and Python 3.12.
- **[verified]** This Intel Mac ships `AppleNeuralEngine.framework`, `ANECompiler.framework`, `MIL.framework` and `MilAneflow.framework`. `_ANEDeviceInfo` reports `hasANE=0`, `numANEs=0`, `aneArchitectureType=h11`. `_ANEInMemoryModelDescriptor modelWithMILText:` **returned nil** for a valid MIL program. I did not isolate the cause, so the private in-memory path can't be used on this machine. Don't probe it further on Intel.

### 2. Capture: which ATen ops a transformer block produces
Files: `spike/aten_op_census.json` and `spike/<variant>_{fw,bw}_graph.py.txt`, all from the variants that actually get converted.

- **[verified]** With default decompositions, SDPA stays fused as `_scaled_dot_product_flash_attention_for_cpu[_backward]`. That op is specific to the device, returns a tuple, and coremltools rejects it.
- **[verified]** `core_aten_decompositions()` combined with the default flash SDPA gives **wrong gradients**. The max abs error on `x.grad` is 1.7e-3, while the largest `x.grad` value is 5.4e-4, so the error is about 3× the gradient. My hypothesis, not checked: the decomposed forward does not produce what the fused flash backward expects. Fix: **math SDPA** (`sdpa_kernel(SDPBackend.MATH)`). Gradients then match eager to **8.7e-11**.
- **[verified]** With math SDPA, PyTorch 2.15 uses `_safe_softmax`. It adds a check that runs on the data, for rows that are entirely masked: `eq(-inf) → logical_not → any → where(full_like 0)`. That check survives conversion as MIL `equal/logical_not/reduce_max/fill_like/select`. Decomposing `aten._safe_softmax` to plain `aten._softmax` removes all five ops. The gradient error stays at 8.7e-11, because a causal mask never masks a whole row. The causal mask itself (`arange/le/where`) **does** fold to a constant; no `arange` survives into MIL.
- **[verified]** Graph inputs: the forward takes **12** (x, 9 parameters, 2 RoPE buffers). The backward takes **35** (saved activations, tangent, weights).

### 3. MIL text format
- **[verified]** coremltools 9.0 builder → `ct.convert(mlprogram)` → Core ML compile (`MLModel.compileModelAtURL`, called by coremltools) writes `model.mil` on this Mac. Its header (`program(1.3)`, `coremlc-component-MIL 3510.2.1`, `coremlc-version 3505.4.1`, `coremltools-version 9.0`) is **byte-identical** to `MIL_HDR` in `training/training_dynamic/mil_dynamic.h`. The function signature (`func main<ios18>`) and op syntax are the same dialect. `inmem_basic.m` also takes its MIL from a compiled `.mlmodelc` **[repo]**.
- **[verified]** `compileModelAtURL` works on macOS 26.2 (Intel). The benchmark report says it is broken on 26.x **[repo]**; that may be specific to Apple Silicon or to a narrower case.
- **[verified]** Graph rewrites such as folding `transpose` into `matmul(transpose_x=true)`, or `sigmoid·x` into `silu`, are done by **coremltools' passes**: the `.mlpackage` already contains them. In this spike, `model.mil` matches the mlpackage program.

### 4. Lowering fw + bw through coremltools
Script: `spike/03_convert_aot_graphs_to_mil.py`.
- **[verified]** These blocked conversion, each with a simple fix:
  - `aten.alias` / `aten.detach`: remove them (rewire their users to the input).
  - `aten.silu_backward`: add it to the decomposition table.
  - The tuple returned by fused SDPA: use math SDPA.
  - `None` outputs (gradients of buffers): filter them out.
  - Non-contiguous saved tensors as inputs: call `.contiguous()`.
- **[verified]** Both graphs convert. MIL ops for the recommended variant (`math_sdpa_plain_softmax`):
  - fw: reshape, mul, cast, transpose, matmul, add, expand_dims, slice_by_index, pow, reduce_mean, rsqrt, sub, concat, tile, softmax, silu
  - bw: mul, cast, reshape, matmul, add, reduce_sum, transpose, slice_by_index, **slice_update**, sub, pow, tile, squeeze, sigmoid
  - The `cast` ops are fp32↔fp16 conversions at the boundaries. They go away if the backend does its I/O in fp16 **[needs-AS]**.
- **[verified]** Numerics, Core ML **CPU** fp16 against torch fp32, on the real tensors from a training step (`spike/04_compare_on_coreml_cpu.py`):
  - fw, 25 outputs: worst relative error **8.1e-4**, median 3.5e-4.
  - bw, 10 gradient outputs: worst relative error **5.0e-3**, median 1.5e-3.
  - The largest gradient values are only 2.5e-4 to 2.6e-3, already close to where fp16 loses precision, so loss scaling is needed for real models. This agrees with the repo's `256 * NLAYERS` loss scale **[repo]**.
  - Two harness traps, avoided above: Core ML returns **zeros** for outputs that only pass a graph input through, and random N(0,1) "saved activations" overflow fp16 in the backward (up to 4e8). Always test with the real captured tensors.
- These numbers come from Core ML's CPU path. The ANE's fp16 accumulation is different hardware **[needs-AS]**.


### 5. Ops used on the ANE in this repo
I counted MIL ops in the training pipeline's MIL generators only (`mil_dynamic.h`, `stories_mil.h`, `ane_classifier.h`, `ane_rmsnorm_bwd.h`, `ane_mil_gen.h`), checked with positive controls (`matmul`=38, `softmax`=8, `conv`=54 hits). Result: reshape, mul, slice_by_size, transpose, matmul, conv, add, concat, cast, reduce_sum, softmax, pow, sub, sigmoid, exp **[repo]**. The converter emits ops that appear **nowhere** in this repo: `expand_dims, slice_by_index, tile, reduce_mean, rsqrt, silu, slice_update, squeeze`. Section 6.3 checks them against ANEForge's data.

### 6. Evidence from related repos (code review)
All the repos are forked and cloned under `~/projects/apple-neural-engine/` (see `~/projects/apple-neural-engine/README.md`). Most of this comes from M4/M5 machines. Facts specific to M1 are marked.

#### 6.1 Multiple inputs: they work
- ane-train runs **up to 98 inputs on M1** (`ane-train/ANE_TRAINING.md:43`) **[code]**. YuE-Studio binds ~30 inputs per program on M4 (`YuE-Studio/docs/apple-silicon.md:222-236`) **[code]**. zpeng11/ANE got bit-exact 2-input results on M4 (upstream issue #56). silicon-ledger EXP-041 got bit-exact results for 1–4 inputs, including unequal sizes, on M4 and M5 under macOS 27.0.1, with no 0x1d (`silicon-ledger-public/results/EXP-041-multi-input/`) **[code]**.
- **The likely origin of "0x1d / single-input only":** `hexStringIdentifier` (the compile-cache key) is derived from the MIL text. Test variants that share MIL text but differ in weights hit a stale bundle and fail with `Code=10 verifyBundleAtPath` (EXP-041; issue #56) **[code]**. **Backend rule:** make the MIL text unique per weight set, for example by putting a hash in the weight file names.
- **The binding order is disputed.** These three conflict; test on the M1 first:
  - ane-train (M1): input slots must be in **ascending size** and outputs descending, otherwise silent zeros (`ANE_TRAINING.md:27-32`).
  - Orion: **alphabetical** by name, with uniform sizes (docs only; the code just follows a naming convention).
  - YuE (M4): binds in MIL signature order (coremltools sorts names), with unequal sizes, and it works.
  - Mitigation: read the compiled model's input symbols (`modelAttributes["ANEFModelDescription"]`, per `ane-infer/docs/ane-internals.md:35-59`) instead of assuming an order.
- So the repo's trick of packing every input into one IOSurface is **not required**. It may still be worth keeping for very small tensors, given the minimum surface sizes: ~2 KB on M1 (ane-train `:35`), and 49 KB in Orion's validator (`Orion/compiler/pass_ane_validate.c:22-56`).

#### 6.2 Compile limit and compile cost
- **Compile once, load the binary:** call `ANECCompile(input, flags, NULL)` from `ANECompiler.framework` in your own process. It writes `model.hwx`. Then `[_ANEModel modelAtURL:hwx key:@""]` → `[inMem setModel:]` → `loadWithQoS:`, with no `compileWithQoS:`. The output is bit-identical to a service compile, and loading takes 0.01–0.1 s (`YuE-Studio/src/yue2/ane/libyue2ane.m:103-135,175-181`, M4) **[code]**. Cache one file per (arch, OS build). The target arch comes from `_ANEDeviceInfo.aneSubTypeAndVariant`; the M1 string isn't recorded anywhere I found.
- Compiler options take effect **only** through the `ANECCompile` flags dictionary, not the descriptor's options plist. Speed flags (`MemCacheStrategy=None`, `DisableCpAllocation`, …) gave a 3× faster compile with identical output (`YuE-Studio/docs/apple-silicon.md:353-390`) **[code]**.
- ane-train caches the compiled temp directory by `hexStringIdentifier` and calls only `loadWithQoS` on a cache hit (`ane-train/ane_runtime.h:49-131`, M1) **[code]**.
- **Drop Orion's "delta compilation".** The path it actually trains with (`Orion/core/ane_runtime.m:344-386`) unloads, overwrites the weight files and reloads, which is the same thing this repo's M5 test found to have no effect. No test in Orion checks that the output changes, and Orion's own `docs/ane_constraints.md:77-81` says reloading doesn't change outputs. Its `exec()` restart doesn't exist in its code either (`git log -S execv` finds nothing) **[code]**.
- Cold compiles take ~1.6–1.8 ms per conv. A graph that will be refused can still take 2.5–23 s before it fails (silicon-ledger EXP-042) **[code]**. After a failed compile, the next one is delayed by ~15 s (`ANEForge/aneforge/_circuit.py`) **[code]**.

#### 6.3 Op support on the ANE (ANEForge data)
ANEForge's sweep (`ANEForge/aneforge/full_mil_vocabulary_sweep.json`, M5) and its per-chip catalog (`aneforge/_op_catalog.py`, `docs/op-catalog.md`) **[code]**:

| Ops from our converter | M5 sweep | Per-chip catalog | Plan |
|---|---|---|---|
| reshape, mul, add, sub, pow, matmul, concat, expand_dims, squeeze, reduce_sum, reduce_mean, rsqrt, sigmoid, silu, tile, transpose | compile, run, correct | native M1–M5 | use as is (tile factors only {2,3,4,8}) |
| softmax | not in the sweep | native everywhere | use |
| slice_by_size | correct | native, but on **M1 a nonzero last-axis begin clamps values above 4094 to ±inf** (`_targets.py:188`, `docs/cross-chip.md:118-129`) | matters for the repo's packed-weight slices; ANEForge rewrites them as transpose→slice→transpose |
| slice_by_index (strided, RoPE) | correct | "~" | avoid: use half-split RoPE, or reshape `[d/2,2]` → slice → squeeze as ANEForge's ONNX importer does (`onnx.py:897-901`) |
| slice_update (RoPE backward) | **fails to compile** | "native" (conflict) | rewrite as slice + concat, as ANEForge's own backward does (`autograd.py:424-444`) |
| cast | **fails to compile** | fp16↔fp32 native on M1 (conflict) | do I/O in fp16 so no casts are emitted |

- `concat` works. Orion's "concat rejected" is probably a malformed op: this repo's working concat passes `interleave`, and Orion's failing version wasn't committed (`Orion` commit `dfa9349`) **[code]**.
- Other limits worth enforcing in the partitioner:
  - rank ≤ 5, axis size ≤ 65536, max dim 16384 on A13-class chips (`ANEForge/docs/developer/op-quirks.md:9-11`, `_targets.py:161-166`)
  - matmul output becomes inf above ~32752 (`docs/numeric-cliffs.md:14`)
  - native SDPA gives garbage when both seq lengths are ≥ 512 or S > 2048 (`op-quirks.md:52-53`)
  - no `bias=` on conv, no `gelu` (use the tanh form), matmul transpose flags must be named consts (`Orion/compiler/codegen.m:69,116-121`)
  - **[code]** for all of these.

#### 6.4 M1-specific warnings (the user's test machine)
- **Conflict to settle first:** ane-train (M1, runtime-weight programs) lists reshape, transpose, slice, gather, concat, split, pad, reduce_*, rsqrt and pow as failing, and "two runtime 4D inputs fail". It uses 3D tensors and builds LayerNorm from a matmul against a ones vector (`ane-train/ANE_TRAINING.md:17,116-127`; `modules/ops/ane_ln.h:47-60`). That list is a table without probe code. ANEForge's catalog marks most of those ops native on M1. This repo's dynamic pipeline, which relies on reshape, transpose and slice with runtime weights, has only been benchmarked on M3 and newer; the M1 results in the benchmark report are for the static pipeline **[repo]**.
- M1 matmul: the inner dim must be a multiple of 32, otherwise silent zeros; with runtime operands, the weight goes in slot 0 and the activation in slot 1 (`ANE_TRAINING.md:51-61`) **[code]**.
- M1 rejects MIL that refers to a shared `weight.bin` by offset **[repo]**. The spike's generated `model.mil` uses that format: 1 `BLOBFILE` reference in fw, 2 in bw **[verified]**.
- M1 dispatch floor is 0.19–0.23 ms (claimed by ane-guide, arXiv 2606.22283), compared with 0.07–0.1 ms on M4/M5 **[prior-art]**.
- On M1, ANEForge warns against `loss_scale` ≥ 512 with conv weight gradients (`autograd.py:691-720`) **[code]**.

#### 6.5 Dispatch
- The cost per call is roughly fixed: about 74 µs on M5 Pro regardless of fusion (ANEForge paper), and 106 µs with `doEvaluateDirectWithModel:` against 117 µs with `evaluateWithQoS:` on M5 (`ane-infer/docs/ane-internals.md:64-83`) **[code]**. Fusion is the main performance lever.
- **Chaining is not usable:** `_ANEChainingRequest` stalls at `buffersReady` (ane-infer `:187-194`) and is rejected at the driver (silicon-ledger EXP-035). ane-train's "verified working" claim has no code behind it **[code]**. Multi-procedure MIL (several `func`s, chosen by `procedureIndex`) does work (ane-infer `:35-59`) **[code]**.

### 7. Lowering options
- **A. coremltools → MIL text → `_ANEInMemoryModel`** (this repo's runtime). Verified here up to `model.mil` (sections 3–4). Needs our own rewrites (slice_update, strided slices, casts, M1 slice clamp), our own fp16 I/O, and an ANECCompile/.hwx cache.
- **B. FX → ANEForge graph.** ANEForge is pure Python (numpy only) and writes MIL text from its own graph (`ANEForge/aneforge/_compile.py:50-56,906-924`). It already includes per-chip workaround rewrites and runtime-weight training with loss scaling and on-ANE Adam (`autograd.py`). It dispatches through Espresso's e5rt API rather than `_ANEClient` (`_lib/ane_e5rt_dispatch.mm:110`). Its ONNX importer (`aneforge/onnx.py`, an `@onnx_op` handler registry) is the template for an FX importer. MIL emission is pure Python and should run on an Intel Mac (`_compile.py:933-946`) **[code]**. Two mismatches: AOTAutograd hands us a backward graph made of ATen ops, which must map onto ANEForge ops (we don't use its VJP); and a runtime-weight `x @ W` becomes `bmm`, not `matmul` (`graph.py:166-176`).
- **Recommendation:** evaluate B first. It already holds most of the per-chip knowledge that A would have to rediscover. Keep A as the fallback and as the cross-check for numbers (Core ML CPU).

## Proposed architecture (not validated)
1. **Backend entry point:** `aot_autograd(fw_compiler, bw_compiler, decompositions=core_aten + {silu_backward, _safe_softmax→_softmax})`, with `dynamic=False` and math SDPA.
2. **Cleanup pass** (section 4), then **rewrites** for the ANE: slice_update → slice+concat, strided slice → reshape/slice/squeeze, fp16 boundaries, and no conv bias.
3. **Partition** with `CapabilityBasedPartitioner`, using the support table for each chip (6.3/6.4). Unsupported parts run in eager on the CPU. Make each ANE part as large as possible (fixed ~0.1–0.2 ms per call).
4. **Lower** each ANE part through option B (or A).
5. **Inputs:** separate IOSurfaces bound by the compiled model's input symbols, with MIL text made unique per weight set. Weights are always inputs, never constants, so optimizer steps don't recompile.
6. **Runtime:** a `.hwx` cache keyed by (MIL hash, arch, OS build), compiled with `ANECCompile` and speed flags, loaded with `setModel:` + `loadWithQoS:`. The compile limit then only applies the first time on a new machine.

## Risks
- **M1 op support with runtime weights** (6.4) could force much more work onto the CPU on your test machine. Settle this first.
- Binding order (6.1) is disputed and fails *silently*. Every kernel needs a CPU oracle test the first time.
- **Utilization:** the hand-fused repo reaches 5–9% of peak **[repo]**. Generic graphs will start lower. Compare against an MPS baseline early.
- **Private APIs** can break with any macOS update **[repo]**. ANEForge's e5rt path is a second private surface with the same risk.

## Checklist for the M1 MacBook Pro
1. **Op support with runtime weights:** for each op in 6.3/6.4, compile and run a tiny kernel whose weights are inputs, and compare against CPU. This settles the ane-train vs ANEForge conflict. ANEForge's sweep tooling may already do this.
2. **Binding order:** a 3-input kernel with unequal sizes, in name order against size order. Confirm with an oracle.
3. **ANECCompile + .hwx load:** get the M1 arch string from `_ANEDeviceInfo`, compile once, reload in a fresh process, and compare outputs.
4. Compile the spike's `math_sdpa_plain_softmax_fw` `model.mil` after rewriting away its `BLOBFILE` offset references. Compare with Core ML CPU.
5. Dispatch floor: back-to-back evals of a no-op kernel; compare with the claimed 0.19–0.23 ms.

## Reproducing the spike
```bash
cd research/torch_compile_backend/spike
# Capture + convert (Linux container, current torch). Writes <variant>_{fw,bw}.mlpackage, *_io.npz, graphs, census.
docker run --rm -v "$PWD":/w -w /w python:3.12-slim sh -c \
  "pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && pip install coremltools numpy && \
   python 03_convert_aot_graphs_to_mil.py"
# Compile + compare on the Mac host (Intel or Apple Silicon)
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python coremltools numpy
.venv/bin/python 02_mil_text_probe.py                                  # MIL header / dialect check
.venv/bin/python 04_compare_on_coreml_cpu.py . math_sdpa_plain_softmax  # numerics + model.mil op census
```
`01_capture_aot_graphs.py` is the first capture: default vs core_aten decompositions with flash SDPA. It shows the wrong-gradient case. `03` reuses its model definition.
