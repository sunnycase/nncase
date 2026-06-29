# PyNTT Design

## Status

Draft design.

## Summary

PyNTT is a Python-first nncase inference backend and megakernel framework. It
uses nncase front-end import, graph optimization, NTT selection, tiling, and
bufferization, then emits a runnable Python model directory instead of a native
kmodel module. The initial backend is Triton, the runtime API uses
`torch.Tensor`, and the first implementation supports static shapes only.

PyNTT is not a rewrite of NTT. It reuses the existing NTT tensor IR and schedule
contracts as the compiler-side source of truth, while replacing the kernel
implementation language and runtime with Python DSL infrastructure.

Product name: `PyNTT`.

Target and module kind: `pyntt`.

## Goals

- Provide a formal nncase inference backend for Python DSL kernels.
- Build an inference megakernel framework where codegen emits Triton top
  kernels directly and the generated Python model launches them.
- Generate a Python model directory that can run with a pure Python runtime.
- Use `torch.Tensor` as the first public runtime tensor interface.
- Reuse existing nncase and NTT compiler passes where their semantics match
  PyNTT:
  - module partitioning
  - auto packing
  - auto vectorization
  - auto distribution with `b` as the initial block placement axis
  - affine selection
  - auto tiling
  - TIR selection
  - bufferization
- Support the first op group:
  - elementwise
  - unary
  - binary
  - cast
  - where
  - matmul
  - reduce
  - softmax
- Integrate into the existing pytest-based test flow.

## Non-Goals

- Phase 1 does not need to emit a native kmodel payload.
- Phase 1 does not need to be executable by the C++ nncase runtime.
- Phase 1 does not support dynamic shapes.
- Phase 1 does not support multiple Python DSL backends. Triton is the only
  backend.
- PyNTT must not silently fall back to C++ NTT kernels. Unsupported operators,
  layouts, dtypes, or schedules should fail fast with actionable diagnostics.

## Existing NTT Context

The current NTT backend has two major parts:

- `ntt/`: C++ runtime and kernel headers.
- `modules/Nncase.Modules.NTT/`: C# target, NTT TIR selection, evaluators, and
  C++/CUDA code generation.

`cpu` and `cuda` targets currently inherit the same NTT target path and use a
native `NTTModuleBuilder`. That builder serializes module metadata, read-only
data, local data sections, generated C++/CUDA source, and a native `.text`
payload. The runtime then loads a native `block_entry`.

PyNTT should reuse the compiler-side NTT semantics but not the native runtime
ABI in the first stage. Its output should be Python artifacts that are loaded
and executed by the PyNTT runtime.

## High-Level Architecture

```text
Model importer
  -> nncase graph IR
  -> graph optimization / fusion / partition
  -> NTT selection and scheduling
  -> NTT TIR PrimFunction + SchedResult + buffers
  -> PyNTT module spec + generated Triton top kernels
  -> Python model directory
  -> generated Python model runtime
  -> generated Triton top kernel
  -> CUDA execution through torch.Tensor inputs and outputs
```

The important boundary is between NTT TIR and PyNTT module spec. NTT TIR is the
compiler-facing representation. PyNTT module spec is the Python-runtime-facing
contract.

## Repository Layout

Recommended first layout:

```text
pyntt/
  README.md
  pyproject.toml
  pyntt/
    __init__.py
    ir/
      __init__.py
      spec.py
      dtype.py
      layout.py
    runtime/
      __init__.py
      module.py
      tensor.py
      executor.py
      errors.py
    backends/
      __init__.py
      triton/
        __init__.py
        backend.py
        lowering.py
        autotune.py
    testing/
      __init__.py
      reference.py
      compare.py
      benchmark.py

modules/Nncase.Modules.NTT/
  Targets/
    PyNTTTarget.cs
    PyNTTModuleCompiler.cs
  CodeGen/
    PyNTT/
      PyNTTModuleBuilder.cs
      PyNTTFunctionBuilder.cs
      PyNTTKernelSourceConvertVisitor.cs
      PyNTTLinkableFunction.cs
      PyNTTLinkableModule.cs
      PyNTTLinkedModule.cs
      Templates/
        Triton/
          Kernels/
            TensorLoad.py.cshtml
            TensorStore.py.cshtml
            ElementwiseBinary.py.cshtml
            ...
```

The `pyntt/` root contains Python runtime helpers, spec definitions, and
optional Triton kernel utilities. The C# codegen layer translates nncase NTT
metadata into stable PyNTT specs, generated Python entry files, and generated
Triton top kernels.

## Target Integration

Add a `PyNTTTarget` with target kind `pyntt`.

`PyNTTTarget` should inherit the existing `NTTTarget` base so it can reuse the
same NTT pass registration path. Its compiler implementation should be a new
`PyNTTModuleCompiler`.

Initial behavior:

- `PyNTTModuleCompiler.ModuleKind` returns `pyntt`.
- `PyNTTModuleCompiler.CreateModuleBuilder` returns `PyNTTModuleBuilder`.
- `PyNTTModuleCompiler.IsSupportedCall` uses an explicit PyNTT allowlist rather
  than inheriting every current CPU/CUDA NTT-supported op.
- The allowlist covers only the phase 1 op set.
- Unsupported calls should include:
  - op name
  - dtype
  - shape
  - layout/vectorization state when available
  - reason for rejection

PyNTT should reuse NTT TIR selection when possible. If an existing NTT TIR op
has semantics that are too C++-NTT-specific, the fix should be made at the
selection/spec boundary, not hidden inside a Triton kernel.

## Generated Output

The compiler should emit a directory, not a single file. A generated directory
is easier to inspect, test, benchmark, and cache.

Suggested output:

```text
<output-dir>/
  model.py
  metadata.json
  specs.py
  generated_kernels.py
  runtime_config.py
  requirements.txt
  README.md
```

Responsibilities:

- `model.py`: public entrypoint, runtime validation, output allocation, and
  direct launch of generated Triton top kernels.
- `metadata.json`: model metadata, version, static shapes, dtype, kernel list,
  schedule IDs, and debug mapping.
- `specs.py`: Python representation of `ModuleSpec`, `FunctionSpec`, and
  `TensorSpec` instances.
- `generated_kernels.py`: generated Triton top kernels.
- `runtime_config.py`: backend choice, cache settings, debug flags, and
  optional autotune controls.
- `requirements.txt`: first stage should include `torch` and `triton`.

The generated code should be deterministic for the same input model, compiler
options, and PyNTT version. Deterministic output is required for review,
debugging, and cache keys.

## Runtime API

The first public API should be `torch.Tensor` based:

```python
import torch
from generated_model import load_model

model = load_model()
outputs = model(*inputs)
```

Runtime constraints:

- Inputs and outputs are `torch.Tensor`.
- CUDA tensors are the primary execution path.
- CPU tensors may be rejected initially unless a copy-to-CUDA policy is
  explicitly configured.
- Shapes are static in phase 1 and must match the generated spec exactly.
- Dtypes must match the generated spec exactly unless an explicit cast kernel is
  part of the graph.
- Output allocation is owned by the PyNTT runtime unless the generated model
  explicitly supports user-provided output buffers.

Core runtime objects:

- `ModuleSpec`: whole generated model metadata.
- `FunctionSpec`: callable graph or megakernel-level function.
- `TensorSpec`: dtype, shape, strides, memory role, layout, and aliasing.

Kernel launch metadata is emitted directly into `model.py` and
`metadata.json`. There is no runtime kernel dispatch layer in M3.

## Codegen Structure

PyNTT follows the same ownership model as the existing NTT CPU codegen path:

- `PyNTTModuleBuilder` owns module-level assembly.
- `PyNTTFunctionBuilder` owns per-function conversion setup.
- `PyNTTKernelSourceConvertVisitor` owns IR-to-Triton source conversion and
  generated-kernel metadata, matching the role of
  `KernelCSourceConvertVisitor` in the native NTT backend.
- `PyNTTLinkableFunction` stores the source conversion result.
- `PyNTTLinkableModule` only writes generated files, metadata, and model launch
  glue. It must not contain op-level Triton lowering.

The generated Triton top kernel is instantiated from Razor templates. Template
models carry TIR-derived dtype, shape, stride, offset, op, and launch metadata,
and the rendered `generated_kernels.py` contains the concrete rank/static-shape
Triton code. This keeps kernel implementation logic out of the visitor without
forcing the Python package to hand-write rank-dispatched device helpers.
Generated load/compute/store fragments should be emitted as named Triton
subfunctions in `generated_kernels.py`; the launchable top kernel should remain
a readable call sequence over those helpers.

`PyNTTKernelSourceConvertVisitor` should select a template and build a strongly
typed template model, mirroring `KernelCSourceConvertVisitor` in the native NTT
backend. It should not inline large Python snippets directly in C# strings.

Launch-shape parameters such as `block_size` are not constants owned by the
source visitor. They are emitted as tuning or auto-tiling parameters in kernel
metadata. The generated model selects the current parameter value, computes the
launch grid from that value and static shape metadata, and passes it as a
Triton constexpr to the top kernel. Later milestones should replace the M3
deterministic selector with real autotune cache entries or NTT auto-tiling
results without changing the generated top-kernel/body split.

## Block-Axis Distribution

PyNTT uses `b` as the initial block placement axis. For Triton, this maps to
the first launch grid dimension: `tl.program_id(0)` is the block shard index and
`tl.num_programs(0)` is the block shard count.

Generated metadata records this as `launch.sharding` with a `local_shard`
strategy. The generated model still owns launch shape selection and passes only
static meta parameters to the top kernel. Generated Triton helper subfunctions
compute the local contiguous shard region:

```text
local_extent = ceil_div(global_extent, block_count)
offset = block_index * local_extent
extent = min(local_extent, max(global_extent - offset, 0))
```

Kernels must loop inside their local shard by the tuned or auto-tiled
`block_size`. They must not use grid-stride traversal for this phase, because
grid-stride interleaves regions across blocks and does not match NTT local shard
semantics.

## Megakernel Model

PyNTT should treat a scheduled `PrimFunction` as the first megakernel unit.
This keeps the initial boundary aligned with existing nncase scheduling and
bufferization.

The term megakernel means:

- A compiled function can contain multiple logical operators.
- Producer-consumer fusion should reduce global memory traffic.
- Schedule and buffer metadata should remain explicit.
- The generated Triton top kernel is the launch boundary for M3.

Phase 1 should not force the whole model into one Triton kernel. Whole-model
megakernel execution is a later optimization and should be introduced only when
the scheduling, register pressure, and memory lifetime model can justify it.

Recommended phases:

- Phase 1: one PyNTT kernel call per NTT `PrimFunction` or selected fused op.
- Phase 2: merge compatible adjacent `PrimFunction`s where buffer lifetimes and
  schedules allow it.
- Phase 3: persistent or grouped execution for LLM-style decode workloads.
- Phase 4: CUDA Graph capture for launch overhead reduction.

## PyNTT Spec

The spec is the compatibility layer between C# codegen and Python runtime
helpers. It should be explicit enough to describe module/function/tensor
contracts. Kernel bodies and launch calls are emitted as generated Python code,
not runtime-dispatched spec objects.

Example conceptual schema:

```python
@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: DType
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    role: TensorRole
    layout: LayoutSpec
    memory: MemorySpace

@dataclass(frozen=True)
class FunctionSpec:
    name: str
    module_kind: str
    is_entry: bool
    parameters: tuple[str, ...]
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
```

The C# emitter may serialize this as JSON plus Python builders. JSON is useful
for external tools. Python builders are useful for direct import and type-safe
runtime construction.

Spec invariants:

- All shapes are static in phase 1.
- Layout and strides are explicit.
- Broadcast semantics are explicit, not inferred from shape alone.
- Memory space is explicit.
- Aliasing and in-place behavior are explicit.
- Every generated kernel has a stable debug name and source mapping back to
  nncase function/op information.
- Every generated kernel launch has explicit static launch metadata in the
  generated Python code and `metadata.json`.

## Triton Backend

The first PyNTT backend is Triton.

Backend responsibilities:

- Generate Triton top-kernel source into `generated_kernels.py`.
- Emit direct launch calls and static launch metadata into `model.py`.
- Validate shape, dtype, layout, and schedule constraints before codegen or
  before launch through generated runtime helpers.
- Surface Triton compilation errors with the generated kernel name and original
  nncase op mapping.

M3 emits generated Triton `@triton.jit` top kernels directly. Runtime helpers
validate `torch.Tensor` inputs and allocate outputs; they do not resolve or
dispatch kernel descriptors.

Kernel validation should be strict in phase 1. The runtime should reject
missing launch metadata, unsupported placement, non-CUDA tensors, non-contiguous
tensors, shape mismatches, or unsupported generated kernel patterns.

## Kernel Templates

PyNTT kernel implementation lives in Razor templates under
`modules/Nncase.Modules.NTT/CodeGen/PyNTT/Templates/Triton`. Codegen
instantiates those templates with concrete TIR dtype, rank, shape, stride,
offset, op, and launch metadata. The rendered output is copied into each
generated model's `generated_kernels.py` as Triton `@triton.jit` helper
subfunctions plus one launchable top kernel. Generated models do not require a
kernel spec dispatch layer or package-level handwritten Triton kernels.

Initial template groups:

- elementwise
  - unary
  - binary
  - cast
  - where
  - simple fused elementwise expressions where schedule allows
- matmul
  - static 2-D dense matmul in M4
  - batched static matmul in a later coverage pass
  - optional bias/scale/activation epilogue when represented in spec
- reduce
  - sum
  - max
  - min
  - single-axis reduction in M4
  - mean if represented as reduce plus scale
- softmax
  - static axis softmax
  - numerically stable max-subtract-exp-sum-div pattern

Template design rules:

- Razor templates should be mostly Python/Triton, with C# logic kept in the
  template model.
- Rendered kernels should include comments naming the source template and TIR
  specialization metadata.
- Generated kernels should compute the PyNTT local shard for the `b` axis and
  then loop inside the local shard by `block_size`.
- Dtype behavior and accumulation dtype must be explicit.
- Layout assumptions must be encoded in kernel constraints and runtime
  validation.
- Meta parameters should be emitted as launch metadata or selected by a controlled
  autotune policy.
- Debug names should include model function, kernel kind, and specialization
  key where possible.

Generated Triton top kernels are the primary execution path. Reusable Python
runtime utilities may support validation, tuning, and launch preparation, but
op-specific Triton implementation should be generated from templates so each
kernel specialization is explicit and inspectable.

## Connecting Specs to PyNTT Kernels

The generated model should directly launch generated Triton top kernels.

Recommended flow:

```text
generated model __call__
  -> validate runtime tensors against FunctionSpec/TensorSpec
  -> allocate outputs/temp buffers
  -> launch generated Triton top kernel with static meta
```

Generated kernel metadata should contain:

- generated top kernel name
- static launch metadata
- optional autotune result
- debug mapping

This keeps PyNTT extensible:

- Generated Triton top kernels can specialize schedule shape, indexing,
  placement, and composition.
- Hand-written PyNTT Triton kernels can still be used as reusable helpers or
  fallback implementation patterns when that is intentional.
- TileLang or CuteDSL can be added later behind the same spec contract.

## Buffer and Memory Model

Phase 1 should use a simple Python memory model:

- Inputs are user-provided `torch.Tensor` objects.
- Outputs are allocated by the runtime with `torch.empty_like`,
  `torch.empty`, or spec-derived allocation.
- Temporary buffers are allocated by the runtime and reused within one model
  invocation when lifetime data is available.
- Constants are emitted as PyTorch tensors or serialized files loaded by
  `model.py`.

NTT bufferization metadata should be preserved in the spec even if phase 1 does
not fully exploit it. That metadata is required for future megakernel fusion and
workspace reuse.

Memory spaces:

- `input`
- `output`
- `constant`
- `temp`
- `local`

Triton-specific memory hierarchy such as SRAM/shared memory should be encoded
as schedule or kernel metadata, not as Python tensor allocations.

## Constants

Constants should be handled in a way that supports review and deployment.

Recommended phase 1 options:

- Small constants may be embedded directly in generated Python.
- Larger constants should be stored as `.pt`, `.npy`, or `.safetensors` files
  under the generated model directory.
- The generated `metadata.json` should include constant names, shapes, dtypes,
  file paths, and checksums.

The runtime should load constants to the target device lazily or during model
initialization, depending on the selected runtime configuration.

## Static Shape Policy

Phase 1 supports static shapes only.

The generated model should validate:

- number of inputs
- tensor rank
- tensor shape
- dtype
- device
- contiguous/layout constraints
- stride constraints when required

Shape mismatch should be an error. It should not trigger dynamic recompilation
in phase 1.

Future dynamic shape support should use specialization:

- shape bucket key
- generated or cached spec variant
- Triton compilation cache key
- runtime dispatch by concrete shape

## Error Handling

PyNTT should fail early and with enough context.

Compiler errors should include:

- target `pyntt`
- op name and module function
- unsupported dtype/shape/layout reason
- relevant source model node name if available

Runtime errors should include:

- generated model name
- generated kernel name
- selected backend
- tensor argument index/name
- expected and actual dtype/shape/device

Triton errors should be wrapped with:

- PyNTT kernel key
- generated source path if any
- original nncase function/op mapping

## Observability

PyNTT should be easy to inspect because that is one of the main advantages of a
Python DSL backend.

Required generated artifacts:

- `metadata.json`
- printed or dumpable `ModuleSpec`
- generated kernel launch log under debug mode
- source map from generated kernel/function to nncase function/op
- optional per-kernel timing report

Runtime debug options:

- print generated kernel launch details
- dump selected Triton meta parameters
- enable Triton IR/PTX dump through environment variables
- run selected kernels against reference implementation
- benchmark selected kernels

## Testing

PyNTT should integrate into the existing pytest flow.

Recommended test layers:

- Python unit tests for PyNTT specs, generated launch, runtime validation, and
  allocation.
- Triton kernel tests for generated top kernels.
- nncase compiler tests that compile small graphs with target `pyntt` and
  execute the generated Python directory.
- Existing importer/basic tests extended with a `pyntt` target subset where
  CUDA and Triton are available.
- Differential correctness tests against existing nncase reference output,
  PyTorch, or NumPy.

Initial test matrix:

- elementwise unary: `abs`, `exp`, `log`, `sqrt`, `neg`
- binary: `add`, `sub`, `mul`, `div`, `max`, `min`
- cast: `float32 <-> float16`, integer casts where supported
- where: boolean mask with static broadcast rules
- matmul: 2-D static shapes first; batched static shapes later
- reduce: sum/max/min over one static axis first; multi-axis later
- softmax: static axis softmax

Correctness tolerances should be dtype-specific and documented in tests.

## Performance Policy

Phase 1 prioritizes functionality and correctness. Performance ownership sits
in generated Triton top kernels for M3. Future work should preserve enough
schedule and launch metadata to improve those generated kernels without adding
a runtime kernel dispatch layer.

Required performance guardrails:

- record per-kernel execution time in benchmark mode
- separate Triton compile time from steady-state runtime
- expose kernel cache behavior
- keep debug mapping to identify slow kernels

Phase 1 should not block on matching native NTT performance. However, the design
should preserve the information needed to tune later:

- layout
- strides
- schedule
- tile sizes
- reduction axes
- accumulation dtype
- launch meta parameters

## Autotuning

Autotuning should be optional in phase 1.

Recommended policy:

- Default mode uses deterministic selection from explicit tuning parameter
  candidate spaces.
- Optional mode enables Triton autotune for selected generated kernels.
- Autotune cache key includes:
  - kernel key
  - dtype
  - shape
  - layout/stride constraints
  - GPU architecture
  - Triton version
  - PyNTT version

Autotune must not change numerical semantics. If a tuned config fails
correctness validation, it must be rejected and reported.

## Compatibility and Versioning

PyNTT should version the generated spec.

`metadata.json` should include:

- `pyntt_spec_version`
- `pyntt_runtime_version`
- `nncase_version`
- `target_kind`
- `backend`
- `backend_version`
- `triton_version`
- `torch_version`

The runtime should reject unsupported spec versions with a clear error.

## Implementation Plan

### Phase 0: Skeleton

- Add `docs/design/pyntt.md`.
- Add `pyntt/` Python package skeleton.
- Add `PyNTTTarget` and `PyNTTModuleCompiler`.
- Add a minimal `PyNTTModuleBuilder` that emits a model directory.
- Add a single generated model smoke test.

### Phase 1: Static Triton Runtime

- Define `ModuleSpec`, `FunctionSpec`, and `TensorSpec`.
- Implement pure Python runtime with `torch.Tensor` validation.
- Emit generated Triton top kernels and direct launch code.
- Implement initial elementwise top-kernel generation.
- Add pytest coverage for compile and run.

### Phase 2: First Kernel Set

- Add initial matmul templates.
- Add initial reduce templates.
- Add initial softmax templates.
- Add dtype-specific correctness tolerances.
- Add generated metadata and debug source maps.
- Add benchmark helpers.

### Phase 3: Megakernel Improvements

- Use buffer lifetime metadata for workspace reuse.
- Add fusion-aware elementwise and epilogue handling.
- Add fused generated-kernel patterns where schedules justify them.
- Add optional Triton autotune.
- Generalize generated Triton kernels as PyNTT TIR lowering coverage grows.

### Phase 4: Production Hardening

- Add packaging integration.
- Add CI gates for PyNTT tests where CUDA/Triton are available.
- Add compatibility checks for spec/runtime versions.
- Add profiling and debug documentation.
- Evaluate CUDA Graph capture or grouped execution for launch overhead.

## Open Design Points

These are not blockers for the first design but should be decided before broad
implementation.

- Whether generated constants use `.pt`, `.npy`, or `.safetensors`.
- Whether CPU tensors should be rejected or copied to CUDA by policy.
- Whether `pyntt/` is packaged as a separate wheel or included in the main
  `nncase` wheel.
- How much of NTT distributed metadata should be serialized before PyNTT has a
  multi-device runtime.
- Whether phase 1 generated output should include a standalone benchmark script
  by default.

## Acceptance Criteria

Phase 1 is complete when:

- `pyntt` is a registered nncase target.
- nncase can compile small static-shape graphs to a PyNTT Python directory.
- The generated directory can be imported and run with `torch.Tensor` inputs.
- The generated model directly launches generated Triton top kernels from
  `generated_kernels.py`.
- Unsupported ops fail during compilation with clear diagnostics.
- pytest covers the initial op group at unit and generated-model levels.
- Generated metadata is deterministic and includes source mapping.
