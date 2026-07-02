# PyNTT Design

## Status

Active implementation design.

## Summary

PyNTT is a Python-first nncase inference backend and megakernel framework. It
uses nncase front-end import, graph optimization, NTT selection, tiling, and
bufferization, then emits a runnable Python model directory instead of a native
kmodel module. The initial backend is Triton, the runtime API uses
`torch.Tensor`, and the generated model runs through the existing nncase
compiler pipeline.

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
  - auto distribution with `b` as the initial block placement axis
  - affine selection
  - auto tiling
  - TIR selection
  - bufferization
- Preserve native NTT vector layout decisions only when PyNTT codegen has an
  explicit Triton template representation for the resulting layout. Layout
  changes must be visible in TIR and generated source; they must not be hidden
  behind native NTT kernels or runtime fallbacks.
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

Current layout:

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
      interpreter.py
      module.py
      tensor.py
      errors.py
      sharding.py
      tuning.py
      workspace.py
    backends/
      __init__.py
      triton/
        __init__.py
        backend.py
    codegen/
      __init__.py
      render.py
      templates/
        triton/
          module.py.jinja
          top_kernel.py.jinja
          kernels/
            TensorLoad.py.jinja
            TensorStore.py.jinja
            ElementwiseBinary.py.jinja
            ...
    testing/
      __init__.py

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
```

The `pyntt/` root contains Python runtime helpers, spec definitions, the Jinja
renderer, and Triton template assets. The C# codegen layer translates nncase NTT
metadata into stable PyNTT specs, generated Python entry files, launch metadata,
and a render manifest. It does not own Python/Triton template source.

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
  kernel_params.json
  specs.py
  generated_kernels.py
  rdata.py
  assets/
    <function>_<section>.bin
  runtime_config.py
  requirements.txt
  README.md
```

Responsibilities:

- `model.py`: public entrypoint, runtime validation, output allocation,
  `generated_kernels.py` refresh on model load, and direct launch of generated
  Triton top kernels.
- `metadata.json`: model metadata, version, static shapes, dtype, kernel list,
  schedule IDs, and debug mapping.
- `kernel_params.json`: codegen manifest consumed by PyNTT's Jinja renderer.
  It contains top-kernel metadata, helper template names, helper model objects,
  and the generated top-kernel call sequence. It is the stable boundary that
  lets PyNTT template edits regenerate kernels without recompiling nncase or
  recompiling the model.
- `specs.py`: Python representation of `ModuleSpec`, `FunctionSpec`, and
  `TensorSpec` instances.
- `generated_kernels.py`: generated Triton helper functions and launchable top
  kernels. The file is initially a placeholder and is refreshed from
  `kernel_params.json` by `model.py` through `pyntt.codegen.render`.
- `rdata.py`: generated bundle table for read-only data sections.
- `assets/*.bin`: binary rdata payloads. Non-empty rdata sections must be
  emitted as binary files, not base64 strings embedded in Python source.
- `runtime_config.py`: backend choice, cache settings, debug flags, and
  optional autotune controls.
- `requirements.txt`: first stage includes `torch`, `triton`, and `jinja2`.

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
- Ranks, layouts, dtypes, and storage plans are fixed by the generated spec.
- Dynamic dimensions are resolved from input tensor shapes before launch and
  must satisfy the generated shape binding constraints.
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

The generated Triton top kernel is instantiated from PyNTT-owned Jinja
templates. C# emits a render manifest instead of final kernel source. Template
models carry TIR-derived dtype, shape, stride, offset, op, and launch metadata;
`pyntt.codegen.render` reads the manifest and produces concrete rank/shape-expr
Triton code in `generated_kernels.py`.

Generated load/compute/store fragments are emitted as named Triton
subfunctions in `generated_kernels.py`; the launchable top kernel remains a
readable call sequence over those helpers. This mirrors the native NTT style:
a generated top kernel calls generated device/helper functions, while the
runtime owns launch, workspace allocation, rdata materialization, and input
validation.

`PyNTTKernelSourceConvertVisitor` selects a Jinja template name and builds a
strongly typed template model, mirroring `KernelCSourceConvertVisitor` in the
native NTT backend. It should not inline large Python snippets directly in C#
strings, and it should not call a C# template engine. The only generated source
body stored in C# is the top-kernel helper call sequence derived by visiting TIR
expressions.

Launch-shape parameters such as `block_size` are not constants owned by the
source visitor. They are emitted as tuning or auto-tiling parameters in kernel
metadata. The generated model selects the current parameter value, computes the
launch grid from that value and resolved logical shape metadata, and passes it
as a Triton constexpr to the top kernel. Later milestones should replace the M3
deterministic selector with real autotune cache entries or NTT auto-tiling
results without changing the generated top-kernel/body split.

Dynamic shape dimensions are not Triton constexpr parameters. Codegen emits them
as regular scalar kernel arguments, because their values are selected by runtime
input tensors and shape bucket bindings. Only structural template choices such
as rank, operator kind, dtype, and tuning meta parameters remain compile-time
constants.

## Block-Axis Distribution

PyNTT uses `b` as the initial block placement axis. For Triton, this maps to
the first launch grid dimension: `tl.program_id(0)` is the block shard index and
`tl.num_programs(0)` is the block shard count.

Generated metadata records this as `launch.sharding` with a `local_shard`
strategy. The generated model still owns launch shape selection and passes
static meta parameters plus runtime dynamic dimension scalars to the top kernel.
Generated Triton helper subfunctions compute the local contiguous shard region:

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
    shape: tuple[int | str, ...]
    strides: tuple[int | str, ...]
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
    shape_bindings: tuple[ShapeBinding, ...]
```

The C# emitter may serialize this as JSON plus Python builders. JSON is useful
for external tools. Python builders are useful for direct import and type-safe
runtime construction.

Spec invariants:

- Tensor ranks are static, while each dimension may be either an integer or a
  generated Python dimension expression.
- Dynamic dimension expressions must be derived from nncase `Dimension`/shape
  expressions and shape bucket bindings, not from ad hoc generated names.
- Layout and strides are explicit.
- Broadcast semantics are explicit, not inferred from shape alone.
- Memory space is explicit.
- Aliasing and in-place behavior are explicit.
- Every generated kernel has a stable debug name and source mapping back to
  nncase function/op information.
- Every generated kernel launch has explicit static launch metadata in the
  generated Python code and `metadata.json`.

## Dynamic Shape Policy

PyNTT dynamic shape support follows nncase shape expressions instead of adding a
separate dispatch system.

The compiler-side source of truth is the existing ranked shape and dimension
expression system:

- `DimConst` is emitted as a literal in Python and Triton.
- `DimVar` is emitted as a generated Python shape binding and as a non-constexpr
  Triton scalar argument.
- Compound expressions such as product, sum, min, max, floor division, ceil
  division, remainder, clamp, and compare/select are emitted as Python
  expressions for runtime allocation/launch and as Triton scalar expressions for
  in-kernel indexing and masking.

The runtime resolves a shape environment before each function call:

```text
inputs
  -> FunctionSpec.shape_bindings
  -> shape_env
  -> validate input/output tensor specs
  -> allocate outputs using resolved shapes
  -> compute numel/grid expressions
  -> launch top kernel with runtime dim scalar arguments
```

Workspace and constant storage are not dynamically resized in this path. After
bufferization, workspace sizes are static upper bounds derived from shape bucket
ranges. Generated kernels use runtime logical shape expressions for masks, loop
extents, and view indexing, while runtime allocation keeps the bufferized
storage plan fixed.

The split between static and dynamic data is:

- Static: rank, dtype, operator specialization, memory space, workspace pool
  size, local rdata layout, structural schedule choices, and Triton meta
  parameters such as `block_size`.
- Dynamic: input-bound dimensions, logical tensor shapes, logical contiguous
  strides when they depend on dynamic dimensions, `numel`, launch grid size, and
  masks.

Dynamic dimensions must not be emitted as `tl.constexpr`. Doing so would turn
runtime tensor shapes into Triton specialization keys and break shape-bucket
execution semantics.

## Triton Backend

The first PyNTT backend is Triton.

Backend responsibilities:

- Generate Triton top-kernel source into `generated_kernels.py`.
- Emit direct launch calls and launch metadata into `model.py`.
- Validate shape, dtype, layout, and schedule constraints before codegen or
  before launch through generated runtime helpers.
- Surface Triton compilation errors with the generated kernel name and original
  nncase op mapping.

PyNTT emits generated Triton `@triton.jit` top kernels through the manifest
renderer. Runtime helpers validate `torch.Tensor` inputs, allocate/reuse
outputs and workspaces, materialize rdata, resolve dynamic shapes, and launch
the generated top kernel. They do not resolve or dispatch op-level kernel
descriptors.

Kernel validation should be strict. The runtime should reject
missing launch metadata, unsupported placement, non-CUDA tensors, non-contiguous
tensors, shape mismatches, or unsupported generated kernel patterns.

## Kernel Templates

PyNTT kernel implementation lives in Jinja templates under
`pyntt/pyntt/codegen/templates/triton`. The nncase C# backend emits a
reader-only manifest, and PyNTT renders that manifest into each generated
model's `generated_kernels.py` as Triton `@triton.jit` helper subfunctions plus
one launchable top kernel. Generated models do not require a kernel spec
dispatch layer or package-level handwritten Triton kernels.

The design intentionally keeps templates in the PyNTT Python package rather
than in `modules/Nncase.Modules.NTT`: changing a template or renderer can
regenerate `generated_kernels.py` from an existing model directory without
recompiling nncase and without rerunning the compiler.

Initial template groups:

- elementwise
  - unary
  - binary
  - cast
  - where
  - simple fused elementwise expressions where schedule allows
- matmul
  - 2-D dense matmul in M4
  - batched matmul in a later coverage pass
  - optional bias/scale/activation epilogue when represented in spec
- reduce
  - sum
  - max
  - min
  - single-axis reduction in M4
  - mean if represented as reduce plus scale
- softmax
  - fixed rank/axis softmax
  - numerically stable max-subtract-exp-sum-div pattern

Template design rules:

- Jinja templates should be reader-only entry points owned by PyNTT. They may
  call Python renderer functions for complex code generation, but the rendered
  source must remain deterministic for a fixed manifest and PyNTT version.
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
runtime utilities may support validation, tuning, launch preparation, and
manifest rendering, but op-specific Triton implementation should be generated
from templates so each kernel specialization is explicit and inspectable.

### Explicit Vector Layout Ops

PyNTT supports explicit vector layout TIR ops and packed matmul layouts when
the corresponding templates are available. Qwen-style paged attention and
packed matmul are the first cases: lowering may emit `TIR.NTT.Pack`/`Unpack`
or vectorized/packed matmul buffers to convert logical head or N dimensions
into vector-lane storage layouts required by Triton kernels.

This boundary is important:

- Pack/vectorize passes may only be enabled when the resulting TIR ops and
  vector element layouts are represented explicitly in PyNTT templates.
- Explicit `Pack`/`Unpack` in TIR are real model semantics and must be lowered.
- Vector element buffers are lowered to scalar physical pointer access in
  Triton templates. The vector lane count is a static template parameter, while
  logical dimensions remain runtime scalar arguments when they are dynamic.
- Tail lanes are masked. Pack writes zero to physical tail lanes; unpack avoids
  writing logical elements outside the source shape.

Generated code should therefore keep the rendered kernel easy to inspect:
logical shape expressions, physical lane expansion, offsets, masks, and stores
are visible in the generated Triton helper instead of hidden in a package-level
runtime dispatch.

### Paged Attention KV Cache Objects

Paged-attention inputs enter PyNTT as nncase runtime objects, but Triton kernels
cannot consume Python objects. The generated model is responsible for
materializing a stable set of tensor arguments before launch:

- `metadata`: flat `int64` tensor containing number of sequences and
  `(context_len, seq_len)` pairs.
- `slot_mapping`: `int64` tensor used by update kernels to map logical token
  positions to physical KV-cache slots.
- `block_tables`: `int64` tensor used by attention kernels to map sequence
  blocks to physical cache blocks.
- `kv_caches`: the backing KV-cache tensor, preferably as a torch tensor view
  over the runtime object's storage so update kernels mutate persistent state.

The C# source converter should not special-case a Python object at the call
site. Instead, when a TIR op references an object buffer, it records synthetic
generated-kernel inputs for the object fields it needs. `model.py` then resolves
those synthetic inputs from the original function argument through runtime
helpers. This keeps the top-kernel signature pure Tensor/scalar while preserving
the existing nncase object input contract at the public function boundary.

Paged-attention templates follow the NTT kernel split:

- `UpdatePagedAttentionKVCache` updates the persistent cache from a local
  K/V slot tensor and a slot-mapping tensor.
- `GatherPagedAttentionKVCache` reads cache blocks into a temporary local
  tensor when TIR asks for a gather stage.
- `PagedAttention` computes attention from Q, cache storage, metadata, scale,
  and output buffers.

For the first implementation, the metadata and storage layout are specialized
from the static `IPagedAttentionKVCache` type information embedded in TIR. The
runtime sequence lengths and dynamic token counts are passed as non-constexpr
Triton scalar arguments.

## Connecting Specs to PyNTT Kernels

The generated model should directly launch generated Triton top kernels.

Recommended flow:

```text
generated model __call__
  -> validate runtime tensors against FunctionSpec/TensorSpec
  -> allocate outputs/temp buffers
  -> resolve dynamic shape environment
  -> launch generated Triton top kernel with runtime dim scalars and static meta
```

Generated kernel metadata and `kernel_params.json` should contain:

- generated top kernel name
- static launch metadata
- runtime dimension scalar names
- Python expressions for dynamic launch values such as `numel`
- optional autotune result
- debug mapping

This keeps PyNTT extensible:

- Generated Triton top kernels can specialize schedule shape, indexing,
  placement, and composition.
- PyNTT package code can provide shared render utilities and runtime helpers.
  Device code for an op specialization should still be rendered into the
  generated model so the final Triton source is inspectable.
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

Current policy:

- Rdata sections are emitted as binary assets under `assets/*.bin`.
- `rdata.py` records the bundle table and `file:` URI for each non-empty
  payload.
- Empty rdata sections are represented as empty strings in the bundle table.
- Base64-embedded rdata is not allowed; C# rejects non-empty inline rdata
  payloads during generated-model writing.
- The generated `metadata.json` and `rdata.py` should include enough byte,
  section, and file-path information to debug runtime materialization.

The runtime should load constants to the target device lazily or during model
initialization, depending on the selected runtime configuration.

## Shape Validation Policy

The generated model should validate:

- number of inputs
- tensor rank
- tensor shape or shape expression
- dtype
- device
- contiguous/layout constraints
- stride constraints when required

Shape mismatch should be an error. It should not trigger dynamic recompilation.
If a runtime input violates the shape bucket range used by compilation, the
runtime should reject it before launching Triton.

Future specialization may add multiple generated variants selected by a shape
bucket key, but the single-variant dynamic path should remain valid for
dimension values inside the compiled range.

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
- where: boolean mask with fixed-rank broadcast rules
- matmul: 2-D shapes first; batched shapes later
- reduce: sum/max/min over one fixed axis first; multi-axis later
- softmax: fixed axis softmax

Correctness tolerances should be dtype-specific and documented in tests.

## Performance Policy

Phase 1 prioritizes functionality and correctness. Performance ownership sits
in generated Triton top kernels and PyNTT templates. Future work should
preserve enough schedule, layout, cost, and launch metadata to improve those
generated kernels without adding a runtime op dispatch layer.

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
  - dynamic shape bucket key or range summary
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

### Phase 1: Triton Runtime

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

### Phase 2.1: Manifest and Reader-Only Templates

- Emit `kernel_params.json` as the stable C# to PyNTT renderer boundary.
- Move PyNTT kernel templates out of the C# codegen tree into
  `pyntt/pyntt/codegen/templates/triton`.
- Render `generated_kernels.py` from `kernel_params.json` at model load via
  `pyntt.codegen.render`.
- Keep generated top kernels as explicit call sequences over generated Triton
  helper functions.
- Allow template-only changes to refresh kernels without recompiling nncase or
  rerunning model compilation.

### Phase 2.5: Dynamic Shape Bring-Up

- Emit PyNTT shape expressions from nncase `Dimension` values.
- Serialize `FunctionSpec.shape_bindings` from shape bucket metadata.
- Resolve dynamic dimensions from `torch.Tensor.shape` in the generated runtime.
- Pass dynamic dimensions as non-constexpr Triton scalar arguments.
- Keep bufferized workspace allocation at static upper-bound sizes.
- Run dynamic-shape importer coverage, starting with Qwen3 decode/prefill
  coverage.

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
- nncase can compile small graphs to a PyNTT Python directory.
- The generated directory can be imported and run with `torch.Tensor` inputs.
- The generated model directly launches generated Triton top kernels from
  `generated_kernels.py`.
- Dynamic input dimensions are resolved through existing nncase shape
  expressions and are passed to Triton top kernels as non-constexpr scalar
  arguments.
- Unsupported ops fail during compilation with clear diagnostics.
- pytest covers the initial op group at unit and generated-model levels.
- Generated metadata is deterministic and includes source mapping.
