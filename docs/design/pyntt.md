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
- Represent storage-preserving reshape and bitcast operations as first-class
  buffer-alias Grids. They must not allocate storage or generate Triton/C++
  kernels.
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
- `specs.py`: Python representation of `ModuleSpec`, `FunctionSpec`,
  `TensorSpec`, and `TensorResultSpec` instances. `FunctionSpec.outputs`
  describes caller-allocated storage, while `FunctionSpec.results` describes
  ordered logical results backed by input or output storage.
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

- Tensor inputs, caller-allocated outputs, and tensor results are
  `torch.Tensor`. Object ABI values remain objects.
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
- `TensorSpec`: dtype, shape, strides, memory role, and layout.
- `TensorResultSpec`: logical tensor metadata plus its input/output storage
  source and byte offset.

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
class TensorResultSpec:
    tensor: TensorSpec
    source: Literal["input", "output"]
    source_index: int
    offset_bytes: int | str

@dataclass(frozen=True)
class FunctionSpec:
    name: str
    module_kind: str
    is_entry: bool
    parameters: tuple[str, ...]
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    results: tuple[TensorResultSpec, ...]
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
- Caller-allocated storage outputs and ordered logical results are distinct;
  every result names exactly one input/output storage owner.
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

## Buffer Alias Views and Function ABI

Storage-preserving `Reshape` and `Bitcast` are view operations, not compute
operators. Their lowering is target-independent until the final TIR buffer
alias is created:

```text
Reshape / Bitcast
  -> view-like Grid(read source, write result, alias-only body)
  -> solution-local, per-use Grid fusion during MCTS
  -> logical TIR.Buffer descriptor over the selected physical storage
  -> bufferize/codegen with no view kernel and no copy
```

The Grid access maps are the canonical scheduling representation. The body op
implements `IBufferAliasOp`, which identifies the source and result operands;
memory-effect analysis therefore distinguishes a zero-copy alias from an op
that merely happens to have no declared effects. `BufferViewTransform` is an
internal materialization helper derived from those Grid maps. It contains a
source map and a result map over one common affine domain. Construction and
verification enforce:

- ranked tensor or distributed tensor source/result types;
- equal physical byte size;
- no partial distributed values;
- compatible placement;
- for distributed-to-distributed views, every physical shard maps to the same
  symbolic common-domain region on both sides, including non-uniform tail
  shards and vector lane width changes;
- explicit singleton-axis projection for dimension insertion/removal;
- byte-aligned dtype reinterpretation and a contiguous reshaped suffix.

MCTS treats a view as a normal zero-cost Grid candidate. Fusion is per use: if a
view is shared or is a function result, the selected solution clones only the
schedule node for the fused consumer and preserves the canonical view for other
uses. It never consumes the semantic expression or mutates another candidate.
After search, an alias-only condensed component remains a residual Grid instead
of being wrapped in a descriptor-only `PrimFunction`. AutoTiling substitutes
its selected producer values but does not restore a concrete `Reshape` or
`Bitcast` op. TIR selection accepts only these residual alias Grids, follows the
generic `IBufferAliasOp` summary, and emits `TIR.Buffer` values sharing the
selected `PhysicalBuffer` and `MemSpan`. Any executable Grid that survives
AutoTiling is an invariant violation. Likewise, a nested descriptor-only
`PrimFunction` reaching backend codegen is rejected rather than silently
elided. No `TIR.NTT.Reshape`, `TIR.NTT.Bitcast`, memcopy, empty device function,
or backend template is emitted for the alias.

`PrimFunction` has two distinct result concepts:

- output parameters are physical storage allocated by the caller;
- `PrimFunction.Results` is the ordered list of logical result buffers/views
  and the input/output ABI storage that owns their bytes.

A direct TIR `PrimFunction` call remains an imperative `Unit` expression. Its
logical results are exposed only through `PrimFunctionWrapper` before wrapper
removal and through the explicit result bindings after lowering. This keeps
TIR control-flow bodies statement-typed while preserving tuple and alias
results without copies.

TIR selection promotes full-span internal result storage to caller-allocated
output storage. A partial internal view is rejected unless an owning pass has
inserted an explicit materialization. Input-backed and in-place results require
no output allocation. At a callee call site, the caller reconstructs each
logical result view from the callee descriptor and the bound storage argument.

PyNTT serializes this ABI directly:

- `FunctionSpec.inputs`: input storage parameters;
- `FunctionSpec.outputs`: caller-allocated output storage parameters;
- `FunctionSpec.results`: ordered logical results with `source`,
  `source_index`, dtype, shape, strides, and `offset_bytes`.

The runtime allocates only `outputs`. After execution it materializes `results`
as zero-copy `torch.Tensor` views over their input/output storage using the
generated dtype, shape, strides, and byte offset. This distinction is required
for root reshape/bitcast results, in-place object results such as KV cache, and
tuple-returning internal functions.

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

## Target Machine and AutoTiling Model

NTT-family targets resolve one immutable `TargetMachineModel` from the target
option `--target-machine`. A named model is the single source of truth shared by
the op cost model, AutoTiling, TIR buffer placement, codegen metadata, and
runtime resource validation. Ad-hoc memory capacity/bandwidth arrays and the
old Triton capability string are not part of the contract.

The model stops at the physical block boundary:

- A CPU block is one core-bound worker.
- A PyNTT block is one persistent GPU CTA.
- PyNTT uses eight compute warps with the target warp width. The block count is
  determined by the configured block hierarchy and must not exceed the target
  compute-unit count.
- The target worker width is also the default PagedAttention query tile. On
  NVIDIA targets this is 32, which bounds the private score/value state without
  exposing warp decomposition to nncase. The cache block size remains an
  independent storage-layout property.
- Occupancy is not an AutoTiling variable. The shared-memory budget is fixed
  per block.
- Warp/thread decomposition and tensor layouts remain the responsibility of
  Triton.

`TargetMemoryResourceSpec` models a physical storage resource and its capacity,
bandwidth, latency, and allocation granularity. `TargetMemorySpaceSpec` models a
logical scheduling space with a stable identity, sharing scope, compiler-managed
allocation limit, and optional `MemoryLocation` binding. Multiple logical
spaces may share one physical resource. The logical allocation limit may be
smaller than the physical capacity: persistent Triton profiles reserve the
remainder of shared memory for backend-private dot-operand staging, lowering
scratch, and the PyNTT device-call ABI. AutoTiling can allocate only the logical
arena limit, while generated-kernel validation checks total compiled usage
against the physical resource capacity. Directed `TargetMemoryTransferSpec`
edges describe adjacent parent/local channels and explicitly distinguish
`DirectAccess` from `ExplicitCopy`. Chip-scoped traffic is aggregated across
active blocks; block-scoped traffic remains per block.

The AutoTiling hierarchy is ordered from the innermost working set to the root:

```text
GPU: Shared -> BlockLocalData -> chip-global root
CPU: L1/Cache -> BlockLocalData -> main-memory root
```

`BlockLocalData` remains a real outer AutoTiling candidate as well as the
block-scoped workspace vocabulary shared with AutoDistributed and TIR
selection. On UMA GPUs it normally shares the global-memory resource with the
root; that adjacent edge is `DirectAccess`, so lowering creates a block-local
logical view without a copy. The `BlockLocalData -> Shared` edge is
`ExplicitCopy`. CPU `BlockLocalData -> L1/Cache` follows the same explicit
staging contract.

All storage resources use the existing TIR buffer vocabulary. A selected
`Shared` or `Cache` placement allocates a target-bound local buffer. Lowering
inserts `TileLoad(local, parent_view)` before reads and
`TileStore(local, parent_view)` after writes according to the operand memory
effects. Existing NTT `TensorLoad`/`TensorStore` operations retain their
distributed model-boundary semantics and may directly materialize their local
buffer in the selected lowest storage space. Direct-access edges never emit a
copy.

Registers are intentionally absent from the nncase target-memory hierarchy and
from TIR buffer placement. Triton owns block-internal SSA values, register
allocation, warp decomposition, and instruction-level scheduling. PyNTT still
rejects a compiled specialization that spills. nncase does not model a
register memory space, register placement, or occupancy. A target profile does,
however, declare a backend-private accumulator byte budget and the minimum
GEMM/GEMV accumulator tile dimensions imposed by its lowering. These values
constrain semantic reduction state before codegen; they are not addressable
storage and do not allocate a TIR buffer.

Reduction axes remain first-class affine grid axes so AutoTiling can choose
their power-of-two tile extents. A `ReductionAccumulator` operand effect marks
the logical result carried across the innermost reduction loop; it is not a
request for an addressable TIR buffer. AutoTiling therefore never places this
state in `Shared`, `Cache`, or `BlockLocalData`. The backend initializes private
state before the reduction loop, updates it for each reduction tile, and
finalizes/casts/stores it once after the loop. PyNTT represents this state as a
Triton SSA tensor, while NTT uses a backend-local tensor object. Only partials
that cross block or PrimFunc boundaries remain explicit distributed TIR
buffers with their required synchronization. Auxiliary reduction state follows
the same ownership rule: tiled `Mean` carries a backend-private element count,
adds each dynamic tile's true reduction extent, and divides the accumulated sum
only once during finalization. `MatrixTileWorkload` derives the live fp32
accumulator bytes from local M/N/multiplicity after applying the backend's
minimum GEMM/GEMV accumulator dimensions. `ReductionTileWorkload` reports the
corresponding state for scalar reductions. OR-Tools rejects tiles whose live
state exceeds the target budget; codegen repeats the same check as a fail-fast
contract assertion.

AutoTiling minimizes block latency using:

```text
max(compute cycles,
    per-memory read/write cycles,
    directed transfer cycles)
+ explicit synchronization cycles
```

Elementwise and SIMT throughput are per block. Matrix candidates use the local
shard/tile shape, model MMA/WGMMA primitive dimensions, and charge padding when
M/N/K do not fit the primitive. Kernel tile extents use power-of-two candidate
sets. Hierarchy decomposition factors remain exact affine factors and are not
kernel tile sizes.

AutoTiling metadata follows three separate ownership rules:

- `ParameterInfo.MemoryEffect` is the only operand read/write/scope contract.
  Grid memory-effect analysis resolves body calls and aliases back to each
  `GridAccess`; operators do not publish a second buffer-state table.
- `GridTileAxisPolicy` records tiling legality on the affine domain. Searchable
  axes use power-of-two candidates, while full and fixed axes are exact and may
  have non-power-of-two extents.
- `TileWorkload` describes target-independent local compute work and, for
  reductions, the semantic state that must remain live between reduction
  tiles. It does not choose or expose that state's physical storage. The target
  machine converts compute work into cycles and applies backend-private state
  limits and lowering granularity.

There is deliberately no `MicroKernelInfo` contract. Block-internal
microkernel implementation remains a backend responsibility.

PyNTT currently provides canonical profiles for RTX 5060 Ti 16 GB and H800 SXM
80 GB, plus generic CPU/CUDA/XPU profiles. The generated manifest records the
resolved profile, managed/physical memory limits, and backend-private
accumulator contract. At first launch, PyNTT tries tuning
candidates in priority order and accepts only a specialization whose compiled
warp count, shared memory, and ptxas spill-store/spill-load counts satisfy that
contract. Successful specialization decisions are cached; Triton
`OutOfResources` rejects only that candidate, while other compilation errors
fail immediately.

## Buffer and Memory Model

- Inputs are user-provided `torch.Tensor` objects.
- Main and internal PrimFuncs use caller-allocated outputs and workspace.
- Constants use binary rdata assets and are materialized once by the runtime.
- `BufferizePass` remains authoritative for buffer shape, strides, size,
  location, workspace offset, and lifetime.
- Shared-memory typed call frames preserve ordinary internal PrimFunc call
  boundaries without keeping every pointer and dynamic tensor-descriptor field
  live in the device-function ABI. Private device functions frame only the
  descriptor fields referenced by their lowered bodies. Caller-allocated
  AutoTiling PrimFuncs remain `noinline` device functions so their temporary SSA
  values do not enlarge the caller's live range; operator-template helpers stay
  inlined within those leaf functions. Call frames occupy a backend-owned arena
  separate from the AutoTiling staging arena. The renderer assigns frame offsets
  from the device-call DAG, keeps caller and callee frames disjoint, and reuses
  storage across sibling or sequential calls.
  The maximum live call stack is rounded independently from the AutoTiling
  arena. The compiler-managed arena must fit its logical allocation limit, and
  the compiled kernel's total usage, including call frames and Triton-private
  lowering scratch, must fit the target's physical per-block shared-memory
  capacity.
- A shared device-context frame stores canonical dynamic launch values once;
  private device functions load only the fields used by their body. Static
  specialization values remain `tl.constexpr` direct parameters. This is an ABI
  representation owned by PyNTT and does not create compiler-visible TIR
  buffers.
- Function call-boundary policy is derived from the IR `FunctionRole`, never
  from function names. Shape-bucket selector and segment-root functions are
  `Dispatch`; the manifest marks them `compose_into_caller`, and the reader-only
  renderer structurally expands them before call-frame allocation, ABI
  liveness, and Triton code generation. This is mandatory composition rather
  than a request to Triton's optional inliner, so orchestration wrappers cannot
  survive as spill-prone device functions. Decoder and ordinary compute
  functions are `Compute`, while AutoTiling-generated leaf PrimFuncs are
  `ScheduledRegion`; both preserve their noinline device boundaries.
  Dispatch trace markers are retained. A direct operator helper owned by a
  Dispatch function moves to the owning top-kernel scope but remains a
  `noinline` compute leaf; composing an orchestration wrapper must not inline
  operator register state into the top kernel. Operator-template helpers may
  still inline inside a scheduled leaf.
- Triton top kernels consume the generated TIR buffer plan directly; Python
  runtime code must not recompute workspace placement.

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
