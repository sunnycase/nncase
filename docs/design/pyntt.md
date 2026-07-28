# PyNTT Design

## Status

Accepted architecture, revised 2026-07-27.

PyNTT is a formal nncase inference backend and the Python-DSL foundation for
megakernels. Triton is the only supported backend. Generated models execute
through a pure Python runtime with `torch.Tensor` inputs and outputs.

This revision deliberately removes compiler-owned block-level AutoTiling from
the PyNTT path. The compiler emits selected, bufferized semantic TIR. PyNTT
kernel templates own all scheduling below a physical block.

## Design Decision

The boundary is:

```text
nncase IR
  -> target-independent optimization
  -> AutoVectorize / AutoPacking
  -> AutoDistributed (chip/die/block placement)
  -> TIR Selection + target microkernel/resource selection
  -> bufferize + shared-arena offsets + function ABI + synchronization
  -> PyNTT manifest v8
  -> reader-only Jinja renderer
  -> Triton kernels and Python model
```

PyNTT does **not** run the nncase AutoTiling stage. CPU/CUDA NTT targets may
continue to use AutoTiling; stage registration is target-owned.

The compiler is responsible for graph semantics, distribution, storage
identity, inter-function ABI, and concrete target resource reservations. The
backend is responsible for turning one selected semantic operation over
global-memory descriptors and byte-offset workspace reservations into an
efficient block-local implementation.

## Goals

- Provide a production inference target named `pyntt`.
- Generate a Python model directory instead of a kmodel.
- Reuse nncase importers, optimization, vectorization, packing,
  AutoDistributed, TIR Selection, bufferization, and pytest infrastructure.
- Preserve chip/die/block sharding selected by AutoDistributed.
- Keep dtype, shape, stride, byte span, distribution, dynamic dimensions,
  constants, side effects, and function ABI explicit at the compiler/backend
  boundary.
- Let Triton templates choose block-local tiles, memory staging, pipelines,
  encodings, microkernels, and tail handling.
- Allow template-only kernel development by re-rendering an existing manifest,
  without recompiling nncase or the model.
- Fail at the boundary when compiler-scheduled TIR or unsupported semantics
  reach PyNTT.

## Non-Goals

- Supporting the old PyNTT manifest or scheduled-TIR protocol.
- Interpreting a kernel spec at runtime.
- Emitting C++ kernels or using the native NTT runtime ABI.
- Supporting non-Triton Python DSL backends in this revision.
- Searching warp/thread/register/shared-memory schedules in nncase.
- Silently falling back to NTT, C++, PyTorch, or an ordinary kernel.

## Ownership

### nncase Compiler

The compiler owns:

- import, shape/range inference, canonicalization, and high-level fusion;
- vector dtypes and packed tensor layouts;
- chip/die/block distribution and reshard decisions;
- TIR semantic operation selection;
- global backing buffers, shapes, strides, byte offsets, byte spans, and
  memory locations;
- caller-allocated output and workspace ABI;
- target microkernel family/variant selection and the static parameters needed
  to determine its resource requirements;
- typed Shared workspace reservations, lifetime/reuse scheduling, alignment,
  allocation-policy rounding, arena size, and byte offsets;
- constants and rdata binary layout;
- nested `PrimFunction` call structure;
- object side effects, grid-wide synchronization, and semantic trace scopes;
- launch hierarchy and target capabilities required by the backend.

The compiler must not choose or serialize:

- warp/thread decomposition inside the selected block microkernel;
- register allocation;
- Triton Shared memdesc shape/layout/encoding or aliases;
- cp.async/TMA/TLE copy pipelines;
- Triton encodings or layout conversions;
- reduction phase plans internal to one operation.

### PyNTT Templates

Templates under `pyntt/pyntt/codegen/templates/triton/` own:

- mapping a block-local semantic operation to Triton programs;
- `tl.arange`/`tl.range` loops and masked tails;
- register tiles and typed Shared memdesc views;
- `tle.gpu.alloc(alias=..., alias_offset_bytes=...)`, copies, and pipeline
  stages within the compiler-reserved byte arena;
- producer/consumer synchronization internal to a template;
- Triton encodings and implementation of the selected microkernel contract;
- use of `tl.dot`, reductions, vectorized accesses, and backend hints;
- architecture-specific decision trees and optional Triton autotuning.

A template receives actual dtype, logical shape, physical strides,
distribution, runtime dimension expressions, target capability, operation
attributes, selected microkernel parameters, and named Shared byte offsets. It
chooses the typed alias shape and layout, including whether an alias uses an MMA
Shared layout. It must not recover facts from generated names or assume a
Qwen-specific shape.

### Generated Python Runtime

Generated `model.py` and the PyNTT runtime own:

- `torch.Tensor` argument validation;
- output allocation and result view materialization;
- one-time rdata loading and reusable workspace allocation;
- shape-bucket selection;
- direct Triton launch binding;
- passing dimensions, strides, workspace pointers, and tuning choices;
- load/run separation and stable model state.

The runtime does not reinterpret TIR, select an operator implementation, or
allocate/pass Shared memory. One Shared arena is created inside the generated
Triton kernel.

## Target Pipeline

`ITarget.IsAutoTilingEnabled` controls whether the compiler creates the
AutoTiling stage. `NTTTarget` enables it. `PyNTTTarget` disables it and exposes
no alternative hidden tiling pipeline.

For PyNTT the relevant late pipeline is:

1. AutoDistributed chooses chip/die/block SBP layouts.
2. `NTTTIRSelectionPass` lowers the distributed graph directly to semantic TIR,
   asks the target selector for a concrete microkernel, and materializes its
   typed Shared workspace reservations.
3. target TIR passes lower reshard storage; Bufferize schedules global and
   Shared storage, assigns arena offsets, and forms function workspace ABI;
   later passes inline single-use helpers where legal, plan semantic memory
   synchronization, and canonicalize index expressions.
4. PyNTT codegen validates the resulting TIR and emits manifest v8.

There must be no `AutoTilingPass` dump for a PyNTT compilation.

## Direct Selected-TIR Contract

PyNTT accepts bufferized semantic TIR, including:

- `PrimFunction`, direct nested `PrimFunction` calls, `Sequential`, and
  `Return`;
- input/output/workspace/rdata buffers in globally addressable memory;
- target-selected Shared reservation buffers carried only by the final
  `shared_workspace` operand of an `NTTKernelOp`;
- logical buffer aliases and subviews with explicit shapes and strides;
- semantic NTT operations such as TensorLoad, TensorStore, reshard,
  elementwise, matmul, reduce, normalization, attention, and packed operators;
- block or grid barriers required by cross-operation memory effects;
- codegen trace scopes that preserve fusion/operator names;
- dynamic `Dimension` expressions with proven ranges.

PyNTT rejects, with the owning function and construct in the error:

- `PrimFunction` with `FunctionRole.ScheduledRegion`;
- executable `Grid`, `For`, or `PipelineFor` schedules;
- executable use of `MemoryLocation.Shared`, or any
  `MemoryLocation.Register` buffer;
- storage encodings selected by the compiler;
- `TileLoad` or `TileStore`;
- AutoTiling block-microkernel metadata; target TIR microkernel metadata is
  required for operations that reserve Shared workspace;
- recursive device-function call graphs;
- unsupported semantic TIR operations.

This validation is mandatory. Removing a check to admit old scheduled TIR is
not a supported migration path.

## Memory Model

Semantic tensor buffers visible at the compiler/backend boundary are globally
addressable. Their TIR descriptor is authoritative.

- `Input` and `Output` are runtime tensors in device global memory.
- `Data` is caller-allocated mutable global workspace.
- `RData` is immutable global constant storage loaded from binary assets.
- `ChipLocalData` and `ChipLocalRData` are chip-scoped globally addressable
  pools used for UMA views and collectives.
- `BlockLocalData` and `BlockLocalRData` are per-block slices of globally
  addressable backing pools. The name describes ownership, not Triton shared
  memory.

`MemoryLocation.Shared` has a narrower contract. Such a `TIR.Buffer` is a
compiler-owned resource reservation, not a semantic pointer and not directly
loadable/storable by PyNTT codegen:

- every `NTTKernelOp` has one final `shared_workspace` operand;
- zero reservations are represented by `None`;
- one reservation is represented by the Buffer itself;
- two or more reservations are represented by a flat Tuple;
- empty/singleton/nested workspace Tuples are invalid;
- Bufferize owns reservation lifetime, reuse, alignment, arena offsets, target
  allocation-size policy, and capacity validation;
- PyNTT serializes named byte offsets and the total arena byte size;
- Jinja creates typed aliases over the byte arena and owns their Triton/MMA
  layout.

Registers remain entirely backend-private and never appear as TIR buffers.

Every compiler-provided buffer descriptor includes enough information to build
an address without guessing:

- scalar dtype and vector lane shape;
- logical dimensions;
- physical element strides;
- byte span, base storage, and byte offset;
- memory location;
- distributed tensor type and placement when applicable;
- runtime stride arguments for externally supplied strided tensors.

Templates may derive a local tile from these descriptors. They may not replace
or reinterpret the backing layout.

## Distribution and Launch

nncase distributes only at chip/die/block levels. A logical topology such as
`yx=4x8` maps both logical axes to the physical block level through required
hierarchy-level metadata.

AutoDistributed determines each block's shard region. Generated code passes the
physical block id and hierarchy. Templates calculate only within that local
shard. They must not expand work to the full undistributed tensor or alter the
selected shard.

The generated model owns launch count. Persistent launch policy normally uses
the configured block hierarchy, for example one block per resident SM. The
kernel owns all work loops performed by that block.

`num_warps`, producer allocation, register partitioning, and resource-neutral
tuning parameters are backend launch choices, not nncase loop schedules. A
selected microkernel parameter such as `num_stages` becomes part of the compiler
contract when it determines a Shared reservation. Dynamic dimensions are
runtime arguments and must not be declared `tl.constexpr`.

The manifest carries target worker capabilities and the distributed launch
hierarchy, but it does not carry `num_warps`, `block_size` candidates, or
resource-neutral renderer tuning candidates. Rendering derives a valid block
execution geometry and emits it in `PYNTT_KERNEL_CONFIGS` beside the Triton
kernels. `model.py` reads that backend-owned table at runtime. Backend policy
changes that fit the existing reservation require only re-rendering the
manifest. Changing the selected microkernel, its reserved pipeline depth, or
target capability requires model recompilation.

## Function ABI

PyNTT uses the common `PrimFunction` ABI:

- input and in/out buffers first;
- caller-allocated output buffers next;
- caller-allocated workspaces last;
- explicit logical results bind to either an input or output storage parameter.

Nested functions receive workspace pointers from their caller. A callee must
not allocate a replacement workspace. Entry allocation includes the transitive
workspace requirement of every reachable callee and shape-bucket branch.

Shared workspace follows the same caller-owned lifetime rule without becoming a
runtime pointer ABI. Bufferize adds one Shared byte-arena workspace to a
`PrimFunction` when required. Nested device functions receive a compile-time
base byte offset into the entry arena; operation-local offsets are relative to
that base. The generated top Triton function creates the arena once, and Python
runtime dispatch never sees it.

Single-use semantic helpers may be inlined in TIR, but reusable decoder-layer
functions remain device functions. Codegen passes arguments directly; there is
no call-frame structure or Python-side offset reconstruction.

## Synchronization

Synchronization has two owners:

- TIR owns synchronization between semantic operations when global workspace,
  chip-local state, object side effects, or cross-block visibility requires it.
- A kernel template owns synchronization between its internal load, pipeline,
  compute, and store phases.

`tl.debug_barrier` is used for block-local semantic dependencies.
`tle.distributed_barrier` is emitted only for a TIR grid-wide dependency.
Template-internal barriers are not represented as manifest pipeline tables.

## Manifest v8

`kernel_params.json` is a reader-only rendering manifest. Version 8 is the only
accepted version.

The root object contains exactly:

```text
pyntt_codegen_manifest_version
target_kind
backend
functions
```

Each function contains exactly its identity, module kind, entry marker, and
`render_kernels`. Each render kernel contains exactly:

```text
metadata
helpers
device_functions
body_source
```

Metadata carries semantic operation attributes, tensor names, distributed
launch metadata, fixed worker geometry, target resource capability, and total
Shared arena bytes. Helper models may carry a selected microkernel
family/variant, static parameters, and named Shared byte-offset expressions.
They do not carry typed Triton aliases, encodings, copies, or pipeline bodies.
Device functions carry a direct parameter ABI and body source.

Each helper contains exactly:

```text
template
model
arguments
workspace_arguments
```

`arguments` are operation-specific parameters. `workspace_arguments` are only
the live compiler-owned pool or Shared-arena parameters referenced by that
helper. This exact list avoids extending every specialization region across
unrelated top-function ABI values.

Version 8 has no pipeline execution tables, typed Shared aliases, local-buffer
tables, storage encodings, or compiler-generated reduction bodies.
`num_warps` and launch tuning remain renderer-owned. A selected microkernel may
include `num_stages` when it changes compiler-reserved workspace capacity. The
reader validates exact object keys and rejects unknown or missing fields. There
is no v7 reader, upgrade shim, or compatibility fallback.

## Generated Package

A generated package includes:

- `kernel_params.json`: stable compiler-to-template boundary;
- `generated_kernels.py`: Jinja-rendered Triton source and backend-owned
  `PYNTT_KERNEL_CONFIGS`;
- `metadata.json`: runtime function/tensor/result ABI;
- `model.py`: direct launch and load/run implementation;
- `specs.py` and `runtime_config.py`: runtime descriptors;
- `rdata.py` plus `assets/*.bin`: constant bundle metadata and binary payloads;
- `requirements.txt`, `README.md`, and `__init__.py`.

Large constants are binary assets, never base64 Python literals. Workspace and
rdata allocations are cached and reused between runs.

## Kernel Template Contract

One selected semantic TIR call maps to one Jinja helper model. A top Triton
kernel executes the selected `PrimFunction` body by directly calling these
`@triton.jit` helpers and nested device functions.

Matrix-like templates are organized first by semantic operation family and then
by selected algorithm:

```text
triton/kernels/
  matmul/
    simt_fma.py.jinja
    simt_fma_smem_pipeline.py.jinja
    mma.py.jinja
  qkv_parallel_linear/
    simt_fma.py.jinja
    mma.py.jinja
  matmul_glu/
    simt_fma.py.jinja
    mma.py.jinja
```

Every target microkernel `(family, variant)` maps to exactly one complete
algorithm file. The file owns its full block-local schedule, including loops,
copies, pipeline, encodings, accumulation, and tails. It may import generic
syntax/address helpers, but algorithm files must not include one another or
act as wrappers around shared algorithm fragments. Packing remains a typed
operand/layout property when it does not change the algorithm.

A partial compiler-generated loop plus a template fragment is forbidden.
Unsupported combinations fail during selection, manifest emission, or
rendering with dtype, shape, layout, operation, family, and variant context.

Typical matrix templates should:

1. validate the selected variant against legal local M/N/K extents, dtype,
   packing, and target capability;
2. implement only that selected algorithm rather than reselecting a
   GEMV/GEMM/MMA path in Jinja;
3. alias the compiler-reserved Shared offsets with template-owned typed shapes,
   encodings, and MMA-layout policy;
4. stage gmem operands and pipeline them when profitable;
5. handle full and tail tiles with masks;
6. accumulate at the required precision and store only the local output shard.

The same principle applies to reduction, normalization, attention, reshard,
and elementwise templates.

### Producer/Consumer Execution

Every executable helper has a consumer function, a producer function, and one
`tle.gpu.warp_specialize` wrapper. This is the common block-internal execution
ABI used by standalone kernels and future megakernel composition.

- A selected algorithm with an independent gmem-to-Shared transfer phase owns a
  real `tle.pipe`. Its producer receives the writer, its consumer receives the
  reader, and the template owns acquire/commit/wait/release ordering.
- A helper without a legal independent transfer phase runs its semantic work in
  the consumer and has an empty producer. It must not allocate fake Shared
  memory, duplicate traffic, or move TIR-visible synchronization merely to make
  the producer non-empty.
- An empty producer receives no helper arguments. Live operation and workspace
  parameters are passed only to the phase that uses them.
- Each helper keeps its own specialization region so TIR operation order,
  caller-owned Shared-arena alias lifetimes, and semantic barriers remain
  explicit.

The renderer derives the worker register count from target aggregate
register-file capacity, register allocation granularity, compute-warps, and the
Triton backend's physical worker-allocation group. The same aligned partition is
used by all helper variants. This avoids PTX `setmaxnreg` save/restore spills for
top-function values that remain live between helper calls.

## Dynamic Shapes

Dynamic shape expressions remain nncase `Dimension` expressions through TIR and
manifest emission. Capacity allocation uses proven range maxima; execution uses
the actual runtime value.

Templates receive actual dimensions and derive masks and loop bounds from them.
Shape buckets are selected in generated runtime code, never inside the Triton
kernel. A dynamic dimension must not be made `tl.constexpr` merely to simplify
template code.

## Template Development Loop

Changing only a Jinja template or `pyntt/pyntt/codegen/render.py` must not require
model recompilation:

```sh
export PYTHONPATH="$PWD/pyntt:${PYTHONPATH}"
python - <<'PY'
from pathlib import Path
from pyntt.codegen.render import render_generated_kernels

generated = Path("tests_output/test_qwen3/infer/pyntt/noptq/CodeGen/pyntt")
render_generated_kernels(generated)
print(generated / "generated_kernels.py")
PY
```

Recompile only when the selected TIR, selected microkernel resource contract,
Shared reservation sizes/count/alignment, manifest schema/model, function ABI,
buffer/rdata layout, launch metadata, runtime signature, or target options
change. Alias layout, copy/pipeline implementation, and other template changes
that fit the existing reserved capacity require only re-rendering. Run
re-rendered packages in a fresh Python process to avoid stale Triton JIT state.

## Error Policy

PyNTT is fail-fast:

- unsupported IR or TIR is a compile error;
- invalid manifest shape is a render error;
- unsupported template dtype/layout/shape is a render or Triton compile error;
- invalid runtime tensor shape/dtype/device/stride is an argument error;
- unavailable CUDA/Triton is an explicit environment error.

Errors include function, operation, buffer, dtype, shape, and template context.
No path returns fake success or changes backend silently.

## Testing

Validation is layered:

1. C# target tests verify stage ownership, direct TIR selection, canonical
   None/value/Tuple Shared operands, Bufferize allocation/offsets, manifest v8,
   strict rejection of scheduled TIR, function ABI, and generated source.
2. Python package tests verify exact manifest reading, rendering, runtime tensor
   validation, workspace/rdata reuse, and removed-field rejection.
3. Existing importer pytest suites run with `NNCASE_TEST_TARGETS=pyntt`.
4. Focused CUDA tests compare generated outputs with PyTorch.
5. Qwen3 one-layer is the integration gate for dynamic dimensions,
   distribution, packed operators, attention, nested functions, and constants.
6. Benchmarks and TTGIR/PTX inspection validate template performance; they do
   not enforce a compiler-owned pipeline representation.

Every PyNTT pipeline test must assert that no AutoTiling stage, scheduled
function, Register buffer, TileLoad/TileStore, storage encoding, or AutoTiling
microkernel metadata survives. Shared reservation buffers and target TIR
microkernel metadata must be validated together.

## Extension Rules

To add an operation:

1. add or reuse the semantic TIR op and its memory effects;
2. select it in `NTTTIRSelectionPass` with authoritative buffers;
3. add target microkernel/resource selection when the operation needs Shared
   storage;
4. add a typed manifest model in PyNTT codegen;
5. map each selected variant to one Jinja algorithm file that owns the complete
   block-local schedule and typed aliases;
6. add strict reader validation only if the manifest schema changes;
7. add a focused compiler-to-CUDA correctness test;
8. inspect performance artifacts for matrix/reduction/attention kernels.

Do not add a compiler `Grid`, TileLoad, pipeline body, Triton encoding, or typed
Shared alias to make a template easier to write. Shared TIR buffers are allowed
only as selected resource reservations.

## Acceptance Criteria

This architecture is implemented when:

- PyNTT compilation skips AutoTiling while NTT compilation retains it;
- PyNTT codegen accepts direct selected/bufferized semantic TIR;
- scheduled TIR is rejected at the codegen boundary;
- target microkernel resource reservations are bufferized into one Shared arena
  and emitted as named byte offsets;
- manifest v8 contains no compiler-generated block schedule or typed Shared
  alias representation;
- templates own loops, tails, shared staging, pipelines, encodings, and
  microkernels;
- template-only edits can be re-rendered without model recompilation;
- focused C# and Python tests pass;
- one-layer Qwen3 compiles and matches the reference on CUDA.
