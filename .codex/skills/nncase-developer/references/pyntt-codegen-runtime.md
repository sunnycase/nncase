# PyNTT Codegen and Runtime ABI Guidelines

Use this reference when changing PyNTT TIR selection, codegen, Jinja rendering,
generated model dispatch, runtime tensor views, workspace allocation, rdata, or
`PrimFunction` calls.

## Hard Boundary

PyNTT does not run nncase AutoTiling. The compiler-to-PyNTT boundary is selected,
bufferized semantic TIR with globally addressable semantic buffers and
target-selected Shared resource reservations.

- AutoDistributed owns chip/die/block sharding.
- TIR Selection owns semantic operations, authoritative buffer descriptors,
  concrete target microkernel selection, and typed Shared workspace
  reservations.
- Bufferize owns storage identity, shape, strides, byte spans, locations, and
  caller-allocated function ABI. It also owns Shared lifetime/reuse, alignment,
  target allocation-policy rounding, arena size, and byte offsets.
- TIR passes own cross-operation memory effects, block/grid barriers, and trace
  scopes.
- PyNTT Jinja templates own all work below a block: warp/thread mapping,
  typed aliases over reserved Shared bytes, registers, copies, pipelines,
  encodings, implementation of selected microkernels, and tails.

PyNTT codegen must reject `ScheduledRegion`, executable `Grid`, `For`,
`PipelineFor`, executable Shared buffers, Register TIR buffers, storage
encodings, TileLoad, TileStore, and AutoTiling block-microkernel metadata.
Shared buffers are valid only as `NTTKernelOp` resource reservations and
caller-owned Shared arena ABI. Do not restore old scheduled-TIR handling or add
a compatibility path.

`BlockLocalData` and `BlockLocalRData` are per-block slices of globally
addressable backing pools. They are not Triton shared memory.

## Component Ownership

- `NTTTIRSelectionPass` lowers distributed IR directly to semantic TIR,
  including TensorLoad/TensorStore/reshard and logical views, then asks the
  target `ITIRMicroKernelSelector` for microkernel parameters and Shared
  reservations.
- `PyNTTLinkableModule` owns generated Python dispatch, input/output binding,
  shape buckets, workspace allocation, rdata materialization, host tensor
  descriptor binding, and launch arguments.
- `PyNTTKernelSourceConvertVisitor` validates direct TIR and emits typed helper
  models plus semantic metadata.
- `pyntt/pyntt/codegen/render.py` is the strict, reader-only manifest renderer.
- Jinja templates under `pyntt/pyntt/codegen/templates/` implement complete
  Triton operation schedules.
- `pyntt/pyntt/runtime/` validates `torch.Tensor` arguments and preserves
  requested pointer/view semantics.

Do not infer storage from names or recalculate compiler-owned byte offsets in
Python. Fix cross-layer contracts at their owning layer and update all users.

For matrix-like families, map each selected `(family, variant)` to one complete
algorithm file, such as `matmul/simt_fma.py.jinja` or
`matmul/mma.py.jinja`. The selector chooses the algorithm; the template
validates and implements that choice. Do not put GEMV, GEMM, and MMA schedules
behind branches in one wrapper, and do not split one algorithm across included
Jinja fragments. Generic address and syntax macros may remain shared.

## Manifest v9

Version 9 is the only accepted codegen manifest. Objects use exact schemas.

Each render kernel has only:

```text
metadata
helpers
device_functions
body_source
```

There are no pipeline execution tables, typed Shared aliases, local-buffer
tables, storage encodings, or compiler-generated reduction-phase plans.
Launch metadata carries the total Shared arena bytes and Bufferize-computed
arena alignment; a helper model may carry selected microkernel
family/variant/static parameters and named Shared byte-offset expressions. Each
helper contains exactly:

```text
template
model
arguments
workspace_arguments
```

`arguments` are operation-specific values. `workspace_arguments` are the exact
live compiler-owned pool/arena parameters referenced by that helper; do not
pass every top-level workspace conservatively. Unknown, missing, and obsolete
pre-v9 fields must fail validation.

Templates receive dtype, vector lanes, shapes, strides, distributed regions,
runtime dimensions, operation attributes, fixed worker geometry, target
resource capabilities, selected microkernel parameters, and Shared offsets.
Those inputs are sufficient to implement the backend schedule. The manifest
must not carry `num_warps` or renderer tuning candidates. A microkernel
`num_stages` may be serialized only when it affects compiler-reserved capacity.
Renderer-owned choices are emitted in `PYNTT_KERNEL_CONFIGS`; changes that fit
the existing resource reservation can be applied by re-rendering without model
compilation.

Launch metadata may also contain host tensor descriptor backing records. These
records contain only compiler-owned facts: descriptor ABI name, backing source,
fixed byte offset, scalar dtype, global logical shape/strides, and vector lane
shape. The renderer combines a backing record with the selected algorithm and
emits concrete `PYNTT_HOST_TENSOR_DESCRIPTOR_SPECS`, including TMA shape,
strides, block shape, and padding. Do not serialize an algorithm-specific TMA
block shape from C#.

## Producer/Consumer Contract

Every executable Jinja helper has one consumer, one producer, and one
`tle.gpu.warp_specialize` wrapper.

- A microkernel with an independent gmem-to-Shared transfer phase owns a real
  `tle.pipe`; the producer receives its writer and the consumer receives its
  reader.
- A helper without such a phase keeps the producer empty and runs semantic work
  in the consumer. Do not allocate fake Shared memory or duplicate loads merely
  to make both phases non-empty.
- The producer receives only values it uses. In particular, an empty producer
  receives `()`, not the helper ABI.
- TIR order and inter-operation barriers remain outside helper-internal phase
  scheduling.

The manifest carries target worker width, threads per block, aggregate
register-file capacity, and register allocation granularity. The renderer owns
the backend's physical warp-allocation rule and derives one aligned worker
register partition for every helper. A fixed low register count is invalid:
`setmaxnreg` may otherwise spill live top-function ABI values between adjacent
specialization regions.

## Caller-Allocated Workspace

Nested `PrimFunction` calls use strict caller-allocated output and workspace
semantics.

- The caller allocates `data` and `block_local_data` once for the active entry
  dispatch and passes workspace views directly to callees.
- Callees do not allocate replacement workspaces.
- Entry sizing includes all reachable callees and shape-bucket branches.
- Recursive call graphs fail before allocation/codegen.

Shared uses the same caller-owned lifetime semantics but is not a Python
runtime pointer:

- Bufferize gives a `PrimFunction` one Shared byte-arena workspace when needed.
- The generated top Triton function allocates the arena once.
- The allocation uses the exact alignment emitted by Bufferize; templates must
  not infer alignment from an operation name or encoding.
- Nested device functions receive a compile-time base byte offset.
- Operation-local named offsets are relative to that base.
- Templates create typed aliases with
  `tle.gpu.alloc(alias=..., alias_offset_bytes=...)`.

The backing `data` allocation is:

```text
max_local_data_bytes_per_shard * max_shard_count + max_collective_data_bytes
```

`data_pool_stride_bytes` is the per-shard stride. Collective storage is appended
once and is not part of that stride. `block_local_data_pool_stride_bytes` is an
independent backing stride.

When walking TIR for transitive requirements, handle `PrimFunction` before a
generic `BaseFunction` early return or the root body will be skipped.

## Buffer and Stride Rules

Keep these concepts distinct:

- a logical buffer's element strides;
- an external runtime tensor's ABI strides;
- a per-shard workspace backing stride.

Use the `TIR.Buffer` shape, strides, `MemSpan`, `PhysicalBuffer`, location, and
distributed type as the source of truth. Templates may derive tile offsets but
must not flatten a multidimensional descriptor and then invent its layout.

Inputs, outputs, Data, RData, ChipLocal, and BlockLocal pools are gmem-visible at
the boundary. Shared TIR buffers are non-addressable reservations. Their
canonical final `NTTKernelOp` operand is:

- `None` for zero buffers;
- the Buffer itself for one buffer;
- a flat Tuple for two or more buffers.

Empty, singleton, or nested workspace Tuples are invalid. Typed Triton Shared
memdescs and all register descriptors are created only inside a kernel
template.

## Host Tensor Descriptors

TMA-capable helpers receive either a host-created
`triton.tools.tensor_descriptor.TensorDescriptor` or one pointer to a device
table of host-encoded `CUtensorMap` entries. Do not call
`tl.make_tensor_descriptor` in generated device code.

- Build every descriptor entry from the authoritative global backing, never
  from a guessed local pointer.
- Use `single` only when one rectangular map describes the globally addressed
  operand. Use `table` for distributed local views: one 128-byte entry per
  hierarchy owner in one aligned CUDA byte buffer, selected by linear owner in
  device code. Derive entries from the complete ordered `SplitStage` mapping.
- For a table entry, use local descriptor coordinates. Acquire tensor-map proxy
  visibility and reinterpret the raw entry against the actual destination
  Shared memdesc before `tle.gpu.copy`.
- Unequal owner extents are represented by unequal entry shapes; TMA OOB fill
  handles the final tile. A partial `BlockCyclic(BlockSize > 1)` block is not a
  rectangle and must fail fast.
- Reject per-shard backing without a stable address/span contract and
  BlockLocal backing.
- Validate static shape, strides, byte offset, block shape, rank, and storage
  span before launch. Do not fall back to device descriptor construction.
- Cache one single descriptor or descriptor table per generated kernel ABI slot
  and replace it when the backing pointer or descriptor configuration changes.
- Keep the concrete descriptor layout renderer-owned so template-only TMA tile
  changes remain re-render-only changes.
- Model fused QKV as one concatenated logical N stream. Generate one N/K tile
  loop pair, route portions of each N tile to the Q/K/V descriptors, and stage
  those portions into disjoint regions of one Shared tile. Do not emit
  projection-specific Q, K, and V tile loops.

The Triton scratch allocator is unrelated to host descriptors. Install it only
for kernels whose grid synchronization path requires runtime scratch storage.

## Synchronization

TIR barriers cover dependencies between semantic helpers, global/chip-local
visibility, collectives, and object side effects. Template barriers cover
internal staging and compute.

- Use `tl.debug_barrier` for a block-local semantic dependency.
- Use `tle.distributed_barrier` only for a grid-wide dependency.
- Do not serialize template pipeline barriers into the manifest.

## RData Safety

`rdata`, `chip_local_rdata`, and `block_local_rdata` are immutable after loading
from binary assets. Never encode large rdata as base64.

If token 0 is correct but later tokens fail, check workspace/rdata overlap
before changing arithmetic templates:

- verify entry allocation includes the collective tail;
- compare Data, collective, and RData pointer ranges;
- verify nested callees use caller-provided strides and pointers;
- verify manifest/bin offsets are used directly.

## Template Fast Loop

For Jinja-template or reader-only renderer changes, re-render the existing
`kernel_params.json` and run the package in a fresh Python process. Do not
recompile nncase or the model first.

Recompile when changing selected TIR, selected microkernel resource contracts,
Shared reservation count/size/alignment, C# helper models, manifest schema,
function ABI, buffer/rdata layout, launch metadata, runtime signatures, or
target options.

## Validation Checklist

- Build `Nncase.Compiler` in Debug without RID-specific restore.
- Verify the PyNTT dump advances from AutoDistributed directly to TIRPass.
- Inspect final TIR for no scheduled regions, Grid/For/PipelineFor, executable
  Shared buffers, Register buffers, TileLoad/TileStore, storage encoding, or
  AutoTiling block-microkernel metadata.
- Verify target TIR microkernel metadata and Shared reservations agree, use
  canonical None/value/Tuple representation, and Bufferize assigns aligned
  in-bounds offsets.
- Validate manifest v9 with the Python reader.
- Verify every executable helper renders producer/consumer functions and one
  specialization wrapper.
- Inspect compiled metadata for the expected physical warp count and zero
  PTXAS stack/spill bytes.
- Re-render `generated_kernels.py` and inspect that tiling/pipeline logic comes
  from Jinja templates.
- For TMA helpers, verify TTGIR contains
  `async_tma_copy_global_to_local`, PTX contains
  `cp.async.bulk.tensor`, and generated source contains no
  `tl.make_tensor_descriptor`.
- Run focused C# target tests and Python package/benchmark tests.
- Run the smallest existing importer pytest with `NNCASE_TEST_TARGETS=pyntt`.
- For shared ABI changes, run one-layer Qwen3 and compare all generated tokens.
- Keep `tests_output/` and build products out of commits.
