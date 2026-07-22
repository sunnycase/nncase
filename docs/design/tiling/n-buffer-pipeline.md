# AutoTiling Loop Pipelines

## Status

This document defines the compiler-visible block-local pipeline contract shared
by AutoTiling, TIR, target models, and backend code generation. It is a clean
break from operation-owned or microkernel-owned pipeline selection. There is no
legacy pipeline configuration on `BlockMicroKernelCandidate`, no phase metadata
on ordinary expressions, and no pass that recovers a pipeline from an already
built loop body.

The first implemented asynchronous schedule is the PyNTT Triton NVIDIA
global-to-shared double buffer:

```text
template:          triton.loop.cp_async.n2.v1
stage count:       2
prefetch distance: 1
tail policy:       serial
execution scope:   one persistent block
```

AutoTiling currently chooses between two schedules for each eligible lexical
loop entry:

```text
N = 1: serial producer then consumer
N = 2: asynchronous fill/steady/drain
```

An ordinary block microkernel remains an operation implementation. It declares
its local execution cost, memory accesses, resource use, and storage-encoding
requirements, but it does not own loop staging or overlap.

## Design Principles

### Loop ownership

Pipeline overlap is a property of a lexical loop, not of a load, buffer,
operator, or microkernel. One loop decision owns:

- the loop entry and canonical domain axis;
- the selected stage count;
- all parent-to-local transfer channels crossing that loop entry;
- the backend template and synchronization protocol;
- the producer and consumer services used by the latency model;
- the physical stage count of every selected channel allocation;
- the final `PipelineFor` region.

This single ownership point prevents cost, storage, and lowering from selecting
different schedules.

### Construction, not recovery

Pipeline structure is created while lowering the selected TileGraph schedule.
The producer and consumer bodies are separate builder inputs from birth.
Lowering never scans an ordinary body to infer phases from call names, buffer
shapes, source text, traversal order, or annotations.

Validation may check a structure that has already been selected, but validation
must not invent missing ownership or phase information.

### Layer boundaries

`Nncase.Core` owns target-independent contracts:

- `PipelineTemplateId`;
- `PipelineSynchronizationProtocol`;
- `PipelineRegionPlan` and `PipelineStageChannelPlan`;
- `StagedBufferLayout`;
- symbolic fill/steady/drain estimates;
- `PipelineFor` and its exact channel bindings.

The target machine owns hardware facts:

- memory spaces and directed transfer edges;
- transfer bandwidth and latency;
- asynchronous transfer capability and supported stage counts;
- commit/wait costs;
- block synchronization cost;
- memory capacity and allocation granularity.

`ILoopPipelineBackend` owns representability and lowering identity:

- whether a stage count is implemented;
- whether one symbolic transfer channel is representable;
- the exact template ID and synchronization protocol.

The backend does not choose tile sizes, storage placement, stage count, or
channels. The hardware model does not contain Triton or Jinja identities.

## Solver Model

### Potential channels

For every iteration `TileNode` and lexical loop entry, AutoTiling enumerates
read-only parent-to-local placements that require an explicit transfer. A
channel identity is derived from stable TileGraph placement provenance:

```text
<buffer endpoint>.entry<loop entry>.<destination memory space>
```

Outputs, chip-scoped effects, direct-access edges, and placements without a
local allocation are not pipeline channels. The target backend receives the
source and destination memory spaces, dtype, symbolic local shape, full shape,
machine, and solver. It returns a symbolic legality expression; it does not
mutate the schedule.

For the current Triton contract a legal channel is an explicit global-to-shared
copy backed by a target edge with two-stage asynchronous support. The local
region must also be a compact rectangle representable by `tle.gpu.copy`.
`StorageLevel` is the sole mapping from a solved placement to a target memory
space. For every explicit input transfer, AutoTiling selects the exact nearest
dominating source placement before pipeline eligibility is considered. The
destination level determines the required directed transfer edge; the selected
source placement's level must map to that edge's source memory space. A
caller-owned root endpoint maps to `TargetMachineModel.RootMemorySpace`.
AutoTiling does not inspect a TIR `MemoryLocation` to derive this mapping;
physical location is an ABI and allocation concern of later TIR lowering.

This transfer-source decision is shared by serial and pipelined lowering. A
pipeline channel references it instead of independently inferring a source from
an internal value, a physical buffer location, or a post-solve traversal. A
shared-memory producer therefore shadows older outer views and cannot be
modeled as a global-to-shared asynchronous channel merely because the consumer
allocates another shared tile.

### Stage decision

Each eligible loop entry creates exactly two Boolean decisions:

```text
serial + asynchronous == 1
stage_count = serial + 2 * asynchronous
```

The asynchronous choice is constrained by construction:

- at least one selected placement is a pipeline channel;
- every selected channel is backend-representable;
- the loop has at least two iterations;
- two asynchronous schedules whose consumer regions overlap cannot both be
  selected.

Only one asynchronous loop schedule may be extracted for a `TileNode`. The
serial choice does not materialize `PipelineFor`; it remains the ordinary TIR
loop and allocation path.

### Storage coupling

The same `stage_count` expression is passed to storage-encoding selection as a
`StagedAllocationContext`. A candidate must provide its per-stage physical size
and stage stride. Capacity is modeled as:

```text
N = 1: physical_bytes
N > 1: N * stage_stride_bytes
```

Therefore an asynchronous decision cannot be selected without reserving its
complete staged allocation. Conversely, a serial decision cannot accidentally
retain a multiplied buffer. Extraction creates `StagedBufferLayout` only when
the selected stage count is greater than one.

The logical `TIR.Buffer` shape, strides, dtype, and storage encoding continue to
describe one stage. `StagedBufferLayout` adds only physical facts:

```text
StageCount
StagePhysicalBytes
StageStrideBytes
PhysicalBytes = StageCount * StageStrideBytes
```

This keeps ordinary buffer indexing independent of the pipeline depth.

## Cost Model

Microkernel candidates expose only `BlockMicroKernelExecutionCost.RegionCycles`.
They do not suppress or replace transfer, memory, synchronization, or pipeline
cost terms.

AutoTiling first builds the complete serial objective from the ordinary,
independently owned terms:

```text
serial_cycles = compute + memory + transfer + synchronization + encoding
```

For each potential loop pipeline it derives producer and consumer services from
the same selected loop region and placements that will be lowered:

- `P`: selected channel transfer bytes / bandwidth plus transfer latency;
- `C`: selected microkernel or base compute cycles in the loop region;
- `I`: iterations per loop invocation;
- `V`: invocation count;
- control costs: commit, wait/acquire, and release from the selected protocol
  and target machine.

The two alternatives are:

```text
serial = V * I * (P + C)

producer_service = P + commit
consumer_service = wait_acquire + C + release
II = max(producer_service, consumer_service)
pipeline = V * (producer_service + (I - 1) * II + consumer_service)
```

The primary objective subtracts only the nonnegative overlap saving when the
two-stage decision is selected:

```text
total = max(0, serial_cycles - asynchronous * max(0, serial - pipeline))
```

This preserves one accounting source for every ordinary cost. No candidate
claims a cost-ownership mask, and no post-solve correction tries to remove
double-counted terms. Pipeline selection priority is only a deterministic
secondary tie break after predicted latency.

Chip-scoped transfer contention is multiplied by the active persistent block
count. Block-scoped compute and shared-memory work remain local to one block.

## Direct TIR Construction

The solver result records every selected explicit transfer as a destination
`(TileNode, BufferIdentity, LoopEntry, StorageLevel)` and an exact source
placement or caller ABI binding. A `SelectedLoopPipeline` contains the exact
loop entry, domain axis, plan, estimate, and references to those selected
transfers. TileGraph lowering checks that the solved loop-order permutation
still maps the selected entry to that axis, then creates a
`PipelineForBuilder` at that loop position.

TIR construction indexes every created view by the same complete placement
identity. An explicit `TileLoad` obtains its source only from the solved
transfer binding. The latest-view index remains an implementation aid for
zero-copy logical views; it is not consulted to choose a memory-transfer
source, and physical `Buffer` locations are never used to reconstruct a lost
storage level.

For each selected channel, placement construction produces all four binding
parts together:

1. the stable channel descriptor;
2. the logical access variable;
3. the allocation/view expression;
4. the staged root `TIR.Buffer` carrying `StagedBufferLayout`.

The explicit parent-to-local `TileLoad` is inserted directly into
`PipelineForBuilder.Produce`. Normal loop work is inserted directly into its
consumer body. A selected channel without an owned allocation, explicit copy,
or exact binding fails immediately.

`PipelineForBuilder.Build` requires every declared channel exactly once and
requires both bodies to be non-empty. It creates one logical stage expression:

```text
logical_stage = (loop_var - loop_start) / loop_step
stage = logical_stage % StageCount
```

Each phase receives a normal single-stage buffer alias sharing the root
`PhysicalBuffer`:

```text
MemSpan.Start = root.Start + stage * StageStrideBytes
MemSpan.Size  = StagePhysicalBytes
```

The phase body is cloned against those explicit aliases. The resulting
`PipelineFor` stores producer body, consumer body, plan, region identity,
channel descriptors, access variables, allocation expressions, and staged root
buffers as first-class operands. No later pass reconstructs phases or ownership
from an already-lowered loop body.

## Full And Tail Loops

Loop peeling preserves `PipelineFor` for both full and tail partitions so the
channel bindings and allocation remain explicit. The backend interprets the
partition according to `PipelineTailPolicy`:

- full or unpartitioned regions use the asynchronous N=2 template;
- tail regions execute the same producer and consumer bodies serially with the
  same staged allocation.

The current tail path intentionally performs a synchronous copy, block barrier,
consumer, and release barrier for each peeled iteration. It does not launch an
incomplete asynchronous fill/drain sequence.

Reduction codegen accepts only homogeneous full/tail pairs: both ordinary
`For`, or both `PipelineFor` with identical plans and bindings. Mixed
representations fail fast.

## Synchronization

`PlanMemorySynchronization` treats `PipelineFor` as one structured protocol.
Its staged accesses and allocations are explicit operands, so synchronization
analysis does not need to rediscover the transfer channel. Template-owned
commit, wait/acquire, and release events discharge the corresponding block
effects and prevent duplicate barriers from being inserted around the region.

Synchronization cost is derived from the directed transfer edge and the target
block synchronization specification. A wait is charged an additional block
barrier when the asynchronous primitive does not itself provide acquire
semantics. Stage reuse is charged a release barrier when required by the
protocol.

## PyNTT Manifest And Rendering

PyNTT codegen emits manifest version 6. Each pipeline execution contains only
the selected structured contract:

- region, schedule, and template identities;
- stage count, prefetch distance, synchronization, tail policy, and partition;
- loop variable and bounds;
- exact channel descriptors and staged allocations;
- producer and consumer source fragments.

The reader validates this schema exactly and dispatches the selected template.
For `triton.loop.cp_async.n2.v1`, Jinja owns the concrete fill/steady/drain
syntax. It emits `tle.gpu.async_commit_group`, `async_wait_group`, and block
barriers according to the manifest protocol. Stage aliases lower through
`buffered_tensor.slot(tl.cast(stage, tl.int32))`, which satisfies the FlagTree
frontend contract for both compile-time constants and dynamic `tl.range`
induction values.

The renderer never scans generated source to infer a pipeline schedule. Editing
only the Jinja template or reader-only renderer can use the existing manifest
re-render workflow without recompiling nncase or the model.

## Extension Rules

Adding N-buffer depths beyond two requires all of the following in one change:

1. target asynchronous transfer capability for the stage count;
2. backend legality and a versioned template descriptor;
3. a symbolic cost formula for its exact fill/steady/drain protocol;
4. storage candidates that provide a legal stage stride;
5. renderer and protocol tests for full and tail partitions.

Adding another backend implements `ILoopPipelineBackend`; it must not add
backend conditions to Core or GraphTiler. Adding another transfer kind extends
the target memory graph and backend channel legality; it must not be inferred
from op names.

## Invariants

- Pipeline stage count belongs to one lexical loop decision.
- The same decision controls legality, cost, capacity, TIR layout, and template.
- A staged allocation exists only for a selected multi-stage schedule.
- Every selected channel has exactly one producer, consumer access, allocation,
  and root buffer binding.
- Producer and consumer bodies are distinct before `PipelineFor` exists.
- No ordinary expression carries pipeline phase metadata.
- No pass recovers a pipeline from an ordinary loop.
- No backend silently changes stage count, tile, placement, or storage encoding.
- No hidden allocation, transfer, synchronization, spill, or serial fallback is
  allowed.
- Unsupported capability or incomplete binding fails before source rendering.

## Validation

Required coverage includes:

- backend and hardware stage-capability intersection;
- N=1 versus N=2 solver selection under changed producer/consumer costs;
- exact capacity multiplication and extracted `StagedBufferLayout`;
- direct `PipelineFor` construction with complete channel bindings;
- clone, rewrite, type inference, canonicalization, and tail peeling;
- synchronization planning without duplicate barriers;
- strict manifest-v6 validation and Jinja full/tail protocol rendering;
- CUDA numerical execution of the structured N=2 path;
- Qwen3 integration through the existing AutoTiling and PyNTT test flow.

Benchmark scripts and TTGIR/PTX inspection are validation tools. Their source
heuristics are not compiler contracts and must not feed policy back into Core,
AutoTiling, TIR, or rendering.
