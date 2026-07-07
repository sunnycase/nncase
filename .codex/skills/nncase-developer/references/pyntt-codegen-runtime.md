# PyNTT Codegen and Runtime ABI Guidelines

Use this reference when changing PyNTT TIR codegen, generated model dispatch, Jinja rendering, runtime tensor views, workspace allocation, rdata materialization, or PrimFunction call lowering.

## Ownership Model

PyNTT follows the same broad contract as NTT: the compiler lowers selected TIR into top-level runtime-dispatched kernels, and backend templates implement the block-local/device work described by the manifest.

- TIR and bufferize own buffer regions, memory locations, offsets, shapes, strides, and function ABI.
- `PyNTTLinkableModule` owns generated Python dispatch: input/output binding, shape-bucket branch dispatch, function call lowering, workspace allocation, rdata materialization, and launch arguments.
- `PyNTTKernelSourceConvertVisitor` owns manifest/kernel metadata emitted from one lowered PrimFunction body.
- PyNTT templates under `pyntt/pyntt/codegen/templates/` consume manifest metadata only. They should not infer global storage layout from names or ad hoc offsets.
- `pyntt/pyntt/runtime/` owns Python tensor views and runtime helpers. It must preserve pointer/view semantics requested by generated model code.

Do not fix a cross-layer mismatch by hard-coding a model, segment, token index, kernel name, or shape bucket. Update the ABI contract and all users together.

## Caller-Allocated Workspace

Nested PrimFunction calls must use strict caller-allocated workspace semantics.

- The caller allocates `data` and `block_local_data` once for the active entry dispatch and passes workspace views to callees.
- Callees must not allocate replacement workspaces when the caller supplied workspace arguments.
- Entry workspace sizing must include the transitive requirements of all reachable callee PrimFunctions, including callees behind shape-bucket `IfThenElse` branches.
- Compute the backing `data` allocation as:

```text
max_local_data_bytes_per_shard * max_shard_count + max_collective_data_bytes
```

- `data_pool_stride_bytes` is the local per-shard stride. The collective tail is appended once after all per-shard local pools and must not be included in the stride.
- `block_local_data_pool_stride_bytes` is independent from `data_pool_stride_bytes`.
- Recursive or cyclic PrimFunction call graphs should fail fast with a clear error instead of silently under-allocating.

When traversing TIR for callee requirements, remember that `PrimFunction` is also a `BaseFunction`. If a traversal returns early for `BaseFunction` before handling `PrimFunction`, it will skip the root body and miss shape-bucket callees.

## Buffer View Strides

Keep two stride concepts separate:

- Runtime tensor view stride: how a specific buffer argument is indexed across shards.
- Workspace backing pool stride: how much storage belongs to one shard in the caller-allocated pool.

For a non-distributed `Data` buffer view, the runtime tensor view stride is `0`; every shard sees the same logical buffer view. If that buffer is also passed as a callee workspace argument, the callee still needs the owning workspace backing stride so its own distributed local buffers and collective tail are addressed correctly.

Do not derive either stride from Python pointer arithmetic in templates. Emit both through generated model arguments from TIR/buffer metadata.

## RData Safety

Treat `rdata`, `chip_local_rdata`, and `block_local_rdata` as immutable after materialization.

If an LLM case has token 0 correct but token 1 or later wrong, check for workspace overwrite before changing math kernels:

- Compare generated entry allocation against callee `collective_data_pool_bytes`.
- Inspect `model.py` for suspicious allocation like `data_local * shard_count + 0` when nested kernels use collective buffers.
- Compare device pointer ranges for `data`, collective tail, and `rdata` if corruption is suspected.
- Check whether the corrupted values match temporary workspace data.
- Validate that generated buffer views use `rdata` offsets from manifest/bin metadata, not recalculated offsets.

A common failure mode is collective storage starting exactly at `data + data_stride * shard_count` while the entry allocation omitted the collective tail. CUDA allocation reuse can then place `rdata` immediately after `data`, so collective stores overwrite constants and the first token may still appear correct.

## Shape-Bucket Dispatch

Shape-bucket entry functions can contain nested `IfThenElse` dispatch and direct PrimFunction calls.

- Codegen must preserve the existing compiler pipeline and consume lowered TIR; do not introduce a separate Python-side dispatcher model.
- Segment kernels generated around call sites must use the same prepared workspace context as the surrounding function.
- Branches that contain only function calls still affect transitive workspace requirements.
- Runtime scalar expressions should be built from existing dimension/range expressions and shape environment values.

## Template Fast Loop

When changing only PyNTT templates or reader-only rendering:

- Re-render from the existing generated manifest with `render_generated_kernels`.
- Do not recompile the model first unless the manifest, TIR/codegen metadata, rdata layout, runtime model signature, or target options changed.
- After re-rendering, run the generated package in a fresh Python process to avoid stale Triton/JIT state.

When changing C# manifest emission, function ABI, workspace allocation, or rdata metadata, recompile the model because the compiler-to-PyNTT boundary changed.

## Validation Checklist

Before finishing PyNTT codegen/runtime ABI changes:

- Build with `dotnet build -c Debug --no-restore`.
- Inspect generated `model.py` and confirm entry workspace allocations include transitive callee local and collective requirements.
- Inspect generated `kernel_params.json` or `metadata.json` for `data_pool_bytes`, `collective_data_pool_bytes`, `block_local_data_pool_bytes`, and shape-bucket dispatch metadata.
- Run the smallest existing pytest case that exercises the changed ABI with `NNCASE_TEST_TARGETS=pyntt`.
- For LLM dispatch/workspace/rdata changes, run one-layer qwen3 and confirm token 0 and later tokens both match.
- Keep generated outputs under `tests_output/` out of source commits unless explicitly requested.
