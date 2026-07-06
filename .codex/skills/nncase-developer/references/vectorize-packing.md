# Vectorize and Packing Guidelines

Use this reference when changing nncase vectorization, NTT/PyNTT vectorized ops, or packed operator layouts.

## Pipeline

The relevant compiler order is:

1. Pre-auto-distributed decomposition runs on high-level semantic ops.
2. `AutoVectorizePass` runs target-registered egraph vectorization rules.
3. Post-vectorize dataflow rules fold simple post-ops into vectorized ops.
4. `AutoPackingPass` rewrites selected vectorized/high-level ops to target packed ABI ops.
5. `AutoDistributedPass` chooses sharding and inserts boxing.
6. TIR selection lowers the selected IR to local TIR kernels.

Do not create a side pipeline for vectorization. Register target-specific vectorize and packing rules through the target, and keep `InferRangePass`/`OptimizeByRangePass` after broad rewrites when types/ranges can change.

## Core Model

`Pack(input, lanes, axes)` is the IR operation that moves logical tensor extents into a vector dtype:

- Each packed shape axis is divided by its lane using ceil division for fixed shapes, or symbolic division for dynamic shapes.
- The dtype becomes `VectorType(elem, lanes...)`.
- Packing an already vectorized tensor prepends new lanes to the existing vector lanes.
- `Unpack(input, lanes, axes)` restores shape extents and removes the first `axes.Count` vector lanes.
- `MaskVectorType` is the boolean vector form. Do not create `VectorType(Boolean, ...)`.

Semantic dimensions must remain in tensor shape. Vector dtype lanes are physical layout lanes only. For example, norm stats components such as `sum` and `sum_sq` belong in a real tensor shape dimension, not in a vector lane. If an op needs dtype widening, such as `vec<f16, lanes> -> vec<fp32, 2, lanes>`, the extra `2` is a physical widening lane, not a semantic component.

## Rule Patterns

Vectorize rules usually follow one of two patterns.

Creation rules rewrite scalar ops into:

```text
PadForVectorize(input)
  -> Pack(...)
  -> VectorizedOp(...)
  -> Unpack(...)
  -> SliceForVectorize(...)
```

Propagation rules move existing vector layout across neighboring ops:

```text
Pack(Op(args))          -> Op(Pack(args))
Op(Unpack(arg), other)  -> Unpack(Op(arg, Pack(other)))
```

When writing a new rule:

- Generate candidates from ranked shapes and target lane bytes: `lane = targetLaneBytes / dtype.SizeInBytes`.
- Use `VectorizeUtility.PadForVectorize` and `SliceForVectorize` for non-divisible extents unless the op semantics forbid padding on that axis.
- Preserve metadata on replacement expressions when the old call carried user-visible metadata.
- Reject candidates by returning `null` or no candidate; do not silently emit an invalid local fallback.
- For broadcast ops, align operand axes to output rank before propagating lanes. Do not vectorize a broadcast dimension of size `1`.
- For shape-changing ops, adjust shape/axis parameters instead of assuming lanes keep the same axis number.
- Prefer existing neutral folds such as `FoldPackTranspose`, `FoldPackReshape`, `FoldPackBitcast`, and `UnpackToBitcast`. Do not add combined local folds when existing rules compose.

## Axis Semantics

Track vectorized axes in logical tensor coordinates, not only by vector lane order.

- Elementwise ops preserve vectorized axes.
- Transpose maps vectorized axes through the permutation.
- Reshape may propagate a lane only when the source and destination shape mapping keeps the lane as a trailing factor of the mapped region.
- Slice and pad require begin/end/padding alignment on vectorized axes.
- Concat on a vectorized concat axis requires every input extent on that axis to divide exactly by the lane.
- Gather can propagate only when the vectorized axis is not the gathered axis in the supported simple case.
- Where vectorizes the boolean condition with `VectorizeMask`; mask vectorization supports one axis.

For reductions and matmul, distinguish preserved lanes from consumed reduction lanes:

- A vectorized reduce axis is consumed by the reduction and must not remain in the output dtype.
- A non-reduced vectorized axis may remain in the output dtype.
- Matmul M/N lanes are output lanes. Matmul K lanes are reduction lanes and disappear from the output.
- If padding is applied to a reduction axis, use the correct neutral element and reject cases where padding would change semantics.

## Vectorized Op Contracts

Every vectorized op needs a precise contract for:

- input vectorized axes and lane order
- output vectorized axes and lane order
- output dtype, especially when input and output scalar dtypes differ
- padding values and whether padding is legal on reduction axes
- distributed type inference for packed tensor types
- evaluator behavior by devectorizing to logical tensors only for reference evaluation

Do not derive output vector dtype by blindly copying input lanes. Use op semantics:

- Cast changes lane count according to byte-size ratio and must validate exact divisibility.
- Reduce removes lanes for reduced axes.
- Matmul removes K lanes and preserves M/N lanes.
- RoPE keeps input/output dtype lanes, while sin/cos can be separately packed and cast to fp32.
- Norm-like stats must produce fp32 stats regardless of input precision; semantic stats components stay in shape.

## AutoPacking

AutoPacking is not generic vector propagation. It rewrites a legal vectorized or high-level op into a target packed ABI.

Current examples:

- `PackMatMulByN` rewrites a suitable N-vectorized matmul into `PackedMatMul`.
- `PackQKVParallelLinearByN` packs Q/K/V weights and biases into `PackedQKVParallelLinear`.
- `PackMatMulGluByN` packs gate/up weights and biases into `PackedMatMulGlu`.

AutoPacking rules should:

- Check exact divisibility for every packed weight/bias/output axis.
- Keep the external logical result shape by adding required `Unpack` calls around packed ABI outputs.
- Reject unsupported quantization, scaling, dtype, or vectorized-input cases explicitly.
- Encode target layout in the packed op contract and evaluator, not in downstream codegen guesses.
- Avoid requiring an outer `Unpack` pattern just to match; match the semantic op or vectorized op directly when possible.

Packed weight layouts must be documented by their evaluator/type inference. For packed matmul-like ops, use the packed vector dtype lanes to describe physical `Nr` and inner vector lanes, while logical N/K/M semantics remain in tensor shape and op attributes.

## Distributed Types

Pack and Unpack are part of distributed type inference:

- Packing a split axis is legal only when the local split and lane are compatible.
- Packing a split axis divides split granularity by the lane when granularity exists.
- Unpacking a split axis multiplies split granularity by the lane.
- Dynamic or non-divisible cases must return `InvalidType` unless the op has a well-defined symbolic contract.

Vectorized op type inference must describe the local shard type only. Do not add placement-wide replication or synchronization math to an op evaluator. If a vectorized op creates a partial result, represent that through `DistributedType.Partial` or explicit boxing requirements so AutoDistributed can insert legal reshards.

## Cost

Vectorized and packed op evaluators report local shard cost:

- load bytes for local inputs
- store bytes for local outputs
- local compute cycles or target-op cost query
- only semantic communication performed by the op itself

Hardware-sensitive cost belongs behind `ITargetOpCostModel`. Derive vector behavior from `VectorType`/`MaskVectorType`; do not add a separate "is vectorized" side channel.

## Lowering And Codegen

TIR selection lowers IR `Pack`/`Unpack` to `TIR.NTT.Pack`/`TIR.NTT.Unpack` kernels. NTT/PyNTT codegen expects the buffer element dtype and shape to already reflect vectorization.

Codegen rules:

- Validate pack/unpack shape and lane-prefix relationships at codegen boundaries.
- Do not reinterpret vectorized axes differently from IR/TIR metadata.
- Do not use scalar shape dimensions to recover semantic lanes hidden in dtype.
- Keep PyNTT templates aligned with manifest dtype/shape metadata; template-only changes should be rerenderable from the manifest.

## Checklist For New Vectorized Or Packed Ops

Before implementing:

- Define which dimensions are semantic shape axes and which are physical vector lanes.
- Define vectorized input axes, output axes, lane order, dtype widening/narrowing, and padding policy.
- Decide whether the op is a vectorized semantic op, an AutoPacking ABI op, or both.
- Add type inference, evaluator, cost evaluator, and distributed type inference together.
- Add candidate generation and propagation rules only where existing generic rules cannot express the transformation.
- Add TIR selection and NTT/PyNTT codegen support using the selected local shard type.
- Add focused tests for type inference, evaluator equivalence, propagation/folding, distributed legality, and codegen smoke where applicable.

Before finishing:

- Inspect IR dumps around `AutoVectorizePass`, `AutoPackingPass`, and `AutoDistributedPass`.
- Check that semantic dimensions are still in tensor shape.
- Check that reduction lanes are not accidentally preserved in output dtype.
- Check that packed ABI outputs are unpacked back to the expected public logical shape.
- Run focused unit tests before qwen or broad importer tests.
