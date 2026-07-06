# AutoDistributed and Cost Model Principles

Use these rules when changing AutoDistributed search, cost evaluators, target cost models, boxing, or sharding propagation.

## Mental Model

AutoDistributed chooses a legal distributed program. The cost model ranks legal candidates. Codegen and runtime execute the chosen local shards. Keep these responsibilities separate.

- AutoDistributed owns candidate generation, legality, reshard/boxing insertion, solver construction, and diagnostics.
- Op type inference owns sharding compatibility for one op and must reject unsupported distributed types with actionable reasons.
- Op cost evaluators own local shard cost for one op invocation.
- Target/hierarchy cost models own hardware aggregation, bandwidth, synchronization, active block count, and target-specific latency.
- Codegen owns lowering the selected distributed IR/TIR to target code. It must not reinterpret the sharding decision.

## Layering Rules

- Do not put global replication, mesh-wide active block count, or placement-wide bandwidth math in an op cost evaluator.
- Do not make an evaluator inspect a concrete backend unless the query is explicitly target-provided through `ITargetOpCostModel`.
- Do not hard-code a model, segment, operator name, or target behavior in AutoDistributed to force a better search result.
- Do not hide illegal candidates by silent fallback. Reject them through type inference or candidate diagnostics.
- Do not introduce duplicate sharding helpers when `DistributedUtility`, `DistributedType`, `Placement`, or existing boxing utilities already describe the contract.

## Local Op Cost Evaluators

An op cost evaluator should describe the work performed by one local shard.

- Use local divided tensor types for distributed inputs and outputs.
- Count local reads, writes, and local compute only.
- Use max shape for cost when shape is dynamic, matching current nncase cost model policy.
- Express compute and memory as distinct cost factors. Do not pre-collapse them into latency.
- Query the target op cost model for hardware-sensitive local kernels such as unary, binary, elementwise, and matmul.
- If a vectorized type is present, derive vector behavior from the dtype/vector type instead of adding an extra boolean side channel.
- Keep communication factors only when the op itself semantically performs that local communication. Placement-wide aggregation belongs above the evaluator.

Example boundary:

- Correct: `VectorizedLayerNorm` returns local shard load/store bytes and local cycles for the divided tensor.
- Incorrect: `VectorizedLayerNorm` multiplies CPU cycles by the number of broadcast replicas in the placement.

## Target And Hierarchy Cost Models

A target cost model converts local cost factors into latency for a placement and hardware capability.

- Treat local compute and local memory as per-block or per-CTA quantities.
- Use `TargetCostAggregationContext` or the equivalent context object for active blocks and hierarchy-level aggregation.
- Model block-local bandwidth separately from chip-global bandwidth.
- Convert memory cost to latency through bandwidth. Convert compute cost to latency through the target's local compute throughput.
- Combine compute and memory latency with the target's overlap rule, usually `max(compute, memory)`, then add synchronization and communication latency.
- Model synchronization in the target cost model, not in individual unrelated op evaluators.
- Keep hardware details such as MMA/WGMMA tile constraints, padding penalties, bandwidth, SM count, and sync cycles behind target capability structures.

## AutoDistributed Search

AutoDistributed should search legal distributed programs, not repair invalid local decisions after the fact.

- Generate candidate policies from tensor type, placement hierarchy, rank, dynamic shape range, and target options.
- Represent split, broadcast, and partial semantics in distributed types and boxing/reshard paths.
- Prefer generic multi-hop reshard planning over special-casing one source/target pair.
- Keep `B`, `S`, and `P` semantics explicit. Do not encode them as target-specific shortcuts.
- For dynamic shape, use the existing dimension/range expressions and max shape for cost, but preserve real shape expressions for TIR/codegen when the pipeline supports them.
- Dump selected candidates, rejected candidates, costs, and latency breakdowns when debugging search. Diagnostics should explain why a candidate is accepted, rejected, or dominated.

## Cost Dump Expectations

Cost dumps should make the layer boundary visible.

- Show local cost factors before latency aggregation.
- Show active block count and effective global memory pressure in target cost breakdowns.
- Show explicit chip-global memory separately from block-local bytes that become globally visible through active blocks.
- Show synchronization and communication as separate latency terms.
- Avoid diagnostic fields that imply op evaluators are responsible for placement-wide replication.

## Validation Checklist

Before finishing a change:

- Build the C# compiler with `dotnet build -c Debug --no-restore`.
- Run focused unit tests for changed utilities or evaluators.
- For AutoDistributed or PyNTT search changes, run a small existing pytest case with `NNCASE_TEST_TARGETS=pyntt`.
- For LLM-path changes, run one-layer qwen3 before broader tests.
- Inspect the dumped IR and cost files. Verify the selected sharding follows from local evaluator cost plus target aggregation, not from an op-local global penalty.
- Check `git status --short` and keep generated outputs out of source changes unless explicitly requested.
