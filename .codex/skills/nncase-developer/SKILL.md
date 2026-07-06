---
name: nncase-developer
description: nncase compiler development guidance for IR, importers, passes, AutoDistributed, vectorize/packing, cost model, NTT/PyNTT codegen/runtime, and tests. Use when modifying nncase compiler internals, target backends, sharding/search/cost logic, vectorized or packed ops, PyNTT templates, or debugging model integration and performance regressions.
---

# nncase Developer

## Overview

Use this skill for nncase compiler work that crosses IR, passes, evaluator, target codegen, runtime, and tests. Read the repository `AGENTS.md` first, then load only the reference files relevant to the change.

## Core Rules

- Fix the owning layer. Do not patch around a cross-layer contract issue at a local call site.
- Preserve the compiler pipeline. Prefer existing passes, evaluator interfaces, target options, dump scopes, and test runners over ad hoc side flows.
- Keep generated artifacts and local outputs out of source changes unless the user explicitly asks for them.
- Validate narrowly first, then broaden to the smallest end-to-end test that exercises the changed contract.

## AutoDistributed And Cost Model

Read [references/auto-dist-cost-model.md](references/auto-dist-cost-model.md) before changing:

- `src/Nncase.Passes/Distributed/AutoDistributed.cs`
- `src/Nncase.Core/CostModel/`
- target cost models such as `TritonTargetOpCostModel`
- op `Visit(ICostEvaluateContext, ...)` implementations
- boxing, sharding, placement, hierarchy, or reshard search behavior

The most important boundary is: op cost evaluators report local shard cost only; target/hierarchy cost models aggregate that local cost over the selected placement and hardware model.

## Vectorize And Packing

Read [references/vectorize-packing.md](references/vectorize-packing.md) before changing:

- `Pack`, `Unpack`, `VectorType`, `MaskVectorType`, or vector lane type inference
- NTT/PyNTT vectorized ops, propagation rules, or AutoPacking rules
- packed matmul/QKV/GLU layouts, vectorized reduce/norm/rope behavior, or vectorized cost/codegen

The most important boundary is: tensor shape carries semantic dimensions, while vector dtype lanes carry physical packed lanes only. Do not encode semantic fields such as stats components or tuple outputs in vector lanes.
