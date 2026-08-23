// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Materializes selected experts' gate/up activations.
/// </summary>
public sealed partial class SparseExpertsGateUp : NTTKernelOp
{
    public static readonly ParameterInfo Q = new(typeof(SparseExpertsGateUp), 0, "q", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo RouterExpertIds = new(typeof(SparseExpertsGateUp), 1, "router_expert_ids", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo MoeExpertGateInputScale = new(typeof(SparseExpertsGateUp), 2, "moe_expert_gate_input_scale", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo MoeExpertGateProjW = new(typeof(SparseExpertsGateUp), 3, "moe_expert_gate_proj_w", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo MoeExpertGateProjScale = new(typeof(SparseExpertsGateUp), 4, "moe_expert_gate_proj_scale", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo MoeExpertUpInputScale = new(typeof(SparseExpertsGateUp), 5, "moe_expert_up_input_scale", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo MoeExpertUpProjW = new(typeof(SparseExpertsGateUp), 6, "moe_expert_up_proj_w", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo MoeExpertUpProjScale = new(typeof(SparseExpertsGateUp), 7, "moe_expert_up_proj_scale", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo Output = new(typeof(SparseExpertsGateUp), 8, "output", memoryEffect: MemoryEffect.Write);

    public long HiddenSize { get; }

    public long MoEIntermediateSize { get; }

    public long NumExpert { get; }

    public long NumTopK { get; }

    public long ChunkSize { get; }

    public override string DisplayProperty() =>
        $"HiddenSize: {HiddenSize}, MoEIntermediateSize: {MoEIntermediateSize}, NumExpert: {NumExpert}, NumTopK: {NumTopK}, ChunkSize: {ChunkSize}";
}

/// <summary>
/// Projects selected expert activations to their weighted hidden-state partials.
/// </summary>
public sealed partial class SparseExpertsDown : NTTKernelOp
{
    public static readonly ParameterInfo Activations = new(typeof(SparseExpertsDown), 0, "activations", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo RouterExpertIds = new(typeof(SparseExpertsDown), 1, "router_expert_ids", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo RouterExpertWeights = new(typeof(SparseExpertsDown), 2, "router_expert_weights", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo MoeExpertDownInputScale = new(typeof(SparseExpertsDown), 3, "moe_expert_down_input_scale", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo MoeExpertDownProjW = new(typeof(SparseExpertsDown), 4, "moe_expert_down_proj_w", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo MoeExpertDownProjScale = new(typeof(SparseExpertsDown), 5, "moe_expert_down_proj_scale", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo Output = new(typeof(SparseExpertsDown), 6, "output", memoryEffect: MemoryEffect.ReductionWrite);

    public long HiddenSize { get; }

    public long MoEIntermediateSize { get; }

    public long NumExpert { get; }

    public long NumTopK { get; }

    public long ChunkSize { get; }

    public override string DisplayProperty() =>
        $"HiddenSize: {HiddenSize}, MoEIntermediateSize: {MoEIntermediateSize}, NumExpert: {NumExpert}, NumTopK: {NumTopK}, ChunkSize: {ChunkSize}";
}
