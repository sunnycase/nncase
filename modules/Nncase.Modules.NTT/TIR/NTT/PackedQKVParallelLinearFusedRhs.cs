// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Packed QKV projection whose block-local Q/K/V RHS shards are stored in one
/// canonical K-major tensor. Projection capacities define the fixed physical N
/// ranges; output shapes define the active tails.
/// </summary>
public sealed partial class PackedQKVParallelLinearFusedRhs : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(PackedQKVParallelLinearFusedRhs), 0, "input", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Weight = new(typeof(PackedQKVParallelLinearFusedRhs), 1, "weight", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo QBias = new(typeof(PackedQKVParallelLinearFusedRhs), 2, "q_bias", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KBias = new(typeof(PackedQKVParallelLinearFusedRhs), 3, "k_bias", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo VBias = new(typeof(PackedQKVParallelLinearFusedRhs), 4, "v_bias", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo QInputScale = new(typeof(PackedQKVParallelLinearFusedRhs), 5, "q_input_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KInputScale = new(typeof(PackedQKVParallelLinearFusedRhs), 6, "k_input_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo VInputScale = new(typeof(PackedQKVParallelLinearFusedRhs), 7, "v_input_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo QWeightScale = new(typeof(PackedQKVParallelLinearFusedRhs), 8, "q_weight_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KWeightScale = new(typeof(PackedQKVParallelLinearFusedRhs), 9, "k_weight_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo VWeightScale = new(typeof(PackedQKVParallelLinearFusedRhs), 10, "v_weight_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo QOutput = new(typeof(PackedQKVParallelLinearFusedRhs), 11, "q_output", memoryEffect: MemoryEffect.ReductionWrite);

    public static readonly ParameterInfo KOutput = new(typeof(PackedQKVParallelLinearFusedRhs), 12, "k_output", memoryEffect: MemoryEffect.ReductionWrite);

    public static readonly ParameterInfo VOutput = new(typeof(PackedQKVParallelLinearFusedRhs), 13, "v_output", memoryEffect: MemoryEffect.ReductionWrite);

    public long NumHeads { get; }

    public long NumKvHeads { get; }

    public IR.NTT.PackedMatMulRhsLayout RhsLayout { get; }

    /// <summary>
    /// Gets fixed scalar N capacities for Q, K, and V in the fused local RHS.
    /// </summary>
    public IRArray<long> ProjectionNCapacities { get; }

    public override string DisplayProperty() =>
        $"NumHeads: {NumHeads}, NumKvHeads: {NumKvHeads}, RhsLayout: {RhsLayout}, " +
        $"ProjectionNCapacities: [{string.Join(",", ProjectionNCapacities)}]";
}
