// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

/// <summary>
/// Reduces distributed normalization statistics, applies normalization while
/// loading the activation, and computes fused NVFP4 gate/up projections.
/// </summary>
public sealed partial class GatherReduceNormApplyNVFP4MatMulGlu : NTTKernelOp
{
    public static readonly ParameterInfo PartialStats = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        0,
        "partial_stats",
        memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo Input = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        1,
        "input",
        memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo NormScale = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        2,
        "norm_scale",
        memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo NormBias = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        3,
        "norm_bias",
        memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateWeightPacked = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        4,
        "gate_weight_packed",
        memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpWeightPacked = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        5,
        "up_weight_packed",
        memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateWeightScale = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        6,
        "gate_weight_scale",
        memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpWeightScale = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        7,
        "up_weight_scale",
        memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateInputGlobalScale = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        8,
        "gate_input_global_scale",
        memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpInputGlobalScale = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        9,
        "up_input_global_scale",
        memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateWeightGlobalScale = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        10,
        "gate_weight_global_scale",
        memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpWeightGlobalScale = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        11,
        "up_weight_global_scale",
        memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(
        typeof(GatherReduceNormApplyNVFP4MatMulGlu),
        12,
        "output",
        memoryEffect: MemoryEffect.Write);

    public DistributedType InStatsType { get; }

    public DistributedType OutStatsType { get; }

    /// <summary>
    /// Gets the logical normalized activation type consumed by the fused
    /// projection after gathering the physical input shards.
    /// </summary>
    public DistributedType NormalizedInputType { get; }

    public int Axis { get; }

    public float Epsilon { get; }

    public bool UseMean { get; }

    public bool HasBias { get; }

    public GluType GluType { get; }

    public long GroupSize { get; }

    public override string DisplayProperty() =>
        $"Axis: {Axis}, Epsilon: {Epsilon}, UseMean: {UseMean}, HasBias: {HasBias}, " +
        $"GluType: {GluType}, GroupSize: {GroupSize}";
}
