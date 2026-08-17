// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Reduces partial normalization statistics across their placement and applies
/// the resulting broadcast statistics without materializing an intermediate buffer.
/// </summary>
public sealed partial class GatherReduceNormApply : NTTKernelOp
{
    public static readonly ParameterInfo PartialStats = new(
        typeof(GatherReduceNormApply),
        0,
        "partial_stats",
        ParameterKind.Input,
        MemoryEffect.ChipRead);

    public static readonly ParameterInfo Input = new(
        typeof(GatherReduceNormApply),
        1,
        "input",
        ParameterKind.Input,
        MemoryEffect.Read);

    public static readonly ParameterInfo Scale = new(
        typeof(GatherReduceNormApply),
        2,
        "scale",
        ParameterKind.Input,
        MemoryEffect.Read);

    public static readonly ParameterInfo Bias = new(
        typeof(GatherReduceNormApply),
        3,
        "bias",
        ParameterKind.Input,
        MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(
        typeof(GatherReduceNormApply),
        4,
        "output",
        ParameterKind.Input,
        MemoryEffect.Write);

    public DistributedType InStatsType { get; }

    public DistributedType OutStatsType { get; }

    public int Axis { get; }

    public float Epsilon { get; }

    public bool UseMean { get; }

    public override string DisplayProperty()
        => $"Axis: {Axis}, Epsilon: {Epsilon}, UseMean: {UseMean}";
}
