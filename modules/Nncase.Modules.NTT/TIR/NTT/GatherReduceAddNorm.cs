// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Materializes a partial tensor, adds a residual, and publishes normalization statistics.
/// </summary>
public sealed partial class GatherReduceAddNormStats : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(GatherReduceAddNormStats), 0, "input", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo Collective = new(typeof(GatherReduceAddNormStats), 1, "collective", memoryEffect: MemoryEffect.ReadWrite);

    public static readonly ParameterInfo Addend = new(typeof(GatherReduceAddNormStats), 2, "addend", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo ValueOutput = new(typeof(GatherReduceAddNormStats), 3, "value_output", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo StatsOutput = new(typeof(GatherReduceAddNormStats), 4, "stats_output", memoryEffect: MemoryEffect.Write);

    public DistributedType InType { get; }

    public DistributedType OutType { get; }

    public int Axis { get; }

    public bool UseMean { get; }

    public override string DisplayProperty() => $"Axis: {Axis}, UseMean: {UseMean}";
}

/// <summary>
/// Materializes a partial tensor, adds a residual, and applies normalization while
/// preserving the unnormalized value for the following residual connection.
/// </summary>
public sealed partial class GatherReduceAddNormApply : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(GatherReduceAddNormApply), 0, "input", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo Collective = new(typeof(GatherReduceAddNormApply), 1, "collective", memoryEffect: MemoryEffect.ReadWrite);

    public static readonly ParameterInfo Addend = new(typeof(GatherReduceAddNormApply), 2, "addend", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo ValueOutput = new(typeof(GatherReduceAddNormApply), 3, "value_output", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo StatsWorkspace = new(typeof(GatherReduceAddNormApply), 4, "stats_workspace", memoryEffect: MemoryEffect.ReadWrite);

    public static readonly ParameterInfo Scale = new(typeof(GatherReduceAddNormApply), 5, "scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Bias = new(typeof(GatherReduceAddNormApply), 6, "bias", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo NormOutput = new(typeof(GatherReduceAddNormApply), 7, "norm_output", memoryEffect: MemoryEffect.Write);

    public DistributedType InType { get; }

    public DistributedType OutType { get; }

    public int Axis { get; }

    public float Epsilon { get; }

    public bool UseMean { get; }

    public override string DisplayProperty() =>
        $"Axis: {Axis}, Epsilon: {Epsilon}, UseMean: {UseMean}";
}
