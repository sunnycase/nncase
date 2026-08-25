// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

/// <summary>
/// Reduces partial Q/K/V projections and applies Q/K normalization, RoPE, and
/// KV-cache updates without materializing the combined projections.
/// </summary>
public sealed partial class GatherReduceQKVRoPEWithCache : NTTKernelOp
{
    public static readonly ParameterInfo Q = new(typeof(GatherReduceQKVRoPEWithCache), 0, "q", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo K = new(typeof(GatherReduceQKVRoPEWithCache), 1, "k", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo V = new(typeof(GatherReduceQKVRoPEWithCache), 2, "v", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo QScale = new(typeof(GatherReduceQKVRoPEWithCache), 3, "q_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KScale = new(typeof(GatherReduceQKVRoPEWithCache), 4, "k_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo QBias = new(typeof(GatherReduceQKVRoPEWithCache), 5, "q_bias", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KBias = new(typeof(GatherReduceQKVRoPEWithCache), 6, "k_bias", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Cos = new(typeof(GatherReduceQKVRoPEWithCache), 7, "cos", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Sin = new(typeof(GatherReduceQKVRoPEWithCache), 8, "sin", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KVCaches = new(
        typeof(GatherReduceQKVRoPEWithCache),
        9,
        "kv_caches",
        memoryEffect: MemoryEffect.ChipWrite.PartitionedByArgument(10));

    public static readonly ParameterInfo LayerId = new(typeof(GatherReduceQKVRoPEWithCache), 10, "layer_id", IR.TypePatternUtility.IsDimensionType(), memoryEffect: MemoryEffect.None);

    public static readonly ParameterInfo QOutput = new(typeof(GatherReduceQKVRoPEWithCache), 11, "q_output", memoryEffect: MemoryEffect.Write);

    public DistributedType QInType { get; }

    public DistributedType QLogicalType { get; }

    public DistributedType KInType { get; }

    public DistributedType KLogicalType { get; }

    public DistributedType VInType { get; }

    public DistributedType VLogicalType { get; }

    public IRArray<Dimension> QShape { get; }

    public IRArray<Dimension> QStrides { get; }

    public IRArray<Dimension> KShape { get; }

    public IRArray<Dimension> KStrides { get; }

    public IRArray<Dimension> VShape { get; }

    public IRArray<Dimension> VStrides { get; }

    public int QAxis { get; }

    public float QEpsilon { get; }

    public bool QUseMean { get; }

    public int KAxis { get; }

    public float KEpsilon { get; }

    public bool KUseMean { get; }

    public IRArray<AttentionDimKind> QKVLayout { get; }

    public IRArray<AttentionDimKind> AttentionLayout { get; }

    public override string DisplayProperty() =>
        $"QAxis: {QAxis}, QEpsilon: {QEpsilon}, QUseMean: {QUseMean}, " +
        $"KAxis: {KAxis}, KEpsilon: {KEpsilon}, KUseMean: {KUseMean}, " +
        $"QKVLayout [{string.Join(',', QKVLayout)}], " +
        $"AttentionLayout [{string.Join(',', AttentionLayout)}]";
}
