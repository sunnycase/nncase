// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR;
using Nncase.Schedule;

namespace Nncase.Targets;

/// <summary>
/// Target-neutral contiguous storage model used by native NTT targets.
/// </summary>
public sealed class DefaultTargetStorageEncodingModel : ITargetStorageEncodingModelProvider
{
    public IReadOnlyList<TargetStorageEncodingCandidate> GetCandidates(TargetStorageEncodingModelContext context)
        =>
        [
            new(
                TargetStorageEncodingIds.Linear,
                context.Solver.MakeIntConst(1),
                context.LogicalBytes,
                GetNaturalAlignment(context.DataType),
                context.Solver.MakeIntConst(0),
                ImmutableArray<TargetStorageEncodingParameter>.Empty),
        ];

    internal static int GetNaturalAlignment(DataType dataType)
    {
        var bytes = Math.Max(1, dataType.SizeInBytes);
        return checked((int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)bytes));
    }

    internal static DataType GetScalarDataType(DataType dataType)
        => dataType is VectorType vector ? GetScalarDataType(vector.ElemType) : dataType;
}

/// <summary>
/// Triton physical shared-memory encodings. The IDs are compiler/backend ABI:
/// AutoTiling selects them and PyNTT lowers them without re-inferring layout
/// from the consuming operation.
/// </summary>
public sealed class TritonTargetStorageEncodingModel : ITargetStorageEncodingModelProvider
{
    public static readonly TargetStorageEncodingId SwizzledShared = new("triton.shared.swizzled");

    public static readonly TargetStorageEncodingId NvidiaMmaShared = new("triton.nvidia.mma-shared");

    public IReadOnlyList<TargetStorageEncodingCandidate> GetCandidates(TargetStorageEncodingModelContext context)
    {
        if (context.Machine.GetMemoryResource(context.MemorySpace).Kind != TargetMemorySpaceKind.Shared)
        {
            return new DefaultTargetStorageEncodingModel().GetCandidates(context);
        }

        var alignment = Math.Max(16, DefaultTargetStorageEncodingModel.GetNaturalAlignment(context.DataType));

        // Ordinary accesses may pay for an incompatible shared layout. Keep
        // the specialized encoding out of unrelated buffers while microkernel
        // requirements select it when required. One cycle is a deterministic
        // tie breaker, not a material data-movement cost.
        var nvidiaMmaPreferencePenalty = context.Solver.MakeIntConst(1);
        var candidates = new List<TargetStorageEncodingCandidate>
        {
            new(
                NvidiaMmaShared,
                context.Solver.MakeIntConst(SupportsNvidiaMmaShared(context.DataType) ? 1 : 0),
                context.LogicalBytes,
                alignment,
                nvidiaMmaPreferencePenalty,
                ImmutableArray<TargetStorageEncodingParameter>.Empty),
            new(
                SwizzledShared,
                context.Solver.MakeIntConst(1),
                context.LogicalBytes,
                alignment,
                context.Solver.MakeIntConst(0),
                ImmutableArray<TargetStorageEncodingParameter>.Empty),
        };
        return candidates;
    }

    private static bool SupportsNvidiaMmaShared(DataType dataType)
    {
        var scalar = DefaultTargetStorageEncodingModel.GetScalarDataType(dataType);
        return scalar == DataTypes.Float16
            || scalar == DataTypes.BFloat16
            || scalar == DataTypes.Float32
            || scalar == DataTypes.Float8E4M3
            || scalar == DataTypes.Float8E5M2
            || scalar == DataTypes.Int8;
    }
}
