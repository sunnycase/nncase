// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Google.OrTools.ConstraintSolver;
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

    public static readonly TargetStorageEncodingId KMajorPackedN = new("triton.shared.k-major-packed-n");

    public IReadOnlyList<TargetStorageEncodingCandidate> GetCandidates(TargetStorageEncodingModelContext context)
    {
        if (context.Machine.GetMemoryResource(context.MemorySpace).Kind != TargetMemorySpaceKind.Shared)
        {
            return new DefaultTargetStorageEncodingModel().GetCandidates(context);
        }

        var alignment = Math.Max(16, DefaultTargetStorageEncodingModel.GetNaturalAlignment(context.DataType));
        var physicalBytes = RoundUpPowerOfTwo(context.LogicalBytes, context.Solver);

        // Ordinary accesses may pay for an incompatible shared layout. Keep
        // the specialized encoding out of unrelated buffers while microkernel
        // requirements select it when required. One cycle is a deterministic
        // tie breaker, not a material data-movement cost.
        var nvidiaMmaPreferencePenalty = context.Solver.MakeIntConst(1);
        var packedNPreferencePenalty = context.Solver.MakeIntConst(1);
        var candidates = new List<TargetStorageEncodingCandidate>
        {
            new(
                NvidiaMmaShared,
                context.Solver.MakeIntConst(SupportsNvidiaMmaShared(context.DataType) ? 1 : 0),
                physicalBytes,
                alignment,
                nvidiaMmaPreferencePenalty,
                ImmutableArray<TargetStorageEncodingParameter>.Empty),
        };
        if (CanRepresentKMajorPackedN(context))
        {
            candidates.Add(new(
                KMajorPackedN,
                context.Solver.MakeIntConst(1),
                physicalBytes,
                alignment,
                packedNPreferencePenalty,
                ImmutableArray<TargetStorageEncodingParameter>.Empty));
        }

        candidates.Add(new(
            SwizzledShared,
            context.Solver.MakeIntConst(1),
            physicalBytes,
            alignment,
            context.Solver.MakeIntConst(0),
            ImmutableArray<TargetStorageEncodingParameter>.Empty));
        return candidates;
    }

    private static IntExpr RoundUpPowerOfTwo(IntExpr logicalBytes, Solver solver)
    {
        var maximumBytes = logicalBytes.Var().Max();
        if (maximumBytes < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(logicalBytes),
                maximumBytes,
                "Triton shared storage cannot have a negative logical size.");
        }

        if (maximumBytes == 0)
        {
            return solver.MakeIntConst(0);
        }

        IntExpr physicalBytes = solver.MakeIntConst(0);
        long previousBytes = 0;
        for (long bytes = 1; bytes > 0 && previousBytes < maximumBytes;)
        {
            var exceedsPrevious = solver.MakeIsGreaterCstVar(logicalBytes, previousBytes);
            var fitsCurrent = 1 - solver.MakeIsGreaterCstVar(logicalBytes, bytes);
            physicalBytes += exceedsPrevious * fitsCurrent * bytes;
            previousBytes = bytes;
            bytes = bytes <= long.MaxValue / 2 ? bytes * 2 : 0;
        }

        if (previousBytes < maximumBytes)
        {
            throw new OverflowException(
                $"Triton shared storage size {maximumBytes} cannot be represented as a power of two.");
        }

        return physicalBytes;
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

    private static bool CanRepresentKMajorPackedN(TargetStorageEncodingModelContext context)
        => context.DataType is VectorType
            && context.LogicalShape.Length == 2
            && SupportsNvidiaMmaShared(context.DataType);
}
