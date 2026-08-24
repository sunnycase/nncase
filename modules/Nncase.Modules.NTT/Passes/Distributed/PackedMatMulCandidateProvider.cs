// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator;
using Nncase.Evaluator.IR.NTT;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;
using ScaledMatMulEvaluator = Nncase.Evaluator.Math.ScaledMatMulEvaluator;

namespace Nncase.Passes.Distributed;

internal static class PackedMatMulDistributedCandidates
{
    public static IEnumerable<PackedMatMulDistributedCandidate> Enumerate(
        DistributedCandidateContext context,
        PackedMatMul target)
    {
        if (context.AvailableInputTypes.Count != 4)
        {
            yield break;
        }

        var hasAddend = context.AvailableInputTypes[PackedMatMul.Addend.Index]
            .Any(type => type is not NoneType);
        foreach (var lhs in context.AvailableInputTypes[PackedMatMul.Lhs.Index]
                     .OfType<DistributedType>()
                     .Where(type => type.Partial is null))
        {
            foreach (var rhs in context.AvailableInputTypes[PackedMatMul.Rhs.Index]
                         .OfType<DistributedType>()
                         .Where(type => type.Partial is null))
            {
                if (!TryAlignRhsReductionPolicy(target.RhsLayout, lhs, rhs, out var alignedRhs))
                {
                    continue;
                }

                foreach (var scale in context.AvailableInputTypes[PackedMatMul.Scale.Index])
                {
                    var outputType = PackedMatMulEvaluator.InferType(
                        target,
                        lhs,
                        alignedRhs,
                        scale,
                        NoneType.Default);
                    if (outputType is InvalidType ||
                        (hasAddend && outputType is DistributedType { Partial: not null }))
                    {
                        continue;
                    }

                    var addend = hasAddend ? outputType : NoneType.Default;
                    var finalOutputType = PackedMatMulEvaluator.InferType(
                        target,
                        lhs,
                        alignedRhs,
                        scale,
                        addend);
                    if (finalOutputType is not InvalidType)
                    {
                        yield return new PackedMatMulDistributedCandidate(
                            lhs,
                            alignedRhs,
                            scale,
                            addend,
                            finalOutputType);
                    }
                }
            }
        }
    }

    internal static bool TryAlignRhsReductionPolicy(
        PackedMatMulRhsLayout rhsLayout,
        DistributedType lhs,
        DistributedType rhs,
        out DistributedType alignedRhs)
    {
        alignedRhs = null!;
        if (lhs.Placement != rhs.Placement ||
            rhs.TensorType.DType is not VectorType rhsVectorType ||
            !PackedMatMulEvaluator.TryGetLayoutInfo(
                rhsLayout,
                rhsVectorType,
                rhs.TensorType.Shape.Rank,
                out var rhsUnpackAxes,
                out _,
                out var transposeB,
                out _) ||
            TypeInference.UnpackType(rhs, rhsUnpackAxes) is not DistributedType logicalRhs)
        {
            return false;
        }

        var dimInfo = VectorizedMatMul.GetDimInfo(
            false,
            transposeB,
            lhs.TensorType.Shape.Rank,
            logicalRhs.TensorType.Shape.Rank);
        var logicalPolicies = logicalRhs.AxisPolicies.ToArray();
        logicalPolicies[dimInfo.Rk] = lhs.AxisPolicies[dimInfo.Lk];
        if (!DistributedUtility.IsDistributable(
                logicalRhs.TensorType,
                logicalPolicies,
                lhs.Placement))
        {
            return false;
        }

        var alignedLogicalRhs = new DistributedType(
            logicalRhs.TensorType,
            logicalPolicies,
            lhs.Placement);
        if (TypeInference.PackType(
                alignedLogicalRhs,
                rhsVectorType.Lanes.ToArray(),
                rhsUnpackAxes) is not DistributedType packedRhs ||
            packedRhs.TensorType != rhs.TensorType)
        {
            return false;
        }

        alignedRhs = packedRhs;
        return true;
    }
}

/// <summary>
/// Propagates the packed reduction layout for scaled low-precision matmul while
/// keeping both scalar scales replicated on the matrix placement.
/// </summary>
internal sealed class PackedScaledMatMulCandidateProvider :
    DistributedCandidateProvider<PackedScaledMatMul>
{
    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedScaledMatMul target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Concat(Enumerate(context, target).Select(candidate => candidate.OutputType))
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedScaledMatMul target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Enumerate(context, target)
            .Where(candidate => candidate.OutputType == returnType)
            .Select(candidate => new DistributedCandidateTuple(
                [candidate.Lhs, candidate.Rhs, candidate.LhsScale, candidate.RhsScale],
                "packed-scaled-matmul-reduction-sbp"))
            .Distinct()
            .ToArray();
        return true;
    }

    private static IEnumerable<PackedScaledMatMulDistributedCandidate> Enumerate(
        DistributedCandidateContext context,
        PackedScaledMatMul target)
    {
        if (context.AvailableInputTypes.Count != 4)
        {
            yield break;
        }

        foreach (var lhs in context.AvailableInputTypes[PackedScaledMatMul.Lhs.Index]
                     .OfType<DistributedType>()
                     .Where(type => type.Partial is null))
        {
            foreach (var rhs in context.AvailableInputTypes[PackedScaledMatMul.Rhs.Index]
                         .OfType<DistributedType>()
                         .Where(type => type.Partial is null))
            {
                if (!PackedMatMulDistributedCandidates.TryAlignRhsReductionPolicy(
                        target.RhsLayout,
                        lhs,
                        rhs,
                        out var alignedRhs))
                {
                    continue;
                }

                foreach (var lhsScale in context.AvailableInputTypes[PackedScaledMatMul.LhsScale.Index]
                             .Where(type => IsReplicatedScale(type, lhs.Placement)))
                {
                    foreach (var rhsScale in context.AvailableInputTypes[PackedScaledMatMul.RhsScale.Index]
                                 .Where(type => IsReplicatedScale(type, lhs.Placement)))
                    {
                        var outputType = PackedScaledMatMulEvaluator.InferType(
                            target,
                            lhs,
                            alignedRhs,
                            lhsScale,
                            rhsScale);
                        if (outputType is not InvalidType)
                        {
                            yield return new(
                                lhs,
                                alignedRhs,
                                lhsScale,
                                rhsScale,
                                outputType);
                        }
                    }
                }
            }
        }
    }

    private static bool IsReplicatedScale(IRType type, Placement placement) => type switch
    {
        TensorType => ScaledMatMulEvaluator.IsScaleType(type),
        DistributedType distributed =>
            distributed.Placement == placement &&
            distributed.Partial is null &&
            distributed.AxisPolicies.All(policy => policy is SBPBroadCast) &&
            ScaledMatMulEvaluator.IsScaleType(distributed),
        _ => false,
    };
}

/// <summary>
/// Propagates the packed matrix policies for block-scaled FP8 matmul. The scale
/// grid remains replicated because matrix shards may start inside a 128-wide
/// quantization block and therefore cannot be represented by a non-overlapping
/// split of the scale tensor.
/// </summary>
internal sealed class PackedBlockScaledMatMulCandidateProvider :
    DistributedCandidateProvider<PackedBlockScaledMatMul>
{
    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedBlockScaledMatMul target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Concat(Enumerate(context, target).Select(candidate => candidate.OutputType))
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedBlockScaledMatMul target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Enumerate(context, target)
            .Where(candidate => candidate.OutputType == returnType)
            .Select(candidate => new DistributedCandidateTuple(
                [candidate.Lhs, candidate.Rhs, candidate.RhsScale],
                "packed-block-scaled-matmul-reduction-sbp"))
            .Distinct()
            .ToArray();
        return true;
    }

    private static IEnumerable<PackedBlockScaledMatMulDistributedCandidate> Enumerate(
        DistributedCandidateContext context,
        PackedBlockScaledMatMul target)
    {
        if (context.AvailableInputTypes.Count != 3)
        {
            yield break;
        }

        foreach (var lhs in context.AvailableInputTypes[PackedBlockScaledMatMul.Lhs.Index]
                     .OfType<DistributedType>()
                     .Where(type => type.Partial is null))
        {
            foreach (var rhs in context.AvailableInputTypes[PackedBlockScaledMatMul.Rhs.Index]
                         .OfType<DistributedType>()
                         .Where(type => type.Partial is null))
            {
                if (!PackedMatMulDistributedCandidates.TryAlignRhsReductionPolicy(
                        target.RhsLayout,
                        lhs,
                        rhs,
                        out var alignedRhs))
                {
                    continue;
                }

                foreach (var rhsScale in context.AvailableInputTypes[PackedBlockScaledMatMul.RhsScale.Index]
                             .Where(type => IsReplicatedScale(type, lhs.Placement)))
                {
                    var outputType = PackedBlockScaledMatMulEvaluator.InferType(
                        target,
                        lhs,
                        alignedRhs,
                        rhsScale);
                    if (outputType is not InvalidType)
                    {
                        yield return new(lhs, alignedRhs, rhsScale, outputType);
                    }
                }
            }
        }
    }

    private static bool IsReplicatedScale(IRType type, Placement placement)
    {
        var scale = ScaledMatMulEvaluator.GetTensorType(type);
        if (scale?.Shape is not RankedShape { Rank: 2 })
        {
            return false;
        }

        return type switch
        {
            TensorType => true,
            DistributedType distributed =>
                distributed.Placement == placement &&
                distributed.Partial is null &&
                distributed.AxisPolicies.All(policy => policy is SBPBroadCast),
            _ => false,
        };
    }
}

/// <summary>
/// Propagates an exact lhs reduction-axis layout to the packed RHS reduction
/// axis, including ordered multi-stage split policies.
/// </summary>
internal sealed class PackedMatMulCandidateProvider :
    DistributedCandidateProvider<PackedMatMul>
{
    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedMatMul target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Concat(PackedMatMulDistributedCandidates.Enumerate(context, target).Select(candidate => candidate.OutputType))
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedMatMul target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = PackedMatMulDistributedCandidates.Enumerate(context, target)
            .Where(candidate => candidate.OutputType == returnType)
            .Select(candidate => new DistributedCandidateTuple(
                [candidate.Lhs, candidate.Rhs, candidate.Scale, candidate.Addend],
                "packed-matmul-reduction-sbp"))
            .Distinct()
            .ToArray();
        return true;
    }
}

/// <summary>
/// Propagates packed-matmul layouts while preserving the coupled value and
/// normalization-statistics return types.
/// </summary>
internal sealed class PackedMatMulNormStatsCandidateProvider :
    DistributedCandidateProvider<PackedMatMulNormStats>
{
    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedMatMulNormStats target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Concat(EnumerateCandidates(context, target).Select(candidate => candidate.OutputType))
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedMatMulNormStats target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = EnumerateCandidates(context, target)
            .Where(candidate => candidate.OutputType == returnType)
            .Select(candidate => new DistributedCandidateTuple(
                [candidate.Lhs, candidate.Rhs, candidate.Scale, candidate.Addend],
                "packed-matmul-norm-stats-reduction-sbp"))
            .Distinct()
            .ToArray();
        return true;
    }

    private static IEnumerable<PackedMatMulDistributedCandidate> EnumerateCandidates(
        DistributedCandidateContext context,
        PackedMatMulNormStats target)
    {
        var packedTarget = new PackedMatMul(
            target.OutputDataType,
            false,
            target.RhsLayout);
        foreach (var candidate in PackedMatMulDistributedCandidates.Enumerate(context, packedTarget))
        {
            var outputType = PackedMatMulNormStatsEvaluator.InferType(
                target,
                candidate.Lhs,
                candidate.Rhs,
                candidate.Scale,
                candidate.Addend);
            if (outputType is not InvalidType)
            {
                yield return candidate with { OutputType = outputType };
            }
        }
    }
}

internal sealed record PackedMatMulDistributedCandidate(
    IRType Lhs,
    IRType Rhs,
    IRType Scale,
    IRType Addend,
    IRType OutputType);

internal sealed record PackedScaledMatMulDistributedCandidate(
    IRType Lhs,
    IRType Rhs,
    IRType LhsScale,
    IRType RhsScale,
    IRType OutputType);

internal sealed record PackedBlockScaledMatMulDistributedCandidate(
    IRType Lhs,
    IRType Rhs,
    IRType RhsScale,
    IRType OutputType);
