// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.Utilities;

namespace Nncase.Passes.Distributed;

internal sealed class NormApplyCandidateProvider : DistributedCandidateProvider<NormApply>
{
    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        NormApply target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (returnType is not DistributedType output
            || HasPartial(output)
            || output.TensorType.Shape is not RankedShape inputShape
            || context.AvailableInputTypes.Count != 4)
        {
            return false;
        }

        var normalizedAxis = NormalizeAxis(target.Axis, inputShape.Rank);
        var input = FindExact(context, NormApply.Input.Index, output);
        if (input is null)
        {
            return true;
        }

        var statsTensorTypes = GetCompatibleStatsTensorTypes(output.TensorType, target.Axis, target.UseMean);
        if (statsTensorTypes.Count == 0)
        {
            return false;
        }

        var statsPolicies = new SBP[inputShape.Rank + 1];
        statsPolicies[0] = SBP.B;
        for (int i = 0; i < inputShape.Rank; i++)
        {
            statsPolicies[i + 1] = i < normalizedAxis ? output.AxisPolicies[i] : SBP.B;
        }

        if (!TryFindDistributed(context, NormApply.Stats.Index, statsTensorTypes, output.Placement, statsPolicies, out var stats))
        {
            return true;
        }

        var parameterRank = inputShape.Rank - normalizedAxis;
        var parameterPolicies = new SBP[parameterRank];
        for (int i = 0; i < parameterRank; i++)
        {
            parameterPolicies[i] = output.AxisPolicies[normalizedAxis + i];
        }

        if (!TryFindDistributed(context, NormApply.Scale.Index, output.Placement, parameterPolicies, out var scale)
            || !TryFindDistributed(context, NormApply.Bias.Index, output.Placement, parameterPolicies, out var bias))
        {
            return true;
        }

        tuples =
        [
            new DistributedCandidateTuple(
                [input, stats, scale, bias],
                "norm-apply-output-sbp")
        ];
        return true;
    }

    private static IRType? FindExact(DistributedCandidateContext context, int index, IRType type)
        => context.AvailableInputTypes[index].FirstOrDefault(candidate => candidate == type);

    private static bool TryFindDistributed(
        DistributedCandidateContext context,
        int index,
        Placement placement,
        IReadOnlyList<SBP> policies,
        out IRType result)
    {
        result = context.AvailableInputTypes[index].FirstOrDefault(candidate =>
            candidate is DistributedType distributed
            && distributed.Placement == placement
            && !HasPartial(distributed)
            && SamePolicies(distributed.AxisPolicies, policies))!;
        return result is not null;
    }

    private static bool TryFindDistributed(
        DistributedCandidateContext context,
        int index,
        IReadOnlyList<TensorType> tensorTypes,
        Placement placement,
        IReadOnlyList<SBP> policies,
        out IRType result)
    {
        result = context.AvailableInputTypes[index].FirstOrDefault(candidate =>
            candidate is DistributedType distributed
            && tensorTypes.Any(tensorType => distributed.TensorType == tensorType)
            && distributed.Placement == placement
            && !HasPartial(distributed)
            && SamePolicies(distributed.AxisPolicies, policies))!;
        return result is not null;
    }

    private static bool SamePolicies(IReadOnlyList<SBP> lhs, IReadOnlyList<SBP> rhs)
    {
        if (lhs.Count != rhs.Count)
        {
            return false;
        }

        for (int i = 0; i < lhs.Count; i++)
        {
            if (!DistributedUtility.IsSamePolicy(lhs[i], rhs[i], checkGranularity: false))
            {
                return false;
            }
        }

        return true;
    }

    private static bool HasPartial(DistributedType distributedType)
        => distributedType.Partial is not null || distributedType.AxisPolicies.Any(policy => policy is SBPPartial);

    private static int NormalizeAxis(int axis, int rank)
    {
        var normalizedAxis = axis < 0 ? axis + rank : axis;
        if (normalizedAxis < 0 || normalizedAxis >= rank)
        {
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for rank {rank}.");
        }

        return normalizedAxis;
    }

    private static TensorType GetStatsTensorType(TensorType input, int axis, bool useMean)
    {
        if (!input.DType.IsFloat())
        {
            return TensorType.Invalid(DataTypes.Float32);
        }

        var statsDType = DataTypes.Float32;
        if (input.Shape.IsUnranked)
        {
            return TensorType.Unranked(statsDType);
        }

        if (input.Shape is not RankedShape shape || shape.Rank == 0)
        {
            return TensorType.Invalid(statsDType);
        }

        var normalizedAxis = NormalizeAxis(axis, shape.Rank);
        var statsShape = new Dimension[shape.Rank + 1];
        statsShape[0] = useMean ? 2 : 1;
        for (int i = 0; i < shape.Rank; i++)
        {
            statsShape[i + 1] = i < normalizedAxis ? shape[i] : 1;
        }

        return new TensorType(statsDType, new RankedShape(statsShape));
    }

    private static IReadOnlyList<TensorType> GetCompatibleStatsTensorTypes(TensorType input, int axis, bool useMean)
    {
        var stats = GetStatsTensorType(input, axis, useMean);
        return stats.Shape.IsInvalid ? Array.Empty<TensorType>() : [stats];
    }
}
