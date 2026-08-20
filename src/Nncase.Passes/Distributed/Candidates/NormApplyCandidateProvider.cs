// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;

namespace Nncase.Passes.Distributed;

internal sealed class NormApplyCandidateProvider : DistributedCandidateProvider<NormApply>
{
    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        NormApply target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        if (context.AvailableInputTypes.Count != 4 ||
            !TryGetSourceTensorType(context, NormApply.Input.Index, out var outputTensorType))
        {
            return defaultReturnTypes;
        }

        return defaultReturnTypes
            .Concat(context.AvailableInputTypes[NormApply.Input.Index]
                .OfType<DistributedType>()
                .Where(input => !HasPartial(input) && input.TensorType == outputTensorType))
            .Distinct()
            .ToArray();
    }

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
            || context.AvailableInputTypes.Count != 4
            || !TryGetSourceTensorType(context, NormApply.Input.Index, out var inputTensorType)
            || inputTensorType != output.TensorType
            || !TryGetSourceTensorType(context, NormApply.Scale.Index, out var scaleTensorType)
            || !TryGetSourceTensorType(context, NormApply.Bias.Index, out var biasTensorType))
        {
            return false;
        }

        var normalizedAxis = NormalizeAxis(target.Axis, inputShape.Rank);
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

        var parameterRank = inputShape.Rank - normalizedAxis;
        var parameterPolicies = new SBP[parameterRank];
        for (int i = 0; i < parameterRank; i++)
        {
            parameterPolicies[i] = output.AxisPolicies[normalizedAxis + i];
        }

        tuples = statsTensorTypes
            .Select(statsTensorType =>
            {
                var stats = new DistributedType(statsTensorType, statsPolicies, output.Placement);
                var scale = new DistributedType(scaleTensorType, parameterPolicies, output.Placement);
                var bias = new DistributedType(biasTensorType, parameterPolicies, output.Placement);
                return (Stats: stats, Scale: scale, Bias: bias);
            })
            .Where(arguments =>
                NormApplyEvaluator.InferType(target, output, arguments.Stats, arguments.Scale, arguments.Bias) == output)
            .Select(arguments => new DistributedCandidateTuple(
                [output, arguments.Stats, arguments.Scale, arguments.Bias],
                "norm-apply-preserve-input-sbp"))
            .ToArray();
        return true;
    }

    private static bool TryGetSourceTensorType(
        DistributedCandidateContext context,
        int index,
        out TensorType tensorType)
    {
        tensorType = context.SourceCall.Arguments[index].CheckedType switch
        {
            TensorType value => value,
            DistributedType value => value.TensorType,
            _ => null!,
        };
        return tensorType is not null;
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
