// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.Math;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Utilities;

namespace Nncase.Passes.Distributed;

/// <summary>
/// Propagates exact producer layouts through broadcast-compatible binary ops.
/// </summary>
internal sealed class BinaryCandidateProvider : DistributedCandidateProvider<Binary>
{
    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        Binary target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        if (context.SourceCall.CheckedType is not TensorType outputTensorType ||
            outputTensorType.Shape is not RankedShape ||
            context.AvailableInputTypes.Count != 2)
        {
            return defaultReturnTypes;
        }

        var results = new List<IRType>(defaultReturnTypes);
        foreach (var inputType in context.AvailableInputTypes
                     .SelectMany(types => types)
                     .OfType<DistributedType>())
        {
            if (TryLiftInputLayout(inputType, outputTensorType, out var outputType))
            {
                results.Add(outputType);
            }
        }

        return results.Distinct().ToArray();
    }

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        Binary target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = System.Array.Empty<DistributedCandidateTuple>();
        if (returnType is not DistributedType outputType ||
            outputType.Partial is not null ||
            outputType.AxisPolicies.Any(policy => policy is SBPPartial) ||
            context.AvailableInputTypes.Count != 2 ||
            !TryGetSourceTensorType(context, Binary.Lhs.Index, out var lhsTensorType) ||
            !TryGetSourceTensorType(context, Binary.Rhs.Index, out var rhsTensorType))
        {
            return false;
        }

        if (!TryProjectOutputLayout(outputType, lhsTensorType, out var lhsType) ||
            !TryProjectOutputLayout(outputType, rhsTensorType, out var rhsType) ||
            BinaryEvaluator.CheckSBP(target.BinaryOp, outputType.TensorType, lhsType, rhsType) != outputType)
        {
            return true;
        }

        tuples =
        [
            new DistributedCandidateTuple(
                [lhsType, rhsType],
                "binary-exact-output-sbp"),
        ];
        return true;
    }

    private static bool TryLiftInputLayout(
        DistributedType inputType,
        TensorType outputTensorType,
        out DistributedType outputType)
    {
        outputType = null!;
        if (inputType.Partial is not null ||
            inputType.AxisPolicies.Any(policy => policy is SBPPartial) ||
            inputType.TensorType.Shape is not RankedShape inputShape ||
            outputTensorType.Shape is not RankedShape outputShape ||
            inputShape.Rank > outputShape.Rank)
        {
            return false;
        }

        var outputPolicies = Enumerable.Repeat<SBP>(SBP.B, outputShape.Rank).ToArray();
        var rankPadding = outputShape.Rank - inputShape.Rank;
        for (var inputAxis = 0; inputAxis < inputShape.Rank; inputAxis++)
        {
            var outputAxis = rankPadding + inputAxis;
            if (inputShape[inputAxis] == outputShape[outputAxis])
            {
                outputPolicies[outputAxis] = inputType.AxisPolicies[inputAxis];
            }
            else if (!IsUnitDimension(inputShape[inputAxis]))
            {
                return false;
            }
        }

        if (!DistributedUtility.IsDistributable(outputTensorType, outputPolicies, inputType.Placement))
        {
            return false;
        }

        outputType = new DistributedType(outputTensorType, outputPolicies, inputType.Placement);
        return true;
    }

    private static bool TryProjectOutputLayout(
        DistributedType outputType,
        TensorType inputTensorType,
        out DistributedType inputType)
    {
        inputType = null!;
        if (outputType.TensorType.Shape is not RankedShape outputShape ||
            inputTensorType.Shape is not RankedShape inputShape ||
            inputShape.Rank > outputShape.Rank)
        {
            return false;
        }

        var inputPolicies = new SBP[inputShape.Rank];
        var rankPadding = outputShape.Rank - inputShape.Rank;
        for (var inputAxis = 0; inputAxis < inputShape.Rank; inputAxis++)
        {
            var outputAxis = rankPadding + inputAxis;
            if (inputShape[inputAxis] == outputShape[outputAxis])
            {
                inputPolicies[inputAxis] = outputType.AxisPolicies[outputAxis];
            }
            else if (IsUnitDimension(inputShape[inputAxis]))
            {
                inputPolicies[inputAxis] = SBP.B;
            }
            else
            {
                return false;
            }
        }

        if (!DistributedUtility.IsDistributable(inputTensorType, inputPolicies, outputType.Placement))
        {
            return false;
        }

        inputType = new DistributedType(inputTensorType, inputPolicies, outputType.Placement);
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

    private static bool IsUnitDimension(Dimension dimension)
        => dimension is { IsFixed: true, FixedValue: 1 };
}
