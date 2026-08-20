// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.Passes.Distributed;

internal static class BroadcastCandidateUtility
{
    public static bool TryLiftInputLayout(
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

    public static bool TryProjectOutputLayout(
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

    public static bool TryGetSourceTensorType(
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
