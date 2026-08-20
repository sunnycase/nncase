// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Utilities;
using Gather = Nncase.IR.Tensors.Gather;

namespace Nncase.Passes.Distributed;

internal sealed class GatherCandidateProvider : DistributedCandidateProvider<Gather>
{
    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        Gather target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (returnType is not DistributedType output ||
            output.Partial is not null ||
            output.AxisPolicies.Any(policy => policy is SBPPartial) ||
            context.AvailableInputTypes.Count != 2 ||
            !BroadcastCandidateUtility.TryGetSourceTensorType(context, Gather.Input.Index, out var inputTensorType) ||
            !BroadcastCandidateUtility.TryGetSourceTensorType(context, Gather.Index.Index, out var indexTensorType) ||
            !TryProjectOutputLayout(target.Axis, output, inputTensorType, indexTensorType, out var inputType, out var indexType))
        {
            return false;
        }

        tuples =
        [
            new DistributedCandidateTuple(
                [inputType, indexType],
                "gather-exact-output-sbp"),
        ];
        return true;
    }

    private static bool TryProjectOutputLayout(
        int axis,
        DistributedType outputType,
        TensorType inputTensorType,
        TensorType indexTensorType,
        out DistributedType inputType,
        out DistributedType indexType)
    {
        inputType = null!;
        indexType = null!;
        if (inputTensorType.Shape is not RankedShape inputShape ||
            indexTensorType.Shape is not RankedShape indexShape ||
            outputType.TensorType.Shape is not RankedShape outputShape)
        {
            return false;
        }

        axis = axis < 0 ? axis + inputShape.Rank : axis;
        if (axis < 0 || axis >= inputShape.Rank ||
            outputShape.Rank != inputShape.Rank - 1 + indexShape.Rank)
        {
            return false;
        }

        var inputPolicies = new SBP[inputShape.Rank];
        for (var inputAxis = 0; inputAxis < inputShape.Rank; inputAxis++)
        {
            inputPolicies[inputAxis] = inputAxis switch
            {
                var value when value < axis => outputType.AxisPolicies[value],
                var value when value == axis => SBP.B,
                _ => outputType.AxisPolicies[inputAxis - 1 + indexShape.Rank],
            };
        }

        var indexPolicies = new SBP[indexShape.Rank];
        for (var indexAxis = 0; indexAxis < indexShape.Rank; indexAxis++)
        {
            indexPolicies[indexAxis] = outputType.AxisPolicies[axis + indexAxis];
            if (indexPolicies[indexAxis] is SBPSplit)
            {
                return false;
            }
        }

        if (!DistributedUtility.IsDistributable(inputTensorType, inputPolicies, outputType.Placement) ||
            !DistributedUtility.IsDistributable(indexTensorType, indexPolicies, outputType.Placement))
        {
            return false;
        }

        inputType = new DistributedType(inputTensorType, inputPolicies, outputType.Placement);
        indexType = new DistributedType(indexTensorType, indexPolicies, outputType.Placement);
        return true;
    }
}
