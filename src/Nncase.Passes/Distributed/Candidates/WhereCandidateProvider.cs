// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Passes.Distributed;

internal sealed class WhereCandidateProvider : DistributedCandidateProvider<Where>
{
    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        Where target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        if (target.IsTfWhere || context.SourceCall.CheckedType is not TensorType outputTensorType)
        {
            return defaultReturnTypes;
        }

        var results = new List<IRType>(defaultReturnTypes);
        foreach (var input in context.AvailableInputTypes.SelectMany(types => types).OfType<DistributedType>())
        {
            if (BroadcastCandidateUtility.TryLiftInputLayout(input, outputTensorType, out var output))
            {
                results.Add(output);
            }
        }

        return results.Distinct().ToArray();
    }

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        Where target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (target.IsTfWhere ||
            returnType is not DistributedType output ||
            output.Partial is not null ||
            output.AxisPolicies.Any(policy => policy is SBPPartial) ||
            context.AvailableInputTypes.Count != 3 ||
            !BroadcastCandidateUtility.TryGetSourceTensorType(context, Where.Cond.Index, out var condTensorType) ||
            !BroadcastCandidateUtility.TryGetSourceTensorType(context, Where.X.Index, out var xTensorType) ||
            !BroadcastCandidateUtility.TryGetSourceTensorType(context, Where.Y.Index, out var yTensorType) ||
            !BroadcastCandidateUtility.TryProjectOutputLayout(output, condTensorType, out var condType) ||
            !BroadcastCandidateUtility.TryProjectOutputLayout(output, xTensorType, out var xType) ||
            !BroadcastCandidateUtility.TryProjectOutputLayout(output, yTensorType, out var yType))
        {
            return false;
        }

        tuples =
        [
            new DistributedCandidateTuple(
                [condType, xType, yType],
                "where-exact-output-sbp"),
        ];
        return true;
    }
}
