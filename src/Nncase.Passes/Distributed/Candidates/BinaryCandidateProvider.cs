// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.Math;
using Nncase.IR;
using Nncase.IR.Math;

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
            if (BroadcastCandidateUtility.TryLiftInputLayout(inputType, outputTensorType, out var outputType))
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
            !BroadcastCandidateUtility.TryGetSourceTensorType(context, Binary.Lhs.Index, out var lhsTensorType) ||
            !BroadcastCandidateUtility.TryGetSourceTensorType(context, Binary.Rhs.Index, out var rhsTensorType))
        {
            return false;
        }

        if (!BroadcastCandidateUtility.TryProjectOutputLayout(outputType, lhsTensorType, out var lhsType) ||
            !BroadcastCandidateUtility.TryProjectOutputLayout(outputType, rhsTensorType, out var rhsType) ||
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
}
