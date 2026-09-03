// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache License. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.Math;
using Nncase.IR;
using Nncase.IR.Math;

namespace Nncase.Passes.Distributed;

/// <summary>
/// Builds block-scaled matmul candidates from the result layout while keeping
/// weight scales replicated and quantization-block boundaries intact.
/// </summary>
internal sealed class BlockScaledMatMulCandidateProvider :
    DistributedCandidateProvider<BlockScaledMatMul>
{
    public override bool IsExhaustive => true;

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        BlockScaledMatMul target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (context.AvailableInputTypes.Count != 3)
        {
            return false;
        }

        if (returnType is not DistributedType outputType)
        {
            return true;
        }

        var scaleCandidates = context.AvailableInputTypes[BlockScaledMatMul.RhsScale.Index]
            .OfType<DistributedType>()
            .Where(type =>
                type.Placement == outputType.Placement &&
                type.Partial is null &&
                type.AxisPolicies.All(policy => policy is SBPBroadCast))
            .ToArray();
        if (scaleCandidates.Length == 0)
        {
            return true;
        }

        var results = new List<DistributedCandidateTuple>();
        foreach (var lhs in context.AvailableInputTypes[BlockScaledMatMul.Lhs.Index]
                     .OfType<DistributedType>()
                     .Where(type => type.Placement == outputType.Placement))
        {
            foreach (var rhs in context.AvailableInputTypes[BlockScaledMatMul.Rhs.Index]
                         .OfType<DistributedType>()
                         .Where(type => type.Placement == outputType.Placement))
            {
                foreach (var scale in scaleCandidates)
                {
                    if (BlockScaledMatMulEvaluator.InferType(target, lhs, rhs, scale) != outputType)
                    {
                        continue;
                    }

                    results.Add(new DistributedCandidateTuple(
                        [lhs, rhs, scale],
                        "block-scaled-matmul-output-sbp"));
                }
            }
        }

        tuples = results;
        return true;
    }
}
