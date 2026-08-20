// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.Passes.Distributed;

internal sealed class BindNormStatsCandidateProvider : DistributedCandidateProvider<BindNormStats>
{
    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        BindNormStats target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes.Where(IsMaterialized).ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        BindNormStats target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (!IsMaterialized(returnType) || context.AvailableInputTypes.Count != 2)
        {
            return false;
        }

        var stats = context.AvailableInputTypes[BindNormStats.Stats.Index]
            .FirstOrDefault(candidate => Equals(candidate, returnType));
        if (stats is null)
        {
            return true;
        }

        var normStats = new NormStats(target.Axis, target.UseMean);
        tuples = context.AvailableInputTypes[BindNormStats.Input.Index]
            .Where(input => BindNormStatsEvaluator.IsCompatibleMaterializedStatsType(
                NormStatsEvaluator.InferType(normStats, input),
                returnType))
            .Select(input => new DistributedCandidateTuple(
                [input, stats],
                "materialized-norm-stats-binding"))
            .ToArray();
        return true;
    }

    private static bool IsMaterialized(IRType type)
        => type is not DistributedType distributed ||
            (distributed.Partial is null && distributed.AxisPolicies.All(policy => policy is not SBPPartial));
}
