// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="BindNormStats"/>.
/// </summary>
public sealed class BindNormStatsEvaluator :
    IEvaluator<BindNormStats>,
    ITypeInferencer<BindNormStats>,
    ICostEvaluator<BindNormStats>
{
    public static bool IsCompatibleMaterializedStatsType(IRType expected, IRType stats)
    {
        if (Equals(expected, stats))
        {
            return true;
        }

        return expected is DistributedType { Partial: { Op: ReduceOp.Sum, Axes.Count: > 0 } } partial &&
            stats is DistributedType { Partial: null } materialized &&
            Equals(partial with { Partial = null }, materialized);
    }

    public IValue Visit(IEvaluateContext context, BindNormStats target)
        => context.GetArgumentValue(target, BindNormStats.Stats);

    public IRType Visit(ITypeInferenceContext context, BindNormStats target)
    {
        var input = context.CheckArgumentType<IRType>(target, BindNormStats.Input);
        var stats = context.CheckArgumentType<IRType>(target, BindNormStats.Stats);
        var expected = NormStatsEvaluator.InferType(
            new NormStats(target.Axis, target.UseMean),
            input);
        if (expected is InvalidType invalid)
        {
            return invalid;
        }

        if (!IsCompatibleMaterializedStatsType(expected, stats))
        {
            return new InvalidType(
                $"BindNormStats stats type {stats} must be the materialized NormStats({input}) type {expected}.");
        }

        if (stats is DistributedType distributed &&
            (distributed.Partial is not null || distributed.AxisPolicies.Any(policy => policy is SBPPartial)))
        {
            return new InvalidType("BindNormStats requires materialized, non-partial statistics.");
        }

        return stats;
    }

    public Cost Visit(ICostEvaluateContext context, BindNormStats target) => new();
}
