// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class GatherReduceAddNormStatsEvaluator :
    ITypeInferencer<GatherReduceAddNormStats>,
    ITileWorkloadEvaluator<GatherReduceAddNormStats>
{
    public IRType Visit(ITypeInferenceContext context, GatherReduceAddNormStats target)
    {
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormStats.Input);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormStats.Collective);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormStats.Addend);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormStats.ValueOutput);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormStats.StatsOutput);
        return TupleType.Void;
    }

    public TileWorkload Visit(GatherReduceAddNormStats op, TileWorkloadContext context)
        => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
        => bufferShapes[3].Aggregate(
            (IntExpr)solver.MakeIntConst(1),
            (acc, dim) => solver.MakeProd(acc, dim));
}

public sealed class GatherReduceAddNormApplyEvaluator :
    ITypeInferencer<GatherReduceAddNormApply>,
    ITileWorkloadEvaluator<GatherReduceAddNormApply>
{
    public IRType Visit(ITypeInferenceContext context, GatherReduceAddNormApply target)
    {
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormApply.Input);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormApply.Collective);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormApply.Addend);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormApply.ValueOutput);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormApply.StatsWorkspace);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormApply.Scale);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormApply.Bias);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceAddNormApply.NormOutput);
        return TupleType.Void;
    }

    public TileWorkload Visit(GatherReduceAddNormApply op, TileWorkloadContext context)
        => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
        => bufferShapes[3].Aggregate(
            (IntExpr)solver.MakeIntConst(1),
            (acc, dim) => solver.MakeProd(acc, dim));
}
