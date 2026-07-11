// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class NormStatsEvaluator : ITypeInferencer<NormStats>, ITileWorkloadEvaluator<NormStats>
{
    public IRType Visit(ITypeInferenceContext context, NormStats target) => TupleType.Void;

    public TileWorkload Visit(NormStats op, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var inputSize = bufferShapes[0].Aggregate((IntExpr)solver.MakeIntConst(1), (acc, dim) => solver.MakeProd(acc, dim));
        return inputSize;
    }
}

public sealed class NormApplyEvaluator : ITypeInferencer<NormApply>, ITileWorkloadEvaluator<NormApply>
{
    public IRType Visit(ITypeInferenceContext context, NormApply target) => TupleType.Void;

    public TileWorkload Visit(NormApply op, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var outputSize = bufferShapes[^1].Aggregate((IntExpr)solver.MakeIntConst(1), (acc, dim) => solver.MakeProd(acc, dim));
        return outputSize;
    }
}
