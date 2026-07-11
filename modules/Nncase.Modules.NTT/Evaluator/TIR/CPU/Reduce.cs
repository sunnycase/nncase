// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class ReduceEvaluator : ITypeInferencer<Reduce>, ITileWorkloadEvaluator<Reduce>
{
    public IRType Visit(ITypeInferenceContext context, Reduce target)
    {
        context.CheckArgumentType<TensorType>(target, Reduce.Input);
        context.CheckArgumentType<TensorType>(target, Reduce.Output);
        return TupleType.Void;
    }

    public TileWorkload Visit(Reduce op, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[0][^1], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][^1], solver.MakeIntConst(factor)));
    }
}
