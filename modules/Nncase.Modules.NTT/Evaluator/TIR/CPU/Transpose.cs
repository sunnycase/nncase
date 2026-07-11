// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class TransposeEvaluator : ITypeInferencer<Transpose>, ITileWorkloadEvaluator<Transpose>
{
    public IRType Visit(ITypeInferenceContext context, Transpose target)
    {
        context.CheckArgumentType<TensorType>(target, Transpose.Input);
        context.CheckArgumentType<TensorType>(target, Transpose.Output);
        return TupleType.Void;
    }

    public TileWorkload Visit(Transpose op, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[0][^1], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][^1], solver.MakeIntConst(factor)));
    }
}
