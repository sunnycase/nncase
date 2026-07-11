// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class SwishEvaluator : ITypeInferencer<Swish>, ITileWorkloadEvaluator<Swish>
{
    public IRType Visit(ITypeInferenceContext context, Swish target)
    {
        context.CheckArgumentType<TensorType>(target, Swish.Input);
        context.CheckArgumentType<TensorType>(target, Swish.Output);
        return TupleType.Void;
    }

    public TileWorkload Visit(Swish swish, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[0][^1], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][^1], solver.MakeIntConst(factor)));
    }
}
