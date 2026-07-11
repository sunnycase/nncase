// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class CastEvaluator : ITypeInferencer<Cast>, ITileWorkloadEvaluator<Cast>
{
    public IRType Visit(ITypeInferenceContext context, Cast target) => TupleType.Void;

    public TileWorkload Visit(Cast op, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[0][^1], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][^1], solver.MakeIntConst(factor)));
    }
}
