// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class GetPositionIdsEvaluator : ITypeInferencer<GetPositionIds>, ITileWorkloadEvaluator<GetPositionIds>
{
    public IRType Visit(ITypeInferenceContext context, GetPositionIds target)
    {
        return TupleType.Void;
    }

    public TileWorkload Visit(GetPositionIds op, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[1][^1], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[1][^1], solver.MakeIntConst(factor)));
    }
}
