// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class PadEvaluator : ITypeInferencer<Pad>, ITileWorkloadEvaluator<Pad>
{
    public IRType Visit(ITypeInferenceContext context, Pad target)
    {
        context.CheckArgumentType<TensorType>(target, Pad.Input);
        context.CheckArgumentType<TensorType>(target, Pad.Output);
        return TupleType.Void;
    }

    public TileWorkload Visit(Pad op, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[0][0], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][0], solver.MakeIntConst(factor)));
    }
}
