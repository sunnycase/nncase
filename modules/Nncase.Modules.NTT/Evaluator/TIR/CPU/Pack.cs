// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Google.OrTools.ConstraintSolver;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class VectorizeEvaluator : ITypeInferencer<Nncase.TIR.NTT.Pack>, ITileWorkloadEvaluator<Nncase.TIR.NTT.Pack>
{
    public IRType Visit(ITypeInferenceContext context, Nncase.TIR.NTT.Pack target) => TupleType.Void;

    public TileWorkload Visit(Nncase.TIR.NTT.Pack op, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[0][^1], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][^1], solver.MakeIntConst(factor)));
    }
}
