// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Google.OrTools.ConstraintSolver;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class VectorizedLayerNormEvaluator : ITypeInferencer<VectorizedLayerNorm>, ITileWorkloadEvaluator<VectorizedLayerNorm>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, VectorizedLayerNorm target) => TupleType.Void;

    public TileWorkload Visit(VectorizedLayerNorm op, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[0][0], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][0], solver.MakeIntConst(factor)));
    }
}
