// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Google.OrTools.ConstraintSolver;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class VectorizedBinaryEvaluator : ITypeInferencer<VectorizedBinary>, ITileWorkloadEvaluator<VectorizedBinary>
{
    public IRType Visit(ITypeInferenceContext context, VectorizedBinary target) => TupleType.Void;

    public TileWorkload Visit(VectorizedBinary op, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var factora = System.Math.Min(context.BufferShapes[0][^1], 32);
        var factorb = System.Math.Min(context.BufferShapes[1][^1], 32);
        return factora * factorb * (1 + solver.MakeIsLessVar(bufferShapes[0][^1], solver.MakeIntConst(factora)) + solver.MakeIsLessVar(bufferShapes[1][^1], solver.MakeIntConst(factorb)));
    }
}
