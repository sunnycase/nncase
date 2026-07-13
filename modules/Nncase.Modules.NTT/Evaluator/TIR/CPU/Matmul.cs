// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.NTT;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class MatmulEvaluator : ITypeInferencer<Matmul>, ITileWorkloadEvaluator<Matmul>
{
    public IRType Visit(ITypeInferenceContext context, Matmul target) => TupleType.Void;

    public TileWorkload Visit(Matmul op, TileWorkloadContext context) => new MatrixTileWorkload(GetMatrixShape, DataTypes.Float32.SizeInBytes);

    private static MatrixTileWorkloadShape GetMatrixShape(Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes, Google.OrTools.ConstraintSolver.Solver solver, TileWorkloadContext context)
    {
        var lhsShape = bufferShapes[0];
        var outputShape = bufferShapes[2];
        var matmul = (Matmul)context.Op;
        var k = lhsShape[matmul.TransposeA ? ^2 : ^1] * TileWorkloadUtility.GetVectorLaneCount(context.BufferDataTypes[0]);
        var m = outputShape[^2];
        var n = outputShape[^1] * TileWorkloadUtility.GetVectorLaneCount(context.BufferDataTypes[2]);
        return new(m, n, k, solver.MakeIntConst(1));
    }
}
