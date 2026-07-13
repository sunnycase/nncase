// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class MatMulGluEvaluator : ITypeInferencer<MatMulGlu>, ITileWorkloadEvaluator<MatMulGlu>
{
    public IRType Visit(ITypeInferenceContext context, MatMulGlu target) => TupleType.Void;

    public TileWorkload Visit(MatMulGlu op, TileWorkloadContext context) => new MatrixTileWorkload(GetMatrixShape, DataTypes.Float32.SizeInBytes);

    private static MatrixTileWorkloadShape GetMatrixShape(Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes, Google.OrTools.ConstraintSolver.Solver solver, TileWorkloadContext context)
    {
        var inputShape = bufferShapes[0];
        var outputShape = bufferShapes[^1];
        var k = inputShape[^1] * TileWorkloadUtility.GetVectorLaneCount(context.BufferDataTypes[0]);
        var m = outputShape[^2];
        var n = outputShape[^1] * TileWorkloadUtility.GetVectorLaneCount(context.BufferDataTypes[^1]);
        return new(m, n, k, solver.MakeIntConst(2));
    }
}
