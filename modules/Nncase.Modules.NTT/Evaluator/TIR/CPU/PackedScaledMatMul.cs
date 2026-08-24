// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class PackedScaledMatMulEvaluator : ITypeInferencer<PackedScaledMatMul>, ITileWorkloadEvaluator<PackedScaledMatMul>
{
    public IRType Visit(ITypeInferenceContext context, PackedScaledMatMul target) => TupleType.Void;

    public TileWorkload Visit(PackedScaledMatMul op, TileWorkloadContext context) =>
        new MatrixTileWorkload(GetMatrixShape, DataTypes.Float32.SizeInBytes);

    private static MatrixTileWorkloadShape GetMatrixShape(
        Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes,
        Google.OrTools.ConstraintSolver.Solver solver,
        TileWorkloadContext context)
    {
        var lhsShape = bufferShapes[PackedScaledMatMul.Lhs.Index];
        var outputShape = bufferShapes[PackedScaledMatMul.Output.Index];
        var k = lhsShape[^1] * TileWorkloadUtility.GetVectorLaneCount(
            context.BufferDataTypes[PackedScaledMatMul.Lhs.Index]);
        var m = outputShape[^2];
        var n = outputShape[^1] * TileWorkloadUtility.GetVectorLaneCount(
            context.BufferDataTypes[PackedScaledMatMul.Output.Index]);
        return new(m, n, k, solver.MakeIntConst(1));
    }
}
