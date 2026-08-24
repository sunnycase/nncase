// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class PackedBlockScaledMatMulEvaluator :
    ITypeInferencer<PackedBlockScaledMatMul>,
    ITileWorkloadEvaluator<PackedBlockScaledMatMul>
{
    public IRType Visit(ITypeInferenceContext context, PackedBlockScaledMatMul target) => TupleType.Void;

    public TileWorkload Visit(PackedBlockScaledMatMul op, TileWorkloadContext context) =>
        new MatrixTileWorkload(GetMatrixShape, DataTypes.Float32.SizeInBytes);

    private static MatrixTileWorkloadShape GetMatrixShape(
        Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes,
        Google.OrTools.ConstraintSolver.Solver solver,
        TileWorkloadContext context)
    {
        var lhsShape = bufferShapes[PackedBlockScaledMatMul.Lhs.Index];
        var outputShape = bufferShapes[PackedBlockScaledMatMul.Output.Index];
        var k = lhsShape[^1] * TileWorkloadUtility.GetVectorLaneCount(
            context.BufferDataTypes[PackedBlockScaledMatMul.Lhs.Index]);
        var m = outputShape[^2];
        var n = outputShape[^1] * TileWorkloadUtility.GetVectorLaneCount(
            context.BufferDataTypes[PackedBlockScaledMatMul.Output.Index]);
        return new(m, n, k, solver.MakeIntConst(1));
    }
}
