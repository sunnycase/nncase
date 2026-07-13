// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class PackedQKVParallelLinearEvaluator : ITypeInferencer<PackedQKVParallelLinear>, ITileWorkloadEvaluator<PackedQKVParallelLinear>
{
    public IRType Visit(ITypeInferenceContext context, PackedQKVParallelLinear target) => TupleType.Void;

    public TileWorkload Visit(PackedQKVParallelLinear op, TileWorkloadContext context) => new MatrixTileWorkload(GetMatrixShape, DataTypes.Float32.SizeInBytes);

    private static MatrixTileWorkloadShape GetMatrixShape(Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes, Google.OrTools.ConstraintSolver.Solver solver, TileWorkloadContext context)
    {
        var inputShape = bufferShapes[0];
        var qOutputShape = bufferShapes[^3];
        var kOutputShape = bufferShapes[^2];
        var vOutputShape = bufferShapes[^1];
        var k = inputShape[^1] * TileWorkloadUtility.GetVectorLaneCount(context.BufferDataTypes[0]);
        var m = qOutputShape[^2];
        var qN = qOutputShape[^1] * TileWorkloadUtility.GetVectorLaneCount(context.BufferDataTypes[^3]);
        var kN = kOutputShape[^1] * TileWorkloadUtility.GetVectorLaneCount(context.BufferDataTypes[^2]);
        var vN = vOutputShape[^1] * TileWorkloadUtility.GetVectorLaneCount(context.BufferDataTypes[^1]);
        var n = solver.MakeSum(solver.MakeSum(qN, kN), vN);
        return new(m, n, k, solver.MakeIntConst(1));
    }
}
