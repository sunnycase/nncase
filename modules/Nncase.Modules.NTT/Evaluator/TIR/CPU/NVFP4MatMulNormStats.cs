// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class NVFP4MatMulNormStatsEvaluator :
    ITypeInferencer<NVFP4MatMulNormStats>,
    ITileWorkloadEvaluator<NVFP4MatMulNormStats>
{
    public IRType Visit(ITypeInferenceContext context, NVFP4MatMulNormStats target) => TupleType.Void;

    public TileWorkload Visit(NVFP4MatMulNormStats op, TileWorkloadContext context) =>
        new MatrixTileWorkload(GetMatrixShape, DataTypes.Float32.SizeInBytes);

    private static MatrixTileWorkloadShape GetMatrixShape(
        Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes,
        Google.OrTools.ConstraintSolver.Solver solver,
        TileWorkloadContext context)
    {
        var lhsShape = bufferShapes[NVFP4MatMulNormStats.Lhs.Index];
        var outputShape = bufferShapes[NVFP4MatMulNormStats.Output.Index];
        return new(
            outputShape[^2],
            outputShape[^1],
            lhsShape[^1],
            solver.MakeIntConst(1));
    }
}
