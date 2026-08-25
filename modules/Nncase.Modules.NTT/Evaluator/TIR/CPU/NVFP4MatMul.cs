// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class NVFP4MatMulEvaluator :
    ITypeInferencer<NVFP4MatMul>,
    ITileWorkloadEvaluator<NVFP4MatMul>
{
    public IRType Visit(ITypeInferenceContext context, NVFP4MatMul target) => TupleType.Void;

    public TileWorkload Visit(NVFP4MatMul op, TileWorkloadContext context) =>
        new MatrixTileWorkload(GetMatrixShape, DataTypes.Float32.SizeInBytes);

    private static MatrixTileWorkloadShape GetMatrixShape(
        Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes,
        Google.OrTools.ConstraintSolver.Solver solver,
        TileWorkloadContext context)
    {
        var lhsShape = bufferShapes[NVFP4MatMul.Lhs.Index];
        var outputShape = bufferShapes[NVFP4MatMul.Output.Index];
        return new(
            outputShape[^2],
            outputShape[^1],
            lhsShape[^1],
            solver.MakeIntConst(1));
    }
}
