// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class NVFP4MatMulGluEvaluator :
    ITypeInferencer<NVFP4MatMulGlu>,
    ITileWorkloadEvaluator<NVFP4MatMulGlu>
{
    public IRType Visit(ITypeInferenceContext context, NVFP4MatMulGlu target) => TupleType.Void;

    public TileWorkload Visit(NVFP4MatMulGlu op, TileWorkloadContext context) =>
        new MatrixTileWorkload(GetMatrixShape, DataTypes.Float32.SizeInBytes);

    private static MatrixTileWorkloadShape GetMatrixShape(
        Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes,
        Google.OrTools.ConstraintSolver.Solver solver,
        TileWorkloadContext context)
    {
        var inputShape = bufferShapes[NVFP4MatMulGlu.Input.Index];
        var outputShape = bufferShapes[NVFP4MatMulGlu.Output.Index];
        return new(
            outputShape[^2],
            outputShape[^1],
            inputShape[^1],
            solver.MakeIntConst(2));
    }
}
