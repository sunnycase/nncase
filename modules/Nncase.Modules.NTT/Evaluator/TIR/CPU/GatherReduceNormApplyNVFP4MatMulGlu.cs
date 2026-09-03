// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class GatherReduceNormApplyNVFP4MatMulGluEvaluator :
    ITypeInferencer<GatherReduceNormApplyNVFP4MatMulGlu>,
    ITileWorkloadEvaluator<GatherReduceNormApplyNVFP4MatMulGlu>
{
    public IRType Visit(
        ITypeInferenceContext context,
        GatherReduceNormApplyNVFP4MatMulGlu target)
    {
        foreach (var parameter in target.Parameters.Take(target.Parameters.Count - 1))
        {
            _ = context.CheckArgumentType<IRType>(target, parameter);
        }

        return TupleType.Void;
    }

    public TileWorkload Visit(
        GatherReduceNormApplyNVFP4MatMulGlu op,
        TileWorkloadContext context) =>
        new MatrixTileWorkload(GetMatrixShape, DataTypes.Float32.SizeInBytes);

    private static MatrixTileWorkloadShape GetMatrixShape(
        Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes,
        Google.OrTools.ConstraintSolver.Solver solver,
        TileWorkloadContext context)
    {
        var inputShape = bufferShapes[GatherReduceNormApplyNVFP4MatMulGlu.Input.Index];
        var outputShape = bufferShapes[GatherReduceNormApplyNVFP4MatMulGlu.Output.Index];
        return new(
            outputShape[^2],
            outputShape[^1],
            inputShape[^1],
            solver.MakeIntConst(2));
    }
}
