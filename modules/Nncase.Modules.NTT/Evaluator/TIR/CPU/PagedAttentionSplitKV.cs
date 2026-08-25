// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class PagedAttentionUseSplitKVEvaluator : ITypeInferencer<PagedAttentionUseSplitKV>
{
    public IRType Visit(ITypeInferenceContext context, PagedAttentionUseSplitKV target)
        => TensorType.Scalar(DataTypes.Boolean);
}

public sealed class PagedAttentionPartialEvaluator : ITypeInferencer<PagedAttentionPartial>
{
    public IRType Visit(ITypeInferenceContext context, PagedAttentionPartial target)
        => TupleType.Void;
}

public sealed class PagedAttentionMergeEvaluator : ITypeInferencer<PagedAttentionMerge>
{
    public IRType Visit(ITypeInferenceContext context, PagedAttentionMerge target)
        => TupleType.Void;
}

public sealed class PagedAttentionMergePackedMatMulEvaluator :
    ITypeInferencer<PagedAttentionMergePackedMatMul>,
    ITileWorkloadEvaluator<PagedAttentionMergePackedMatMul>
{
    public IRType Visit(ITypeInferenceContext context, PagedAttentionMergePackedMatMul target)
        => TupleType.Void;

    public TileWorkload Visit(PagedAttentionMergePackedMatMul op, TileWorkloadContext context)
        => new MatrixTileWorkload(GetMatrixShape, DataTypes.Float32.SizeInBytes);

    private static MatrixTileWorkloadShape GetMatrixShape(
        Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes,
        Google.OrTools.ConstraintSolver.Solver solver,
        TileWorkloadContext context)
    {
        var lhsShape = bufferShapes[4];
        var outputShape = bufferShapes[6];
        var k = lhsShape[^1] * TileWorkloadUtility.GetVectorLaneCount(context.BufferDataTypes[4]);
        var m = outputShape[^2];
        var n = outputShape[^1] * TileWorkloadUtility.GetVectorLaneCount(context.BufferDataTypes[6]);
        return new(m, n, k, solver.MakeIntConst(1));
    }
}
