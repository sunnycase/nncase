// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class UpdatePagedAttentionKVCacheEvaluator : ITypeInferencer<UpdatePagedAttentionKVCache>, ITileWorkloadEvaluator<UpdatePagedAttentionKVCache>
{
    public IRType Visit(ITypeInferenceContext context, UpdatePagedAttentionKVCache target) => TupleType.Void;

    public TileWorkload Visit(UpdatePagedAttentionKVCache op, TileWorkloadContext context) => new ElementwiseTileWorkload(GetComputeWork);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        return bufferShapes[0].Aggregate((IntExpr)solver.MakeIntConst(1), (acc, dim) => solver.MakeProd(acc, dim));
    }
}
