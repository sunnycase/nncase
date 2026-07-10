// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class UpdatePagedAttentionKVCacheEvaluator : ITypeInferencer<UpdatePagedAttentionKVCache>, IKernelInfoEvaluator<UpdatePagedAttentionKVCache>
{
    public IRType Visit(ITypeInferenceContext context, UpdatePagedAttentionKVCache target) => TupleType.Void;

    public MicroKernelInfo Visit(UpdatePagedAttentionKVCache op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var tileBounds = Enumerable.Repeat(new ValueRange<long>(1, int.MaxValue), domain.Length).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (INTTTargetOptions)context.TargetOptions;
        if (bufferInfos.Length != 3)
        {
            throw new InvalidOperationException($"UpdatePagedAttentionKVCache expects slots, input kv-cache, and output kv-cache buffers, got {bufferInfos.Length}.");
        }

        bufferInfos[0] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[1] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[2] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Write);
        return new MicroKernelInfo(tileBounds, bufferInfos, GetComputeCycle);
    }

    private static IntExpr GetComputeCycle(IntExpr[][] bufferShapes, Solver solver, MicroKernelContext context)
    {
        return bufferShapes[0].Aggregate((IntExpr)solver.MakeIntConst(1), (acc, dim) => solver.MakeProd(acc, dim));
    }
}
