// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class NormStatsEvaluator : ITypeInferencer<NormStats>, IKernelInfoEvaluator<NormStats>
{
    public IRType Visit(ITypeInferenceContext context, NormStats target) => TupleType.Void;

    public MicroKernelInfo Visit(NormStats op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var tileBounds = Enumerable.Repeat(new ValueRange<long>(1, int.MaxValue), domain.Length).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (INTTTargetOptions)context.TargetOptions;
        bufferInfos[0] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[1] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Write);
        return new MicroKernelInfo(tileBounds, bufferInfos, GetComputeCycle);
    }

    private static IntExpr GetComputeCycle(IntExpr[][] bufferShapes, Solver solver, MicroKernelContext context)
    {
        var inputSize = bufferShapes[0].Aggregate((IntExpr)solver.MakeIntConst(1), (acc, dim) => solver.MakeProd(acc, dim));
        return inputSize;
    }
}

public sealed class NormApplyEvaluator : ITypeInferencer<NormApply>, IKernelInfoEvaluator<NormApply>
{
    public IRType Visit(ITypeInferenceContext context, NormApply target) => TupleType.Void;

    public MicroKernelInfo Visit(NormApply op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var tileBounds = Enumerable.Repeat(new ValueRange<long>(1, int.MaxValue), domain.Length).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (INTTTargetOptions)context.TargetOptions;
        for (int i = 0; i < bufferInfos.Length; i++)
        {
            var state = i == bufferInfos.Length - 1
                ? MicroKernelBufferInfo.BufferState.Write
                : MicroKernelBufferInfo.BufferState.Read;
            bufferInfos[i] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], state);
        }

        return new MicroKernelInfo(tileBounds, bufferInfos, GetComputeCycle);
    }

    private static IntExpr GetComputeCycle(IntExpr[][] bufferShapes, Solver solver, MicroKernelContext context)
    {
        var outputSize = bufferShapes[^1].Aggregate((IntExpr)solver.MakeIntConst(1), (acc, dim) => solver.MakeProd(acc, dim));
        return outputSize;
    }
}
