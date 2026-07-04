// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class PackedQKVParallelLinearEvaluator : ITypeInferencer<PackedQKVParallelLinear>, IKernelInfoEvaluator<PackedQKVParallelLinear>
{
    public IRType Visit(ITypeInferenceContext context, PackedQKVParallelLinear target) => TupleType.Void;

    public MicroKernelInfo Visit(PackedQKVParallelLinear op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var tileBounds = Enumerable.Repeat(new ValueRange<long>(1, int.MaxValue), domain.Length).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (INTTTargetOptions)context.TargetOptions;
        for (var i = 0; i < bufferInfos.Length; i++)
        {
            var state = i < bufferInfos.Length - 3 ? MicroKernelBufferInfo.BufferState.Read : MicroKernelBufferInfo.BufferState.Read | MicroKernelBufferInfo.BufferState.Write;
            bufferInfos[i] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], state);
        }

        return new MicroKernelInfo(tileBounds, bufferInfos, GetComputeCycle);
    }

    private static Google.OrTools.ConstraintSolver.IntExpr GetComputeCycle(Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes, Google.OrTools.ConstraintSolver.Solver solver, MicroKernelContext context)
    {
        var inputShape = bufferShapes[0];
        var qOutputShape = bufferShapes[^3];
        var kOutputShape = bufferShapes[^2];
        var vOutputShape = bufferShapes[^1];
        var k = inputShape[^1];
        var m = qOutputShape[^2];
        var n = solver.MakeSum(solver.MakeSum(qOutputShape[^1], kOutputShape[^1]), vOutputShape[^1]);
        return solver.MakeProd(solver.MakeProd(m, n), k);
    }
}
