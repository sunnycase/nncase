// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class MatMulGluEvaluator : ITypeInferencer<MatMulGlu>, IKernelInfoEvaluator<MatMulGlu>
{
    public IRType Visit(ITypeInferenceContext context, MatMulGlu target) => TupleType.Void;

    public MicroKernelInfo Visit(MatMulGlu op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var tileBounds = Enumerable.Repeat(new ValueRange<long>(1, int.MaxValue), domain.Length).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (INTTTargetOptions)context.TargetOptions;
        for (var i = 0; i < bufferInfos.Length; i++)
        {
            var state = i < bufferInfos.Length - 1 ? MicroKernelBufferInfo.BufferState.Read : MicroKernelBufferInfo.BufferState.Read | MicroKernelBufferInfo.BufferState.Write;
            bufferInfos[i] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], state);
        }

        return new MicroKernelInfo(tileBounds, bufferInfos, GetComputeCycle);
    }

    private static Google.OrTools.ConstraintSolver.IntExpr GetComputeCycle(Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes, Google.OrTools.ConstraintSolver.Solver solver, MicroKernelContext context)
    {
        var inputShape = bufferShapes[0];
        var outputShape = bufferShapes[^1];
        var k = inputShape[^1];
        var m = outputShape[^2];
        var n = outputShape[^1];
        var matmulCycles = solver.MakeProd(solver.MakeProd(solver.MakeProd(m, n), k), 2);
        var gluCycles = solver.MakeProd(solver.MakeProd(m, n), 9);
        return solver.MakeSum(matmulCycles, gluCycles);
    }
}
