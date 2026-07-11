// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Schedule;

namespace Nncase.Evaluator.TIR.NTT;

internal static class TransferKernelInfo
{
    public static MicroKernelInfo Create(MicroKernelContext context)
    {
        if (context.BufferShapes.Length != 2)
        {
            throw new InvalidOperationException($"Affine transfer expects one input and one output, got {context.BufferShapes.Length} buffers.");
        }

        var domain = context.AccessMaps[0].Domains;
        var tileBounds = Enumerable.Range(0, domain.Length)
            .Select(_ => new ValueRange<long>(1, int.MaxValue))
            .ToArray();
        var options = (INTTTargetOptions)context.TargetOptions;
        var bandwidth = options.MemoryBandWidths[1];
        var bufferInfos = new[]
        {
            new MicroKernelBufferInfo(bandwidth, bandwidth, MicroKernelBufferInfo.BufferState.Read),
            new MicroKernelBufferInfo(bandwidth, bandwidth, MicroKernelBufferInfo.BufferState.Write),
        };
        return new MicroKernelInfo(tileBounds, bufferInfos, GetComputeCycle);
    }

    private static IntExpr GetComputeCycle(IntExpr[][] bufferShapes, Solver solver, MicroKernelContext context)
        => bufferShapes[0].Aggregate((IntExpr)solver.MakeIntConst(1), (acc, dim) => solver.MakeProd(acc, dim));
}
