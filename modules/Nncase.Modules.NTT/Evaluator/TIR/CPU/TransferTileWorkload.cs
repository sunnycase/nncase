// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Schedule;

namespace Nncase.Evaluator.TIR.NTT;

internal static class TransferTileWorkload
{
    public static TileWorkload Create(TileWorkloadContext context)
    {
        if (context.BufferShapes.Length != 2)
        {
            throw new InvalidOperationException($"Affine transfer expects one input and one output, got {context.BufferShapes.Length} buffers.");
        }

        return new ElementwiseTileWorkload(GetComputeWork);
    }

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
        => bufferShapes[0].Aggregate((IntExpr)solver.MakeIntConst(1), (acc, dim) => solver.MakeProd(acc, dim));
}
