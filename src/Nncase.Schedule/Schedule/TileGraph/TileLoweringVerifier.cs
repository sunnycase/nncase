// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Verifies invariants required at the AutoTiling-to-TIR boundary.
/// </summary>
internal static class TileLoweringVerifier
{
    public static void Verify(Expr body, string functionName)
    {
        if (body is Sequential { Count: 0 })
        {
            throw new InvalidOperationException(
                $"AutoTiling produced an empty body for {functionName}.");
        }

        foreach (var expression in ExprCollector.Collect(body))
        {
            switch (expression)
            {
                case TIR.For { Body.Count: 0 } loop:
                    throw new InvalidOperationException(
                        $"AutoTiling produced empty loop {loop.LoopVar.Name} in {functionName}.");
                case Let { Body.Count: 0 } let:
                    throw new InvalidOperationException(
                        $"AutoTiling produced empty let binding {let.Var.Name} in {functionName}.");
            }
        }
    }
}
