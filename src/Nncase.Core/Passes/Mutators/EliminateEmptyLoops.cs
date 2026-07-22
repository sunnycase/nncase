// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Mutators;

/// <summary>
/// Eliminates statically empty positive-step TIR loops.
/// </summary>
public sealed class EliminateEmptyLoops : ExprRewriter
{
    /// <inheritdoc/>
    protected override BaseExpr RewriteLeafFor(For expr)
        => RewriteLoop(expr, expr.LoopVar, expr.Domain);

    /// <inheritdoc/>
    protected override BaseExpr RewriteLeafPipelineFor(PipelineFor expr)
        => RewriteLoop(expr, expr.LoopVar, expr.Domain);

    private BaseExpr RewriteLoop(BaseExpr loop, DimVar loopVar, TIR.Range domain)
    {
        if (!domain.Start.IsFixed || !domain.Stop.IsFixed || !domain.Step.IsFixed)
        {
            return loop;
        }

        if (domain.Step.FixedValue <= 0)
        {
            throw new InvalidOperationException(
                $"TIR loop {loopVar.Name} has non-positive step {domain.Step.FixedValue}; " +
                "empty-loop elimination requires positive-step half-open loop semantics.");
        }

        if (domain.Start.FixedValue < domain.Stop.FixedValue)
        {
            return loop;
        }

        SetMutated();
        return T.Nop();
    }
}
