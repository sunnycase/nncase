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
    {
        var domain = expr.Domain;
        if (!domain.Start.IsFixed || !domain.Stop.IsFixed || !domain.Step.IsFixed)
        {
            return expr;
        }

        if (domain.Step.FixedValue <= 0)
        {
            throw new InvalidOperationException(
                $"TIR loop {expr.LoopVar.Name} has non-positive step {domain.Step.FixedValue}; " +
                "empty-loop elimination requires positive-step half-open loop semantics.");
        }

        if (domain.Start.FixedValue < domain.Stop.FixedValue)
        {
            return expr;
        }

        SetMutated();
        return T.Nop();
    }
}
