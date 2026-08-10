// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// A one-shot consumer-to-producer dependency edge.
/// </summary>
public sealed class PipelineHandoff : Expr
{
    public PipelineHandoff(string handoffId)
        : base([])
    {
        if (string.IsNullOrWhiteSpace(handoffId))
        {
            throw new ArgumentException("Pipeline handoff ID must not be empty.", nameof(handoffId));
        }

        HandoffId = handoffId;
    }

    public string HandoffId { get; }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(
        ExprFunctor<TExprResult, TTypeResult, TContext> functor,
        TContext context)
        => functor.VisitPipelineHandoff(this, context);

    public PipelineHandoff With(string? handoffId = null)
        => new(handoffId ?? HandoffId);

    public override bool Equals(object? obj)
        => ReferenceEquals(this, obj) ||
            (obj is PipelineHandoff other && HandoffId == other.HandoffId);

    protected override int GetHashCodeCore()
        => HandoffId.GetHashCode(StringComparison.Ordinal);
}
