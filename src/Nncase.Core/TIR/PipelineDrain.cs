// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// Terminal rendezvous for a cyclic pipeline before any of its Shared backing
/// storage is reused.
/// </summary>
public sealed class PipelineDrain : Expr
{
    public PipelineDrain(string stageId)
        : base([])
    {
        if (string.IsNullOrWhiteSpace(stageId))
        {
            throw new ArgumentException("Pipeline drain stage ID must not be empty.", nameof(stageId));
        }

        StageId = stageId;
    }

    public string StageId { get; }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(
        ExprFunctor<TExprResult, TTypeResult, TContext> functor,
        TContext context)
        => functor.VisitPipelineDrain(this, context);

    public PipelineDrain With(string? stageId = null)
        => new(stageId ?? StageId);

    public override bool Equals(object? obj)
        => ReferenceEquals(this, obj) ||
            (obj is PipelineDrain other && StageId == other.StageId);

    protected override int GetHashCodeCore() => StageId.GetHashCode(StringComparison.Ordinal);
}
