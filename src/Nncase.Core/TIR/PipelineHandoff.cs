// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// A one-shot consumer-to-producer ownership edge for Shared storage whose
/// previous owner is not a cyclic pipeline.
/// </summary>
public sealed class PipelineHandoff : Expr
{
    public PipelineHandoff(string handoffId, long sharedOffsetBytes)
        : base([])
    {
        if (string.IsNullOrWhiteSpace(handoffId))
        {
            throw new ArgumentException("Pipeline handoff ID must not be empty.", nameof(handoffId));
        }

        if (sharedOffsetBytes < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sharedOffsetBytes),
                sharedOffsetBytes,
                "Pipeline handoff Shared offset must be non-negative.");
        }

        HandoffId = handoffId;
        SharedOffsetBytes = sharedOffsetBytes;
    }

    public string HandoffId { get; }

    public long SharedOffsetBytes { get; }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(
        ExprFunctor<TExprResult, TTypeResult, TContext> functor,
        TContext context)
        => functor.VisitPipelineHandoff(this, context);

    public PipelineHandoff With(
        string? handoffId = null,
        long? sharedOffsetBytes = null)
        => new(handoffId ?? HandoffId, sharedOffsetBytes ?? SharedOffsetBytes);

    public override bool Equals(object? obj)
        => ReferenceEquals(this, obj) ||
            (obj is PipelineHandoff other &&
             HandoffId == other.HandoffId &&
             SharedOffsetBytes == other.SharedOffsetBytes);

    protected override int GetHashCodeCore()
        => HashCode.Combine(
            HandoffId.GetHashCode(StringComparison.Ordinal),
            SharedOffsetBytes);
}
