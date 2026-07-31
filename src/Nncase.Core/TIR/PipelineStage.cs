// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// One pipelined operation invocation viewed as a producer or consumer stage
/// according to its enclosing <see cref="ProducerConsumerRegion"/> body.
/// The operation is either a directly selected microkernel or a call to a
/// pipeline-bearing <see cref="PrimFunction"/>.
/// </summary>
public sealed class PipelineStage : Expr
{
    public PipelineStage(string stageId, Call operation)
        : base([operation])
    {
        if (string.IsNullOrWhiteSpace(stageId))
        {
            throw new ArgumentException("Pipeline stage ID must not be empty.", nameof(stageId));
        }

        StageId = stageId;
    }

    public string StageId { get; }

    public Call Operation => (Call)Operands[0];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(
        ExprFunctor<TExprResult, TTypeResult, TContext> functor,
        TContext context)
        => functor.VisitPipelineStage(this, context);

    public PipelineStage With(string? stageId = null, Call? operation = null)
        => new(stageId ?? StageId, operation ?? Operation);

    public override bool Equals(object? obj)
        => ReferenceEquals(this, obj) ||
            (obj is PipelineStage other &&
             StageId == other.StageId &&
             base.Equals(other));

    protected override int GetHashCodeCore()
        => HashCode.Combine(StageId, base.GetHashCodeCore());
}
