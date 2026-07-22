// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// A regular TIR loop. Compiler-visible producer/consumer schedules are
/// represented directly by <see cref="PipelineFor"/> instead of annotations
/// on this node.
/// </summary>
public sealed class For : Expr
{
    public For(
        DimVar loopVar,
        Range domain,
        LoopMode mode,
        Sequential body,
        LoopPartition partition = LoopPartition.Unpartitioned)
        : base([loopVar, domain, body])
    {
        Mode = mode;
        Partition = partition;
    }

    public For(
        DimVar loopVar,
        Range domain,
        LoopMode mode,
        LoopPartition partition = LoopPartition.Unpartitioned)
        : this(loopVar, domain, mode, Sequential.Empty, partition)
    {
    }

    public DimVar LoopVar => (DimVar)Operands[0];

    public Range Domain => (Range)Operands[1];

    public LoopMode Mode { get; }

    public LoopPartition Partition { get; }

    public Sequential Body => (Sequential)Operands[2];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(
        ExprFunctor<TExprResult, TTypeResult, TContext> functor,
        TContext context)
        => functor.VisitFor(this, context);

    public For With(
        DimVar? loopVar = null,
        Range? domain = null,
        LoopMode? loopMode = null,
        Sequential? body = null,
        LoopPartition? partition = null)
        => new(
            loopVar ?? LoopVar,
            domain ?? Domain,
            loopMode ?? Mode,
            body ?? Body,
            partition ?? Partition);

    public override bool Equals(object? obj)
    {
        if (ReferenceEquals(this, obj))
        {
            return true;
        }

        return obj is For other &&
            Mode == other.Mode &&
            Partition == other.Partition &&
            base.Equals(other);
    }

    protected override int GetHashCodeCore()
        => HashCode.Combine(Mode, Partition, base.GetHashCodeCore());
}
