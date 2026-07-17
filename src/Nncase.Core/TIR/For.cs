// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// A for loop, with poissible type annotations.
/// <example>
/// <code>
///   for (loop_var = min; loop_var &lt; min + extent; ++loop_var) {
///     body
///    }
/// </code>
/// </example>
/// </summary>
public sealed class For : Expr
{
    /// <summary>
    /// Initializes a new instance of the <see cref="For"/> class.
    /// </summary>
    /// <param name="loopVar">The loop variable.</param>
    /// <param name="domain">The domain of for range.</param>
    /// <param name="mode">The kind of the for loop.</param>
    /// <param name="body">The body sequence.</param>
    /// <param name="partition">The structured full/tail partition.</param>
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

    /// <summary>
    /// Initializes a new instance of the <see cref="For"/> class.
    /// </summary>
    /// <param name="loopVar">The loop variable.</param>
    /// <param name="domain">The domain of for range.</param>
    /// <param name="mode">The kind of the for loop.</param>
    /// <param name="partition">The structured full/tail partition.</param>
    public For(
        DimVar loopVar,
        Range domain,
        LoopMode mode,
        LoopPartition partition = LoopPartition.Unpartitioned)
        : this(loopVar, domain, mode, new(), partition)
    {
    }

    /// <summary>
    /// Gets the loop variable.
    /// </summary>
    public DimVar LoopVar => (DimVar)Operands[0];

    /// <summary>
    /// Gets the domain of for range.
    /// </summary>
    public Range Domain => (Range)Operands[1];

    /// <summary>
    /// Gets the kind of the for loop.
    /// </summary>
    public LoopMode Mode { get; }

    /// <summary>
    /// Gets the structured full/tail partition represented by this loop.
    /// </summary>
    public LoopPartition Partition { get; }

    /// <summary>
    /// Gets the body sequence.
    /// </summary>
    public Sequential Body => (Sequential)Operands[2];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitFor(this, context);

    public For With(
        DimVar? loopVar = null,
        Range? domain = null,
        LoopMode? loopMode = null,
        Sequential? body = null,
        LoopPartition? partition = null)
        => new For(loopVar ?? LoopVar, domain ?? Domain, loopMode ?? Mode, body ?? Body, partition ?? Partition);

    /// <inheritdoc/>
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

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Mode, Partition, base.GetHashCodeCore());
}
