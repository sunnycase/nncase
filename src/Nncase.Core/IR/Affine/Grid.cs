// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.IR.Affine;

public sealed class Grid : Expr
{
    private readonly int _accessesCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="Grid"/> class.
    /// </summary>
    /// <param name="domainParameter">the grid domain parameter. </param>
    /// <param name="accesses">Grid accesses.</param>
    /// <param name="body">The body sequence.</param>
    public Grid(Var domainParameter, ReadOnlySpan<GridAccess> accesses, Sequential body)
        : base(new BaseExpr[] { domainParameter }.Concat(accesses.ToArray()).Append(body))
    {
        _accessesCount = accesses.Length;

        if (accesses.IsEmpty)
        {
            throw new ArgumentException("Grid must have at least one access.", nameof(accesses));
        }

        var affineDomainRanks = accesses
            .ToArray()
            .Where(access => access.IsAffine)
            .Select(access => access.AffineMap.Domains.Length)
            .Distinct()
            .ToArray();
        if (affineDomainRanks.Length != 1)
        {
            throw new ArgumentException("All affine grid access regions must have the same domain rank.", nameof(accesses));
        }

        if (!accesses.ToArray().Any(access => access.IsWrite))
        {
            throw new ArgumentException("Grid must have at least one write access.", nameof(accesses));
        }

        if (!accesses.ToArray().Any(access => access.IsAffine && access.DomainMode == GridDomainMode.Constraint))
        {
            throw new ArgumentException("Grid must have at least one affine domain constraint.", nameof(accesses));
        }
    }

    public Var DomainParameter => (Var)Operands[0];

    public ReadOnlySpan<GridAccess> Accesses => SpanUtility.UnsafeCast<BaseExpr, GridAccess>(Operands.Slice(1, _accessesCount));

    public Sequential Body => (Sequential)Operands[1 + _accessesCount];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitGrid(this, context);

    public Grid With(Var? domainParameter = null, GridAccess[]? accesses = null, Sequential? body = null)
        => new(domainParameter ?? DomainParameter, accesses ?? Accesses, body ?? Body);
}
