// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.IR.Affine;

public enum GridTileExtentKind
{
    Search,
    FullExtent,
    Fixed,
}

/// <summary>
/// Target-independent tiling legality for one grid domain axis.
/// </summary>
public sealed record GridTileAxisPolicy
{
    private GridTileAxisPolicy(GridTileExtentKind extentKind, long extent, long alignment)
    {
        if (alignment <= 0 || !System.Numerics.BitOperations.IsPow2((ulong)alignment))
        {
            throw new ArgumentOutOfRangeException(nameof(alignment), alignment, "Tile alignment must be a positive power of two.");
        }

        if (extentKind == GridTileExtentKind.Fixed && extent <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(extent), extent, "A fixed tile extent must be positive.");
        }

        ExtentKind = extentKind;
        Extent = extent;
        Alignment = alignment;
    }

    public GridTileExtentKind ExtentKind { get; }

    public long Extent { get; }

    public long Alignment { get; }

    public static GridTileAxisPolicy FullExtent { get; } = new(GridTileExtentKind.FullExtent, 0, 1);

    public static GridTileAxisPolicy Search(long alignment = 1)
        => new(GridTileExtentKind.Search, 0, alignment);

    public static GridTileAxisPolicy Fixed(long extent)
        => new(GridTileExtentKind.Fixed, extent, 1);

    public override string ToString()
        => ExtentKind switch
        {
            GridTileExtentKind.Search when Alignment == 1 => "search",
            GridTileExtentKind.Search => $"search-align-{Alignment}",
            GridTileExtentKind.FullExtent => "full",
            GridTileExtentKind.Fixed => $"fixed-{Extent}",
            _ => throw new ArgumentOutOfRangeException(),
        };
}

public sealed class Grid : Expr
{
    private readonly int _accessesCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="Grid"/> class.
    /// </summary>
    /// <param name="domainParameter">the grid domain parameter. </param>
    /// <param name="accesses">Grid accesses.</param>
    /// <param name="body">The body sequence.</param>
    /// <param name="tileAxisPolicies">Tiling legality for each domain axis.</param>
    public Grid(Var domainParameter, ReadOnlySpan<GridAccess> accesses, Sequential body, ReadOnlySpan<GridTileAxisPolicy> tileAxisPolicies)
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

        var domainRank = affineDomainRanks[0];
        if (tileAxisPolicies.Length != domainRank)
        {
            throw new ArgumentException($"Grid has a rank-{domainRank} domain but {tileAxisPolicies.Length} tile-axis policies.", nameof(tileAxisPolicies));
        }

        if (tileAxisPolicies.Contains(null!))
        {
            throw new ArgumentException("Grid tile-axis policies must not contain null.", nameof(tileAxisPolicies));
        }

        TileAxisPolicies = tileAxisPolicies.ToArray();
    }

    public Var DomainParameter => (Var)Operands[0];

    public ReadOnlySpan<GridAccess> Accesses => SpanUtility.UnsafeCast<BaseExpr, GridAccess>(Operands.Slice(1, _accessesCount));

    public Sequential Body => (Sequential)Operands[1 + _accessesCount];

    public IReadOnlyList<GridTileAxisPolicy> TileAxisPolicies { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitGrid(this, context);

    public Grid With(Var? domainParameter = null, GridAccess[]? accesses = null, Sequential? body = null, GridTileAxisPolicy[]? tileAxisPolicies = null)
        => new(domainParameter ?? DomainParameter, accesses ?? Accesses, body ?? Body, tileAxisPolicies ?? TileAxisPolicies.ToArray());
}
