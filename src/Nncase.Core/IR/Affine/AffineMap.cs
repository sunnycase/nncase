// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using NetFabric.Hyperlinq;
using Nncase.Utilities;

namespace Nncase.IR.Affine;

public sealed class AffineDomain : BaseExpr
{
    public AffineDomain(AffineDim offset, AffineExtent extent)
        : base(new BaseExpr[] { offset, extent })
    {
    }

    public AffineDim Offset => (AffineDim)Operands[0];

    public AffineExtent Extent => (AffineExtent)Operands[1];

    public override BaseExpr this[Dimension index] => throw new NotSupportedException();

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineDomain(this, context);

    public AffineDomain With(AffineDim? offset = null, AffineExtent? extent = null)
        => new AffineDomain(offset ?? Offset, extent ?? Extent);

    public override string ToString() => $"({Offset}, {Extent})";
}

public sealed class AffineRange : BaseExpr
{
    public AffineRange(AffineExpr offset, AffineExpr extent)
        : base(new BaseExpr[] { offset, extent })
    {
    }

    public AffineExpr Offset => (AffineExpr)Operands[0];

    public AffineExpr Extent => (AffineExpr)Operands[1];

    public override BaseExpr this[Dimension index] => throw new NotSupportedException();

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineRange(this, context);

    public AffineRange With(AffineExpr? offset = null, AffineExpr? extent = null)
        => new AffineRange(offset ?? Offset, extent ?? Extent);

    public (Dimension Offset, Dimension Extent) Apply(ReadOnlySpan<Dimension> dims, ReadOnlySpan<Dimension> extents, IReadOnlyDictionary<AffineSymbol, Dimension>? symbols = null)
    {
        var offset = Offset.Apply(dims, extents, symbols);
        var extent = Extent.Apply(dims, extents, symbols);
        return (offset, extent);
    }

    public ValueRange<long> Apply(ReadOnlySpan<long> dims, ReadOnlySpan<long> extents, IReadOnlyDictionary<AffineSymbol, long>? symbols = null)
    {
        var offset = Offset.Apply(dims, extents, symbols);
        var extent = Extent.Apply(dims, extents, symbols);
        return (offset, extent);
    }

    internal string GetDisplayString(ReadOnlySpan<AffineSymbol> symbols)
        => $"({Offset.GetDisplayString(symbols)}, {Extent.GetDisplayString(symbols)})";

    internal AffineRange ReplaceDomains(ReadOnlySpan<AffineRange> newDomains)
        => new AffineRange(Offset.ReplaceDomainsAndSymbols(newDomains, Array.Empty<AffineSymbol>()), Extent.ReplaceDomainsAndSymbols(newDomains, Array.Empty<AffineSymbol>()));
}

public sealed class AffineMap : BaseExpr
{
    private readonly int _domainsCount;
    private readonly int _symbolsCount;

    public AffineMap(ReadOnlySpan<AffineDomain> domains, ReadOnlySpan<AffineSymbol> symbols, ReadOnlySpan<AffineRange> results)
        : base(domains.ToArray().AsEnumerable<BaseExpr>().Concat(symbols.ToArray()).Concat(results.ToArray()))
    {
        _domainsCount = domains.Length;
        _symbolsCount = symbols.Length;
    }

    public ReadOnlySpan<AffineDomain> Domains => SpanUtility.UnsafeCast<BaseExpr, AffineDomain>(Operands.Slice(0, _domainsCount));

    public ReadOnlySpan<AffineSymbol> Symbols => SpanUtility.UnsafeCast<BaseExpr, AffineSymbol>(Operands.Slice(_domainsCount, _symbolsCount));

    public ReadOnlySpan<AffineRange> Results => SpanUtility.UnsafeCast<BaseExpr, AffineRange>(Operands.Slice(_domainsCount + _symbolsCount));

    public override BaseExpr this[Dimension index] => throw new NotSupportedException();

    public static AffineMap operator *(AffineMap lhs, AffineMap rhs)
    {
        if (lhs.Results.Length != rhs.Domains.Length)
        {
            throw new ArgumentException("Cannot compose AffineMaps with mismatching dimensions and results.");
        }

        var results = rhs.Results.AsValueEnumerable().Select(x => x.ReplaceDomains(lhs.Results)).ToArray();
        var symbols = lhs.Symbols.ToArray().Concat(rhs.Symbols.ToArray()).ToArray();
        return new AffineMap(lhs.Domains, symbols, results);
    }

    public static AffineMap FromCallable(Func<AffineDomain[], AffineSymbol[], AffineRange[]> func, int dimsCount, int symbolsCount = 0)
    {
        var domains = F.Affine.Domains(dimsCount);
        var symbols = F.Affine.Symbols(symbolsCount);
        var results = func(domains, symbols);
        return new AffineMap(domains, symbols, results);
    }

    public static AffineMap FromCallable(Delegate func)
    {
        var parameters = func.Method.GetParameters();
        var arguments = new object[parameters.Length];
        var domains = new List<AffineDomain>();
        var symbols = new List<AffineSymbol>();
        for (int i = 0; i < arguments.Length; i++)
        {
            var type = parameters[i].ParameterType;
            if (type == typeof(AffineDomain))
            {
                var domain = F.Affine.Domain(i);
                domains.Add(domain);
                arguments[i] = domain;
            }
            else if (type == typeof(AffineSymbol))
            {
                var symbol = F.Affine.Symbol(symbols.Count);
                symbols.Add(symbol);
                arguments[i] = symbol;
            }
            else
            {
                throw new ArgumentException("Invalid callable argument");
            }
        }

        var results = (AffineRange[])func.DynamicInvoke(arguments)!;
        return new AffineMap(CollectionsMarshal.AsSpan(domains), CollectionsMarshal.AsSpan(symbols), results);
    }

    public static AffineMap Identity(int rank)
    {
        var domains = F.Affine.Domains(rank);
        var results = domains.Select(x => new AffineRange(x.Offset, x.Extent)).ToArray();
        return new AffineMap(domains, default, results);
    }

    public static AffineMap Permutation(int[] perms)
    {
        var domains = F.Affine.Domains(perms.Length);
        var results = Enumerable.Range(0, perms.Length).Select(i => new AffineRange(domains[perms[i]].Offset, domains[perms[i]].Extent)).ToArray();
        return new AffineMap(domains, default, results);
    }

    /// <summary>
    /// Gets the number of adjacent result points produced by one domain point
    /// for a rectangular range result. Identity and downscale results have a
    /// multiplicity of one; an extent such as <c>4 * t0</c> has a
    /// multiplicity of four.
    /// </summary>
    public long GetPointMultiplicity(int resultIndex)
    {
        if ((uint)resultIndex >= (uint)Results.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(resultIndex), resultIndex, "Affine result index is out of range.");
        }

        if (Symbols.Length > 0)
        {
            throw new NotSupportedException("Point multiplicity does not support affine symbols.");
        }

        var zeroOffsets = new long[Domains.Length];
        var unitExtents = Enumerable.Repeat(1L, Domains.Length).ToArray();
        var multiplicity = Results[resultIndex].Extent.Apply(zeroOffsets, unitExtents);
        if (multiplicity < 0)
        {
            throw new InvalidOperationException(
                $"Affine result axis {resultIndex} has negative point multiplicity {multiplicity}: {Results[resultIndex]}.");
        }

        // A floor-divided extent evaluates to zero at unit extent, but its
        // point relation is a single-valued downscale rather than an empty
        // relation.
        return System.Math.Max(1, multiplicity);
    }

    public bool IsProjectedPermutation(bool allowConstInResults)
    {
        if (Symbols.Length > 0)
        {
            return false;
        }

        // Having more results than inputs means that results have duplicated dims or
        // zeros that can't be mapped to input dims.
        if (Results.Length > Domains.Length && !allowConstInResults)
        {
            return false;
        }

        var seen = Enumerable.Repeat(false, Domains.Length).ToArray();
        foreach (var range in Results)
        {
            switch (range.Offset, range.Extent)
            {
                case (AffineDim dim, AffineExtent extent) when dim.Position == extent.Position:
                    if (seen[dim.Position])
                    {
                        return false;
                    }

                    seen[dim.Position] = true;
                    break;
                case (AffineDivBinary { BinaryOp: AffineDivBinaryOp.FloorDiv, Lhs: AffineDim dim, Rhs: AffineConstant }, AffineExtent extent) when dim.Position == extent.Position:
                    if (seen[dim.Position])
                    {
                        return false;
                    }

                    seen[dim.Position] = true;
                    break;
                case (AffineConstant, AffineConstant):
                    if (!allowConstInResults)
                    {
                        return false;
                    }

                    break;
                default:
                    return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Tests whether every non-constant result axis is an independent,
    /// positive integral scale or downscale of one domain axis. Such maps
    /// preserve rectangular tile regions when the selected tile satisfies the
    /// scale alignment constraints.
    /// </summary>
    public bool IsRectangularProjection(bool allowConstInResults)
    {
        if (Symbols.Length > 0)
        {
            return false;
        }

        if (Results.Length > Domains.Length && !allowConstInResults)
        {
            return false;
        }

        var seen = new bool[Domains.Length];
        foreach (var range in Results)
        {
            if (range.Offset is AffineConstant && range.Extent is AffineConstant)
            {
                if (!allowConstInResults)
                {
                    return false;
                }

                continue;
            }

            if (!TryMatchAxisTransform(range.Offset, range.Extent, out var axis) || seen[axis])
            {
                return false;
            }

            seen[axis] = true;
        }

        return true;

        static bool TryMatchAxisTransform(AffineExpr offset, AffineExpr extent, out int axis)
        {
            axis = -1;
            switch (offset, extent)
            {
                case (AffineDim dim, AffineExtent axisExtent) when dim.Position == axisExtent.Position:
                    axis = dim.Position;
                    return true;
                case (AffineMulBinary offsetMul, AffineMulBinary extentMul)
                    when TryExtractPositiveConstantFactor(offsetMul, out var offsetScale, out var offsetInner) &&
                         TryExtractPositiveConstantFactor(extentMul, out var extentScale, out var extentInner) &&
                         offsetScale == extentScale:
                    return TryMatchAxisTransform(offsetInner, extentInner, out axis);
                case (
                    AffineDivBinary
                    {
                        BinaryOp: var offsetOp,
                        Lhs: var offsetInner,
                        Rhs: AffineConstant offsetDivisor,
                    },
                    AffineDivBinary
                    {
                        BinaryOp: var extentOp,
                        Lhs: var extentInner,
                        Rhs: AffineConstant extentDivisor,
                    })
                    when offsetOp is AffineDivBinaryOp.FloorDiv or AffineDivBinaryOp.CeilDiv &&
                         offsetOp == extentOp &&
                         offsetDivisor.Value > 0 &&
                         offsetDivisor.Value == extentDivisor.Value:
                    return TryMatchAxisTransform(offsetInner, extentInner, out axis);
                default:
                    return false;
            }

            static bool TryExtractPositiveConstantFactor(
                AffineMulBinary expression,
                out long factor,
                out AffineExpr inner)
            {
                switch (expression)
                {
                    case { Lhs: AffineConstant constant, Rhs: var rhs } when constant.Value > 0:
                        factor = constant.Value;
                        inner = rhs;
                        return true;
                    case { Lhs: var lhs, Rhs: AffineConstant constant } when constant.Value > 0:
                        factor = constant.Value;
                        inner = lhs;
                        return true;
                    default:
                        factor = 0;
                        inner = null!;
                        return false;
                }
            }
        }
    }

    public bool IsPermutation()
    {
        if (Domains.Length != Results.Length)
        {
            return false;
        }

        return IsProjectedPermutation(false);
    }

    public TIR.Range[] Apply(ReadOnlySpan<Dimension> dims, ReadOnlySpan<Dimension> extents, IReadOnlyDictionary<AffineSymbol, Dimension>? symbols = null)
    {
        var newResults = new TIR.Range[Results.Length];
        for (int i = 0; i < newResults.Length; i++)
        {
            newResults[i] = Results[i].Apply(dims, extents, symbols);
        }

        return newResults;
    }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineMap(this, context);

    public AffineMap With(AffineDomain[]? domains = null, AffineSymbol[]? symbols = null, AffineRange[]? results = null)
        => new AffineMap(domains ?? Domains, symbols ?? Symbols, results ?? Results);

    public override string ToString()
    {
        var domains = string.Join(", ", Enumerable.Range(0, Domains.Length).Select(i => $"(d{i}, t{i})"));
        var syms = string.Join(", ", Enumerable.Range(0, Symbols.Length).Select(i => $"s{i}"));
        var results = StringUtility.Join(", ", Results.AsValueEnumerable().Select(expr => expr.GetDisplayString(Symbols)));

        return Symbols.Length == 0 ? $"({domains}) -> ({results})" : $"({domains})[{syms}] -> ({results})";
    }
}
