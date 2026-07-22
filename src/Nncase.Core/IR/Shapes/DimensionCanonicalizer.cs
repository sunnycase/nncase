// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR.Shapes;

namespace Nncase.IR;

/// <summary>
/// Canonicalizes dimension expressions without changing their integer semantics.
/// </summary>
public sealed class DimensionCanonicalizer
{
    private readonly Func<DimVar, Dimension?>? _resolveVariable;
    private readonly Func<Dimension, Dimension, bool>? _canProveLessOrEqual;
    private readonly Dictionary<Dimension, Dimension> _memo = new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<DimVar> _resolvingVariables = new(ReferenceEqualityComparer.Instance);

    /// <summary>
    /// Initializes a new instance of the <see cref="DimensionCanonicalizer"/> class.
    /// </summary>
    /// <param name="resolveVariable">Optional scoped variable substitution.</param>
    /// <param name="canProveLessOrEqual">Optional scoped proof for <c>lhs &lt;= rhs</c>.</param>
    public DimensionCanonicalizer(
        Func<DimVar, Dimension?>? resolveVariable = null,
        Func<Dimension, Dimension, bool>? canProveLessOrEqual = null)
    {
        _resolveVariable = resolveVariable;
        _canProveLessOrEqual = canProveLessOrEqual;
    }

    /// <summary>
    /// Canonicalizes <paramref name="value"/>.
    /// </summary>
    public Dimension Canonicalize(Dimension value)
    {
        if (_memo.TryGetValue(value, out var cached))
        {
            return cached;
        }

        var result = CanonicalizeCore(value);
        if (result.Equals(value))
        {
            result = value;
        }

        _memo[value] = result;
        return result;
    }

    internal static Dimension SimplifyMin(
        ReadOnlySpan<Dimension> operands,
        Func<Dimension, Dimension, bool>? canProveLessOrEqual = null)
        => SimplifyExtremum(operands, isMin: true, canProveLessOrEqual);

    internal static Dimension SimplifyMax(
        ReadOnlySpan<Dimension> operands,
        Func<Dimension, Dimension, bool>? canProveLessOrEqual = null)
        => SimplifyExtremum(operands, isMin: false, canProveLessOrEqual);

    private static bool DefaultCanProveLessOrEqual(Dimension lhs, Dimension rhs)
    {
        if (lhs.Equals(rhs))
        {
            return true;
        }

        if (lhs.IsFixed && rhs.IsFixed)
        {
            return lhs.FixedValue <= rhs.FixedValue;
        }

        return lhs.Metadata.Range is { } lhsRange
            && rhs.Metadata.Range is { } rhsRange
            && lhsRange.Max <= rhsRange.Min;
    }

    private static Dimension SimplifyExtremum(
        ReadOnlySpan<Dimension> operands,
        bool isMin,
        Func<Dimension, Dimension, bool>? canProveLessOrEqual)
    {
        if (operands.Length == 0)
        {
            throw new ArgumentException("At least one dimension is required.", nameof(operands));
        }

        var flattened = new List<Dimension>(operands.Length);
        foreach (var operand in operands)
        {
            if (operand.IsUnknown)
            {
                return Dimension.Unknown;
            }

            if (isMin && operand is DimMin nestedMin)
            {
                flattened.AddRange(nestedMin.Operands.ToArray());
            }
            else if (!isMin && operand is DimMax nestedMax)
            {
                flattened.AddRange(nestedMax.Operands.ToArray());
            }
            else
            {
                flattened.Add(operand);
            }
        }

        var candidates = new List<Dimension>(flattened.Count);
        foreach (var candidate in flattened)
        {
            var redundant = false;
            for (var i = candidates.Count - 1; i >= 0; i--)
            {
                var existing = candidates[i];
                if (existing.Equals(candidate))
                {
                    redundant = true;
                    break;
                }

                var existingDominates = isMin
                    ? ProveLessOrEqual(existing, candidate, canProveLessOrEqual)
                    : ProveLessOrEqual(candidate, existing, canProveLessOrEqual);
                if (existingDominates)
                {
                    redundant = true;
                    break;
                }

                var candidateDominates = isMin
                    ? ProveLessOrEqual(candidate, existing, canProveLessOrEqual)
                    : ProveLessOrEqual(existing, candidate, canProveLessOrEqual);
                if (candidateDominates)
                {
                    candidates.RemoveAt(i);
                }
            }

            if (!redundant)
            {
                candidates.Add(candidate);
            }
        }

        return candidates.Count switch
        {
            0 => throw new InvalidOperationException("Dimension extremum canonicalization removed every operand."),
            1 => candidates[0],
            _ when candidates.All(x => x.IsFixed) => new DimConst(
                isMin
                    ? candidates.Min(x => x.FixedValue)
                    : candidates.Max(x => x.FixedValue)),
            _ when isMin => new DimMin(candidates.ToArray()),
            _ => new DimMax(candidates.ToArray()),
        };
    }

    private static bool ProveLessOrEqual(
        Dimension lhs,
        Dimension rhs,
        Func<Dimension, Dimension, bool>? canProveLessOrEqual)
        => DefaultCanProveLessOrEqual(lhs, rhs)
            || (canProveLessOrEqual?.Invoke(lhs, rhs) ?? false);

    private Dimension CanonicalizeCore(Dimension value)
    {
        return value switch
        {
            DimConst or UnknownDim => value,
            DimVar variable => ResolveVariable(variable),
            DimProduct product => product.With(
                operands: product.Operands.ToArray().Select(Canonicalize).ToArray()),
            DimSum sum => sum.With(
                operands: sum.Operands.ToArray().Select(Canonicalize).ToArray()),
            DimFraction fraction => fraction.With(
                numerator: Canonicalize(fraction.Numerator),
                denominator: Canonicalize(fraction.Denominator)),
            DimRemainder remainder => remainder.With(
                numerator: Canonicalize(remainder.Numerator),
                denominator: Canonicalize(remainder.Denominator)),
            DimAbs abs => Dimension.Abs(Canonicalize(abs.Operand)),
            DimClamp clamp => Dimension.Clamp(
                Canonicalize(clamp.Operand),
                Canonicalize(clamp.MinValue),
                Canonicalize(clamp.MaxValue)),
            DimCompareAndSelect select => CanonicalizeSelect(select),
            DimMin min => SimplifyMin(
                min.Operands.ToArray().Select(Canonicalize).ToArray(),
                CanProveLessOrEqual),
            DimMax max => SimplifyMax(
                max.Operands.ToArray().Select(Canonicalize).ToArray(),
                CanProveLessOrEqual),
            DimPositive positive => Dimension.Positive(
                Canonicalize(positive.Operand),
                Canonicalize(positive.Extent)),
            DimAt at => at.With(
                shape: CanonicalizeShape(at.Shape),
                index: Canonicalize(at.Index)),
            DimPower power => Dimension.Pow(Canonicalize(power.Dim), power.Power),
            AsDim => value,
            _ => value,
        };
    }

    private Dimension ResolveVariable(DimVar variable)
    {
        if (_resolveVariable is null || !_resolvingVariables.Add(variable))
        {
            return variable;
        }

        try
        {
            var replacement = _resolveVariable(variable);
            return replacement is null || ReferenceEquals(replacement, variable)
                ? variable
                : Canonicalize(replacement);
        }
        finally
        {
            _resolvingVariables.Remove(variable);
        }
    }

    private Shape CanonicalizeShape(Shape shape)
        => shape is RankedShape ranked
            ? new RankedShape(ranked.Dimensions.ToArray().Select(Canonicalize).ToArray())
            : shape;

    private bool CanProveLessOrEqual(Dimension lhs, Dimension rhs)
        => DefaultCanProveLessOrEqual(lhs, rhs)
            || (_canProveLessOrEqual?.Invoke(lhs, rhs) ?? false);

    private Dimension CanonicalizeSelect(DimCompareAndSelect select)
    {
        var value = Canonicalize(select.Value);
        var expected = Canonicalize(select.Expected);
        var trueValue = Canonicalize(select.TrueValue);
        var falseValue = Canonicalize(select.FalseValue);
        var condition = select.CompareOp switch
        {
            CompareOp.Equal when CanProveEqual(value, expected) => true,
            CompareOp.Equal when CanProveLessThan(value, expected) || CanProveLessThan(expected, value) => false,
            CompareOp.NotEqual when CanProveLessThan(value, expected) || CanProveLessThan(expected, value) => true,
            CompareOp.NotEqual when CanProveEqual(value, expected) => false,
            CompareOp.LowerThan when CanProveLessThan(value, expected) => true,
            CompareOp.LowerThan when CanProveLessOrEqual(expected, value) => false,
            CompareOp.LowerOrEqual when CanProveLessOrEqual(value, expected) => true,
            CompareOp.LowerOrEqual when CanProveLessThan(expected, value) => false,
            CompareOp.GreaterThan when CanProveLessThan(expected, value) => true,
            CompareOp.GreaterThan when CanProveLessOrEqual(value, expected) => false,
            CompareOp.GreaterOrEqual when CanProveLessOrEqual(expected, value) => true,
            CompareOp.GreaterOrEqual when CanProveLessThan(value, expected) => false,
            _ => (bool?)null,
        };
        return condition switch
        {
            true => trueValue,
            false => falseValue,
            null => Dimension.Select(value, expected, trueValue, falseValue, select.CompareOp),
        };
    }

    private bool CanProveEqual(Dimension lhs, Dimension rhs)
        => CanProveLessOrEqual(lhs, rhs) && CanProveLessOrEqual(rhs, lhs);

    private bool CanProveLessThan(Dimension lhs, Dimension rhs)
    {
        try
        {
            return CanProveLessOrEqual(lhs + 1, rhs);
        }
        catch (OverflowException)
        {
            return false;
        }
    }
}
