// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Reactive;
using System.Text.Json.Serialization;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;

namespace Nncase.CodeGen.PyNTT;

internal sealed class PyNTTDimEquivalence : IEquatable<PyNTTDimEquivalence>
{
    private const int MaxAffineTerms = 16;
    private readonly KeyValuePair<string, long>[] _terms;

    private PyNTTDimEquivalence(long constant, IEnumerable<KeyValuePair<string, long>> terms)
    {
        Constant = constant;
        _terms = terms
            .Where(term => term.Value != 0)
            .OrderBy(term => term.Key, StringComparer.Ordinal)
            .ToArray();
    }

    public long Constant { get; }

    public int TermCount => _terms.Length;

    public static PyNTTDimEquivalence FromConstant(long value) => new(value, []);

    public static PyNTTDimEquivalence FromAtom(string atom)
        => new(0, [new KeyValuePair<string, long>(atom, 1)]);

    public static PyNTTDimEquivalence Add(PyNTTDimEquivalence lhs, PyNTTDimEquivalence rhs)
        => Combine(lhs, rhs, 1);

    public static PyNTTDimEquivalence Subtract(PyNTTDimEquivalence lhs, PyNTTDimEquivalence rhs)
        => Combine(lhs, rhs, -1);

    public static PyNTTDimEquivalence? TryAdd(PyNTTDimEquivalence lhs, PyNTTDimEquivalence rhs)
        => lhs.TermCount + rhs.TermCount <= MaxAffineTerms ? Add(lhs, rhs) : null;

    public static PyNTTDimEquivalence? TrySubtract(PyNTTDimEquivalence lhs, PyNTTDimEquivalence rhs)
        => lhs.TermCount + rhs.TermCount <= MaxAffineTerms ? Subtract(lhs, rhs) : null;

    public static PyNTTDimEquivalence Scale(PyNTTDimEquivalence value, long scale)
        => new(
            checked(value.Constant * scale),
            value._terms.Select(term => new KeyValuePair<string, long>(term.Key, checked(term.Value * scale))));

    public bool Equals(PyNTTDimEquivalence? other)
        => other is not null &&
            Constant == other.Constant &&
            _terms.AsSpan().SequenceEqual(other._terms);

    public override bool Equals(object? obj) => Equals(obj as PyNTTDimEquivalence);

    public override int GetHashCode()
    {
        var hash = default(HashCode);
        hash.Add(Constant);
        foreach (var term in _terms)
        {
            hash.Add(term.Key, StringComparer.Ordinal);
            hash.Add(term.Value);
        }

        return hash.ToHashCode();
    }

    private static PyNTTDimEquivalence Combine(
        PyNTTDimEquivalence lhs,
        PyNTTDimEquivalence rhs,
        long rhsScale)
    {
        var terms = new Dictionary<string, long>(StringComparer.Ordinal);
        AddTerms(lhs, 1);
        AddTerms(rhs, rhsScale);
        return new(checked(lhs.Constant + (rhs.Constant * rhsScale)), terms);

        void AddTerms(PyNTTDimEquivalence value, long scale)
        {
            foreach (var term in value._terms)
            {
                terms.TryGetValue(term.Key, out var coefficient);
                terms[term.Key] = checked(coefficient + (term.Value * scale));
            }
        }
    }
}

public sealed record PyNTTDimExpression(
    string PythonExpression,
    string TritonExpression,
    long? FixedValue = null,
    long? RangeMin = null,
    long? RangeMax = null)
{
    public static PyNTTDimExpression Zero { get; } = new("0", "0", 0)
    {
        Equivalence = PyNTTDimEquivalence.FromConstant(0),
    };

    public static PyNTTDimExpression One { get; } = new("1", "1", 1)
    {
        Equivalence = PyNTTDimEquivalence.FromConstant(1),
    };

    public bool IsFixed => FixedValue.HasValue;

    public bool IsFixedOne => FixedValue == 1;

    public bool IsFixedNonOne => FixedValue.HasValue && FixedValue.Value != 1;

    public long? MinValue => FixedValue ?? RangeMin;

    public long? MaxValue => FixedValue ?? RangeMax;

    [JsonIgnore]
    internal PyNTTDimEquivalence? Equivalence { get; init; }

    public object ToPythonLiteral() => FixedValue.HasValue ? FixedValue.Value : PythonExpression;

    public override string ToString() => TritonExpression;

    internal bool IsEquivalentTo(PyNTTDimExpression other)
    {
        if (FixedValue.HasValue && other.FixedValue.HasValue)
        {
            return FixedValue.Value == other.FixedValue.Value;
        }

        if (Equivalence is { } equivalence &&
            other.Equivalence is { } otherEquivalence &&
            equivalence.Equals(otherEquivalence))
        {
            return true;
        }

        return PythonExpression == other.PythonExpression ||
            TritonExpression == other.TritonExpression;
    }

    internal PyNTTDimExpression EnsureEquivalence()
        => Equivalence is null
            ? this with { Equivalence = PyNTTDimEquivalence.FromAtom(TritonExpression) }
            : this;
}

internal sealed class PyNTTDimExpressionEmitter : ExprFunctor<PyNTTDimExpression, Unit>
{
    private readonly Action<string>? _registerRuntimeScalar;
    private readonly Func<string, string> _formatRuntimeScalar;
    private readonly string _threadIdExpression;
    private readonly Func<DimVar, PyNTTDimExpression?>? _resolveDimVar;

    public PyNTTDimExpressionEmitter(
        Action<string>? registerRuntimeScalar = null,
        Func<string, string>? formatRuntimeScalar = null,
        string? threadIdExpression = null,
        Func<DimVar, PyNTTDimExpression?>? resolveDimVar = null)
    {
        _registerRuntimeScalar = registerRuntimeScalar;
        _formatRuntimeScalar = formatRuntimeScalar ?? (name => name);
        _threadIdExpression = threadIdExpression ?? "pyntt_thread_id";
        _resolveDimVar = resolveDimVar;
    }

    public PyNTTDimExpression Emit(Dimension dimension)
    {
        var expression = Visit(dimension).EnsureEquivalence();
        return WithRangeFromMetadata(expression, dimension);
    }

    protected override PyNTTDimExpression VisitDimAbs(DimAbs expr)
    {
        var operand = Visit(expr.Operand);
        return new PyNTTDimExpression($"abs({operand.PythonExpression})", $"tl.abs({operand.TritonExpression})")
            .EnsureEquivalence();
    }

    protected override PyNTTDimExpression VisitDimClamp(DimClamp expr)
    {
        var operand = Visit(expr.Operand);
        var min = Visit(expr.MinValue);
        var max = Visit(expr.MaxValue);
        return WithRangeFromMetadata(Minimum(Maximum(operand, min), max), expr);
    }

    protected override PyNTTDimExpression VisitDimCompareAndSelect(DimCompareAndSelect expr)
    {
        var value = Visit(expr.Value);
        var expected = Visit(expr.Expected);
        var trueValue = Visit(expr.TrueValue);
        var falseValue = Visit(expr.FalseValue);
        var op = CompareOpToPython(expr.CompareOp);
        var predicate = $"{value.PythonExpression} {op} {expected.PythonExpression}";
        var tritonPredicate = $"{value.TritonExpression} {op} {expected.TritonExpression}";
        return new PyNTTDimExpression(
            $"({trueValue.PythonExpression} if ({predicate}) else {falseValue.PythonExpression})",
            $"tl.where({tritonPredicate}, {trueValue.TritonExpression}, {falseValue.TritonExpression})")
            .EnsureEquivalence();
    }

    protected override PyNTTDimExpression VisitDimConst(DimConst expr)
        => Const(expr.Value);

    protected override PyNTTDimExpression VisitDimFraction(DimFraction expr)
    {
        var numerator = Visit(expr.Numerator);
        var denominator = Visit(expr.Denominator);
        if (expr.DivMode == DimDivideMode.FloorDiv)
        {
            var expression = new PyNTTDimExpression(
                $"(({numerator.PythonExpression}) // ({denominator.PythonExpression}))",
                $"(({numerator.TritonExpression}) // ({denominator.TritonExpression}))")
                .EnsureEquivalence();
            return WithRangeFromMetadata(expression, expr);
        }

        var ceilExpression = new PyNTTDimExpression(
            $"(({numerator.PythonExpression} + {denominator.PythonExpression} - 1) // ({denominator.PythonExpression}))",
            $"(({numerator.TritonExpression} + {denominator.TritonExpression} - 1) // ({denominator.TritonExpression}))")
            .EnsureEquivalence();
        return WithRangeFromMetadata(ceilExpression, expr);
    }

    protected override PyNTTDimExpression VisitDimMax(DimMax expr)
    {
        var operands = expr.Operands.ToArray().Select(Visit).ToArray();
        if (operands.Length == 1)
        {
            return operands[0];
        }

        return WithRangeFromMetadata(operands.Skip(1).Aggregate(operands[0], Maximum), expr);
    }

    protected override PyNTTDimExpression VisitDimMin(DimMin expr)
    {
        var operands = expr.Operands.ToArray().Select(Visit).ToArray();
        if (operands.Length == 1)
        {
            return operands[0];
        }

        return WithRangeFromMetadata(operands.Skip(1).Aggregate(operands[0], Minimum), expr);
    }

    protected override PyNTTDimExpression VisitDimPositive(DimPositive expr)
    {
        var operand = Visit(expr.Operand);
        var extent = Visit(expr.Extent);
        return new PyNTTDimExpression(
            $"(({operand.PythonExpression}) if ({operand.PythonExpression} >= 0) else ({operand.PythonExpression} + {extent.PythonExpression}))",
            $"tl.where({operand.TritonExpression} >= 0, {operand.TritonExpression}, {operand.TritonExpression} + {extent.TritonExpression})")
            .EnsureEquivalence();
    }

    protected override PyNTTDimExpression VisitDimPower(DimPower expr)
    {
        var dim = Visit((Dimension)expr.Dim);
        return new PyNTTDimExpression(
            $"(({dim.PythonExpression}) ** {expr.Power.ToString(CultureInfo.InvariantCulture)})",
            $"(({dim.TritonExpression}) ** {expr.Power.ToString(CultureInfo.InvariantCulture)})")
            .EnsureEquivalence();
    }

    protected override PyNTTDimExpression VisitDimProduct(DimProduct expr)
    {
        var parts = new List<PyNTTDimExpression>();
        if (expr.Scale != 1)
        {
            parts.Add(new(
                expr.Scale.ToString(CultureInfo.InvariantCulture),
                expr.Scale.ToString(CultureInfo.InvariantCulture),
                expr.Scale));
        }

        parts.AddRange(expr.Operands.ToArray().Select(Visit));
        return WithRangeFromMetadata(BuildBinaryChain(parts, "*", 1), expr);
    }

    protected override PyNTTDimExpression VisitDimRemainder(DimRemainder expr)
    {
        var numerator = Visit(expr.Numerator);
        var denominator = Visit(expr.Denominator);
        var expression = new PyNTTDimExpression(
            $"(({numerator.PythonExpression}) % ({denominator.PythonExpression}))",
            $"(({numerator.TritonExpression}) % ({denominator.TritonExpression}))")
            .EnsureEquivalence();
        return WithRangeFromMetadata(expression, expr);
    }

    protected override PyNTTDimExpression VisitDimSum(DimSum expr)
    {
        var parts = new List<PyNTTDimExpression>();
        if (expr.Bias != 0)
        {
            parts.Add(new(
                expr.Bias.ToString(CultureInfo.InvariantCulture),
                expr.Bias.ToString(CultureInfo.InvariantCulture),
                expr.Bias));
        }

        parts.AddRange(expr.Operands.ToArray().Select(Visit));
        return WithRangeFromMetadata(BuildBinaryChain(parts, "+", 0), expr);
    }

    protected override PyNTTDimExpression VisitDimVar(DimVar expr)
    {
        var name = SanitizePythonIdentifier(expr.Name);
        _registerRuntimeScalar?.Invoke(name);
        var formattedName = _formatRuntimeScalar(name);
        if (_resolveDimVar?.Invoke(expr) is { } resolved)
        {
            if (resolved.FixedValue is { } fixedValue)
            {
                return Const(fixedValue);
            }

            return resolved with
            {
                PythonExpression = formattedName,
                TritonExpression = formattedName,
                Equivalence = PyNTTDimEquivalence.FromAtom(formattedName),
            };
        }

        return WithRangeFromMetadata(
            new PyNTTDimExpression(formattedName, formattedName)
            {
                Equivalence = PyNTTDimEquivalence.FromAtom(formattedName),
            },
            expr);
    }

    protected override PyNTTDimExpression VisitAsDim(AsDim expr)
        => EmitScalarExpression(expr.Dim);

    protected override PyNTTDimExpression VisitThreadIdDim(ThreadIdDim expr)
        => new(_threadIdExpression, _threadIdExpression)
        {
            Equivalence = PyNTTDimEquivalence.FromAtom(_threadIdExpression),
        };

    private PyNTTDimExpression EmitScalarExpression(BaseExpr expr)
    {
        return expr switch
        {
            Dimension dimension => Visit(dimension),
            TensorConst tensorConst when tensorConst.Value.Shape.IsScalar => FormatScalarConst(tensorConst.Value.ToScalar<long>()),
            Call call => EmitScalarCall(call),
            _ => throw new NotSupportedException($"Unsupported PyNTT dimension scalar expression: {expr.GetType().Name}."),
        };
    }

    private PyNTTDimExpression EmitScalarCall(Call call)
    {
        var args = call.Arguments.ToArray();
        return call.Target switch
        {
            AsTensor when args.Length == 1 => EmitScalarExpression(args[0]),
            LocalShardDim localShardDim when args.Length == 1 => EmitLocalShardDim(localShardDim, args[0]),
            _ => throw new NotSupportedException($"Unsupported PyNTT dimension scalar call target: {call.Target.GetType().Name}."),
        };
    }

    private PyNTTDimExpression EmitLocalShardDim(LocalShardDim op, BaseExpr dimExpr)
    {
        var globalDim = EmitScalarExpression(dimExpr);
        if (op.AxisPolicy is not SBPSplit split || split.Axes.Count == 0)
        {
            return globalDim;
        }

        var axes = split.Axes.ToArray();
        var hierarchy = op.Placement.Hierarchy.ToArray();
        foreach (var axis in axes)
        {
            if (axis < 0 || axis >= hierarchy.Length)
            {
                throw new NotSupportedException($"PyNTT LocalShardDim split axis {axis} is outside placement rank {hierarchy.Length}.");
            }
        }

        var shardCount = axes.Select(axis => hierarchy[axis]).Aggregate(1, checked((lhs, rhs) => lhs * rhs));
        if (shardCount == 1)
        {
            return globalDim;
        }

        var localDim = split.Granularity is { } granularity
            ? Visit(granularity)
            : CeilDiv(globalDim, Const(shardCount));
        var shardOffset = Multiply(BuildSubShardLinearIndex(axes, hierarchy), localDim);
        if (CanUseFullLocalDim(globalDim, localDim, shardCount))
        {
            return localDim;
        }

        return Maximum(PyNTTDimExpression.Zero, Minimum(localDim, Subtract(globalDim, shardOffset)));
    }

    private static PyNTTDimExpression BuildSubShardLinearIndex(IReadOnlyList<int> axes, IReadOnlyList<int> hierarchy)
    {
        var parts = new List<PyNTTDimExpression>();
        for (var i = 0; i < axes.Count; i++)
        {
            var stride = 1;
            for (var j = i + 1; j < axes.Count; j++)
            {
                stride = checked(stride * hierarchy[axes[j]]);
            }

            var coord = BuildShardCoordExpression(axes[i], hierarchy);
            parts.Add(stride == 1 ? coord : Multiply(Const(stride), coord));
        }

        return BuildBinaryChain(parts, "+", 0);
    }

    private static PyNTTDimExpression BuildShardCoordExpression(int axis, IReadOnlyList<int> hierarchy)
    {
        var divisor = 1;
        for (var i = axis + 1; i < hierarchy.Count; i++)
        {
            divisor = checked(divisor * hierarchy[i]);
        }

        var dividend = divisor == 1
            ? "shard_index"
            : $"(shard_index // {divisor.ToString(CultureInfo.InvariantCulture)})";
        var extent = hierarchy[axis];
        if (extent == 1)
        {
            return PyNTTDimExpression.Zero;
        }

        var expression = $"(({dividend}) % {extent.ToString(CultureInfo.InvariantCulture)})";
        return new(expression, expression)
        {
            Equivalence = PyNTTDimEquivalence.FromAtom(expression),
        };
    }

    private static bool CanUseFullLocalDim(PyNTTDimExpression globalDim, PyNTTDimExpression localDim, int shardCount)
    {
        if (!globalDim.FixedValue.HasValue || !localDim.FixedValue.HasValue)
        {
            return false;
        }

        var globalValue = globalDim.FixedValue.Value;
        var localValue = localDim.FixedValue.Value;
        return localValue > 0 && globalValue >= localValue * shardCount && globalValue % localValue == 0;
    }

    private static PyNTTDimExpression BuildBinaryChain(IReadOnlyList<PyNTTDimExpression> parts, string op, long identity)
    {
        if (op == "*" && parts.Any(part => part.FixedValue == 0))
        {
            return Const(0);
        }

        var effectiveParts = op switch
        {
            "+" => parts.Where(part => part.FixedValue != 0).ToArray(),
            "*" => parts.Where(part => part.FixedValue != 1).ToArray(),
            _ => parts.ToArray(),
        };
        if (effectiveParts.Length == 0)
        {
            return Const(identity);
        }

        if (effectiveParts.Length == 1)
        {
            return effectiveParts[0];
        }

        long? fixedValue = null;
        if (effectiveParts.All(part => part.FixedValue.HasValue))
        {
            fixedValue = op == "*"
                ? effectiveParts.Aggregate(1L, (value, part) => checked(value * part.FixedValue!.Value))
                : effectiveParts.Aggregate(0L, (value, part) => checked(value + part.FixedValue!.Value));
        }

        long? rangeMin = null;
        long? rangeMax = null;
        if (op == "+")
        {
            rangeMin = effectiveParts.All(part => part.MinValue.HasValue)
                ? effectiveParts.Aggregate(0L, (value, part) => checked(value + part.MinValue!.Value))
                : null;
            rangeMax = effectiveParts.All(part => part.MaxValue.HasValue)
                ? effectiveParts.Aggregate(0L, (value, part) => checked(value + part.MaxValue!.Value))
                : null;
        }
        else if (op == "*" && effectiveParts.All(part => part.MinValue.HasValue && part.MaxValue.HasValue))
        {
            var intervalMin = 1L;
            var intervalMax = 1L;
            foreach (var part in effectiveParts)
            {
                var products = new[]
                {
                    checked(intervalMin * part.MinValue!.Value),
                    checked(intervalMin * part.MaxValue!.Value),
                    checked(intervalMax * part.MinValue!.Value),
                    checked(intervalMax * part.MaxValue!.Value),
                };
                intervalMin = products.Min();
                intervalMax = products.Max();
            }

            rangeMin = intervalMin;
            rangeMax = intervalMax;
        }

        var expression = new PyNTTDimExpression(
            $"({string.Join($" {op} ", effectiveParts.Select(part => part.PythonExpression))})",
            $"({string.Join($" {op} ", effectiveParts.Select(part => part.TritonExpression))})",
            fixedValue,
            rangeMin,
            rangeMax);
        return WithBinaryEquivalence(expression, effectiveParts, op);
    }

    private static PyNTTDimExpression Const(long value)
    {
        var text = value.ToString(CultureInfo.InvariantCulture);
        return new(text, text, value)
        {
            Equivalence = PyNTTDimEquivalence.FromConstant(value),
        };
    }

    private static PyNTTDimExpression FormatScalarConst(long value) => Const(value);

    private static PyNTTDimExpression WithRangeFromMetadata(PyNTTDimExpression expression, BaseExpr source)
    {
        if (expression.FixedValue.HasValue || source.Metadata.Range is not { } range)
        {
            return expression;
        }

        if (!double.IsFinite(range.Min) ||
            !double.IsFinite(range.Max) ||
            range.Min < long.MinValue ||
            range.Min > long.MaxValue ||
            range.Max < long.MinValue ||
            range.Max > long.MaxValue)
        {
            return expression;
        }

        var metadataMin = checked((long)Math.Floor(range.Min));
        var metadataMax = checked((long)Math.Ceiling(range.Max));
        var rangeMin = expression.RangeMin.HasValue
            ? Math.Max(expression.RangeMin.Value, metadataMin)
            : metadataMin;
        var rangeMax = expression.RangeMax.HasValue
            ? Math.Min(expression.RangeMax.Value, metadataMax)
            : metadataMax;
        if (rangeMin > rangeMax)
        {
            throw new InvalidOperationException(
                $"Dimension range inference is inconsistent for {source}: " +
                $"derived=[{expression.RangeMin}, {expression.RangeMax}], metadata=[{metadataMin}, {metadataMax}].");
        }

        return expression with
        {
            RangeMin = rangeMin,
            RangeMax = rangeMax,
        };
    }

    private static PyNTTDimExpression Add(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
        => BuildBinaryChain([lhs, rhs], "+", 0);

    private static PyNTTDimExpression Subtract(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
    {
        if (rhs.FixedValue == 0)
        {
            return lhs;
        }

        if (lhs.FixedValue.HasValue && rhs.FixedValue.HasValue)
        {
            return Const(checked(lhs.FixedValue.Value - rhs.FixedValue.Value));
        }

        long? rangeMin = lhs.MinValue.HasValue && rhs.MaxValue.HasValue
            ? checked(lhs.MinValue.Value - rhs.MaxValue.Value)
            : null;
        long? rangeMax = lhs.MaxValue.HasValue && rhs.MinValue.HasValue
            ? checked(lhs.MaxValue.Value - rhs.MinValue.Value)
            : null;
        var expression = new PyNTTDimExpression(
            $"(({lhs.PythonExpression}) - ({rhs.PythonExpression}))",
            $"(({lhs.TritonExpression}) - ({rhs.TritonExpression}))",
            null,
            rangeMin,
            rangeMax);
        return lhs.Equivalence is { } lhsEquivalence && rhs.Equivalence is { } rhsEquivalence
            ? PyNTTDimEquivalence.TrySubtract(lhsEquivalence, rhsEquivalence) is { } equivalence
                ? expression with { Equivalence = equivalence }
                : expression.EnsureEquivalence()
            : expression.EnsureEquivalence();
    }

    private static PyNTTDimExpression Multiply(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
        => BuildBinaryChain([lhs, rhs], "*", 1);

    private static PyNTTDimExpression CeilDiv(PyNTTDimExpression numerator, PyNTTDimExpression denominator)
    {
        if (numerator.FixedValue.HasValue && denominator.FixedValue.HasValue)
        {
            var denominatorValue = denominator.FixedValue.Value;
            return Const(checked((numerator.FixedValue.Value + denominatorValue - 1) / denominatorValue));
        }

        long? rangeMin = numerator.MinValue.HasValue && denominator.FixedValue is > 0
            ? checked((numerator.MinValue.Value + denominator.FixedValue.Value - 1) / denominator.FixedValue.Value)
            : null;
        long? rangeMax = numerator.MaxValue.HasValue && denominator.FixedValue is > 0
            ? checked((numerator.MaxValue.Value + denominator.FixedValue.Value - 1) / denominator.FixedValue.Value)
            : null;
        return new PyNTTDimExpression(
            $"(({numerator.PythonExpression} + {denominator.PythonExpression} - 1) // ({denominator.PythonExpression}))",
            $"(({numerator.TritonExpression} + {denominator.TritonExpression} - 1) // ({denominator.TritonExpression}))",
            null,
            rangeMin,
            rangeMax)
            .EnsureEquivalence();
    }

    private static PyNTTDimExpression Minimum(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
    {
        if (lhs.FixedValue.HasValue && rhs.FixedValue.HasValue)
        {
            return Const(Math.Min(lhs.FixedValue.Value, rhs.FixedValue.Value));
        }

        var minValues = new[] { lhs.MinValue, rhs.MinValue }.OfType<long>().ToArray();
        var maxValues = new[] { lhs.MaxValue, rhs.MaxValue }.OfType<long>().ToArray();
        long? rangeMin = minValues.Length == 2 ? minValues.Min() : null;
        long? rangeMax = maxValues.Length > 0 ? maxValues.Min() : null;
        return new PyNTTDimExpression(
            $"min({lhs.PythonExpression}, {rhs.PythonExpression})",
            $"tl.minimum({lhs.TritonExpression}, {rhs.TritonExpression})",
            null,
            rangeMin,
            rangeMax)
            .EnsureEquivalence();
    }

    private static PyNTTDimExpression Maximum(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
    {
        if (lhs.FixedValue.HasValue && rhs.FixedValue.HasValue)
        {
            return Const(Math.Max(lhs.FixedValue.Value, rhs.FixedValue.Value));
        }

        var minValues = new[] { lhs.MinValue, rhs.MinValue }.OfType<long>().ToArray();
        var maxValues = new[] { lhs.MaxValue, rhs.MaxValue }.OfType<long>().ToArray();
        long? rangeMin = minValues.Length > 0 ? minValues.Max() : null;
        long? rangeMax = maxValues.Length == 2 ? maxValues.Max() : null;
        return new PyNTTDimExpression(
            $"max({lhs.PythonExpression}, {rhs.PythonExpression})",
            $"tl.maximum({lhs.TritonExpression}, {rhs.TritonExpression})",
            null,
            rangeMin,
            rangeMax)
            .EnsureEquivalence();
    }

    private static PyNTTDimExpression WithBinaryEquivalence(
        PyNTTDimExpression expression,
        IReadOnlyList<PyNTTDimExpression> parts,
        string op)
    {
        if (parts.Any(part => part.Equivalence is null))
        {
            return expression.EnsureEquivalence();
        }

        if (op == "+")
        {
            PyNTTDimEquivalence? equivalence = PyNTTDimEquivalence.FromConstant(0);
            foreach (var part in parts)
            {
                equivalence = PyNTTDimEquivalence.TryAdd(equivalence!, part.Equivalence!);
                if (equivalence is null)
                {
                    return expression.EnsureEquivalence();
                }
            }

            return expression with { Equivalence = equivalence };
        }

        if (op == "*")
        {
            var dynamicParts = parts.Where(part => !part.FixedValue.HasValue).ToArray();
            if (dynamicParts.Length <= 1)
            {
                var scale = parts
                    .Where(part => part.FixedValue.HasValue)
                    .Aggregate(1L, (value, part) => checked(value * part.FixedValue!.Value));
                var equivalence = dynamicParts.Length == 0
                    ? PyNTTDimEquivalence.FromConstant(scale)
                    : PyNTTDimEquivalence.Scale(dynamicParts[0].Equivalence!, scale);
                return expression with { Equivalence = equivalence };
            }
        }

        return expression.EnsureEquivalence();
    }

    private static string CompareOpToPython(CompareOp compareOp) => compareOp switch
    {
        CompareOp.Equal => "==",
        CompareOp.NotEqual => "!=",
        CompareOp.LowerThan => "<",
        CompareOp.LowerOrEqual => "<=",
        CompareOp.GreaterThan => ">",
        CompareOp.GreaterOrEqual => ">=",
        _ => throw new NotSupportedException($"Unsupported PyNTT dimension compare op: {compareOp}."),
    };

    private static string SanitizePythonIdentifier(string value)
    {
        var chars = value.Select(ch => char.IsAsciiLetterOrDigit(ch) || ch == '_' ? ch : '_').ToArray();
        if (chars.Length == 0 || char.IsDigit(chars[0]))
        {
            return "_" + new string(chars);
        }

        return new string(chars);
    }
}

public static class PyNTTTemplateUtility
{
    public static int SelectBlockAxis(IReadOnlyList<PyNTTDimExpression> shape, IReadOnlyList<PyNTTDimExpression> strides)
    {
        if (shape.Count == 0)
        {
            return 0;
        }

        for (var i = shape.Count - 1; i >= 0; i--)
        {
            if (!shape[i].IsFixedOne && strides[i].IsFixedOne)
            {
                return i;
            }
        }

        for (var i = shape.Count - 1; i >= 0; i--)
        {
            if (!shape[i].IsFixedOne)
            {
                return i;
            }
        }

        return shape.Count - 1;
    }

    public static PyNTTDimExpression[] ContiguousStrides(IReadOnlyList<PyNTTDimExpression> shape)
    {
        var strides = new PyNTTDimExpression[shape.Count];
        var stride = PyNTTDimExpression.One;
        for (var i = shape.Count - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride = Multiply(stride, shape[i]);
        }

        return strides;
    }

    public static string ShapeTuple(IReadOnlyList<PyNTTDimExpression> shape)
        => "(" + string.Join(", ", shape.Select(dim => dim.TritonExpression)) + (shape.Count == 1 ? "," : string.Empty) + ")";

    public static string RuntimeParameterSuffix(IReadOnlyList<string> runtimeShapeArgs)
    {
        if (runtimeShapeArgs.Count == 0)
        {
            return ", block_size: tl.constexpr";
        }

        return ", " + string.Join(", ", runtimeShapeArgs) + ", block_size: tl.constexpr";
    }

    public static long RequireFixed(PyNTTDimExpression dimension, string context)
        => dimension.FixedValue ?? throw new NotSupportedException($"{context} must be fixed for the current PyNTT Triton template, got {dimension.PythonExpression}.");

    private static PyNTTDimExpression Multiply(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
    {
        if (lhs.FixedValue == 0 || rhs.FixedValue == 0)
        {
            return PyNTTDimExpression.Zero;
        }

        if (lhs.FixedValue == 1)
        {
            return rhs;
        }

        if (rhs.FixedValue == 1)
        {
            return lhs;
        }

        long? fixedValue = lhs.FixedValue.HasValue && rhs.FixedValue.HasValue
            ? checked(lhs.FixedValue.Value * rhs.FixedValue.Value)
            : null;
        return new($"({lhs.PythonExpression} * {rhs.PythonExpression})", $"({lhs.TritonExpression} * {rhs.TritonExpression})", fixedValue);
    }
}
