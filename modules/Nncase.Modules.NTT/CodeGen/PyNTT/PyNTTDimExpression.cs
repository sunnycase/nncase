// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Reactive;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Shapes;

namespace Nncase.CodeGen.PyNTT;

public sealed record PyNTTDimExpression(string PythonExpression, string TritonExpression, long? FixedValue = null, long? RangeMin = null, long? RangeMax = null)
{
    public static PyNTTDimExpression Zero { get; } = new("0", "0", 0);

    public static PyNTTDimExpression One { get; } = new("1", "1", 1);

    public bool IsFixed => FixedValue.HasValue;

    public bool IsFixedOne => FixedValue == 1;

    public bool IsFixedNonOne => FixedValue.HasValue && FixedValue.Value != 1;

    public long? MinValue => FixedValue ?? RangeMin;

    public long? MaxValue => FixedValue ?? RangeMax;

    public object ToPythonLiteral() => FixedValue.HasValue ? FixedValue.Value : PythonExpression;

    public override string ToString() => TritonExpression;
}

internal sealed class PyNTTDimExpressionEmitter : ExprFunctor<PyNTTDimExpression, Unit>
{
    private readonly Action<string>? _registerRuntimeScalar;
    private readonly Func<string, string> _formatRuntimeScalar;
    private readonly string _threadIdExpression;

    public PyNTTDimExpressionEmitter(Action<string>? registerRuntimeScalar = null, Func<string, string>? formatRuntimeScalar = null, string? threadIdExpression = null)
    {
        _registerRuntimeScalar = registerRuntimeScalar;
        _formatRuntimeScalar = formatRuntimeScalar ?? (name => name);
        _threadIdExpression = threadIdExpression ?? "pyntt_thread_id";
    }

    public PyNTTDimExpression Emit(Dimension dimension)
    {
        var expression = Visit(dimension);
        if (expression.FixedValue.HasValue || dimension.Metadata.Range is not { } range)
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

        return expression with
        {
            RangeMin = checked((long)Math.Floor(range.Min)),
            RangeMax = checked((long)Math.Ceiling(range.Max)),
        };
    }

    protected override PyNTTDimExpression VisitDimAbs(DimAbs expr)
    {
        var operand = Visit(expr.Operand);
        return new($"abs({operand.PythonExpression})", $"tl.abs({operand.TritonExpression})");
    }

    protected override PyNTTDimExpression VisitDimClamp(DimClamp expr)
    {
        var operand = Visit(expr.Operand);
        var min = Visit(expr.MinValue);
        var max = Visit(expr.MaxValue);
        return new(
            $"min(max({operand.PythonExpression}, {min.PythonExpression}), {max.PythonExpression})",
            $"tl.minimum(tl.maximum({operand.TritonExpression}, {min.TritonExpression}), {max.TritonExpression})");
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
        return new(
            $"({trueValue.PythonExpression} if ({predicate}) else {falseValue.PythonExpression})",
            $"tl.where({tritonPredicate}, {trueValue.TritonExpression}, {falseValue.TritonExpression})");
    }

    protected override PyNTTDimExpression VisitDimConst(DimConst expr)
    {
        var value = expr.Value.ToString(CultureInfo.InvariantCulture);
        return new(value, value, expr.Value);
    }

    protected override PyNTTDimExpression VisitDimFraction(DimFraction expr)
    {
        var numerator = Visit(expr.Numerator);
        var denominator = Visit(expr.Denominator);
        if (expr.DivMode == DimDivideMode.FloorDiv)
        {
            return new(
                $"(({numerator.PythonExpression}) // ({denominator.PythonExpression}))",
                $"(({numerator.TritonExpression}) // ({denominator.TritonExpression}))");
        }

        return new(
            $"(({numerator.PythonExpression} + {denominator.PythonExpression} - 1) // ({denominator.PythonExpression}))",
            $"(({numerator.TritonExpression} + {denominator.TritonExpression} - 1) // ({denominator.TritonExpression}))");
    }

    protected override PyNTTDimExpression VisitDimMax(DimMax expr)
    {
        var operands = expr.Operands.ToArray().Select(Visit).ToArray();
        if (operands.Length == 1)
        {
            return operands[0];
        }

        var python = $"max({string.Join(", ", operands.Select(operand => operand.PythonExpression))})";
        var triton = operands.Skip(1).Aggregate(
            operands[0].TritonExpression,
            (current, operand) => $"tl.maximum({current}, {operand.TritonExpression})");
        return new(python, triton);
    }

    protected override PyNTTDimExpression VisitDimMin(DimMin expr)
    {
        var operands = expr.Operands.ToArray().Select(Visit).ToArray();
        if (operands.Length == 1)
        {
            return operands[0];
        }

        var python = $"min({string.Join(", ", operands.Select(operand => operand.PythonExpression))})";
        var triton = operands.Skip(1).Aggregate(
            operands[0].TritonExpression,
            (current, operand) => $"tl.minimum({current}, {operand.TritonExpression})");
        return new(python, triton);
    }

    protected override PyNTTDimExpression VisitDimPositive(DimPositive expr)
    {
        var operand = Visit(expr.Operand);
        var extent = Visit(expr.Extent);
        return new(
            $"(({operand.PythonExpression}) if ({operand.PythonExpression} >= 0) else ({operand.PythonExpression} + {extent.PythonExpression}))",
            $"tl.where({operand.TritonExpression} >= 0, {operand.TritonExpression}, {operand.TritonExpression} + {extent.TritonExpression})");
    }

    protected override PyNTTDimExpression VisitDimPower(DimPower expr)
    {
        var dim = Visit((Dimension)expr.Dim);
        return new(
            $"(({dim.PythonExpression}) ** {expr.Power.ToString(CultureInfo.InvariantCulture)})",
            $"(({dim.TritonExpression}) ** {expr.Power.ToString(CultureInfo.InvariantCulture)})");
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
        return BuildBinaryChain(parts, "*", 1);
    }

    protected override PyNTTDimExpression VisitDimRemainder(DimRemainder expr)
    {
        var numerator = Visit(expr.Numerator);
        var denominator = Visit(expr.Denominator);
        return new(
            $"(({numerator.PythonExpression}) % ({denominator.PythonExpression}))",
            $"(({numerator.TritonExpression}) % ({denominator.TritonExpression}))");
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
        return BuildBinaryChain(parts, "+", 0);
    }

    protected override PyNTTDimExpression VisitDimVar(DimVar expr)
    {
        var name = SanitizePythonIdentifier(expr.Name);
        _registerRuntimeScalar?.Invoke(name);
        var formattedName = _formatRuntimeScalar(name);
        return new(formattedName, formattedName);
    }

    protected override PyNTTDimExpression VisitThreadIdDim(ThreadIdDim expr)
        => new(_threadIdExpression, _threadIdExpression);

    private static PyNTTDimExpression BuildBinaryChain(IReadOnlyList<PyNTTDimExpression> parts, string op, long identity)
    {
        if (parts.Count == 0)
        {
            var text = identity.ToString(CultureInfo.InvariantCulture);
            return new(text, text, identity);
        }

        if (parts.Count == 1)
        {
            return parts[0];
        }

        long? fixedValue = null;
        if (parts.All(part => part.FixedValue.HasValue))
        {
            fixedValue = op == "*"
                ? parts.Aggregate(1L, (value, part) => checked(value * part.FixedValue!.Value))
                : parts.Aggregate(0L, (value, part) => checked(value + part.FixedValue!.Value));
        }

        return new(
            $"({string.Join($" {op} ", parts.Select(part => part.PythonExpression))})",
            $"({string.Join($" {op} ", parts.Select(part => part.TritonExpression))})",
            fixedValue);
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
