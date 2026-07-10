// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Reactive;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;

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
        return WithRangeFromMetadata(expression, dimension);
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
            var expression = new PyNTTDimExpression(
                $"(({numerator.PythonExpression}) // ({denominator.PythonExpression}))",
                $"(({numerator.TritonExpression}) // ({denominator.TritonExpression}))");
            return WithRangeFromMetadata(expression, expr);
        }

        var ceilExpression = new PyNTTDimExpression(
            $"(({numerator.PythonExpression} + {denominator.PythonExpression} - 1) // ({denominator.PythonExpression}))",
            $"(({numerator.TritonExpression} + {denominator.TritonExpression} - 1) // ({denominator.TritonExpression}))");
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
        return WithRangeFromMetadata(BuildBinaryChain(parts, "*", 1), expr);
    }

    protected override PyNTTDimExpression VisitDimRemainder(DimRemainder expr)
    {
        var numerator = Visit(expr.Numerator);
        var denominator = Visit(expr.Denominator);
        var expression = new PyNTTDimExpression(
            $"(({numerator.PythonExpression}) % ({denominator.PythonExpression}))",
            $"(({numerator.TritonExpression}) % ({denominator.TritonExpression}))");
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
        return WithRangeFromMetadata(new(formattedName, formattedName), expr);
    }

    protected override PyNTTDimExpression VisitAsDim(AsDim expr)
        => EmitScalarExpression(expr.Dim);

    protected override PyNTTDimExpression VisitThreadIdDim(ThreadIdDim expr)
        => new(_threadIdExpression, _threadIdExpression);

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
        return new(expression, expression);
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

        long? rangeMin = null;
        long? rangeMax = null;
        if (op == "+")
        {
            rangeMin = parts.All(part => part.MinValue.HasValue)
                ? parts.Aggregate(0L, (value, part) => checked(value + part.MinValue!.Value))
                : null;
            rangeMax = parts.All(part => part.MaxValue.HasValue)
                ? parts.Aggregate(0L, (value, part) => checked(value + part.MaxValue!.Value))
                : null;
        }
        else if (op == "*" && parts.All(part => part.MinValue.HasValue && part.MaxValue.HasValue && part.MinValue.Value >= 0))
        {
            rangeMin = parts.Aggregate(1L, (value, part) => checked(value * part.MinValue!.Value));
            rangeMax = parts.Aggregate(1L, (value, part) => checked(value * part.MaxValue!.Value));
        }

        return new(
            $"({string.Join($" {op} ", parts.Select(part => part.PythonExpression))})",
            $"({string.Join($" {op} ", parts.Select(part => part.TritonExpression))})",
            fixedValue,
            rangeMin,
            rangeMax);
    }

    private static PyNTTDimExpression Const(long value)
    {
        var text = value.ToString(CultureInfo.InvariantCulture);
        return new(text, text, value);
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

        return expression with
        {
            RangeMin = checked((long)Math.Floor(range.Min)),
            RangeMax = checked((long)Math.Ceiling(range.Max)),
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
        return new(
            $"(({lhs.PythonExpression}) - ({rhs.PythonExpression}))",
            $"(({lhs.TritonExpression}) - ({rhs.TritonExpression}))",
            null,
            rangeMin,
            rangeMax);
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
        return new(
            $"(({numerator.PythonExpression} + {denominator.PythonExpression} - 1) // ({denominator.PythonExpression}))",
            $"(({numerator.TritonExpression} + {denominator.TritonExpression} - 1) // ({denominator.TritonExpression}))",
            null,
            rangeMin,
            rangeMax);
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
        return new(
            $"min({lhs.PythonExpression}, {rhs.PythonExpression})",
            $"tl.minimum({lhs.TritonExpression}, {rhs.TritonExpression})",
            null,
            rangeMin,
            rangeMax);
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
        return new(
            $"max({lhs.PythonExpression}, {rhs.PythonExpression})",
            $"tl.maximum({lhs.TritonExpression}, {rhs.TritonExpression})",
            null,
            rangeMin,
            rangeMax);
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
