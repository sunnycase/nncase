// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="ScaledMatMul"/>.
/// </summary>
public sealed class ScaledMatMulEvaluator :
    IEvaluator<ScaledMatMul>,
    ITypeInferencer<ScaledMatMul>,
    ICostEvaluator<ScaledMatMul>
{
    public IValue Visit(IEvaluateContext context, ScaledMatMul target)
    {
        var lhs = context.GetArgumentValueAsTensor(target, ScaledMatMul.Lhs);
        var rhs = context.GetArgumentValueAsTensor(target, ScaledMatMul.Rhs);
        var lhsScale = context.GetArgumentValueAsTensor(target, ScaledMatMul.LhsScale);
        var rhsScale = context.GetArgumentValueAsTensor(target, ScaledMatMul.RhsScale);
        return Evaluate(lhs, rhs, lhsScale, rhsScale, target.OutputDataType);
    }

    public static IValue Evaluate(
        Tensor lhs,
        Tensor rhs,
        Tensor lhsScale,
        Tensor rhsScale,
        DataType outputDataType)
    {
        if (ValidateDataTypes(lhs.ElementType, rhs.ElementType, outputDataType) is { } typeError)
        {
            throw new InvalidOperationException(typeError.Reason);
        }
        var lhsScaleValue = GetScaleValue(lhsScale, "lhs");
        var rhsScaleValue = GetScaleValue(rhsScale, "rhs");
        if (!float.IsFinite(lhsScaleValue) || lhsScaleValue <= 0F ||
            !float.IsFinite(rhsScaleValue) || rhsScaleValue <= 0F)
        {
            throw new InvalidOperationException(
                $"ScaledMatMul scales must be finite and positive, got lhs={lhsScaleValue}, rhs={rhsScaleValue}.");
        }

        var lhsFloat = lhs.CastElementTo(DataTypes.Float32).ToOrtTensor();
        var quantizedLhs = OrtKI.Div(lhsFloat, lhsScaleValue)
            .ToTensor()
            .CastElementTo(rhs.ElementType)
            .CastElementTo(DataTypes.Float32)
            .ToOrtTensor();
        var rhsFloat = rhs.CastElementTo(DataTypes.Float32).ToOrtTensor();
        var result = OrtKI.MatMul(quantizedLhs, rhsFloat);
        result = OrtKI.Mul(result, lhsScaleValue * rhsScaleValue);
        return Value.FromTensor(result.ToTensor().CastElementTo(outputDataType));
    }

    public IRType Visit(ITypeInferenceContext context, ScaledMatMul target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, ScaledMatMul.Lhs),
            context.CheckArgumentType<IRType>(target, ScaledMatMul.Rhs),
            context.CheckArgumentType<IRType>(target, ScaledMatMul.LhsScale),
            context.CheckArgumentType<IRType>(target, ScaledMatMul.RhsScale));

    public static IRType InferType(
        ScaledMatMul target,
        IRType lhs,
        IRType rhs,
        IRType lhsScale,
        IRType rhsScale)
    {
        if (ValidateScaleType(lhsScale, "lhs") is { } lhsScaleError)
        {
            return lhsScaleError;
        }

        if (ValidateScaleType(rhsScale, "rhs") is { } rhsScaleError)
        {
            return rhsScaleError;
        }

        return (lhs, rhs) switch
        {
            (TensorType lhsTensor, TensorType rhsTensor) => InferTensorType(target, lhsTensor, rhsTensor),
            (DistributedType lhsDistributed, DistributedType rhsDistributed) =>
                InferDistributedType(target, lhsDistributed, rhsDistributed),
            _ => new InvalidType($"ScaledMatMul expects tensor operands, got lhs={lhs}, rhs={rhs}."),
        };
    }

    public Cost Visit(ICostEvaluateContext context, ScaledMatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, ScaledMatMul.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, ScaledMatMul.Rhs);
        var lhsScale = context.GetArgumentType<IRType>(target, ScaledMatMul.LhsScale);
        var rhsScale = context.GetArgumentType<IRType>(target, ScaledMatMul.RhsScale);
        var output = context.GetReturnType<IRType>();
        if (TargetCostTensor.TryFromType(lhs, out var lhsTensor) &&
            TargetCostTensor.TryFromType(rhs, out var rhsTensor) &&
            TargetCostTensor.TryFromType(output, out var outputTensor) &&
            context.TargetCostModel.TryGetMatMulCost(
                new(lhsTensor, rhsTensor, outputTensor, target.OutputDataType),
                out var cost))
        {
            Add(cost, CostFactorNames.BlockLocalMemoryLoadBytes, CostUtility.GetMemoryAccess(lhsScale));
            Add(cost, CostFactorNames.BlockLocalMemoryLoadBytes, CostUtility.GetMemoryAccess(rhsScale));
            Add(cost, CostFactorNames.CPUCycles, CostUtility.GetCPUCycles(lhs, 2));
            return cost;
        }

        var k = GetReductionExtent(lhs);
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs) +
                CostUtility.GetMemoryAccess(lhsScale) + CostUtility.GetMemoryAccess(rhsScale),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(output, checked(k + 2U)),
        };
    }

    public static bool IsScaleType(IRType type) => ValidateScaleType(type, "scale") is null;

    public static TensorType? GetTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => distributed.TensorType,
        _ => null,
    };

    private static IRType InferTensorType(ScaledMatMul target, TensorType lhs, TensorType rhs)
    {
        if (ValidateDataTypes(lhs.DType, rhs.DType, target.OutputDataType) is { } error)
        {
            return error;
        }

        return MatMulEvaluator.VisitTensorType(
            lhs,
            rhs with { DType = lhs.DType },
            NoneType.Default,
            outputDataType: target.OutputDataType);
    }

    private static IRType InferDistributedType(
        ScaledMatMul target,
        DistributedType lhs,
        DistributedType rhs)
    {
        if (ValidateDataTypes(lhs.TensorType.DType, rhs.TensorType.DType, target.OutputDataType) is { } error)
        {
            return error;
        }

        return MatMulEvaluator.VisitDistributedType(
            lhs,
            rhs with { TensorType = rhs.TensorType with { DType = lhs.TensorType.DType } },
            NoneType.Default,
            outputDataType: target.OutputDataType);
    }

    private static InvalidType? ValidateDataTypes(
        DataType lhsType,
        DataType rhsType,
        DataType outputType)
    {
        if (lhsType is VectorType || rhsType is VectorType)
        {
            return new InvalidType("ScaledMatMul high-level operands must have scalar element types.");
        }

        if (lhsType is not PrimType || !lhsType.IsFloat() ||
            rhsType != DataTypes.Float8E4M3 && rhsType != DataTypes.Float8E5M2)
        {
            return new InvalidType(
                $"ScaledMatMul requires a floating lhs and an E4M3/E5M2 rhs, got {lhsType}/{rhsType}.");
        }

        return outputType is PrimType && outputType.IsFloat()
            ? null
            : new InvalidType($"ScaledMatMul output type must be floating, got {outputType}.");
    }

    private static InvalidType? ValidateScaleType(IRType type, string name)
    {
        var tensor = GetTensorType(type);
        if (tensor is null ||
            tensor.DType is not PrimType scaleType ||
            !scaleType.IsFloat() ||
            tensor.Shape is not RankedShape shape)
        {
            return new InvalidType(
                $"ScaledMatMul {name} scale must be a ranked floating-point tensor, got {type}.");
        }

        if (shape.Rank == 0 || shape.IsFixed && shape.Aggregate(1L, (value, dim) => checked(value * dim.FixedValue)) == 1L)
        {
            return null;
        }

        return new InvalidType(
            $"ScaledMatMul {name} scale must contain exactly one element, got shape {shape}.");
    }

    private static float GetScaleValue(Tensor scale, string name)
    {
        var values = scale.CastElementTo(DataTypes.Float32).ToArray<float>();
        if (values.Length != 1)
        {
            throw new InvalidOperationException(
                $"ScaledMatMul {name} scale must contain exactly one element, got {values.Length}.");
        }

        return values[0];
    }

    private static uint GetReductionExtent(IRType lhs)
    {
        var tensor = lhs is DistributedType distributed
            ? DistributedUtility.GetDividedTensorType(distributed)
            : GetTensorType(lhs);
        return tensor?.Shape is RankedShape { Rank: > 0 } shape && shape[^1].IsFixed
            ? checked((uint)shape[^1].FixedValue)
            : 1U;
    }

    private static void Add(Cost cost, string name, UInt128 value)
    {
        if (value == 0)
        {
            return;
        }

        cost.Factors[name] = cost.Factors.TryGetValue(name, out var oldValue)
            ? oldValue + value
            : value;
    }
}
