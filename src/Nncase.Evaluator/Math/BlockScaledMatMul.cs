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
/// Evaluator for <see cref="BlockScaledMatMul"/>.
/// </summary>
public sealed class BlockScaledMatMulEvaluator :
    IEvaluator<BlockScaledMatMul>,
    ITypeInferencer<BlockScaledMatMul>,
    ICostEvaluator<BlockScaledMatMul>
{
    public IValue Visit(IEvaluateContext context, BlockScaledMatMul target)
        => Evaluate(
            context.GetArgumentValueAsTensor(target, BlockScaledMatMul.Lhs),
            context.GetArgumentValueAsTensor(target, BlockScaledMatMul.Rhs),
            context.GetArgumentValueAsTensor(target, BlockScaledMatMul.RhsScale),
            target.OutputDataType,
            target.WeightBlockN,
            target.WeightBlockK);

    public static IValue Evaluate(
        Tensor lhs,
        Tensor rhs,
        Tensor rhsScale,
        DataType outputDataType,
        long weightBlockN,
        long weightBlockK)
    {
        ValidateRuntimeContract(lhs, rhs, rhsScale, outputDataType, weightBlockN, weightBlockK);

        var lhsShape = lhs.Dimensions.ToArray();
        var rhsShape = rhs.Dimensions.ToArray();
        var k = checked((int)lhsShape[^1]);
        var n = checked((int)rhsShape[^1]);
        var rows = checked((int)(lhs.Length / k));
        var blockN = checked((int)weightBlockN);
        var blockK = checked((int)weightBlockK);

        var lhsValues = lhs.CastElementTo(DataTypes.Float32).ToArray<float>();
        var normalizedLhs = new float[lhsValues.Length];
        var lhsScales = new float[checked(rows * MathUtility.CeilDiv(k, blockK))];
        var kBlocks = MathUtility.CeilDiv(k, blockK);
        for (var row = 0; row < rows; row++)
        {
            for (var block = 0; block < kBlocks; block++)
            {
                var begin = block * blockK;
                var end = System.Math.Min(k, begin + blockK);
                var maxAbs = 0F;
                for (var index = begin; index < end; index++)
                {
                    maxAbs = System.Math.Max(
                        maxAbs,
                        System.Math.Abs(lhsValues[(row * k) + index]));
                }

                var scale = System.Math.Max(maxAbs, 1E-12F) / (float)Float8E4M3.MaxNormal;
                lhsScales[(row * kBlocks) + block] = scale;
                for (var index = begin; index < end; index++)
                {
                    normalizedLhs[(row * k) + index] = lhsValues[(row * k) + index] / scale;
                }
            }
        }

        var quantizedLhs = Tensor.From<float>(normalizedLhs, lhsShape)
            .CastElementTo(rhs.ElementType)
            .CastElementTo(DataTypes.Float32)
            .ToArray<float>();
        for (var row = 0; row < rows; row++)
        {
            for (var index = 0; index < k; index++)
            {
                quantizedLhs[(row * k) + index] *= lhsScales[(row * kBlocks) + (index / blockK)];
            }
        }

        var rhsValues = rhs.CastElementTo(DataTypes.Float32).ToArray<float>();
        var scaleValues = rhsScale.CastElementTo(DataTypes.Float32).ToArray<float>();
        var rhsK = checked((int)rhsShape[^2]);
        var scaleK = MathUtility.CeilDiv(rhsK, blockK);
        for (var ki = 0; ki < rhsK; ki++)
        {
            for (var ni = 0; ni < n; ni++)
            {
                var scaleIndex = (ni / blockN * scaleK) + (ki / blockK);
                rhsValues[(ki * n) + ni] *= scaleValues[scaleIndex];
            }
        }

        var lhsTensor = Tensor.From<float>(quantizedLhs, lhsShape).ToOrtTensor();
        var rhsTensor = Tensor.From<float>(rhsValues, rhsShape).ToOrtTensor();
        return Value.FromTensor(
            OrtKI.MatMul(lhsTensor, rhsTensor).ToTensor().CastElementTo(outputDataType));
    }

    public IRType Visit(ITypeInferenceContext context, BlockScaledMatMul target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, BlockScaledMatMul.Lhs),
            context.CheckArgumentType<IRType>(target, BlockScaledMatMul.Rhs),
            context.CheckArgumentType<IRType>(target, BlockScaledMatMul.RhsScale));

    public static IRType InferType(
        BlockScaledMatMul target,
        IRType lhs,
        IRType rhs,
        IRType rhsScale)
    {
        if (target.WeightBlockN <= 0 || target.WeightBlockK <= 0)
        {
            return new InvalidType(
                $"BlockScaledMatMul block dimensions must be positive, got [{target.WeightBlockN}, {target.WeightBlockK}].");
        }

        var lhsTensor = ScaledMatMulEvaluator.GetTensorType(lhs);
        var rhsTensor = ScaledMatMulEvaluator.GetTensorType(rhs);
        var scaleTensor = ScaledMatMulEvaluator.GetTensorType(rhsScale);
        if (lhsTensor is null || rhsTensor is null || scaleTensor is null)
        {
            return new InvalidType(
                $"BlockScaledMatMul expects tensor-like operands, got lhs={lhs}, rhs={rhs}, rhs_scale={rhsScale}.");
        }

        if (lhsTensor.DType is VectorType || rhsTensor.DType is VectorType ||
            lhsTensor.DType is not PrimType || !lhsTensor.DType.IsFloat() ||
            rhsTensor.DType != DataTypes.Float8E4M3)
        {
            return new InvalidType(
                $"BlockScaledMatMul requires a scalar floating lhs and E4M3 rhs, got {lhsTensor.DType}/{rhsTensor.DType}.");
        }

        if (target.OutputDataType is not PrimType outputType || !outputType.IsFloat())
        {
            return new InvalidType(
                $"BlockScaledMatMul output type must be floating, got {target.OutputDataType}.");
        }

        if (scaleTensor.DType is not PrimType scaleType || !scaleType.IsFloat() ||
            rhsTensor.Shape is not RankedShape { Rank: >= 2 } rhsShape ||
            scaleTensor.Shape is not RankedShape { Rank: 2 } scaleShape)
        {
            return new InvalidType(
                $"BlockScaledMatMul rhs scale must be a rank-2 floating tensor, got {rhsScale}.");
        }

        var expectedScaleN = Dimension.CeilDiv(rhsShape[^1], target.WeightBlockN);
        var expectedScaleK = Dimension.CeilDiv(rhsShape[^2], target.WeightBlockK);
        if (scaleShape[0] != expectedScaleN || scaleShape[1] != expectedScaleK)
        {
            return new InvalidType(
                $"BlockScaledMatMul rhs scale shape must be [{expectedScaleN}, {expectedScaleK}] " +
                $"for rhs {rhsShape}, got {scaleShape}.");
        }

        return (lhs, rhs) switch
        {
            (TensorType lhsValue, TensorType rhsValue) => MatMulEvaluator.VisitTensorType(
                lhsValue,
                rhsValue with { DType = lhsValue.DType },
                NoneType.Default,
                outputDataType: target.OutputDataType),
            (DistributedType lhsValue, DistributedType rhsValue) => MatMulEvaluator.VisitDistributedType(
                lhsValue,
                rhsValue with { TensorType = rhsValue.TensorType with { DType = lhsValue.TensorType.DType } },
                NoneType.Default,
                outputDataType: target.OutputDataType),
            _ => new InvalidType(
                $"BlockScaledMatMul lhs/rhs distribution kinds must match, got lhs={lhs}, rhs={rhs}."),
        };
    }

    public Cost Visit(ICostEvaluateContext context, BlockScaledMatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, BlockScaledMatMul.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, BlockScaledMatMul.Rhs);
        var rhsScale = context.GetArgumentType<IRType>(target, BlockScaledMatMul.RhsScale);
        var output = context.GetReturnType<IRType>();
        var k = GetReductionExtent(lhs);
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs) +
                CostUtility.GetMemoryAccess(rhsScale),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(output, checked(k + 4U)),
        };
    }

    private static void ValidateRuntimeContract(
        Tensor lhs,
        Tensor rhs,
        Tensor rhsScale,
        DataType outputDataType,
        long weightBlockN,
        long weightBlockK)
    {
        var inferred = InferType(
            new BlockScaledMatMul(outputDataType, weightBlockN, weightBlockK),
            new TensorType(lhs.ElementType, lhs.Shape),
            new TensorType(rhs.ElementType, rhs.Shape),
            new TensorType(rhsScale.ElementType, rhsScale.Shape));
        if (inferred is InvalidType invalid)
        {
            throw new InvalidOperationException(invalid.Reason);
        }
    }

    private static uint GetReductionExtent(IRType lhs)
    {
        var tensor = lhs is DistributedType distributed
            ? DistributedUtility.GetDividedTensorType(distributed)
            : ScaledMatMulEvaluator.GetTensorType(lhs);
        return tensor?.Shape is RankedShape { Rank: > 0 } shape && shape[^1].IsFixed
            ? checked((uint)shape[^1].FixedValue)
            : 1U;
    }
}
