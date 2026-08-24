// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NTT;
using OrtKISharp;
using BlockScaledMatMulEvaluator = Nncase.Evaluator.Math.BlockScaledMatMulEvaluator;
using ScaledMatMulEvaluator = Nncase.Evaluator.Math.ScaledMatMulEvaluator;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="PackedBlockScaledMatMul"/>.
/// </summary>
public sealed class PackedBlockScaledMatMulEvaluator :
    IEvaluator<PackedBlockScaledMatMul>,
    ITypeInferencer<PackedBlockScaledMatMul>,
    ICostEvaluator<PackedBlockScaledMatMul>
{
    public IValue Visit(IEvaluateContext context, PackedBlockScaledMatMul target)
    {
        var lhs = context.GetArgumentValueAsTensor(target, PackedBlockScaledMatMul.Lhs);
        var rhs = context.GetArgumentValueAsTensor(target, PackedBlockScaledMatMul.Rhs);
        var rhsScale = context.GetArgumentValueAsTensor(target, PackedBlockScaledMatMul.RhsScale);
        string? errorMessage = null;
        if (rhs.ElementType is not VectorType rhsVectorType ||
            !PackedMatMulEvaluator.TryGetLayoutInfo(
                target.RhsLayout,
                rhsVectorType,
                rhs.Rank,
                out var rhsUnpackAxes,
                out var outputLanes,
                out var transposeB,
                out errorMessage) ||
            transposeB)
        {
            throw new InvalidOperationException(
                errorMessage ?? "PackedBlockScaledMatMul requires a K-major vector RHS.");
        }

        var logicalRhs = rhs.ToOrtTensor().Unpack(rhsVectorType.Lanes.Count, rhsUnpackAxes).ToTensor();
        var logical = BlockScaledMatMulEvaluator.Evaluate(
            lhs,
            logicalRhs,
            rhsScale,
            target.OutputDataType,
            target.WeightBlockN,
            target.WeightBlockK).AsTensor().ToOrtTensor();
        var outputAxis = logical.Rank - 1;
        return logical
            .Pack(0, outputLanes, Enumerable.Repeat(outputAxis, outputLanes.Length).ToArray())
            .ToValue(new VectorType(target.OutputDataType, outputLanes));
    }

    public IRType Visit(ITypeInferenceContext context, PackedBlockScaledMatMul target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, PackedBlockScaledMatMul.Lhs),
            context.CheckArgumentType<IRType>(target, PackedBlockScaledMatMul.Rhs),
            context.CheckArgumentType<IRType>(target, PackedBlockScaledMatMul.RhsScale));

    public static IRType InferType(
        PackedBlockScaledMatMul target,
        IRType lhs,
        IRType rhs,
        IRType rhsScale)
    {
        var rhsTensor = ScaledMatMulEvaluator.GetTensorType(rhs);
        string? errorMessage = null;
        if (rhsTensor?.DType is not VectorType rhsVectorType ||
            !PackedMatMulEvaluator.TryGetLayoutInfo(
                target.RhsLayout,
                rhsVectorType,
                rhsTensor.Shape.Rank,
                out var rhsUnpackAxes,
                out var outputLanes,
                out var transposeB,
                out errorMessage) ||
            transposeB)
        {
            return new InvalidType(
                errorMessage ?? "PackedBlockScaledMatMul requires a K-major vector RHS.");
        }

        var logicalRhs = UnpackType(rhs, rhsUnpackAxes);
        if (logicalRhs is InvalidType)
        {
            return logicalRhs;
        }

        var logical = BlockScaledMatMulEvaluator.InferType(
            new BlockScaledMatMul(
                target.OutputDataType,
                target.WeightBlockN,
                target.WeightBlockK),
            lhs,
            logicalRhs,
            rhsScale);
        return logical is InvalidType ? logical : PackOutput(logical, outputLanes);
    }

    public Cost Visit(ICostEvaluateContext context, PackedBlockScaledMatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, PackedBlockScaledMatMul.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, PackedBlockScaledMatMul.Rhs);
        var rhsScale = context.GetArgumentType<IRType>(target, PackedBlockScaledMatMul.RhsScale);
        var output = context.GetReturnType<IRType>();
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs) +
                CostUtility.GetMemoryAccess(rhsScale),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(output, GetK(lhs) + 4U),
        };
    }

    private static IRType UnpackType(IRType input, int[] axes) => input switch
    {
        DistributedType distributed => TypeInference.UnpackType(distributed, axes),
        TensorType tensor => TypeInference.UnpackType(tensor, axes),
        _ => new InvalidType($"Cannot unpack PackedBlockScaledMatMul RHS type {input}."),
    };

    private static IRType PackOutput(IRType output, int[] lanes) => output switch
    {
        DistributedType distributed => TypeInference.PackType(
            distributed,
            lanes,
            Enumerable.Repeat(distributed.TensorType.Shape.Rank - 1, lanes.Length).ToArray()),
        TensorType tensor => TypeInference.PackType(
            tensor,
            lanes,
            Enumerable.Repeat(tensor.Shape.Rank - 1, lanes.Length).ToArray()),
        _ => new InvalidType($"PackedBlockScaledMatMul produced a non-tensor type {output}."),
    };

    private static uint GetK(IRType lhs)
    {
        var tensor = ScaledMatMulEvaluator.GetTensorType(lhs);
        return tensor?.Shape is RankedShape { Rank: > 0 } shape && shape[^1].IsFixed
            ? checked((uint)shape[^1].FixedValue)
            : 1U;
    }
}
