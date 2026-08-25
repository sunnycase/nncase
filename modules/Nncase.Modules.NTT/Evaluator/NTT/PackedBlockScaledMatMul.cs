// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NTT;
using Nncase.Utilities;
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
        => Evaluate(
            target,
            context.GetArgumentValueAsTensor(target, PackedBlockScaledMatMul.Lhs),
            context.GetArgumentValueAsTensor(target, PackedBlockScaledMatMul.Rhs),
            context.GetArgumentValueAsTensor(target, PackedBlockScaledMatMul.RhsScale),
            context.GetArgumentValue(target, PackedBlockScaledMatMul.Addend));

    public static IValue Evaluate(
        PackedBlockScaledMatMul target,
        Tensor lhs,
        Tensor rhs,
        Tensor rhsScale,
        IValue addend)
    {
        string? errorMessage = null;
        if (rhs.ElementType is not VectorType rhsVectorType ||
            !PackedMatMulEvaluator.TryGetRhsLayoutInfo(
                target.RhsLayout,
                rhsVectorType,
                rhs.Rank,
                out var rhsUnpackAxes,
                out var transposeB,
                out errorMessage) ||
            target.OutputNVectorLaneCount <= 0)
        {
            throw new InvalidOperationException(
                errorMessage ??
                $"PackedBlockScaledMatMul requires a positive output N vector lane count, got {target.OutputNVectorLaneCount}.");
        }

        var logicalRhsOrt = rhs.ToOrtTensor().Unpack(rhsVectorType.Lanes.Count, rhsUnpackAxes);
        if (transposeB)
        {
            var permutation = Enumerable.Range(0, logicalRhsOrt.Rank).Select(index => (long)index).ToArray();
            (permutation[^2], permutation[^1]) = (permutation[^1], permutation[^2]);
            logicalRhsOrt = OrtKI.Transpose(logicalRhsOrt, permutation);
        }

        var logicalRhs = logicalRhsOrt.ToTensor();
        var logical = BlockScaledMatMulEvaluator.Evaluate(
            lhs,
            logicalRhs,
            rhsScale,
            target.OutputDataType,
            target.WeightBlockN,
            target.WeightBlockK).AsTensor().ToOrtTensor();
        var outputAxis = logical.Rank - 1;
        var outputLanes = new[] { target.OutputNVectorLaneCount };
        var packed = logical.Pack(
            0,
            outputLanes,
            Enumerable.Repeat(outputAxis, outputLanes.Length).ToArray());
        if (!IsNone(addend))
        {
            packed = OrtKI.Add(packed, addend.AsTensor().ToOrtTensor());
        }

        return packed.ToValue(new VectorType(target.OutputDataType, outputLanes));
    }

    public IRType Visit(ITypeInferenceContext context, PackedBlockScaledMatMul target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, PackedBlockScaledMatMul.Lhs),
            context.CheckArgumentType<IRType>(target, PackedBlockScaledMatMul.Rhs),
            context.CheckArgumentType<IRType>(target, PackedBlockScaledMatMul.RhsScale),
            context.CheckArgumentType<IRType>(target, PackedBlockScaledMatMul.Addend));

    public static IRType InferType(
        PackedBlockScaledMatMul target,
        IRType lhs,
        IRType rhs,
        IRType rhsScale,
        IRType addend)
    {
        var rhsTensor = ScaledMatMulEvaluator.GetTensorType(rhs);
        string? errorMessage = null;
        if (rhsTensor?.DType is not VectorType rhsVectorType ||
            !PackedMatMulEvaluator.TryGetRhsLayoutInfo(
                target.RhsLayout,
                rhsVectorType,
                rhsTensor.Shape.Rank,
                out var rhsUnpackAxes,
                out var transposeB,
                out errorMessage) ||
            target.OutputNVectorLaneCount <= 0)
        {
            return new InvalidType(
                errorMessage ??
                $"PackedBlockScaledMatMul requires a positive output N vector lane count, got {target.OutputNVectorLaneCount}.");
        }

        var logicalRhs = UnpackType(rhs, rhsUnpackAxes);
        if (logicalRhs is InvalidType)
        {
            return logicalRhs;
        }

        if (transposeB)
        {
            logicalRhs = TransposeLastTwo(logicalRhs);
            if (logicalRhs is InvalidType)
            {
                return logicalRhs;
            }
        }

        var logical = BlockScaledMatMulEvaluator.InferType(
            new BlockScaledMatMul(
                target.OutputDataType,
                target.WeightBlockN,
                target.WeightBlockK),
            lhs,
            logicalRhs,
            rhsScale);
        var output = logical is InvalidType
            ? logical
            : PackOutput(logical, [target.OutputNVectorLaneCount]);
        if (output is InvalidType || addend is NoneType)
        {
            return output;
        }

        return Equals(output, addend)
            ? output
            : new InvalidType(
                $"PackedBlockScaledMatMul addend must have exactly the packed output type, " +
                $"got output={output}, addend={addend}.");
    }

    public Cost Visit(ICostEvaluateContext context, PackedBlockScaledMatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, PackedBlockScaledMatMul.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, PackedBlockScaledMatMul.Rhs);
        var rhsScale = context.GetArgumentType<IRType>(target, PackedBlockScaledMatMul.RhsScale);
        var addend = context.GetArgumentType<IRType>(target, PackedBlockScaledMatMul.Addend);
        var output = context.GetReturnType<IRType>();
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs) +
                CostUtility.GetMemoryAccess(rhsScale) + CostUtility.GetMemoryAccess(addend),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(
                output,
                checked(GetK(lhs) + 4U + (addend is NoneType ? 0U : 1U))),
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

    private static IRType TransposeLastTwo(IRType input)
    {
        var tensor = ScaledMatMulEvaluator.GetTensorType(input);
        if (tensor?.Shape is not RankedShape { Rank: >= 2 } shape)
        {
            return new InvalidType($"Cannot transpose PackedBlockScaledMatMul logical RHS type {input}.");
        }

        var permutation = Enumerable.Range(0, shape.Rank).ToArray();
        (permutation[^2], permutation[^1]) = (permutation[^1], permutation[^2]);
        var permutationShape = new RankedShape(
            permutation.Select(axis => (Dimension)axis).ToArray());
        if (TypeInference.TransposeType(tensor, permutationShape) is not TensorType transposedTensor)
        {
            return new InvalidType($"Cannot transpose PackedBlockScaledMatMul logical RHS type {input}.");
        }

        return input switch
        {
            TensorType => transposedTensor,
            DistributedType distributed => new DistributedType(
                transposedTensor,
                permutation.Select(axis => distributed.AxisPolicies[axis]).ToArray(),
                distributed.Placement,
                distributed.Partial),
            _ => new InvalidType($"Cannot transpose PackedBlockScaledMatMul logical RHS type {input}."),
        };
    }

    private static uint GetK(IRType lhs)
    {
        var tensor = ScaledMatMulEvaluator.GetTensorType(lhs);
        return tensor?.Shape is RankedShape { Rank: > 0 } shape && shape[^1].IsFixed
            ? checked((uint)shape[^1].FixedValue)
            : 1U;
    }

    private static bool IsNone(IValue value) => value is NoneValue || value.Type is NoneType;
}
