// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;
using NVFP4MatMulEvaluator = Nncase.Evaluator.Math.NVFP4MatMulEvaluator;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="PackedNVFP4MatMul"/>.
/// </summary>
public sealed class PackedNVFP4MatMulEvaluator :
    IEvaluator<PackedNVFP4MatMul>,
    ITypeInferencer<PackedNVFP4MatMul>,
    ICostEvaluator<PackedNVFP4MatMul>
{
    public IValue Visit(IEvaluateContext context, PackedNVFP4MatMul target)
        => Value.FromTensor(Evaluate(
            target,
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMul.Lhs),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMul.RhsPacked),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMul.RhsScale),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMul.LhsGlobalScale),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMul.RhsGlobalScale),
            context.GetArgumentValue(target, PackedNVFP4MatMul.Addend)));

    public IRType Visit(ITypeInferenceContext context, PackedNVFP4MatMul target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMul.Lhs),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMul.RhsPacked),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMul.RhsScale),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMul.LhsGlobalScale),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMul.RhsGlobalScale),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMul.Addend));

    public Cost Visit(ICostEvaluateContext context, PackedNVFP4MatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, PackedNVFP4MatMul.Lhs);
        var addend = context.GetArgumentType<IRType>(target, PackedNVFP4MatMul.Addend);
        var output = context.GetReturnType<IRType>();
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                CostUtility.GetMemoryAccess(lhs) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, PackedNVFP4MatMul.RhsPacked)) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, PackedNVFP4MatMul.RhsScale)) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, PackedNVFP4MatMul.LhsGlobalScale)) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, PackedNVFP4MatMul.RhsGlobalScale)) +
                (addend is NoneType ? 0 : CostUtility.GetMemoryAccess(addend)),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(
                output,
                checked(GetLogicalK(lhs) + 8U + (addend is NoneType ? 0U : 1U))),
        };
    }

    public static Tensor Evaluate(
        PackedNVFP4MatMul target,
        Tensor lhs,
        Tensor rhsPacked,
        Tensor rhsScale,
        Tensor lhsGlobalScale,
        Tensor rhsGlobalScale,
        IValue addend)
    {
        RequireVectorType(
            lhs.ElementType,
            DataTypes.BFloat16,
            [target.InputKVectorLaneCount],
            "PackedNVFP4MatMul lhs");
        RequireVectorType(
            rhsPacked.ElementType,
            DataTypes.UInt8,
            [target.RhsKPackLaneCount, target.RhsKVectorLaneCount],
            "PackedNVFP4MatMul rhs");

        var logicalLhs = UnpackTensor(lhs);
        var logicalRhs = UnpackTensor(rhsPacked);
        var logicalOutput = NVFP4MatMulEvaluator.Evaluate(
            logicalLhs,
            logicalRhs,
            rhsScale,
            lhsGlobalScale,
            rhsGlobalScale,
            target.OutputDataType,
            target.GroupSize);
        var packedOutput = PackTensor(logicalOutput, target.OutputNVectorLaneCount);
        return IsNone(addend)
            ? packedOutput
            : OrtKI.Add(packedOutput.ToOrtTensor(), addend.AsTensor().ToOrtTensor())
                .ToTensor(packedOutput.ElementType);
    }

    public static IRType InferType(
        PackedNVFP4MatMul target,
        IRType lhs,
        IRType rhsPacked,
        IRType rhsScale,
        IRType lhsGlobalScale,
        IRType rhsGlobalScale,
        IRType addend)
    {
        if (ValidateTargetContract(target) is { } targetError)
        {
            return targetError;
        }

        if (ValidateVectorType(
                GetTensorType(lhs)?.DType,
                DataTypes.BFloat16,
                [target.InputKVectorLaneCount],
                "PackedNVFP4MatMul lhs") is { } lhsError)
        {
            return lhsError;
        }

        if (ValidateVectorType(
                GetTensorType(rhsPacked)?.DType,
                DataTypes.UInt8,
                [target.RhsKPackLaneCount, target.RhsKVectorLaneCount],
                "PackedNVFP4MatMul rhs") is { } rhsError)
        {
            return rhsError;
        }

        var logicalLhs = UnpackType(lhs, 1, "lhs");
        var logicalRhs = UnpackType(rhsPacked, 2, "rhs");
        if (logicalLhs is InvalidType)
        {
            return logicalLhs;
        }

        if (logicalRhs is InvalidType)
        {
            return logicalRhs;
        }

        var logicalOutput = NVFP4MatMulEvaluator.InferType(
            new Nncase.IR.Math.NVFP4MatMul(target.OutputDataType, target.GroupSize),
            logicalLhs,
            logicalRhs,
            rhsScale,
            lhsGlobalScale,
            rhsGlobalScale);
        if (logicalOutput is InvalidType)
        {
            return logicalOutput;
        }

        var outputTensor = GetTensorType(logicalOutput);
        if (outputTensor?.Shape is not RankedShape { Rank: > 0 } outputShape ||
            !Dimension.TryDivExactly(outputShape[^1], target.OutputNVectorLaneCount, out _))
        {
            return new InvalidType(
                $"PackedNVFP4MatMul output N must be divisible by lane count " +
                $"{target.OutputNVectorLaneCount}, got {logicalOutput}.");
        }

        var output = PackOutputType(logicalOutput, target.OutputNVectorLaneCount);
        if (output is InvalidType || addend is NoneType)
        {
            return output;
        }

        return Equals(output, addend)
            ? output
            : new InvalidType(
                $"PackedNVFP4MatMul addend must have exactly the packed output type, " +
                $"got output={output}, addend={addend}.");
    }

    internal static IRType InferProjectionType(
        DataType outputDataType,
        long groupSize,
        int inputKVectorLaneCount,
        int rhsKPackLaneCount,
        int rhsKVectorLaneCount,
        int outputNVectorLaneCount,
        IRType input,
        IRType weightPacked,
        IRType weightScale,
        IRType inputGlobalScale,
        IRType weightGlobalScale,
        IRType? addend = null)
        => InferType(
            new PackedNVFP4MatMul(
                outputDataType,
                groupSize,
                inputKVectorLaneCount,
                rhsKPackLaneCount,
                rhsKVectorLaneCount,
                outputNVectorLaneCount),
            input,
            weightPacked,
            weightScale,
            inputGlobalScale,
            weightGlobalScale,
            addend ?? NoneType.Default);

    internal static Tensor EvaluateProjection(
        DataType outputDataType,
        long groupSize,
        int inputKVectorLaneCount,
        int rhsKPackLaneCount,
        int rhsKVectorLaneCount,
        int outputNVectorLaneCount,
        Tensor input,
        Tensor weightPacked,
        Tensor weightScale,
        Tensor inputGlobalScale,
        Tensor weightGlobalScale)
        => Evaluate(
            new PackedNVFP4MatMul(
                outputDataType,
                groupSize,
                inputKVectorLaneCount,
                rhsKPackLaneCount,
                rhsKVectorLaneCount,
                outputNVectorLaneCount),
            input,
            weightPacked,
            weightScale,
            inputGlobalScale,
            weightGlobalScale,
            NoneValue.Default);

    internal static TensorType? GetTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => distributed.TensorType,
        _ => null,
    };

    internal static uint GetLogicalK(IRType input)
    {
        var local = input is DistributedType distributed
            ? DistributedUtility.GetDividedTensorType(distributed)
            : GetTensorType(input);
        if (local?.Shape is not RankedShape { Rank: > 0 } shape || !shape[^1].IsFixed)
        {
            return 1U;
        }

        var laneProduct = local.DType is VectorType vector
            ? vector.Lanes.Aggregate(1, checked((product, lane) => product * lane))
            : 1;
        return checked((uint)(shape[^1].FixedValue * laneProduct));
    }

    private static InvalidType? ValidateTargetContract(PackedNVFP4MatMul target)
    {
        if (target.InputKVectorLaneCount <= 0 || target.RhsKPackLaneCount <= 0 ||
            target.RhsKVectorLaneCount <= 0 || target.OutputNVectorLaneCount <= 0)
        {
            return new InvalidType(
                "PackedNVFP4MatMul vector lane counts must all be positive.");
        }

        return null;
    }

    private static InvalidType? ValidateVectorType(
        DataType? actual,
        DataType expectedElementType,
        int[] expectedLanes,
        string name)
    {
        if (actual is not VectorType vector || vector.ElemType != expectedElementType ||
            !vector.Lanes.SequenceEqual(expectedLanes))
        {
            return new InvalidType(
                $"{name} must have dtype vec<{expectedElementType}," +
                $"[{string.Join(",", expectedLanes)}]>, got {actual}.");
        }

        return null;
    }

    private static void RequireVectorType(
        DataType actual,
        DataType expectedElementType,
        int[] expectedLanes,
        string name)
    {
        if (ValidateVectorType(actual, expectedElementType, expectedLanes, name) is { } error)
        {
            throw new InvalidOperationException(error.Reason);
        }
    }

    private static Tensor UnpackTensor(Tensor input)
    {
        var vector = (VectorType)input.ElementType;
        var axes = Enumerable.Repeat(input.Rank - 1, vector.Lanes.Count).ToArray();
        return input.ToOrtTensor().Unpack(vector.Lanes.Count, axes).ToTensor();
    }

    private static Tensor PackTensor(Tensor input, int lanes)
    {
        var outputAxis = input.Rank - 1;
        return input.ToOrtTensor()
            .Pack(0, [lanes], [outputAxis])
            .ToTensor(new VectorType(input.ElementType, [lanes]));
    }

    private static IRType UnpackType(IRType input, int laneRank, string name)
    {
        var tensor = GetTensorType(input);
        if (tensor?.Shape is not RankedShape { Rank: > 0 } shape)
        {
            return new InvalidType($"Cannot unpack PackedNVFP4MatMul {name} type {input}.");
        }

        var axes = Enumerable.Repeat(shape.Rank - 1, laneRank).ToArray();
        return input switch
        {
            DistributedType distributed => TypeInference.UnpackType(distributed, axes),
            TensorType plain => TypeInference.UnpackType(plain, axes),
            _ => new InvalidType($"Cannot unpack PackedNVFP4MatMul {name} type {input}."),
        };
    }

    private static IRType PackOutputType(IRType output, int lanes)
    {
        var tensor = GetTensorType(output)!;
        var axes = new[] { tensor.Shape.Rank - 1 };
        return output switch
        {
            DistributedType distributed => TypeInference.PackType(distributed, [lanes], axes),
            TensorType plain => TypeInference.PackType(plain, [lanes], axes),
            _ => new InvalidType($"Cannot pack PackedNVFP4MatMul output type {output}."),
        };
    }

    private static bool IsNone(IValue value) => value is NoneValue || value.Type is NoneType;
}
