// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Math;
using Nncase.Utilities;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="NVFP4MatMul"/>.
/// </summary>
public sealed class NVFP4MatMulEvaluator :
    IEvaluator<NVFP4MatMul>,
    ITypeInferencer<NVFP4MatMul>,
    ICostEvaluator<NVFP4MatMul>
{
    private const int NVFP4GroupSize = 16;

    private static readonly float[] E2M1Values = [0F, 0.5F, 1F, 1.5F, 2F, 3F, 4F, 6F];

    public IValue Visit(IEvaluateContext context, NVFP4MatMul target)
        => Value.FromTensor(Evaluate(
            context.GetArgumentValueAsTensor(target, NVFP4MatMul.Lhs),
            context.GetArgumentValueAsTensor(target, NVFP4MatMul.RhsPacked),
            context.GetArgumentValueAsTensor(target, NVFP4MatMul.RhsScale),
            context.GetArgumentValueAsTensor(target, NVFP4MatMul.LhsGlobalScale),
            context.GetArgumentValueAsTensor(target, NVFP4MatMul.RhsGlobalScale),
            target.OutputDataType,
            target.GroupSize));

    public IRType Visit(ITypeInferenceContext context, NVFP4MatMul target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, NVFP4MatMul.Lhs),
            context.CheckArgumentType<IRType>(target, NVFP4MatMul.RhsPacked),
            context.CheckArgumentType<IRType>(target, NVFP4MatMul.RhsScale),
            context.CheckArgumentType<IRType>(target, NVFP4MatMul.LhsGlobalScale),
            context.CheckArgumentType<IRType>(target, NVFP4MatMul.RhsGlobalScale));

    public Cost Visit(ICostEvaluateContext context, NVFP4MatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, NVFP4MatMul.Lhs);
        var output = context.GetReturnType<IRType>();
        var localLhs = lhs is DistributedType distributed
            ? DistributedUtility.GetDividedTensorType(distributed)
            : GetTensorType(lhs);
        var k = localLhs?.Shape is RankedShape { Rank: > 0 } shape && shape[^1].IsFixed
            ? checked((uint)shape[^1].FixedValue)
            : 1U;
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                CostUtility.GetMemoryAccess(lhs) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, NVFP4MatMul.RhsPacked)) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, NVFP4MatMul.RhsScale)) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, NVFP4MatMul.LhsGlobalScale)) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, NVFP4MatMul.RhsGlobalScale)),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(output, checked(k + 8U)),
        };
    }

    public static Tensor Evaluate(
        Tensor lhs,
        Tensor rhsPacked,
        Tensor rhsScale,
        Tensor lhsGlobalScale,
        Tensor rhsGlobalScale,
        DataType outputDataType,
        long groupSize)
    {
        var inferred = InferType(
            new NVFP4MatMul(outputDataType, groupSize),
            new TensorType(lhs.ElementType, lhs.Shape),
            new TensorType(rhsPacked.ElementType, rhsPacked.Shape),
            new TensorType(rhsScale.ElementType, rhsScale.Shape),
            new TensorType(lhsGlobalScale.ElementType, lhsGlobalScale.Shape),
            new TensorType(rhsGlobalScale.ElementType, rhsGlobalScale.Shape));
        if (inferred is InvalidType invalid)
        {
            throw new InvalidOperationException(invalid.Reason);
        }

        var inputGlobal = GetPositiveScale(lhsGlobalScale, "lhs global");
        var weightGlobal = GetPositiveScale(rhsGlobalScale, "rhs global");
        var lhsShape = lhs.Dimensions.ToArray();
        var rhsShape = rhsPacked.Dimensions.ToArray();
        var k = checked((int)lhsShape[^1]);
        var n = checked((int)rhsShape[0]);
        var rows = checked((int)(lhs.Length / k));
        var groups = checked(k / (int)groupSize);
        var lhsValues = lhs.CastElementTo(DataTypes.Float32).ToArray<float>();
        var packedValues = rhsPacked.ToArray<byte>();
        var weightScales = rhsScale.CastElementTo(DataTypes.Float32).ToArray<float>();

        var activationScales = new float[checked(rows * groups)];
        for (var row = 0; row < rows; row++)
        {
            for (var group = 0; group < groups; group++)
            {
                var maxAbs = 0F;
                var begin = group * (int)groupSize;
                for (var index = 0; index < groupSize; index++)
                {
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(lhsValues[(row * k) + begin + index]));
                }

                activationScales[(row * groups) + group] = inputGlobal * maxAbs / 6F;
            }
        }

        activationScales = Tensor.From<float>(activationScales, [rows, groups])
            .CastElementTo(DataTypes.Float8E4M3)
            .CastElementTo(DataTypes.Float32)
            .ToArray<float>();

        var result = new float[checked(rows * n)];
        for (var row = 0; row < rows; row++)
        {
            for (var ni = 0; ni < n; ni++)
            {
                var accumulator = 0F;
                for (var ki = 0; ki < k; ki++)
                {
                    var group = ki / (int)groupSize;
                    var activationScale = activationScales[(row * groups) + group];
                    var activation = activationScale == 0F
                        ? 0F
                        : QuantizeE2M1(lhsValues[(row * k) + ki] * inputGlobal / activationScale);
                    var packed = packedValues[(ni * (k / 2)) + (ki / 2)];
                    var code = (ki & 1) == 0 ? packed & 0x0F : packed >> 4;
                    var weight = DecodeE2M1(code);
                    accumulator += activation * weight * activationScale *
                        weightScales[(ni * groups) + group];
                }

                result[(row * n) + ni] = accumulator / (inputGlobal * weightGlobal);
            }
        }

        var outputShape = lhsShape.ToArray();
        outputShape[^1] = n;
        return Tensor.From<float>(result, outputShape).CastElementTo(outputDataType);
    }

    public static IRType InferType(
        NVFP4MatMul target,
        IRType lhs,
        IRType rhsPacked,
        IRType rhsScale,
        IRType lhsGlobalScale,
        IRType rhsGlobalScale)
    {
        if (target.GroupSize != NVFP4GroupSize)
        {
            return new InvalidType(
                $"NVFP4MatMul requires the NVFP4 group size {NVFP4GroupSize}, got {target.GroupSize}.");
        }

        var lhsTensor = GetTensorType(lhs);
        var rhsTensor = GetTensorType(rhsPacked);
        var scaleTensor = GetTensorType(rhsScale);
        if (lhsTensor?.Shape is not RankedShape { Rank: >= 2 } lhsShape ||
            rhsTensor?.Shape is not RankedShape { Rank: 2 } rhsShape ||
            scaleTensor?.Shape is not RankedShape { Rank: 2 } scaleShape)
        {
            return new InvalidType(
                $"NVFP4MatMul expects lhs rank >= 2 and rank-2 packed rhs/scales, got " +
                $"lhs={lhs}, rhs_packed={rhsPacked}, rhs_scale={rhsScale}.");
        }

        if (lhsTensor.DType is not PrimType lhsType || !lhsType.IsFloat() ||
            rhsTensor.DType != DataTypes.UInt8 ||
            scaleTensor.DType != DataTypes.Float8E4M3 ||
            target.OutputDataType is not PrimType outputType || !outputType.IsFloat())
        {
            return new InvalidType(
                $"NVFP4MatMul requires floating lhs/output, U8 packed rhs, and E4M3 rhs scales, got " +
                $"{lhsTensor.DType}/{rhsTensor.DType}/{scaleTensor.DType}/{target.OutputDataType}.");
        }

        if (!Dimension.TryDivExactly(lhsShape[^1], 2, out var packedK) ||
            !Dimension.TryDivExactly(lhsShape[^1], target.GroupSize, out var scaleK) ||
            rhsShape[1] != packedK || scaleShape[0] != rhsShape[0] || scaleShape[1] != scaleK)
        {
            return new InvalidType(
                $"NVFP4MatMul expects rhs_packed=[N,K/2] and rhs_scale=[N,K/{target.GroupSize}] " +
                $"for lhs [...,M,K], got lhs={lhsShape}, rhs_packed={rhsShape}, rhs_scale={scaleShape}.");
        }

        if (ValidateGlobalScale(lhsGlobalScale, "lhs") is { } lhsScaleError)
        {
            return lhsScaleError;
        }

        if (ValidateGlobalScale(rhsGlobalScale, "rhs") is { } rhsScaleError)
        {
            return rhsScaleError;
        }

        var outputShape = lhsShape.ToArray();
        outputShape[^1] = rhsShape[0];
        var outputTensor = new TensorType(target.OutputDataType, new RankedShape(outputShape));
        if (lhs is TensorType && rhsPacked is TensorType && rhsScale is TensorType &&
            lhsGlobalScale is TensorType && rhsGlobalScale is TensorType)
        {
            return outputTensor;
        }

        if (lhs is not DistributedType lhsDistributed ||
            rhsPacked is not DistributedType rhsDistributed ||
            rhsScale is not DistributedType scaleDistributed)
        {
            return new InvalidType(
                "NVFP4MatMul requires lhs, packed rhs, and rhs scales to be either all tensors or all distributed tensors.");
        }

        if (lhsDistributed.Placement != rhsDistributed.Placement ||
            lhsDistributed.Placement != scaleDistributed.Placement)
        {
            return new InvalidType("NVFP4MatMul distributed operands must use the same placement.");
        }

        if (lhsDistributed.Partial is not null || rhsDistributed.Partial is not null ||
            scaleDistributed.Partial is not null)
        {
            return new InvalidType("NVFP4MatMul does not accept partial input operands.");
        }

        var lhsKPolicy = lhsDistributed.AxisPolicies[^1];
        var rhsNPolicy = rhsDistributed.AxisPolicies[0];
        var rhsKPolicy = rhsDistributed.AxisPolicies[1];
        var scaleKPolicy = scaleDistributed.AxisPolicies[1];
        if (!DistributedUtility.TryScaleAxisPolicyUnits(rhsKPolicy, 2, 1, out var logicalRhsKPolicy) ||
            !DistributedUtility.TryScaleAxisPolicyUnits(scaleKPolicy, target.GroupSize, 1, out var logicalScaleKPolicy) ||
            !DistributedUtility.IsSamePolicy(lhsKPolicy, logicalRhsKPolicy) ||
            !DistributedUtility.IsSamePolicy(lhsKPolicy, logicalScaleKPolicy) ||
            !DistributedUtility.IsSamePolicy(rhsNPolicy, scaleDistributed.AxisPolicies[0]))
        {
            return new InvalidType(
                "NVFP4MatMul distributed operands must preserve identical logical shard boundaries: " +
                "one packed-rhs byte represents two K values and one rhs scale represents GroupSize K values.");
        }

        if (!IsReplicatedGlobalScale(lhsGlobalScale, lhsDistributed.Placement) ||
            !IsReplicatedGlobalScale(rhsGlobalScale, lhsDistributed.Placement))
        {
            return new InvalidType("NVFP4MatMul global scales must be replicated on the operand placement.");
        }

        var outputPolicies = lhsDistributed.AxisPolicies.ToArray();
        outputPolicies[^1] = rhsNPolicy;
        var partial = lhsKPolicy is SBPSplit split ? SBP.P(split.HierarchyAxes) : null;
        if (!DistributedUtility.IsDistributable(outputTensor, outputPolicies, lhsDistributed.Placement))
        {
            return new InvalidType("NVFP4MatMul output policies are not distributable for the selected placement.");
        }

        return new DistributedType(outputTensor, outputPolicies, lhsDistributed.Placement, Partial: partial);
    }

    private static TensorType? GetTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => distributed.TensorType,
        _ => null,
    };

    private static InvalidType? ValidateGlobalScale(IRType type, string name)
    {
        var tensor = GetTensorType(type);
        if (tensor?.DType != DataTypes.Float32 ||
            tensor.Shape is not RankedShape shape ||
            shape.Rank == 0 || !shape.IsFixed ||
            shape.Aggregate(1L, (value, dim) => checked(value * dim.FixedValue)) != 1L)
        {
            return new InvalidType(
                $"NVFP4MatMul {name} global scale must be a one-element F32 tensor, got {type}.");
        }

        return null;
    }

    private static bool IsReplicatedGlobalScale(IRType type, Placement placement) => type switch
    {
        TensorType => true,
        DistributedType distributed =>
            distributed.Placement == placement &&
            distributed.Partial is null &&
            distributed.AxisPolicies.All(policy => policy is SBPBroadCast),
        _ => false,
    };

    private static float GetPositiveScale(Tensor scale, string name)
    {
        var values = scale.CastElementTo(DataTypes.Float32).ToArray<float>();
        if (values.Length != 1 || !float.IsFinite(values[0]) || values[0] <= 0F)
        {
            throw new InvalidOperationException(
                $"NVFP4MatMul {name} scale must contain one finite positive value, got " +
                $"[{string.Join(", ", values)}].");
        }

        return values[0];
    }

    private static float DecodeE2M1(int code)
    {
        var value = E2M1Values[code & 0x07];
        return (code & 0x08) == 0 ? value : -value;
    }

    private static float QuantizeE2M1(float value)
    {
        var sign = MathF.CopySign(1F, value);
        var magnitude = MathF.Abs(value);
        var quantized = magnitude switch
        {
            <= 0.25F => 0F,
            < 0.75F => 0.5F,
            <= 1.25F => 1F,
            < 1.75F => 1.5F,
            <= 2.5F => 2F,
            < 3.5F => 3F,
            <= 5F => 4F,
            _ => 6F,
        };
        return quantized * sign;
    }
}
