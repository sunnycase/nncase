// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NTT;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="VectorizedRoPE"/>.
/// </summary>
public class VectorizedRoPEEvaluator : IEvaluator<VectorizedRoPE>, ITypeInferencer<VectorizedRoPE>, ICostEvaluator<VectorizedRoPE>,
    IMetricEvaluator<VectorizedRoPE>
{
    public static bool AxisEqual(IRArray<SBP> a, IRArray<SBP> b, int startA, int startB)
    {
        var lenA = a.Count - startA;
        if (lenA != b.Count - startB)
        {
            return false;
        }

        for (int i = 0; i < lenA; i++)
        {
            if (!Equals(a[startA + i], b[startB + i]))
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, VectorizedRoPE target)
    {
        var inputTensor = context.GetArgumentValueAsTensor(target, VectorizedRoPE.Input);
        var cosTensor = context.GetArgumentValueAsTensor(target, VectorizedRoPE.Cos);
        var sinTensor = context.GetArgumentValueAsTensor(target, VectorizedRoPE.Sin);

        var originDtype = inputTensor.ElementType;
        var rotaryAxis = inputTensor.Dimensions.Length - 1;
        var input = OrtKI.Cast(ToLogicalTensor(inputTensor, rotaryAxis), (long)OrtDataType.Float);
        var cos = OrtKI.Cast(ToLogicalTensor(cosTensor, rotaryAxis), (long)OrtDataType.Float);
        var sin = OrtKI.Cast(ToLogicalTensor(sinTensor, rotaryAxis), (long)OrtDataType.Float);

        var sliceAxis = input.Rank - 1;
        var sliceDim = input.Shape[sliceAxis] / 2;
        var parts = OrtKI.Split(input, new[] { sliceDim, sliceDim }, sliceAxis);

        // rotate half
        var rotated = OrtKI.Concat([OrtKI.Neg(parts[1]), parts[0]], sliceAxis);
        var output = OrtKI.Add(OrtKI.Mul(input, cos), OrtKI.Mul(rotated, sin));
        output = OrtKI.Cast(output, (long)GetScalarDataType(originDtype).ToOrtType());
        output = FromLogicalTensor(output, originDtype, rotaryAxis);
        return output.ToValue(originDtype);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, VectorizedRoPE target)
    {
        var input = context.CheckArgumentType<IRType>(target, VectorizedRoPE.Input);
        var cos = context.CheckArgumentType<IRType>(target, VectorizedRoPE.Cos);
        var sin = context.CheckArgumentType<IRType>(target, VectorizedRoPE.Sin);

        return (input, cos, sin) switch
        {
            (DistributedType a, DistributedType b, DistributedType c) => Visit(a, b, c),
            (TensorType a, TensorType, TensorType) => Visit(a),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, VectorizedRoPE target)
    {
        var inputType = context.GetArgumentType<IRType>(target, VectorizedRoPE.Input);
        var cosType = context.GetArgumentType<IRType>(target, VectorizedRoPE.Cos);
        var sinType = context.GetArgumentType<IRType>(target, VectorizedRoPE.Sin);
        var returnType = context.GetReturnType<IRType>();
        if (TargetOpCostModelUtility.TryGetTargetElementwiseCost(context.TargetCostModel, "vectorized_rope", [inputType, cosType, sinType], returnType, workPerElement: 4.0, out var targetCost))
        {
            return targetCost;
        }

        var macPerElement = 4; // 2 for mul, 1 for add, 1 for neg and concat
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(cosType) + CostUtility.GetMemoryAccess(sinType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, macPerElement),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, VectorizedRoPE target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, VectorizedRoPE.Input);
        var cosType = context.GetArgumentType<TensorType>(target, VectorizedRoPE.Cos);
        var sinType = context.GetArgumentType<TensorType>(target, VectorizedRoPE.Sin);
        var returnType = context.GetReturnType<TensorType>();
        var macPerElement = 2; // 1 for mul, 1 for add

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(cosType) + CostUtility.GetMemoryAccess(sinType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(returnType, macPerElement),
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }

    private IRType Visit(DistributedType input, DistributedType cos, DistributedType sin)
    {
        // only unsupported print without to-string
        if (input.Placement != cos.Placement || cos.Placement != sin.Placement
            || !Equals(input.AxisPolicies[0], cos.AxisPolicies[0])
            || cos.AxisPolicies[1] is not SBPBroadCast
            || !Equals(input.AxisPolicies[2], cos.AxisPolicies[2])
            || !AxisEqual(cos.AxisPolicies, sin.AxisPolicies, startA: 0, startB: 0)
            || input.AxisPolicies[^1] is not SBPBroadCast)
        {
            return new InvalidType("RoPE: distributed types mismatch (placement/axis/SBP)");

            // optional(still ToString)：
            // return new InvalidType($"RoPE mismatch: in={input.GetType().Name}, cos={cos.GetType().Name}, sin={sin.GetType().Name}");
        }

        return input;
    }

    private static OrtKISharp.Tensor ToLogicalTensor(Tensor tensor, int vectorizedAxis)
    {
        var ort = tensor.ToOrtTensor();
        var lanes = GetVectorLanes(tensor.ElementType);
        return lanes.Length == 0
            ? ort
            : ort.Unpack(lanes.Length, Enumerable.Repeat(vectorizedAxis, lanes.Length).ToArray());
    }

    private static OrtKISharp.Tensor FromLogicalTensor(OrtKISharp.Tensor tensor, DataType dataType, int vectorizedAxis)
    {
        var lanes = GetVectorLanes(dataType);
        return lanes.Length == 0
            ? tensor
            : tensor.Pack(0, lanes, Enumerable.Repeat(vectorizedAxis, lanes.Length).ToArray());
    }

    private static DataType GetScalarDataType(DataType dataType)
    {
        return dataType switch
        {
            VectorType vectorType => vectorType.ElemType,
            MaskVectorType => DataTypes.Boolean,
            _ => dataType,
        };
    }

    private static int[] GetVectorLanes(DataType dataType)
    {
        return dataType switch
        {
            VectorType vectorType => vectorType.Lanes.ToArray(),
            MaskVectorType maskVectorType => [maskVectorType.Lanes],
            _ => [],
        };
    }
}
