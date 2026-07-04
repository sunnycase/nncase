// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="QKVParallelLinear"/>.
/// </summary>
public sealed class QKVParallelLinearEvaluator : IEvaluator<QKVParallelLinear>, ITypeInferencer<QKVParallelLinear>, ICostEvaluator<QKVParallelLinear>
{
    public IValue Visit(IEvaluateContext context, QKVParallelLinear target)
    {
        var input = context.GetArgumentValueAsTensor(target, QKVParallelLinear.Input);
        var qWeight = context.GetArgumentValueAsTensor(target, QKVParallelLinear.QWeight);
        var kWeight = context.GetArgumentValueAsTensor(target, QKVParallelLinear.KWeight);
        var vWeight = context.GetArgumentValueAsTensor(target, QKVParallelLinear.VWeight);
        return Value.FromTensors(
            Project(input, qWeight, context.GetArgumentValue(target, QKVParallelLinear.QBias), context.GetArgumentValue(target, QKVParallelLinear.QInputScale), context.GetArgumentValue(target, QKVParallelLinear.QWeightScale), target.OutputDataType),
            Project(input, kWeight, context.GetArgumentValue(target, QKVParallelLinear.KBias), context.GetArgumentValue(target, QKVParallelLinear.KInputScale), context.GetArgumentValue(target, QKVParallelLinear.KWeightScale), target.OutputDataType),
            Project(input, vWeight, context.GetArgumentValue(target, QKVParallelLinear.VBias), context.GetArgumentValue(target, QKVParallelLinear.VInputScale), context.GetArgumentValue(target, QKVParallelLinear.VWeightScale), target.OutputDataType));
    }

    public IRType Visit(ITypeInferenceContext context, QKVParallelLinear target)
    {
        var input = context.CheckArgumentType<IRType>(target, QKVParallelLinear.Input);
        var qWeight = context.CheckArgumentType<IRType>(target, QKVParallelLinear.QWeight);
        var kWeight = context.CheckArgumentType<IRType>(target, QKVParallelLinear.KWeight);
        var vWeight = context.CheckArgumentType<IRType>(target, QKVParallelLinear.VWeight);
        var qBias = context.CheckArgumentType<IRType>(target, QKVParallelLinear.QBias);
        var kBias = context.CheckArgumentType<IRType>(target, QKVParallelLinear.KBias);
        var vBias = context.CheckArgumentType<IRType>(target, QKVParallelLinear.VBias);
        var q = VisitProjection(input, qWeight, target.OutputDataType);
        var k = VisitProjection(input, kWeight, target.OutputDataType);
        var v = VisitProjection(input, vWeight, target.OutputDataType);
        if (q is InvalidType)
        {
            return q;
        }

        if (k is InvalidType)
        {
            return k;
        }

        if (v is InvalidType)
        {
            return v;
        }

        var biasCheck = CheckBiasType("q", q, qBias) ?? CheckBiasType("k", k, kBias) ?? CheckBiasType("v", v, vBias);
        if (biasCheck is not null)
        {
            return biasCheck;
        }

        var headCheck = CheckHeadShape(target, q, k, v);
        if (headCheck is not null)
        {
            return headCheck;
        }

        return new TupleType(new[] { q, k, v });
    }

    public Cost Visit(ICostEvaluateContext context, QKVParallelLinear target)
    {
        var input = context.GetArgumentType<IRType>(target, QKVParallelLinear.Input);
        var qWeight = context.GetArgumentType<IRType>(target, QKVParallelLinear.QWeight);
        var kWeight = context.GetArgumentType<IRType>(target, QKVParallelLinear.KWeight);
        var vWeight = context.GetArgumentType<IRType>(target, QKVParallelLinear.VWeight);
        var qBias = context.GetArgumentType<IRType>(target, QKVParallelLinear.QBias);
        var kBias = context.GetArgumentType<IRType>(target, QKVParallelLinear.KBias);
        var vBias = context.GetArgumentType<IRType>(target, QKVParallelLinear.VBias);
        var qInputScale = context.GetArgumentType<IRType>(target, QKVParallelLinear.QInputScale);
        var kInputScale = context.GetArgumentType<IRType>(target, QKVParallelLinear.KInputScale);
        var vInputScale = context.GetArgumentType<IRType>(target, QKVParallelLinear.VInputScale);
        var qWeightScale = context.GetArgumentType<IRType>(target, QKVParallelLinear.QWeightScale);
        var kWeightScale = context.GetArgumentType<IRType>(target, QKVParallelLinear.KWeightScale);
        var vWeightScale = context.GetArgumentType<IRType>(target, QKVParallelLinear.VWeightScale);
        var output = context.GetReturnType<TupleType>();
        var macPerElement = GetK(input);
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input)
                + CostUtility.GetMemoryAccess(qWeight) + CostUtility.GetMemoryAccess(kWeight) + CostUtility.GetMemoryAccess(vWeight)
                + CostUtility.GetMemoryAccess(qBias) + CostUtility.GetMemoryAccess(kBias) + CostUtility.GetMemoryAccess(vBias)
                + CostUtility.GetMemoryAccess(qInputScale) + CostUtility.GetMemoryAccess(kInputScale) + CostUtility.GetMemoryAccess(vInputScale)
                + CostUtility.GetMemoryAccess(qWeightScale) + CostUtility.GetMemoryAccess(kWeightScale) + CostUtility.GetMemoryAccess(vWeightScale),
            [CostFactorNames.MemoryStore] = output.Fields.Aggregate((UInt128)0, (sum, type) => sum + CostUtility.GetMemoryAccess(type)),
            [CostFactorNames.CPUCycles] = output.Fields.Aggregate((UInt128)0, (sum, type) => sum + CostUtility.GetCPUCycles(type, macPerElement)),
        };
    }

    private static Tensor Project(Tensor input, Tensor weight, IValue bias, IValue inputScale, IValue weightScale, DataType outputDataType)
    {
        _ = inputScale;
        var effectiveWeight = DequantizeWeight(weight, weightScale, input.ElementType);
        var result = Math.MatMulEvaluator.InferValue(input.ElementType, input, effectiveWeight, outputDataType).AsTensor();
        if (!IsNone(bias))
        {
            var biasTensor = bias.AsTensor().CastElementTo(result.ElementType);
            result = OrtKI.Add(result.ToOrtTensor(), biasTensor.ToOrtTensor()).ToTensor().CastElementTo(result.ElementType);
        }

        return result;
    }

    private static Tensor DequantizeWeight(Tensor weight, IValue weightScale, DataType outputType)
    {
        if (IsNone(weightScale))
        {
            return weight.ElementType == outputType ? weight : weight.CastElementTo(outputType);
        }

        var weightFloat = weight.CastElementTo(DataTypes.Float32).ToOrtTensor();
        var scaleFloat = weightScale.AsTensor().CastElementTo(DataTypes.Float32).ToOrtTensor();
        return OrtKI.Mul(weightFloat, scaleFloat).ToTensor().CastElementTo(outputType);
    }

    private static bool IsNone(IValue value) => value is NoneValue || value.Type is NoneType;

    private static IRType VisitProjection(IRType input, IRType weight, DataType outputDataType)
    {
        return (input, weight) switch
        {
            (DistributedType a, DistributedType b) => Math.MatMulEvaluator.VisitDistributedType(a, b with { TensorType = b.TensorType with { DType = a.TensorType.DType } }, NoneType.Default, outputDataType: outputDataType),
            (TensorType a, TensorType b) => Math.MatMulEvaluator.VisitTensorType(a, b with { DType = a.DType }, NoneType.Default, outputDataType: outputDataType),
            _ => new InvalidType($"QKVParallelLinear input/weight types are not supported: {input}, {weight}."),
        };
    }

    private static InvalidType? CheckBiasType(string name, IRType output, IRType bias)
    {
        if (bias is NoneType)
        {
            return null;
        }

        if (bias is not TensorType and not DistributedType)
        {
            return new InvalidType($"QKVParallelLinear {name} bias should be a tensor or None, got {bias}.");
        }

        var outDim = GetLastDimension(output);
        var biasDim = GetLastDimension(bias);
        if (outDim is { IsFixed: true } && biasDim is { IsFixed: true } && biasDim.FixedValue != outDim.FixedValue)
        {
            return new InvalidType($"QKVParallelLinear {name} bias last dimension {biasDim.FixedValue} does not match output dimension {outDim.FixedValue}.");
        }

        return null;
    }

    private static InvalidType? CheckHeadShape(QKVParallelLinear target, IRType q, IRType k, IRType v)
    {
        if (GetLastDimension(q) is { IsFixed: true } qDim && qDim.FixedValue % target.NumHeads != 0)
        {
            return new InvalidType($"QKVParallelLinear q dimension {qDim.FixedValue} is not divisible by num_heads {target.NumHeads}.");
        }

        foreach (var (name, type) in new[] { ("k", k), ("v", v) })
        {
            if (GetLastDimension(type) is { IsFixed: true } dim && dim.FixedValue % target.NumKvHeads != 0)
            {
                return new InvalidType($"QKVParallelLinear {name} dimension {dim.FixedValue} is not divisible by num_kv_heads {target.NumKvHeads}.");
            }
        }

        return null;
    }

    private static Dimension? GetLastDimension(IRType type)
    {
        var tensorType = type switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => distributed.TensorType,
            _ => null,
        };
        return tensorType?.Shape is RankedShape shape && shape.Rank > 0 ? shape[^1] : null;
    }

    private static uint GetK(IRType type)
    {
        var tensorType = type switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => DistributedUtility.GetDividedTensorType(distributed),
            _ => null,
        };
        if (tensorType?.Shape is RankedShape shape && shape.Rank > 0 && shape[^1].IsFixed)
        {
            return checked((uint)shape[^1].FixedValue);
        }

        return 1;
    }
}
