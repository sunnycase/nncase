// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
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
            Project(input, qWeight, context.GetArgumentValue(target, QKVParallelLinear.QBias), context.GetArgumentValue(target, QKVParallelLinear.QInputScale), context.GetArgumentValue(target, QKVParallelLinear.QWeightScale), target.OutputDataType, target.QuantizationMode),
            Project(input, kWeight, context.GetArgumentValue(target, QKVParallelLinear.KBias), context.GetArgumentValue(target, QKVParallelLinear.KInputScale), context.GetArgumentValue(target, QKVParallelLinear.KWeightScale), target.OutputDataType, target.QuantizationMode),
            Project(input, vWeight, context.GetArgumentValue(target, QKVParallelLinear.VBias), context.GetArgumentValue(target, QKVParallelLinear.VInputScale), context.GetArgumentValue(target, QKVParallelLinear.VWeightScale), target.OutputDataType, target.QuantizationMode));
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
        var qInputScale = context.CheckArgumentType<IRType>(target, QKVParallelLinear.QInputScale);
        var kInputScale = context.CheckArgumentType<IRType>(target, QKVParallelLinear.KInputScale);
        var vInputScale = context.CheckArgumentType<IRType>(target, QKVParallelLinear.VInputScale);
        var qWeightScale = context.CheckArgumentType<IRType>(target, QKVParallelLinear.QWeightScale);
        var kWeightScale = context.CheckArgumentType<IRType>(target, QKVParallelLinear.KWeightScale);
        var vWeightScale = context.CheckArgumentType<IRType>(target, QKVParallelLinear.VWeightScale);
        if (CheckScaleContract(
                target,
                qInputScale,
                kInputScale,
                vInputScale,
                qWeightScale,
                kWeightScale,
                vWeightScale) is { } scaleError)
        {
            return scaleError;
        }

        var q = VisitProjection(input, qWeight, qInputScale, qWeightScale, target, "q");
        var k = VisitProjection(input, kWeight, kInputScale, kWeightScale, target, "k");
        var v = VisitProjection(input, vWeight, vInputScale, vWeightScale, target, "v");
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
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(input)
                + CostUtility.GetMemoryAccess(qWeight) + CostUtility.GetMemoryAccess(kWeight) + CostUtility.GetMemoryAccess(vWeight)
                + CostUtility.GetMemoryAccess(qBias) + CostUtility.GetMemoryAccess(kBias) + CostUtility.GetMemoryAccess(vBias)
                + CostUtility.GetMemoryAccess(qInputScale) + CostUtility.GetMemoryAccess(kInputScale) + CostUtility.GetMemoryAccess(vInputScale)
                + CostUtility.GetMemoryAccess(qWeightScale) + CostUtility.GetMemoryAccess(kWeightScale) + CostUtility.GetMemoryAccess(vWeightScale),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = output.Fields.Aggregate((UInt128)0, (sum, type) => sum + CostUtility.GetMemoryAccess(type)),
            [CostFactorNames.CPUCycles] = output.Fields.Aggregate((UInt128)0, (sum, type) => sum + CostUtility.GetCPUCycles(type, macPerElement)),
        };
    }

    private static Tensor Project(
        Tensor input,
        Tensor weight,
        IValue bias,
        IValue inputScale,
        IValue weightScale,
        DataType outputDataType,
        MatMulQuantizationMode quantizationMode)
    {
        var result = quantizationMode switch
        {
            MatMulQuantizationMode.None => Math.MatMulEvaluator.InferValue(
                input.ElementType,
                input,
                weight.ElementType == input.ElementType ? weight : weight.CastElementTo(input.ElementType),
                outputDataType).AsTensor(),
            MatMulQuantizationMode.StaticTensor => Math.ScaledMatMulEvaluator.Evaluate(
                input,
                weight,
                inputScale.AsTensor(),
                weightScale.AsTensor(),
                outputDataType).AsTensor(),
            MatMulQuantizationMode.DynamicTensor => EvaluateDynamicTensorProjection(
                input,
                weight,
                weightScale.AsTensor(),
                outputDataType),
            _ => throw new NotSupportedException(
                $"Unsupported QKVParallelLinear quantization mode: {quantizationMode}."),
        };
        if (!IsNone(bias))
        {
            var biasTensor = bias.AsTensor().CastElementTo(result.ElementType);
            result = OrtKI.Add(result.ToOrtTensor(), biasTensor.ToOrtTensor()).ToTensor().CastElementTo(result.ElementType);
        }

        return result;
    }

    public static Tensor EvaluateDynamicTensorProjection(
        Tensor input,
        Tensor weight,
        Tensor weightScale,
        DataType outputDataType)
    {
        if (input.Rank != 2 || weight.Rank != 2 || weightScale.Rank != 1 ||
            input.Dimensions[1] != weight.Dimensions[0] ||
            weightScale.Dimensions[0] != weight.Dimensions[1])
        {
            throw new InvalidOperationException(
                "Dynamic-tensor QKV projection expects input=[M,K], weight=[K,N], and weight_scale=[N], " +
                $"got input=[{string.Join(",", input.Dimensions.ToArray())}], " +
                $"weight=[{string.Join(",", weight.Dimensions.ToArray())}], " +
                $"weight_scale=[{string.Join(",", weightScale.Dimensions.ToArray())}].");
        }

        var m = checked((int)input.Dimensions[0]);
        var k = checked((int)input.Dimensions[1]);
        var n = checked((int)weight.Dimensions[1]);
        var inputValues = input.CastElementTo(DataTypes.Float32).ToArray<float>();
        var normalizedInput = new float[inputValues.Length];
        var rowScales = new float[m];
        for (var row = 0; row < m; row++)
        {
            var maxAbs = 0F;
            for (var index = 0; index < k; index++)
            {
                maxAbs = System.Math.Max(
                    maxAbs,
                    System.Math.Abs(inputValues[(row * k) + index]));
            }

            var scale = System.Math.Max(maxAbs, 1E-12F) / (float)Float8E4M3.MaxNormal;
            rowScales[row] = scale;
            for (var index = 0; index < k; index++)
            {
                normalizedInput[(row * k) + index] = inputValues[(row * k) + index] / scale;
            }
        }

        var dequantizedInput = Tensor.From<float>(normalizedInput, input.Dimensions.ToArray())
            .CastElementTo(DataTypes.Float8E4M3)
            .CastElementTo(DataTypes.Float32)
            .ToArray<float>();
        for (var row = 0; row < m; row++)
        {
            for (var index = 0; index < k; index++)
            {
                dequantizedInput[(row * k) + index] *= rowScales[row];
            }
        }

        var dequantizedWeight = weight.CastElementTo(DataTypes.Float32).ToArray<float>();
        var weightScaleValues = weightScale.CastElementTo(DataTypes.Float32).ToArray<float>();
        for (var index = 0; index < k; index++)
        {
            for (var column = 0; column < n; column++)
            {
                dequantizedWeight[(index * n) + column] *= weightScaleValues[column];
            }
        }

        return OrtKI.MatMul(
                Tensor.From<float>(dequantizedInput, input.Dimensions.ToArray()).ToOrtTensor(),
                Tensor.From<float>(dequantizedWeight, weight.Dimensions.ToArray()).ToOrtTensor())
            .ToTensor()
            .CastElementTo(outputDataType);
    }

    private static bool IsNone(IValue value) => value is NoneValue || value.Type is NoneType;

    private static IRType VisitProjection(
        IRType input,
        IRType weight,
        IRType inputScale,
        IRType weightScale,
        QKVParallelLinear target,
        string name)
    {
        var output = target.QuantizationMode switch
        {
            MatMulQuantizationMode.None or MatMulQuantizationMode.DynamicTensor =>
                (input, weight) switch
                {
                    (DistributedType a, DistributedType b) => Math.MatMulEvaluator.VisitDistributedType(
                        a,
                        b with { TensorType = b.TensorType with { DType = a.TensorType.DType } },
                        NoneType.Default,
                        outputDataType: target.OutputDataType),
                    (TensorType a, TensorType b) => Math.MatMulEvaluator.VisitTensorType(
                        a,
                        b with { DType = a.DType },
                        NoneType.Default,
                        outputDataType: target.OutputDataType),
                    _ => new InvalidType($"QKVParallelLinear input/weight types are not supported: {input}, {weight}."),
                },
            MatMulQuantizationMode.StaticTensor => Math.ScaledMatMulEvaluator.InferType(
                new ScaledMatMul(target.OutputDataType),
                input,
                weight,
                inputScale,
                weightScale),
            _ => new InvalidType(
                $"Unsupported QKVParallelLinear quantization mode: {target.QuantizationMode}."),
        };
        if (output is InvalidType || target.QuantizationMode != MatMulQuantizationMode.DynamicTensor)
        {
            return output;
        }

        return CheckDynamicTensorWeightScale(name, output, weightScale) ?? output;
    }

    private static InvalidType? CheckScaleContract(
        QKVParallelLinear target,
        IRType qInputScale,
        IRType kInputScale,
        IRType vInputScale,
        IRType qWeightScale,
        IRType kWeightScale,
        IRType vWeightScale)
    {
        var inputScales = new[] { qInputScale, kInputScale, vInputScale };
        var weightScales = new[] { qWeightScale, kWeightScale, vWeightScale };
        var valid = target.QuantizationMode switch
        {
            MatMulQuantizationMode.None =>
                inputScales.All(scale => scale is NoneType) &&
                weightScales.All(scale => scale is NoneType),
            MatMulQuantizationMode.StaticTensor =>
                inputScales.All(Math.ScaledMatMulEvaluator.IsScaleType) &&
                weightScales.All(Math.ScaledMatMulEvaluator.IsScaleType),
            MatMulQuantizationMode.DynamicTensor =>
                inputScales.All(scale => scale is NoneType) &&
                weightScales.All(scale => scale is TensorType or DistributedType),
            _ => false,
        };
        return valid
            ? null
            : new InvalidType(
                $"QKVParallelLinear scale operands do not match quantization mode {target.QuantizationMode}; " +
                $"input scales=[{string.Join(",", inputScales.Select(scale => scale is not NoneType))}], " +
                $"weight scales=[{string.Join(",", weightScales.Select(scale => scale is not NoneType))}].");
    }

    private static InvalidType? CheckDynamicTensorWeightScale(
        string name,
        IRType output,
        IRType weightScale)
    {
        var outputTensor = GetTensorType(output);
        var scaleTensor = GetTensorType(weightScale);
        if (outputTensor is null || scaleTensor is null ||
            scaleTensor.DType is not PrimType scaleType || !scaleType.IsFloat() ||
            scaleTensor.Shape is not RankedShape { Rank: 1 } ||
            outputTensor.Shape is not RankedShape outputShape ||
            scaleTensor.Shape[0] != outputShape[^1])
        {
            return new InvalidType(
                $"Dynamic-tensor QKV {name} weight scale must be a floating [N] tensor matching the output, " +
                $"got scale={weightScale}, output={output}.");
        }

        if (output is DistributedType outputDistributed)
        {
            if (weightScale is not DistributedType scaleDistributed ||
                scaleDistributed.Placement != outputDistributed.Placement ||
                scaleDistributed.Partial is not null ||
                scaleDistributed.AxisPolicies[0] != outputDistributed.AxisPolicies[^1])
            {
                return new InvalidType(
                    $"Dynamic-tensor QKV {name} weight scale must use the output N distribution, " +
                    $"got scale={weightScale}, output={output}.");
            }
        }

        return null;
    }

    private static TensorType? GetTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => distributed.TensorType,
        _ => null,
    };

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
