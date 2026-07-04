// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="PackedQKVParallelLinear"/>.
/// </summary>
public sealed class PackedQKVParallelLinearEvaluator : IEvaluator<PackedQKVParallelLinear>, ITypeInferencer<PackedQKVParallelLinear>, ICostEvaluator<PackedQKVParallelLinear>
{
    public IValue Visit(IEvaluateContext context, PackedQKVParallelLinear target)
    {
        ValidateNoScales(
            context.GetArgumentValue(target, PackedQKVParallelLinear.QInputScale),
            context.GetArgumentValue(target, PackedQKVParallelLinear.KInputScale),
            context.GetArgumentValue(target, PackedQKVParallelLinear.VInputScale),
            context.GetArgumentValue(target, PackedQKVParallelLinear.QWeightScale),
            context.GetArgumentValue(target, PackedQKVParallelLinear.KWeightScale),
            context.GetArgumentValue(target, PackedQKVParallelLinear.VWeightScale));

        var input = context.GetArgumentValueAsTensor(target, PackedQKVParallelLinear.Input);
        return Value.FromTensors(
            Project(input, context.GetArgumentValueAsTensor(target, PackedQKVParallelLinear.QWeight), context.GetArgumentValue(target, PackedQKVParallelLinear.QBias), target.OutputDataType),
            Project(input, context.GetArgumentValueAsTensor(target, PackedQKVParallelLinear.KWeight), context.GetArgumentValue(target, PackedQKVParallelLinear.KBias), target.OutputDataType),
            Project(input, context.GetArgumentValueAsTensor(target, PackedQKVParallelLinear.VWeight), context.GetArgumentValue(target, PackedQKVParallelLinear.VBias), target.OutputDataType));
    }

    public IRType Visit(ITypeInferenceContext context, PackedQKVParallelLinear target)
    {
        var input = context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.Input);
        var qWeight = context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.QWeight);
        var kWeight = context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.KWeight);
        var vWeight = context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.VWeight);
        var qBias = context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.QBias);
        var kBias = context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.KBias);
        var vBias = context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.VBias);
        var scaleCheck = CheckScales(
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.QInputScale),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.KInputScale),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.VInputScale),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.QWeightScale),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.KWeightScale),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.VWeightScale));
        if (scaleCheck is not null)
        {
            return scaleCheck;
        }

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

        return new TupleType(new[] { q, k, v });
    }

    public Cost Visit(ICostEvaluateContext context, PackedQKVParallelLinear target)
    {
        var input = context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.Input);
        var qWeight = context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.QWeight);
        var kWeight = context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.KWeight);
        var vWeight = context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.VWeight);
        var qBias = context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.QBias);
        var kBias = context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.KBias);
        var vBias = context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.VBias);
        var output = context.GetReturnType<TupleType>();
        if (TryGetTargetCost(context, target, input, qWeight, kWeight, vWeight, output, out var targetCost))
        {
            AddBiasCost(targetCost, output, qBias, kBias, vBias);
            return targetCost;
        }

        var macPerElement = GetK(input);
        var cost = new Cost()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(input)
                + CostUtility.GetMemoryAccess(qWeight) + CostUtility.GetMemoryAccess(kWeight) + CostUtility.GetMemoryAccess(vWeight)
                + CostUtility.GetMemoryAccess(qBias) + CostUtility.GetMemoryAccess(kBias) + CostUtility.GetMemoryAccess(vBias),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = output.Fields.Aggregate((UInt128)0, (sum, type) => sum + CostUtility.GetMemoryAccess(type)),
            [CostFactorNames.CPUCycles] = output.Fields.Aggregate((UInt128)0, (sum, type) => sum + CostUtility.GetCPUCycles(type, macPerElement)),
        };

        return cost;
    }

    private static Tensor Project(Tensor input, Tensor packedWeight, IValue packedBias, DataType outputDataType)
    {
        var weightVectorType = (VectorType)packedWeight.ElementType;
        var nr = weightVectorType.Lanes[0];
        var nLanes = weightVectorType.Lanes[1];
        var weightOrt = packedWeight.ToOrtTensor();
        var rN = packedWeight.Rank - 2;
        weightOrt = weightOrt.Unpack(weightVectorType.Lanes.Count, [rN, rN]);

        var perm = Enumerable.Range(0, weightOrt.Rank).Select(i => (long)i).ToArray();
        (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
        weightOrt = OrtKI.Transpose(weightOrt, perm);

        var result = Math.MatMulEvaluator.InferValue(input.ElementType, input, weightOrt.ToTensor(), outputDataType).AsTensor().ToOrtTensor();
        result = result.Pack(0, [nr, nLanes], [result.Rank - 1, result.Rank - 1]);
        if (!IsNone(packedBias))
        {
            result = OrtKI.Add(result, packedBias.AsTensor().ToOrtTensor());
        }

        return result.ToTensor(new VectorType(outputDataType, [nr, nLanes]));
    }

    private static IRType VisitProjection(IRType input, IRType packedWeight, DataType outputDataType)
    {
        return (input, packedWeight) switch
        {
            (DistributedType a, DistributedType b) => PackOutput(Math.MatMulEvaluator.VisitDistributedType(a, b with { TensorType = UnpackedBType(b.TensorType) }, NoneType.Default, dimInfo: VectorizedMatMul.GetDimInfo(false, true, a.TensorType.Shape.Rank, UnpackedBType(b.TensorType).Shape.Rank), transB: true, outputDataType: outputDataType), GetNr(b.TensorType)),
            (TensorType a, TensorType b) => PackOutput(Math.MatMulEvaluator.VisitTensorType(a, UnpackedBType(b), NoneType.Default, dimInfo: VectorizedMatMul.GetDimInfo(false, true, a.Shape.Rank, UnpackedBType(b).Shape.Rank), outputDataType: outputDataType), GetNr(b)),
            _ => new InvalidType($"PackedQKVParallelLinear input/weight types are not supported: {input}, {packedWeight}."),
        };
    }

    private static IRType PackOutput(IRType output, int nr) => output switch
    {
        DistributedType distributed => distributed with { TensorType = (TensorType)TypeInference.PackType(distributed.TensorType, [nr], [distributed.TensorType.Shape.Rank - 1]) },
        TensorType tensor => TypeInference.PackType(tensor, [nr], [tensor.Shape.Rank - 1]),
        _ => output,
    };

    private static TensorType UnpackedBType(TensorType tensorType)
    {
        var vectorType = (VectorType)tensorType.DType;
        var nr = vectorType.Lanes[0];
        var nLanes = vectorType.Lanes[1];
        var newShape = tensorType.Shape.ToArray();
        newShape[^2] *= nr;
        return tensorType with { DType = vectorType with { Lanes = [nLanes] }, Shape = newShape };
    }

    private static int GetNr(TensorType tensorType) => ((VectorType)tensorType.DType).Lanes[0];

    private static bool TryGetTargetCost(
        ICostEvaluateContext context,
        PackedQKVParallelLinear target,
        IRType input,
        IRType qWeight,
        IRType kWeight,
        IRType vWeight,
        TupleType output,
        out Cost cost)
    {
        cost = Cost.Zero;
        if (output.Fields.Count != 3
            || !TargetCostTensor.TryFromType(input, out var inputTensor))
        {
            return false;
        }

        var weights = new[] { qWeight, kWeight, vWeight };
        for (int i = 0; i < weights.Length; i++)
        {
            if (!TargetCostTensor.TryFromType(weights[i], out var weightTensor)
                || !TargetCostTensor.TryFromType(output.Fields[i], out var outputTensor)
                || !context.TargetCostModel.TryGetMatMulCost(
                    new(inputTensor, weightTensor, outputTensor, GetScalarType(target.OutputDataType), MatMulOpCostKind.Simt),
                    out var projectionCost))
            {
                cost = Cost.Zero;
                return false;
            }

            cost += projectionCost;
        }

        if (TryGetMemoryBytes(input, out var inputBytes))
        {
            SubtractCostFactor(cost, CostFactorNames.BlockLocalMemoryLoadBytes, inputBytes * 2);
        }

        return true;
    }

    private static void AddBiasCost(Cost cost, TupleType outputType, IRType qBias, IRType kBias, IRType vBias)
    {
        var biases = new[] { qBias, kBias, vBias };
        for (int i = 0; i < biases.Length; i++)
        {
            if (biases[i] is NoneType)
            {
                continue;
            }

            AddCostFactor(cost, CostFactorNames.BlockLocalMemoryLoadBytes, CostUtility.GetMemoryAccess(biases[i]));
            AddCostFactor(cost, CostFactorNames.CPUCycles, CostUtility.GetCPUCycles(outputType.Fields[i], 1));
        }
    }

    private static void AddCostFactor(Cost cost, string name, UInt128 value)
    {
        if (value == 0)
        {
            return;
        }

        if (cost.Factors.TryGetValue(name, out var oldValue))
        {
            cost.Factors[name] = oldValue + value;
        }
        else
        {
            cost.Factors.Add(name, value);
        }
    }

    private static void SubtractCostFactor(Cost cost, string name, UInt128 value)
    {
        if (value == 0 || !cost.Factors.TryGetValue(name, out var oldValue))
        {
            return;
        }

        cost.Factors[name] = oldValue > value ? oldValue - value : 0;
    }

    private static bool TryGetMemoryBytes(IRType type, out UInt128 count)
    {
        count = CostUtility.GetMemoryAccess(type);
        return count > 0;
    }

    private static DataType GetScalarType(DataType dtype) => dtype switch
    {
        VectorType vectorType => GetScalarType(vectorType.ElemType),
        _ => dtype,
    };

    private static InvalidType? CheckBiasType(string name, IRType output, IRType bias)
    {
        if (bias is NoneType)
        {
            return null;
        }

        if (bias is not TensorType and not DistributedType)
        {
            return new InvalidType($"PackedQKVParallelLinear {name} bias should be a packed tensor or None, got {bias}.");
        }

        if (GetLastDimension(output) is { IsFixed: true } outDim && GetLastDimension(bias) is { IsFixed: true } biasDim && biasDim.FixedValue != outDim.FixedValue)
        {
            return new InvalidType($"PackedQKVParallelLinear {name} bias last dimension {biasDim.FixedValue} does not match packed output dimension {outDim.FixedValue}.");
        }

        return null;
    }

    private static InvalidType? CheckScales(params IRType[] scales)
    {
        return scales.All(scale => scale is NoneType)
            ? null
            : new InvalidType("PackedQKVParallelLinear currently supports only None input/weight scales.");
    }

    private static void ValidateNoScales(params IValue[] scales)
    {
        if (scales.Any(scale => !IsNone(scale)))
        {
            throw new NotSupportedException("PackedQKVParallelLinear currently supports only None input/weight scales.");
        }
    }

    private static bool IsNone(IValue value) => value is NoneValue || value.Type is NoneType;

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
