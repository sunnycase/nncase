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
            Project(input, context.GetArgumentValueAsTensor(target, PackedQKVParallelLinear.QWeight), context.GetArgumentValue(target, PackedQKVParallelLinear.QBias), target.OutputDataType, target.RhsLayout),
            Project(input, context.GetArgumentValueAsTensor(target, PackedQKVParallelLinear.KWeight), context.GetArgumentValue(target, PackedQKVParallelLinear.KBias), target.OutputDataType, target.RhsLayout),
            Project(input, context.GetArgumentValueAsTensor(target, PackedQKVParallelLinear.VWeight), context.GetArgumentValue(target, PackedQKVParallelLinear.VBias), target.OutputDataType, target.RhsLayout));
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

        var q = VisitProjection(input, qWeight, target.OutputDataType, target.RhsLayout);
        var k = VisitProjection(input, kWeight, target.OutputDataType, target.RhsLayout);
        var v = VisitProjection(input, vWeight, target.OutputDataType, target.RhsLayout);
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

    private static Tensor Project(
        Tensor input,
        Tensor packedWeight,
        IValue packedBias,
        DataType outputDataType,
        PackedMatMulRhsLayout rhsLayout)
    {
        string? errorMessage = null;
        if (packedWeight.ElementType is not VectorType weightVectorType ||
            !TryGetLayoutInfo(
                rhsLayout,
                weightVectorType,
                packedWeight.Rank,
                out var rhsUnpackAxes,
                out var outputLanes,
                out var transposeB,
                out errorMessage))
        {
            throw new InvalidOperationException(
                errorMessage ?? $"PackedQKVParallelLinear expects a vector weight, got {packedWeight.ElementType}.");
        }

        var weightOrt = packedWeight.ToOrtTensor();
        weightOrt = weightOrt.Unpack(weightVectorType.Lanes.Count, rhsUnpackAxes);
        if (transposeB)
        {
            var perm = Enumerable.Range(0, weightOrt.Rank).Select(i => (long)i).ToArray();
            (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
            weightOrt = OrtKI.Transpose(weightOrt, perm);
        }

        var result = Math.MatMulEvaluator.InferValue(input.ElementType, input, weightOrt.ToTensor(), outputDataType).AsTensor().ToOrtTensor();
        result = result.Pack(
            0,
            outputLanes,
            Enumerable.Repeat(result.Rank - 1, outputLanes.Length).ToArray());
        if (!IsNone(packedBias))
        {
            result = OrtKI.Add(result, packedBias.AsTensor().ToOrtTensor());
        }

        return result.ToTensor(new VectorType(outputDataType, outputLanes));
    }

    private static IRType VisitProjection(
        IRType input,
        IRType packedWeight,
        DataType outputDataType,
        PackedMatMulRhsLayout rhsLayout)
    {
        switch (input, packedWeight)
        {
            case (DistributedType a, DistributedType b):
                {
                    string? errorMessage = null;
                    if (b.TensorType.DType is not VectorType vectorType ||
                        !TryGetLayoutInfo(
                            rhsLayout,
                            vectorType,
                            b.TensorType.Shape.Rank,
                            out var rhsUnpackAxes,
                            out var outputLanes,
                            out var transposeB,
                            out errorMessage))
                    {
                        return new InvalidType(
                            errorMessage ?? $"PackedQKVParallelLinear expects a vector weight, got {b.TensorType.DType}.");
                    }

                    var unpackedWeightType = UnpackType(b, rhsUnpackAxes);
                    if (unpackedWeightType is not DistributedType unpackedWeight)
                    {
                        return unpackedWeightType;
                    }

                    var dimInfo = VectorizedMatMul.GetDimInfo(
                        false,
                        transposeB,
                        a.TensorType.Shape.Rank,
                        unpackedWeight.TensorType.Shape.Rank);
                    if (a.AxisPolicies[dimInfo.Lk] != unpackedWeight.AxisPolicies[dimInfo.Rk])
                    {
                        return new InvalidType(
                            "PackedQKVParallelLinear requires input and weight reduction axes to use the same " +
                            $"distributed policy, got input={a.AxisPolicies[dimInfo.Lk]} and " +
                            $"weight={unpackedWeight.AxisPolicies[dimInfo.Rk]}.");
                    }

                    return PackOutput(
                        Math.MatMulEvaluator.VisitDistributedType(
                            a,
                            unpackedWeight,
                            NoneType.Default,
                            dimInfo: dimInfo,
                            transB: transposeB,
                            outputDataType: outputDataType),
                        outputLanes);
                }

            case (TensorType a, TensorType b):
                {
                    string? errorMessage = null;
                    if (b.DType is not VectorType vectorType ||
                        !TryGetLayoutInfo(
                            rhsLayout,
                            vectorType,
                            b.Shape.Rank,
                            out var rhsUnpackAxes,
                            out var outputLanes,
                            out var transposeB,
                            out errorMessage))
                    {
                        return new InvalidType(
                            errorMessage ?? $"PackedQKVParallelLinear expects a vector weight, got {b.DType}.");
                    }

                    var unpackedWeightType = UnpackType(b, rhsUnpackAxes);
                    if (unpackedWeightType is not TensorType unpackedWeight)
                    {
                        return unpackedWeightType;
                    }

                    var dimInfo = VectorizedMatMul.GetDimInfo(
                        false,
                        transposeB,
                        a.Shape.Rank,
                        unpackedWeight.Shape.Rank);
                    return PackOutput(
                        Math.MatMulEvaluator.VisitTensorType(
                            a,
                            unpackedWeight,
                            NoneType.Default,
                            dimInfo: dimInfo,
                            outputDataType: outputDataType),
                        outputLanes);
                }

            default:
                return new InvalidType(
                    $"PackedQKVParallelLinear input/weight types are not supported: {input}, {packedWeight}.");
        }
    }

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
        _ => output,
    };

    private static bool TryGetLayoutInfo(
        PackedMatMulRhsLayout layout,
        VectorType vectorType,
        int weightRank,
        out int[] rhsUnpackAxes,
        out int[] outputLanes,
        out bool transposeB,
        out string? errorMessage)
    {
        switch (layout, vectorType.Lanes.Count)
        {
            case (PackedMatMulRhsLayout.NMajor, 2):
                rhsUnpackAxes = [weightRank - 2, weightRank - 2];
                outputLanes = vectorType.Lanes.ToArray();
                transposeB = true;
                errorMessage = null;
                return true;
            case (PackedMatMulRhsLayout.KMajor, 3):
                rhsUnpackAxes = [weightRank - 1, weightRank - 2, weightRank - 2];
                outputLanes = [vectorType.Lanes[0]];
                transposeB = false;
                errorMessage = null;
                return true;
            default:
                rhsUnpackAxes = [];
                outputLanes = [];
                transposeB = false;
                errorMessage =
                    $"PackedQKVParallelLinear {layout} expects " +
                    $"{(layout == PackedMatMulRhsLayout.NMajor ? 2 : 3)} weight vector lanes, " +
                    $"got [{string.Join(",", vectorType.Lanes)}].";
                return false;
        }
    }

    private static IRType UnpackType(IRType input, int[] axes) => input switch
    {
        DistributedType distributed => TypeInference.UnpackType(distributed, axes),
        TensorType tensor => TypeInference.UnpackType(tensor, axes),
        _ => new InvalidType($"Cannot unpack {input} with axes [{string.Join(",", axes)}]."),
    };

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
            var logicalWeight = weights[i];
            var logicalOutput = output.Fields[i];
            if (target.RhsLayout == PackedMatMulRhsLayout.KMajor)
            {
                var weightTensorType = GetTensorType(logicalWeight);
                var outputTensorType = GetTensorType(logicalOutput);
                if (weightTensorType?.DType is not VectorType weightVectorType ||
                    !TryGetLayoutInfo(
                        target.RhsLayout,
                        weightVectorType,
                        weightTensorType.Shape.Rank,
                        out var rhsUnpackAxes,
                        out var outputLanes,
                        out _,
                        out _) ||
                    outputTensorType?.DType is not VectorType outputVectorType ||
                    !outputVectorType.Lanes.SequenceEqual(outputLanes))
                {
                    cost = Cost.Zero;
                    return false;
                }

                logicalWeight = UnpackType(logicalWeight, rhsUnpackAxes);
                logicalOutput = UnpackType(
                    logicalOutput,
                    Enumerable.Repeat(outputTensorType.Shape.Rank - 1, outputLanes.Length).ToArray());
                if (logicalWeight is InvalidType || logicalOutput is InvalidType)
                {
                    cost = Cost.Zero;
                    return false;
                }
            }

            if (!TargetCostTensor.TryFromType(logicalWeight, out var weightTensor)
                || !TargetCostTensor.TryFromType(logicalOutput, out var outputTensor)
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

    private static TensorType? GetTensorType(IRType type) => type switch
    {
        TensorType tensorType => tensorType,
        DistributedType distributedType => distributedType.TensorType,
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
