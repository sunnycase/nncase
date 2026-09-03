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
using ScaledMatMulEvaluator = Nncase.Evaluator.Math.ScaledMatMulEvaluator;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="PackedQKVParallelLinear"/>.
/// </summary>
public sealed class PackedQKVParallelLinearEvaluator : IEvaluator<PackedQKVParallelLinear>, ITypeInferencer<PackedQKVParallelLinear>, ICostEvaluator<PackedQKVParallelLinear>
{
    public IValue Visit(IEvaluateContext context, PackedQKVParallelLinear target)
    {
        var input = context.GetArgumentValueAsTensor(target, PackedQKVParallelLinear.Input);
        return Value.FromTensors(
            Project(input, context.GetArgumentValueAsTensor(target, PackedQKVParallelLinear.QWeight), context.GetArgumentValue(target, PackedQKVParallelLinear.QBias), context.GetArgumentValue(target, PackedQKVParallelLinear.QInputScale), context.GetArgumentValue(target, PackedQKVParallelLinear.QWeightScale), target),
            Project(input, context.GetArgumentValueAsTensor(target, PackedQKVParallelLinear.KWeight), context.GetArgumentValue(target, PackedQKVParallelLinear.KBias), context.GetArgumentValue(target, PackedQKVParallelLinear.KInputScale), context.GetArgumentValue(target, PackedQKVParallelLinear.KWeightScale), target),
            Project(input, context.GetArgumentValueAsTensor(target, PackedQKVParallelLinear.VWeight), context.GetArgumentValue(target, PackedQKVParallelLinear.VBias), context.GetArgumentValue(target, PackedQKVParallelLinear.VInputScale), context.GetArgumentValue(target, PackedQKVParallelLinear.VWeightScale), target));
    }

    public IRType Visit(ITypeInferenceContext context, PackedQKVParallelLinear target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.Input),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.QWeight),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.KWeight),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.VWeight),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.QBias),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.KBias),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.VBias),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.QInputScale),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.KInputScale),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.VInputScale),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.QWeightScale),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.KWeightScale),
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinear.VWeightScale));

    internal static IRType InferType(
        PackedQKVParallelLinear target,
        IRType input,
        IRType qWeight,
        IRType kWeight,
        IRType vWeight,
        IRType qBias,
        IRType kBias,
        IRType vBias,
        IRType qInputScale,
        IRType kInputScale,
        IRType vInputScale,
        IRType qWeightScale,
        IRType kWeightScale,
        IRType vWeightScale)
    {
        var scaleCheck = CheckScaleContract(
            target,
            qInputScale,
            kInputScale,
            vInputScale,
            qWeightScale,
            kWeightScale,
            vWeightScale);
        if (scaleCheck is not null)
        {
            return scaleCheck;
        }

        var q = InferProjectionType("q", input, qWeight, qInputScale, qWeightScale, target);
        var k = InferProjectionType("k", input, kWeight, kInputScale, kWeightScale, target);
        var v = InferProjectionType("v", input, vWeight, vInputScale, vWeightScale, target);
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
        var scales = new[]
        {
            context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.QInputScale),
            context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.KInputScale),
            context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.VInputScale),
            context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.QWeightScale),
            context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.KWeightScale),
            context.GetArgumentType<IRType>(target, PackedQKVParallelLinear.VWeightScale),
        };
        var output = context.GetReturnType<TupleType>();
        if (TryGetTargetCost(
                context,
                target,
                input,
                qWeight,
                kWeight,
                vWeight,
                qBias,
                kBias,
                vBias,
                output,
                out var targetCost))
        {
            AddBiasCost(targetCost, output, qBias, kBias, vBias);
            return targetCost;
        }

        var macPerElement = GetK(input);
        var cost = new Cost()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(input)
                + CostUtility.GetMemoryAccess(qWeight) + CostUtility.GetMemoryAccess(kWeight) + CostUtility.GetMemoryAccess(vWeight)
                + CostUtility.GetMemoryAccess(qBias) + CostUtility.GetMemoryAccess(kBias) + CostUtility.GetMemoryAccess(vBias)
                + scales.Aggregate((UInt128)0, (sum, scale) => sum + CostUtility.GetMemoryAccess(scale)),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = output.Fields.Aggregate((UInt128)0, (sum, type) => sum + CostUtility.GetMemoryAccess(type)),
            [CostFactorNames.CPUCycles] = output.Fields.Aggregate((UInt128)0, (sum, type) => sum + CostUtility.GetCPUCycles(type, macPerElement)),
        };

        return cost;
    }

    private static Tensor Project(
        Tensor input,
        Tensor packedWeight,
        IValue packedBias,
        IValue inputScale,
        IValue weightScale,
        PackedQKVParallelLinear target)
    {
        string? errorMessage = null;
        if (packedWeight.ElementType is not VectorType weightVectorType ||
            !TryGetLayoutInfo(
                target.RhsLayout,
                weightVectorType,
                packedWeight.Rank,
                target.OutputNVectorLaneCount,
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

        var logicalWeight = weightOrt.ToTensor();
        var result = target.QuantizationMode switch
        {
            MatMulQuantizationMode.None => Math.MatMulEvaluator.InferValue(
                input.ElementType,
                input,
                logicalWeight,
                target.OutputDataType).AsTensor().ToOrtTensor(),
            MatMulQuantizationMode.StaticTensor => ScaledMatMulEvaluator.Evaluate(
                input,
                logicalWeight,
                inputScale.AsTensor(),
                weightScale.AsTensor(),
                target.OutputDataType).AsTensor().ToOrtTensor(),
            MatMulQuantizationMode.DynamicTensor =>
                Nncase.Evaluator.NN.QKVParallelLinearEvaluator.EvaluateDynamicTensorProjection(
                    input,
                    logicalWeight,
                    UnpackDynamicTensorWeightScale(weightScale.AsTensor(), outputLanes),
                    target.OutputDataType).ToOrtTensor(),
            _ => throw new NotSupportedException(
                $"Unsupported PackedQKVParallelLinear quantization mode: {target.QuantizationMode}."),
        };
        result = result.Pack(
            0,
            outputLanes,
            Enumerable.Repeat(result.Rank - 1, outputLanes.Length).ToArray());
        if (!IsNone(packedBias))
        {
            result = OrtKI.Add(result, packedBias.AsTensor().ToOrtTensor());
        }

        return result.ToTensor(new VectorType(target.OutputDataType, outputLanes));
    }

    internal static IRType InferProjectionType(
        string name,
        IRType input,
        IRType packedWeight,
        IRType inputScale,
        IRType weightScale,
        PackedQKVParallelLinear target)
    {
        switch (input, packedWeight)
        {
            case (DistributedType a, DistributedType b):
                {
                    string? errorMessage = null;
                    if (b.TensorType.DType is not VectorType vectorType ||
                        !TryGetLayoutInfo(
                            target.RhsLayout,
                            vectorType,
                            b.TensorType.Shape.Rank,
                            target.OutputNVectorLaneCount,
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

                    var projection = target.QuantizationMode switch
                    {
                        MatMulQuantizationMode.None => Math.MatMulEvaluator.VisitDistributedType(
                            a,
                            unpackedWeight,
                            NoneType.Default,
                            dimInfo: dimInfo,
                            transB: transposeB,
                            outputDataType: target.OutputDataType),
                        MatMulQuantizationMode.StaticTensor => ScaledMatMulEvaluator.InferType(
                            new ScaledMatMul(target.OutputDataType),
                            a,
                            unpackedWeight,
                            inputScale,
                            weightScale),
                        MatMulQuantizationMode.DynamicTensor => Math.MatMulEvaluator.VisitDistributedType(
                            a,
                            unpackedWeight with
                            {
                                TensorType = unpackedWeight.TensorType with { DType = a.TensorType.DType },
                            },
                            NoneType.Default,
                            dimInfo: dimInfo,
                            transB: transposeB,
                            outputDataType: target.OutputDataType),
                        _ => new InvalidType(
                            $"Unsupported PackedQKVParallelLinear quantization mode: {target.QuantizationMode}."),
                    };
                    var packedOutput = PackOutput(projection, outputLanes);
                    return target.QuantizationMode == MatMulQuantizationMode.DynamicTensor
                        ? CheckDynamicTensorWeightScale(name, packedOutput, weightScale, outputLanes) ?? packedOutput
                        : packedOutput;
                }

            case (TensorType a, TensorType b):
                {
                    string? errorMessage = null;
                    if (b.DType is not VectorType vectorType ||
                        !TryGetLayoutInfo(
                            target.RhsLayout,
                            vectorType,
                            b.Shape.Rank,
                            target.OutputNVectorLaneCount,
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
                    var projection = target.QuantizationMode switch
                    {
                        MatMulQuantizationMode.None => Math.MatMulEvaluator.VisitTensorType(
                            a,
                            unpackedWeight,
                            NoneType.Default,
                            dimInfo: dimInfo,
                            outputDataType: target.OutputDataType),
                        MatMulQuantizationMode.StaticTensor => ScaledMatMulEvaluator.InferType(
                            new ScaledMatMul(target.OutputDataType),
                            a,
                            unpackedWeight,
                            inputScale,
                            weightScale),
                        MatMulQuantizationMode.DynamicTensor => Math.MatMulEvaluator.VisitTensorType(
                            a,
                            unpackedWeight with { DType = a.DType },
                            NoneType.Default,
                            dimInfo: dimInfo,
                            outputDataType: target.OutputDataType),
                        _ => new InvalidType(
                            $"Unsupported PackedQKVParallelLinear quantization mode: {target.QuantizationMode}."),
                    };
                    var packedOutput = PackOutput(projection, outputLanes);
                    return target.QuantizationMode == MatMulQuantizationMode.DynamicTensor
                        ? CheckDynamicTensorWeightScale(name, packedOutput, weightScale, outputLanes) ?? packedOutput
                        : packedOutput;
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

    internal static bool TryGetLayoutInfo(
        PackedMatMulRhsLayout layout,
        VectorType vectorType,
        int weightRank,
        int outputNVectorLaneCount,
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
            case (PackedMatMulRhsLayout.NMajorKPacked, 2) when outputNVectorLaneCount > 0:
                rhsUnpackAxes = [weightRank - 1, weightRank - 1];
                outputLanes = [outputNVectorLaneCount];
                transposeB = true;
                errorMessage = null;
                return true;
            default:
                rhsUnpackAxes = [];
                outputLanes = [];
                transposeB = false;
                errorMessage =
                    $"PackedQKVParallelLinear {layout} expects " +
                    $"{(layout == PackedMatMulRhsLayout.KMajor ? 3 : 2)} weight vector lanes " +
                    $"and a positive output N vector lane count, got " +
                    $"lanes=[{string.Join(",", vectorType.Lanes)}], " +
                    $"output_n_lanes={outputNVectorLaneCount}.";
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
        IRType qBias,
        IRType kBias,
        IRType vBias,
        TupleType output,
        out Cost cost)
    {
        cost = Cost.Zero;
        if (output.Fields.Count != 3
            || !TargetCostTensor.TryFromType(input, out var inputTensor))
        {
            return false;
        }

        var matmulProfile = GetMatMulCostProfile(
            target,
            inputTensor,
            output,
            qBias,
            kBias,
            vBias);
        var weights = new[] { qWeight, kWeight, vWeight };
        for (int i = 0; i < weights.Length; i++)
        {
            var logicalWeight = weights[i];
            var logicalOutput = output.Fields[i];
            if (target.RhsLayout is PackedMatMulRhsLayout.KMajor or PackedMatMulRhsLayout.NMajorKPacked)
            {
                var weightTensorType = GetTensorType(logicalWeight);
                var outputTensorType = GetTensorType(logicalOutput);
                if (weightTensorType?.DType is not VectorType weightVectorType ||
                    !TryGetLayoutInfo(
                        target.RhsLayout,
                        weightVectorType,
                        weightTensorType.Shape.Rank,
                        target.OutputNVectorLaneCount,
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
                    new(inputTensor, weightTensor, outputTensor, GetScalarType(target.OutputDataType), matmulProfile.Kind),
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

        if (matmulProfile.KTileCount > 0 &&
            cost.Factors.TryGetValue(CostFactorNames.CPUCycles, out var computeCycles))
        {
            AddCostFactor(
                cost,
                CostFactorNames.PipelineDrainCycles,
                DivideRoundUp(computeCycles, (UInt128)matmulProfile.KTileCount));
        }

        return true;
    }

    private static MatMulCostProfile GetMatMulCostProfile(
        PackedQKVParallelLinear target,
        TargetCostTensor input,
        TupleType output,
        params IRType[] biases)
    {
        if (target.RhsLayout != PackedMatMulRhsLayout.KMajor ||
            biases.Any(bias => bias is not NoneType) ||
            GetScalarType(input.DType) != DataTypes.BFloat16 ||
            !TryGetFixedShape(input, out var inputShape) ||
            inputShape is not [1, _] ||
            output.Fields.Cast<IRType>().ToArray() is not { Length: 3 } fields ||
            fields.Any(field => field is not DistributedType))
        {
            return MatMulCostProfile.Simt;
        }

        var distributedOutputs = fields.Cast<DistributedType>().ToArray();
        long totalN = 0;
        foreach (var outputType in distributedOutputs)
        {
            if (!TargetCostTensor.TryFromType(outputType, out var outputTensor) ||
                GetScalarType(outputTensor.DType) != DataTypes.BFloat16 ||
                !TryGetFixedShape(outputTensor, out var outputShape) ||
                outputShape.Length != 2 ||
                outputShape[0] != 1)
            {
                return MatMulCostProfile.Simt;
            }

            totalN = checked(totalN +
                (outputShape[1] * GetVectorLaneCount(outputTensor.DType)));
        }

        var partial = distributedOutputs[0].Partial;
        var splitKProfile =
            inputShape[1] == 256 &&
            totalN == 256 &&
            partial is { Op: ReduceOp.Sum } &&
            distributedOutputs.Skip(1).All(outputType =>
                outputType.Partial is { Op: ReduceOp.Sum } other &&
                other.Axes.SequenceEqual(partial.Axes));
        var directProfile =
            inputShape[1] == 2048 &&
            totalN == 32 &&
            distributedOutputs.All(outputType => outputType.Partial is null);

        return splitKProfile
            ? new(MatMulOpCostKind.Mma, inputShape[1] / 64)
            : directProfile
                ? new(MatMulOpCostKind.Mma, inputShape[1] / 1024)
                : MatMulCostProfile.Simt;
    }

    private static UInt128 DivideRoundUp(UInt128 value, UInt128 divisor)
        => (value + divisor - 1) / divisor;

    private static bool TryGetFixedShape(
        TargetCostTensor tensor,
        out long[] shape)
        => CompilerServices.TryGetMaxShape(tensor.Shape, out shape) &&
            tensor.Shape.IsFixed;

    private static int GetVectorLaneCount(DataType dtype) => dtype switch
    {
        VectorType vectorType => checked(
            vectorType.Lanes.Aggregate(1, static (product, lane) => product * lane) *
            GetVectorLaneCount(vectorType.ElemType)),
        _ => 1,
    };

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

    private readonly record struct MatMulCostProfile(MatMulOpCostKind Kind, long KTileCount)
    {
        public static MatMulCostProfile Simt => new(MatMulOpCostKind.Simt, 0);
    }

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

    private static InvalidType? CheckScaleContract(
        PackedQKVParallelLinear target,
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
                inputScales.All(ScaledMatMulEvaluator.IsScaleType) &&
                weightScales.All(ScaledMatMulEvaluator.IsScaleType),
            MatMulQuantizationMode.DynamicTensor =>
                inputScales.All(scale => scale is NoneType) &&
                weightScales.All(scale => scale is TensorType or DistributedType),
            _ => false,
        };
        return valid
            ? null
            : new InvalidType(
                $"PackedQKVParallelLinear scale operands do not match quantization mode {target.QuantizationMode}; " +
                $"input scales=[{string.Join(",", inputScales.Select(scale => scale is not NoneType))}], " +
                $"weight scales=[{string.Join(",", weightScales.Select(scale => scale is not NoneType))}].");
    }

    private static InvalidType? CheckDynamicTensorWeightScale(
        string name,
        IRType output,
        IRType weightScale,
        int[] outputLanes)
    {
        var outputTensor = GetTensorType(output);
        var scaleTensor = GetTensorType(weightScale);
        if (outputTensor?.DType is not VectorType outputVectorType ||
            scaleTensor?.DType is not VectorType scaleVectorType ||
            scaleVectorType.ElemType is not PrimType scaleType || !scaleType.IsFloat() ||
            !outputVectorType.Lanes.SequenceEqual(outputLanes) ||
            !scaleVectorType.Lanes.SequenceEqual(outputLanes) ||
            outputTensor.Shape is not RankedShape outputShape ||
            scaleTensor.Shape is not RankedShape { Rank: 1 } scaleShape ||
            scaleShape[0] != outputShape[^1])
        {
            return new InvalidType(
                $"Dynamic-tensor PackedQKVParallelLinear {name} weight scale must be a packed floating " +
                $"[N/{string.Join("x", outputLanes)}]<{string.Join(",", outputLanes)}> tensor matching the output, " +
                $"got scale={weightScale}, output={output}.");
        }

        if (output is DistributedType outputDistributed &&
            (weightScale is not DistributedType scaleDistributed ||
             scaleDistributed.Placement != outputDistributed.Placement ||
             scaleDistributed.Partial is not null ||
             scaleDistributed.AxisPolicies[0] != outputDistributed.AxisPolicies[^1]))
        {
            return new InvalidType(
                $"Dynamic-tensor PackedQKVParallelLinear {name} weight scale must use the packed output N " +
                $"distribution, got scale={weightScale}, output={output}.");
        }

        return null;
    }

    private static Tensor UnpackDynamicTensorWeightScale(Tensor packedScale, int[] outputLanes)
    {
        if (packedScale.ElementType is not VectorType scaleVectorType ||
            !scaleVectorType.Lanes.SequenceEqual(outputLanes) ||
            packedScale.Rank != 1)
        {
            throw new InvalidOperationException(
                $"Dynamic-tensor PackedQKVParallelLinear weight scale must be rank-1 vec<{string.Join(",", outputLanes)}>, " +
                $"got shape=[{string.Join(",", packedScale.Dimensions.ToArray())}], dtype={packedScale.ElementType}.");
        }

        return packedScale.ToOrtTensor().Unpack(outputLanes.Length, new int[outputLanes.Length]).ToTensor();
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
