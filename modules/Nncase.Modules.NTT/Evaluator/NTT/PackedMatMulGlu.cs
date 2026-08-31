// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;
using ScaledMatMulEvaluator = Nncase.Evaluator.Math.ScaledMatMulEvaluator;
using BlockScaledMatMulEvaluator = Nncase.Evaluator.Math.BlockScaledMatMulEvaluator;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="PackedMatMulGlu"/>.
/// </summary>
public sealed class PackedMatMulGluEvaluator : IEvaluator<PackedMatMulGlu>, ITypeInferencer<PackedMatMulGlu>, ICostEvaluator<PackedMatMulGlu>
{
    public IValue Visit(IEvaluateContext context, PackedMatMulGlu target)
    {
        var input = context.GetArgumentValueAsTensor(target, PackedMatMulGlu.Input);
        var gate = Project(input, context.GetArgumentValueAsTensor(target, PackedMatMulGlu.GateWeight), context.GetArgumentValue(target, PackedMatMulGlu.GateBias), context.GetArgumentValue(target, PackedMatMulGlu.GateInputScale), context.GetArgumentValue(target, PackedMatMulGlu.GateWeightScale), target);
        var up = Project(input, context.GetArgumentValueAsTensor(target, PackedMatMulGlu.UpWeight), context.GetArgumentValue(target, PackedMatMulGlu.UpBias), context.GetArgumentValue(target, PackedMatMulGlu.UpInputScale), context.GetArgumentValue(target, PackedMatMulGlu.UpWeightScale), target);
        return Value.FromTensor(ApplyGlu(gate, up, target.GluType));
    }

    public IRType Visit(ITypeInferenceContext context, PackedMatMulGlu target)
    {
        var input = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.Input);
        var gateWeight = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.GateWeight);
        var upWeight = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.UpWeight);
        var gateBias = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.GateBias);
        var upBias = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.UpBias);
        var gateInputScale = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.GateInputScale);
        var upInputScale = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.UpInputScale);
        var gateWeightScale = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.GateWeightScale);
        var upWeightScale = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.UpWeightScale);
        return InferType(
            target,
            input,
            gateWeight,
            upWeight,
            gateBias,
            upBias,
            gateInputScale,
            upInputScale,
            gateWeightScale,
            upWeightScale);
    }

    public static IRType InferType(
        PackedMatMulGlu target,
        IRType input,
        IRType gateWeight,
        IRType upWeight,
        IRType gateBias,
        IRType upBias,
        IRType gateInputScale,
        IRType upInputScale,
        IRType gateWeightScale,
        IRType upWeightScale)
    {
        var scaleCheck = CheckScaleContract(
            target,
            gateInputScale,
            upInputScale,
            gateWeightScale,
            upWeightScale);
        if (scaleCheck is not null)
        {
            return scaleCheck;
        }

        var gate = VisitProjection(input, gateWeight, gateInputScale, gateWeightScale, target);
        var up = VisitProjection(input, upWeight, upInputScale, upWeightScale, target);
        if (gate is InvalidType)
        {
            return gate;
        }

        if (up is InvalidType)
        {
            return up;
        }

        if (!SameProjectionType(gate, up))
        {
            return new InvalidType($"PackedMatMulGlu gate/up projections must have the same distributed type, got gate={gate}, up={up}.");
        }

        if (IsPartialProjection(gate))
        {
            if (gateBias is not NoneType || upBias is not NoneType)
            {
                return new InvalidType(
                    "PackedMatMulGlu split-K projections cannot apply bias before their partial sums are materialized.");
            }

            return target.GluType switch
            {
                GluType.SwiGLU => new TupleType(new[] { gate, up }),
                _ => new InvalidType($"Unsupported PackedMatMulGlu type: {target.GluType}."),
            };
        }

        var biasCheck = CheckBiasType("gate", gate, gateBias) ?? CheckBiasType("up", up, upBias);
        if (biasCheck is not null)
        {
            return biasCheck;
        }

        return target.GluType switch
        {
            GluType.SwiGLU => gate,
            _ => new InvalidType($"Unsupported PackedMatMulGlu type: {target.GluType}."),
        };
    }

    public Cost Visit(ICostEvaluateContext context, PackedMatMulGlu target)
    {
        var input = context.GetArgumentType<IRType>(target, PackedMatMulGlu.Input);
        var gateWeight = context.GetArgumentType<IRType>(target, PackedMatMulGlu.GateWeight);
        var upWeight = context.GetArgumentType<IRType>(target, PackedMatMulGlu.UpWeight);
        var gateBias = context.GetArgumentType<IRType>(target, PackedMatMulGlu.GateBias);
        var upBias = context.GetArgumentType<IRType>(target, PackedMatMulGlu.UpBias);
        var scales = new[]
        {
            context.GetArgumentType<IRType>(target, PackedMatMulGlu.GateInputScale),
            context.GetArgumentType<IRType>(target, PackedMatMulGlu.UpInputScale),
            context.GetArgumentType<IRType>(target, PackedMatMulGlu.GateWeightScale),
            context.GetArgumentType<IRType>(target, PackedMatMulGlu.UpWeightScale),
        };
        var output = context.GetReturnType<IRType>();
        var partialOutput = output as TupleType;
        var isPartial = partialOutput is { Count: 2 };
        var projectionOutput = isPartial ? partialOutput![0] : output;
        if (TryGetTargetCost(
                context,
                target,
                input,
                gateWeight,
                upWeight,
                projectionOutput,
                includeGlu: !isPartial,
                out var targetCost))
        {
            if (!isPartial)
            {
                AddBiasCost(targetCost, output, gateBias, upBias);
            }

            return targetCost;
        }

        var macPerElement = GetK(input);
        var outputAccess = output is TupleType tuple
            ? tuple.Fields.Aggregate((UInt128)0, (sum, field) => sum + CostUtility.GetMemoryAccess(field))
            : CostUtility.GetMemoryAccess(output);
        var outputCycles = output is TupleType outputTuple
            ? outputTuple.Fields.Aggregate(
                (UInt128)0,
                (sum, field) => sum + CostUtility.GetCPUCycles(field, macPerElement))
            : CostUtility.GetCPUCycles(output, checked((macPerElement * 2U) + 9U));
        return new()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(input)
                + CostUtility.GetMemoryAccess(gateWeight) + CostUtility.GetMemoryAccess(upWeight)
                + CostUtility.GetMemoryAccess(gateBias) + CostUtility.GetMemoryAccess(upBias)
                + scales.Aggregate((UInt128)0, (sum, scale) => sum + CostUtility.GetMemoryAccess(scale)),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = outputAccess,
            [CostFactorNames.CPUCycles] = outputCycles,
        };
    }

    private static Tensor Project(
        Tensor input,
        Tensor packedWeight,
        IValue packedBias,
        IValue inputScale,
        IValue weightScale,
        PackedMatMulGlu target)
    {
        string? errorMessage = null;
        if (packedWeight.ElementType is not VectorType weightVectorType ||
            !TryGetLayoutInfo(
                target.RhsLayout,
                weightVectorType,
                packedWeight.Rank,
                target.OutputDataType,
                out var rhsUnpackAxes,
                out var outputLanes,
                out var transposeB,
                out errorMessage))
        {
            throw new InvalidOperationException(
                errorMessage ?? $"PackedMatMulGlu expects a vector weight, got {packedWeight.ElementType}.");
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
            MatMulQuantizationMode.DynamicBlock => BlockScaledMatMulEvaluator.Evaluate(
                input,
                logicalWeight,
                weightScale.AsTensor(),
                target.OutputDataType,
                target.WeightBlockN,
                target.WeightBlockK).AsTensor().ToOrtTensor(),
            _ => throw new NotSupportedException(
                $"Unsupported PackedMatMulGlu quantization mode: {target.QuantizationMode}."),
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

    internal static Tensor ApplyGlu(Tensor gate, Tensor up, GluType gluType)
    {
        return gluType switch
        {
            GluType.SwiGLU => ApplySwiGLU(gate, up),
            _ => throw new NotSupportedException($"Unsupported PackedMatMulGlu type: {gluType}."),
        };
    }

    private static Tensor ApplySwiGLU(Tensor gate, Tensor up)
    {
        var gateOrt = gate.ToOrtTensor();
        var gateType = gateOrt.DataType;
        var gateFloat = OrtKI.Cast(gateOrt, (long)OrtDataType.Float);
        var swish = OrtKI.Sigmoid(gateFloat) * gateFloat;
        var swishCast = OrtKI.Cast(swish, (long)gateType);
        return OrtKI.Mul(swishCast, up.ToOrtTensor()).ToTensor(gate.ElementType);
    }

    private static IRType VisitProjection(
        IRType input,
        IRType packedWeight,
        IRType inputScale,
        IRType weightScale,
        PackedMatMulGlu target)
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
                            target.OutputDataType,
                            out var rhsUnpackAxes,
                            out var outputLanes,
                            out var transposeB,
                            out errorMessage))
                    {
                        return new InvalidType(
                            errorMessage ?? $"PackedMatMulGlu expects a vector weight, got {b.TensorType.DType}.");
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
                            "PackedMatMulGlu requires input and weight reduction axes to use the same " +
                            $"distributed policy, got input={a.AxisPolicies[dimInfo.Lk]} and " +
                            $"weight={unpackedWeight.AxisPolicies[dimInfo.Rk]}.");
                    }

                    var quantizedWeight = target.QuantizationMode == MatMulQuantizationMode.None || !transposeB
                        ? unpackedWeight
                        : TransposeLastTwo(unpackedWeight);
                    if (quantizedWeight is InvalidType)
                    {
                        return quantizedWeight;
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
                            quantizedWeight,
                            inputScale,
                            weightScale),
                        MatMulQuantizationMode.DynamicBlock => BlockScaledMatMulEvaluator.InferType(
                            new BlockScaledMatMul(
                                target.OutputDataType,
                                target.WeightBlockN,
                                target.WeightBlockK),
                            a,
                            quantizedWeight,
                            weightScale),
                        _ => new InvalidType(
                            $"Unsupported PackedMatMulGlu quantization mode: {target.QuantizationMode}."),
                    };
                    return PackOutput(projection, outputLanes);
                }

            case (TensorType a, TensorType b):
                {
                    string? errorMessage = null;
                    if (b.DType is not VectorType vectorType ||
                        !TryGetLayoutInfo(
                            target.RhsLayout,
                            vectorType,
                            b.Shape.Rank,
                            target.OutputDataType,
                            out var rhsUnpackAxes,
                            out var outputLanes,
                            out var transposeB,
                            out errorMessage))
                    {
                        return new InvalidType(
                            errorMessage ?? $"PackedMatMulGlu expects a vector weight, got {b.DType}.");
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
                    var quantizedWeight = target.QuantizationMode == MatMulQuantizationMode.None || !transposeB
                        ? unpackedWeight
                        : TransposeLastTwo(unpackedWeight);
                    if (quantizedWeight is InvalidType)
                    {
                        return quantizedWeight;
                    }

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
                            quantizedWeight,
                            inputScale,
                            weightScale),
                        MatMulQuantizationMode.DynamicBlock => BlockScaledMatMulEvaluator.InferType(
                            new BlockScaledMatMul(
                                target.OutputDataType,
                                target.WeightBlockN,
                                target.WeightBlockK),
                            a,
                            quantizedWeight,
                            weightScale),
                        _ => new InvalidType(
                            $"Unsupported PackedMatMulGlu quantization mode: {target.QuantizationMode}."),
                    };
                    return PackOutput(projection, outputLanes);
                }

            default:
                return new InvalidType(
                    $"PackedMatMulGlu input/weight types are not supported: {input}, {packedWeight}.");
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

    private static bool IsPartialProjection(IRType projection)
        => projection is DistributedType distributed &&
            (distributed.Partial is not null || distributed.AxisPolicies.Any(policy => policy is SBPPartial));

    internal static bool TryGetLayoutInfo(
        PackedMatMulRhsLayout layout,
        VectorType vectorType,
        int weightRank,
        DataType outputDataType,
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
            case (PackedMatMulRhsLayout.NMajorKPacked, 2)
                when outputDataType is PrimType { SizeInBytes: > 0 }:
                var vectorBytes = checked(
                    vectorType.Lanes[1] * vectorType.ElemType.SizeInBytes);
                if (vectorBytes % outputDataType.SizeInBytes != 0)
                {
                    break;
                }

                rhsUnpackAxes = [weightRank - 1, weightRank - 1];
                outputLanes = [vectorBytes / outputDataType.SizeInBytes];
                transposeB = true;
                errorMessage = null;
                return true;
            default:
                break;
        }

        rhsUnpackAxes = [];
        outputLanes = [];
        transposeB = false;
        errorMessage =
            $"PackedMatMulGlu {layout} has incompatible weight lanes " +
            $"[{string.Join(",", vectorType.Lanes)}] and output dtype {outputDataType}.";
        return false;
    }

    private static IRType UnpackType(IRType input, int[] axes) => input switch
    {
        DistributedType distributed => TypeInference.UnpackType(distributed, axes),
        TensorType tensor => TypeInference.UnpackType(tensor, axes),
        _ => new InvalidType($"Cannot unpack {input} with axes [{string.Join(",", axes)}]."),
    };

    private static IRType TransposeLastTwo(IRType input)
    {
        var tensor = input switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => null,
        };
        if (tensor?.Shape is not RankedShape { Rank: >= 2 } shape)
        {
            return new InvalidType($"Cannot transpose PackedMatMulGlu logical weight type {input}.");
        }

        var permutation = Enumerable.Range(0, shape.Rank).ToArray();
        (permutation[^2], permutation[^1]) = (permutation[^1], permutation[^2]);
        var permutationShape = new RankedShape(
            permutation.Select(axis => (Dimension)axis).ToArray());
        if (TypeInference.TransposeType(tensor, permutationShape) is not TensorType transposed)
        {
            return new InvalidType($"Cannot transpose PackedMatMulGlu logical weight type {input}.");
        }

        return input switch
        {
            TensorType => transposed,
            DistributedType distributed => new DistributedType(
                transposed,
                permutation.Select(axis => distributed.AxisPolicies[axis]).ToArray(),
                distributed.Placement,
                distributed.Partial),
            _ => new InvalidType($"Cannot transpose PackedMatMulGlu logical weight type {input}."),
        };
    }

    private static bool TryGetTargetCost(
        ICostEvaluateContext context,
        PackedMatMulGlu target,
        IRType input,
        IRType gateWeight,
        IRType upWeight,
        IRType output,
        bool includeGlu,
        out Cost cost)
    {
        cost = Cost.Zero;
        if (!TargetCostTensor.TryFromType(input, out var inputTensor))
        {
            return false;
        }

        var logicalOutput = output;
        if (target.RhsLayout != PackedMatMulRhsLayout.NMajor)
        {
            var outputTensorType = GetTensorType(output);
            var gateWeightTensorType = GetTensorType(gateWeight);
            if (outputTensorType?.DType is not VectorType outputVectorType ||
                gateWeightTensorType?.DType is not VectorType gateWeightVectorType ||
                !TryGetLayoutInfo(
                    target.RhsLayout,
                    gateWeightVectorType,
                    gateWeightTensorType.Shape.Rank,
                    target.OutputDataType,
                    out _,
                    out var outputLanes,
                    out _,
                    out _) ||
                !outputVectorType.Lanes.SequenceEqual(outputLanes))
            {
                return false;
            }

            logicalOutput = UnpackType(
                output,
                Enumerable.Repeat(outputTensorType.Shape.Rank - 1, outputLanes.Length).ToArray());
            if (logicalOutput is InvalidType)
            {
                return false;
            }
        }

        if (!TargetCostTensor.TryFromType(logicalOutput, out var outputTensor))
        {
            return false;
        }

        foreach (var weight in new[] { gateWeight, upWeight })
        {
            var logicalWeight = weight;
            if (target.RhsLayout != PackedMatMulRhsLayout.NMajor)
            {
                var weightTensorType = GetTensorType(weight);
                if (weightTensorType?.DType is not VectorType vectorType ||
                    !TryGetLayoutInfo(
                        target.RhsLayout,
                        vectorType,
                        weightTensorType.Shape.Rank,
                        target.OutputDataType,
                        out var rhsUnpackAxes,
                        out _,
                        out _,
                        out _))
                {
                    cost = Cost.Zero;
                    return false;
                }

                logicalWeight = UnpackType(weight, rhsUnpackAxes);
                if (logicalWeight is InvalidType)
                {
                    cost = Cost.Zero;
                    return false;
                }
            }

            if (!TargetCostTensor.TryFromType(logicalWeight, out var weightTensor)
                || !context.TargetCostModel.TryGetMatMulCost(
                    new(inputTensor, weightTensor, outputTensor, GetScalarType(target.OutputDataType), MatMulOpCostKind.Simt),
                    out var projectionCost))
            {
                cost = Cost.Zero;
                return false;
            }

            cost += projectionCost;
        }

        if (includeGlu && context.TargetCostModel.TryGetElementwiseCost(new("packed_matmul_glu", [outputTensor, outputTensor], outputTensor, WorkPerElement: 9.0), out var gluCost))
        {
            cost += gluCost;
        }
        else if (includeGlu)
        {
            AddCostFactor(cost, CostFactorNames.CPUCycles, CostUtility.GetCPUCycles(output, 9));
            AddCostFactor(cost, CostFactorNames.BlockLocalMemoryLoadBytes, CostUtility.GetMemoryAccess(output));
            AddCostFactor(cost, CostFactorNames.BlockLocalMemoryStoreBytes, CostUtility.GetMemoryAccess(output));
        }

        if (TryGetMemoryBytes(input, out var inputBytes))
        {
            SubtractCostFactor(cost, CostFactorNames.BlockLocalMemoryLoadBytes, inputBytes);
        }

        return true;
    }

    private static void AddBiasCost(Cost cost, IRType outputType, IRType gateBias, IRType upBias)
    {
        foreach (var bias in new[] { gateBias, upBias })
        {
            if (bias is NoneType)
            {
                continue;
            }

            AddCostFactor(cost, CostFactorNames.BlockLocalMemoryLoadBytes, CostUtility.GetMemoryAccess(bias));
            AddCostFactor(cost, CostFactorNames.CPUCycles, CostUtility.GetCPUCycles(outputType, 1));
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
            return new InvalidType($"PackedMatMulGlu {name} bias should be a packed tensor or None, got {bias}.");
        }

        if (GetLastDimension(output) is { IsFixed: true } outDim && GetLastDimension(bias) is { IsFixed: true } biasDim && biasDim.FixedValue != outDim.FixedValue)
        {
            return new InvalidType($"PackedMatMulGlu {name} bias last dimension {biasDim.FixedValue} does not match packed output dimension {outDim.FixedValue}.");
        }

        return null;
    }

    private static InvalidType? CheckScaleContract(
        PackedMatMulGlu target,
        IRType gateInputScale,
        IRType upInputScale,
        IRType gateWeightScale,
        IRType upWeightScale)
    {
        var hasGateInputScale = gateInputScale is not NoneType;
        var hasUpInputScale = upInputScale is not NoneType;
        var hasGateWeightScale = gateWeightScale is not NoneType;
        var hasUpWeightScale = upWeightScale is not NoneType;
        var valid = target.QuantizationMode switch
        {
            MatMulQuantizationMode.None =>
                !hasGateInputScale && !hasUpInputScale &&
                !hasGateWeightScale && !hasUpWeightScale &&
                target.WeightBlockN == 0 && target.WeightBlockK == 0,
            MatMulQuantizationMode.StaticTensor =>
                ScaledMatMulEvaluator.IsScaleType(gateInputScale) &&
                ScaledMatMulEvaluator.IsScaleType(upInputScale) &&
                ScaledMatMulEvaluator.IsScaleType(gateWeightScale) &&
                ScaledMatMulEvaluator.IsScaleType(upWeightScale) &&
                target.WeightBlockN == 0 && target.WeightBlockK == 0,
            MatMulQuantizationMode.DynamicBlock =>
                !hasGateInputScale && !hasUpInputScale &&
                hasGateWeightScale && hasUpWeightScale &&
                target.WeightBlockN > 0 && target.WeightBlockK > 0,
            _ => false,
        };
        return valid
            ? null
            : new InvalidType(
                $"PackedMatMulGlu scale operands do not match quantization mode {target.QuantizationMode}; " +
                $"input scales={hasGateInputScale}/{hasUpInputScale}, weight scales=" +
                $"{hasGateWeightScale}/{hasUpWeightScale}, block=" +
                $"[{target.WeightBlockN}, {target.WeightBlockK}].");
    }

    private static bool SameProjectionType(IRType lhs, IRType rhs) => (lhs, rhs) switch
    {
        (DistributedType lhsDistributed, DistributedType rhsDistributed) => lhsDistributed.Equals(rhsDistributed),
        (TensorType lhsTensor, TensorType rhsTensor) => lhsTensor.Equals(rhsTensor),
        _ => false,
    };

    private static TensorType? GetTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => distributed.TensorType,
        _ => null,
    };

    private static bool IsNone(IValue value) => value is NoneValue || value.Type is NoneType;

    private static Dimension? GetLastDimension(IRType type)
    {
        var tensorType = GetTensorType(type);
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
