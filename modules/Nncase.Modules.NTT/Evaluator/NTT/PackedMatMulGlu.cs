// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="PackedMatMulGlu"/>.
/// </summary>
public sealed class PackedMatMulGluEvaluator : IEvaluator<PackedMatMulGlu>, ITypeInferencer<PackedMatMulGlu>, ICostEvaluator<PackedMatMulGlu>
{
    public IValue Visit(IEvaluateContext context, PackedMatMulGlu target)
    {
        ValidateNoScales(
            context.GetArgumentValue(target, PackedMatMulGlu.GateInputScale),
            context.GetArgumentValue(target, PackedMatMulGlu.UpInputScale),
            context.GetArgumentValue(target, PackedMatMulGlu.GateWeightScale),
            context.GetArgumentValue(target, PackedMatMulGlu.UpWeightScale));

        var input = context.GetArgumentValueAsTensor(target, PackedMatMulGlu.Input);
        var gate = Project(input, context.GetArgumentValueAsTensor(target, PackedMatMulGlu.GateWeight), context.GetArgumentValue(target, PackedMatMulGlu.GateBias), target.OutputDataType, target.RhsLayout);
        var up = Project(input, context.GetArgumentValueAsTensor(target, PackedMatMulGlu.UpWeight), context.GetArgumentValue(target, PackedMatMulGlu.UpBias), target.OutputDataType, target.RhsLayout);
        return Value.FromTensor(ApplyGlu(gate, up, target.GluType));
    }

    public IRType Visit(ITypeInferenceContext context, PackedMatMulGlu target)
    {
        var input = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.Input);
        var gateWeight = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.GateWeight);
        var upWeight = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.UpWeight);
        var gateBias = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.GateBias);
        var upBias = context.CheckArgumentType<IRType>(target, PackedMatMulGlu.UpBias);
        var scaleCheck = CheckScales(
            context.CheckArgumentType<IRType>(target, PackedMatMulGlu.GateInputScale),
            context.CheckArgumentType<IRType>(target, PackedMatMulGlu.UpInputScale),
            context.CheckArgumentType<IRType>(target, PackedMatMulGlu.GateWeightScale),
            context.CheckArgumentType<IRType>(target, PackedMatMulGlu.UpWeightScale));
        if (scaleCheck is not null)
        {
            return scaleCheck;
        }

        var gate = VisitProjection(input, gateWeight, target.OutputDataType, target.RhsLayout);
        var up = VisitProjection(input, upWeight, target.OutputDataType, target.RhsLayout);
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

        if (RejectPartialProjection("gate", gate) is { } partialCheck)
        {
            return partialCheck;
        }

        if (RejectPartialProjection("up", up) is { } upPartialCheck)
        {
            return upPartialCheck;
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
        var output = context.GetReturnType<IRType>();
        if (TryGetTargetCost(context, target, input, gateWeight, upWeight, output, out var targetCost))
        {
            AddBiasCost(targetCost, output, gateBias, upBias);
            return targetCost;
        }

        var macPerElement = GetK(input);
        return new()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(input)
                + CostUtility.GetMemoryAccess(gateWeight) + CostUtility.GetMemoryAccess(upWeight)
                + CostUtility.GetMemoryAccess(gateBias) + CostUtility.GetMemoryAccess(upBias),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(output, checked(macPerElement * 2U + 9U)),
        };
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

    private static Tensor ApplyGlu(Tensor gate, Tensor up, GluType gluType)
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

    private static InvalidType? RejectPartialProjection(string name, IRType projection)
    {
        if (projection is DistributedType { Partial: not null } distributed)
        {
            return new InvalidType($"PackedMatMulGlu does not support reduction-axis sharding because SwiGLU is nonlinear; {name} projection type has partial {distributed.Partial}.");
        }

        if (projection is DistributedType { AxisPolicies: var policies } && policies.Any(policy => policy is SBPPartial))
        {
            return new InvalidType($"PackedMatMulGlu does not support partial axis policies because SwiGLU is nonlinear; {name} projection is partial.");
        }

        return null;
    }

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
                    $"PackedMatMulGlu {layout} expects " +
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

    private static bool TryGetTargetCost(ICostEvaluateContext context, PackedMatMulGlu target, IRType input, IRType gateWeight, IRType upWeight, IRType output, out Cost cost)
    {
        cost = Cost.Zero;
        if (!TargetCostTensor.TryFromType(input, out var inputTensor))
        {
            return false;
        }

        var logicalOutput = output;
        if (target.RhsLayout == PackedMatMulRhsLayout.KMajor)
        {
            var outputTensorType = GetTensorType(output);
            var gateWeightTensorType = GetTensorType(gateWeight);
            if (outputTensorType?.DType is not VectorType outputVectorType ||
                gateWeightTensorType?.DType is not VectorType gateWeightVectorType ||
                !TryGetLayoutInfo(
                    target.RhsLayout,
                    gateWeightVectorType,
                    gateWeightTensorType.Shape.Rank,
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
            if (target.RhsLayout == PackedMatMulRhsLayout.KMajor)
            {
                var weightTensorType = GetTensorType(weight);
                if (weightTensorType?.DType is not VectorType vectorType ||
                    !TryGetLayoutInfo(
                        target.RhsLayout,
                        vectorType,
                        weightTensorType.Shape.Rank,
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

        if (context.TargetCostModel.TryGetElementwiseCost(new("packed_matmul_glu", [outputTensor, outputTensor], outputTensor, WorkPerElement: 9.0), out var gluCost))
        {
            cost += gluCost;
        }
        else
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

    private static InvalidType? CheckScales(params IRType[] scales)
    {
        return scales.All(scale => scale is NoneType)
            ? null
            : new InvalidType("PackedMatMulGlu currently supports only None input/weight scales.");
    }

    private static void ValidateNoScales(params IValue[] scales)
    {
        if (scales.Any(scale => !IsNone(scale)))
        {
            throw new NotSupportedException("PackedMatMulGlu currently supports only None input/weight scales.");
        }
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
