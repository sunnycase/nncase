// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="MatMulGlu"/>.
/// </summary>
public sealed class MatMulGluEvaluator : IEvaluator<MatMulGlu>, ITypeInferencer<MatMulGlu>, ICostEvaluator<MatMulGlu>
{
    public IValue Visit(IEvaluateContext context, MatMulGlu target)
    {
        var input = context.GetArgumentValueAsTensor(target, MatMulGlu.Input);
        var gate = Project(
            input,
            context.GetArgumentValueAsTensor(target, MatMulGlu.GateWeight),
            context.GetArgumentValue(target, MatMulGlu.GateBias),
            context.GetArgumentValue(target, MatMulGlu.GateInputScale),
            context.GetArgumentValue(target, MatMulGlu.GateWeightScale),
            target.OutputDataType);
        var up = Project(
            input,
            context.GetArgumentValueAsTensor(target, MatMulGlu.UpWeight),
            context.GetArgumentValue(target, MatMulGlu.UpBias),
            context.GetArgumentValue(target, MatMulGlu.UpInputScale),
            context.GetArgumentValue(target, MatMulGlu.UpWeightScale),
            target.OutputDataType);
        return Value.FromTensor(ApplyGlu(gate, up, target.GluType, target.OutputDataType));
    }

    public IRType Visit(ITypeInferenceContext context, MatMulGlu target)
    {
        var input = context.CheckArgumentType<IRType>(target, MatMulGlu.Input);
        var gateWeight = context.CheckArgumentType<IRType>(target, MatMulGlu.GateWeight);
        var upWeight = context.CheckArgumentType<IRType>(target, MatMulGlu.UpWeight);
        var gateBias = context.CheckArgumentType<IRType>(target, MatMulGlu.GateBias);
        var upBias = context.CheckArgumentType<IRType>(target, MatMulGlu.UpBias);
        var gate = VisitProjection(input, gateWeight, target.OutputDataType);
        var up = VisitProjection(input, upWeight, target.OutputDataType);
        if (gate is InvalidType)
        {
            return gate;
        }

        if (up is InvalidType)
        {
            return up;
        }

        if (!SameTensorShape(gate, up))
        {
            return new InvalidType($"MatMulGlu gate/up projections must have the same shape, got gate={gate}, up={up}.");
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
            _ => new InvalidType($"Unsupported MatMulGlu type: {target.GluType}."),
        };
    }

    public Cost Visit(ICostEvaluateContext context, MatMulGlu target)
    {
        var input = context.GetArgumentType<IRType>(target, MatMulGlu.Input);
        var gateWeight = context.GetArgumentType<IRType>(target, MatMulGlu.GateWeight);
        var upWeight = context.GetArgumentType<IRType>(target, MatMulGlu.UpWeight);
        var gateBias = context.GetArgumentType<IRType>(target, MatMulGlu.GateBias);
        var upBias = context.GetArgumentType<IRType>(target, MatMulGlu.UpBias);
        var gateInputScale = context.GetArgumentType<IRType>(target, MatMulGlu.GateInputScale);
        var upInputScale = context.GetArgumentType<IRType>(target, MatMulGlu.UpInputScale);
        var gateWeightScale = context.GetArgumentType<IRType>(target, MatMulGlu.GateWeightScale);
        var upWeightScale = context.GetArgumentType<IRType>(target, MatMulGlu.UpWeightScale);
        var output = context.GetReturnType<IRType>();
        if (TryGetTargetCost(context, target, input, gateWeight, upWeight, output, out var targetCost))
        {
            AddBiasCost(targetCost, output, gateBias, upBias);
            return targetCost;
        }

        var macPerElement = GetK(input);
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input)
                + CostUtility.GetMemoryAccess(gateWeight) + CostUtility.GetMemoryAccess(upWeight)
                + CostUtility.GetMemoryAccess(gateBias) + CostUtility.GetMemoryAccess(upBias)
                + CostUtility.GetMemoryAccess(gateInputScale) + CostUtility.GetMemoryAccess(upInputScale)
                + CostUtility.GetMemoryAccess(gateWeightScale) + CostUtility.GetMemoryAccess(upWeightScale),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(output, checked(macPerElement * 2U + 9U)),
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

    private static Tensor ApplyGlu(Tensor gate, Tensor up, GluType gluType, DataType outputDataType)
    {
        return gluType switch
        {
            GluType.SwiGLU => ApplySwiGLU(gate, up, outputDataType),
            _ => throw new NotSupportedException($"Unsupported MatMulGlu type: {gluType}."),
        };
    }

    private static Tensor ApplySwiGLU(Tensor gate, Tensor up, DataType outputDataType)
    {
        var gateOrt = gate.ToOrtTensor();
        var gateType = gateOrt.DataType;
        var gateFloat = OrtKI.Cast(gateOrt, (long)OrtDataType.Float);
        var swish = OrtKI.Sigmoid(gateFloat) * gateFloat;
        var swishCast = OrtKI.Cast(swish, (long)gateType);
        return OrtKI.Mul(swishCast, up.ToOrtTensor()).ToTensor().CastElementTo(outputDataType);
    }

    private static IRType VisitProjection(IRType input, IRType weight, DataType outputDataType)
    {
        return (input, weight) switch
        {
            (DistributedType a, DistributedType b) => Math.MatMulEvaluator.VisitDistributedType(a, b with { TensorType = b.TensorType with { DType = a.TensorType.DType } }, NoneType.Default, outputDataType: outputDataType),
            (TensorType a, TensorType b) => Math.MatMulEvaluator.VisitTensorType(a, b with { DType = a.DType }, NoneType.Default, outputDataType: outputDataType),
            _ => new InvalidType($"MatMulGlu input/weight types are not supported: {input}, {weight}."),
        };
    }

    private static InvalidType? RejectPartialProjection(string name, IRType projection)
    {
        if (projection is DistributedType { Partial: not null } distributed)
        {
            return new InvalidType($"MatMulGlu does not support reduction-axis sharding because SwiGLU is nonlinear; {name} projection type has partial {distributed.Partial}.");
        }

        if (projection is DistributedType { AxisPolicies: var policies } && policies.Any(policy => policy is SBPPartial))
        {
            return new InvalidType($"MatMulGlu does not support partial axis policies because SwiGLU is nonlinear; {name} projection is partial.");
        }

        return null;
    }

    private static bool TryGetTargetCost(ICostEvaluateContext context, MatMulGlu target, IRType input, IRType gateWeight, IRType upWeight, IRType output, out Cost cost)
    {
        cost = Cost.Zero;
        if (!TargetCostTensor.TryFromType(input, out var inputTensor)
            || !TargetCostTensor.TryFromType(output, out var outputTensor))
        {
            return false;
        }

        foreach (var weight in new[] { gateWeight, upWeight })
        {
            if (!TargetCostTensor.TryFromType(weight, out var weightTensor)
                || !context.TargetCostModel.TryGetMatMulCost(new(inputTensor, weightTensor, outputTensor, target.OutputDataType), out var projectionCost))
            {
                cost = Cost.Zero;
                return false;
            }

            cost += projectionCost;
        }

        if (context.TargetCostModel.TryGetElementwiseCost(new("matmul_glu", [outputTensor, outputTensor], outputTensor, WorkPerElement: 9.0), out var gluCost))
        {
            cost += gluCost;
        }
        else
        {
            AddCostFactor(cost, CostFactorNames.CPUCycles, CostUtility.GetCPUCycles(output, 9));
            AddCostFactor(cost, CostFactorNames.MemoryLoad, CostUtility.GetMemoryAccess(output));
            AddCostFactor(cost, CostFactorNames.MemoryStore, CostUtility.GetMemoryAccess(output));
        }

        if (TryGetScalarElementCount(input, out var inputElements))
        {
            SubtractCostFactor(cost, CostFactorNames.MemoryLoad, inputElements);
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

            AddCostFactor(cost, CostFactorNames.MemoryLoad, CostUtility.GetMemoryAccess(bias));
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

    private static bool TryGetScalarElementCount(IRType type, out UInt128 count)
    {
        if (!TargetCostTensor.TryFromType(type, out var tensor)
            || !CompilerServices.TryGetMaxShape(tensor.Shape, out var shape))
        {
            count = 0;
            return false;
        }

        count = 1;
        foreach (var dim in shape)
        {
            count *= (UInt128)System.Math.Max(0, dim);
        }

        count *= (UInt128)GetVectorLaneCount(tensor.DType);
        return true;
    }

    private static long GetVectorLaneCount(DataType dtype) => dtype switch
    {
        VectorType vectorType => vectorType.Lanes.Aggregate(1L, static (acc, lane) => acc * lane) * GetVectorLaneCount(vectorType.ElemType),
        _ => 1,
    };

    private static InvalidType? CheckBiasType(string name, IRType output, IRType bias)
    {
        if (bias is NoneType)
        {
            return null;
        }

        if (bias is not TensorType and not DistributedType)
        {
            return new InvalidType($"MatMulGlu {name} bias should be a tensor or None, got {bias}.");
        }

        var outDim = GetLastDimension(output);
        var biasDim = GetLastDimension(bias);
        if (outDim is { IsFixed: true } && biasDim is { IsFixed: true } && biasDim.FixedValue != outDim.FixedValue)
        {
            return new InvalidType($"MatMulGlu {name} bias last dimension {biasDim.FixedValue} does not match output dimension {outDim.FixedValue}.");
        }

        return null;
    }

    private static bool SameTensorShape(IRType lhs, IRType rhs)
    {
        var lhsTensor = GetTensorType(lhs);
        var rhsTensor = GetTensorType(rhs);
        return lhsTensor is not null && rhsTensor is not null && lhsTensor.Shape.Equals(rhsTensor.Shape);
    }

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
