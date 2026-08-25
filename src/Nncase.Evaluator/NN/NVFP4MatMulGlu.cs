// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="NVFP4MatMulGlu"/>.
/// </summary>
public sealed class NVFP4MatMulGluEvaluator :
    IEvaluator<NVFP4MatMulGlu>,
    ITypeInferencer<NVFP4MatMulGlu>,
    ICostEvaluator<NVFP4MatMulGlu>
{
    public IValue Visit(IEvaluateContext context, NVFP4MatMulGlu target)
    {
        var input = context.GetArgumentValueAsTensor(target, NVFP4MatMulGlu.Input);
        var gate = Math.NVFP4MatMulEvaluator.Evaluate(
            input,
            context.GetArgumentValueAsTensor(target, NVFP4MatMulGlu.GateWeightPacked),
            context.GetArgumentValueAsTensor(target, NVFP4MatMulGlu.GateWeightScale),
            context.GetArgumentValueAsTensor(target, NVFP4MatMulGlu.GateInputGlobalScale),
            context.GetArgumentValueAsTensor(target, NVFP4MatMulGlu.GateWeightGlobalScale),
            target.OutputDataType,
            target.GroupSize);
        var up = Math.NVFP4MatMulEvaluator.Evaluate(
            input,
            context.GetArgumentValueAsTensor(target, NVFP4MatMulGlu.UpWeightPacked),
            context.GetArgumentValueAsTensor(target, NVFP4MatMulGlu.UpWeightScale),
            context.GetArgumentValueAsTensor(target, NVFP4MatMulGlu.UpInputGlobalScale),
            context.GetArgumentValueAsTensor(target, NVFP4MatMulGlu.UpWeightGlobalScale),
            target.OutputDataType,
            target.GroupSize);
        return Value.FromTensor(ApplyGlu(gate, up, target.GluType, target.OutputDataType));
    }

    public IRType Visit(ITypeInferenceContext context, NVFP4MatMulGlu target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, NVFP4MatMulGlu.Input),
            context.CheckArgumentType<IRType>(target, NVFP4MatMulGlu.GateWeightPacked),
            context.CheckArgumentType<IRType>(target, NVFP4MatMulGlu.UpWeightPacked),
            context.CheckArgumentType<IRType>(target, NVFP4MatMulGlu.GateWeightScale),
            context.CheckArgumentType<IRType>(target, NVFP4MatMulGlu.UpWeightScale),
            context.CheckArgumentType<IRType>(target, NVFP4MatMulGlu.GateInputGlobalScale),
            context.CheckArgumentType<IRType>(target, NVFP4MatMulGlu.UpInputGlobalScale),
            context.CheckArgumentType<IRType>(target, NVFP4MatMulGlu.GateWeightGlobalScale),
            context.CheckArgumentType<IRType>(target, NVFP4MatMulGlu.UpWeightGlobalScale));

    public Cost Visit(ICostEvaluateContext context, NVFP4MatMulGlu target)
    {
        var output = context.GetReturnType<IRType>();
        UInt128 loads = 0;
        foreach (var parameter in new[]
        {
            NVFP4MatMulGlu.Input,
            NVFP4MatMulGlu.GateWeightPacked,
            NVFP4MatMulGlu.UpWeightPacked,
            NVFP4MatMulGlu.GateWeightScale,
            NVFP4MatMulGlu.UpWeightScale,
            NVFP4MatMulGlu.GateInputGlobalScale,
            NVFP4MatMulGlu.UpInputGlobalScale,
            NVFP4MatMulGlu.GateWeightGlobalScale,
            NVFP4MatMulGlu.UpWeightGlobalScale,
        })
        {
            loads += CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, parameter));
        }

        var input = context.GetArgumentType<IRType>(target, NVFP4MatMulGlu.Input);
        var tensor = input is DistributedType distributed
            ? DistributedUtility.GetDividedTensorType(distributed)
            : input as TensorType;
        var k = tensor?.Shape is RankedShape { Rank: > 0 } shape && shape[^1].IsFixed
            ? checked((uint)shape[^1].FixedValue)
            : 1U;
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = loads,
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(output, checked((2U * k) + 17U)),
        };
    }

    public static IRType InferType(
        NVFP4MatMulGlu target,
        IRType input,
        IRType gateWeightPacked,
        IRType upWeightPacked,
        IRType gateWeightScale,
        IRType upWeightScale,
        IRType gateInputGlobalScale,
        IRType upInputGlobalScale,
        IRType gateWeightGlobalScale,
        IRType upWeightGlobalScale)
    {
        var gate = Math.NVFP4MatMulEvaluator.InferType(
            new NVFP4MatMul(target.OutputDataType, target.GroupSize),
            input,
            gateWeightPacked,
            gateWeightScale,
            gateInputGlobalScale,
            gateWeightGlobalScale);
        if (gate is InvalidType)
        {
            return gate;
        }

        var up = Math.NVFP4MatMulEvaluator.InferType(
            new NVFP4MatMul(target.OutputDataType, target.GroupSize),
            input,
            upWeightPacked,
            upWeightScale,
            upInputGlobalScale,
            upWeightGlobalScale);
        if (up is InvalidType)
        {
            return up;
        }

        if (gate != up)
        {
            return new InvalidType(
                $"NVFP4MatMulGlu gate/up projections must have the same type, got gate={gate}, up={up}.");
        }

        if (gate is DistributedType { Partial: not null })
        {
            return new InvalidType(
                "NVFP4MatMulGlu cannot split the reduction axis because GLU is nonlinear.");
        }

        return target.GluType switch
        {
            GluType.SwiGLU => gate,
            _ => new InvalidType($"Unsupported NVFP4MatMulGlu type: {target.GluType}."),
        };
    }

    private static Tensor ApplyGlu(Tensor gate, Tensor up, GluType gluType, DataType outputDataType)
    {
        if (gluType != GluType.SwiGLU)
        {
            throw new NotSupportedException($"Unsupported NVFP4MatMulGlu type: {gluType}.");
        }

        var gateFloat = gate.CastElementTo(DataTypes.Float32).ToOrtTensor();
        var upFloat = up.CastElementTo(DataTypes.Float32).ToOrtTensor();
        var result = OrtKI.Mul(OrtKI.Mul(gateFloat, OrtKI.Sigmoid(gateFloat)), upFloat);
        return result.ToTensor().CastElementTo(outputDataType);
    }
}
