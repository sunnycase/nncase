// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="PackedNVFP4MatMulGlu"/>.
/// </summary>
public sealed class PackedNVFP4MatMulGluEvaluator :
    IEvaluator<PackedNVFP4MatMulGlu>,
    ITypeInferencer<PackedNVFP4MatMulGlu>,
    ICostEvaluator<PackedNVFP4MatMulGlu>
{
    public IValue Visit(IEvaluateContext context, PackedNVFP4MatMulGlu target)
    {
        var input = context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulGlu.Input);
        var gate = Project(
            target,
            input,
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulGlu.GateWeightPacked),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulGlu.GateWeightScale),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulGlu.GateInputGlobalScale),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulGlu.GateWeightGlobalScale));
        var up = Project(
            target,
            input,
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulGlu.UpWeightPacked),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulGlu.UpWeightScale),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulGlu.UpInputGlobalScale),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulGlu.UpWeightGlobalScale));
        return Value.FromTensor(ApplyGlu(gate, up, target.GluType));
    }

    public IRType Visit(ITypeInferenceContext context, PackedNVFP4MatMulGlu target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulGlu.Input),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulGlu.GateWeightPacked),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulGlu.UpWeightPacked),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulGlu.GateWeightScale),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulGlu.UpWeightScale),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulGlu.GateInputGlobalScale),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulGlu.UpInputGlobalScale),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulGlu.GateWeightGlobalScale),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulGlu.UpWeightGlobalScale));

    public Cost Visit(ICostEvaluateContext context, PackedNVFP4MatMulGlu target)
    {
        UInt128 loads = 0;
        foreach (var parameter in new[]
        {
            PackedNVFP4MatMulGlu.Input,
            PackedNVFP4MatMulGlu.GateWeightPacked,
            PackedNVFP4MatMulGlu.UpWeightPacked,
            PackedNVFP4MatMulGlu.GateWeightScale,
            PackedNVFP4MatMulGlu.UpWeightScale,
            PackedNVFP4MatMulGlu.GateInputGlobalScale,
            PackedNVFP4MatMulGlu.UpInputGlobalScale,
            PackedNVFP4MatMulGlu.GateWeightGlobalScale,
            PackedNVFP4MatMulGlu.UpWeightGlobalScale,
        })
        {
            loads += CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, parameter));
        }

        var input = context.GetArgumentType<IRType>(target, PackedNVFP4MatMulGlu.Input);
        var output = context.GetReturnType<IRType>();
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = loads,
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(
                output,
                checked((2U * PackedNVFP4MatMulEvaluator.GetLogicalK(input)) + 17U)),
        };
    }

    public static IRType InferType(
        PackedNVFP4MatMulGlu target,
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
        var gate = ProjectType(
            target,
            input,
            gateWeightPacked,
            gateWeightScale,
            gateInputGlobalScale,
            gateWeightGlobalScale);
        if (gate is InvalidType)
        {
            return gate;
        }

        var up = ProjectType(
            target,
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
                $"PackedNVFP4MatMulGlu gate/up projections must have the same type, " +
                $"got gate={gate}, up={up}.");
        }

        if (gate is DistributedType { Partial: not null })
        {
            return new InvalidType(
                "PackedNVFP4MatMulGlu cannot split the reduction axis because GLU is nonlinear.");
        }

        return target.GluType switch
        {
            GluType.SwiGLU => gate,
            _ => new InvalidType($"Unsupported PackedNVFP4MatMulGlu type: {target.GluType}."),
        };
    }

    private static Tensor Project(
        PackedNVFP4MatMulGlu target,
        Tensor input,
        Tensor weightPacked,
        Tensor weightScale,
        Tensor inputGlobalScale,
        Tensor weightGlobalScale)
        => PackedNVFP4MatMulEvaluator.EvaluateProjection(
            target.OutputDataType,
            target.GroupSize,
            target.InputKVectorLaneCount,
            target.RhsKPackLaneCount,
            target.RhsKVectorLaneCount,
            target.OutputNVectorLaneCount,
            input,
            weightPacked,
            weightScale,
            inputGlobalScale,
            weightGlobalScale);

    private static IRType ProjectType(
        PackedNVFP4MatMulGlu target,
        IRType input,
        IRType weightPacked,
        IRType weightScale,
        IRType inputGlobalScale,
        IRType weightGlobalScale)
        => PackedNVFP4MatMulEvaluator.InferProjectionType(
            target.OutputDataType,
            target.GroupSize,
            target.InputKVectorLaneCount,
            target.RhsKPackLaneCount,
            target.RhsKVectorLaneCount,
            target.OutputNVectorLaneCount,
            input,
            weightPacked,
            weightScale,
            inputGlobalScale,
            weightGlobalScale);

    private static Tensor ApplyGlu(Tensor gate, Tensor up, GluType gluType)
    {
        if (gluType != GluType.SwiGLU)
        {
            throw new NotSupportedException($"Unsupported PackedNVFP4MatMulGlu type: {gluType}.");
        }

        var gateOrt = gate.ToOrtTensor();
        var gateType = gateOrt.DataType;
        var gateFloat = OrtKI.Cast(gateOrt, (long)OrtDataType.Float);
        var upFloat = OrtKI.Cast(up.ToOrtTensor(), (long)OrtDataType.Float);
        var result = OrtKI.Mul(OrtKI.Mul(gateFloat, OrtKI.Sigmoid(gateFloat)), upFloat);
        return OrtKI.Cast(result, (long)gateType).ToTensor(gate.ElementType);
    }
}
