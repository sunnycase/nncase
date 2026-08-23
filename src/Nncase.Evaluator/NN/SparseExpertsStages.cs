// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;

namespace Nncase.Evaluator.NN;

public sealed class SparseExpertsGateUpEvaluator :
    IEvaluator<SparseExpertsGateUp>,
    ITypeInferencer<SparseExpertsGateUp>,
    ICostEvaluator<SparseExpertsGateUp>
{
    public IRType Visit(ITypeInferenceContext context, SparseExpertsGateUp target)
    {
        var arguments = target.Parameters
            .Select(parameter => context.CheckArgumentType<IRType>(target, parameter))
            .ToArray();
        return InferType(target, arguments);
    }

    public Cost Visit(ICostEvaluateContext context, SparseExpertsGateUp target)
    {
        var q = SparseExpertsStageUtility.GetLogicalLocalTensorType(
            context.GetArgumentType<IRType>(target, SparseExpertsGateUp.Q), 1);
        var gateWeight = SparseExpertsStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, SparseExpertsGateUp.MoeExpertGateProjW));
        var output = SparseExpertsStageUtility.GetLogicalLocalTensorType(context.GetReturnType<IRType>(), 2);
        if (!SparseExpertsStageUtility.TryGetMaxShape(q, out var qShape) ||
            !SparseExpertsStageUtility.TryGetMaxShape(gateWeight, out var gateShape) ||
            !SparseExpertsStageUtility.TryGetMaxShape(output, out var outputShape))
        {
            return Cost.Zero;
        }

        var inputTensor = new TargetCostTensor(q.DType, new RankedShape(qShape[0], qShape[1]));
        var activationTensor = new TargetCostTensor(output.DType, new RankedShape(outputShape[0], outputShape[2]));
        var matrix = new TargetCostTensor(gateWeight.DType, new RankedShape(qShape[1], gateShape[1]));
        var cost = Cost.Zero;
        if (context.TargetCostModel.TryGetMatMulCost(
                new(inputTensor, matrix, activationTensor, output.DType, MatMulOpCostKind.Simt),
                out var projectionCost))
        {
            cost = projectionCost * checked((UInt128)target.NumTopK * 2U);
        }
        else
        {
            cost[CostFactorNames.CPUCycles] = checked(
                (UInt128)qShape[0] * (UInt128)target.NumTopK * 2U *
                (UInt128)qShape[1] * (UInt128)gateShape[1]);
        }

        if (context.TargetCostModel.TryGetElementwiseCost(
                new("sparse_experts_swiglu", [activationTensor, activationTensor], activationTensor, 9.0),
                out var activationCost))
        {
            cost += activationCost * (UInt128)target.NumTopK;
        }

        SparseExpertsStageUtility.AddSelectedExpertsMemoryCost(
            cost,
            context,
            target,
            denseLoadParameters:
            [
                SparseExpertsGateUp.Q,
                SparseExpertsGateUp.RouterExpertIds,
            ],
            selectedExpertLoadParameters:
            [
                SparseExpertsGateUp.MoeExpertGateInputScale,
                SparseExpertsGateUp.MoeExpertGateProjW,
                SparseExpertsGateUp.MoeExpertGateProjScale,
                SparseExpertsGateUp.MoeExpertUpInputScale,
                SparseExpertsGateUp.MoeExpertUpProjW,
                SparseExpertsGateUp.MoeExpertUpProjScale,
            ],
            target.NumExpert,
            checked(outputShape[0] * target.NumTopK),
            context.GetReturnType<IRType>());
        return cost;
    }

    public IValue Visit(IEvaluateContext context, SparseExpertsGateUp target)
    {
        var qValue = context.GetArgumentValueAsTensor(target, SparseExpertsGateUp.Q);
        var q = SparseExpertsStageUtility.GetLogicalValues(qValue, 1);
        var ids = context.GetArgumentValueAsTensor(target, SparseExpertsGateUp.RouterExpertIds).ToArray<long>();
        var gateInputScales = SparseExpertsStageUtility.GetFloatValues(
            context.GetArgumentValueAsTensor(target, SparseExpertsGateUp.MoeExpertGateInputScale));
        var gateWeights = SparseExpertsStageUtility.GetFloatValues(
            context.GetArgumentValueAsTensor(target, SparseExpertsGateUp.MoeExpertGateProjW));
        var gateScales = SparseExpertsStageUtility.GetFloatValues(
            context.GetArgumentValueAsTensor(target, SparseExpertsGateUp.MoeExpertGateProjScale));
        var upInputScales = SparseExpertsStageUtility.GetFloatValues(
            context.GetArgumentValueAsTensor(target, SparseExpertsGateUp.MoeExpertUpInputScale));
        var upWeights = SparseExpertsStageUtility.GetFloatValues(
            context.GetArgumentValueAsTensor(target, SparseExpertsGateUp.MoeExpertUpProjW));
        var upScales = SparseExpertsStageUtility.GetFloatValues(
            context.GetArgumentValueAsTensor(target, SparseExpertsGateUp.MoeExpertUpProjScale));
        var output = new float[checked((int)(target.ChunkSize * target.NumTopK * target.MoEIntermediateSize))];
        for (var token = 0L; token < target.ChunkSize; token++)
        {
            for (var topK = 0L; topK < target.NumTopK; topK++)
            {
                var expert = ids[checked((int)((token * target.NumTopK) + topK))];
                SparseExpertsStageUtility.ValidateExpertIndex(expert, target.NumExpert);
                var gateInputScale = gateInputScales[expert];
                var gateScale = gateScales[expert];
                var upInputScale = upInputScales[expert];
                var upScale = upScales[expert];
                for (var intermediate = 0L; intermediate < target.MoEIntermediateSize; intermediate++)
                {
                    var gate = 0F;
                    var up = 0F;
                    for (var hidden = 0L; hidden < target.HiddenSize; hidden++)
                    {
                        var input = q[checked((int)((token * target.HiddenSize) + hidden))];
                        var weightOffset = checked(
                            (int)(((expert * target.MoEIntermediateSize) + intermediate) * target.HiddenSize + hidden));
                        gate += (input / gateInputScale) * gateWeights[weightOffset];
                        up += (input / upInputScale) * upWeights[weightOffset];
                    }

                    gate *= gateInputScale * gateScale;
                    up *= upInputScale * upScale;
                    var swish = gate / (1F + MathF.Exp(-gate));
                    output[checked((int)(((token * target.NumTopK) + topK) * target.MoEIntermediateSize + intermediate))] = swish * up;
                }
            }
        }

        return SparseExpertsStageUtility.CreateOutputValue(
            output,
            [target.ChunkSize, target.NumTopK, target.MoEIntermediateSize],
            target.OutputDataType,
            2);
    }

    public static IRType InferType(SparseExpertsGateUp target, IReadOnlyList<IRType> arguments)
    {
        if (arguments.Any(type => type is AnyType))
        {
            return AnyType.Default;
        }

        if (SparseExpertsStageUtility.ValidateCommonArguments(target, arguments) is { } commonError)
        {
            return commonError;
        }

        var tensors = arguments.Select(SparseExpertsStageUtility.GetTensorType).ToArray();
        if (ValidateTensorContract(target, tensors) is { } tensorError)
        {
            return tensorError;
        }

        var outputTensor = SparseExpertsStageUtility.CreatePackedTensorType(
            target.OutputDataType,
            [target.ChunkSize, target.NumTopK, target.MoEIntermediateSize],
            2);
        if (outputTensor is InvalidType)
        {
            return outputTensor;
        }

        if (arguments.All(type => type is TensorType))
        {
            return outputTensor;
        }

        var distributed = arguments.Cast<DistributedType>().ToArray();
        if (SparseExpertsStageUtility.ValidateDistributedInputs(target, distributed) is { } distributedError)
        {
            return distributedError;
        }

        var placement = distributed[0].Placement;
        var token = distributed[SparseExpertsGateUp.Q.Index].AxisPolicies[0];
        var scalarIntermediate = distributed[SparseExpertsGateUp.MoeExpertGateProjW.Index].AxisPolicies[1];
        if (!SparseExpertsStageUtility.TryScalePolicy(
                scalarIntermediate,
                1,
                SparseExpertsStageUtility.GetVectorLaneCount(target.OutputDataType),
                out var intermediate))
        {
            return new InvalidType("SparseExpertsGateUp cannot scale the intermediate split to its packed output type.");
        }

        if (distributed[SparseExpertsGateUp.Q.Index].AxisPolicies[1] is not SBPBroadCast ||
            distributed[SparseExpertsGateUp.RouterExpertIds.Index].AxisPolicies is not [var idsToken, SBPBroadCast] ||
            idsToken != token ||
            distributed[SparseExpertsGateUp.MoeExpertGateProjW.Index].AxisPolicies is not [SBPBroadCast, var gateIntermediate, SBPBroadCast] ||
            gateIntermediate != scalarIntermediate ||
            distributed[SparseExpertsGateUp.MoeExpertUpProjW.Index].AxisPolicies is not [SBPBroadCast, var upIntermediate, SBPBroadCast] ||
            upIntermediate != scalarIntermediate)
        {
            return new InvalidType("SparseExpertsGateUp requires token-aligned ids, broadcast hidden inputs, and matching gate/up intermediate sharding.");
        }

        foreach (var parameter in SparseExpertsStageUtility.GateUpScaleParameters)
        {
            if (!SparseExpertsStageUtility.IsBroadcast(distributed[parameter.Index]))
            {
                return new InvalidType($"SparseExpertsGateUp {parameter.Name} must be broadcast because experts are selected dynamically.");
            }
        }

        if (!SparseExpertsStageUtility.TryGetDisjointRoleAxes(placement, token, intermediate, out _, out _))
        {
            return new InvalidType("SparseExpertsGateUp token and intermediate sharding must use disjoint hierarchy axes.");
        }

        return new DistributedType((TensorType)outputTensor, [token, SBP.B, intermediate], placement);
    }

    private static InvalidType? ValidateTensorContract(
        SparseExpertsGateUp target,
        IReadOnlyList<TensorType> arguments)
    {
        var q = SparseExpertsStageUtility.GetLogicalTensorType(arguments[SparseExpertsGateUp.Q.Index], 1);
        return SparseExpertsStageUtility.ValidateShapes(
            "SparseExpertsGateUp",
            [
                (SparseExpertsGateUp.Q, q, new long[] { target.ChunkSize, target.HiddenSize }),
                (SparseExpertsGateUp.RouterExpertIds, arguments[SparseExpertsGateUp.RouterExpertIds.Index], new long[] { target.ChunkSize, target.NumTopK }),
                (SparseExpertsGateUp.MoeExpertGateInputScale, arguments[SparseExpertsGateUp.MoeExpertGateInputScale.Index], new long[] { target.NumExpert, 1 }),
                (SparseExpertsGateUp.MoeExpertGateProjW, arguments[SparseExpertsGateUp.MoeExpertGateProjW.Index], new long[] { target.NumExpert, target.MoEIntermediateSize, target.HiddenSize }),
                (SparseExpertsGateUp.MoeExpertGateProjScale, arguments[SparseExpertsGateUp.MoeExpertGateProjScale.Index], new long[] { target.NumExpert, 1 }),
                (SparseExpertsGateUp.MoeExpertUpInputScale, arguments[SparseExpertsGateUp.MoeExpertUpInputScale.Index], new long[] { target.NumExpert, 1 }),
                (SparseExpertsGateUp.MoeExpertUpProjW, arguments[SparseExpertsGateUp.MoeExpertUpProjW.Index], new long[] { target.NumExpert, target.MoEIntermediateSize, target.HiddenSize }),
                (SparseExpertsGateUp.MoeExpertUpProjScale, arguments[SparseExpertsGateUp.MoeExpertUpProjScale.Index], new long[] { target.NumExpert, 1 }),
            ],
            SparseExpertsGateUp.RouterExpertIds,
            SparseExpertsGateUp.Q,
            target.OutputDataType);
    }
}

public sealed class SparseExpertsDownEvaluator :
    IEvaluator<SparseExpertsDown>,
    ITypeInferencer<SparseExpertsDown>,
    ICostEvaluator<SparseExpertsDown>
{
    public IRType Visit(ITypeInferenceContext context, SparseExpertsDown target)
    {
        var arguments = target.Parameters
            .Select(parameter => context.CheckArgumentType<IRType>(target, parameter))
            .ToArray();
        return InferType(target, arguments);
    }

    public Cost Visit(ICostEvaluateContext context, SparseExpertsDown target)
    {
        var activations = SparseExpertsStageUtility.GetLogicalLocalTensorType(
            context.GetArgumentType<IRType>(target, SparseExpertsDown.Activations), 2);
        var downWeight = SparseExpertsStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, SparseExpertsDown.MoeExpertDownProjW));
        var physicalOutput = SparseExpertsStageUtility.GetLocalTensorType(context.GetReturnType<IRType>());
        var output = SparseExpertsStageUtility.GetLogicalTensorType(physicalOutput, 1);
        if (!SparseExpertsStageUtility.TryGetMaxShape(activations, out var activationShape) ||
            !SparseExpertsStageUtility.TryGetMaxShape(downWeight, out var downShape) ||
            !SparseExpertsStageUtility.TryGetMaxShape(output, out var outputShape))
        {
            return Cost.Zero;
        }

        // The selected experts form one logical matrix by concatenating their
        // local K slices. This is both the semantic reduction and the PyNTT
        // microkernel contract; treating each route as an independent GEMV
        // overcharges launch/control work and hides the benefit of K sharding.
        var concatenatedK = checked(activationShape[2] * target.NumTopK);
        var inputTensor = new TargetCostTensor(
            activations.DType,
            new RankedShape(activationShape[0], concatenatedK));
        var matrix = new TargetCostTensor(
            downWeight.DType,
            new RankedShape(concatenatedK, downShape[1]));

        // Keep the packed output dtype in the target query. The logical shape
        // carries scalar N, while the vector dtype records that the selected
        // implementation is eligible for a vector/matrix microkernel.
        var outputTensor = new TargetCostTensor(physicalOutput.DType, new RankedShape(outputShape[0], outputShape[1]));
        var cost = Cost.Zero;
        UInt128 segmentedWeightReadPenalty = 0;
        if (context.TargetCostModel.TryGetMatMulCost(
                new(
                    inputTensor,
                    matrix,
                    outputTensor,
                    output.DType,
                    RhsMemoryAccess: new TargetCostMemoryAccessPattern(
                        checked(activationShape[2] * downWeight.DType.SizeInBytes))),
                out var downCost))
        {
            cost = downCost;
            var usefulMatMulLoadBytes = checked(
                ((UInt128)activationShape[0] * (UInt128)concatenatedK * (UInt128)activations.DType.SizeInBytes) +
                ((UInt128)concatenatedK * (UInt128)downShape[1] * (UInt128)downWeight.DType.SizeInBytes));
            if (cost.Factors.TryGetValue(CostFactorNames.BlockLocalMemoryLoadBytes, out var effectiveLoadBytes) &&
                effectiveLoadBytes > usefulMatMulLoadBytes)
            {
                segmentedWeightReadPenalty = effectiveLoadBytes - usefulMatMulLoadBytes;
            }
        }
        else
        {
            cost[CostFactorNames.CPUCycles] = checked(
                (UInt128)outputShape[0] * (UInt128)concatenatedK *
                (UInt128)outputShape[1]);
        }

        SparseExpertsStageUtility.AddSelectedExpertsMemoryCost(
            cost,
            context,
            target,
            denseLoadParameters:
            [
                SparseExpertsDown.Activations,
                SparseExpertsDown.RouterExpertIds,
                SparseExpertsDown.RouterExpertWeights,
            ],
            selectedExpertLoadParameters:
            [
                SparseExpertsDown.MoeExpertDownInputScale,
                SparseExpertsDown.MoeExpertDownProjW,
                SparseExpertsDown.MoeExpertDownProjScale,
            ],
            target.NumExpert,
            checked(outputShape[0] * target.NumTopK),
            context.GetReturnType<IRType>());
        if (segmentedWeightReadPenalty > 0)
        {
            cost[CostFactorNames.BlockLocalMemoryLoadBytes] = checked(
                cost[CostFactorNames.BlockLocalMemoryLoadBytes] + segmentedWeightReadPenalty);
        }

        return cost;
    }

    public IValue Visit(IEvaluateContext context, SparseExpertsDown target)
    {
        var activations = SparseExpertsStageUtility.GetLogicalValues(
            context.GetArgumentValueAsTensor(target, SparseExpertsDown.Activations), 2);
        var ids = context.GetArgumentValueAsTensor(target, SparseExpertsDown.RouterExpertIds).ToArray<long>();
        var routerWeights = SparseExpertsStageUtility.GetFloatValues(
            context.GetArgumentValueAsTensor(target, SparseExpertsDown.RouterExpertWeights));
        var inputScales = SparseExpertsStageUtility.GetFloatValues(
            context.GetArgumentValueAsTensor(target, SparseExpertsDown.MoeExpertDownInputScale));
        var downWeights = SparseExpertsStageUtility.GetFloatValues(
            context.GetArgumentValueAsTensor(target, SparseExpertsDown.MoeExpertDownProjW));
        var downScales = SparseExpertsStageUtility.GetFloatValues(
            context.GetArgumentValueAsTensor(target, SparseExpertsDown.MoeExpertDownProjScale));
        var output = new float[checked((int)(target.ChunkSize * target.HiddenSize))];
        for (var token = 0L; token < target.ChunkSize; token++)
        {
            for (var topK = 0L; topK < target.NumTopK; topK++)
            {
                var routeOffset = checked((int)((token * target.NumTopK) + topK));
                var expert = ids[routeOffset];
                SparseExpertsStageUtility.ValidateExpertIndex(expert, target.NumExpert);
                var inputScale = inputScales[expert];
                var downScale = downScales[expert];
                for (var hidden = 0L; hidden < target.HiddenSize; hidden++)
                {
                    var result = 0F;
                    for (var intermediate = 0L; intermediate < target.MoEIntermediateSize; intermediate++)
                    {
                        var activationOffset = checked(
                            (int)(((token * target.NumTopK) + topK) * target.MoEIntermediateSize + intermediate));
                        var weightOffset = checked(
                            (int)(((expert * target.HiddenSize) + hidden) * target.MoEIntermediateSize + intermediate));
                        result += (activations[activationOffset] / inputScale) * downWeights[weightOffset];
                    }

                    output[checked((int)((token * target.HiddenSize) + hidden))] +=
                        routerWeights[routeOffset] * result * inputScale * downScale;
                }
            }
        }

        return SparseExpertsStageUtility.CreateOutputValue(
            output,
            [target.ChunkSize, target.HiddenSize],
            target.OutputDataType,
            1);
    }

    public static IRType InferType(SparseExpertsDown target, IReadOnlyList<IRType> arguments)
    {
        if (arguments.Any(type => type is AnyType))
        {
            return AnyType.Default;
        }

        if (SparseExpertsStageUtility.ValidateCommonArguments(target, arguments) is { } commonError)
        {
            return commonError;
        }

        var tensors = arguments.Select(SparseExpertsStageUtility.GetTensorType).ToArray();
        if (ValidateTensorContract(target, tensors) is { } tensorError)
        {
            return tensorError;
        }

        var outputTensor = SparseExpertsStageUtility.CreatePackedTensorType(
            target.OutputDataType,
            [target.ChunkSize, target.HiddenSize],
            1);
        if (outputTensor is InvalidType)
        {
            return outputTensor;
        }

        if (arguments.All(type => type is TensorType))
        {
            return outputTensor;
        }

        var distributed = arguments.Cast<DistributedType>().ToArray();
        if (SparseExpertsStageUtility.ValidateDistributedInputs(target, distributed) is { } distributedError)
        {
            return distributedError;
        }

        var placement = distributed[0].Placement;
        var activation = distributed[SparseExpertsDown.Activations.Index];
        var token = activation.AxisPolicies[0];
        if (!SparseExpertsStageUtility.TryScalePolicy(
                activation.AxisPolicies[2],
                SparseExpertsStageUtility.GetVectorLaneCount(activation.TensorType.DType),
                1,
                out var scalarIntermediate))
        {
            return new InvalidType("SparseExpertsDown cannot scale its packed activation split to the scalar reduction axis.");
        }

        var scalarOutput = distributed[SparseExpertsDown.MoeExpertDownProjW.Index].AxisPolicies[1];
        if (!SparseExpertsStageUtility.TryScalePolicy(
                scalarOutput,
                1,
                SparseExpertsStageUtility.GetVectorLaneCount(target.OutputDataType),
                out var output))
        {
            return new InvalidType("SparseExpertsDown cannot scale the scalar output split to its packed output type.");
        }

        if (activation.AxisPolicies[1] is not SBPBroadCast ||
            distributed[SparseExpertsDown.RouterExpertIds.Index].AxisPolicies is not [var idsToken, SBPBroadCast] ||
            idsToken != token ||
            distributed[SparseExpertsDown.RouterExpertWeights.Index].AxisPolicies is not [var weightsToken, SBPBroadCast] ||
            weightsToken != token ||
            distributed[SparseExpertsDown.MoeExpertDownProjW.Index].AxisPolicies is not [SBPBroadCast, var downOutput, var downIntermediate] ||
            downOutput != scalarOutput || downIntermediate != scalarIntermediate)
        {
            return new InvalidType("SparseExpertsDown requires token-aligned router inputs and matching activation/down intermediate sharding.");
        }

        foreach (var parameter in SparseExpertsStageUtility.DownScaleParameters)
        {
            if (!SparseExpertsStageUtility.IsBroadcast(distributed[parameter.Index]))
            {
                return new InvalidType($"SparseExpertsDown {parameter.Name} must be broadcast because experts are selected dynamically.");
            }
        }

        if (!SparseExpertsStageUtility.TryGetDisjointRoleAxes(
                placement,
                token,
                scalarIntermediate,
                output,
                out _,
                out var intermediateAxes,
                out _))
        {
            return new InvalidType("SparseExpertsDown token, intermediate, and output sharding must use disjoint hierarchy axes.");
        }

        return new DistributedType(
            (TensorType)outputTensor,
            [token, output],
            placement,
            intermediateAxes.Length == 0 ? null : SBP.P(intermediateAxes));
    }

    private static InvalidType? ValidateTensorContract(
        SparseExpertsDown target,
        IReadOnlyList<TensorType> arguments)
    {
        var activations = SparseExpertsStageUtility.GetLogicalTensorType(
            arguments[SparseExpertsDown.Activations.Index],
            2);
        return SparseExpertsStageUtility.ValidateShapes(
            "SparseExpertsDown",
            [
                (SparseExpertsDown.Activations, activations, new long[] { target.ChunkSize, target.NumTopK, target.MoEIntermediateSize }),
                (SparseExpertsDown.RouterExpertIds, arguments[SparseExpertsDown.RouterExpertIds.Index], new long[] { target.ChunkSize, target.NumTopK }),
                (SparseExpertsDown.RouterExpertWeights, arguments[SparseExpertsDown.RouterExpertWeights.Index], new long[] { target.ChunkSize, target.NumTopK }),
                (SparseExpertsDown.MoeExpertDownInputScale, arguments[SparseExpertsDown.MoeExpertDownInputScale.Index], new long[] { target.NumExpert, 1 }),
                (SparseExpertsDown.MoeExpertDownProjW, arguments[SparseExpertsDown.MoeExpertDownProjW.Index], new long[] { target.NumExpert, target.HiddenSize, target.MoEIntermediateSize }),
                (SparseExpertsDown.MoeExpertDownProjScale, arguments[SparseExpertsDown.MoeExpertDownProjScale.Index], new long[] { target.NumExpert, 1 }),
            ],
            SparseExpertsDown.RouterExpertIds,
            SparseExpertsDown.Activations,
            target.OutputDataType);
    }
}

internal static class SparseExpertsStageUtility
{
    public static readonly ParameterInfo[] GateUpScaleParameters =
    [
        SparseExpertsGateUp.MoeExpertGateInputScale,
        SparseExpertsGateUp.MoeExpertGateProjScale,
        SparseExpertsGateUp.MoeExpertUpInputScale,
        SparseExpertsGateUp.MoeExpertUpProjScale,
    ];

    public static readonly ParameterInfo[] DownScaleParameters =
    [
        SparseExpertsDown.MoeExpertDownInputScale,
        SparseExpertsDown.MoeExpertDownProjScale,
    ];

    public static InvalidType? ValidateCommonArguments<TOp>(TOp target, IReadOnlyList<IRType> arguments)
        where TOp : Op
    {
        if (arguments.Count != target.Parameters.Count)
        {
            return new InvalidType($"{typeof(TOp).Name} expects {target.Parameters.Count} inputs, got {arguments.Count}.");
        }

        if (arguments.OfType<InvalidType>().FirstOrDefault() is { } invalid)
        {
            return invalid;
        }

        if (!arguments.All(type => type is TensorType) && !arguments.All(type => type is DistributedType))
        {
            return new InvalidType($"{typeof(TOp).Name} requires either tensor inputs or distributed inputs with one common placement.");
        }

        return null;
    }

    public static InvalidType? ValidateDistributedInputs<TOp>(TOp target, IReadOnlyList<DistributedType> arguments)
        where TOp : Op
    {
        var placement = arguments[0].Placement;
        if (arguments.Any(type => type.Placement != placement))
        {
            return new InvalidType($"{typeof(TOp).Name} distributed inputs must use one placement.");
        }

        if (arguments.Any(type => type.Partial is not null || type.AxisPolicies.Any(policy => policy is SBPPartial)))
        {
            return new InvalidType($"{typeof(TOp).Name} does not accept partial inputs.");
        }

        return null;
    }

    public static InvalidType? ValidateShapes(
        string opName,
        IReadOnlyList<(ParameterInfo Parameter, TensorType Type, long[] Shape)> checks,
        ParameterInfo idsParameter,
        ParameterInfo activationParameter,
        DataType outputDataType)
    {
        foreach (var (parameter, type, expectedShape) in checks)
        {
            if (type.Shape is not RankedShape shape || shape.Rank != expectedShape.Length)
            {
                return new InvalidType($"{opName} {parameter.Name} must have rank {expectedShape.Length}, got {type.Shape}.");
            }

            for (var axis = 0; axis < expectedShape.Length; axis++)
            {
                if (shape[axis].IsFixed && shape[axis].FixedValue != expectedShape[axis])
                {
                    return new InvalidType($"{opName} {parameter.Name} axis {axis} must be {expectedShape[axis]}, got {shape[axis]}.");
                }
            }
        }

        var ids = checks.Single(item => item.Parameter == idsParameter).Type;
        if (ids.DType != DataTypes.Int64)
        {
            return new InvalidType($"{opName} {idsParameter.Name} must be int64, got {ids.DType}.");
        }

        var activation = checks.Single(item => item.Parameter == activationParameter).Type;
        if (!GetScalarDataType(activation.DType).IsFloat() || !GetScalarDataType(outputDataType).IsFloat())
        {
            return new InvalidType($"{opName} activation and output dtypes must be floating point.");
        }

        if (GetScalarDataType(activation.DType) != GetScalarDataType(outputDataType))
        {
            return new InvalidType(
                $"{opName} activation and output scalar dtypes must match, got {activation.DType} and {outputDataType}.");
        }

        return null;
    }

    public static IRType CreatePackedTensorType(DataType dataType, long[] logicalShape, int packedAxis)
    {
        var scalarType = new TensorType(GetScalarDataType(dataType), logicalShape);
        return dataType is VectorType vector
            ? TypeInference.PackType(
                scalarType,
                vector.Lanes,
                Enumerable.Repeat(packedAxis, vector.Lanes.Count).ToArray())
            : scalarType;
    }

    public static TensorType GetTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => distributed.TensorType,
        _ => TensorType.Invalid(DataTypes.Float32),
    };

    public static TensorType GetLocalTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => DistributedUtility.GetDividedTensorType(
            distributed,
            DistributedUtility.DivideFlags.MaxShape),
        _ => TensorType.Invalid(DataTypes.Float32),
    };

    public static TensorType GetLogicalLocalTensorType(IRType type, int packedAxis) =>
        GetLogicalTensorType(GetLocalTensorType(type), packedAxis);

    public static TensorType GetLogicalTensorType(TensorType type, int packedAxis)
    {
        if (type.DType is not VectorType vector)
        {
            return type;
        }

        return TypeInference.UnpackType(
            type,
            Enumerable.Repeat(packedAxis, vector.Lanes.Count).ToArray()) switch
        {
            TensorType tensor => tensor,
            IRType invalid => throw new InvalidOperationException(
                $"Cannot unpack sparse-experts tensor type {type} on axis {packedAxis}: {invalid}."),
        };
    }

    public static bool IsBroadcast(DistributedType type) =>
        type.AxisPolicies.All(policy => policy is SBPBroadCast);

    public static bool TryGetDisjointRoleAxes(
        Placement placement,
        SBP first,
        SBP second,
        out int[] firstAxes,
        out int[] secondAxes)
    {
        firstAxes = Array.Empty<int>();
        secondAxes = Array.Empty<int>();
        return TryGetRoleAxes(first, placement.Rank, out firstAxes) &&
            TryGetRoleAxes(second, placement.Rank, out secondAxes) &&
            AreDisjoint(firstAxes, secondAxes);
    }

    public static bool TryGetDisjointRoleAxes(
        Placement placement,
        SBP first,
        SBP second,
        SBP third,
        out int[] firstAxes,
        out int[] secondAxes,
        out int[] thirdAxes)
    {
        firstAxes = Array.Empty<int>();
        secondAxes = Array.Empty<int>();
        thirdAxes = Array.Empty<int>();
        return TryGetRoleAxes(first, placement.Rank, out firstAxes) &&
            TryGetRoleAxes(second, placement.Rank, out secondAxes) &&
            TryGetRoleAxes(third, placement.Rank, out thirdAxes) &&
            AreDisjoint(firstAxes, secondAxes, thirdAxes);
    }

    public static bool TryGetMaxShape(TensorType type, out long[] shape)
    {
        if (CompilerServices.TryGetMaxShape(type.Shape, out var maxShape) && maxShape is not null)
        {
            shape = maxShape;
            return true;
        }

        shape = Array.Empty<long>();
        return false;
    }

    public static DataType GetScalarDataType(DataType type) =>
        type is VectorType vector ? GetScalarDataType(vector.ElemType) : type;

    public static long GetVectorLaneCount(DataType type) => type switch
    {
        VectorType vector => vector.Lanes.Aggregate(1L, (product, lane) => checked(product * lane)) *
            GetVectorLaneCount(vector.ElemType),
        _ => 1,
    };

    public static bool TryScalePolicy(
        SBP policy,
        long numerator,
        long denominator,
        out SBP scaledPolicy)
    {
        if (policy is not SBPSplit split)
        {
            scaledPolicy = policy;
            return policy is SBPBroadCast;
        }

        if (DistributedUtility.TryScaleSplitUnits(split, numerator, denominator, out var scaledSplit))
        {
            scaledPolicy = scaledSplit;
            return true;
        }

        scaledPolicy = null!;
        return false;
    }

    public static float[] GetLogicalValues(Tensor tensor, int packedAxis)
    {
        var value = tensor.ToOrtTensor();
        if (tensor.ElementType is VectorType vector)
        {
            value = value.Unpack(
                vector.Lanes.Count,
                Enumerable.Repeat(packedAxis, vector.Lanes.Count).ToArray());
        }

        return value.Cast(OrtKISharp.OrtDataType.Float).ToArray<float>();
    }

    public static float[] GetFloatValues(Tensor tensor) =>
        tensor.ToOrtTensor().Cast(OrtKISharp.OrtDataType.Float).ToArray<float>();

    public static IValue CreateOutputValue(
        float[] values,
        long[] logicalShape,
        DataType outputDataType,
        int packedAxis)
    {
        var scalarType = GetScalarDataType(outputDataType);
        var output = Tensor.From(values, logicalShape).CastElementTo(scalarType);
        if (outputDataType is not VectorType vector)
        {
            return Value.FromTensor(output);
        }

        var axes = Enumerable.Repeat(packedAxis, vector.Lanes.Count).ToArray();
        return output
            .ToOrtTensor()
            .Pack(0, vector.Lanes, axes)
            .ToValue(TypeInference.PackType(scalarType, vector.Lanes));
    }

    public static void ValidateExpertIndex(long expert, long numExperts)
    {
        if (expert < 0 || expert >= numExperts)
        {
            throw new InvalidOperationException(
                $"Sparse experts selected expert index {expert} is outside [0, {numExperts}).");
        }
    }

    public static void AddSelectedExpertsMemoryCost<TOp>(
        Cost cost,
        ICostEvaluateContext context,
        TOp target,
        IReadOnlyList<ParameterInfo> denseLoadParameters,
        IReadOnlyList<ParameterInfo> selectedExpertLoadParameters,
        long numExperts,
        long localSelectionCount,
        IRType storeType)
        where TOp : Op
    {
        if (numExperts <= 0 || localSelectionCount < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numExperts),
                $"Sparse-experts memory cost requires positive numExperts and non-negative local selections, got {numExperts}/{localSelectionCount}.");
        }

        var loadBytes = denseLoadParameters.Aggregate(
            (UInt128)0,
            (total, parameter) => total + CostUtility.GetMemoryAccess(
                context.GetArgumentType<IRType>(target, parameter)));
        foreach (var parameter in selectedExpertLoadParameters)
        {
            var allExpertBytes = CostUtility.GetMemoryAccess(
                context.GetArgumentType<IRType>(target, parameter));
            if ((allExpertBytes % (UInt128)numExperts) != 0)
            {
                throw new InvalidOperationException(
                    $"Sparse-experts parameter {parameter.Name} has {allExpertBytes} local bytes, " +
                    $"which cannot be divided into {numExperts} equal expert slices.");
            }

            loadBytes = checked(
                loadBytes +
                ((allExpertBytes / (UInt128)numExperts) * (UInt128)localSelectionCount));
        }

        cost[CostFactorNames.BlockLocalMemoryLoadBytes] = loadBytes;
        cost[CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(storeType);
    }

    private static bool TryGetRoleAxes(SBP policy, int placementRank, out int[] axes)
    {
        axes = policy switch
        {
            SBPBroadCast => Array.Empty<int>(),
            SBPSplit split => split.HierarchyAxes.ToArray(),
            _ => null!,
        };
        return axes is not null &&
            axes.Distinct().Count() == axes.Length &&
            axes.All(axis => axis >= 0 && axis < placementRank);
    }

    private static bool AreDisjoint(params IReadOnlyList<int>[] groups) =>
        groups.SelectMany(static group => group).Distinct().Count() == groups.Sum(static group => group.Count);
}
