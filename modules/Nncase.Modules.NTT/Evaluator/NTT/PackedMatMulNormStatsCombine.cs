// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluates and validates packed-matmul partial materialization followed by
/// residual addition and normalization-statistics production.
/// </summary>
public sealed class PackedMatMulNormStatsCombineEvaluator :
    IEvaluator<PackedMatMulNormStatsCombine>,
    ITypeInferencer<PackedMatMulNormStatsCombine>,
    ICostEvaluator<PackedMatMulNormStatsCombine>
{
    public IValue Visit(IEvaluateContext context, PackedMatMulNormStatsCombine target)
    {
        var input = context.GetArgumentValueAsTensor(target, PackedMatMulNormStatsCombine.Input);
        var addend = context.GetArgumentValueAsTensor(target, PackedMatMulNormStatsCombine.Addend);
        var output = OrtKI.Add(input.ToOrtTensor(), addend.ToOrtTensor())
            .ToTensor()
            .CastElementTo(input.ElementType);
        return Value.FromTensors(
            output,
            NormStatsEvaluator.Evaluate(output, target.Axis, target.UseMean));
    }

    public IRType Visit(ITypeInferenceContext context, PackedMatMulNormStatsCombine target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, PackedMatMulNormStatsCombine.Input),
            context.CheckArgumentType<IRType>(target, PackedMatMulNormStatsCombine.Addend),
            target.OutputType);

    public static IRType InferType(
        PackedMatMulNormStatsCombine target,
        IRType inputType,
        IRType addendType,
        IRType outputType)
    {
        if (outputType is not TupleType { Count: 2 } output)
        {
            return new InvalidType(
                $"PackedMatMulNormStatsCombine requires a value/statistics output tuple, got {outputType}.");
        }

        var valueOutput = output[0];
        if (!Equals(addendType, valueOutput))
        {
            return new InvalidType(
                $"PackedMatMulNormStatsCombine addend must match the materialized value output, got {addendType} and {valueOutput}.");
        }

        if (!CanMaterialize(inputType, valueOutput))
        {
            return new InvalidType(
                $"PackedMatMulNormStatsCombine cannot materialize {inputType} into {valueOutput}.");
        }

        var statsType = NormStatsEvaluator.InferType(
            new NormStats(target.Axis, target.UseMean),
            valueOutput);
        if (statsType is InvalidType)
        {
            return statsType;
        }

        return Equals(statsType, output[1])
            ? outputType
            : new InvalidType(
                $"PackedMatMulNormStatsCombine statistics output must be {statsType}, got {output[1]}.");
    }

    public Cost Visit(ICostEvaluateContext context, PackedMatMulNormStatsCombine target)
    {
        var input = context.GetArgumentType<IRType>(target, PackedMatMulNormStatsCombine.Input);
        var addend = context.GetArgumentType<IRType>(target, PackedMatMulNormStatsCombine.Addend);
        var output = context.GetReturnType<TupleType>();
        var valueOutput = output[0];
        var statsOutput = output[1];
        var hasCollective = input is DistributedType distributedInput &&
            valueOutput is DistributedType distributedValueOutput &&
            !Equals(distributedInput, distributedValueOutput);
        var localInput = input is DistributedType distributedInputType
            ? DistributedUtility.GetDividedTensorType(
                distributedInputType,
                DistributedUtility.DivideFlags.MaxShape)
            : (TensorType)input;
        var localValue = valueOutput is DistributedType distributedOutput
            ? DistributedUtility.GetDividedTensorType(
                distributedOutput,
                DistributedUtility.DivideFlags.MaxShape)
            : (TensorType)valueOutput;
        var localStats = statsOutput is DistributedType distributedStats
            ? DistributedUtility.GetDividedTensorType(
                distributedStats,
                DistributedUtility.DivideFlags.MaxShape)
            : (TensorType)statsOutput;

        var result = new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                CostUtility.GetMemoryAccess(localInput) +
                CostUtility.GetMemoryAccess(localValue) +
                CostUtility.GetMemoryAccess(addend),
            [CostFactorNames.BlockLocalMemoryStoreBytes] =
                CostUtility.GetMemoryAccess(localValue) +
                CostUtility.GetMemoryAccess(localStats),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(
                localValue,
                target.UseMean ? 4U : 3U),
        };
        if (hasCollective)
        {
            result[CostFactorNames.GridSynchronization] = 1;
        }

        return result;
    }

    public static bool CanMaterialize(IRType inputType, IRType outputType)
    {
        if (Equals(inputType, outputType) && IsMaterialized(outputType))
        {
            return true;
        }

        return inputType is DistributedType input &&
            outputType is DistributedType output &&
            CanCombineTo(input, output);
    }

    private static bool IsMaterialized(IRType type)
        => type is not DistributedType distributed ||
            (distributed.Partial is null &&
             distributed.AxisPolicies.All(policy => policy is not SBPPartial));

    private static bool CanCombineTo(DistributedType input, DistributedType output)
    {
        if (input.TensorType != output.TensorType ||
            input.Placement != output.Placement ||
            input.Partial is not { Op: ReduceOp.Sum } partial ||
            output.Partial is not null ||
            input.AxisPolicies.Count != output.AxisPolicies.Count)
        {
            return false;
        }

        var inputHierarchy = DistributedUtility.GetHierarchyAxisPolicies(
            input.AxisPolicies,
            input.Placement.Rank);
        var outputHierarchy = DistributedUtility.GetHierarchyAxisPolicies(
            output.AxisPolicies,
            output.Placement.Rank);
        for (var axis = 0; axis < inputHierarchy.Count; axis++)
        {
            if (partial.Axes.Contains(axis))
            {
                if (inputHierarchy[axis] is not HierarchyAxisBroadcast ||
                    outputHierarchy[axis] is not (HierarchyAxisBroadcast or HierarchyAxisSplit))
                {
                    return false;
                }
            }
            else if (!CanMaterializeHierarchyPolicy(inputHierarchy[axis], outputHierarchy[axis]))
            {
                return false;
            }
        }

        return true;
    }

    private static bool CanMaterializeHierarchyPolicy(
        HierarchyAxisPolicy input,
        HierarchyAxisPolicy output)
        => (input, output) switch
        {
            (HierarchyAxisBroadcast, HierarchyAxisBroadcast) => true,
            (HierarchyAxisBroadcast, HierarchyAxisSplit) => true,
            (HierarchyAxisSplit, HierarchyAxisBroadcast) => true,
            (HierarchyAxisSplit lhs, HierarchyAxisSplit rhs) =>
                lhs.TensorAxis == rhs.TensorAxis,
            _ => false,
        };
}
