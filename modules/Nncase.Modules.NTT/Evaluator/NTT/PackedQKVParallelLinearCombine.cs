// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluates and validates materialization of direct or partial packed QKV results.
/// </summary>
public sealed class PackedQKVParallelLinearCombineEvaluator :
    IEvaluator<PackedQKVParallelLinearCombine>,
    ITypeInferencer<PackedQKVParallelLinearCombine>,
    ICostEvaluator<PackedQKVParallelLinearCombine>
{
    public IValue Visit(IEvaluateContext context, PackedQKVParallelLinearCombine target)
        => context.GetArgumentValue(target, PackedQKVParallelLinearCombine.QKV);

    public IRType Visit(ITypeInferenceContext context, PackedQKVParallelLinearCombine target)
        => InferType(
            context.CheckArgumentType<IRType>(target, PackedQKVParallelLinearCombine.QKV),
            target.OutputType);

    public static IRType InferType(IRType inputType, IRType outputType)
    {
        if (inputType is not TupleType { Count: 3 } input ||
            outputType is not TupleType { Count: 3 } output)
        {
            return new InvalidType(
                $"PackedQKVParallelLinearCombine requires three-field input/output tuples, got {inputType} -> {outputType}.");
        }

        if (input == output && input.Fields.All(IsMaterialized))
        {
            return output;
        }

        var inputFields = input.Fields.OfType<DistributedType>().ToArray();
        var outputFields = output.Fields.OfType<DistributedType>().ToArray();
        if (inputFields.Length != 3 || outputFields.Length != 3 ||
            !inputFields.All(field => field.Partial is { Op: ReduceOp.Sum }) ||
            outputFields.Any(field => field.Partial is not null) ||
            !inputFields.Skip(1).All(field => HasSamePartial(field.Partial!, inputFields[0].Partial!)) ||
            !inputFields.Zip(outputFields).All(pair => CanCombineTo(pair.First, pair.Second)))
        {
            return new InvalidType(
                $"PackedQKVParallelLinearCombine cannot materialize {inputType} into {outputType}.");
        }

        return output;
    }

    public Cost Visit(ICostEvaluateContext context, PackedQKVParallelLinearCombine target)
    {
        var input = context.GetArgumentType<TupleType>(target, PackedQKVParallelLinearCombine.QKV);
        var output = context.GetReturnType<TupleType>();
        if (input == output && input.Fields.All(IsMaterialized))
        {
            return Cost.Zero;
        }

        var inputFields = input.Fields.Cast<DistributedType>().ToArray();
        var outputFields = output.Fields.Cast<DistributedType>().ToArray();
        var cost = Cost.Zero;
        foreach (var (inputField, outputField) in inputFields.Zip(outputFields))
        {
            var localOutput = DistributedUtility.GetDividedTensorType(
                outputField,
                DistributedUtility.DivideFlags.MaxShape);
            var fanIn = inputField.Partial!.Axes.Aggregate(
                1.0,
                (product, axis) => product * inputField.Placement.Hierarchy[axis]);
            var inputTensor = new TargetCostTensor(inputField.TensorType.DType, localOutput.Shape);
            var outputTensor = new TargetCostTensor(localOutput.DType, localOutput.Shape);
            if (!context.TargetCostModel.TryGetElementwiseCost(
                    new(
                        "packed_qkv_partial_reduce",
                        [inputTensor],
                        outputTensor,
                        WorkPerElement: System.Math.Max(0.0, fanIn - 1.0),
                        InputReadMultiplicity: fanIn),
                    out var fieldCost))
            {
                var bytes = CostUtility.GetMemoryAccess(localOutput);
                fieldCost = new Cost
                {
                    [CostFactorNames.BlockLocalMemoryLoadBytes] = (UInt128)(fanIn * (double)bytes),
                    [CostFactorNames.BlockLocalMemoryStoreBytes] = bytes,
                    [CostFactorNames.CPUCycles] = (UInt128)((fanIn - 1.0) *
                        (double)CostUtility.GetCPUCycles(localOutput, 1)),
                };
            }

            cost += fieldCost;
        }

        return cost + new Cost { [CostFactorNames.GridSynchronization] = 1 };
    }

    private static bool IsMaterialized(IRType type)
        => type is not DistributedType distributed ||
            (distributed.Partial is null && distributed.AxisPolicies.All(policy => policy is not SBPPartial));

    private static bool HasSamePartial(SBPPartial lhs, SBPPartial rhs)
        => lhs.Op == rhs.Op && lhs.Axes.SequenceEqual(rhs.Axes);

    private static bool CanCombineTo(DistributedType input, DistributedType output)
    {
        if (input.TensorType != output.TensorType ||
            input.Placement != output.Placement ||
            input.Partial is not { } partial ||
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
            else if (!HasSameHierarchyPolicy(inputHierarchy[axis], outputHierarchy[axis]))
            {
                return false;
            }
        }

        return true;
    }

    private static bool HasSameHierarchyPolicy(
        HierarchyAxisPolicy input,
        HierarchyAxisPolicy output)
        => (input, output) switch
        {
            (HierarchyAxisBroadcast, HierarchyAxisBroadcast) => true,
            (HierarchyAxisSplit lhs, HierarchyAxisSplit rhs) =>
                lhs.TensorAxis == rhs.TensorAxis && lhs.Distribution == rhs.Distribution,
            _ => false,
        };
}
