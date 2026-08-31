// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Math;
using Nncase.IR.NTT;
using Nncase.Utilities;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluates and validates materialization of direct or split-K GLU projections.
/// </summary>
public sealed class PackedMatMulGluCombineEvaluator :
    IEvaluator<PackedMatMulGluCombine>,
    ITypeInferencer<PackedMatMulGluCombine>,
    ICostEvaluator<PackedMatMulGluCombine>
{
    public IValue Visit(IEvaluateContext context, PackedMatMulGluCombine target)
    {
        var input = context.GetArgumentValue(target, PackedMatMulGluCombine.Projections);
        if (input is not TupleValue { Count: 2 } projections)
        {
            return input;
        }

        return Value.FromTensor(PackedMatMulGluEvaluator.ApplyGlu(
            projections[0].AsTensor(),
            projections[1].AsTensor(),
            target.GluType));
    }

    public IRType Visit(ITypeInferenceContext context, PackedMatMulGluCombine target)
        => InferType(
            context.CheckArgumentType<IRType>(target, PackedMatMulGluCombine.Projections),
            target.OutputType,
            target.GluType);

    public static IRType InferType(
        IRType inputType,
        IRType outputType,
        global::Nncase.IR.NN.GluType gluType)
    {
        if (gluType != global::Nncase.IR.NN.GluType.SwiGLU)
        {
            return new InvalidType($"Unsupported PackedMatMulGluCombine type: {gluType}.");
        }

        if (inputType == outputType && IsMaterialized(inputType))
        {
            return outputType;
        }

        if (inputType is not TupleType { Count: 2 } input ||
            input.Fields[0] is not DistributedType gate ||
            input.Fields[1] is not DistributedType up ||
            outputType is not DistributedType output ||
            gate.Partial is not { Op: ReduceOp.Sum } partial ||
            up.Partial is not { Op: ReduceOp.Sum } upPartial ||
            !HasSamePartial(partial, upPartial) ||
            !CanCombineTo(gate, output) ||
            !CanCombineTo(up, output))
        {
            return new InvalidType(
                $"PackedMatMulGluCombine cannot materialize {inputType} into {outputType}.");
        }

        return outputType;
    }

    public Cost Visit(ICostEvaluateContext context, PackedMatMulGluCombine target)
    {
        var input = context.GetArgumentType<IRType>(target, PackedMatMulGluCombine.Projections);
        var output = context.GetReturnType<IRType>();
        if (input == output && IsMaterialized(input))
        {
            return Cost.Zero;
        }

        if (input is not TupleType { Count: 2 } tuple ||
            tuple.Fields[0] is not DistributedType gate ||
            output is not DistributedType distributedOutput)
        {
            return new Cost { [CostFactorNames.CPUCycles] = 1 };
        }

        var localOutput = DistributedUtility.GetDividedTensorType(
            distributedOutput,
            DistributedUtility.DivideFlags.MaxShape);
        var fanIn = gate.Partial!.Axes.Aggregate(
            1.0,
            (product, axis) => product * gate.Placement.Hierarchy[axis]);
        var inputTensor = new TargetCostTensor(gate.TensorType.DType, localOutput.Shape);
        var outputTensor = new TargetCostTensor(localOutput.DType, localOutput.Shape);
        Cost cost;
        if (!context.TargetCostModel.TryGetElementwiseCost(
                new(
                    "packed_matmul_glu_partial_reduce",
                    [inputTensor, inputTensor],
                    outputTensor,
                    WorkPerElement: checked((2.0 * System.Math.Max(0.0, fanIn - 1.0)) + 9.0),
                    InputReadMultiplicity: fanIn),
                out cost))
        {
            var bytes = CostUtility.GetMemoryAccess(localOutput);
            cost = new Cost
            {
                [CostFactorNames.BlockLocalMemoryLoadBytes] = (UInt128)(2.0 * fanIn * (double)bytes),
                [CostFactorNames.BlockLocalMemoryStoreBytes] = bytes,
                [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(
                    localOutput,
                    checked((uint)((2.0 * System.Math.Max(0.0, fanIn - 1.0)) + 9.0))),
            };
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
                    outputHierarchy[axis] is not HierarchyAxisBroadcast)
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
