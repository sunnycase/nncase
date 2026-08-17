// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Utilities;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="PackedMatMulNormStats"/>.
/// </summary>
public sealed class PackedMatMulNormStatsEvaluator :
    IEvaluator<PackedMatMulNormStats>,
    ITypeInferencer<PackedMatMulNormStats>,
    ICostEvaluator<PackedMatMulNormStats>
{
    public IValue Visit(IEvaluateContext context, PackedMatMulNormStats target)
    {
        var packedTarget = CreatePackedMatMul(target);
        var output = PackedMatMulEvaluator.Evaluate(
            packedTarget,
            context.GetOrtArgumentValue(target, PackedMatMulNormStats.Lhs),
            context.GetArgumentValueAsTensor(target, PackedMatMulNormStats.Rhs),
            context.GetArgumentValue(target, PackedMatMulNormStats.Scale),
            context.GetArgumentValue(target, PackedMatMulNormStats.Addend)).AsTensor();
        return Value.FromTensors(
            output,
            NormStatsEvaluator.Evaluate(output, target.Axis, target.UseMean));
    }

    public IRType Visit(ITypeInferenceContext context, PackedMatMulNormStats target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, PackedMatMulNormStats.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, PackedMatMulNormStats.Rhs);
        var scale = context.CheckArgumentType<IRType>(target, PackedMatMulNormStats.Scale);
        var addend = context.CheckArgumentType<IRType>(target, PackedMatMulNormStats.Addend);
        return InferType(target, lhs, rhs, scale, addend);
    }

    public static IRType InferType(
        PackedMatMulNormStats target,
        IRType lhs,
        IRType rhs,
        IRType scale,
        IRType addend)
    {
        var output = PackedMatMulEvaluator.InferType(
            CreatePackedMatMul(target),
            lhs,
            rhs,
            scale,
            addend);
        if (output is InvalidType)
        {
            return output;
        }

        if (output is DistributedType { Partial: not null })
        {
            return new InvalidType(
                "PackedMatMulNormStats requires a non-partial matmul output; reduce the matmul result before forming normalization statistics.");
        }

        var stats = NormStatsEvaluator.InferType(
            new NormStats(target.Axis, target.UseMean),
            output);
        return stats is InvalidType ? stats : new TupleType(new[] { output, stats });
    }

    public Cost Visit(ICostEvaluateContext context, PackedMatMulNormStats target)
    {
        var lhs = context.GetArgumentType<IRType>(target, PackedMatMulNormStats.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, PackedMatMulNormStats.Rhs);
        var addend = context.GetArgumentType<IRType>(target, PackedMatMulNormStats.Addend);
        var output = context.GetReturnType<TupleType>();
        var valueOutput = output.Fields[0];
        var statsOutput = output.Fields[1];
        var packedTarget = CreatePackedMatMul(target);
        if (PackedMatMulEvaluator.TryGetTargetCost(
                context,
                packedTarget,
                lhs,
                rhs,
                valueOutput,
                out var targetCost,
                out _))
        {
            PackedMatMulEvaluator.AddAddendCost(targetCost, valueOutput, addend);
            AddStatsCost(context, target, targetCost, valueOutput, statsOutput);
            return targetCost;
        }

        var macPerElement = GetReductionExtent(lhs);
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(lhs)
                + CostUtility.GetMemoryAccess(rhs)
                + CostUtility.GetMemoryAccess(addend),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(valueOutput)
                + CostUtility.GetMemoryAccess(statsOutput),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(
                valueOutput,
                checked(macPerElement + (addend is NoneType ? 0U : 1U) + (target.UseMean ? 3U : 2U))),
        };
    }

    private static void AddStatsCost(
        ICostEvaluateContext context,
        PackedMatMulNormStats target,
        Cost cost,
        IRType valueOutput,
        IRType statsOutput)
    {
        var workPerElement = target.UseMean ? 3.0 : 2.0;
        if (TargetCostTensor.TryFromType(valueOutput, out var valueTensor)
            && context.TargetCostModel.TryGetElementwiseCost(
                new(
                    "packed_mat_mul_norm_stats",
                    [valueTensor],
                    valueTensor,
                    workPerElement),
                out var statsCost)
            && statsCost.Factors.TryGetValue(CostFactorNames.CPUCycles, out var statsCycles))
        {
            AddCostFactor(cost, CostFactorNames.CPUCycles, statsCycles);
        }
        else
        {
            AddCostFactor(
                cost,
                CostFactorNames.CPUCycles,
                CostUtility.GetCPUCycles(valueOutput, workPerElement));
        }

        AddCostFactor(
            cost,
            CostFactorNames.BlockLocalMemoryStoreBytes,
            CostUtility.GetMemoryAccess(statsOutput));
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

    private static PackedMatMul CreatePackedMatMul(PackedMatMulNormStats target) =>
        new(target.OutputDataType, false, target.RhsLayout);

    private static uint GetReductionExtent(IRType input)
    {
        var localType = input is DistributedType distributed
            ? DistributedUtility.GetDividedTensorType(distributed)
            : input as TensorType;
        if (localType is not TensorType { Shape: RankedShape shape })
        {
            return 1;
        }

        var extent = shape[^1];
        return extent.IsFixed ? checked((uint)extent.FixedValue) : 1U;
    }
}
