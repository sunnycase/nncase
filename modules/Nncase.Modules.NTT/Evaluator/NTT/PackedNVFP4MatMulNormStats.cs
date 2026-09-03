// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="PackedNVFP4MatMulNormStats"/>.
/// </summary>
public sealed class PackedNVFP4MatMulNormStatsEvaluator :
    IEvaluator<PackedNVFP4MatMulNormStats>,
    ITypeInferencer<PackedNVFP4MatMulNormStats>,
    ICostEvaluator<PackedNVFP4MatMulNormStats>
{
    public IValue Visit(IEvaluateContext context, PackedNVFP4MatMulNormStats target)
    {
        var output = PackedNVFP4MatMulEvaluator.Evaluate(
            CreatePackedMatMul(target),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulNormStats.Lhs),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulNormStats.RhsPacked),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulNormStats.RhsScale),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulNormStats.LhsGlobalScale),
            context.GetArgumentValueAsTensor(target, PackedNVFP4MatMulNormStats.RhsGlobalScale),
            context.GetArgumentValue(target, PackedNVFP4MatMulNormStats.Addend));
        return Value.FromTensors(
            output,
            NormStatsEvaluator.Evaluate(output, target.Axis, target.UseMean));
    }

    public IRType Visit(ITypeInferenceContext context, PackedNVFP4MatMulNormStats target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.Lhs),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.RhsPacked),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.RhsScale),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.LhsGlobalScale),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.RhsGlobalScale),
            context.CheckArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.Addend));

    public static IRType InferType(
        PackedNVFP4MatMulNormStats target,
        IRType lhs,
        IRType rhsPacked,
        IRType rhsScale,
        IRType lhsGlobalScale,
        IRType rhsGlobalScale,
        IRType addend)
    {
        var output = PackedNVFP4MatMulEvaluator.InferType(
            CreatePackedMatMul(target),
            lhs,
            rhsPacked,
            rhsScale,
            lhsGlobalScale,
            rhsGlobalScale,
            addend);
        if (output is InvalidType)
        {
            return output;
        }

        if (output is DistributedType { Partial: not null })
        {
            return new InvalidType(
                "PackedNVFP4MatMulNormStats requires a non-partial matmul output.");
        }

        var stats = NormStatsEvaluator.InferType(
            new NormStats(target.Axis, target.UseMean),
            output);
        return stats is InvalidType ? stats : new TupleType(new[] { output, stats });
    }

    public Cost Visit(ICostEvaluateContext context, PackedNVFP4MatMulNormStats target)
    {
        var lhs = context.GetArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.Lhs);
        var addend = context.GetArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.Addend);
        var output = context.GetReturnType<TupleType>();
        var valueOutput = output.Fields[0];
        var statsOutput = output.Fields[1];
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                CostUtility.GetMemoryAccess(lhs) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.RhsPacked)) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.RhsScale)) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.LhsGlobalScale)) +
                CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, PackedNVFP4MatMulNormStats.RhsGlobalScale)) +
                (addend is NoneType ? 0 : CostUtility.GetMemoryAccess(addend)),
            [CostFactorNames.BlockLocalMemoryStoreBytes] =
                CostUtility.GetMemoryAccess(valueOutput) + CostUtility.GetMemoryAccess(statsOutput),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(
                valueOutput,
                checked(
                    PackedNVFP4MatMulEvaluator.GetLogicalK(lhs) + 8U +
                    (addend is NoneType ? 0U : 1U) + (target.UseMean ? 3U : 2U))),
        };
    }

    private static PackedNVFP4MatMul CreatePackedMatMul(PackedNVFP4MatMulNormStats target) =>
        new(
            target.OutputDataType,
            target.GroupSize,
            target.InputKVectorLaneCount,
            target.RhsKPackLaneCount,
            target.RhsKVectorLaneCount,
            target.OutputNVectorLaneCount);
}
