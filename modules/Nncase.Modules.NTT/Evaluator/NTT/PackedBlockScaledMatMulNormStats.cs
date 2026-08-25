// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using ScaledMatMulEvaluator = Nncase.Evaluator.Math.ScaledMatMulEvaluator;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="PackedBlockScaledMatMulNormStats"/>.
/// </summary>
public sealed class PackedBlockScaledMatMulNormStatsEvaluator :
    IEvaluator<PackedBlockScaledMatMulNormStats>,
    ITypeInferencer<PackedBlockScaledMatMulNormStats>,
    ICostEvaluator<PackedBlockScaledMatMulNormStats>
{
    public IValue Visit(IEvaluateContext context, PackedBlockScaledMatMulNormStats target)
    {
        var output = PackedBlockScaledMatMulEvaluator.Evaluate(
            CreatePackedBlockScaledMatMul(target),
            context.GetArgumentValueAsTensor(target, PackedBlockScaledMatMulNormStats.Lhs),
            context.GetArgumentValueAsTensor(target, PackedBlockScaledMatMulNormStats.Rhs),
            context.GetArgumentValueAsTensor(target, PackedBlockScaledMatMulNormStats.RhsScale),
            context.GetArgumentValue(target, PackedBlockScaledMatMulNormStats.Addend)).AsTensor();
        return Value.FromTensors(
            output,
            NormStatsEvaluator.Evaluate(output, target.Axis, target.UseMean));
    }

    public IRType Visit(ITypeInferenceContext context, PackedBlockScaledMatMulNormStats target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, PackedBlockScaledMatMulNormStats.Lhs),
            context.CheckArgumentType<IRType>(target, PackedBlockScaledMatMulNormStats.Rhs),
            context.CheckArgumentType<IRType>(target, PackedBlockScaledMatMulNormStats.RhsScale),
            context.CheckArgumentType<IRType>(target, PackedBlockScaledMatMulNormStats.Addend));

    public static IRType InferType(
        PackedBlockScaledMatMulNormStats target,
        IRType lhs,
        IRType rhs,
        IRType rhsScale,
        IRType addend)
    {
        var output = PackedBlockScaledMatMulEvaluator.InferType(
            CreatePackedBlockScaledMatMul(target),
            lhs,
            rhs,
            rhsScale,
            addend);
        if (output is InvalidType)
        {
            return output;
        }

        if (output is DistributedType { Partial: not null })
        {
            return new InvalidType(
                "PackedBlockScaledMatMulNormStats requires a non-partial matmul output.");
        }

        var stats = NormStatsEvaluator.InferType(
            new NormStats(target.Axis, target.UseMean),
            output);
        return stats is InvalidType ? stats : new TupleType(new[] { output, stats });
    }

    public Cost Visit(ICostEvaluateContext context, PackedBlockScaledMatMulNormStats target)
    {
        var lhs = context.GetArgumentType<IRType>(target, PackedBlockScaledMatMulNormStats.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, PackedBlockScaledMatMulNormStats.Rhs);
        var rhsScale = context.GetArgumentType<IRType>(target, PackedBlockScaledMatMulNormStats.RhsScale);
        var addend = context.GetArgumentType<IRType>(target, PackedBlockScaledMatMulNormStats.Addend);
        var output = context.GetReturnType<TupleType>();
        var valueOutput = output.Fields[0];
        var statsOutput = output.Fields[1];
        var addendWork = addend is NoneType ? 0U : 1U;
        var statsWork = target.UseMean ? 3U : 2U;
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs) +
                CostUtility.GetMemoryAccess(rhsScale) +
                (addend is NoneType ? 0 : CostUtility.GetMemoryAccess(addend)),
            [CostFactorNames.BlockLocalMemoryStoreBytes] =
                CostUtility.GetMemoryAccess(valueOutput) + CostUtility.GetMemoryAccess(statsOutput),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(
                valueOutput,
                checked(GetK(lhs) + 4U + addendWork + statsWork)),
        };
    }

    private static PackedBlockScaledMatMul CreatePackedBlockScaledMatMul(
        PackedBlockScaledMatMulNormStats target) =>
        new(
            target.OutputDataType,
            target.RhsLayout,
            target.OutputNVectorLaneCount,
            target.WeightBlockN,
            target.WeightBlockK);

    private static uint GetK(IRType lhs)
    {
        var tensor = ScaledMatMulEvaluator.GetTensorType(lhs);
        return tensor?.Shape is RankedShape { Rank: > 0 } shape && shape[^1].IsFixed
            ? checked((uint)shape[^1].FixedValue)
            : 1U;
    }
}
