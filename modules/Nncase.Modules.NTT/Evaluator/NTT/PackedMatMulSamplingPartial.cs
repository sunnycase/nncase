// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluates and infers the post-distribution LM-head/sampling-partial fusion.
/// </summary>
public sealed class PackedMatMulSamplingPartialEvaluator :
    IEvaluator<PackedMatMulSamplingPartial>,
    ITypeInferencer<PackedMatMulSamplingPartial>
{
    public IValue Visit(IEvaluateContext context, PackedMatMulSamplingPartial target)
    {
        var packed = PackedMatMulEvaluator.Evaluate(
                CreatePackedMatMul(target),
                context.GetOrtArgumentValue(target, PackedMatMulSamplingPartial.Lhs),
                context.GetArgumentValueAsTensor(target, PackedMatMulSamplingPartial.Rhs),
                context.GetArgumentValue(target, PackedMatMulSamplingPartial.Scale),
                context.GetArgumentValue(target, PackedMatMulSamplingPartial.Addend))
            .AsTensor();
        var logits = packed.CastTo(target.OutputDataType, CastMode.Reinterpret);
        var partial = SamplingPartialEvaluator.Evaluate(
            logits,
            context.GetArgumentValue(target, PackedMatMulSamplingPartial.State),
            target.Config);
        return new TupleValue([
            Value.FromTensor(logits),
            partial[0],
            partial[1],
        ]);
    }

    public IRType Visit(ITypeInferenceContext context, PackedMatMulSamplingPartial target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, PackedMatMulSamplingPartial.Lhs),
            context.CheckArgumentType<IRType>(target, PackedMatMulSamplingPartial.Rhs),
            context.CheckArgumentType<TensorType>(target, PackedMatMulSamplingPartial.State),
            context.CheckArgumentType<IRType>(target, PackedMatMulSamplingPartial.Scale),
            context.CheckArgumentType<IRType>(target, PackedMatMulSamplingPartial.Addend));

    public static IRType InferType(
        PackedMatMulSamplingPartial target,
        IRType lhsType,
        IRType rhsType,
        TensorType stateType,
        IRType scaleType,
        IRType addendType)
    {
        var packedOutputType = PackedMatMulEvaluator.InferType(
            CreatePackedMatMul(target),
            lhsType,
            rhsType,
            scaleType,
            addendType);
        if (packedOutputType is InvalidType)
        {
            return packedOutputType;
        }

        if (packedOutputType is DistributedType { Partial: not null })
        {
            return new InvalidType(
                "PackedMatMulSamplingPartial requires a non-partial packed matmul output.");
        }

        var logitsType = BitcastUtility.InferType(
            packedOutputType,
            target.OutputDataType);
        if (logitsType is InvalidType)
        {
            return logitsType;
        }

        var partialType = SamplingPartialEvaluator.InferType(
            new SamplingPartial(target.Config),
            logitsType,
            stateType);
        if (partialType is not TupleType { Fields.Count: 2 } partial)
        {
            return partialType;
        }

        return new TupleType([
            logitsType,
            partial.Fields[0],
            partial.Fields[1],
        ]);
    }

    internal static PackedMatMul CreatePackedMatMul(PackedMatMulSamplingPartial target)
        => new(target.OutputDataType, false, target.RhsLayout);
}
