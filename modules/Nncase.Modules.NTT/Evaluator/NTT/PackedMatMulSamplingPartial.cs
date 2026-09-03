// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NTT;

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
        packed = ApplyScale(
            packed,
            context.GetArgumentValue(target, PackedMatMulSamplingPartial.LhsScale));
        packed = ApplyScale(
            packed,
            context.GetArgumentValue(target, PackedMatMulSamplingPartial.RhsScale));
        var logits = ToLogicalLogits(packed, target.OutputDataType);
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
            context.CheckArgumentType<IRType>(target, PackedMatMulSamplingPartial.Addend),
            context.CheckArgumentType<IRType>(target, PackedMatMulSamplingPartial.LhsScale),
            context.CheckArgumentType<IRType>(target, PackedMatMulSamplingPartial.RhsScale));

    public static IRType InferType(
        PackedMatMulSamplingPartial target,
        IRType lhsType,
        IRType rhsType,
        TensorType stateType,
        IRType scaleType,
        IRType addendType,
        IRType lhsScaleType,
        IRType rhsScaleType)
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

        var hasLhsScale = lhsScaleType is not NoneType;
        var hasRhsScale = rhsScaleType is not NoneType;
        if (hasLhsScale != hasRhsScale)
        {
            return new InvalidType(
                "PackedMatMulSamplingPartial requires both lhs/rhs output scales or neither.");
        }

        if (hasLhsScale)
        {
            if (ValidateScaleType(packedOutputType, lhsScaleType, "lhs") is { } lhsError)
            {
                return lhsError;
            }

            if (ValidateScaleType(packedOutputType, rhsScaleType, "rhs") is { } rhsError)
            {
                return rhsError;
            }
        }

        var logitsType = InferLogicalLogitsType(
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
        => new(target.AccumulatorDataType, false, target.RhsLayout);

    private static Tensor ApplyScale(Tensor input, IValue scale)
        => scale is NoneValue || scale.Type is NoneType
            ? input
            : global::Nncase.IR.F.Math.Mul(input, scale.AsTensor()).Evaluate().AsTensor();

    private static Tensor ToLogicalLogits(Tensor packed, DataType outputDataType)
    {
        if (packed.ElementType is not VectorType vectorType)
        {
            throw new InvalidOperationException(
                $"PackedMatMulSamplingPartial expects a vector packed result, got {packed.ElementType}.");
        }

        var outputAxis = packed.Rank - 1;
        var logical = global::Nncase.IR.F.Tensors.Unpack(
                packed,
                vectorType.Lanes.ToArray(),
                Enumerable.Repeat(outputAxis, vectorType.Lanes.Count).ToArray())
            .Evaluate()
            .AsTensor();
        return logical.CastElementTo(outputDataType);
    }

    private static IRType InferLogicalLogitsType(
        IRType packedOutputType,
        DataType outputDataType)
    {
        var tensorType = packedOutputType switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => distributed.TensorType,
            _ => null,
        };
        if (tensorType?.DType is not VectorType)
        {
            return new InvalidType(
                $"PackedMatMulSamplingPartial expects a vector packed result, got {packedOutputType}.");
        }

        var outputAxis = tensorType.Shape.Rank - 1;
        var logicalType = packedOutputType switch
        {
            TensorType tensor => TypeInference.UnpackType(tensor, [outputAxis]),
            DistributedType distributed => TypeInference.UnpackType(distributed, [outputAxis]),
            _ => throw new InvalidOperationException(),
        };
        return logicalType switch
        {
            TensorType tensor => tensor with { DType = outputDataType },
            DistributedType distributed => distributed with
            {
                TensorType = distributed.TensorType with { DType = outputDataType },
            },
            _ => logicalType,
        };
    }

    private static InvalidType? ValidateScaleType(
        IRType packedOutputType,
        IRType scaleType,
        string name)
    {
        IRType broadcast = (packedOutputType, scaleType) switch
        {
            (TensorType output, TensorType scale) => TypeInference.BroadcastType(output, scale),
            (DistributedType output, DistributedType scale)
                when output.Placement == scale.Placement =>
                TypeInference.BroadcastType(output.TensorType, scale.TensorType) is TensorType tensor
                    ? Nncase.Evaluator.Math.BinaryEvaluator.CheckSBP(
                        BinaryOp.Mul,
                        tensor,
                        output,
                        scale)
                    : new InvalidType(
                        $"PackedMatMulSamplingPartial {name} scale cannot broadcast to its packed output."),
            _ => new InvalidType(
                $"PackedMatMulSamplingPartial {name} scale must use the same tensor/distributed domain as its packed output, " +
                $"got output={packedOutputType}, scale={scaleType}."),
        };

        if (broadcast is InvalidType invalid)
        {
            return invalid;
        }

        return Equals(broadcast, packedOutputType)
            ? null
            : new InvalidType(
                $"PackedMatMulSamplingPartial {name} scale must broadcast without changing the packed output type, " +
                $"got output={packedOutputType}, result={broadcast}.");
    }
}
