// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="NormApply"/>.
/// </summary>
public sealed class NormApplyEvaluator : IEvaluator<NormApply>, ITypeInferencer<NormApply>, ICostEvaluator<NormApply>
{
    public IValue Visit(IEvaluateContext context, NormApply target)
    {
        var inputRaw = context.GetArgumentValueAsTensor(target, NormApply.Input);
        var statsRaw = context.GetArgumentValueAsTensor(target, NormApply.Stats);
        var scaleRaw = context.GetArgumentValueAsTensor(target, NormApply.Scale);
        var biasRaw = context.GetArgumentValueAsTensor(target, NormApply.Bias);
        if (inputRaw.ElementType is VectorType || statsRaw.ElementType is VectorType)
        {
            throw new NotSupportedException("NormApply evaluator does not support vector tensor values.");
        }

        var input = inputRaw.CastElementTo(DataTypes.Float32).Cast<float>();
        var stats = statsRaw.CastElementTo(DataTypes.Float32).Cast<float>();
        var scale = scaleRaw.CastElementTo(DataTypes.Float32).Cast<float>();
        var bias = biasRaw.CastElementTo(DataTypes.Float32).Cast<float>();
        var originType = context.CurrentCall.Arguments[NormApply.Input.Index].CheckedDataType;

        var shape = input.Shape.ToValueArray();
        var rank = shape.Length;
        var normalizedAxis = NormUtility.NormalizeAxis(target.Axis, rank);
        var outerSize = TensorUtilities.GetProduct(shape.AsSpan(0, normalizedAxis));
        var innerSize = TensorUtilities.GetProduct(shape.AsSpan(normalizedAxis));
        var normalizationSize = GetNormalizationSize(context, target, innerSize);

        var output = new float[input.Length];
        var inputSpan = input.Buffer.Span;
        var statsSpan = stats.Buffer.Span;
        var scaleSpan = scale.Buffer.Span;
        var biasSpan = bias.Buffer.Span;
        for (int outer = 0; outer < outerSize; outer++)
        {
            float mean = 0f;
            float sumSq;
            if (target.UseMean)
            {
                mean = statsSpan[outer] / normalizationSize;
                sumSq = statsSpan[checked((int)(outerSize + outer))];
            }
            else
            {
                sumSq = statsSpan[outer];
            }

            var variance = sumSq / normalizationSize;
            if (target.UseMean)
            {
                variance -= mean * mean;
            }

            variance = MathF.Max(variance, 0f);
            var rstd = 1f / MathF.Sqrt(variance + target.Epsilon);
            var baseOffset = outer * innerSize;
            for (int inner = 0; inner < innerSize; inner++)
            {
                var localOffset = checked((int)(baseOffset + inner));
                var value = inputSpan[localOffset];
                var centered = target.UseMean ? value - mean : value;
                output[localOffset] = (centered * rstd * scaleSpan[inner % scaleSpan.Length]) + biasSpan[inner % biasSpan.Length];
            }
        }

        return Value.FromTensor(Tensor.From(output, shape).CastTo(originType));
    }

    public IRType Visit(ITypeInferenceContext context, NormApply target)
    {
        var input = context.CheckArgumentType<IRType>(target, NormApply.Input);
        var stats = context.CheckArgumentType<IRType>(target, NormApply.Stats);
        var scale = context.CheckArgumentType<IRType>(target, NormApply.Scale);
        var bias = context.CheckArgumentType<IRType>(target, NormApply.Bias);

        return (input, stats, scale, bias) switch
        {
            (TensorType inputTensor, TensorType statsTensor, TensorType scaleTensor, TensorType biasTensor) => Visit(target, inputTensor, statsTensor, scaleTensor, biasTensor),
            (DistributedType inputDistributed, DistributedType statsDistributed, DistributedType scaleDistributed, DistributedType biasDistributed) => Visit(target, inputDistributed, statsDistributed, scaleDistributed, biasDistributed),
            _ => new InvalidType($"{nameof(NormApply)} arguments must all be tensor or all be distributed tensors."),
        };
    }

    public Cost Visit(ICostEvaluateContext context, NormApply target)
    {
        var input = context.GetArgumentType<IRType>(target, NormApply.Input);
        var stats = context.GetArgumentType<IRType>(target, NormApply.Stats);
        var scale = context.GetArgumentType<IRType>(target, NormApply.Scale);
        var bias = context.GetArgumentType<IRType>(target, NormApply.Bias);
        var output = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                CostUtility.GetMemoryAccess(input)
                + CostUtility.GetMemoryAccess(stats)
                + CostUtility.GetMemoryAccess(scale)
                + CostUtility.GetMemoryAccess(bias),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(input, target.UseMean ? 7 : 5),
        };
    }

    private static IRType Visit(NormApply target, TensorType input, TensorType stats, TensorType scale, TensorType bias)
    {
        if (!input.DType.IsFloat() || !scale.DType.IsFloat() || !bias.DType.IsFloat())
        {
            return new InvalidType("NormApply input, scale and bias must be float.");
        }

        if (!IsStatsTypeCompatible(target, input, stats, out var expectedStats))
        {
            return new InvalidType($"NormApply stats type {stats} does not match expected {expectedStats}.");
        }

        return input;
    }

    private static IRType Visit(NormApply target, DistributedType input, DistributedType stats, DistributedType scale, DistributedType bias)
    {
        if (input.Placement != stats.Placement || input.Placement != scale.Placement || input.Placement != bias.Placement)
        {
            return new InvalidType("NormApply argument placements must be equal.");
        }

        if (NormUtility.HasPartial(input) || NormUtility.HasPartial(stats) || NormUtility.HasPartial(scale) || NormUtility.HasPartial(bias))
        {
            return new InvalidType("NormApply requires non-partial input, stats, scale and bias.");
        }

        if (Visit(target, input.TensorType, stats.TensorType, scale.TensorType, bias.TensorType) is InvalidType invalid)
        {
            return invalid;
        }

        if (input.TensorType.Shape is not RankedShape inputShape)
        {
            return new InvalidType("NormApply distributed input must have ranked shape.");
        }

        var normalizedAxis = NormUtility.NormalizeAxis(target.Axis, inputShape.Rank);
        if (stats.AxisPolicies.Count != inputShape.Rank + 1 || stats.AxisPolicies[0] is not SBPBroadCast)
        {
            return new InvalidType("NormApply stats policies must match NormStats output rank.");
        }

        for (int i = 0; i < inputShape.Rank; i++)
        {
            var statsPolicy = stats.AxisPolicies[i + 1];
            if (i < normalizedAxis)
            {
                if (!NormUtility.IsSamePolicy(input.AxisPolicies[i], statsPolicy))
                {
                    return new InvalidType($"NormApply stats policy {statsPolicy} does not match input policy {input.AxisPolicies[i]} on outer axis {i}.");
                }
            }
            else if (statsPolicy is not SBPBroadCast)
            {
                return new InvalidType($"NormApply stats policy on normalized axis {i} must be broadcast.");
            }
        }

        var parameterRank = inputShape.Rank - normalizedAxis;
        if (scale.AxisPolicies.Count != parameterRank || bias.AxisPolicies.Count != parameterRank)
        {
            return new InvalidType("NormApply scale and bias policies must match normalized rank.");
        }

        for (int j = 0; j < parameterRank; j++)
        {
            var inputPolicy = input.AxisPolicies[normalizedAxis + j];
            if (!NormUtility.IsParameterPolicyCompatible(inputPolicy, scale.AxisPolicies[j]))
            {
                return new InvalidType($"NormApply scale policy {scale.AxisPolicies[j]} is not compatible with input policy {inputPolicy}.");
            }

            if (!NormUtility.IsParameterPolicyCompatible(inputPolicy, bias.AxisPolicies[j]))
            {
                return new InvalidType($"NormApply bias policy {bias.AxisPolicies[j]} is not compatible with input policy {inputPolicy}.");
            }
        }

        return new DistributedType(input.TensorType, input.AxisPolicies, input.Placement);
    }

    private static long GetNormalizationSize(IEvaluateContext context, NormApply target, long fallback)
    {
        var inputType = context.CurrentCall.Arguments[NormApply.Input.Index].CheckedType;
        return inputType switch
        {
            TensorType tensor => NormUtility.GetNormalizationSize(tensor, target.Axis, fallback),
            DistributedType distributed => NormUtility.GetNormalizationSize(distributed.TensorType, target.Axis, fallback),
            _ => fallback,
        };
    }

    private static bool IsStatsTypeCompatible(NormApply target, TensorType input, TensorType stats, out TensorType expectedStats)
    {
        expectedStats = NormUtility.GetStatsTensorType(input, target.Axis, target.UseMean);
        if (stats == expectedStats)
        {
            return true;
        }

        if (input.DType is not VectorType vectorInput || stats.DType != DataTypes.Float32)
        {
            return false;
        }

        var scalarInput = new TensorType(vectorInput.ElemType, input.Shape);
        expectedStats = NormUtility.GetStatsTensorType(scalarInput, target.Axis, target.UseMean);
        return stats == expectedStats;
    }
}
