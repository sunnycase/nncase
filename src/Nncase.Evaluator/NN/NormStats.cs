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

/// <summary>
/// Evaluator for <see cref="NormStats"/>.
/// </summary>
public sealed class NormStatsEvaluator : IEvaluator<NormStats>, ITypeInferencer<NormStats>, ICostEvaluator<NormStats>
{
    public IValue Visit(IEvaluateContext context, NormStats target)
    {
        var inputRaw = UnpackVectorInput(context.GetArgumentValueAsTensor(target, NormStats.Input));
        var input = inputRaw.CastElementTo(DataTypes.Float32).Cast<float>();
        var shape = input.Shape.ToValueArray();
        var rank = shape.Length;
        var normalizedAxis = NormUtility.NormalizeAxis(target.Axis, rank);
        var outerSize = TensorUtilities.GetProduct(shape.AsSpan(0, normalizedAxis));
        var innerSize = TensorUtilities.GetProduct(shape.AsSpan(normalizedAxis));
        var components = target.UseMean ? 2 : 1;

        var statsShape = new long[rank + 1];
        statsShape[0] = components;
        for (int i = 0; i < rank; i++)
        {
            statsShape[i + 1] = i < normalizedAxis ? shape[i] : 1;
        }

        var stats = new float[checked(components * outerSize)];
        var inputSpan = input.Buffer.Span;
        for (int outer = 0; outer < outerSize; outer++)
        {
            var baseOffset = outer * innerSize;
            float sum = 0f;
            float sumSq = 0f;
            for (int inner = 0; inner < innerSize; inner++)
            {
                var value = inputSpan[checked((int)(baseOffset + inner))];
                sum += value;
                sumSq += value * value;
            }

            if (target.UseMean)
            {
                stats[outer] = sum;
                stats[checked((int)(outerSize + outer))] = sumSq;
            }
            else
            {
                stats[outer] = sumSq;
            }
        }

        return Value.FromTensor(Tensor.From(stats, statsShape));
    }

    public IRType Visit(ITypeInferenceContext context, NormStats target)
    {
        var input = context.CheckArgumentType<IRType>(target, NormStats.Input);
        return input switch
        {
            TensorType tensor => Visit(target, tensor),
            DistributedType distributed => Visit(target, distributed),
            _ => new InvalidType($"{nameof(NormStats)} input must be tensor-like, but got {input}."),
        };
    }

    public Cost Visit(ICostEvaluateContext context, NormStats target)
    {
        var input = context.GetArgumentType<IRType>(target, NormStats.Input);
        var output = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(input, target.UseMean ? 3 : 2),
        };
    }

    private static IRType Visit(NormStats target, TensorType input)
    {
        if (!input.DType.IsFloat())
        {
            return new InvalidType("NormStats input must be float.");
        }

        try
        {
            return NormUtility.GetStatsTensorType(input, target.Axis, target.UseMean);
        }
        catch (ArgumentOutOfRangeException ex)
        {
            return new InvalidType(ex.Message);
        }
    }

    private static IRType Visit(NormStats target, DistributedType input)
    {
        if (NormUtility.HasPartial(input))
        {
            return new InvalidType("NormStats input must not be partial.");
        }

        if (NormUtility.GetStatsTensorType(input, target.Axis, target.UseMean) is not TensorType statsType || statsType.Shape.IsInvalid)
        {
            return new InvalidType("NormStats input has invalid shape or dtype.");
        }

        if (input.TensorType.Shape is not RankedShape inputShape)
        {
            return new InvalidType("NormStats distributed input must have ranked shape.");
        }

        var normalizedAxis = NormUtility.NormalizeAxis(target.Axis, inputShape.Rank);
        var policies = Enumerable.Repeat(SBP.B, inputShape.Rank + 1).Select(policy => (SBP)policy).ToArray();
        var partialAxes = new HashSet<int>();
        var preservedSplitAxes = new HashSet<int>();

        for (int i = 0; i < inputShape.Rank; i++)
        {
            var policy = input.AxisPolicies[i];
            if (i < normalizedAxis)
            {
                policies[i + 1] = policy;
                if (policy is SBPSplit preservedSplit)
                {
                    foreach (var axis in preservedSplit.Axes)
                    {
                        preservedSplitAxes.Add(axis);
                    }
                }
            }
            else if (policy is SBPSplit reduceSplit)
            {
                foreach (var axis in reduceSplit.Axes)
                {
                    partialAxes.Add(axis);
                }
            }
            else if (policy is not SBPBroadCast)
            {
                return new InvalidType($"NormStats does not support policy {policy} on input axis {i}.");
            }
        }

        if (partialAxes.Overlaps(preservedSplitAxes))
        {
            return new InvalidType("NormStats cannot preserve and reduce on the same placement axis.");
        }

        if (!DistributedUtility.IsDistributable(statsType, policies, input.Placement))
        {
            return new InvalidType("NormStats output policies are not distributable.");
        }

        var partial = partialAxes.Count == 0 ? null : SBP.P(partialAxes.OrderBy(axis => axis).ToArray(), ReduceOp.Sum);
        return new DistributedType(statsType, policies, input.Placement, partial);
    }

    private static Tensor UnpackVectorInput(Tensor input)
    {
        if (input.ElementType is not VectorType vectorType)
        {
            return input;
        }

        if (input.Shape.Count == 0)
        {
            throw new NotSupportedException("NormStats evaluator does not support vector scalar tensor values.");
        }

        var axes = Enumerable.Repeat(input.Shape.Count - 1, vectorType.Lanes.Count).ToArray();
        return Tensors.UnpackEvaluator.UnpackImpl(input, axes);
    }
}
