// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1010, SA1008
using System;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Tensors;

public sealed class PackEvaluator : ITypeInferencer<Pack>, ICostEvaluator<Pack>, IEvaluator<Pack>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Pack target)
    {
        var input = context.GetArgumentValueAsTensor(target, Pack.Input);
        var dt = input.ElementType;
        var elementType = dt is VectorType vt ? vt.ElemType : dt;
        var oldLanesCount = dt switch
        {
            VectorType vt2 => vt2.Lanes.Count,
            MaskVectorType => 1,
            _ => 0,
        };

        var physicalShape = input.Dimensions.ToArray().AsEnumerable();
        if (dt is VectorType vectorType)
        {
            physicalShape = physicalShape.Concat(vectorType.Lanes.Select(lane => (long)lane));
        }

        var output = input.Reinterpret(elementType, physicalShape.ToArray());
        for (var index = target.Axes.Count - 1; index >= 0; index--)
        {
            output = PackTensor(output, oldLanesCount++, target.Lanes[index], target.Axes[index]);
        }

        var outputType = context.CurrentCall.CheckedTensorType;
        var outputShape = context.Evaluate(context.CurrentCall.CheckedShape).AsTensor().ToArray<long>();
        return Value.FromTensor(output.Reinterpret(outputType.DType, outputShape));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Pack target)
    {
        var input = context.CheckArgumentType<IRType>(target, Pack.Input);

        return input switch
        {
            DistributedType d => Visit(context, target, d),
            TensorType t => Visit(context, target, t),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Pack target)
    {
        var inputType = context.GetArgumentType<IRType>(target, Pack.Input);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Pack target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
        };
    }

    private static Tensor PackTensor(Tensor input, int oldLanesCount, int lanes, int axis)
    {
        if (axis < 0)
        {
            return input;
        }

        var shape = input.Dimensions.ToArray();
        if (axis >= shape.Length - oldLanesCount)
        {
            throw new ArgumentOutOfRangeException(nameof(axis), "Pack axis must refer to a logical tensor dimension.");
        }

        if (shape[axis] % lanes != 0)
        {
            throw new InvalidOperationException($"Pack axis extent {shape[axis]} is not divisible by lane count {lanes}.");
        }

        var dividedShape = shape.Take(axis)
            .Concat(new[] { shape[axis] / lanes, lanes })
            .Concat(shape.Skip(axis + 1))
            .ToArray();
        var permutation = Enumerable.Range(0, axis + 1)
            .Concat(Enumerable.Range(axis + 2, dividedShape.Length - (axis + oldLanesCount + 2)))
            .Append(axis + 1)
            .Concat(Enumerable.Range(dividedShape.Length - oldLanesCount, oldLanesCount))
            .Select(index => (long)index)
            .ToArray();
        var reshaped = input.Reshape(dividedShape);
        return permutation.Select(index => (int)index).SequenceEqual(Enumerable.Range(0, dividedShape.Length))
            ? reshaped
            : reshaped.Transpose(permutation);
    }

    private IRType Visit(ITypeInferenceContext context, Pack target, TensorType input)
    {
        if (target.Lanes.Count != target.Axes.Count)
        {
            return new InvalidType("pack lanes and axes must have the same length");
        }

        if (target.Lanes.Any(lane => lane <= 0))
        {
            return new InvalidType("pack lane <= 0");
        }

        return TypeInference.PackType(input, target.Lanes, target.Axes);
    }

    private IRType Visit(ITypeInferenceContext context, Pack target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType)
        {
            throw new InvalidOperationException();
        }

        return TypeInference.PackType(input, target.Lanes, target.Axes);
    }
}
