// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1010, SA1008
using System;
using System.Diagnostics;
using System.Linq;
using System.Net.Http.Headers;
using System.Numerics;
using System.Runtime.InteropServices;
using CommunityToolkit.HighPerformance;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

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
        if (elementType == DataTypes.Float8E4M3 || elementType == DataTypes.Float8E5M2)
        {
            var inputCasted = input.CastElement<float>();
            var inputOrt = inputCasted.ToOrtTensor();
            inputOrt = inputOrt.Pack(oldLanesCount, target.Lanes, target.Axes);
            var output = inputOrt.ToTensor().CastElementTo(elementType);
            var outputType = context.CurrentCall.CheckedTensorType;
            var outputShape = context.Evaluate(context.CurrentCall.CheckedShape).AsTensor().ToArray<long>();
            output = output.CastTo(outputType.DType, CastMode.Reinterpret, outputShape);
            return Value.FromTensor(output);
        }
        else
        {
            var inputOrt = input.ToOrtTensor();
            inputOrt = inputOrt.Pack(oldLanesCount, target.Lanes, target.Axes);
            return inputOrt.ToValue(TypeInference.PackType(input.ElementType, target.Lanes));
        }
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
