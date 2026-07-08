// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1010, SA1008
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http.Headers;
using System.Numerics;
using System.Runtime.InteropServices;
using CommunityToolkit.HighPerformance;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

public sealed class PackEvaluator : ITypeInferencer<Pack>, ICostEvaluator<Pack>, IEvaluator<Pack>
{
    private static Dictionary<int, int> GetAxisLaneProducts(IRArray<int> lanes, IRArray<int> axes)
    {
        var products = new Dictionary<int, int>();
        for (int i = 0; i < axes.Count; i++)
        {
            products[axes[i]] = products.TryGetValue(axes[i], out var product)
                ? checked(product * lanes[i])
                : lanes[i];
        }

        return products;
    }

    private static bool TryDivideSplitGranularity(Dimension dim, SBPSplit split, Placement placement, int laneProduct, out Dimension? dividedGranularity)
    {
        dividedGranularity = null;
        if (split.Granularity is { } granularity)
        {
            if (!Dimension.TryDivExactly(granularity, laneProduct, out var divided))
            {
                return false;
            }

            dividedGranularity = divided;
            return true;
        }

        var divisor = split.Axes.Select(axis => placement.Hierarchy[axis]).Aggregate(1, (a, b) => a * b);
        if (!dim.IsFixed)
        {
            return false;
        }

        var localDim = (Dimension)MathUtility.CeilDiv(dim.FixedValue, divisor);
        return Dimension.TryDivExactly(localDim, laneProduct, out _);
    }

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
            var output = inputOrt.ToTensor().CastElementTo(context.CurrentCall.Arguments[Pack.Input.Index].CheckedDataType);
            output = output.CastTo(TypeInference.PackType(input.ElementType, target.Lanes), CastMode.Reinterpret);
            output = output.Squeeze(Enumerable.Range(output.Rank - target.Lanes.Count, target.Lanes.Count).Select(i => (long)i).ToArray());
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
        if (Visit(context, target, input.TensorType) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        var axisLaneProducts = GetAxisLaneProducts(target.Lanes, target.Axes);
        var ndsbp = new SBP[input.TensorType.Shape.Rank];
        for (int i = 0; i < input.TensorType.Shape.Rank; i++)
        {
            if (input.AxisPolicies[i] is SBPSplit split && axisLaneProducts.TryGetValue(i, out var laneProduct))
            {
                if (TryDivideSplitGranularity(input.TensorType.Shape[i], split, input.Placement, laneProduct, out var granularity))
                {
                    ndsbp[i] = SBP.S(split.Axes, granularity);
                }
                else
                {
                    return new InvalidType($"{input}, pack axis {i} split cuts vector lane group {laneProduct}");
                }
            }
            else
            {
                ndsbp[i] = input.AxisPolicies[i];
            }
        }

        return new DistributedType(tensorType, ndsbp, input.Placement);
    }
}
