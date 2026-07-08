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

public sealed class VectorizeMaskEvaluator : ITypeInferencer<VectorizeMask>, ICostEvaluator<VectorizeMask>, IEvaluator<VectorizeMask>
{
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
    public IValue Visit(IEvaluateContext context, VectorizeMask target)
    {
        var input = context.GetOrtArgumentValue(target, VectorizeMask.Input);
        input = input.Pack(0, target.Lanes, target.Axis);
        return input.ToValue(new MaskVectorType(target.Style, target.ElementBits, target.Lanes));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, VectorizeMask target)
    {
        var input = context.CheckArgumentType<IRType>(target, VectorizeMask.Input);

        return input switch
        {
            DistributedType d => Visit(context, target, d),
            TensorType t => Visit(context, target, t),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, VectorizeMask target)
    {
        var inputType = context.GetArgumentType<IRType>(target, VectorizeMask.Input);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, VectorizeMask target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
        };
    }

    private IRType Visit(ITypeInferenceContext context, VectorizeMask target, TensorType input)
    {
        if (target.Lanes <= 0)
        {
            return new InvalidType("vectorize mask lane <= 0");
        }

        return TypeInference.VectorizeMaskType(input, target.Style, target.ElementBits, target.Lanes, target.Axis);
    }

    private IRType Visit(ITypeInferenceContext context, VectorizeMask target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        var ndsbp = new SBP[input.TensorType.Shape.Rank];
        for (int i = 0; i < input.TensorType.Shape.Rank; i++)
        {
            if (input.AxisPolicies[i] is SBPSplit split && target.Axis == i)
            {
                if (TryDivideSplitGranularity(input.TensorType.Shape[i], split, input.Placement, target.Lanes, out var granularity))
                {
                    ndsbp[i] = SBP.S(split.Axes, granularity);
                }
                else
                {
                    return new InvalidType($"{input}, vectorize mask axis {i} split cuts vector lane group {target.Lanes}");
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
