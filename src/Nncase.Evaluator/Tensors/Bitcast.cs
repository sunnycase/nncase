// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Bitcast"/>.
/// </summary>
public class BitcastEvaluator : IEvaluator<Bitcast>, ITypeInferencer<Bitcast>, ICostEvaluator<Bitcast>, IMetricEvaluator<Bitcast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Bitcast cast)
    {
        var input = context.GetArgumentValue(cast, Bitcast.Input).AsTensor();
        return Value.FromTensor(input.CastTo(cast.NewType, CastMode.Reinterpret));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Bitcast target)
    {
        var input = context.CheckArgumentType<IRType>(target, Bitcast.Input);
        return BitcastUtility.InferType(input, target.NewType);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Bitcast target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Bitcast target)
    {
        return new();
    }
}
