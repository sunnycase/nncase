// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Distributed;

namespace Nncase.Evaluator.IR.Distributed;

public sealed class ShardedViewEvaluator : ITypeInferencer<ShardedView>, ICostEvaluator<ShardedView>
{
    public IRType Visit(ITypeInferenceContext context, ShardedView target)
    {
        var inputType = context.GetArgumentType(target, ShardedView.Input);
        if (inputType is not TensorType tensorType)
        {
            return new InvalidType($"ShardedView expects an unsharded tensor input, got {inputType}.");
        }

        if (tensorType != target.NewType.TensorType)
        {
            return new InvalidType($"ShardedView input type {tensorType} does not match view tensor type {target.NewType.TensorType}.");
        }

        if (target.NewType.AxisPolicies.Any(policy => policy is SBPPartial) || target.NewType.Partial is not null)
        {
            return new InvalidType("ShardedView does not support partial distributed types.");
        }

        return target.NewType;
    }

    public Cost Visit(ICostEvaluateContext context, ShardedView target)
    {
        return Cost.Zero;
    }
}
