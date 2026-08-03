// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.Utilities;

namespace Nncase.Evaluator.IR.Distributed;

public sealed class ShardedViewEvaluator : ITypeInferencer<ShardedView>, ICostEvaluator<ShardedView>
{
    public IRType Visit(ITypeInferenceContext context, ShardedView target)
    {
        var inputType = context.GetArgumentType(target, ShardedView.Input);
        if (!DistributedUtility.TryValidateShardedView(inputType, target.NewType, out var reason))
        {
            return new InvalidType(reason);
        }

        return target.NewType;
    }

    public Cost Visit(ICostEvaluateContext context, ShardedView target)
    {
        var inputType = context.GetArgumentType<IRType>(target, ShardedView.Input);
        return inputType is DistributedType sourceType &&
               !DistributedUtility.IsLocalShardSubview(sourceType, target.NewType)
                ? new Cost { [CostFactorNames.GridSynchronization] = 1 }
                : Cost.Zero;
    }
}
