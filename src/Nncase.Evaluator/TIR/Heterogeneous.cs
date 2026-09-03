// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Evaluator.TIR;

public sealed class ChannelProduceEvaluator : ITypeInferencer<ChannelProduce>
{
    public IRType Visit(ITypeInferenceContext context, ChannelProduce target)
    {
        _ = context.GetArgumentType(target, ChannelProduce.Channel);
        _ = context.GetArgumentType(target, ChannelProduce.Value);
        return TupleType.Void;
    }
}

public sealed class ChannelConsumeEvaluator : ITypeInferencer<ChannelConsume>
{
    public IRType Visit(ITypeInferenceContext context, ChannelConsume target)
    {
        _ = context.GetArgumentType(target, ChannelConsume.Channel);
        _ = context.GetArgumentType(target, ChannelConsume.Destination);
        return TupleType.Void;
    }
}
