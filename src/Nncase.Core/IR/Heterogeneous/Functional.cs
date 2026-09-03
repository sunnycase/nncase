// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Heterogeneous;

namespace Nncase.IR.F;

public static class Heterogeneous
{
    public static Call CreatePipelineChannel(
        string channelId,
        string producerModuleKind,
        string consumerModuleKind,
        IRType payloadType,
        int capacity = 1)
        => new(new IR.Heterogeneous.CreatePipelineChannel(
            channelId,
            producerModuleKind,
            consumerModuleKind,
            payloadType,
            capacity));

    public static Call Produce(Expr channel, Expr value, Expr dependency, string channelId, int phase)
        => new(new IR.Heterogeneous.Produce(channelId, phase), channel, value, dependency);

    public static Call Consume(Expr channel, Expr dependency, string channelId, int phase, IRType payloadType)
        => new(new IR.Heterogeneous.Consume(channelId, phase, payloadType), channel, dependency);

    public static Call PipelineToken(Expr value)
        => new(new IR.Heterogeneous.PipelineToken(), value);

    public static Call PipelineYield(BaseExpr value, Expr dependency)
        => new(new IR.Heterogeneous.PipelineYield(), value, dependency);

    public static Call PipelineLaunch(IR.Tuple workers, int resultWorkerIndex)
        => new(new IR.Heterogeneous.PipelineLaunch(resultWorkerIndex), workers);
}
