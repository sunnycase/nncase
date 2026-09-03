// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Heterogeneous;

namespace Nncase.Passes.Distributed;

internal sealed class PipelineChannelConsumeCandidateProvider : DistributedCandidateProvider<Consume>
{
    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        Consume target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Where(type => IsMaterializedPayloadType(type, target.PayloadType))
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        Consume target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (context.AvailableInputTypes.Count != 2 ||
            !IsMaterializedPayloadType(returnType, target.PayloadType))
        {
            return false;
        }

        tuples = context.AvailableInputTypes[Consume.Channel.Index]
            .SelectMany(
                channelType => context.AvailableInputTypes[Consume.Dependency.Index],
                (channelType, dependencyType) => new DistributedCandidateTuple(
                    [channelType, dependencyType],
                    "pipeline-channel-consume-output-sbp"))
            .ToArray();
        return true;
    }

    public override Consume CreateCandidateTarget(
        DistributedCandidateContext context,
        Consume target,
        IRType returnType)
        => new(target.ChannelId, target.Phase, returnType);

    private static bool IsMaterializedPayloadType(IRType candidate, IRType payloadType)
    {
        var payloadTensorType = payloadType switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => null,
        };
        return candidate switch
        {
            TensorType tensorType => tensorType == payloadTensorType,
            DistributedType distributedType =>
                distributedType.TensorType == payloadTensorType &&
                distributedType.Partial is null &&
                distributedType.AxisPolicies.All(policy => policy is not SBPPartial),
            _ => false,
        };
    }
}
