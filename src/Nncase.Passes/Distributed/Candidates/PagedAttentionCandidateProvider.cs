// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.Passes.Distributed;

internal sealed class PagedAttentionCandidateProvider :
    DistributedCandidateProvider<PagedAttention>
{
    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PagedAttention target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        if (context.AvailableInputTypes.Count != 6)
        {
            return Array.Empty<IRType>();
        }

        var queryTypes = context.AvailableInputTypes[PagedAttention.Q.Index].ToHashSet();
        var gateTypes = context.AvailableInputTypes[PagedAttention.OutputGate.Index].ToHashSet();
        var hasOptionalGate = gateTypes.Any(type => type is NoneType);
        return defaultReturnTypes
            .Where(type => queryTypes.Contains(type) && (hasOptionalGate || gateTypes.Contains(type)))
            .Distinct()
            .ToArray();
    }

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PagedAttention target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (context.AvailableInputTypes.Count != 6)
        {
            return false;
        }

        if (!context.AvailableInputTypes[PagedAttention.Q.Index].Contains(returnType))
        {
            return true;
        }

        var gateTypes = context.AvailableInputTypes[PagedAttention.OutputGate.Index]
            .Where(type => type is NoneType || type == returnType)
            .ToArray();
        if (gateTypes.Length == 0)
        {
            return true;
        }

        var results = new List<DistributedCandidateTuple>();
        foreach (var kvCaches in context.AvailableInputTypes[PagedAttention.KVCaches.Index].OfType<TensorType>())
        {
            foreach (var extra in context.AvailableInputTypes[PagedAttention.Extra.Index])
            {
                foreach (var scale in context.AvailableInputTypes[PagedAttention.Scale.Index].OfType<TensorType>())
                {
                    foreach (var layerId in context.AvailableInputTypes[PagedAttention.LayerId.Index].OfType<DimensionType>())
                    {
                        foreach (var outputGate in gateTypes)
                        {
                            if (PagedAttentionEvaluator.InferType(
                                    target,
                                    returnType,
                                    extra,
                                    scale,
                                    kvCaches,
                                    outputGate) != returnType)
                            {
                                continue;
                            }

                            results.Add(new DistributedCandidateTuple(
                                [returnType, kvCaches, extra, scale, layerId, outputGate],
                                "paged-attention-output-sbp"));
                        }
                    }
                }
            }
        }

        tuples = results;
        return true;
    }
}
