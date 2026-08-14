// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.Passes.Distributed;

internal sealed class QKVRoPEWithCacheCandidateProvider : DistributedCandidateProvider<QKVRoPEWithCache>
{
    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        QKVRoPEWithCache target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        var results = new List<DistributedCandidateTuple>();
        tuples = results;
        if (returnType is not TupleType { Count: 2 } || context.AvailableInputTypes.Count != 9)
        {
            return false;
        }

        foreach (var qkv in context.AvailableInputTypes[QKVRoPEWithCache.QKV.Index].OfType<TupleType>())
        {
            if (qkv.Count != 3)
            {
                continue;
            }

            var qNormArguments = GetNormArguments(
                qkv[0],
                context.AvailableInputTypes[QKVRoPEWithCache.QScale.Index],
                context.AvailableInputTypes[QKVRoPEWithCache.QBias.Index],
                target.QAxis,
                target.QEpsilon,
                target.QUseMean);
            var kNormArguments = GetNormArguments(
                qkv[1],
                context.AvailableInputTypes[QKVRoPEWithCache.KScale.Index],
                context.AvailableInputTypes[QKVRoPEWithCache.KBias.Index],
                target.KAxis,
                target.KEpsilon,
                target.KUseMean);
            var ropeArguments = GetRoPEArguments(
                qkv[0],
                qkv[1],
                context.AvailableInputTypes[QKVRoPEWithCache.Cos.Index],
                context.AvailableInputTypes[QKVRoPEWithCache.Sin.Index]);

            foreach (var qNorm in qNormArguments)
            {
                foreach (var kNorm in kNormArguments)
                {
                    foreach (var rope in ropeArguments)
                    {
                        foreach (var kvCaches in context.AvailableInputTypes[QKVRoPEWithCache.KVCaches.Index])
                        {
                            foreach (var layerId in context.AvailableInputTypes[QKVRoPEWithCache.LayerId.Index])
                            {
                                var inferred = QKVRoPEWithCacheEvaluator.InferType(
                                    target,
                                    qkv,
                                    qNorm.Scale,
                                    kNorm.Scale,
                                    qNorm.Bias,
                                    kNorm.Bias,
                                    rope.Cos,
                                    rope.Sin,
                                    kvCaches);
                                if (inferred != returnType)
                                {
                                    continue;
                                }

                                results.Add(new DistributedCandidateTuple(
                                    [
                                        qkv,
                                        qNorm.Scale,
                                        kNorm.Scale,
                                        qNorm.Bias,
                                        kNorm.Bias,
                                        rope.Cos,
                                        rope.Sin,
                                        kvCaches,
                                        layerId,
                                    ],
                                    "qkv-rope-cache-output-sbp"));
                            }
                        }
                    }
                }
            }
        }

        return true;
    }

    private static IReadOnlyList<NormArguments> GetNormArguments(
        IRType input,
        IReadOnlyList<IRType> scaleCandidates,
        IReadOnlyList<IRType> biasCandidates,
        int axis,
        float epsilon,
        bool useMean)
    {
        var target = new NormApply(axis, epsilon, useMean);
        var stats = NormStatsEvaluator.InferType(new NormStats(axis, useMean), input);
        var results = new List<NormArguments>();
        if (stats is InvalidType)
        {
            return results;
        }

        foreach (var scale in scaleCandidates)
        {
            foreach (var bias in biasCandidates)
            {
                if (NormApplyEvaluator.InferType(target, input, stats, scale, bias) == input)
                {
                    results.Add(new NormArguments(scale, bias));
                }
            }
        }

        return results;
    }

    private static IReadOnlyList<RoPEArguments> GetRoPEArguments(
        IRType q,
        IRType k,
        IReadOnlyList<IRType> cosCandidates,
        IReadOnlyList<IRType> sinCandidates)
    {
        var results = new List<RoPEArguments>();
        foreach (var cos in cosCandidates)
        {
            foreach (var sin in sinCandidates)
            {
                if (RoPEEvaluator.InferType(q, cos, sin) == q &&
                    RoPEEvaluator.InferType(k, cos, sin) == k)
                {
                    results.Add(new RoPEArguments(cos, sin));
                }
            }
        }

        return results;
    }

    private sealed record NormArguments(IRType Scale, IRType Bias);

    private sealed record RoPEArguments(IRType Cos, IRType Sin);
}
