// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.IR.NTT;
using Nncase.IR;
using Nncase.IR.NTT;

namespace Nncase.Passes.Distributed;

internal sealed class PagedAttentionPartialCandidateProvider :
    DistributedCandidateProvider<PagedAttentionPartial>
{
    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PagedAttentionPartial target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        var results = new List<IRType>();
        foreach (var inputTypes in EnumerateInputTypes(context))
        {
            var result = PagedAttentionPartialEvaluator.InferType(
                target,
                inputTypes[PagedAttentionPartial.Q.Index],
                (TensorType)inputTypes[PagedAttentionPartial.KVCaches.Index],
                inputTypes[PagedAttentionPartial.Extra.Index],
                (TensorType)inputTypes[PagedAttentionPartial.Scale.Index],
                (DimensionType)inputTypes[PagedAttentionPartial.LayerId.Index]);
            if (result is not InvalidType)
            {
                results.Add(result);
            }
        }

        return results.Distinct().ToArray();
    }

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PagedAttentionPartial target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        var results = new List<DistributedCandidateTuple>();
        foreach (var inputTypes in EnumerateInputTypes(context))
        {
            if (PagedAttentionPartialEvaluator.InferType(
                    target,
                    inputTypes[PagedAttentionPartial.Q.Index],
                    (TensorType)inputTypes[PagedAttentionPartial.KVCaches.Index],
                    inputTypes[PagedAttentionPartial.Extra.Index],
                    (TensorType)inputTypes[PagedAttentionPartial.Scale.Index],
                    (DimensionType)inputTypes[PagedAttentionPartial.LayerId.Index]) == returnType)
            {
                results.Add(new DistributedCandidateTuple(inputTypes, "paged-attention-partial-sbp"));
            }
        }

        tuples = results;
        return true;
    }

    private static IEnumerable<IRType[]> EnumerateInputTypes(DistributedCandidateContext context)
    {
        if (context.AvailableInputTypes.Count != 5)
        {
            yield break;
        }

        foreach (var query in context.AvailableInputTypes[PagedAttentionPartial.Q.Index])
        {
            foreach (var kvCaches in context.AvailableInputTypes[PagedAttentionPartial.KVCaches.Index].OfType<TensorType>())
            {
                foreach (var extra in context.AvailableInputTypes[PagedAttentionPartial.Extra.Index])
                {
                    foreach (var scale in context.AvailableInputTypes[PagedAttentionPartial.Scale.Index].OfType<TensorType>())
                    {
                        foreach (var layerId in context.AvailableInputTypes[PagedAttentionPartial.LayerId.Index].OfType<DimensionType>())
                        {
                            yield return [query, kvCaches, extra, scale, layerId];
                        }
                    }
                }
            }
        }
    }
}

internal sealed class PagedAttentionCombineCandidateProvider :
    DistributedCandidateProvider<PagedAttentionCombine>
{
    public override bool AllowsPartialInputs => true;

    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PagedAttentionCombine target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Where(returnType => IsLogicalOutputType(target, returnType))
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PagedAttentionCombine target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        var results = new List<DistributedCandidateTuple>();
        var candidateTarget = WithOutputType(target, returnType);
        foreach (var inputTypes in EnumerateInputTypes(context))
        {
            if (PagedAttentionCombineEvaluator.InferType(
                    candidateTarget,
                    inputTypes[PagedAttentionCombine.MaxState.Index],
                    inputTypes[PagedAttentionCombine.SumState.Index],
                    inputTypes[PagedAttentionCombine.AccState.Index]) == returnType)
            {
                results.Add(new DistributedCandidateTuple(inputTypes, "paged-attention-combine-sbp"));
            }
        }

        tuples = results;
        return true;
    }

    public override PagedAttentionCombine CreateCandidateTarget(
        DistributedCandidateContext context,
        PagedAttentionCombine target,
        IRType returnType)
        => WithOutputType(target, returnType);

    private static IEnumerable<IRType[]> EnumerateInputTypes(DistributedCandidateContext context)
    {
        if (context.AvailableInputTypes.Count != 3)
        {
            yield break;
        }

        foreach (var maxState in context.AvailableInputTypes[PagedAttentionCombine.MaxState.Index])
        {
            foreach (var sumState in context.AvailableInputTypes[PagedAttentionCombine.SumState.Index])
            {
                foreach (var accState in context.AvailableInputTypes[PagedAttentionCombine.AccState.Index])
                {
                    yield return [maxState, sumState, accState];
                }
            }
        }
    }

    private static PagedAttentionCombine WithOutputType(
        PagedAttentionCombine target,
        IRType outputType)
        => new(
            target.Layout,
            target.HiddenSize,
            target.OutputDataType,
            outputType,
            target.SplitHierarchyAxis,
            target.SplitCount);

    private static bool IsLogicalOutputType(PagedAttentionCombine target, IRType returnType)
    {
        var tensorType = PagedAttentionSplitTypeUtility.GetTensorType(returnType);
        var expectedType = PagedAttentionSplitTypeUtility.GetTensorType(target.OutputType);
        return tensorType is not null &&
            expectedType is not null &&
            tensorType == expectedType &&
            returnType is not DistributedType { Partial: not null };
    }
}
