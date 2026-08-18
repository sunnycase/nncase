// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.IR.NTT;
using Nncase.IR;
using Nncase.IR.NTT;

namespace Nncase.Passes.Distributed;

internal sealed class SamplingPartialCandidateProvider :
    DistributedCandidateProvider<SamplingPartial>
{
    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        SamplingPartial target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => Enumerate(context, target)
            .Select(candidate => candidate.ReturnType)
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        SamplingPartial target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Enumerate(context, target)
            .Where(candidate => candidate.ReturnType == returnType)
            .Select(candidate => new DistributedCandidateTuple(
                [candidate.LogitsType, candidate.StateType],
                "sampling-partial-vocabulary-sbp"))
            .Distinct()
            .ToArray();
        return true;
    }

    private static IEnumerable<SamplingPartialCandidate> Enumerate(
        DistributedCandidateContext context,
        SamplingPartial target)
    {
        if (context.AvailableInputTypes.Count != 2)
        {
            yield break;
        }

        foreach (var stateType in context.AvailableInputTypes[SamplingPartial.State.Index].OfType<TensorType>())
        {
            foreach (var logitsType in context.AvailableInputTypes[SamplingPartial.Logits.Index])
            {
                var returnType = SamplingPartialEvaluator.InferType(target, logitsType, stateType);
                if (returnType is not InvalidType)
                {
                    yield return new SamplingPartialCandidate(logitsType, stateType, returnType);
                }
            }
        }
    }

    private sealed record SamplingPartialCandidate(
        IRType LogitsType,
        TensorType StateType,
        IRType ReturnType);
}

internal sealed class SamplingCombineCandidateProvider :
    DistributedCandidateProvider<SamplingCombine>
{
    public override bool AllowsPartialInputs => true;

    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        SamplingCombine target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => Enumerate(context, target)
            .Select(candidate => candidate.ReturnType)
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        SamplingCombine target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Enumerate(context, target)
            .Where(candidate => candidate.ReturnType == returnType)
            .Select(candidate => new DistributedCandidateTuple(
                candidate.InputTypes,
                "sampling-combine-vocabulary-sbp"))
            .Distinct()
            .ToArray();
        return true;
    }

    private static IEnumerable<SamplingCombineCandidate> Enumerate(
        DistributedCandidateContext context,
        SamplingCombine target)
    {
        if (context.AvailableInputTypes.Count != 4)
        {
            yield break;
        }

        foreach (var stateType in context.AvailableInputTypes[SamplingCombine.State.Index].OfType<TensorType>())
        {
            foreach (var logitsType in context.AvailableInputTypes[SamplingCombine.Logits.Index])
            {
                foreach (var processedType in context.AvailableInputTypes[SamplingCombine.ProcessedLogits.Index])
                {
                    foreach (var argMaxStateType in context.AvailableInputTypes[SamplingCombine.ArgMaxState.Index])
                    {
                        var returnType = SamplingCombineEvaluator.InferType(
                            target,
                            logitsType,
                            processedType,
                            argMaxStateType,
                            stateType);
                        if (returnType is not InvalidType)
                        {
                            yield return new SamplingCombineCandidate(
                                [logitsType, processedType, argMaxStateType, stateType],
                                returnType);
                        }
                    }
                }
            }
        }
    }

    private sealed record SamplingCombineCandidate(
        IRType[] InputTypes,
        IRType ReturnType);
}
