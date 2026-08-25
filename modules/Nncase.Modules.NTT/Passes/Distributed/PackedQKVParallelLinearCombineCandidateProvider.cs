// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator.IR.NTT;
using Nncase.IR;
using Nncase.IR.NTT;

namespace Nncase.Passes.Distributed;

internal sealed class PackedQKVParallelLinearCombineCandidateProvider :
    DistributedCandidateProvider<PackedQKVParallelLinearCombine>
{
    public override bool AllowsPartialInputs => true;

    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedQKVParallelLinearCombine target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Where(type => HasSameTensorTypes(type, target.OutputType))
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedQKVParallelLinearCombine target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = context.AvailableInputTypes.Count == 1
            ? context.AvailableInputTypes[PackedQKVParallelLinearCombine.QKV.Index]
                .Where(input => PackedQKVParallelLinearCombineEvaluator.InferType(input, returnType) == returnType)
                .Select(input => new DistributedCandidateTuple(
                    [input],
                    "packed-qkv-combine-sbp"))
                .ToArray()
            : Array.Empty<DistributedCandidateTuple>();
        return true;
    }

    public override PackedQKVParallelLinearCombine CreateCandidateTarget(
        DistributedCandidateContext context,
        PackedQKVParallelLinearCombine target,
        IRType returnType)
        => new(returnType);

    private static bool HasSameTensorTypes(IRType candidate, IRType expected)
    {
        if (candidate is not TupleType candidateTuple || expected is not TupleType expectedTuple ||
            candidateTuple.Count != expectedTuple.Count)
        {
            return false;
        }

        return candidateTuple.Fields.Zip(expectedTuple.Fields).All(pair =>
            GetTensorType(pair.First) == GetTensorType(pair.Second));
    }

    private static TensorType? GetTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => distributed.TensorType,
        _ => null,
    };
}
