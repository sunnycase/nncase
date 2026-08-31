// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator.IR.NTT;
using Nncase.IR;
using Nncase.IR.NTT;

namespace Nncase.Passes.Distributed;

internal sealed class PackedMatMulGluCombineCandidateProvider :
    DistributedCandidateProvider<PackedMatMulGluCombine>
{
    public override bool AllowsPartialInputs => true;

    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedMatMulGluCombine target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Where(type => HasSameTensorType(type, target.OutputType))
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedMatMulGluCombine target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = context.AvailableInputTypes.Count == 1
            ? context.AvailableInputTypes[PackedMatMulGluCombine.Projections.Index]
                .Where(input =>
                    PackedMatMulGluCombineEvaluator.InferType(input, returnType, target.GluType) == returnType)
                .Select(input => new DistributedCandidateTuple(
                    [input],
                    "packed-matmul-glu-combine-sbp"))
                .ToArray()
            : Array.Empty<DistributedCandidateTuple>();
        return true;
    }

    public override PackedMatMulGluCombine CreateCandidateTarget(
        DistributedCandidateContext context,
        PackedMatMulGluCombine target,
        IRType returnType)
        => new(returnType, target.GluType);

    private static bool HasSameTensorType(IRType candidate, IRType expected)
        => GetTensorType(candidate) == GetTensorType(expected);

    private static TensorType? GetTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => distributed.TensorType,
        _ => null,
    };
}
