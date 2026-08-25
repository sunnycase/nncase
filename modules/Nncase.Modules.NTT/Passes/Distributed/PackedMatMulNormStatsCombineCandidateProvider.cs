// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator.IR.NTT;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;

namespace Nncase.Passes.Distributed;

internal sealed class PackedMatMulNormStatsCombineCandidateProvider :
    DistributedCandidateProvider<PackedMatMulNormStatsCombine>
{
    public override bool AllowsPartialInputs => true;

    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedMatMulNormStatsCombine target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Where(type => HasSameTensorTypes(type, target.OutputType))
            .Concat(EnumerateOutputTypes(context, target))
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedMatMulNormStatsCombine target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        if (context.AvailableInputTypes.Count != 2 ||
            returnType is not TupleType { Count: 2 } output)
        {
            tuples = Array.Empty<DistributedCandidateTuple>();
            return true;
        }

        var valueOutput = output[0];
        tuples = context.AvailableInputTypes[PackedMatMulNormStatsCombine.Input.Index]
            .Where(input => PackedMatMulNormStatsCombineEvaluator.CanMaterialize(input, valueOutput))
            .Select(input => new DistributedCandidateTuple(
                [input, valueOutput],
                "packed-matmul-norm-stats-combine-sbp"))
            .Where(tuple => PackedMatMulNormStatsCombineEvaluator.InferType(
                target,
                tuple.InputTypes[0],
                tuple.InputTypes[1],
                returnType) == returnType)
            .ToArray();
        return true;
    }

    public override PackedMatMulNormStatsCombine CreateCandidateTarget(
        DistributedCandidateContext context,
        PackedMatMulNormStatsCombine target,
        IRType returnType)
        => new(returnType, target.Axis, target.UseMean);

    private static IEnumerable<IRType> EnumerateOutputTypes(
        DistributedCandidateContext context,
        PackedMatMulNormStatsCombine target)
    {
        if (context.AvailableInputTypes.Count != 2)
        {
            yield break;
        }

        foreach (var addend in context.AvailableInputTypes[PackedMatMulNormStatsCombine.Addend.Index])
        {
            var stats = NormStatsEvaluator.InferType(
                new NormStats(target.Axis, target.UseMean),
                addend);
            if (stats is InvalidType)
            {
                continue;
            }

            var output = new TupleType([addend, stats]);
            if (!HasSameTensorTypes(output, target.OutputType))
            {
                continue;
            }

            if (context.AvailableInputTypes[PackedMatMulNormStatsCombine.Input.Index]
                .Any(input =>
                    PackedMatMulNormStatsCombineEvaluator.CanMaterialize(input, addend) &&
                    PackedMatMulNormStatsCombineEvaluator.InferType(
                        target,
                        input,
                        addend,
                        output) == output))
            {
                yield return output;
            }
        }
    }

    private static bool HasSameTensorTypes(IRType candidate, IRType expected)
    {
        if (candidate is not TupleType { Count: 2 } candidateTuple ||
            expected is not TupleType { Count: 2 } expectedTuple)
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
