// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.Evaluator.IR.NTT;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;

namespace Nncase.Passes.Distributed;

/// <summary>
/// Couples the packed Q/K/V reduction layouts so AutoDistributed can evaluate
/// mixed split-K/split-N plans instead of requiring independently discovered
/// weight candidates to align by chance.
/// </summary>
internal sealed class PackedQKVParallelLinearCandidateProvider :
    DistributedCandidateProvider<PackedQKVParallelLinear>
{
    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedQKVParallelLinear target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Concat(defaultReturnTypes
                .Select(TryMaterializeOutput)
                .Where(output => output is not null)
                .SelectMany(output => Enumerate(context, target, output!))
                .Select(candidate => candidate.OutputType))
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedQKVParallelLinear target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        var materializedOutput = TryMaterializeOutput(returnType);
        tuples = materializedOutput is null
            ? Array.Empty<DistributedCandidateTuple>()
            : Enumerate(context, target, materializedOutput)
                .Where(candidate => candidate.OutputType == returnType)
                .Select(candidate => new DistributedCandidateTuple(
                    candidate.Arguments,
                    "packed-qkv-reduction-sbp"))
                .Distinct()
                .ToArray();
        return true;
    }

    private static IEnumerable<PackedQKVCandidate> Enumerate(
        DistributedCandidateContext context,
        PackedQKVParallelLinear target,
        TupleType materializedOutput)
    {
        if (context.AvailableInputTypes.Count != 13 ||
            materializedOutput.Fields.ToArray() is not { Length: 3 } outputFields ||
            outputFields.Any(field => field is not DistributedType { Partial: null }))
        {
            yield break;
        }

        foreach (var input in context.AvailableInputTypes[PackedQKVParallelLinear.Input.Index]
                     .OfType<DistributedType>()
                     .Where(type => type.Partial is null))
        {
            var qWeights = AlignWeights(
                context,
                target,
                input,
                (DistributedType)outputFields[0],
                PackedQKVParallelLinear.QWeight.Index);
            var kWeights = AlignWeights(
                context,
                target,
                input,
                (DistributedType)outputFields[1],
                PackedQKVParallelLinear.KWeight.Index);
            var vWeights = AlignWeights(
                context,
                target,
                input,
                (DistributedType)outputFields[2],
                PackedQKVParallelLinear.VWeight.Index);
            foreach (var weights in new[] { qWeights, kWeights, vWeights }.CartesianProduct())
            {
                var weightArray = weights.ToArray();
                foreach (var tail in context.AvailableInputTypes.Skip(4).CartesianProduct())
                {
                    var tailArray = tail.ToArray();
                    IRType[] arguments = [input, .. weightArray, .. tailArray];
                    var outputType = PackedQKVParallelLinearEvaluator.InferType(
                        target,
                        arguments[0],
                        arguments[1],
                        arguments[2],
                        arguments[3],
                        arguments[4],
                        arguments[5],
                        arguments[6],
                        arguments[7],
                        arguments[8],
                        arguments[9],
                        arguments[10],
                        arguments[11],
                        arguments[12]);
                    if (IsCoupledOutput(outputType) &&
                        PackedQKVParallelLinearCombineEvaluator.InferType(
                            outputType,
                            materializedOutput) == materializedOutput &&
                        (!HasPartialOutput(outputType) ||
                         arguments.Skip(4).Take(3).All(argument => argument is NoneType)))
                    {
                        yield return new PackedQKVCandidate(arguments, outputType);
                    }
                }
            }
        }
    }

    private static bool IsCoupledOutput(IRType outputType)
    {
        if (outputType is not TupleType { Count: 3 } tuple ||
            tuple.Fields.Any(field => field is not DistributedType))
        {
            return false;
        }

        var outputs = tuple.Fields.Cast<DistributedType>().ToArray();
        var partial = outputs[0].Partial;
        if (partial is not null && partial.Op != ReduceOp.Sum)
        {
            return false;
        }

        var outputNPolicy = outputs[0].AxisPolicies[^1];
        return outputs.All(output =>
            output.Placement == outputs[0].Placement &&
            HasSamePartial(output.Partial, partial) &&
            HasCoupledOutputPolicy(output.AxisPolicies[^1], outputNPolicy));
    }

    private static bool HasCoupledOutputPolicy(SBP lhs, SBP rhs)
        => (lhs, rhs) switch
        {
            (SBPBroadCast, SBPBroadCast) => true,
            (SBPSplit left, SBPSplit right) =>
                left.Stages.Count == right.Stages.Count &&
                left.Stages.Zip(right.Stages).All(stages =>
                    stages.First.HierarchyAxes.SequenceEqual(stages.Second.HierarchyAxes) &&
                    HasCoupledDistribution(
                        stages.First.Distribution,
                        stages.Second.Distribution)),
            _ => false,
        };

    private static bool HasCoupledDistribution(
        SplitDistribution lhs,
        SplitDistribution rhs)
        => (lhs, rhs) switch
        {
            (BlockCyclicSplit, BlockCyclicSplit) => true,
            (ContiguousSplit left, ContiguousSplit right) => left == right,
            _ => false,
        };

    private static bool HasPartialOutput(IRType outputType)
        => outputType is TupleType tuple &&
            tuple.Fields.OfType<DistributedType>().Any(output => output.Partial is not null);

    private static bool HasSamePartial(SBPPartial? lhs, SBPPartial? rhs)
        => (lhs, rhs) switch
        {
            (null, null) => true,
            ({ } left, { } right) => left.Op == right.Op &&
                left.Axes.SequenceEqual(right.Axes),
            _ => false,
        };

    private static IEnumerable<DistributedType> AlignWeights(
        DistributedCandidateContext context,
        PackedQKVParallelLinear target,
        DistributedType input,
        DistributedType materializedOutput,
        int argumentIndex)
        => context.AvailableInputTypes[argumentIndex]
            .OfType<DistributedType>()
            .Where(type => type.Partial is null)
            .Select(weight => TryAlignMatMulPolicies(target, input, weight, materializedOutput))
            .Where(weight => weight is not null)
            .Select(weight => weight!)
            .Distinct();

    private static DistributedType? TryAlignMatMulPolicies(
        PackedQKVParallelLinear target,
        DistributedType input,
        DistributedType weight,
        DistributedType materializedOutput)
    {
        if (input.Placement != weight.Placement ||
            input.Placement != materializedOutput.Placement ||
            weight.TensorType.DType is not VectorType vectorType ||
            !PackedQKVParallelLinearEvaluator.TryGetLayoutInfo(
                target.RhsLayout,
                vectorType,
                weight.TensorType.Shape.Rank,
                out var unpackAxes,
                out var outputLanes,
                out var transposeB,
                out _) ||
            TypeInference.UnpackType(weight, unpackAxes) is not DistributedType logicalWeight ||
            materializedOutput.TensorType.DType is not VectorType outputVectorType ||
            !outputVectorType.Lanes.SequenceEqual(outputLanes) ||
            TypeInference.UnpackType(
                materializedOutput,
                Enumerable.Repeat(
                    materializedOutput.TensorType.Shape.Rank - 1,
                    outputLanes.Length).ToArray()) is not DistributedType logicalOutput)
        {
            return null;
        }

        var dimInfo = VectorizedMatMul.GetDimInfo(
            false,
            transposeB,
            input.TensorType.Shape.Rank,
            logicalWeight.TensorType.Shape.Rank);
        var policies = logicalWeight.AxisPolicies.ToArray();
        policies[dimInfo.Rk] = input.AxisPolicies[dimInfo.Lk];
        policies[dimInfo.Rn] = logicalOutput.AxisPolicies[^1];
        if (!DistributedUtility.IsDistributable(
                logicalWeight.TensorType,
                policies,
                input.Placement))
        {
            return null;
        }

        var alignedLogicalWeight = new DistributedType(
            logicalWeight.TensorType,
            policies,
            input.Placement);
        return TypeInference.PackType(
            alignedLogicalWeight,
            vectorType.Lanes.ToArray(),
            unpackAxes) is DistributedType packedWeight &&
            packedWeight.TensorType == weight.TensorType
                ? packedWeight
                : null;
    }

    private static TupleType? TryMaterializeOutput(IRType outputType)
    {
        if (outputType is not TupleType { Count: 3 } tuple ||
            tuple.Fields.Any(field => field is not DistributedType))
        {
            return null;
        }

        return new TupleType(tuple.Fields
            .Cast<DistributedType>()
            .Select(field => (IRType)new DistributedType(
                field.TensorType,
                field.AxisPolicies,
                field.Placement))
            .ToArray());
    }

    private sealed record PackedQKVCandidate(
        IReadOnlyList<IRType> Arguments,
        IRType OutputType);
}
