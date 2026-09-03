// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator;
using Nncase.Evaluator.IR.NTT;
using Nncase.Evaluator.Math;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NTT;
using Nncase.Utilities;

namespace Nncase.Passes.Distributed;

/// <summary>
/// Derives packed Q/K/V projection contracts from the requested logical output
/// and one common input layout.
/// </summary>
internal sealed class PackedQKVParallelLinearCandidateProvider :
    DistributedCandidateProvider<PackedQKVParallelLinear>
{
    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedQKVParallelLinear target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        var directCandidates = EnumerateDirect(context, target).ToArray();
        return defaultReturnTypes
            .Concat(directCandidates.Select(candidate => candidate.Output))
            .Select(TryMaterializeOutput)
            .Where(output => output is not null)
            .SelectMany(output => EnumeratePartial(context, target, output!))
            .Select(candidate => candidate.Output)
            .Concat(defaultReturnTypes)
            .Concat(directCandidates.Select(candidate => candidate.Output))
            .Distinct()
            .ToArray();
    }

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedQKVParallelLinear target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        var direct = EnumerateDirect(context, target)
            .Where(candidate => candidate.Output == returnType)
            .Select(candidate => new DistributedCandidateTuple(
                [
                    candidate.Input,
                    candidate.Q.Weight,
                    candidate.K.Weight,
                    candidate.V.Weight,
                    candidate.Q.Bias,
                    candidate.K.Bias,
                    candidate.V.Bias,
                    candidate.Q.InputScale,
                    candidate.K.InputScale,
                    candidate.V.InputScale,
                    candidate.Q.WeightScale,
                    candidate.K.WeightScale,
                    candidate.V.WeightScale,
                ],
                "packed-qkv-parallel-linear-output-sbp"))
            .ToArray();
        var materializedOutput = TryMaterializeOutput(returnType);
        var partial = materializedOutput is null
            ? Array.Empty<DistributedCandidateTuple>()
            : EnumeratePartial(context, target, materializedOutput)
                .Where(candidate => candidate.Output == returnType)
                .Select(candidate => new DistributedCandidateTuple(
                    candidate.Arguments,
                    "packed-qkv-reduction-sbp"))
                .ToArray();
        tuples = direct.Concat(partial).Distinct().ToArray();
        return true;
    }

    private static IEnumerable<PackedQKVCandidate> EnumerateDirect(
        DistributedCandidateContext context,
        PackedQKVParallelLinear target)
    {
        if (context.AvailableInputTypes.Count != 13 ||
            context.SourceCall.CheckedType is not TupleType { Count: 3 } outputTuple ||
            !TryGetSourceTensorType(context, PackedQKVParallelLinear.Input.Index, out var inputTensor) ||
            !TryGetTensorType(outputTuple.Fields[0], out var qOutputTensor) ||
            !TryGetTensorType(outputTuple.Fields[1], out var kOutputTensor) ||
            !TryGetTensorType(outputTuple.Fields[2], out var vOutputTensor))
        {
            yield break;
        }

        foreach (var placement in GetPlacements(context))
        {
            foreach (var input in context.GetLeafCandidateTypes(inputTensor, [placement])
                         .Where(type => type.Partial is null))
            {
                var qCandidates = EnumerateProjection(
                    context,
                    target,
                    "q",
                    input,
                    qOutputTensor,
                    PackedQKVParallelLinear.QWeight.Index,
                    PackedQKVParallelLinear.QBias.Index,
                    PackedQKVParallelLinear.QInputScale.Index,
                    PackedQKVParallelLinear.QWeightScale.Index).ToArray();
                var kCandidates = EnumerateProjection(
                    context,
                    target,
                    "k",
                    input,
                    kOutputTensor,
                    PackedQKVParallelLinear.KWeight.Index,
                    PackedQKVParallelLinear.KBias.Index,
                    PackedQKVParallelLinear.KInputScale.Index,
                    PackedQKVParallelLinear.KWeightScale.Index).ToArray();
                var vCandidates = EnumerateProjection(
                    context,
                    target,
                    "v",
                    input,
                    vOutputTensor,
                    PackedQKVParallelLinear.VWeight.Index,
                    PackedQKVParallelLinear.VBias.Index,
                    PackedQKVParallelLinear.VInputScale.Index,
                    PackedQKVParallelLinear.VWeightScale.Index).ToArray();

                foreach (var q in qCandidates)
                {
                    foreach (var k in kCandidates)
                    {
                        foreach (var v in vCandidates)
                        {
                            yield return new(
                                input,
                                q,
                                k,
                                v,
                                new TupleType([q.Output, k.Output, v.Output]));
                        }
                    }
                }
            }
        }
    }

    private static IEnumerable<PackedQKVProjectionCandidate> EnumerateProjection(
        DistributedCandidateContext context,
        PackedQKVParallelLinear target,
        string name,
        DistributedType input,
        TensorType outputTensor,
        int weightIndex,
        int biasIndex,
        int inputScaleIndex,
        int weightScaleIndex)
    {
        if (!TryGetSourceTensorType(context, weightIndex, out var weightTensor))
        {
            yield break;
        }

        foreach (var requestedOutput in context.GetLeafCandidateTypes(outputTensor, [input.Placement])
                     .Where(type => type.Partial is null && HasMatchingOuterPolicies(input, type)))
        {
            if (!TryCreatePackedWeight(target, input, requestedOutput, weightTensor, out var weight))
            {
                continue;
            }

            foreach (var (inputScale, weightScale) in GetScaleCandidates(
                         context,
                         target.QuantizationMode,
                         requestedOutput,
                         inputScaleIndex,
                         weightScaleIndex))
            {
                var output = PackedQKVParallelLinearEvaluator.InferProjectionType(
                    name,
                    input,
                    weight,
                    inputScale,
                    weightScale,
                    target);
                if (output is not DistributedType distributedOutput ||
                    distributedOutput.Placement != input.Placement ||
                    distributedOutput.TensorType != requestedOutput.TensorType ||
                    !distributedOutput.AxisPolicies.SequenceEqual(requestedOutput.AxisPolicies) ||
                    !TryGetBiasType(context, biasIndex, distributedOutput, out var bias))
                {
                    continue;
                }

                yield return new(weight, bias, inputScale, weightScale, distributedOutput);
            }
        }
    }

    private static bool TryCreatePackedWeight(
        PackedQKVParallelLinear target,
        DistributedType input,
        DistributedType requestedOutput,
        TensorType packedWeightTensor,
        out DistributedType packedWeight)
    {
        packedWeight = null!;
        if (packedWeightTensor.DType is not VectorType weightVector ||
            !PackedQKVParallelLinearEvaluator.TryGetLayoutInfo(
                target.RhsLayout,
                weightVector,
                packedWeightTensor.Shape.Rank,
                target.OutputNVectorLaneCount,
                out var rhsUnpackAxes,
                out var outputLanes,
                out var transposeB,
                out _) ||
            TypeInference.UnpackType(packedWeightTensor, rhsUnpackAxes) is not TensorType logicalWeightTensor ||
            TypeInference.UnpackType(
                requestedOutput,
                Enumerable.Repeat(requestedOutput.TensorType.Shape.Rank - 1, outputLanes.Length).ToArray()) is not
                DistributedType logicalOutput)
        {
            return false;
        }

        var dimInfo = VectorizedMatMul.GetDimInfo(
            false,
            transposeB,
            input.TensorType.Shape.Rank,
            logicalWeightTensor.Shape.Rank);
        var logicalWeightPolicies = Enumerable.Repeat<SBP>(SBP.B, logicalWeightTensor.Shape.Rank).ToArray();
        logicalWeightPolicies[dimInfo.Rk] = input.AxisPolicies[dimInfo.Lk];
        logicalWeightPolicies[dimInfo.Rn] = logicalOutput.AxisPolicies[^1];
        var logicalWeight = new DistributedType(
            logicalWeightTensor,
            logicalWeightPolicies,
            input.Placement);
        if (!DistributedUtility.IsDistributable(
                logicalWeight.TensorType,
                logicalWeight.AxisPolicies,
                logicalWeight.Placement) ||
            TypeInference.PackType(
                logicalWeight,
                weightVector.Lanes.ToArray(),
                rhsUnpackAxes) is not DistributedType candidate ||
            candidate.TensorType != packedWeightTensor ||
            !DistributedUtility.IsDistributable(
                candidate.TensorType,
                candidate.AxisPolicies,
                candidate.Placement))
        {
            return false;
        }

        packedWeight = candidate;
        return true;
    }

    private static IEnumerable<(IRType InputScale, IRType WeightScale)> GetScaleCandidates(
        DistributedCandidateContext context,
        MatMulQuantizationMode mode,
        DistributedType requestedOutput,
        int inputScaleIndex,
        int weightScaleIndex)
    {
        switch (mode)
        {
            case MatMulQuantizationMode.None:
                if (IsNoneSource(context, inputScaleIndex) && IsNoneSource(context, weightScaleIndex))
                {
                    yield return (NoneType.Default, NoneType.Default);
                }

                yield break;
            case MatMulQuantizationMode.DynamicTensor:
                if (!IsNoneSource(context, inputScaleIndex) ||
                    !TryGetSourceTensorType(context, weightScaleIndex, out var weightScaleTensor) ||
                    weightScaleTensor.Shape.Rank != 1)
                {
                    yield break;
                }

                var dynamicWeightScale = new DistributedType(
                    weightScaleTensor,
                    [requestedOutput.AxisPolicies[^1]],
                    requestedOutput.Placement);
                if (DistributedUtility.IsDistributable(
                        dynamicWeightScale.TensorType,
                        dynamicWeightScale.AxisPolicies,
                        dynamicWeightScale.Placement))
                {
                    yield return (NoneType.Default, dynamicWeightScale);
                }

                yield break;
            case MatMulQuantizationMode.StaticTensor:
                foreach (var inputScale in GetStaticScaleCandidates(context, inputScaleIndex, requestedOutput.Placement))
                {
                    foreach (var weightScale in GetStaticScaleCandidates(context, weightScaleIndex, requestedOutput.Placement))
                    {
                        yield return (inputScale, weightScale);
                    }
                }

                yield break;
            default:
                yield break;
        }
    }

    private static IReadOnlyList<IRType> GetStaticScaleCandidates(
        DistributedCandidateContext context,
        int index,
        Placement placement)
        => TryGetSourceTensorType(context, index, out var tensorType)
            ? context.GetLeafCandidateTypes(tensorType, [placement])
                .Where(ScaledMatMulEvaluator.IsScaleType)
                .Cast<IRType>()
                .Distinct()
                .ToArray()
            : [];

    private static bool HasMatchingOuterPolicies(DistributedType input, DistributedType output)
        => input.Placement == output.Placement &&
           input.Partial is null &&
           output.Partial is null &&
           input.AxisPolicies.Count == output.AxisPolicies.Count &&
           input.AxisPolicies.Take(input.AxisPolicies.Count - 1)
               .SequenceEqual(output.AxisPolicies.Take(output.AxisPolicies.Count - 1));

    private static IEnumerable<Placement> GetPlacements(DistributedCandidateContext context)
        => context.AvailableInputTypes
            .SelectMany(types => types)
            .OfType<DistributedType>()
            .Select(type => type.Placement)
            .Distinct();

    private static bool IsNoneSource(DistributedCandidateContext context, int index)
        => context.SourceCall.Arguments[index].CheckedType is NoneType;

    private static bool TryGetSourceTensorType(
        DistributedCandidateContext context,
        int index,
        out TensorType tensorType)
        => TryGetTensorType(context.SourceCall.Arguments[index].CheckedType, out tensorType);

    private static bool TryGetTensorType(IRType type, out TensorType tensorType)
    {
        tensorType = type switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => distributed.TensorType,
            _ => null!,
        };
        return tensorType is not null;
    }

    private static bool TryGetBiasType(
        DistributedCandidateContext context,
        int index,
        DistributedType output,
        out IRType type)
    {
        var sourceType = context.SourceCall.Arguments[index].CheckedType switch
        {
            DistributedType distributed => distributed.TensorType,
            IRType value => value,
        };
        if (sourceType is NoneType)
        {
            type = NoneType.Default;
            return true;
        }

        if (output.Partial is not null ||
            sourceType is not TensorType tensorType ||
            !BroadcastCandidateUtility.TryProjectOutputLayout(output, tensorType, out var projected))
        {
            type = null!;
            return false;
        }

        type = projected;
        return true;
    }

    private static IEnumerable<PackedQKVPartialCandidate> EnumeratePartial(
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
                        yield return new PackedQKVPartialCandidate(arguments, outputType);
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
                target.OutputNVectorLaneCount,
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

    private sealed record PackedQKVProjectionCandidate(
        DistributedType Weight,
        IRType Bias,
        IRType InputScale,
        IRType WeightScale,
        DistributedType Output);

    private sealed record PackedQKVCandidate(
        DistributedType Input,
        PackedQKVProjectionCandidate Q,
        PackedQKVProjectionCandidate K,
        PackedQKVProjectionCandidate V,
        TupleType Output);

    private sealed record PackedQKVPartialCandidate(
        IReadOnlyList<IRType> Arguments,
        IRType Output);
}
