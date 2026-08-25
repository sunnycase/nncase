// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator;
using Nncase.Evaluator.IR.NTT;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;

namespace Nncase.Passes.Distributed;

internal static class PackedNVFP4DistributedCandidates
{
    public static IEnumerable<PackedNVFP4ProjectionCandidate> EnumerateProjection(
        DistributedCandidateContext context,
        DataType outputDataType,
        long groupSize,
        int inputKVectorLaneCount,
        int rhsKPackLaneCount,
        int rhsKVectorLaneCount,
        int outputNVectorLaneCount,
        int inputIndex,
        int weightPackedIndex,
        int weightScaleIndex,
        int inputGlobalScaleIndex,
        int weightGlobalScaleIndex,
        bool allowReductionSplit)
    {
        if (!TryGetSourceTensorType(context, inputIndex, out var inputTensor) ||
            !TryGetSourceTensorType(context, weightPackedIndex, out var weightPackedTensor) ||
            !TryGetSourceTensorType(context, weightScaleIndex, out var weightScaleTensor) ||
            !TryGetSourceTensorType(context, inputGlobalScaleIndex, out var inputGlobalScaleTensor) ||
            !TryGetSourceTensorType(context, weightGlobalScaleIndex, out var weightGlobalScaleTensor) ||
            context.SourceCall.CheckedType is not TensorType outputTensor)
        {
            yield break;
        }

        foreach (var placement in GetPlacements(context))
        {
            var outputCandidates = context.GetLeafCandidateTypes(outputTensor, [placement])
                .Where(type => type.Partial is null)
                .ToArray();
            var inputCandidates = context.GetLeafCandidateTypes(inputTensor, [placement]);
            foreach (var requestedOutput in outputCandidates)
            {
                foreach (var input in inputCandidates.Where(candidate =>
                             HasMatchingOuterPolicies(candidate, requestedOutput) &&
                             (candidate.AxisPolicies[^1] is SBPBroadCast || allowReductionSplit)))
                {
                    if (TryCreateProjectionCandidate(
                            outputDataType,
                            groupSize,
                            inputKVectorLaneCount,
                            rhsKPackLaneCount,
                            rhsKVectorLaneCount,
                            outputNVectorLaneCount,
                            requestedOutput,
                            input,
                            weightPackedTensor,
                            weightScaleTensor,
                            inputGlobalScaleTensor,
                            weightGlobalScaleTensor,
                            out var candidate))
                    {
                        yield return candidate;
                    }
                }
            }
        }
    }

    private static bool TryCreateProjectionCandidate(
        DataType outputDataType,
        long groupSize,
        int inputKVectorLaneCount,
        int rhsKPackLaneCount,
        int rhsKVectorLaneCount,
        int outputNVectorLaneCount,
        DistributedType requestedOutput,
        DistributedType input,
        TensorType weightPackedTensor,
        TensorType weightScaleTensor,
        TensorType inputGlobalScaleTensor,
        TensorType weightGlobalScaleTensor,
        out PackedNVFP4ProjectionCandidate candidate)
    {
        candidate = null!;
        var inputRank = input.TensorType.Shape.Rank;
        var outputRank = requestedOutput.TensorType.Shape.Rank;
        if (inputRank < 2 || outputRank != inputRank ||
            TypeInference.UnpackType(input, [inputRank - 1]) is not DistributedType logicalInput ||
            TypeInference.UnpackType(requestedOutput, [outputRank - 1]) is not DistributedType logicalOutput ||
            TypeInference.UnpackType(
                weightPackedTensor,
                [weightPackedTensor.Shape.Rank - 1, weightPackedTensor.Shape.Rank - 1]) is not
                TensorType logicalWeightPackedTensor)
        {
            return false;
        }

        if (!DistributedUtility.TryScaleAxisPolicyUnits(
                logicalInput.AxisPolicies[^1],
                1,
                2,
                out var compressedWeightKPolicy) ||
            !DistributedUtility.TryScaleAxisPolicyUnits(
                logicalInput.AxisPolicies[^1],
                1,
                groupSize,
                out var weightScaleKPolicy))
        {
            return false;
        }

        var logicalWeightPacked = new DistributedType(
            logicalWeightPackedTensor,
            [logicalOutput.AxisPolicies[^1], compressedWeightKPolicy],
            input.Placement);
        if (TypeInference.PackType(
                logicalWeightPacked,
                [rhsKPackLaneCount, rhsKVectorLaneCount],
                [logicalWeightPackedTensor.Shape.Rank - 1, logicalWeightPackedTensor.Shape.Rank - 1]) is not
            DistributedType weightPacked ||
            weightPacked.TensorType != weightPackedTensor ||
            !DistributedUtility.IsDistributable(
                weightPacked.TensorType,
                weightPacked.AxisPolicies,
                weightPacked.Placement))
        {
            return false;
        }

        var weightScale = new DistributedType(
            weightScaleTensor,
            [logicalOutput.AxisPolicies[^1], weightScaleKPolicy],
            input.Placement);
        if (!DistributedUtility.IsDistributable(
                weightScale.TensorType,
                weightScale.AxisPolicies,
                weightScale.Placement))
        {
            return false;
        }

        var inputGlobalScale = Replicate(inputGlobalScaleTensor, input.Placement);
        var weightGlobalScale = Replicate(weightGlobalScaleTensor, input.Placement);
        var output = PackedNVFP4MatMulEvaluator.InferProjectionType(
            outputDataType,
            groupSize,
            inputKVectorLaneCount,
            rhsKPackLaneCount,
            rhsKVectorLaneCount,
            outputNVectorLaneCount,
            input,
            weightPacked,
            weightScale,
            inputGlobalScale,
            weightGlobalScale);
        if (output is not DistributedType distributedOutput ||
            distributedOutput.TensorType != requestedOutput.TensorType ||
            !distributedOutput.AxisPolicies.SequenceEqual(requestedOutput.AxisPolicies))
        {
            return false;
        }

        candidate = new(
            input,
            weightPacked,
            weightScale,
            inputGlobalScale,
            weightGlobalScale,
            distributedOutput);
        return true;
    }

    private static bool HasMatchingOuterPolicies(
        DistributedType input,
        DistributedType output)
    {
        if (input.Placement != output.Placement ||
            input.Partial is not null ||
            input.AxisPolicies.Count != output.AxisPolicies.Count)
        {
            return false;
        }

        return input.AxisPolicies
            .Take(input.AxisPolicies.Count - 1)
            .SequenceEqual(output.AxisPolicies.Take(output.AxisPolicies.Count - 1));
    }

    private static IEnumerable<Placement> GetPlacements(DistributedCandidateContext context)
        => context.AvailableInputTypes
            .SelectMany(types => types)
            .OfType<DistributedType>()
            .Select(type => type.Placement)
            .Distinct();

    private static bool TryGetSourceTensorType(
        DistributedCandidateContext context,
        int index,
        out TensorType tensorType)
    {
        tensorType = context.SourceCall.Arguments[index].CheckedType switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => distributed.TensorType,
            _ => null!,
        };
        return tensorType is not null;
    }

    private static DistributedType Replicate(TensorType tensorType, Placement placement)
        => new(
            tensorType,
            Enumerable.Repeat<SBP>(SBP.B, tensorType.Shape.Rank).ToArray(),
            placement);
}

internal sealed class PackedNVFP4MatMulCandidateProvider :
    DistributedCandidateProvider<PackedNVFP4MatMul>
{
    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedNVFP4MatMul target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => Enumerate(context, target)
            .Select(candidate => candidate.Output)
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedNVFP4MatMul target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Enumerate(context, target)
            .Where(candidate => candidate.Output == returnType)
            .Select(candidate => new DistributedCandidateTuple(
                [
                    candidate.Input,
                    candidate.WeightPacked,
                    candidate.WeightScale,
                    candidate.InputGlobalScale,
                    candidate.WeightGlobalScale,
                ],
                "packed-nvfp4-matmul-sbp"))
            .Distinct()
            .ToArray();
        return true;
    }

    private static IEnumerable<PackedNVFP4ProjectionCandidate> Enumerate(
        DistributedCandidateContext context,
        PackedNVFP4MatMul target)
    {
        if (context.AvailableInputTypes.Count != 5)
        {
            return [];
        }

        return PackedNVFP4DistributedCandidates.EnumerateProjection(
            context,
            target.OutputDataType,
            target.GroupSize,
            target.InputKVectorLaneCount,
            target.RhsKPackLaneCount,
            target.RhsKVectorLaneCount,
            target.OutputNVectorLaneCount,
            PackedNVFP4MatMul.Lhs.Index,
            PackedNVFP4MatMul.RhsPacked.Index,
            PackedNVFP4MatMul.RhsScale.Index,
            PackedNVFP4MatMul.LhsGlobalScale.Index,
            PackedNVFP4MatMul.RhsGlobalScale.Index,
            allowReductionSplit: true);
    }
}

internal sealed class PackedNVFP4MatMulGluCandidateProvider :
    DistributedCandidateProvider<PackedNVFP4MatMulGlu>
{
    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedNVFP4MatMulGlu target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => Enumerate(context, target)
            .Select(candidate => candidate.Output)
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedNVFP4MatMulGlu target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Enumerate(context, target)
            .Where(candidate => candidate.Output == returnType)
            .Select(candidate => new DistributedCandidateTuple(
                [
                    candidate.Input,
                    candidate.GateWeightPacked,
                    candidate.UpWeightPacked,
                    candidate.GateWeightScale,
                    candidate.UpWeightScale,
                    candidate.GateInputGlobalScale,
                    candidate.UpInputGlobalScale,
                    candidate.GateWeightGlobalScale,
                    candidate.UpWeightGlobalScale,
                ],
                "packed-nvfp4-matmul-glu-sbp"))
            .Distinct()
            .ToArray();
        return true;
    }

    private static IEnumerable<PackedNVFP4MatMulGluCandidate> Enumerate(
        DistributedCandidateContext context,
        PackedNVFP4MatMulGlu target)
    {
        if (context.AvailableInputTypes.Count != 9)
        {
            yield break;
        }

        var gateCandidates = PackedNVFP4DistributedCandidates.EnumerateProjection(
            context,
            target.OutputDataType,
            target.GroupSize,
            target.InputKVectorLaneCount,
            target.RhsKPackLaneCount,
            target.RhsKVectorLaneCount,
            target.OutputNVectorLaneCount,
            PackedNVFP4MatMulGlu.Input.Index,
            PackedNVFP4MatMulGlu.GateWeightPacked.Index,
            PackedNVFP4MatMulGlu.GateWeightScale.Index,
            PackedNVFP4MatMulGlu.GateInputGlobalScale.Index,
            PackedNVFP4MatMulGlu.GateWeightGlobalScale.Index,
            allowReductionSplit: false)
            .ToArray();
        var upCandidates = PackedNVFP4DistributedCandidates.EnumerateProjection(
            context,
            target.OutputDataType,
            target.GroupSize,
            target.InputKVectorLaneCount,
            target.RhsKPackLaneCount,
            target.RhsKVectorLaneCount,
            target.OutputNVectorLaneCount,
            PackedNVFP4MatMulGlu.Input.Index,
            PackedNVFP4MatMulGlu.UpWeightPacked.Index,
            PackedNVFP4MatMulGlu.UpWeightScale.Index,
            PackedNVFP4MatMulGlu.UpInputGlobalScale.Index,
            PackedNVFP4MatMulGlu.UpWeightGlobalScale.Index,
            allowReductionSplit: false)
            .ToArray();

        foreach (var gate in gateCandidates)
        {
            foreach (var up in upCandidates.Where(candidate =>
                         candidate.Input == gate.Input && candidate.Output == gate.Output))
            {
                var output = PackedNVFP4MatMulGluEvaluator.InferType(
                    target,
                    gate.Input,
                    gate.WeightPacked,
                    up.WeightPacked,
                    gate.WeightScale,
                    up.WeightScale,
                    gate.InputGlobalScale,
                    up.InputGlobalScale,
                    gate.WeightGlobalScale,
                    up.WeightGlobalScale);
                if (output is not InvalidType)
                {
                    yield return new(
                        gate.Input,
                        gate.WeightPacked,
                        up.WeightPacked,
                        gate.WeightScale,
                        up.WeightScale,
                        gate.InputGlobalScale,
                        up.InputGlobalScale,
                        gate.WeightGlobalScale,
                        up.WeightGlobalScale,
                        output);
                }
            }
        }
    }
}

internal sealed record PackedNVFP4ProjectionCandidate(
    IRType Input,
    IRType WeightPacked,
    IRType WeightScale,
    IRType InputGlobalScale,
    IRType WeightGlobalScale,
    IRType Output);

internal sealed record PackedNVFP4MatMulGluCandidate(
    IRType Input,
    IRType GateWeightPacked,
    IRType UpWeightPacked,
    IRType GateWeightScale,
    IRType UpWeightScale,
    IRType GateInputGlobalScale,
    IRType UpInputGlobalScale,
    IRType GateWeightGlobalScale,
    IRType UpWeightGlobalScale,
    IRType Output);
