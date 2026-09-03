// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.Math;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Shapes;

namespace Nncase.Passes.Distributed;

/// <summary>
/// Derives the three Q/K/V projection contracts from one common input layout.
/// </summary>
internal sealed class QKVParallelLinearCandidateProvider : DistributedCandidateProvider<QKVParallelLinear>
{
    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        QKVParallelLinear target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        if (!HasExpectedSignature(context))
        {
            return Array.Empty<IRType>();
        }

        var results = new HashSet<IRType>();
        foreach (var input in GetDistributedCandidates(context, QKVParallelLinear.Input.Index))
        {
            var qOutputs = GetProjectionCandidates(context, target, input, QKVParallelLinear.QWeight.Index)
                .Select(candidate => candidate.Output)
                .Distinct()
                .ToArray();
            var kOutputs = GetProjectionCandidates(context, target, input, QKVParallelLinear.KWeight.Index)
                .Select(candidate => candidate.Output)
                .Distinct()
                .ToArray();
            var vOutputs = GetProjectionCandidates(context, target, input, QKVParallelLinear.VWeight.Index)
                .Select(candidate => candidate.Output)
                .Distinct()
                .ToArray();

            foreach (var qOutput in qOutputs)
            {
                foreach (var kOutput in kOutputs)
                {
                    foreach (var vOutput in vOutputs)
                    {
                        if (TryGetAuxiliaryTypes(context, target, qOutput, kOutput, vOutput, out _))
                        {
                            results.Add(new TupleType([qOutput, kOutput, vOutput]));
                        }
                    }
                }
            }
        }

        return results.ToArray();
    }

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        QKVParallelLinear target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (!HasExpectedSignature(context) ||
            returnType is not TupleType { Count: 3 } output ||
            output[0] is not DistributedType qOutput ||
            output[1] is not DistributedType kOutput ||
            output[2] is not DistributedType vOutput ||
            qOutput.Placement != kOutput.Placement ||
            qOutput.Placement != vOutput.Placement ||
            !TryGetAuxiliaryTypes(context, target, qOutput, kOutput, vOutput, out var auxiliaryTypes))
        {
            return true;
        }

        var results = new List<DistributedCandidateTuple>();
        foreach (var input in GetDistributedCandidates(context, QKVParallelLinear.Input.Index)
                     .Where(candidate => candidate.Placement == qOutput.Placement))
        {
            var qWeights = GetProjectionCandidates(context, target, input, QKVParallelLinear.QWeight.Index)
                .Where(candidate => candidate.Output == qOutput)
                .Select(candidate => candidate.Weight)
                .Distinct()
                .ToArray();
            var kWeights = GetProjectionCandidates(context, target, input, QKVParallelLinear.KWeight.Index)
                .Where(candidate => candidate.Output == kOutput)
                .Select(candidate => candidate.Weight)
                .Distinct()
                .ToArray();
            var vWeights = GetProjectionCandidates(context, target, input, QKVParallelLinear.VWeight.Index)
                .Where(candidate => candidate.Output == vOutput)
                .Select(candidate => candidate.Weight)
                .Distinct()
                .ToArray();

            foreach (var qWeight in qWeights)
            {
                foreach (var kWeight in kWeights)
                {
                    foreach (var vWeight in vWeights)
                    {
                        results.Add(new DistributedCandidateTuple(
                            [
                                input,
                                qWeight,
                                kWeight,
                                vWeight,
                                auxiliaryTypes.QBias,
                                auxiliaryTypes.KBias,
                                auxiliaryTypes.VBias,
                                auxiliaryTypes.QInputScale,
                                auxiliaryTypes.KInputScale,
                                auxiliaryTypes.VInputScale,
                                auxiliaryTypes.QWeightScale,
                                auxiliaryTypes.KWeightScale,
                                auxiliaryTypes.VWeightScale,
                            ],
                            "qkv-parallel-linear-output-sbp"));
                    }
                }
            }
        }

        tuples = results;
        return true;
    }

    private static bool HasExpectedSignature(DistributedCandidateContext context)
        => context.AvailableInputTypes.Count == 13 &&
           context.SourceCall.CheckedType is TupleType { Count: 3 };

    private static IEnumerable<DistributedType> GetDistributedCandidates(
        DistributedCandidateContext context,
        int index)
        => context.AvailableInputTypes[index]
            .OfType<DistributedType>()
            .Where(type => type.Partial is null)
            .Distinct();

    private static IEnumerable<ProjectionCandidate> GetProjectionCandidates(
        DistributedCandidateContext context,
        QKVParallelLinear target,
        DistributedType input,
        int weightIndex)
    {
        foreach (var weight in GetDistributedCandidates(context, weightIndex)
                     .Where(candidate => candidate.Placement == input.Placement))
        {
            var inferenceWeight = weight with
            {
                TensorType = weight.TensorType with { DType = input.TensorType.DType },
            };
            var output = MatMulEvaluator.VisitDistributedType(
                input,
                inferenceWeight,
                NoneType.Default,
                outputDataType: target.OutputDataType);
            if (output is DistributedType distributedOutput)
            {
                yield return new ProjectionCandidate(weight, distributedOutput);
            }
        }
    }

    private static bool TryGetAuxiliaryTypes(
        DistributedCandidateContext context,
        QKVParallelLinear target,
        DistributedType qOutput,
        DistributedType kOutput,
        DistributedType vOutput,
        out AuxiliaryTypes types)
    {
        types = null!;
        if (!TryGetBiasType(context, QKVParallelLinear.QBias.Index, qOutput, out var qBias) ||
            !TryGetBiasType(context, QKVParallelLinear.KBias.Index, kOutput, out var kBias) ||
            !TryGetBiasType(context, QKVParallelLinear.VBias.Index, vOutput, out var vBias) ||
            !TryGetReplicatedType(context, QKVParallelLinear.QInputScale.Index, qOutput.Placement, out var qInputScale) ||
            !TryGetReplicatedType(context, QKVParallelLinear.KInputScale.Index, qOutput.Placement, out var kInputScale) ||
            !TryGetReplicatedType(context, QKVParallelLinear.VInputScale.Index, qOutput.Placement, out var vInputScale) ||
            !TryGetWeightScaleType(context, target, QKVParallelLinear.QWeightScale.Index, qOutput, out var qWeightScale) ||
            !TryGetWeightScaleType(context, target, QKVParallelLinear.KWeightScale.Index, kOutput, out var kWeightScale) ||
            !TryGetWeightScaleType(context, target, QKVParallelLinear.VWeightScale.Index, vOutput, out var vWeightScale))
        {
            return false;
        }

        types = new(
            qBias,
            kBias,
            vBias,
            qInputScale,
            kInputScale,
            vInputScale,
            qWeightScale,
            kWeightScale,
            vWeightScale);
        return true;
    }

    private static bool TryGetWeightScaleType(
        DistributedCandidateContext context,
        QKVParallelLinear target,
        int index,
        DistributedType output,
        out IRType type)
    {
        if (target.QuantizationMode != IR.Math.MatMulQuantizationMode.DynamicTensor)
        {
            return TryGetReplicatedType(context, index, output.Placement, out type);
        }

        if (!TryGetSourceType(context, index, out var sourceType) ||
            sourceType is not TensorType { Shape: RankedShape { Rank: 1 } } tensorType)
        {
            type = null!;
            return false;
        }

        type = new DistributedType(
            tensorType,
            [output.AxisPolicies[^1]],
            output.Placement);
        return true;
    }

    private static bool TryGetBiasType(
        DistributedCandidateContext context,
        int index,
        DistributedType output,
        out IRType type)
    {
        if (TryGetSourceType(context, index, out var sourceType) && sourceType is NoneType)
        {
            type = NoneType.Default;
            return true;
        }

        if (output.Partial is not null ||
            sourceType is not TensorType tensorType ||
            !BroadcastCandidateUtility.TryProjectOutputLayout(output, tensorType, out var distributedType))
        {
            type = null!;
            return false;
        }

        type = distributedType;
        return true;
    }

    private static bool TryGetReplicatedType(
        DistributedCandidateContext context,
        int index,
        Placement placement,
        out IRType type)
    {
        if (!TryGetSourceType(context, index, out var sourceType))
        {
            type = null!;
            return false;
        }

        switch (sourceType)
        {
            case NoneType:
                type = NoneType.Default;
                return true;
            case TensorType tensorType:
                type = new DistributedType(
                    tensorType,
                    Enumerable.Repeat<SBP>(SBP.B, tensorType.Shape.Rank).ToArray(),
                    placement);
                return true;
            default:
                type = null!;
                return false;
        }
    }

    private static bool TryGetSourceType(
        DistributedCandidateContext context,
        int index,
        out IRType type)
    {
        type = context.SourceCall.Arguments[index].CheckedType switch
        {
            DistributedType distributed => distributed.TensorType,
            IRType value => value,
        };
        return type is TensorType or NoneType;
    }

    private sealed record ProjectionCandidate(DistributedType Weight, DistributedType Output);

    private sealed record AuxiliaryTypes(
        IRType QBias,
        IRType KBias,
        IRType VBias,
        IRType QInputScale,
        IRType KInputScale,
        IRType VInputScale,
        IRType QWeightScale,
        IRType KWeightScale,
        IRType VWeightScale);
}
