// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;

namespace Nncase.Passes.Distributed;

internal static class GatedDeltaNetCandidateUtility
{
    public static IEnumerable<Placement> GetPlacements(IReadOnlyList<IRType> defaultReturnTypes) =>
        defaultReturnTypes
            .OfType<TupleType>()
            .SelectMany(tuple => tuple.Fields.OfType<DistributedType>().Select(type => type.Placement))
            .Distinct();

    public static bool TryGetSplitAxes(SBP policy, int placementRank, out int[] axes)
    {
        axes = policy switch
        {
            SBPBroadCast => Array.Empty<int>(),
            SBPSplit { IsContiguous: true } split => split.HierarchyAxes.ToArray(),
            _ => null!,
        };
        return axes is not null && AreValidAxes(axes, placementRank);
    }

    public static bool IsSumPartialPartition(
        DistributedType type,
        TensorType tensorType,
        Placement placement,
        int splitTensorAxis)
    {
        if (type.TensorType != tensorType ||
            type.Placement != placement ||
            type.Partial is not { Op: ReduceOp.Sum } partial ||
            partial.Axes.Count == 0 ||
            type.AxisPolicies.Count != tensorType.Shape.Rank ||
            type.AxisPolicies.Where((_, axis) => axis != splitTensorAxis)
                .Any(policy => policy is not SBPBroadCast))
        {
            return false;
        }

        var splitAxes = type.AxisPolicies[splitTensorAxis] switch
        {
            SBPBroadCast => Array.Empty<int>(),
            SBPSplit split => split.HierarchyAxes.ToArray(),
            _ => null,
        };
        var partialAxes = partial.Axes.ToArray();
        return splitAxes is not null &&
            AreValidAxes(splitAxes, placement.Rank) &&
            AreValidAxes(partialAxes, placement.Rank) &&
            !splitAxes.Intersect(partialAxes).Any() &&
            CoversPlacement(splitAxes.Concat(partialAxes).ToArray(), placement.Rank);
    }

    public static bool CoversPlacement(IReadOnlyList<int> axes, int placementRank) =>
        axes.OrderBy(axis => axis).SequenceEqual(Enumerable.Range(0, placementRank));

    public static SBP CreateSplitPolicy(IReadOnlyList<int> axes) =>
        axes.Count == 0 ? SBP.B : SBP.SContiguous(axes.ToArray());

    public static DistributedType Create(
        TensorType tensorType,
        IReadOnlyList<SBP> policies,
        Placement placement) =>
        new(tensorType, policies.ToArray(), placement);

    public static DistributedType Broadcast(TensorType tensorType, Placement placement) =>
        Create(tensorType, Enumerable.Repeat<SBP>(SBP.B, tensorType.Shape.Rank).ToArray(), placement);

    public static bool TryGetSourceTensorTypes<TOp>(
        DistributedCandidateContext context,
        TOp target,
        out IRType[] tensorTypes)
        where TOp : Op
    {
        tensorTypes = new IRType[target.Parameters.Count];
        foreach (var parameter in target.Parameters)
        {
            tensorTypes[parameter.Index] = context.SourceCall.Arguments[parameter.Index].CheckedType switch
            {
                TensorType tensor => tensor,
                DistributedType distributed => distributed.TensorType,
                DimensionType dimension => dimension,
                NoneType none => none,
                _ => null!,
            };
            if (tensorTypes[parameter.Index] is null)
            {
                return false;
            }
        }

        return true;
    }

    public static TensorType GetTensorType(IRType type) => type as TensorType
        ?? throw new InvalidOperationException($"Expected a tensor type, got {type}.");

    public static long GetVectorLaneCount(DataType type) => type switch
    {
        VectorType vector => vector.Lanes.Aggregate(
            GetVectorLaneCount(vector.ElemType),
            static (product, lane) => checked(product * lane)),
        _ => 1,
    };

    private static bool AreValidAxes(IReadOnlyList<int> axes, int placementRank) =>
        axes.Distinct().Count() == axes.Count && axes.All(axis => axis >= 0 && axis < placementRank);
}

/// <summary>
/// Derives channel-parallel convolution candidates independently from recurrent-head placement.
/// </summary>
internal sealed class GatedDeltaNetConvolutionCandidateProvider :
    DistributedCandidateProvider<GatedDeltaNetConvolution>
{
    public override bool AllowsPartialInputs => true;

    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        GatedDeltaNetConvolution target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        if (!GatedDeltaNetCandidateUtility.TryGetSourceTensorTypes(context, target, out var inputs) ||
            GatedDeltaNetConvolutionEvaluator.InferTensorType(target, inputs) is not TupleType output)
        {
            return Array.Empty<IRType>();
        }

        var results = new HashSet<IRType>();
        foreach (var placement in GatedDeltaNetCandidateUtility.GetPlacements(defaultReturnTypes))
        {
            var channelAxes = Enumerable.Range(0, placement.Rank).ToArray();
            var channel = GatedDeltaNetCandidateUtility.CreateSplitPolicy(channelAxes);
            var candidate = new TupleType([
                GatedDeltaNetCandidateUtility.Create((TensorType)output[0], [SBP.B, channel], placement),
                output[1],
            ]);
            if (DistributedUtility.IsDistributable(
                    ((DistributedType)candidate[0]).TensorType,
                    ((DistributedType)candidate[0]).AxisPolicies,
                    ((DistributedType)candidate[0]).Placement))
            {
                results.Add(candidate);
            }
        }

        return results.ToArray();
    }

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        GatedDeltaNetConvolution target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (returnType is not TupleType { Count: 2 } output ||
            output[0] is not DistributedType qkvOutput ||
            output[1] is not TensorType stateOutput ||
            qkvOutput.Partial is not null ||
            qkvOutput.AxisPolicies.Count != 2 ||
            qkvOutput.AxisPolicies[0] is not SBPBroadCast ||
            !GatedDeltaNetCandidateUtility.TryGetSplitAxes(
                qkvOutput.AxisPolicies[1],
                qkvOutput.Placement.Rank,
                out var channelAxes) ||
            !GatedDeltaNetCandidateUtility.CoversPlacement(channelAxes, qkvOutput.Placement.Rank) ||
            !GatedDeltaNetCandidateUtility.TryGetSourceTensorTypes(context, target, out var sourceTypes))
        {
            return true;
        }

        var placement = qkvOutput.Placement;
        var channel = GatedDeltaNetCandidateUtility.CreateSplitPolicy(channelAxes);
        var qkvTensorType = GatedDeltaNetCandidateUtility.GetTensorType(
            sourceTypes[GatedDeltaNetConvolution.QKV.Index]);
        var qkvCandidates = new List<DistributedType>
        {
            GatedDeltaNetCandidateUtility.Create(qkvTensorType, [SBP.B, channel], placement),
        };
        if (context.AvailableInputTypes.Count == target.Parameters.Count)
        {
            qkvCandidates.AddRange(context.AvailableInputTypes[GatedDeltaNetConvolution.QKV.Index]
                .OfType<DistributedType>()
                .Where(type => GatedDeltaNetCandidateUtility.IsSumPartialPartition(
                    type,
                    qkvTensorType,
                    placement,
                    1)));
        }

        var convWeightType = GatedDeltaNetCandidateUtility.Create(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetConvolution.ConvWeight.Index]),
            [channel, SBP.B],
            placement);
        tuples = qkvCandidates
            .Distinct()
            .Select(qkvInput =>
            {
                IRType[] inputs = sourceTypes.ToArray();
                inputs[GatedDeltaNetConvolution.QKV.Index] = qkvInput;
                inputs[GatedDeltaNetConvolution.State.Index] = stateOutput;
                inputs[GatedDeltaNetConvolution.ConvWeight.Index] = convWeightType;
                return (Inputs: inputs, QKV: qkvInput);
            })
            .Where(candidate => GatedDeltaNetConvolutionEvaluator.InferType(target, candidate.Inputs) == output)
            .Select(candidate => new DistributedCandidateTuple(
                candidate.Inputs,
                candidate.QKV.Partial is null
                    ? $"gated-delta-net-convolution-channel=[{string.Join(',', channelAxes)}]"
                    : $"gated-delta-net-convolution-direct-sum-partial=[{string.Join(',', candidate.QKV.Partial.Axes)}]"))
            .ToArray();
        return true;
    }
}

/// <summary>
/// Derives the unique value-head ownership required by the recurrent state update.
/// </summary>
internal sealed class GatedDeltaNetRecurrentCoreCandidateProvider :
    DistributedCandidateProvider<GatedDeltaNetRecurrentCore>
{
    public override bool AllowsPartialInputs => true;

    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        GatedDeltaNetRecurrentCore target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        if (!GatedDeltaNetCandidateUtility.TryGetSourceTensorTypes(context, target, out var inputs) ||
            GatedDeltaNetRecurrentCoreEvaluator.InferTensorType(target, inputs) is not TupleType output)
        {
            return Array.Empty<IRType>();
        }

        var results = new HashSet<IRType>();
        var zType = GatedDeltaNetCandidateUtility.GetTensorType(
            inputs[GatedDeltaNetRecurrentCore.Z.Index]);
        var zLaneCount = GatedDeltaNetCandidateUtility.GetVectorLaneCount(zType.DType);
        var scalarValueElements = checked(target.NumValueHeads * target.ValueHeadDim);
        if (scalarValueElements % zLaneCount != 0)
        {
            return Array.Empty<IRType>();
        }

        foreach (var placement in GatedDeltaNetCandidateUtility.GetPlacements(defaultReturnTypes))
        {
            var headAxes = Enumerable.Range(0, placement.Rank).ToArray();
            var value = DistributedUtility.CreateUnitAlignedContiguousSplit(
                headAxes,
                placement,
                scalarValueElements / zLaneCount,
                zLaneCount);
            var candidate = new TupleType([
                new DistributedType((TensorType)output[0], [SBP.B, value], placement),
                output[1],
            ]);
            if (DistributedUtility.IsDistributable(
                    ((DistributedType)candidate[0]).TensorType,
                    ((DistributedType)candidate[0]).AxisPolicies,
                    ((DistributedType)candidate[0]).Placement))
            {
                results.Add(candidate);
            }
        }

        return results.ToArray();
    }

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        GatedDeltaNetRecurrentCore target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (returnType is not TupleType { Count: 2 } output ||
            output[0] is not DistributedType gatedOutput ||
            output[1] is not TensorType stateOutput ||
            gatedOutput.Partial is not null ||
            gatedOutput.AxisPolicies.Count != 2 ||
            gatedOutput.AxisPolicies[0] is not SBPBroadCast ||
            !GatedDeltaNetCandidateUtility.TryGetSplitAxes(
                gatedOutput.AxisPolicies[1],
                gatedOutput.Placement.Rank,
                out var headAxes) ||
            !GatedDeltaNetCandidateUtility.CoversPlacement(headAxes, gatedOutput.Placement.Rank) ||
            !GatedDeltaNetCandidateUtility.TryGetSourceTensorTypes(context, target, out var sourceTypes))
        {
            return true;
        }

        var placement = gatedOutput.Placement;
        var zType = GatedDeltaNetCandidateUtility.GetTensorType(
            sourceTypes[GatedDeltaNetRecurrentCore.Z.Index]);
        var zLaneCount = GatedDeltaNetCandidateUtility.GetVectorLaneCount(zType.DType);
        var scalarValueElements = checked(target.NumValueHeads * target.ValueHeadDim);
        if (scalarValueElements % zLaneCount != 0)
        {
            return true;
        }

        var value = DistributedUtility.CreateUnitAlignedContiguousSplit(
            headAxes,
            placement,
            scalarValueElements / zLaneCount,
            zLaneCount);
        if (!DistributedUtility.TryScaleSplitUnits(
                value,
                1,
                zLaneCount,
                out var packedValue))
        {
            return true;
        }

        if (gatedOutput.AxisPolicies[1] != value)
        {
            return true;
        }

        var zCandidates = new List<DistributedType>
        {
            GatedDeltaNetCandidateUtility.Create(zType, [SBP.B, packedValue], placement),
        };
        if (context.AvailableInputTypes.Count == target.Parameters.Count)
        {
            zCandidates.AddRange(context.AvailableInputTypes[GatedDeltaNetRecurrentCore.Z.Index]
                .OfType<DistributedType>()
                .Where(type => GatedDeltaNetCandidateUtility.IsSumPartialPartition(
                    type,
                    zType,
                    placement,
                    1)));
        }

        tuples = zCandidates
            .Distinct()
            .Select(zInput =>
            {
                IRType[] inputs = sourceTypes.ToArray();
                inputs[GatedDeltaNetRecurrentCore.State.Index] = stateOutput;
                inputs[GatedDeltaNetRecurrentCore.QKV.Index] = GatedDeltaNetCandidateUtility.Broadcast(
                    GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.QKV.Index]), placement);
                inputs[GatedDeltaNetRecurrentCore.Z.Index] = zInput;
                inputs[GatedDeltaNetRecurrentCore.ProjectionInput.Index] = GatedDeltaNetCandidateUtility.Broadcast(
                    GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.ProjectionInput.Index]), placement);
                inputs[GatedDeltaNetRecurrentCore.BWeight.Index] = GatedDeltaNetCandidateUtility.Broadcast(
                    GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.BWeight.Index]), placement);
                inputs[GatedDeltaNetRecurrentCore.AWeight.Index] = GatedDeltaNetCandidateUtility.Broadcast(
                    GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.AWeight.Index]), placement);
                inputs[GatedDeltaNetRecurrentCore.ALog.Index] = GatedDeltaNetCandidateUtility.Broadcast(
                    GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.ALog.Index]), placement);
                inputs[GatedDeltaNetRecurrentCore.DtBias.Index] = GatedDeltaNetCandidateUtility.Broadcast(
                    GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.DtBias.Index]), placement);
                inputs[GatedDeltaNetRecurrentCore.NormWeight.Index] = GatedDeltaNetCandidateUtility.Broadcast(
                    GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.NormWeight.Index]), placement);
                return (Inputs: inputs, Z: zInput);
            })
            .Where(candidate => GatedDeltaNetRecurrentCoreEvaluator.InferType(target, candidate.Inputs) == output)
            .Select(candidate => new DistributedCandidateTuple(
                candidate.Inputs,
                candidate.Z.Partial is null
                    ? $"gated-delta-net-recurrent-core-head=[{string.Join(',', headAxes)}]"
                    : $"gated-delta-net-recurrent-core-direct-sum-partial=[{string.Join(',', candidate.Z.Partial.Axes)}]"))
            .ToArray();
        return true;
    }
}
