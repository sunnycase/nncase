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

    private static bool AreValidAxes(IReadOnlyList<int> axes, int placementRank) =>
        axes.Distinct().Count() == axes.Count && axes.All(axis => axis >= 0 && axis < placementRank);
}

/// <summary>
/// Derives channel-parallel projection candidates independently from recurrent-head placement.
/// </summary>
internal sealed class GatedDeltaNetProjectionCandidateProvider :
    DistributedCandidateProvider<GatedDeltaNetProjection>
{
    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        GatedDeltaNetProjection target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        if (!GatedDeltaNetCandidateUtility.TryGetSourceTensorTypes(context, target, out var inputs) ||
            GatedDeltaNetProjectionEvaluator.InferTensorType(target, inputs) is not TupleType output)
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
        GatedDeltaNetProjection target,
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
        IRType[] inputs = sourceTypes.ToArray();
        inputs[GatedDeltaNetProjection.Input.Index] =
            GatedDeltaNetCandidateUtility.Broadcast(
                GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetProjection.Input.Index]),
                placement);
        inputs[GatedDeltaNetProjection.State.Index] = stateOutput;
        inputs[GatedDeltaNetProjection.QKVWeight.Index] =
            GatedDeltaNetCandidateUtility.Create(
                GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetProjection.QKVWeight.Index]),
                [SBP.B, channel],
                placement);
        inputs[GatedDeltaNetProjection.ConvWeight.Index] =
            GatedDeltaNetCandidateUtility.Create(
                GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetProjection.ConvWeight.Index]),
                [channel, SBP.B],
                placement);
        if (GatedDeltaNetProjectionEvaluator.InferType(target, inputs) != output)
        {
            return true;
        }

        tuples =
        [
            new DistributedCandidateTuple(
                inputs,
                $"gated-delta-net-projection-channel=[{string.Join(',', channelAxes)}]"),
        ];
        return true;
    }
}

/// <summary>
/// Derives the unique value-head ownership required by the recurrent state update.
/// </summary>
internal sealed class GatedDeltaNetRecurrentCoreCandidateProvider :
    DistributedCandidateProvider<GatedDeltaNetRecurrentCore>
{
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
        foreach (var placement in GatedDeltaNetCandidateUtility.GetPlacements(defaultReturnTypes))
        {
            var headAxes = Enumerable.Range(0, placement.Rank).ToArray();
            var headShardCount = headAxes.Aggregate(
                1L,
                (product, axis) => checked(product * placement.Hierarchy[axis]));
            if (target.NumValueHeads % headShardCount != 0)
            {
                continue;
            }

            var head = GatedDeltaNetCandidateUtility.CreateSplitPolicy(headAxes);
            var candidate = new TupleType([
                new DistributedType((TensorType)output[0], [SBP.B, head], placement),
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
        var head = GatedDeltaNetCandidateUtility.CreateSplitPolicy(headAxes);
        IRType[] inputs = sourceTypes.ToArray();
        inputs[GatedDeltaNetRecurrentCore.Input.Index] = GatedDeltaNetCandidateUtility.Broadcast(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.Input.Index]), placement);
        inputs[GatedDeltaNetRecurrentCore.State.Index] = stateOutput;
        inputs[GatedDeltaNetRecurrentCore.QKV.Index] = GatedDeltaNetCandidateUtility.Broadcast(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.QKV.Index]), placement);
        inputs[GatedDeltaNetRecurrentCore.ZWeight.Index] = GatedDeltaNetCandidateUtility.Create(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.ZWeight.Index]), [SBP.B, head], placement);
        inputs[GatedDeltaNetRecurrentCore.BWeight.Index] = GatedDeltaNetCandidateUtility.Create(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.BWeight.Index]), [SBP.B, head], placement);
        inputs[GatedDeltaNetRecurrentCore.AWeight.Index] = GatedDeltaNetCandidateUtility.Create(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.AWeight.Index]), [SBP.B, head], placement);
        inputs[GatedDeltaNetRecurrentCore.ALog.Index] = GatedDeltaNetCandidateUtility.Create(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.ALog.Index]), [head], placement);
        inputs[GatedDeltaNetRecurrentCore.DtBias.Index] = GatedDeltaNetCandidateUtility.Create(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.DtBias.Index]), [head], placement);
        inputs[GatedDeltaNetRecurrentCore.NormWeight.Index] = GatedDeltaNetCandidateUtility.Broadcast(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.NormWeight.Index]), placement);
        if (GatedDeltaNetRecurrentCoreEvaluator.InferType(target, inputs) != output)
        {
            return true;
        }

        tuples =
        [
            new DistributedCandidateTuple(
                inputs,
                $"gated-delta-net-recurrent-core-head=[{string.Join(',', headAxes)}]"),
        ];
        return true;
    }
}
