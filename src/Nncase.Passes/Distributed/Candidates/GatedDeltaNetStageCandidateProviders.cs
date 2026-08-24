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
        IRType[] inputs = sourceTypes.ToArray();
        inputs[GatedDeltaNetConvolution.QKV.Index] =
            GatedDeltaNetCandidateUtility.Create(
                GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetConvolution.QKV.Index]),
                [SBP.B, channel],
                placement);
        inputs[GatedDeltaNetConvolution.State.Index] = stateOutput;
        inputs[GatedDeltaNetConvolution.ConvWeight.Index] =
            GatedDeltaNetCandidateUtility.Create(
                GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetConvolution.ConvWeight.Index]),
                [channel, SBP.B],
                placement);
        if (GatedDeltaNetConvolutionEvaluator.InferType(target, inputs) != output)
        {
            return true;
        }

        tuples =
        [
            new DistributedCandidateTuple(
                inputs,
                $"gated-delta-net-convolution-channel=[{string.Join(',', channelAxes)}]"),
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
            var value = DistributedUtility.CreateUnitAlignedContiguousSplit(
                headAxes,
                placement,
                target.NumValueHeads,
                target.ValueHeadDim);
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
        var head = DistributedUtility.CreateUnitAlignedContiguousSplit(
            headAxes,
            placement,
            target.NumValueHeads);
        var value = DistributedUtility.CreateUnitAlignedContiguousSplit(
            headAxes,
            placement,
            target.NumValueHeads,
            target.ValueHeadDim);
        var zType = GatedDeltaNetCandidateUtility.GetTensorType(
            sourceTypes[GatedDeltaNetRecurrentCore.Z.Index]);
        if (!DistributedUtility.TryScaleSplitUnits(
                value,
                1,
                GatedDeltaNetCandidateUtility.GetVectorLaneCount(zType.DType),
                out var packedValue))
        {
            return true;
        }

        if (gatedOutput.AxisPolicies[1] != value)
        {
            return true;
        }

        IRType[] inputs = sourceTypes.ToArray();
        inputs[GatedDeltaNetRecurrentCore.State.Index] = stateOutput;
        inputs[GatedDeltaNetRecurrentCore.QKV.Index] = GatedDeltaNetCandidateUtility.Broadcast(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.QKV.Index]), placement);
        inputs[GatedDeltaNetRecurrentCore.Z.Index] = GatedDeltaNetCandidateUtility.Create(
            zType, [SBP.B, packedValue], placement);
        inputs[GatedDeltaNetRecurrentCore.BProjection.Index] = GatedDeltaNetCandidateUtility.Broadcast(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.BProjection.Index]), placement);
        inputs[GatedDeltaNetRecurrentCore.AProjection.Index] = GatedDeltaNetCandidateUtility.Broadcast(
            GatedDeltaNetCandidateUtility.GetTensorType(sourceTypes[GatedDeltaNetRecurrentCore.AProjection.Index]), placement);
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
