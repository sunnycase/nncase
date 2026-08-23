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

internal static class SparseExpertsStageCandidateUtility
{
    public static bool TryGetSourceTensorTypes<TOp>(
        DistributedCandidateContext context,
        TOp target,
        out TensorType[] tensorTypes)
        where TOp : Op
    {
        tensorTypes = new TensorType[target.Parameters.Count];
        foreach (var parameter in target.Parameters)
        {
            tensorTypes[parameter.Index] = context.SourceCall.Arguments[parameter.Index].CheckedType switch
            {
                TensorType tensor => tensor,
                DistributedType distributed => distributed.TensorType,
                _ => null!,
            };
            if (tensorTypes[parameter.Index] is null)
            {
                return false;
            }
        }

        return true;
    }

    public static bool TryGetRoleAxes(SBP policy, int placementRank, out int[] axes)
    {
        axes = policy switch
        {
            SBPBroadCast => Array.Empty<int>(),
            SBPSplit split => split.HierarchyAxes.ToArray(),
            _ => null!,
        };
        return axes is not null && AreValidAxes(axes, placementRank);
    }

    public static bool TryGetPartialAxes(SBPPartial? partial, int placementRank, out int[] axes)
    {
        axes = partial switch
        {
            null => Array.Empty<int>(),
            { Op: ReduceOp.Sum } => partial.Axes.ToArray(),
            _ => null!,
        };
        return axes is not null && AreValidAxes(axes, placementRank);
    }

    public static bool AreDisjoint(params IReadOnlyList<int>[] groups) =>
        groups.SelectMany(static group => group).Distinct().Count() == groups.Sum(static group => group.Count);

    public static bool HaveSameAxes(IReadOnlyList<int> lhs, IReadOnlyList<int> rhs) =>
        lhs.Count == rhs.Count && lhs.OrderBy(static axis => axis).SequenceEqual(rhs.OrderBy(static axis => axis));

    public static bool TryScalePolicy(
        SBP policy,
        long numerator,
        long denominator,
        out SBP scaledPolicy)
    {
        if (policy is not SBPSplit split)
        {
            scaledPolicy = policy;
            return policy is SBPBroadCast;
        }

        if (DistributedUtility.TryScaleSplitUnits(split, numerator, denominator, out var scaledSplit))
        {
            scaledPolicy = scaledSplit;
            return true;
        }

        scaledPolicy = null!;
        return false;
    }

    public static long GetVectorLaneCount(DataType dataType) => dataType switch
    {
        VectorType vector => vector.Lanes.Aggregate(1L, (product, lane) => checked(product * lane)) *
            GetVectorLaneCount(vector.ElemType),
        _ => 1,
    };

    public static DistributedType Create(
        TensorType tensorType,
        IReadOnlyList<SBP> policies,
        Placement placement) =>
        new(tensorType, policies.ToArray(), placement);

    public static DistributedType Broadcast(TensorType tensorType, Placement placement) =>
        Create(
            tensorType,
            Enumerable.Repeat<SBP>(SBP.B, tensorType.Shape.Rank).ToArray(),
            placement);

    private static bool AreValidAxes(IReadOnlyList<int> axes, int placementRank) =>
        axes.Distinct().Count() == axes.Count && axes.All(axis => axis >= 0 && axis < placementRank);
}

internal sealed class SparseExpertsGateUpCandidateProvider :
    DistributedCandidateProvider<SparseExpertsGateUp>
{
    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        SparseExpertsGateUp target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        if (!SparseExpertsStageCandidateUtility.TryGetSourceTensorTypes(context, target, out var sourceTypes) ||
            SparseExpertsGateUpEvaluator.InferType(target, sourceTypes) is not TensorType outputTensor)
        {
            return Array.Empty<IRType>();
        }

        var results = new HashSet<IRType>();
        foreach (var output in defaultReturnTypes
                     .OfType<DistributedType>()
                     .Where(type => type.TensorType == outputTensor && type.Partial is null))
        {
            if (output.AxisPolicies is not [var token, SBPBroadCast, var intermediate] ||
                !SparseExpertsStageCandidateUtility.TryGetRoleAxes(token, output.Placement.Rank, out var tokenAxes) ||
                !SparseExpertsStageCandidateUtility.TryGetRoleAxes(intermediate, output.Placement.Rank, out var intermediateAxes) ||
                !SparseExpertsStageCandidateUtility.AreDisjoint(tokenAxes, intermediateAxes))
            {
                continue;
            }

            if (DistributedUtility.IsDistributable(output.TensorType, output.AxisPolicies, output.Placement))
            {
                results.Add(output);
            }
        }

        return results.ToArray();
    }

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        SparseExpertsGateUp target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (returnType is not DistributedType output ||
            output.Partial is not null ||
            output.AxisPolicies is not [var token, SBPBroadCast, var intermediate] ||
            !SparseExpertsStageCandidateUtility.TryGetRoleAxes(token, output.Placement.Rank, out var tokenAxes) ||
            !SparseExpertsStageCandidateUtility.TryGetRoleAxes(intermediate, output.Placement.Rank, out var intermediateAxes) ||
            !SparseExpertsStageCandidateUtility.AreDisjoint(tokenAxes, intermediateAxes) ||
            !SparseExpertsStageCandidateUtility.TryGetSourceTensorTypes(context, target, out var sourceTypes))
        {
            return true;
        }

        if (!SparseExpertsStageCandidateUtility.TryScalePolicy(
                intermediate,
                SparseExpertsStageCandidateUtility.GetVectorLaneCount(output.TensorType.DType),
                1,
                out var scalarIntermediate))
        {
            return true;
        }

        var inputTypes = sourceTypes.Cast<IRType>().ToArray();
        inputTypes[SparseExpertsGateUp.Q.Index] = SparseExpertsStageCandidateUtility.Create(
            sourceTypes[SparseExpertsGateUp.Q.Index],
            [token, SBP.B],
            output.Placement);
        inputTypes[SparseExpertsGateUp.RouterExpertIds.Index] = SparseExpertsStageCandidateUtility.Create(
            sourceTypes[SparseExpertsGateUp.RouterExpertIds.Index],
            [token, SBP.B],
            output.Placement);
        inputTypes[SparseExpertsGateUp.MoeExpertGateProjW.Index] = SparseExpertsStageCandidateUtility.Create(
            sourceTypes[SparseExpertsGateUp.MoeExpertGateProjW.Index],
            [SBP.B, scalarIntermediate, SBP.B],
            output.Placement);
        inputTypes[SparseExpertsGateUp.MoeExpertUpProjW.Index] = SparseExpertsStageCandidateUtility.Create(
            sourceTypes[SparseExpertsGateUp.MoeExpertUpProjW.Index],
            [SBP.B, scalarIntermediate, SBP.B],
            output.Placement);
        foreach (var parameter in new[]
                 {
                     SparseExpertsGateUp.MoeExpertGateInputScale,
                     SparseExpertsGateUp.MoeExpertGateProjScale,
                     SparseExpertsGateUp.MoeExpertUpInputScale,
                     SparseExpertsGateUp.MoeExpertUpProjScale,
                 })
        {
            inputTypes[parameter.Index] = SparseExpertsStageCandidateUtility.Broadcast(
                sourceTypes[parameter.Index],
                output.Placement);
        }

        if (SparseExpertsGateUpEvaluator.InferType(target, inputTypes) != output)
        {
            return true;
        }

        tuples =
        [
            new DistributedCandidateTuple(
                inputTypes,
                $"sparse-experts-gate-up-token=[{string.Join(',', tokenAxes)}]-intermediate=[{string.Join(',', intermediateAxes)}]"),
        ];
        return true;
    }
}

internal sealed class SparseExpertsDownCandidateProvider :
    DistributedCandidateProvider<SparseExpertsDown>
{
    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        SparseExpertsDown target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        if (!SparseExpertsStageCandidateUtility.TryGetSourceTensorTypes(context, target, out var sourceTypes) ||
            SparseExpertsDownEvaluator.InferType(target, sourceTypes) is not TensorType outputTensor)
        {
            return Array.Empty<IRType>();
        }

        var activationCandidates = context.AvailableInputTypes[SparseExpertsDown.Activations.Index]
            .OfType<DistributedType>()
            .Where(type => type.TensorType == sourceTypes[SparseExpertsDown.Activations.Index] && type.Partial is null)
            .ToArray();
        var results = new HashSet<IRType>();
        foreach (var baseOutput in defaultReturnTypes
                     .OfType<DistributedType>()
                     .Where(type => type.TensorType == outputTensor && type.Partial is null))
        {
            if (baseOutput.AxisPolicies is not [var token, var outputFeature] ||
                !SparseExpertsStageCandidateUtility.TryGetRoleAxes(token, baseOutput.Placement.Rank, out var tokenAxes) ||
                !SparseExpertsStageCandidateUtility.TryGetRoleAxes(outputFeature, baseOutput.Placement.Rank, out var outputAxes) ||
                !SparseExpertsStageCandidateUtility.AreDisjoint(tokenAxes, outputAxes))
            {
                continue;
            }

            foreach (var activation in activationCandidates.Where(type => type.Placement == baseOutput.Placement))
            {
                if (activation.AxisPolicies is not [var activationToken, SBPBroadCast, var intermediate] ||
                    activationToken != token ||
                    !SparseExpertsStageCandidateUtility.TryGetRoleAxes(
                        intermediate,
                        baseOutput.Placement.Rank,
                        out var intermediateAxes) ||
                    !SparseExpertsStageCandidateUtility.AreDisjoint(tokenAxes, intermediateAxes, outputAxes))
                {
                    continue;
                }

                results.Add(baseOutput with
                {
                    Partial = intermediateAxes.Length == 0 ? null : SBP.P(intermediateAxes),
                });
            }
        }

        return results.ToArray();
    }

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        SparseExpertsDown target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Array.Empty<DistributedCandidateTuple>();
        if (returnType is not DistributedType output ||
            output.AxisPolicies is not [var token, var outputFeature] ||
            !SparseExpertsStageCandidateUtility.TryGetRoleAxes(token, output.Placement.Rank, out var tokenAxes) ||
            !SparseExpertsStageCandidateUtility.TryGetRoleAxes(outputFeature, output.Placement.Rank, out var outputAxes) ||
            !SparseExpertsStageCandidateUtility.TryGetPartialAxes(output.Partial, output.Placement.Rank, out var intermediateAxes) ||
            !SparseExpertsStageCandidateUtility.AreDisjoint(tokenAxes, intermediateAxes, outputAxes) ||
            !SparseExpertsStageCandidateUtility.TryGetSourceTensorTypes(context, target, out var sourceTypes))
        {
            return true;
        }

        var outputLaneCount = SparseExpertsStageCandidateUtility.GetVectorLaneCount(output.TensorType.DType);
        if (!SparseExpertsStageCandidateUtility.TryScalePolicy(
                outputFeature,
                outputLaneCount,
                1,
                out var scalarOutputFeature))
        {
            return true;
        }

        var result = new List<DistributedCandidateTuple>();
        foreach (var activation in context.AvailableInputTypes[SparseExpertsDown.Activations.Index]
                     .OfType<DistributedType>()
                     .Where(type =>
                         type.TensorType == sourceTypes[SparseExpertsDown.Activations.Index] &&
                         type.Placement == output.Placement &&
                         type.Partial is null))
        {
            if (activation.AxisPolicies is not [var activationToken, SBPBroadCast, var intermediate] ||
                activationToken != token ||
                !SparseExpertsStageCandidateUtility.TryGetRoleAxes(
                    intermediate,
                    output.Placement.Rank,
                    out var activationIntermediateAxes) ||
                !SparseExpertsStageCandidateUtility.HaveSameAxes(activationIntermediateAxes, intermediateAxes))
            {
                continue;
            }

            var activationLaneCount = SparseExpertsStageCandidateUtility.GetVectorLaneCount(activation.TensorType.DType);
            if (!SparseExpertsStageCandidateUtility.TryScalePolicy(
                    intermediate,
                    activationLaneCount,
                    1,
                    out var scalarIntermediate))
            {
                continue;
            }

            var inputTypes = sourceTypes.Cast<IRType>().ToArray();
            inputTypes[SparseExpertsDown.Activations.Index] = activation;
            inputTypes[SparseExpertsDown.RouterExpertIds.Index] = SparseExpertsStageCandidateUtility.Create(
                sourceTypes[SparseExpertsDown.RouterExpertIds.Index],
                [token, SBP.B],
                output.Placement);
            inputTypes[SparseExpertsDown.RouterExpertWeights.Index] = SparseExpertsStageCandidateUtility.Create(
                sourceTypes[SparseExpertsDown.RouterExpertWeights.Index],
                [token, SBP.B],
                output.Placement);
            inputTypes[SparseExpertsDown.MoeExpertDownProjW.Index] = SparseExpertsStageCandidateUtility.Create(
                sourceTypes[SparseExpertsDown.MoeExpertDownProjW.Index],
                [SBP.B, scalarOutputFeature, scalarIntermediate],
                output.Placement);
            foreach (var parameter in new[]
                     {
                         SparseExpertsDown.MoeExpertDownInputScale,
                         SparseExpertsDown.MoeExpertDownProjScale,
                     })
            {
                inputTypes[parameter.Index] = SparseExpertsStageCandidateUtility.Broadcast(
                    sourceTypes[parameter.Index],
                    output.Placement);
            }

            if (SparseExpertsDownEvaluator.InferType(target, inputTypes) != output ||
                result.Any(tuple => tuple.InputTypes.SequenceEqual(inputTypes)))
            {
                continue;
            }

            result.Add(new DistributedCandidateTuple(
                inputTypes,
                $"sparse-experts-down-token=[{string.Join(',', tokenAxes)}]-intermediate={intermediate}-output=[{string.Join(',', outputAxes)}]"));
        }

        tuples = result;
        return true;
    }
}
