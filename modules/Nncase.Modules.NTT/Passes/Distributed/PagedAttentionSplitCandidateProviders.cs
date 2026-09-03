// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator.IR.NTT;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;

namespace Nncase.Passes.Distributed;

internal sealed class PagedAttentionPartialCandidateProvider :
    DistributedCandidateProvider<PagedAttentionPartial>
{
    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PagedAttentionPartial target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        var results = new List<IRType>();
        foreach (var inputTypes in EnumerateInputTypes(context))
        {
            var result = PagedAttentionPartialEvaluator.InferType(
                target,
                inputTypes[PagedAttentionPartial.Q.Index],
                (TensorType)inputTypes[PagedAttentionPartial.KVCaches.Index],
                inputTypes[PagedAttentionPartial.Extra.Index],
                (TensorType)inputTypes[PagedAttentionPartial.Scale.Index],
                (DimensionType)inputTypes[PagedAttentionPartial.LayerId.Index]);
            if (result is not InvalidType)
            {
                results.Add(result);
            }
        }

        return results.Distinct().ToArray();
    }

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PagedAttentionPartial target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        var results = new List<DistributedCandidateTuple>();
        foreach (var inputTypes in EnumerateInputTypes(context))
        {
            if (PagedAttentionPartialEvaluator.InferType(
                    target,
                    inputTypes[PagedAttentionPartial.Q.Index],
                    (TensorType)inputTypes[PagedAttentionPartial.KVCaches.Index],
                    inputTypes[PagedAttentionPartial.Extra.Index],
                    (TensorType)inputTypes[PagedAttentionPartial.Scale.Index],
                    (DimensionType)inputTypes[PagedAttentionPartial.LayerId.Index]) == returnType)
            {
                results.Add(new DistributedCandidateTuple(inputTypes, "paged-attention-partial-sbp"));
            }
        }

        tuples = results;
        return true;
    }

    private static IEnumerable<IRType[]> EnumerateInputTypes(DistributedCandidateContext context)
    {
        if (context.AvailableInputTypes.Count != 5)
        {
            yield break;
        }

        foreach (var query in context.AvailableInputTypes[PagedAttentionPartial.Q.Index])
        {
            foreach (var kvCaches in context.AvailableInputTypes[PagedAttentionPartial.KVCaches.Index].OfType<TensorType>())
            {
                foreach (var extra in context.AvailableInputTypes[PagedAttentionPartial.Extra.Index])
                {
                    foreach (var scale in context.AvailableInputTypes[PagedAttentionPartial.Scale.Index].OfType<TensorType>())
                    {
                        foreach (var layerId in context.AvailableInputTypes[PagedAttentionPartial.LayerId.Index].OfType<DimensionType>())
                        {
                            yield return [query, kvCaches, extra, scale, layerId];
                        }
                    }
                }
            }
        }
    }
}

internal sealed class PagedAttentionCombineCandidateProvider :
    DistributedCandidateProvider<PagedAttentionCombine>
{
    public override bool AllowsPartialInputs => true;

    public override bool IsExhaustive => true;

    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PagedAttentionCombine target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Where(returnType => IsLogicalOutputType(target, returnType))
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PagedAttentionCombine target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        var results = new List<DistributedCandidateTuple>();
        var candidateTarget = WithOutputType(target, returnType);
        if (context.AvailableInputTypes.Count != 4)
        {
            tuples = results;
            return false;
        }

        var sumStateTypes = context.AvailableInputTypes[PagedAttentionCombine.SumState.Index].ToHashSet();
        var accStateTypes = context.AvailableInputTypes[PagedAttentionCombine.AccState.Index].ToHashSet();
        var outputGateTypes = context.AvailableInputTypes[PagedAttentionCombine.OutputGate.Index]
            .Where(type => type is NoneType || type == returnType)
            .ToArray();
        foreach (var maxStateType in context.AvailableInputTypes[PagedAttentionCombine.MaxState.Index])
        {
            if (!TryCreateCompanionStateTypes(
                    candidateTarget,
                    returnType,
                    maxStateType,
                    out var sumStateType,
                    out var accStateType) ||
                !sumStateTypes.Contains(sumStateType) ||
                !accStateTypes.Contains(accStateType))
            {
                continue;
            }

            foreach (var outputGateType in outputGateTypes)
            {
                var inputTypes = new IRType[]
                {
                    maxStateType,
                    sumStateType,
                    accStateType,
                    outputGateType,
                };
                if (PagedAttentionCombineEvaluator.InferType(
                        candidateTarget,
                        maxStateType,
                        sumStateType,
                        accStateType,
                        outputGateType) == returnType)
                {
                    results.Add(new DistributedCandidateTuple(inputTypes, "paged-attention-combine-sbp"));
                }
            }
        }

        tuples = results;
        return true;
    }

    public override PagedAttentionCombine CreateCandidateTarget(
        DistributedCandidateContext context,
        PagedAttentionCombine target,
        IRType returnType)
        => WithOutputType(target, returnType);

    private static bool TryCreateCompanionStateTypes(
        PagedAttentionCombine target,
        IRType returnType,
        IRType maxStateType,
        out IRType sumStateType,
        out IRType accStateType)
    {
        sumStateType = AnyType.Default;
        accStateType = AnyType.Default;
        var outputTensorType = PagedAttentionSplitTypeUtility.GetTensorType(returnType);
        if (outputTensorType?.Shape is not RankedShape outputShape ||
            outputShape.Rank != target.Layout.Count)
        {
            return false;
        }

        var dimAxis = PagedAttentionSplitTypeUtility.GetAxis(target.Layout, AttentionDimKind.Dim);
        var scalarDimensions = outputShape.Dimensions.ToArray();
        scalarDimensions[dimAxis] = (scalarDimensions[dimAxis] *
            PagedAttentionSplitTypeUtility.GetVectorLaneCount(target.OutputDataType)).Simplify();
        var statsDimensions = scalarDimensions.ToArray();
        statsDimensions[dimAxis] = Dimension.One;
        var statsTensorType = new TensorType(DataTypes.Float32, new RankedShape(statsDimensions));
        var accTensorType = new TensorType(DataTypes.Float32, new RankedShape(scalarDimensions));
        if (PagedAttentionSplitTypeUtility.GetTensorType(maxStateType) != statsTensorType)
        {
            return false;
        }

        switch (maxStateType)
        {
            case TensorType when returnType is TensorType:
                sumStateType = statsTensorType;
                accStateType = accTensorType;
                return true;
            case DistributedType { Partial: { Op: ReduceOp.Max } partial } distributedMax
                when returnType is DistributedType:
                sumStateType = new DistributedType(
                    statsTensorType,
                    distributedMax.AxisPolicies,
                    distributedMax.Placement,
                    SBP.P(partial.Axes, ReduceOp.Sum));
                accStateType = new DistributedType(
                    accTensorType,
                    distributedMax.AxisPolicies,
                    distributedMax.Placement,
                    SBP.P(partial.Axes, ReduceOp.Sum));
                return true;
            default:
                return false;
        }
    }

    private static PagedAttentionCombine WithOutputType(
        PagedAttentionCombine target,
        IRType outputType)
        => new(
            target.Layout,
            target.HiddenSize,
            target.OutputDataType,
            outputType,
            target.SplitHierarchyAxis,
            target.SplitCount);

    private static bool IsLogicalOutputType(PagedAttentionCombine target, IRType returnType)
    {
        var tensorType = PagedAttentionSplitTypeUtility.GetTensorType(returnType);
        var expectedType = PagedAttentionSplitTypeUtility.GetTensorType(target.OutputType);
        return tensorType is not null &&
            expectedType is not null &&
            tensorType == expectedType &&
            returnType is not DistributedType { Partial: not null };
    }
}
