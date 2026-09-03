// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

public sealed class PagedAttentionPartialEvaluator :
    IEvaluator<PagedAttentionPartial>,
    ITypeInferencer<PagedAttentionPartial>,
    ICostEvaluator<PagedAttentionPartial>
{
    public IValue Visit(IEvaluateContext context, PagedAttentionPartial target)
    {
        var query = context.GetOrtArgumentValue(target, PagedAttentionPartial.Q);
        var kvCaches = context.GetArgumentValueAsTensor<Reference<IPagedAttentionKVCache>>(
            target,
            PagedAttentionPartial.KVCaches);
        var scale = context.GetOrtArgumentValue(target, PagedAttentionPartial.Scale);
        var layerId = checked((int)context
            .GetArgumentValue(target, PagedAttentionPartial.LayerId)
            .AsTensor()
            .ToScalar<long>());
        var attention = PagedAttentionEvaluator.RefPagedAttn(
            query,
            kvCaches,
            scale,
            layerId,
            target.Layout);
        var dimAxis = PagedAttentionSplitTypeUtility.GetAxis(target.Layout, AttentionDimKind.Dim);
        var queryDataType = context.GetArgumentValue(target, PagedAttentionPartial.Q).AsTensor().ElementType;
        var scalarAttention = PagedAttentionSplitTypeUtility.UnpackToScalar(
            attention,
            queryDataType,
            dimAxis).Cast(OrtDataType.Float);
        var stateType = (TupleType)context.GetReturnType();
        var maxType = PagedAttentionSplitTypeUtility.GetTensorType(stateType[0])!;
        var sumType = PagedAttentionSplitTypeUtility.GetTensorType(stateType[1])!;
        var accType = PagedAttentionSplitTypeUtility.GetTensorType(stateType[2])!;
        var maxShape = maxType.Shape.ToValueArray();
        var maxState = OrtKI.Expand(
            OrtKISharp.Tensor.FromScalar(0.0F),
            OrtKISharp.Tensor.MakeTensor(maxShape));
        var sumState = OrtKI.Expand(
            OrtKISharp.Tensor.FromScalar(1.0F),
            OrtKISharp.Tensor.MakeTensor(maxShape));

        return new TupleValue([
            maxState.ToValue(maxType),
            sumState.ToValue(sumType),
            scalarAttention.ToValue(accType),
        ]);
    }

    public IRType Visit(ITypeInferenceContext context, PagedAttentionPartial target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, PagedAttentionPartial.Q),
            context.CheckArgumentType<TensorType>(target, PagedAttentionPartial.KVCaches),
            context.CheckArgumentType<IRType>(target, PagedAttentionPartial.Extra),
            context.CheckArgumentType<TensorType>(target, PagedAttentionPartial.Scale),
            context.CheckArgumentType<DimensionType>(target, PagedAttentionPartial.LayerId));

    public static IRType InferType(
        PagedAttentionPartial target,
        IRType query,
        TensorType kvCaches,
        IRType extra,
        TensorType scale,
        DimensionType layerId)
    {
        _ = layerId;
        var attentionType = PagedAttentionEvaluator.InferType(
            new PagedAttention(target.Layout, target.HiddenSize),
            query,
            extra,
            scale,
            kvCaches,
            NoneType.Default);
        if (attentionType is InvalidType)
        {
            return attentionType;
        }

        return PagedAttentionSplitTypeUtility.CreatePartialStateType(
            attentionType,
            target.Layout,
            target.HiddenSize,
            target.SplitHierarchyAxis,
            target.SplitCount);
    }

    public Cost Visit(ICostEvaluateContext context, PagedAttentionPartial target)
    {
        var queryType = context.GetArgumentType<IRType>(target, PagedAttentionPartial.Q);
        var extraType = context.GetArgumentType<IRType>(target, PagedAttentionPartial.Extra);
        var kvCachesType = context.GetArgumentType<TensorType>(target, PagedAttentionPartial.KVCaches);
        var returnType = context.GetReturnType<IRType>();
        var cost = new Cost();
        Add(cost, CostFactorNames.BlockLocalMemoryLoadBytes, CostUtility.GetMemoryAccess(queryType));
        Add(cost, CostFactorNames.BlockLocalMemoryStoreBytes, GetTupleMemoryAccess(returnType));
        if (PagedAttentionExecutionPlanQuery.TryCreate(
                queryType,
                extraType,
                kvCachesType,
                target.Layout,
                target.HiddenSize,
                out var query))
        {
            Add(
                cost,
                CostFactorNames.BlockLocalMemoryLoadBytes,
                ToCostFactor(query.KVScalarElements * query.KVElementSizeBytes / target.SplitCount));
            Add(
                cost,
                CostFactorNames.CPUCycles,
                ToCostFactor(query.ComputeWork / target.SplitCount));
        }

        return cost;
    }

    private static UInt128 GetTupleMemoryAccess(IRType type)
        => type is TupleType tuple
            ? tuple.Fields.Aggregate((UInt128)0, (sum, field) => sum + CostUtility.GetMemoryAccess(field))
            : CostUtility.GetMemoryAccess(type);

    private static UInt128 ToCostFactor(double value)
        => value <= 0 || !double.IsFinite(value)
            ? 0
            : (UInt128)(ulong)System.Math.Ceiling(System.Math.Min(value, ulong.MaxValue));

    private static void Add(Cost cost, string factor, UInt128 value)
    {
        if (value == 0)
        {
            return;
        }

        cost.Factors[factor] = cost.Factors.TryGetValue(factor, out var oldValue)
            ? oldValue + value
            : value;
    }
}

public sealed class PagedAttentionCombineEvaluator :
    IEvaluator<PagedAttentionCombine>,
    ITypeInferencer<PagedAttentionCombine>,
    ICostEvaluator<PagedAttentionCombine>
{
    public IValue Visit(IEvaluateContext context, PagedAttentionCombine target)
    {
        var maxState = context.GetOrtArgumentValue(target, PagedAttentionCombine.MaxState);
        var sumState = context.GetOrtArgumentValue(target, PagedAttentionCombine.SumState);
        var accState = context.GetOrtArgumentValue(target, PagedAttentionCombine.AccState);
        var outputGate = context.GetArgumentValue(target, PagedAttentionCombine.OutputGate);
        return Evaluate(target, maxState, sumState, accState, outputGate, context.GetReturnType());
    }

    internal static IValue Evaluate(
        PagedAttentionCombine target,
        OrtKISharp.Tensor maxState,
        OrtKISharp.Tensor sumState,
        OrtKISharp.Tensor accState,
        IValue outputGate,
        IRType returnType)
    {
        _ = maxState;
        var output = accState / sumState;
        var dimAxis = PagedAttentionSplitTypeUtility.GetAxis(target.Layout, AttentionDimKind.Dim);
        output = output.Cast(PagedAttentionSplitTypeUtility.GetScalarDataType(target.OutputDataType).ToOrtType());
        if (outputGate is not NoneValue && outputGate.Type is not NoneType)
        {
            var scalarGate = PagedAttentionSplitTypeUtility.UnpackToScalar(
                outputGate.AsTensor().ToOrtTensor(),
                target.OutputDataType,
                dimAxis);
            output *= OrtKI.Sigmoid(scalarGate);
        }

        output = PagedAttentionSplitTypeUtility.PackFromScalar(output, target.OutputDataType, dimAxis);
        return output.ToValue(returnType);
    }

    public IRType Visit(ITypeInferenceContext context, PagedAttentionCombine target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, PagedAttentionCombine.MaxState),
            context.CheckArgumentType<IRType>(target, PagedAttentionCombine.SumState),
            context.CheckArgumentType<IRType>(target, PagedAttentionCombine.AccState),
            context.CheckArgumentType<IRType>(target, PagedAttentionCombine.OutputGate));

    public static IRType InferType(
        PagedAttentionCombine target,
        IRType maxState,
        IRType sumState,
        IRType accState,
        IRType outputGate)
    {
        var output = PagedAttentionSplitTypeUtility.CreateCombineOutputType(
            maxState,
            sumState,
            accState,
            target.Layout,
            target.HiddenSize,
            target.OutputDataType,
            target.OutputType,
            target.SplitHierarchyAxis,
            target.SplitCount);
        if (output is InvalidType || outputGate is NoneType)
        {
            return output;
        }

        return outputGate == output
            ? output
            : new InvalidType(
                $"PagedAttentionCombine output gate must have exactly the output type, " +
                $"got output={output}, gate={outputGate}.");
    }

    public Cost Visit(ICostEvaluateContext context, PagedAttentionCombine target)
    {
        var maxState = context.GetArgumentType<IRType>(target, PagedAttentionCombine.MaxState);
        var outputGate = context.GetArgumentType<IRType>(target, PagedAttentionCombine.OutputGate);
        var output = context.GetReturnType<IRType>();
        var localOutput = output switch
        {
            DistributedType distributed => DistributedUtility.GetDividedTensorType(
                distributed,
                DistributedUtility.DivideFlags.MaxShape),
            TensorType tensor => tensor,
            _ => throw new InvalidOperationException(
                $"PagedAttentionCombine cost requires a tensor output, got {output}."),
        };
        if (localOutput.Shape is not RankedShape localShape || localShape.Rank != target.Layout.Count)
        {
            throw new InvalidOperationException(
                $"PagedAttentionCombine cost requires a ranked local output, got {localOutput}.");
        }

        var dimAxis = PagedAttentionSplitTypeUtility.GetAxis(target.Layout, AttentionDimKind.Dim);
        var scalarDimensions = localShape.Dimensions.ToArray();
        scalarDimensions[dimAxis] = (scalarDimensions[dimAxis] *
            PagedAttentionSplitTypeUtility.GetVectorLaneCount(localOutput.DType)).Simplify();
        var statsDimensions = scalarDimensions.ToArray();
        statsDimensions[dimAxis] = Dimension.One;
        var localStatsType = new TensorType(DataTypes.Float32, new RankedShape(statsDimensions));
        var localAccType = new TensorType(DataTypes.Float32, new RankedShape(scalarDimensions));
        var fanIn = maxState is DistributedType { Partial: not null }
            ? checked((UInt128)target.SplitCount)
            : (UInt128)1;
        var stateBytes = checked(
            (CostUtility.GetMemoryAccess(localStatsType) * 2 +
             CostUtility.GetMemoryAccess(localAccType)) * fanIn);
        var cost = new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = stateBytes,
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = stateBytes / sizeof(float),
        };
        if (outputGate is not NoneType)
        {
            var gateBytes = CostUtility.GetMemoryAccess(outputGate);
            cost[CostFactorNames.BlockLocalMemoryLoadBytes] += gateBytes;
            cost[CostFactorNames.CPUCycles] += gateBytes /
                (UInt128)System.Math.Max(1, localOutput.DType.SizeInBytes) * 3;
        }

        return cost;
    }
}

internal static class PagedAttentionSplitTypeUtility
{
    public static IRType CreatePartialStateType(
        IRType attentionType,
        IRArray<AttentionDimKind> layout,
        int hiddenSize,
        int splitHierarchyAxis,
        int splitCount)
    {
        if (!TryGetLayoutAxes(layout, out _, out var headAxis, out var dimAxis, out var reason) ||
            splitCount <= 1)
        {
            return new InvalidType(reason ?? $"PagedAttentionPartial SplitCount must be greater than one, got {splitCount}.");
        }

        var attentionTensor = GetTensorType(attentionType);
        if (attentionTensor?.Shape is not RankedShape shape || shape.Rank != layout.Count)
        {
            return new InvalidType($"PagedAttentionPartial requires a ranked attention tensor, got {attentionType}.");
        }

        var laneCount = GetVectorLaneCount(attentionTensor.DType);
        var scalarDimensions = shape.Dimensions.ToArray();
        scalarDimensions[dimAxis] = (scalarDimensions[dimAxis] * laneCount).Simplify();
        if ((scalarDimensions[headAxis] * scalarDimensions[dimAxis]).Simplify() is { IsFixed: true } hidden &&
            hidden.FixedValue != hiddenSize)
        {
            return new InvalidType(
                $"PagedAttentionPartial hidden extent {hidden.FixedValue} does not match HiddenSize {hiddenSize}.");
        }

        var scalarStateDimensions = scalarDimensions.ToArray();
        scalarStateDimensions[dimAxis] = Dimension.One;
        var maxType = new TensorType(DataTypes.Float32, new RankedShape(scalarStateDimensions));
        var accType = new TensorType(DataTypes.Float32, new RankedShape(scalarDimensions));
        if (attentionType is TensorType)
        {
            return new TupleType([maxType, maxType, accType]);
        }

        if (attentionType is not DistributedType distributed ||
            distributed.Partial is not null ||
            distributed.AxisPolicies.Count != shape.Rank ||
            distributed.AxisPolicies.Any(policy => policy is SBPPartial) ||
            splitHierarchyAxis < 0 ||
            splitHierarchyAxis >= distributed.Placement.Rank ||
            !distributed.Placement.IsPhysicalBlockAxis(splitHierarchyAxis) ||
            distributed.Placement.Hierarchy[splitHierarchyAxis] != splitCount ||
            distributed.AxisPolicies.Any(policy => UsesHierarchyAxis(policy, splitHierarchyAxis)))
        {
            return new InvalidType(
                $"PagedAttentionPartial requires an unused physical block hierarchy axis " +
                $"{splitHierarchyAxis} with extent {splitCount}, got {attentionType}.");
        }

        return new TupleType([
            new DistributedType(
                maxType,
                distributed.AxisPolicies,
                distributed.Placement,
                SBP.P([splitHierarchyAxis], ReduceOp.Max)),
            new DistributedType(
                maxType,
                distributed.AxisPolicies,
                distributed.Placement,
                SBP.P([splitHierarchyAxis], ReduceOp.Sum)),
            new DistributedType(
                accType,
                distributed.AxisPolicies,
                distributed.Placement,
                SBP.P([splitHierarchyAxis], ReduceOp.Sum)),
        ]);
    }

    public static IRType CreateCombineOutputType(
        IRType maxState,
        IRType sumState,
        IRType accState,
        IRArray<AttentionDimKind> layout,
        int hiddenSize,
        DataType outputDataType,
        IRType outputType,
        int splitHierarchyAxis,
        int splitCount)
    {
        if (!TryGetLayoutAxes(layout, out _, out var headAxis, out var dimAxis, out var reason))
        {
            return new InvalidType(reason!);
        }

        var maxTensor = GetTensorType(maxState);
        var sumTensor = GetTensorType(sumState);
        var accTensor = GetTensorType(accState);
        if (maxTensor?.Shape is not RankedShape maxShape ||
            sumTensor?.Shape is not RankedShape sumShape ||
            accTensor?.Shape is not RankedShape accShape ||
            maxTensor.DType != DataTypes.Float32 ||
            sumTensor.DType != DataTypes.Float32 ||
            accTensor.DType != DataTypes.Float32 ||
            maxShape.Rank != layout.Count ||
            maxShape != sumShape ||
            accShape.Rank != maxShape.Rank ||
            maxShape[dimAxis] != 1 ||
            maxShape.Dimensions.ToArray().Where((_, axis) => axis != dimAxis).Zip(
                accShape.Dimensions.ToArray().Where((_, axis) => axis != dimAxis)).Any(
                    pair => pair.First != pair.Second))
        {
            return new InvalidType(
                "PagedAttentionCombine requires compatible FP32 max, sum, and accumulator state tensors.");
        }

        var outputDimensions = accShape.Dimensions.ToArray();
        var laneCount = GetVectorLaneCount(outputDataType);
        if (laneCount <= 0 ||
            !outputDimensions[dimAxis].IsFixed ||
            outputDimensions[dimAxis].FixedValue % laneCount != 0)
        {
            return new InvalidType(
                $"PagedAttentionCombine Dim extent {outputDimensions[dimAxis]} is not divisible by output lanes {laneCount}.");
        }

        outputDimensions[dimAxis] = outputDimensions[dimAxis].FixedValue / laneCount;
        if ((outputDimensions[headAxis] * outputDimensions[dimAxis] * laneCount).Simplify() is { IsFixed: true } hidden &&
            hidden.FixedValue != hiddenSize)
        {
            return new InvalidType(
                $"PagedAttentionCombine hidden extent {hidden.FixedValue} does not match HiddenSize {hiddenSize}.");
        }

        var outputTensor = new TensorType(outputDataType, new RankedShape(outputDimensions));
        if (maxState is TensorType && sumState is TensorType && accState is TensorType)
        {
            return outputType == outputTensor
                ? outputTensor
                : new InvalidType(
                    $"PagedAttentionCombine output contract {outputType} does not match {outputTensor}.");
        }

        if (maxState is not DistributedType distributedMax ||
            sumState is not DistributedType distributedSum ||
            accState is not DistributedType distributedAcc ||
            distributedMax.Placement != distributedSum.Placement ||
            distributedMax.Placement != distributedAcc.Placement ||
            distributedMax.Partial is not { Op: ReduceOp.Max } maxPartial ||
            distributedSum.Partial is not { Op: ReduceOp.Sum } sumPartial ||
            distributedAcc.Partial is not { Op: ReduceOp.Sum } accPartial ||
            !maxPartial.Axes.SequenceEqual(sumPartial.Axes) ||
            !maxPartial.Axes.SequenceEqual(accPartial.Axes) ||
            !maxPartial.Axes.SequenceEqual(new[] { splitHierarchyAxis }) ||
            !distributedMax.AxisPolicies.SequenceEqual(distributedSum.AxisPolicies) ||
            !distributedMax.AxisPolicies.SequenceEqual(distributedAcc.AxisPolicies) ||
            distributedMax.AxisPolicies.Count != maxShape.Rank ||
            distributedMax.AxisPolicies.Any(policy => UsesHierarchyAxis(policy, splitHierarchyAxis)) ||
            splitHierarchyAxis < 0 ||
            splitHierarchyAxis >= distributedMax.Placement.Rank ||
            distributedMax.Placement.Hierarchy[splitHierarchyAxis] != splitCount)
        {
            return new InvalidType(
                "PagedAttentionCombine requires matching FP32 P(Max)/P(Sum)/P(Sum) states on one placement.");
        }

        if (outputType is not DistributedType distributedOutput ||
            distributedOutput.TensorType != outputTensor ||
            distributedOutput.Placement != distributedMax.Placement ||
            distributedOutput.Partial is not null ||
            distributedOutput.AxisPolicies.Count != outputTensor.Shape.Rank ||
            distributedOutput.AxisPolicies.Any(policy => policy is SBPPartial) ||
            !CanCombineTo(distributedAcc, distributedOutput, splitHierarchyAxis))
        {
            return new InvalidType(
                $"PagedAttentionCombine cannot discharge P([{splitHierarchyAxis}]) into {outputType}.");
        }

        return distributedOutput;
    }

    public static bool CanCombineTo(
        DistributedType partialState,
        DistributedType outputType,
        int splitHierarchyAxis)
    {
        if (partialState.Placement != outputType.Placement ||
            partialState.Partial is not { } partial ||
            !partial.Axes.SequenceEqual(new[] { splitHierarchyAxis }) ||
            outputType.Partial is not null ||
            partialState.AxisPolicies.Count != outputType.AxisPolicies.Count)
        {
            return false;
        }

        var inputHierarchy = DistributedUtility.GetHierarchyAxisPolicies(
            partialState.AxisPolicies,
            partialState.Placement.Rank);
        var outputHierarchy = DistributedUtility.GetHierarchyAxisPolicies(
            outputType.AxisPolicies,
            outputType.Placement.Rank);
        for (var axis = 0; axis < inputHierarchy.Count; axis++)
        {
            if (axis == splitHierarchyAxis)
            {
                if (inputHierarchy[axis] is not HierarchyAxisBroadcast ||
                    outputHierarchy[axis] is not (HierarchyAxisBroadcast or HierarchyAxisSplit))
                {
                    return false;
                }

                continue;
            }

            if (inputHierarchy[axis] != outputHierarchy[axis])
            {
                return false;
            }
        }

        return true;
    }

    public static int GetAxis(IRArray<AttentionDimKind> layout, AttentionDimKind kind)
        => layout.IndexOf(kind) is var axis && axis >= 0
            ? axis
            : throw new InvalidOperationException(
                $"Attention layout [{string.Join(',', layout)}] does not contain {kind}.");

    public static DataType GetScalarDataType(DataType dataType)
        => dataType is VectorType vector ? GetScalarDataType(vector.ElemType) : dataType;

    public static OrtKISharp.Tensor UnpackToScalar(
        OrtKISharp.Tensor tensor,
        DataType dataType,
        int dimAxis)
    {
        if (dataType is not VectorType vector)
        {
            return tensor;
        }

        tensor = tensor.Unpack(
            vector.Lanes.Count,
            Enumerable.Repeat(dimAxis, vector.Lanes.Count).ToArray());
        return UnpackToScalar(tensor, vector.ElemType, dimAxis);
    }

    public static OrtKISharp.Tensor PackFromScalar(
        OrtKISharp.Tensor tensor,
        DataType dataType,
        int dimAxis)
    {
        if (dataType is not VectorType vector)
        {
            return tensor;
        }

        tensor = PackFromScalar(tensor, vector.ElemType, dimAxis);
        return tensor.Pack(
            0,
            vector.Lanes,
            Enumerable.Repeat(dimAxis, vector.Lanes.Count).ToArray());
    }

    private static bool TryGetLayoutAxes(
        IRArray<AttentionDimKind> layout,
        out int seqAxis,
        out int headAxis,
        out int dimAxis,
        out string? reason)
    {
        seqAxis = layout.IndexOf(AttentionDimKind.Seq);
        headAxis = layout.IndexOf(AttentionDimKind.Head);
        dimAxis = layout.IndexOf(AttentionDimKind.Dim);
        reason = null;
        if (layout.Count == 3 &&
            layout.Distinct().Count() == 3 &&
            seqAxis >= 0 &&
            headAxis >= 0 &&
            dimAxis >= 0)
        {
            return true;
        }

        reason = $"Paged attention requires one Seq, Head, and Dim axis, got [{string.Join(',', layout)}].";
        return false;
    }

    public static TensorType? GetTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => distributed.TensorType,
        _ => null,
    };

    public static int GetVectorLaneCount(DataType dataType)
        => dataType is VectorType vector
            ? vector.Lanes.Aggregate(1, static (product, lane) => checked(product * lane)) *
                GetVectorLaneCount(vector.ElemType)
            : 1;

    private static bool UsesHierarchyAxis(SBP policy, int hierarchyAxis) => policy switch
    {
        SBPSplit split => split.HierarchyAxes.Contains(hierarchyAxis),
        SBPPartial partial => partial.Axes.Contains(hierarchyAxis),
        _ => false,
    };
}
