// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using Range = Nncase.IR.Tensors.Range;
using Reshape = Nncase.IR.Tensors.Reshape;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Range"/>.
/// </summary>
public class ReshapeEvaluator : IEvaluator<Reshape>, ITypeInferencer<Reshape>, ICostEvaluator<Reshape>, IMetricEvaluator<Reshape>
{
    public static IRType VisitDistributedType(DistributedType inType, RankedShape newShape)
    {
        var invalidType = new InvalidType($"not supported reshape {inType} to {newShape}");
        var inShape = (RankedShape)inType.TensorType.Shape;
        var maxInShape = CompilerServices.GetMaxShape(inShape);
        var maxNewShape = CompilerServices.GetMaxShape(newShape);
        if (!TryGetReshapeShapeMapMatrix(maxInShape, maxNewShape, out var mat))
        {
            if (inType.AxisPolicies.All(x => x is SBPBroadCast))
            {
                // If all axes are broadcast, we can still reshape.
                return inType with
                {
                    TensorType = inType.TensorType with { Shape = newShape },
                    AxisPolicies = Enumerable.Repeat(SBP.B, newShape.Rank).ToArray(),
                };
            }

            return invalidType;
        }

        var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsCompleteDict(mat);
        var newAxisPolicies = new SBP[newShape.Rank];

        // 1. [1024@t] -> [8@t, 128]
        foreach ((var inAxis, var newAxes) in forwardDict)
        {
            var inPolicy = inType.AxisPolicies[inAxis];
            if (inPolicy is SBPSplit split)
            {
                var newAxesOffset = newAxes[0];
                var newDims = newAxes.Select(newAxis => newShape[newAxis]).ToArray().AsReadOnlySpan();
                var newSplitAxis = newDims.FirstIndexOfNotEqual(1);
                newSplitAxis = newSplitAxis < 0 ? 0 : newSplitAxis;
                if (newDims[newSplitAxis] != inShape[inAxis]
                    && !Dimension.TryDivExactly(newDims[newSplitAxis], DistributedUtility.GetDivisor(split, inType.Placement), out _))
                {
                    return invalidType;
                }

                var reshapedSplit = split;
                if (newDims[newSplitAxis] != inShape[inAxis])
                {
                    var unitDivisor = newAxes
                        .Except([newAxesOffset + newSplitAxis])
                        .Aggregate(1L, (product, axis) => checked(product * maxNewShape[axis]));
                    if (!DistributedUtility.TryScaleSplitUnits(split, 1, unitDivisor, out reshapedSplit))
                    {
                        return invalidType;
                    }
                }

                foreach (var newAxis in newAxes)
                {
                    newAxisPolicies[newAxis] = newAxis == (newAxesOffset + newSplitAxis) ? reshapedSplit : SBP.B;
                }
            }
            else
            {
                foreach (var newAxis in newAxes)
                {
                    newAxisPolicies[newAxis] = inPolicy;
                }
            }
        }

        // 2. [8@t, 128] -> [1024@t]
        foreach ((var newAxis, var inAxes) in backwardDict)
        {
            if (newAxisPolicies[newAxis] is not null)
            {
                continue; // already set
            }

            var splitAxes = (from inAxis in inAxes
                             let inPolicy = inType.AxisPolicies[inAxis]
                             where inPolicy is SBPSplit
                             select inAxis).ToArray();
            if (splitAxes.Length > 1)
            {
                if (!TryFlattenBlockCyclicSplits(
                        inType,
                        inAxes,
                        maxInShape,
                        out var flattenedSplit))
                {
                    return invalidType;
                }

                newAxisPolicies[newAxis] = flattenedSplit;
                continue;
            }

            var firstSplitAxis = splitAxes.Cast<int?>().FirstOrDefault();
            if (firstSplitAxis is not null)
            {
                // Either the axis is the first axis or all of the dimensions before it are 1.
                if (firstSplitAxis != inAxes[0]
                    && inAxes.TakeWhile(a => a < firstSplitAxis).Any(a => inShape[a] != 1))
                {
                    return invalidType;
                }

                var split = (SBPSplit)inType.AxisPolicies[firstSplitAxis.Value];
                var unitMultiplier = inAxes
                    .Except([firstSplitAxis.Value])
                    .Aggregate(1L, (product, axis) => checked(product * maxInShape[axis]));
                if (!DistributedUtility.TryScaleSplitUnits(split, unitMultiplier, 1, out var reshapedSplit))
                {
                    return invalidType;
                }

                newAxisPolicies[newAxis] = reshapedSplit;
            }
            else
            {
                newAxisPolicies[newAxis] = SBP.B; // no split axis, use B
            }
        }

        if (newAxisPolicies.Any(a => a is null))
        {
            var mappedInputAxes = forwardDict.Keys
                .Concat(backwardDict.Values.SelectMany(axes => axes))
                .ToHashSet();
            if (inType.AxisPolicies
                .Select((policy, axis) => (policy, axis))
                .Where(item => !mappedInputAxes.Contains(item.axis))
                .All(item => item.policy is SBPBroadCast))
            {
                // If all axes that are not in the forward mapping are broadcast, we can still reshape.
                for (int i = 0; i < newAxisPolicies.Length; i++)
                {
                    if (newAxisPolicies[i] is null)
                    {
                        newAxisPolicies[i] = SBP.B;
                    }
                }
            }
            else
            {
                // If there are axes that are not in the forward mapping and they are not broadcast, we cannot reshape.
                return invalidType;
            }
        }

        if (!DistributedUtility.IsDistributable(newAxisPolicies))
        {
            return invalidType;
        }

        return new DistributedType(inType.TensorType with { Shape = newShape }, newAxisPolicies, inType.Placement);
    }

    private static bool TryGetReshapeShapeMapMatrix(
        long[] inputShape,
        long[] outputShape,
        out int[,] matrix)
    {
        if (!inputShape.Contains(1L) && !outputShape.Contains(1L))
        {
            return IRUtility.TryGetShapeMapMatrix(inputShape, outputShape, out matrix);
        }

        var inputAxes = inputShape
            .Select((extent, axis) => (extent, axis))
            .Where(item => item.extent != 1)
            .ToArray();
        var outputAxes = outputShape
            .Select((extent, axis) => (extent, axis))
            .Where(item => item.extent != 1)
            .ToArray();
        var reducedInputShape = inputAxes.Select(item => item.extent).ToArray();
        var reducedOutputShape = outputAxes.Select(item => item.extent).ToArray();
        if (reducedInputShape.Length == 0 && reducedOutputShape.Length == 0)
        {
            matrix = new int[outputShape.Length, inputShape.Length];
            return true;
        }

        if (reducedInputShape.Length == 0 || reducedOutputShape.Length == 0)
        {
            return IRUtility.TryGetShapeMapMatrix(inputShape, outputShape, out matrix);
        }

        if (IRUtility.TryGetShapeMapMatrix(
                reducedInputShape,
                reducedOutputShape,
                out var reducedMatrix))
        {
            matrix = new int[outputShape.Length, inputShape.Length];
            for (var outputIndex = 0; outputIndex < outputAxes.Length; outputIndex++)
            {
                for (var inputIndex = 0; inputIndex < inputAxes.Length; inputIndex++)
                {
                    matrix[outputAxes[outputIndex].axis, inputAxes[inputIndex].axis] =
                        reducedMatrix[outputIndex, inputIndex];
                }
            }

            return true;
        }

        return IRUtility.TryGetShapeMapMatrix(inputShape, outputShape, out matrix);
    }

    private static bool TryFlattenBlockCyclicSplits(
        DistributedType inputType,
        IReadOnlyList<int> inputAxes,
        IReadOnlyList<long> inputShape,
        out SBPSplit flattenedSplit)
    {
        var stages = new List<SplitStage>();
        for (var position = 0; position < inputAxes.Count; position++)
        {
            var inputAxis = inputAxes[position];
            if (inputType.AxisPolicies[inputAxis] is not SBPSplit split)
            {
                continue;
            }

            var parentExtent = inputShape[inputAxis];
            foreach (var stage in split.Stages)
            {
                if (stage.Distribution is not BlockCyclicSplit blockCyclic)
                {
                    flattenedSplit = null!;
                    return false;
                }

                var shardCount = stage.HierarchyAxes.Aggregate(
                    1L,
                    (product, hierarchyAxis) => checked(
                        product * inputType.Placement.Hierarchy[hierarchyAxis]));
                var period = checked(shardCount * blockCyclic.BlockSize);
                if (parentExtent % period != 0)
                {
                    flattenedSplit = null!;
                    return false;
                }

                parentExtent /= shardCount;
            }

            var trailingExtent = inputAxes
                .Skip(position + 1)
                .Aggregate(1L, (product, axis) => checked(product * inputShape[axis]));
            if (!DistributedUtility.TryScaleSplitUnits(
                    split,
                    trailingExtent,
                    1,
                    out var scaledSplit))
            {
                flattenedSplit = null!;
                return false;
            }

            stages.AddRange(scaledSplit.Stages);
        }

        if (stages.Count == 0)
        {
            flattenedSplit = null!;
            return false;
        }

        try
        {
            flattenedSplit = SBP.S(stages.ToArray());
            return true;
        }
        catch (ArgumentException)
        {
            flattenedSplit = null!;
            return false;
        }
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Reshape reshape)
    {
        OrtKISharp.Tensor input;

        var inputOrg = context.GetArgumentValue(reshape, Reshape.Input).AsTensor();
        var dataType = inputOrg.ElementType;
        if (dataType is VectorType { ElemType: DataType dataTypes } vType && dataTypes != DataTypes.Float32)
        {
            var interType = new VectorType(DataTypes.Float32, vType.Lanes);
            input = Nncase.IR.F.Tensors.Cast(inputOrg, interType).Evaluate().AsTensor().ToOrtTensor();
        }
        else if (dataType is not VectorType && dataType.IsFloat() && dataType != DataTypes.Float32)
        {
            input = Nncase.IR.F.Tensors.Cast(inputOrg, DataTypes.Float32).Evaluate().AsTensor().ToOrtTensor();
        }
        else
        {
            input = context.GetOrtArgumentValue(reshape, Reshape.Input);
        }

        var shape = context.GetArgumentValueAsArray<long>(reshape, Reshape.Shape);
        if (context.CurrentCall.CheckedType is AnyType)
        {
            return Value.FromTensor(OrtKI.Reshape(input, shape, 0).ToTensor());
        }

        var tensorType = context.CurrentCall.CheckedTensorType;
        var allowzero = tensorType.Shape is RankedShape rankedShape && rankedShape.Contains(0) ? 1L : 0L;
        if (tensorType.DType is VectorType vtype)
        {
            shape = shape.Concat(vtype.Lanes.Select(i => (long)i)).ToArray();
        }

        var reshaped = OrtKI.Reshape(input, shape, allowzero);
        if (dataType is not VectorType && dataType != DataTypes.Float32)
        {
            reshaped = OrtKI.Cast(OrtKI.Reshape(input, shape, allowzero), (int)dataType.ToOrtType());
        }

        return reshaped.ToValue(dataType);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Reshape target)
    {
        var input = context.CheckArgumentType<IRType>(target, Reshape.Input);
        return input switch
        {
            TensorType tensorType => Visit(context, target, tensorType),
            DistributedType distributedType => Visit(context, target, distributedType),
            AnyType => AnyType.Default,
            InvalidType => input,
            _ => new InvalidType($"Not Support Input Type {input.GetType().Name}"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, Reshape target)
    {
        return CostUtility.GetReshapeCost();
    }

    Cost ICostEvaluator<Reshape>.Visit(ICostEvaluateContext context, Reshape target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Reshape target)
    {
        return Metric.Zero;
    }

    private IRType Visit(ITypeInferenceContext context, Reshape target, TensorType input)
    {
        var shape = (Shape)context.GetArgument(target, Reshape.Shape);
        var outShape = TypeInference.ReshapeShape(input.Shape, shape);
        return input with { Shape = outShape };
    }

    private IRType Visit(ITypeInferenceContext context, Reshape target, DistributedType inputType)
    {
        var outType = Visit(context, target, inputType.TensorType);
        if (outType is not TensorType outTensorType)
        {
            return outType;
        }

        var invalid = new InvalidType(inputType.ToString());
        if (outTensorType.Shape.IsUnranked)
        {
            return invalid;
        }

        return VisitDistributedType(inputType, (RankedShape)outTensorType.Shape);
    }
}
