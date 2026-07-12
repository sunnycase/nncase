// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.Tensors;

/// <summary>
/// Shared type and index rules for storage-preserving bitcasts.
/// </summary>
public static class BitcastUtility
{
    public static IRType InferType(IRType inputType, DataType newType)
    {
        return inputType switch
        {
            TensorType { Shape: RankedShape } tensorType => InferTensorType(tensorType, newType),
            DistributedType { TensorType.Shape: RankedShape } distributedType => InferDistributedType(distributedType, newType),
            _ => new InvalidType(inputType.GetType().ToString()),
        };
    }

    public static TensorType InferTensorType(TensorType inputType, DataType newType)
    {
        if (inputType.Shape is not RankedShape rankedInputShape)
        {
            throw new InvalidOperationException($"Bitcast requires a ranked input shape, got {inputType.Shape}.");
        }

        var dimensions = rankedInputShape.Dimensions.ToArray();
        if (inputType.DType.SizeInBytes != newType.SizeInBytes)
        {
            if (dimensions.Length == 0)
            {
                dimensions = [inputType.DType.SizeInBytes / newType.SizeInBytes];
            }
            else
            {
                dimensions[^1] = (dimensions[^1] * inputType.DType.SizeInBytes / newType.SizeInBytes).Simplify();
            }
        }

        return new TensorType(newType, dimensions);
    }

    private static IRType InferDistributedType(DistributedType inputType, DataType newType)
    {
        var outputTensorType = InferTensorType(inputType.TensorType, newType);
        var invalid = new InvalidType(inputType.ToString());
        var axisPolicies = new SBP[inputType.AxisPolicies.Count];
        for (var index = 0; index < axisPolicies.Length; index++)
        {
            if (inputType.AxisPolicies[index] is SBPPartial)
            {
                return invalid;
            }

            if (inputType.AxisPolicies[index] is SBPSplit split)
            {
                var granularity = split.Granularity is null
                    ? null
                    : split.Granularity * outputTensorType.Shape[index] / inputType.TensorType.Shape[index];
                axisPolicies[index] = SBP.S(split.Axes, granularity);
            }
            else
            {
                axisPolicies[index] = inputType.AxisPolicies[index];
            }
        }

        return new DistributedType(outputTensorType, axisPolicies, inputType.Placement);
    }
}
