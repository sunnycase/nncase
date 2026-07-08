// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.Evaluator.NN;

internal static class NormUtility
{
    public static int NormalizeAxis(int axis, int rank)
    {
        var normalizedAxis = axis < 0 ? axis + rank : axis;
        if (normalizedAxis < 0 || normalizedAxis >= rank)
        {
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for rank {rank}.");
        }

        return normalizedAxis;
    }

    public static TensorType GetStatsTensorType(TensorType input, int axis, bool useMean)
    {
        if (!input.DType.IsFloat())
        {
            return TensorType.Invalid(DataTypes.Float32);
        }

        var statsDType = DataTypes.Float32;
        if (input.Shape.IsUnranked)
        {
            return TensorType.Unranked(statsDType);
        }

        if (input.Shape is not RankedShape shape || shape.Rank == 0)
        {
            return TensorType.Invalid(statsDType);
        }

        var normalizedAxis = NormalizeAxis(axis, shape.Rank);
        var statsShape = new Dimension[shape.Rank + 1];
        statsShape[0] = useMean ? 2 : 1;
        for (int i = 0; i < shape.Rank; i++)
        {
            statsShape[i + 1] = i < normalizedAxis ? shape[i] : 1;
        }

        return new TensorType(statsDType, new RankedShape(statsShape));
    }

    public static TensorType GetStatsTensorType(DistributedType input, int axis, bool useMean)
        => GetStatsTensorType(input.TensorType, axis, useMean);

    public static long GetNormalizationSize(TensorType input, int axis, long fallback)
    {
        if (input.Shape is not RankedShape shape || !shape.IsFixed)
        {
            return fallback;
        }

        var normalizedAxis = NormalizeAxis(axis, shape.Rank);
        long size = 1;
        for (int i = normalizedAxis; i < shape.Rank; i++)
        {
            size = checked(size * shape[i].FixedValue);
        }

        return input.DType is VectorType vectorType
            ? checked(size * vectorType.Lanes.Aggregate(1L, static (acc, lane) => acc * lane))
            : size;
    }

    public static bool HasPartial(DistributedType distributedType)
        => distributedType.Partial is not null || distributedType.AxisPolicies.Any(policy => policy is SBPPartial);

    public static bool IsSamePolicy(SBP a, SBP b)
        => DistributedUtility.IsSamePolicy(a, b, checkGranularity: false);

    public static bool IsParameterPolicyCompatible(SBP inputPolicy, SBP parameterPolicy)
        => IsSamePolicy(inputPolicy, parameterPolicy);
}
