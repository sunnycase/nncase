// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    private static AffineRange[] BuildPrefixFullTileRanges(AffineDomain[] domains, Shape shape, int prefixRank)
    {
        var ranges = new AffineRange[shape.Rank];
        for (var axis = 0; axis < shape.Rank; axis++)
        {
            ranges[axis] = axis < prefixRank
                ? new AffineRange(domains[axis].Offset, domains[axis].Extent)
                : new AffineRange(0, GetFixedDimension(shape[axis]));
        }

        return ranges;
    }

    private static bool HasFixedSuffix(Shape shape, int start)
    {
        for (var axis = start; axis < shape.Rank; axis++)
        {
            if (shape[axis] is not DimConst)
            {
                return false;
            }
        }

        return true;
    }

    private static long GetFixedDimension(Dimension dimension)
        => dimension is DimConst dimConst
            ? dimConst.Value
            : throw new ArgumentException($"Expected fixed dimension, got {dimension}.");

    private static bool IsSameDimension(Dimension lhs, Dimension rhs)
        => lhs.Equals(rhs) || (lhs is DimConst l && rhs is DimConst r && l.Value == r.Value);
}
