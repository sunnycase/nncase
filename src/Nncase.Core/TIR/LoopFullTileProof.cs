// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// Proves whether every logical iteration of a tiled loop owns one complete
/// <see cref="Range.Step"/>-sized tile.
/// </summary>
public static class LoopFullTileProof
{
    /// <summary>
    /// Returns true only when the loop contract or its static range proves
    /// that no iteration is a clipped boundary tile.
    /// </summary>
    public static bool CanProveEveryIterationIsFull(Range domain, LoopPartition partition)
    {
        if (!domain.Step.IsFixed || domain.Step.FixedValue <= 0 || partition == LoopPartition.Tail)
        {
            return false;
        }

        if (partition == LoopPartition.Full || domain.Step.FixedValue == 1)
        {
            return true;
        }

        if (!domain.Start.IsFixed || !domain.Stop.IsFixed)
        {
            return false;
        }

        var extent = new BigInteger(domain.Stop.FixedValue) - domain.Start.FixedValue;
        return extent >= 0 && (extent % domain.Step.FixedValue) == 0;
    }

    /// <summary>
    /// Recovers the logical, unaligned stop from a full-loop domain produced
    /// by tail peeling. The returned bound proves
    /// <c>loop_var + step &lt;= logical_stop</c> for every full iteration.
    /// </summary>
    public static bool TryGetLogicalStop(
        Range domain,
        LoopPartition partition,
        out Dimension logicalStop)
    {
        logicalStop = domain.Stop;
        if (partition != LoopPartition.Full ||
            !domain.Step.IsFixed ||
            domain.Step.FixedValue <= 1)
        {
            return false;
        }

        var step = domain.Step.FixedValue;
        var alignedSpan = domain.Stop - domain.Start;
        if (alignedSpan is not DimProduct product ||
            product.Scale != step ||
            product.Count != 1 ||
            product[0] is not DimFraction fraction ||
            fraction.DivMode != DimDivideMode.FloorDiv ||
            fraction.Denominator is not DimConst denominator ||
            denominator.Value != step ||
            fraction.Numerator.Metadata.Range is not { Min: >= 0 })
        {
            return false;
        }

        logicalStop = domain.Start + fraction.Numerator;
        return true;
    }
}
