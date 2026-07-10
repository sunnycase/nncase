// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectNormStats(IR.NN.NormStats normStats, Call call, Expr output)
    {
        var input = (Expr)call[IR.NN.NormStats.Input];
        var inputShape = input.CheckedShape;
        var outputShape = output.CheckedShape;
        if (inputShape is not { Rank: > 0 } ||
            outputShape.Rank != inputShape.Rank + 1)
        {
            return call;
        }

        var normalizedAxis = NormalizeAxis(normStats.Axis, inputShape.Rank);
        if (!HasFixedSuffix(inputShape, normalizedAxis) ||
            outputShape[0] is not DimConst ||
            !HasFixedReducedSuffix(outputShape, normalizedAxis + 1))
        {
            return call;
        }

        var domains = IR.F.Affine.Domains(normalizedAxis);
        var inputMap = new AffineMap(domains, default, BuildPrefixFullTileRanges(domains, inputShape, normalizedAxis));
        var outputMap = new AffineMap(domains, default, BuildStatsRanges(domains, outputShape, normalizedAxis));
        return IR.F.Affine.Grid()
            .Domain(domains.Length, out var _)
            .Read(input, inputMap, out var inputTile)
            .Write(output, outputMap, out var outputTile)
            .Body(TIR.F.NTT.NormStats(inputTile, outputTile, normStats.Axis, normStats.UseMean))
            .Build();
    }

    public Expr SelectNormApply(IR.NN.NormApply normApply, Call call, Expr output)
    {
        var input = (Expr)call[IR.NN.NormApply.Input];
        var stats = (Expr)call[IR.NN.NormApply.Stats];
        var scale = (Expr)call[IR.NN.NormApply.Scale];
        var bias = (Expr)call[IR.NN.NormApply.Bias];
        var outputShape = output.CheckedShape;
        if (outputShape is not { Rank: > 0 })
        {
            return call;
        }

        var normalizedAxis = NormalizeAxis(normApply.Axis, outputShape.Rank);
        if (input.CheckedShape.Rank != outputShape.Rank ||
            stats.CheckedShape.Rank != outputShape.Rank + 1 ||
            !HasFixedReducedSuffix(stats.CheckedShape, normalizedAxis + 1))
        {
            return call;
        }

        var domains = IR.F.Affine.Domains(outputShape.Rank);
        var identityMap = AffineMap.Identity(outputShape.Rank);
        var statsMap = new AffineMap(domains, default, BuildStatsRanges(domains, stats.CheckedShape, normalizedAxis));
        if (!TryBuildParameterMap(domains, outputShape, scale.CheckedShape, normalizedAxis, out var scaleMap) ||
            !TryBuildParameterMap(domains, outputShape, bias.CheckedShape, normalizedAxis, out var biasMap))
        {
            return call;
        }

        return IR.F.Affine.Grid()
            .Domain(domains.Length, out var _)
            .Read(input, identityMap, out var inputTile)
            .Read(stats, statsMap, out var statsTile)
            .Read(scale, scaleMap, out var scaleTile)
            .Read(bias, biasMap, out var biasTile)
            .Write(output, identityMap, out var outputTile)
            .Body(TIR.F.NTT.NormApply(inputTile, statsTile, scaleTile, biasTile, outputTile, normApply.Axis, normApply.Epsilon, normApply.UseMean))
            .Build();
    }

    private static AffineRange[] BuildStatsRanges(AffineDomain[] domains, Shape statsShape, int normalizedAxis)
    {
        var ranges = new AffineRange[statsShape.Rank];
        ranges[0] = new AffineRange(0, GetFixedDimension(statsShape[0]));
        for (int axis = 1; axis < statsShape.Rank; axis++)
        {
            var inputAxis = axis - 1;
            ranges[axis] = inputAxis < normalizedAxis
                ? new AffineRange(domains[inputAxis].Offset, domains[inputAxis].Extent)
                : new AffineRange(0, 1);
        }

        return ranges;
    }

    private static bool TryBuildParameterMap(AffineDomain[] domains, Shape outputShape, Shape parameterShape, int normalizedAxis, out AffineMap map)
    {
        map = null!;
        var parameterRank = outputShape.Rank - normalizedAxis;
        if (parameterShape.Rank != parameterRank)
        {
            return false;
        }

        var ranges = new AffineRange[parameterRank];
        for (int axis = 0; axis < parameterRank; axis++)
        {
            var outputAxis = normalizedAxis + axis;
            var parameterDim = parameterShape[axis];
            if (parameterDim is DimConst { Value: 1 })
            {
                ranges[axis] = new AffineRange(0, 1);
            }
            else if (IsSameDimension(parameterDim, outputShape[outputAxis]))
            {
                ranges[axis] = new AffineRange(domains[outputAxis].Offset, domains[outputAxis].Extent);
            }
            else
            {
                return false;
            }
        }

        map = new AffineMap(domains, default, ranges);
        return true;
    }

    private static bool HasFixedReducedSuffix(Shape shape, int start)
    {
        for (int axis = start; axis < shape.Rank; axis++)
        {
            if (shape[axis] is not DimConst { Value: 1 })
            {
                return false;
            }
        }

        return true;
    }

    private static int NormalizeAxis(int axis, int rank)
        => axis < 0 ? axis + rank : axis;
}
