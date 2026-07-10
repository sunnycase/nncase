// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectReshape(Call call, Expr output)
    {
        var input = (Expr)call[IR.Tensors.Reshape.Input];
        if (input.CheckedDataType != output.CheckedDataType)
        {
            throw new InvalidOperationException($"Affine Reshape must preserve dtype and lanes, got input={input.CheckedDataType}, output={output.CheckedDataType}.");
        }

        if (!TryGetViewLikeMaps(input, output, 1, 1, out var domains, out var inputMap, out var outputMap))
        {
            return call;
        }

        return IR.F.Affine.Grid()
            .Domain(domains.Length, out var _)
            .Read(input, inputMap, out var inputTile)
            .Write(output, outputMap, out var outputTile)
            .Body(TIR.F.NTT.Reshape(inputTile, outputTile))
            .Build();
    }

    public Expr SelectBitcast(Call call, Expr output)
    {
        var input = (Expr)call[IR.Tensors.Bitcast.Input];
        if (GetScalarDataType(input.CheckedDataType) != GetScalarDataType(output.CheckedDataType))
        {
            return call;
        }

        var inputLanes = GetVectorLaneElementCount(input.CheckedDataType);
        var outputLanes = GetVectorLaneElementCount(output.CheckedDataType);
        if (!TryGetViewLikeMaps(input, output, inputLanes, outputLanes, out var domains, out var inputMap, out var outputMap))
        {
            return call;
        }

        return IR.F.Affine.Grid()
            .Domain(domains.Length, out var _)
            .Read(input, inputMap, out var inputTile)
            .Write(output, outputMap, out var outputTile)
            .Body(TIR.F.NTT.Bitcast(inputTile, outputTile))
            .Build();
    }

    private static bool TryGetViewLikeMaps(Expr input, Expr output, int inputLanes, int outputLanes, out AffineDomain[] domains, out AffineMap inputMap, out AffineMap outputMap)
    {
        domains = Array.Empty<AffineDomain>();
        inputMap = null!;
        outputMap = null!;

        var inputShape = input.CheckedShape;
        var outputShape = output.CheckedShape;
        if (inputShape is not { Rank: > 0 } || outputShape is not { Rank: > 0 })
        {
            return false;
        }

        var prefixRank = GetCommonPrefixRank(inputShape, outputShape);
        return TryGetFlatToFlatMaps(inputShape, outputShape, prefixRank, inputLanes, outputLanes, out domains, out inputMap, out outputMap) ||
            TryGetFlatInputToOutputMajorMaps(inputShape, outputShape, prefixRank, inputLanes, outputLanes, out domains, out inputMap, out outputMap) ||
            TryGetInputMajorToFlatOutputMaps(inputShape, outputShape, prefixRank, inputLanes, outputLanes, out domains, out inputMap, out outputMap) ||
            (!HasSuffixSplit(input.CheckedType, prefixRank) &&
                !HasSuffixSplit(output.CheckedType, prefixRank) &&
                TryGetPrefixFullTileMaps(inputShape, outputShape, prefixRank, inputLanes, outputLanes, out domains, out inputMap, out outputMap));
    }

    private static bool TryGetFlatToFlatMaps(Shape inputShape, Shape outputShape, int prefixRank, int inputLane, int outputLane, out AffineDomain[] domains, out AffineMap inputMap, out AffineMap outputMap)
    {
        domains = Array.Empty<AffineDomain>();
        inputMap = null!;
        outputMap = null!;

        if (inputShape.Rank != prefixRank + 1 ||
            outputShape.Rank != prefixRank + 1 ||
            !HasFixedSuffix(inputShape, prefixRank) ||
            !HasFixedSuffix(outputShape, prefixRank) ||
            GetScalarSuffixElementCount(inputShape, prefixRank, inputLane) != GetScalarSuffixElementCount(outputShape, prefixRank, outputLane))
        {
            return false;
        }

        domains = IR.F.Affine.Domains(prefixRank + 1);
        if (outputLane % inputLane == 0)
        {
            var inputScale = outputLane / inputLane;
            inputMap = new AffineMap(domains, default, BuildFlatSuffixRanges(domains, inputShape, prefixRank, inputScale, suffixDomainIsIdentity: false));
            outputMap = new AffineMap(domains, default, BuildFlatSuffixRanges(domains, outputShape, prefixRank, 1, suffixDomainIsIdentity: true));
            return true;
        }

        if (inputLane % outputLane == 0)
        {
            var outputScale = inputLane / outputLane;
            inputMap = new AffineMap(domains, default, BuildFlatSuffixRanges(domains, inputShape, prefixRank, 1, suffixDomainIsIdentity: true));
            outputMap = new AffineMap(domains, default, BuildFlatSuffixRanges(domains, outputShape, prefixRank, outputScale, suffixDomainIsIdentity: false));
            return true;
        }

        return false;
    }

    private static bool TryGetFlatInputToOutputMajorMaps(Shape inputShape, Shape outputShape, int prefixRank, int inputLane, int outputLane, out AffineDomain[] domains, out AffineMap inputMap, out AffineMap outputMap)
    {
        domains = Array.Empty<AffineDomain>();
        inputMap = null!;
        outputMap = null!;

        if (inputShape.Rank != prefixRank + 1 ||
            outputShape.Rank <= prefixRank + 1 ||
            !HasFixedSuffix(inputShape, prefixRank) ||
            !HasFixedSuffix(outputShape, prefixRank) ||
            GetScalarSuffixElementCount(inputShape, prefixRank, inputLane) != GetScalarSuffixElementCount(outputShape, prefixRank, outputLane))
        {
            return false;
        }

        var scalarElementsPerOutputMajor = ProductFixedSuffix(outputShape, prefixRank + 1) * outputLane;
        if (scalarElementsPerOutputMajor % inputLane != 0)
        {
            return false;
        }

        domains = IR.F.Affine.Domains(prefixRank + 1);
        inputMap = new AffineMap(domains, default, BuildFlatSuffixRanges(domains, inputShape, prefixRank, scalarElementsPerOutputMajor / inputLane, suffixDomainIsIdentity: false));
        outputMap = new AffineMap(domains, default, BuildMajorSuffixRanges(domains, outputShape, prefixRank));
        return true;
    }

    private static bool TryGetInputMajorToFlatOutputMaps(Shape inputShape, Shape outputShape, int prefixRank, int inputLane, int outputLane, out AffineDomain[] domains, out AffineMap inputMap, out AffineMap outputMap)
    {
        domains = Array.Empty<AffineDomain>();
        inputMap = null!;
        outputMap = null!;

        if (inputShape.Rank <= prefixRank + 1 ||
            outputShape.Rank != prefixRank + 1 ||
            !HasFixedSuffix(inputShape, prefixRank) ||
            !HasFixedSuffix(outputShape, prefixRank) ||
            GetScalarSuffixElementCount(inputShape, prefixRank, inputLane) != GetScalarSuffixElementCount(outputShape, prefixRank, outputLane))
        {
            return false;
        }

        var scalarElementsPerInputMajor = ProductFixedSuffix(inputShape, prefixRank + 1) * inputLane;
        if (scalarElementsPerInputMajor % outputLane != 0)
        {
            return false;
        }

        domains = IR.F.Affine.Domains(prefixRank + 1);
        inputMap = new AffineMap(domains, default, BuildMajorSuffixRanges(domains, inputShape, prefixRank));
        outputMap = new AffineMap(domains, default, BuildFlatSuffixRanges(domains, outputShape, prefixRank, scalarElementsPerInputMajor / outputLane, suffixDomainIsIdentity: false));
        return true;
    }

    private static bool TryGetPrefixFullTileMaps(Shape inputShape, Shape outputShape, int prefixRank, int inputLane, int outputLane, out AffineDomain[] domains, out AffineMap inputMap, out AffineMap outputMap)
    {
        domains = Array.Empty<AffineDomain>();
        inputMap = null!;
        outputMap = null!;

        if (!HasFixedSuffix(inputShape, prefixRank) || !HasFixedSuffix(outputShape, prefixRank))
        {
            return false;
        }

        if (GetScalarSuffixElementCount(inputShape, prefixRank, inputLane) != GetScalarSuffixElementCount(outputShape, prefixRank, outputLane))
        {
            return false;
        }

        domains = IR.F.Affine.Domains(prefixRank);
        inputMap = new AffineMap(domains, default, BuildPrefixFullTileRanges(domains, inputShape, prefixRank));
        outputMap = new AffineMap(domains, default, BuildPrefixFullTileRanges(domains, outputShape, prefixRank));
        return true;
    }

    private static AffineRange[] BuildPrefixFullTileRanges(AffineDomain[] domains, Shape shape, int prefixRank)
    {
        var ranges = new AffineRange[shape.Rank];
        for (int axis = 0; axis < shape.Rank; axis++)
        {
            ranges[axis] = axis < prefixRank
                ? new AffineRange(domains[axis].Offset, domains[axis].Extent)
                : new AffineRange(0, GetFixedDimension(shape[axis]));
        }

        return ranges;
    }

    private static AffineRange[] BuildFlatSuffixRanges(AffineDomain[] domains, Shape shape, int prefixRank, long scale, bool suffixDomainIsIdentity)
    {
        var ranges = new AffineRange[shape.Rank];
        for (int axis = 0; axis < prefixRank; axis++)
        {
            ranges[axis] = new AffineRange(domains[axis].Offset, domains[axis].Extent);
        }

        var suffixDomain = domains[prefixRank];
        ranges[prefixRank] = suffixDomainIsIdentity
            ? new AffineRange(suffixDomain.Offset, suffixDomain.Extent)
            : ScaledRange(suffixDomain, scale);
        return ranges;
    }

    private static AffineRange[] BuildMajorSuffixRanges(AffineDomain[] domains, Shape shape, int prefixRank)
    {
        var ranges = new AffineRange[shape.Rank];
        for (int axis = 0; axis < shape.Rank; axis++)
        {
            ranges[axis] = axis switch
            {
                _ when axis < prefixRank => new AffineRange(domains[axis].Offset, domains[axis].Extent),
                _ when axis == prefixRank => new AffineRange(domains[prefixRank].Offset, domains[prefixRank].Extent),
                _ => new AffineRange(0, GetFixedDimension(shape[axis])),
            };
        }

        return ranges;
    }

    private static AffineRange ScaledRange(AffineDomain domain, long scale)
    {
        return scale == 1
            ? new AffineRange(domain.Offset, domain.Extent)
            : new AffineRange(domain.Offset * scale, domain.Extent * scale);
    }

    private static bool HasFixedSuffix(Shape shape, int start)
    {
        for (int axis = start; axis < shape.Rank; axis++)
        {
            if (shape[axis] is not DimConst)
            {
                return false;
            }
        }

        return true;
    }

    private static int GetCommonPrefixRank(Shape inputShape, Shape outputShape)
    {
        var prefixRank = 0;
        var minRank = Math.Min(inputShape.Rank, outputShape.Rank);
        while (prefixRank < minRank && IsSameDimension(inputShape[prefixRank], outputShape[prefixRank]))
        {
            prefixRank++;
        }

        return prefixRank;
    }

    private static bool HasSuffixSplit(IRType type, int prefixRank)
    {
        if (type is not DistributedType distributedType)
        {
            return false;
        }

        return distributedType.AxisPolicies.Any(policy =>
            policy is SBPSplit split && split.Axes.Any(axis => axis >= prefixRank));
    }

    private static DataType GetScalarDataType(DataType dataType)
        => dataType switch
        {
            VectorType vectorType => vectorType.ElemType,
            MaskVectorType => DataTypes.Boolean,
            _ => dataType,
        };

    private static int GetVectorLaneElementCount(DataType dataType)
        => dataType switch
        {
            VectorType vectorType => vectorType.Lanes.ToArray().Aggregate(1, (acc, lane) => acc * lane),
            MaskVectorType maskVectorType => maskVectorType.Lanes,
            _ => 1,
        };

    private static long GetScalarSuffixElementCount(Shape shape, int start, int lanes)
        => ProductFixedSuffix(shape, start) * lanes;

    private static long ProductFixedSuffix(Shape shape, int start)
    {
        var product = 1L;
        for (int axis = start; axis < shape.Rank; axis++)
        {
            product *= GetFixedDimension(shape[axis]);
        }

        return product;
    }

    private static long GetFixedDimension(Dimension dimension)
    {
        if (dimension is not DimConst dimConst)
        {
            throw new ArgumentException($"Expected fixed dimension, got {dimension}.");
        }

        return dimConst.Value;
    }

    private static bool IsSameDimension(Dimension lhs, Dimension rhs)
        => lhs.Equals(rhs) || (lhs is DimConst l && rhs is DimConst r && l.Value == r.Value);
}
