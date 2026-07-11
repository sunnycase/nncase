// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectRoPE(IR.NTT.VectorizedRoPE rope, Call call, Expr output)
    {
        var input = (Expr)call[IR.NTT.VectorizedRoPE.Input];
        var cos = (Expr)call[IR.NTT.VectorizedRoPE.Cos];
        var sin = (Expr)call[IR.NTT.VectorizedRoPE.Sin];

        var rank = input.CheckedShape.Rank;
        var domains = IR.F.Affine.Domains(rank);
        var inOutResults = domains.Select(x => new AffineRange(x.Offset, x.Extent)).ToArray();
        var inOutMap = new AffineMap(domains, default, inOutResults);

        // [seq, head, dim], with sin/cos broadcast over head.
        var sinCosResults = inOutResults.ToArray();
        sinCosResults[1] = new AffineRange(0, 1);
        var sinCosPackFactor = GetRoPESinCosVectorPackFactor(input, cos, sin);
        if (sinCosPackFactor > 1)
        {
            var rotaryAxis = rank - 1;
            var range = sinCosResults[rotaryAxis];
            var offset = new AffineDivBinary(AffineDivBinaryOp.FloorDiv, range.Offset, (long)sinCosPackFactor);
            var end = new AffineDivBinary(AffineDivBinaryOp.CeilDiv, range.Offset + range.Extent, (long)sinCosPackFactor);
            sinCosResults[rotaryAxis] = new AffineRange(offset, end - offset);
        }

        var sinCosMap = new AffineMap(domains, default, sinCosResults);
        var tileAxisPolicies = Enumerable.Repeat(GridTileAxisPolicy.Search(), rank).ToArray();
        tileAxisPolicies[^2] = GridTileAxisPolicy.FullExtent;

        return IR.F.Affine.Grid()
            .Domain(tileAxisPolicies, out var _)
            .Read(input, inOutMap, out var inTile)
            .Read(sin, sinCosMap, out var sinTile)
            .Read(cos, sinCosMap, out var cosTile)
            .Write(output, inOutMap, out var outTile)
            .Body(TIR.F.NTT.RoPE(inTile, cosTile, sinTile, outTile))
            .Build();
    }

    private static int GetRoPESinCosVectorPackFactor(Expr input, Expr cos, Expr sin)
    {
        var inputLanes = GetVectorLanes(input.CheckedDataType);
        var cosLanes = GetVectorLanes(cos.CheckedDataType);
        var sinLanes = GetVectorLanes(sin.CheckedDataType);
        if (!cosLanes.SequenceEqual(sinLanes))
        {
            throw new NotSupportedException($"VectorizedRoPE requires matching sin/cos vector lanes, got cos=[{string.Join(",", cosLanes)}], sin=[{string.Join(",", sinLanes)}].");
        }

        if (inputLanes.Length == 1 && cosLanes.Length == 2 && cosLanes[0] == 2 && cosLanes[1] == inputLanes[0])
        {
            return 2;
        }

        return 1;
    }

    private static int[] GetVectorLanes(DataType dataType)
    {
        return dataType switch
        {
            VectorType vectorType => vectorType.Lanes.ToArray(),
            MaskVectorType maskVectorType => [maskVectorType.Lanes],
            _ => [],
        };
    }
}
