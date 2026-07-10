// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Affine.Builders;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectPackedQKVParallelLinear(IR.NTT.PackedQKVParallelLinear qkv, Call call, BaseExpr output)
    {
        if (output is not IR.Tuple outputs || outputs.Count != 3)
        {
            return call;
        }

        var input = (Expr)call[IR.NTT.PackedQKVParallelLinear.Input];
        var qWeight = (Expr)call[IR.NTT.PackedQKVParallelLinear.QWeight];
        var kWeight = (Expr)call[IR.NTT.PackedQKVParallelLinear.KWeight];
        var vWeight = (Expr)call[IR.NTT.PackedQKVParallelLinear.VWeight];
        var qOutput = (Expr)outputs[0];
        var kOutput = (Expr)outputs[1];
        var vOutput = (Expr)outputs[2];

        if (!TryGetPackedQKVMaps(
            input.CheckedShape,
            qWeight.CheckedShape,
            kWeight.CheckedShape,
            vWeight.CheckedShape,
            qOutput.CheckedShape,
            kOutput.CheckedShape,
            vOutput.CheckedShape,
            out var domains,
            out var inputMap,
            out var qWeightMap,
            out var kWeightMap,
            out var vWeightMap,
            out var qOutputMap,
            out var kOutputMap,
            out var vOutputMap))
        {
            return call;
        }

        var builder = IR.F.Affine.Grid()
            .Domain(domains.Length, out var _)
            .Read(input, inputMap, out var inputTile)
            .Read(qWeight, qWeightMap, out var qWeightTile)
            .Read(kWeight, kWeightMap, out var kWeightTile)
            .Read(vWeight, vWeightMap, out var vWeightTile);

        var qBiasArg = ReadOptionalProjectionParameter(builder, domains, (Expr)call[IR.NTT.PackedQKVParallelLinear.QBias], qOutputMap.Results[1], out var qBiasTile);
        var kBiasArg = ReadOptionalProjectionParameter(builder, domains, (Expr)call[IR.NTT.PackedQKVParallelLinear.KBias], kOutputMap.Results[1], out var kBiasTile);
        var vBiasArg = ReadOptionalProjectionParameter(builder, domains, (Expr)call[IR.NTT.PackedQKVParallelLinear.VBias], vOutputMap.Results[1], out var vBiasTile);

        var qInputScale = (Expr)call[IR.NTT.PackedQKVParallelLinear.QInputScale];
        var kInputScale = (Expr)call[IR.NTT.PackedQKVParallelLinear.KInputScale];
        var vInputScale = (Expr)call[IR.NTT.PackedQKVParallelLinear.VInputScale];
        var qWeightScale = (Expr)call[IR.NTT.PackedQKVParallelLinear.QWeightScale];
        var kWeightScale = (Expr)call[IR.NTT.PackedQKVParallelLinear.KWeightScale];
        var vWeightScale = (Expr)call[IR.NTT.PackedQKVParallelLinear.VWeightScale];
        if (qInputScale is not None || kInputScale is not None || vInputScale is not None ||
            qWeightScale is not None || kWeightScale is not None || vWeightScale is not None)
        {
            return call;
        }

        return builder
            .Write(qOutput, qOutputMap, out var qOutputTile)
            .Write(kOutput, kOutputMap, out var kOutputTile)
            .Write(vOutput, vOutputMap, out var vOutputTile)
            .Body(TIR.F.NTT.PackedQKVParallelLinear(
                inputTile,
                qWeightTile,
                kWeightTile,
                vWeightTile,
                qBiasArg is None ? qBiasArg : qBiasTile,
                kBiasArg is None ? kBiasArg : kBiasTile,
                vBiasArg is None ? vBiasArg : vBiasTile,
                qInputScale,
                kInputScale,
                vInputScale,
                qWeightScale,
                kWeightScale,
                vWeightScale,
                qOutputTile,
                kOutputTile,
                vOutputTile,
                qkv.NumHeads,
                qkv.NumKvHeads))
            .Build();
    }

    private static bool TryGetPackedQKVMaps(
        Shape inputShape,
        Shape qWeightShape,
        Shape kWeightShape,
        Shape vWeightShape,
        Shape qOutputShape,
        Shape kOutputShape,
        Shape vOutputShape,
        out AffineDomain[] domains,
        out AffineMap inputMap,
        out AffineMap qWeightMap,
        out AffineMap kWeightMap,
        out AffineMap vWeightMap,
        out AffineMap qOutputMap,
        out AffineMap kOutputMap,
        out AffineMap vOutputMap)
    {
        domains = Array.Empty<AffineDomain>();
        inputMap = qWeightMap = kWeightMap = vWeightMap = qOutputMap = kOutputMap = vOutputMap = null!;

        if (inputShape.Rank != 2 ||
            qWeightShape.Rank != 2 ||
            kWeightShape.Rank != 2 ||
            vWeightShape.Rank != 2 ||
            qOutputShape.Rank != 2 ||
            kOutputShape.Rank != 2 ||
            vOutputShape.Rank != 2 ||
            !IsSameDimension(inputShape[0], qOutputShape[0]) ||
            !IsSameDimension(inputShape[0], kOutputShape[0]) ||
            !IsSameDimension(inputShape[0], vOutputShape[0]) ||
            !IsSameDimension(inputShape[1], qWeightShape[1]) ||
            !IsSameDimension(inputShape[1], kWeightShape[1]) ||
            !IsSameDimension(inputShape[1], vWeightShape[1]) ||
            !IsSameDimension(qWeightShape[0], qOutputShape[1]) ||
            !IsSameDimension(kWeightShape[0], kOutputShape[1]) ||
            !IsSameDimension(vWeightShape[0], vOutputShape[1]) ||
            inputShape[1] is not DimConst kDim ||
            qWeightShape[0] is not DimConst qNDim ||
            kWeightShape[0] is not DimConst kNDim ||
            vWeightShape[0] is not DimConst vNDim ||
            qNDim.Value <= 0 ||
            kNDim.Value <= 0 ||
            vNDim.Value <= 0)
        {
            return false;
        }

        var commonN = GreatestCommonDivisor(GreatestCommonDivisor(qNDim.Value, kNDim.Value), vNDim.Value);
        var qNScale = qNDim.Value / commonN;
        var kNScale = kNDim.Value / commonN;
        var vNScale = vNDim.Value / commonN;

        domains = IR.F.Affine.Domains(2);
        inputMap = new AffineMap(domains, default, new[]
        {
            new AffineRange(domains[0].Offset, domains[0].Extent),
            new AffineRange(0, kDim.Value),
        });
        qWeightMap = BuildPackedProjectionWeightMap(domains, qNScale, kDim.Value);
        kWeightMap = BuildPackedProjectionWeightMap(domains, kNScale, kDim.Value);
        vWeightMap = BuildPackedProjectionWeightMap(domains, vNScale, kDim.Value);
        qOutputMap = BuildPackedProjectionOutputMap(domains, qNScale);
        kOutputMap = BuildPackedProjectionOutputMap(domains, kNScale);
        vOutputMap = BuildPackedProjectionOutputMap(domains, vNScale);
        return true;
    }

    private static AffineMap BuildPackedProjectionWeightMap(AffineDomain[] domains, long nScale, long k)
        => new(domains, default, new[]
        {
            BuildScaledProjectionRange(domains[1], nScale),
            new AffineRange(0, k),
        });

    private static AffineMap BuildPackedProjectionOutputMap(AffineDomain[] domains, long nScale)
        => new(domains, default, new[]
        {
            new AffineRange(domains[0].Offset, domains[0].Extent),
            BuildScaledProjectionRange(domains[1], nScale),
        });

    private static AffineRange BuildScaledProjectionRange(AffineDomain domain, long scale)
        => scale == 1
            ? new AffineRange(domain.Offset, domain.Extent)
            : new AffineRange(domain.Offset * scale, domain.Extent * scale);

    private static long GreatestCommonDivisor(long lhs, long rhs)
    {
        while (rhs != 0)
        {
            (lhs, rhs) = (rhs, lhs % rhs);
        }

        return lhs;
    }

    private static Expr ReadOptionalProjectionParameter(IGridBuilder builder, AffineDomain[] domains, Expr parameter, AffineRange nRange, out Var tile)
    {
        tile = null!;
        if (parameter is None)
        {
            return parameter;
        }

        if (parameter.CheckedShape.Rank != 1)
        {
            throw new NotSupportedException($"PackedQKVParallelLinear optional parameter expects rank 1 tensor, got rank {parameter.CheckedShape.Rank}.");
        }

        builder.Read(parameter, new AffineMap(domains, default, new[] { nRange }), out tile);
        return tile;
    }
}
