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
    public Expr SelectPackedMatMulGlu(IR.NTT.PackedMatMulGlu matMulGlu, Call call, Expr output)
    {
        var input = (Expr)call[IR.NTT.PackedMatMulGlu.Input];
        var gateWeight = (Expr)call[IR.NTT.PackedMatMulGlu.GateWeight];
        var upWeight = (Expr)call[IR.NTT.PackedMatMulGlu.UpWeight];
        var gateInputScale = (Expr)call[IR.NTT.PackedMatMulGlu.GateInputScale];
        var upInputScale = (Expr)call[IR.NTT.PackedMatMulGlu.UpInputScale];
        var gateWeightScale = (Expr)call[IR.NTT.PackedMatMulGlu.GateWeightScale];
        var upWeightScale = (Expr)call[IR.NTT.PackedMatMulGlu.UpWeightScale];
        if (gateInputScale is not None ||
            upInputScale is not None ||
            gateWeightScale is not None ||
            upWeightScale is not None)
        {
            return call;
        }

        if (!TryGetPackedMatMulGluMaps(
            input.CheckedShape,
            gateWeight.CheckedShape,
            upWeight.CheckedShape,
            output.CheckedShape,
            out var domains,
            out var inputMap,
            out var weightMap,
            out var outputMap))
        {
            return call;
        }

        var builder = IR.F.Affine.Grid()
            .Domain(domains.Length, out var _)
            .Read(input, inputMap, out var inputTile)
            .Read(gateWeight, weightMap, out var gateWeightTile)
            .Read(upWeight, weightMap, out var upWeightTile);

        var gateBiasArg = ReadOptionalPackedMatMulGluBias(
            builder,
            domains,
            (Expr)call[IR.NTT.PackedMatMulGlu.GateBias],
            output.CheckedShape[^1],
            outputMap.Results[^1],
            "gate",
            out var gateBiasTile);
        var upBiasArg = ReadOptionalPackedMatMulGluBias(
            builder,
            domains,
            (Expr)call[IR.NTT.PackedMatMulGlu.UpBias],
            output.CheckedShape[^1],
            outputMap.Results[^1],
            "up",
            out var upBiasTile);

        return builder
            .Write(output, outputMap, out var outputTile)
            .Body(TIR.F.NTT.PackedMatMulGlu(
                inputTile,
                gateWeightTile,
                upWeightTile,
                gateBiasArg is None ? gateBiasArg : gateBiasTile,
                upBiasArg is None ? upBiasArg : upBiasTile,
                gateInputScale,
                upInputScale,
                gateWeightScale,
                upWeightScale,
                outputTile,
                matMulGlu.GluType))
            .Build();
    }

    private static bool TryGetPackedMatMulGluMaps(
        Shape inputShape,
        Shape gateWeightShape,
        Shape upWeightShape,
        Shape outputShape,
        out AffineDomain[] domains,
        out AffineMap inputMap,
        out AffineMap weightMap,
        out AffineMap outputMap)
    {
        domains = Array.Empty<AffineDomain>();
        inputMap = weightMap = outputMap = null!;

        if (inputShape.Rank < 2 ||
            gateWeightShape.Rank < 2 ||
            gateWeightShape.Rank != upWeightShape.Rank ||
            outputShape.Rank != Math.Max(inputShape.Rank, gateWeightShape.Rank) ||
            !HaveSameShape(gateWeightShape, upWeightShape) ||
            inputShape[^1] is not DimConst kDim ||
            !IsSameDimension(inputShape[^1], gateWeightShape[^1]) ||
            !IsSameDimension(inputShape[^1], upWeightShape[^1]) ||
            !IsSameDimension(inputShape[^2], outputShape[^2]) ||
            !IsSameDimension(gateWeightShape[^2], outputShape[^1]))
        {
            return false;
        }

        domains = IR.F.Affine.Domains(outputShape.Rank);
        var inputResults = new AffineRange[inputShape.Rank];
        var weightResults = new AffineRange[gateWeightShape.Rank];
        if (!TryBuildBatchRanges(inputShape, outputShape, domains, inputResults) ||
            !TryBuildBatchRanges(gateWeightShape, outputShape, domains, weightResults))
        {
            return false;
        }

        inputResults[^2] = DomainRange(domains[^2]);
        inputResults[^1] = new AffineRange(0, kDim.Value);
        weightResults[^2] = DomainRange(domains[^1]);
        weightResults[^1] = new AffineRange(0, kDim.Value);
        inputMap = new AffineMap(domains, default, inputResults);
        weightMap = new AffineMap(domains, default, weightResults);
        outputMap = new AffineMap(domains, default, domains.Select(DomainRange).ToArray());
        return true;
    }

    private static bool TryBuildBatchRanges(Shape operandShape, Shape outputShape, AffineDomain[] domains, AffineRange[] results)
    {
        var rankOffset = outputShape.Rank - operandShape.Rank;
        for (int operandAxis = 0; operandAxis < operandShape.Rank - 2; operandAxis++)
        {
            var outputAxis = operandAxis + rankOffset;
            if (operandShape[operandAxis] is DimConst { Value: 1 })
            {
                results[operandAxis] = new AffineRange(0, 1);
            }
            else if (IsSameDimension(operandShape[operandAxis], outputShape[outputAxis]))
            {
                results[operandAxis] = DomainRange(domains[outputAxis]);
            }
            else
            {
                return false;
            }
        }

        return true;
    }

    private static bool HaveSameShape(Shape lhs, Shape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            return false;
        }

        for (int axis = 0; axis < lhs.Rank; axis++)
        {
            if (!IsSameDimension(lhs[axis], rhs[axis]))
            {
                return false;
            }
        }

        return true;
    }

    private static AffineRange DomainRange(AffineDomain domain) => new(domain.Offset, domain.Extent);

    private static Expr ReadOptionalPackedMatMulGluBias(
        IGridBuilder builder,
        AffineDomain[] domains,
        Expr bias,
        Dimension outputN,
        AffineRange nRange,
        string projectionName,
        out Var tile)
    {
        tile = null!;
        if (bias is None)
        {
            return bias;
        }

        if (bias.CheckedShape.Rank != 1 || !IsSameDimension(bias.CheckedShape[0], outputN))
        {
            throw new NotSupportedException($"PackedMatMulGlu {projectionName} bias expects rank 1 with packed N extent {outputN}, got {bias.CheckedShape}.");
        }

        builder.Read(bias, new AffineMap(domains, default, new[] { nRange }), out tile);
        return tile;
    }
}
