// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;

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

        if (!BufferViewUtility.TryCreate(input.CheckedType, output.CheckedType, out var transform))
        {
            return call;
        }

        return IR.F.Affine.Grid()
            .Domain(transform.DomainBounds.Count, out var _)
            .Read(input, transform.SourceMap, out var inputTile)
            .Write(output, transform.ResultMap, out var outputTile)
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

        if (!BufferViewUtility.TryCreate(input.CheckedType, output.CheckedType, out var transform))
        {
            return call;
        }

        return IR.F.Affine.Grid()
            .Domain(transform.DomainBounds.Count, out var _)
            .Read(input, transform.SourceMap, out var inputTile)
            .Write(output, transform.ResultMap, out var outputTile)
            .Body(TIR.F.NTT.Bitcast(inputTile, outputTile))
            .Build();
    }

    private static DataType GetScalarDataType(DataType dataType) => dataType switch
    {
        VectorType vectorType => vectorType.ElemType,
        MaskVectorType => DataTypes.Boolean,
        _ => dataType,
    };
}
