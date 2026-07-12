// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class BitcastEvaluator : ITypeInferencer<Bitcast>, ITileWorkloadEvaluator<Bitcast>
{
    public IRType Visit(ITypeInferenceContext context, Bitcast target)
    {
        var inputType = context.CheckArgumentType<TensorType>(target, Bitcast.Input);
        var outputType = context.CheckArgumentType<TensorType>(target, Bitcast.Output);
        return GetScalarDataType(inputType.DType) == GetScalarDataType(outputType.DType)
            ? TupleType.Void
            : new InvalidType($"TIR Bitcast requires one scalar element type, got input={inputType.DType}, output={outputType.DType}.");
    }

    public TileWorkload Visit(Bitcast op, TileWorkloadContext context) => BufferAliasTileWorkload.Default;

    private static DataType GetScalarDataType(DataType dataType) => dataType switch
    {
        VectorType vectorType => vectorType.ElemType,
        MaskVectorType => DataTypes.Boolean,
        _ => dataType,
    };
}
