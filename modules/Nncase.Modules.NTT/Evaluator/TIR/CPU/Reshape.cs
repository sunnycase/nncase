// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class ReshapeEvaluator : ITypeInferencer<Reshape>, ITileWorkloadEvaluator<Reshape>
{
    public IRType Visit(ITypeInferenceContext context, Reshape target)
    {
        var inputType = context.CheckArgumentType<TensorType>(target, Reshape.Input);
        var outputType = context.CheckArgumentType<TensorType>(target, Reshape.Output);
        return inputType.DType == outputType.DType
            ? TupleType.Void
            : new InvalidType($"TIR Reshape must preserve dtype and lanes, got input={inputType.DType}, output={outputType.DType}.");
    }

    public TileWorkload Visit(Reshape op, TileWorkloadContext context) => BufferAliasTileWorkload.Default;
}
