// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.CodeGen.PyNTT;

internal static class PyNTTFunctionOutputs
{
    public static BufferVar[] GetOutputs(BaseFunction function)
    {
        if (function is not PrimFunction primFunction)
        {
            throw new NotSupportedException($"PyNTT requires PrimFunction output ABI, got {function.GetType().Name} {function.Name}.");
        }

        var outputs = primFunction.GetAbiView().Outputs.ToArray();
        if (outputs.Length == 0)
        {
            throw new NotSupportedException($"PyNTT PrimFunction {primFunction.Name} does not declare caller-allocated output BufferVar parameters.");
        }

        return outputs;
    }

    public static IRType[] GetOutputTypes(BaseFunction function)
        => GetOutputs(function).Select(output => output.CheckedType).ToArray();
}
