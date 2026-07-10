// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.CodeGen.PyNTT;

internal static class PyNTTFunctionOutputs
{
    public static BufferVar[] GetOutputParameters(BaseFunction function)
    {
        if (function is not PrimFunction primFunction)
        {
            throw new NotSupportedException($"PyNTT requires PrimFunction output ABI, got {function.GetType().Name} {function.Name}.");
        }

        return primFunction.GetAbiView().OutputParameters.ToArray();
    }

    public static IRType[] GetOutputParameterTypes(BaseFunction function)
        => GetOutputParameters(function).Select(output => output.CheckedType).ToArray();

    public static PrimFunctionResultBinding[] GetResults(BaseFunction function)
    {
        if (function is not PrimFunction primFunction)
        {
            throw new NotSupportedException($"PyNTT requires PrimFunction result ABI, got {function.GetType().Name} {function.Name}.");
        }

        return primFunction.GetAbiView().Results.ToArray();
    }
}
