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
        => function switch
        {
            PrimFunction => GetOutputParameters(function).Select(output => output.CheckedType).ToArray(),
            Function => FlattenTensorTypes(((CallableType)function.CheckedType).ReturnType).ToArray(),
            _ => throw new NotSupportedException(
                $"PyNTT requires Function or PrimFunction output ABI, got {function.GetType().Name} {function.Name}."),
        };

    public static PrimFunctionResultBinding[] GetResults(BaseFunction function)
    {
        if (function is not PrimFunction primFunction)
        {
            throw new NotSupportedException($"PyNTT requires PrimFunction result ABI, got {function.GetType().Name} {function.Name}.");
        }

        return primFunction.GetAbiView().Results.ToArray();
    }

    private static IEnumerable<IRType> FlattenTensorTypes(IRType type)
    {
        if (type is TupleType tuple)
        {
            foreach (var field in tuple.Fields)
            {
                foreach (var flattened in FlattenTensorTypes(field))
                {
                    yield return flattened;
                }
            }

            yield break;
        }

        if (type is TensorType or DistributedType)
        {
            yield return type;
            yield break;
        }

        throw new NotSupportedException($"PyNTT function result must be tensor-like, got {type}.");
    }
}
