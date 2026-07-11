// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public class TensorLoadEvaluator : ITypeInferencer<TensorLoad>, ITileWorkloadEvaluator<TensorLoad>
{
    public TileWorkload Visit(TensorLoad target, TileWorkloadContext context) => TransferTileWorkload.Create(context);

    public IRType Visit(ITypeInferenceContext context, TensorLoad target)
    {
        _ = CheckTensorOrDistributedBuffer(context, target, TensorLoad.Dest);
        _ = context.CheckArgumentType<IRType>(target, TensorLoad.Src);
        return TupleType.Void;
    }

    private static IRType CheckTensorOrDistributedBuffer(ITypeInferenceContext context, TensorLoad target, ParameterInfo parameter)
        => context.GetArgumentType(target, parameter) switch
        {
            TensorType type => type,
            DistributedType type => type,
            AnyType type => throw new TypeInferenceInterruptException(type),
            InvalidType type => throw new TypeInferenceInterruptException(type),
            var type => throw new TypeInferenceInterruptException(new InvalidType($"{target.GetType().Name}.{parameter.Name} Must Be TensorType or DistributedType But Give {type.GetType().Name}.")),
        };
}
