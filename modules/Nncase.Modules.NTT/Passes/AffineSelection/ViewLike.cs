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

        return CreateAffineView(call, input, output);
    }

    public Expr SelectBitcast(Call call, Expr output)
    {
        var input = (Expr)call[IR.Tensors.Bitcast.Input];
        return CreateAffineView(call, input, output);
    }

    private static Expr CreateAffineView(Call originalCall, Expr input, Expr output)
    {
        if (!AffineViewUtility.TryCreate(input.CheckedType, output.CheckedType, out var transform))
        {
            throw new NotSupportedException(
                $"{originalCall.Target.GetType().Name} from {input.CheckedType} to {output.CheckedType} cannot be represented as a zero-copy affine view.");
        }

        return IR.F.Affine.View(input, output.CheckedType, transform);
    }
}
