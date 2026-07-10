// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Evaluator.Affine;

public sealed class AffineViewEvaluator : ITypeInferencer<AffineView>, ICostEvaluator<AffineView>
{
    public IRType Visit(ITypeInferenceContext context, AffineView target)
    {
        var sourceType = context.GetArgumentType(target, AffineView.Input);
        return AffineViewUtility.Verify(sourceType, target.NewType, target.Transform) is { } error
            ? new InvalidType(error)
            : target.NewType;
    }

    public Cost Visit(ICostEvaluateContext context, AffineView target) => Cost.Zero;
}
