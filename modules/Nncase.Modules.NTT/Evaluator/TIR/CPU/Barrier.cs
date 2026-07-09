// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.Evaluator.TIR.NTT;

public class BarrierEvaluator : ITypeInferencer<Nncase.TIR.NTT.Barrier>
{
    public IRType Visit(ITypeInferenceContext context, Nncase.TIR.NTT.Barrier target)
    {
        return TupleType.Void;
    }
}
