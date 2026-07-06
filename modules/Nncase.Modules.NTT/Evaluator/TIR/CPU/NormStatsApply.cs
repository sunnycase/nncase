// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class NormStatsEvaluator : ITypeInferencer<NormStats>
{
    public IRType Visit(ITypeInferenceContext context, NormStats target) => TupleType.Void;
}

public sealed class NormApplyEvaluator : ITypeInferencer<NormApply>
{
    public IRType Visit(ITypeInferenceContext context, NormApply target) => TupleType.Void;
}
