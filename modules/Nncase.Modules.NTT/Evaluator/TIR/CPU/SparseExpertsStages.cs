// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class SparseExpertsGateUpEvaluator : ITypeInferencer<SparseExpertsGateUp>
{
    public IRType Visit(ITypeInferenceContext context, SparseExpertsGateUp target) => TupleType.Void;
}

public sealed class SparseExpertsDownEvaluator : ITypeInferencer<SparseExpertsDown>
{
    public IRType Visit(ITypeInferenceContext context, SparseExpertsDown target) => TupleType.Void;
}
