// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class SamplingPartialEvaluator : ITypeInferencer<SamplingPartial>
{
    public IRType Visit(ITypeInferenceContext context, SamplingPartial target)
        => TupleType.Void;
}

public sealed class SamplingCombineEvaluator : ITypeInferencer<SamplingCombine>
{
    public IRType Visit(ITypeInferenceContext context, SamplingCombine target)
        => TupleType.Void;
}

public sealed class PackedMatMulSamplingPartialEvaluator : ITypeInferencer<PackedMatMulSamplingPartial>
{
    public IRType Visit(ITypeInferenceContext context, PackedMatMulSamplingPartial target)
        => TupleType.Void;
}
