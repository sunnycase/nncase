// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class PagedAttentionPartialEvaluator : ITypeInferencer<PagedAttentionPartial>
{
    public IRType Visit(ITypeInferenceContext context, PagedAttentionPartial target)
        => TupleType.Void;
}

public sealed class PagedAttentionMergeEvaluator : ITypeInferencer<PagedAttentionMerge>
{
    public IRType Visit(ITypeInferenceContext context, PagedAttentionMerge target)
        => TupleType.Void;
}
