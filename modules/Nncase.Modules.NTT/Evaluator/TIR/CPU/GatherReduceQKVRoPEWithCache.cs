// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class GatherReduceQKVRoPEWithCacheEvaluator : ITypeInferencer<GatherReduceQKVRoPEWithCache>
{
    public IRType Visit(ITypeInferenceContext context, GatherReduceQKVRoPEWithCache target)
    {
        _ = context.CheckArgumentType<IRType>(target, GatherReduceQKVRoPEWithCache.Q);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceQKVRoPEWithCache.K);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceQKVRoPEWithCache.V);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceQKVRoPEWithCache.QOutput);
        return TupleType.Void;
    }
}
