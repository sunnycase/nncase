// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public class GatherReduceScatterEvaluator : ITypeInferencer<GatherReduceScatter>, IKernelInfoEvaluator<GatherReduceScatter>
{
    public MicroKernelInfo Visit(GatherReduceScatter target, MicroKernelContext context)
        => TransferKernelInfo.Create(context);

    public IRType Visit(ITypeInferenceContext context, GatherReduceScatter target)
    {
        _ = context.CheckArgumentType<IRType>(target, GatherReduceScatter.Input);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceScatter.Output);
        return TupleType.Void;
    }
}
