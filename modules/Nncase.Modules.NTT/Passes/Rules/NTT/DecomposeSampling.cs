// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;

namespace Nncase.Passes.Rules.NTT;

/// <summary>
/// Makes vocabulary-shard sampling and its cross-shard combine explicit before
/// distributed search.
/// </summary>
[RuleGenerator]
public sealed partial class DecomposeSampling : IRewriteRule
{
    public IPattern Pattern { get; } = IsSampling(
        "sampling",
        "samplingCall",
        _ => true,
        IsWildcard("logits"),
        IsWildcard("state"));

    private Expr? GetReplace(
        Sampling sampling,
        Call samplingCall,
        Expr logits,
        Expr state)
    {
        var partial = IR.F.NTT.SamplingPartial(logits, state, sampling.Config);
        return IR.F.NTT.SamplingCombine(
                logits,
                IR.F.Tensors.GetItem(partial, 0),
                IR.F.Tensors.GetItem(partial, 1),
                state,
                sampling.Config)
            .InheritMetaData(samplingCall);
    }
}
