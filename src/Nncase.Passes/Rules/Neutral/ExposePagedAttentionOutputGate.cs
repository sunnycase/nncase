// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Exposes the optional paged-attention output gate as independently placeable
/// dataflow before heterogeneous pipeline formation.
/// </summary>
[RuleGenerator]
public sealed partial class ExposePagedAttentionOutputGate : IRewriteRule
{
    public IPattern Pattern { get; } = IsPagedAttention(
        "pagedAttention",
        "call",
        _ => true,
        IsWildcard("q"),
        IsWildcard("kvCaches"),
        IsWildcard("extra"),
        IsWildcard("scale"),
        IsWildcard("layerId"),
        IsWildcard("outputGate"));

    private Expr? GetReplace(
        PagedAttention pagedAttention,
        Call call,
        Expr q,
        Expr kvCaches,
        Expr extra,
        Expr scale,
        Dimension layerId,
        Expr outputGate)
    {
        if (outputGate is None)
        {
            return null;
        }

        var attention = IR.F.NN.PagedAttention(
            q,
            kvCaches,
            extra,
            scale,
            layerId,
            None.Default,
            pagedAttention.Layout.ToArray(),
            pagedAttention.HiddenSize);
        attention.Metadata.SemanticRegion = call.Metadata.SemanticRegion;

        return IR.F.Math.Mul(attention, IR.F.NN.Sigmoid(outputGate))
            .InheritMetaData(call);
    }
}
