// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldBitcastBitcast : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsBitcast(
            target_name: "outer",
            "outerCall",
            _ => true,
            IsBitcast(
                target_name: "inner",
                "innerCall",
                _ => true,
                IsWildcard("input") with { TypePattern = HasRankedShape() }));

    public Expr? GetReplace(Expr outerCall, Bitcast outer, Expr input)
    {
        var direct = IR.F.Tensors.Bitcast(input, outer.NewType);
        return CompilerServices.InferenceType(direct) && Equals(direct.CheckedType, outerCall.CheckedType)
            ? direct
            : null;
    }
}
