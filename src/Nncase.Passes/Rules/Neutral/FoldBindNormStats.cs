// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Removes the distribution-only normalization statistics binding.
/// </summary>
[RuleGenerator]
public sealed partial class FoldBindNormStats : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = IsBindNormStats(
        "bindNormStats",
        _ => true,
        IsWildcard("input"),
        IsWildcard("stats"));

    private Expr GetReplace(Expr stats) => stats;
}
