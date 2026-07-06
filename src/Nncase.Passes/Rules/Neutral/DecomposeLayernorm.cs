// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Decompose layernorm.
/// </summary>
[RuleGenerator]
public sealed partial class DecomposeLayerNorm : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
    IsLayerNorm(
      "ln",
      "call",
      _ => true,
      IsWildcard("input") with { TypePattern = IsFloat() },
      IsWildcard("scale") with { TypePattern = IsFloat() },
      IsWildcard("bias") with { TypePattern = IsFloat() });

    private Expr? GetReplace(Expr input, Call call, LayerNorm ln, Expr scale, Expr bias)
    {
        var rank = input.CheckedShape.Rank;
        if (rank <= 0)
        {
            return null;
        }

        var normalizedAxis = ln.Axis < 0 ? ln.Axis + rank : ln.Axis;
        if (normalizedAxis < 0 || normalizedAxis >= rank)
        {
            return null;
        }

        var stats = IR.F.NN.NormStats(normalizedAxis, input, ln.UseMean);
        return IR.F.NN.NormApply(normalizedAxis, ln.Epsilon, input, stats, scale, bias, ln.UseMean).InheritMetaData(call);
    }
}
