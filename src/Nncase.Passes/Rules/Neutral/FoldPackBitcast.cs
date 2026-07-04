// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldPackBitcast : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsPack(
            target_name: "pack",
            "call",
            _ => true,
            IsBitcast(
                target_name: "bitcast",
                "bitcastCall",
                _ => true,
                IsWildcard("input") with { TypePattern = HasRankedShape() }));

    public Expr? GetReplace(Expr call, Pack pack, Expr bitcastCall, Bitcast bitcast, Expr input)
    {
        return Equals(call.CheckedType, input.CheckedType) ? input : null;
    }
}

[RuleGenerator]
public sealed partial class FoldPackReshapeBitcast : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsPack(
            target_name: "pack",
            "call",
            _ => true,
            IsReshape(
                target_name: "reshape",
                "reshapeCall",
                IsBitcast(
                    target_name: "bitcast",
                    "bitcastCall",
                    _ => true,
                    IsWildcard("input") with { TypePattern = HasRankedShape() }),
                IsWildcard("shape")));

    public Expr? GetReplace(Expr call, Pack pack, Reshape reshape, Bitcast bitcast, Expr input)
    {
        var rank = call.CheckedShape.Rank;
        var trailingAxes = Enumerable.Range(rank - pack.Axes.Count, pack.Axes.Count);
        if (!pack.Axes.SequenceEqual(trailingAxes) || !Equals(call.CheckedDataType, input.CheckedDataType))
        {
            return null;
        }

        return IR.F.Tensors.Reshape(input, call.CheckedShape);
    }
}
