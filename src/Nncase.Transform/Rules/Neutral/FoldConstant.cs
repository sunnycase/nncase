﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Fold call of constants.
/// </summary>
[RuleGenerator]
public partial class FoldConstCall : RewriteRule<CallPattern>
{
    public FoldConstCall()
    {
        IsMultiBranchSafe = true;
    }

    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsCall(
        "call",
        IsOp<Op>(op => op.CanFoldConstCall),
        IsVArgsRepeat("constArgs", () => IsAlt(IsConst(), IsConstTuple()))) with
    {
        TypePattern = IsType(x => !(x is InvalidType)),
    };

    private Const GetReplace(Call call, IReadOnlyList<Expr> constArgs)
    {
        // note for egraphs.
        var new_call = call with { Parameters = new(constArgs) };
        return Const.FromValue(new_call.Evaluate());
    }
}

/// <summary>
/// Fold shape of.
/// </summary>
[RuleGenerator]
public partial class FoldShapeOf : RewriteRule<CallPattern>
{
    public FoldShapeOf()
    {
        IsMultiBranchSafe = true;
    }

    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsShapeOf(IsWildcard("wc") with { TypePattern = HasFixedShape() });

    private Const GetReplace(Expr wc)
    {
        return Const.FromTensor(wc.CheckedShape.ToValueArray().Select(x => (long)x).ToArray());
    }
}
