// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class FoldGetItemTuple : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsGetItem(null, "getItem", IsTuple("input"), IsWildcard("index"));

    private static int? TryGetScalarIndex(BaseExpr index)
    {
        return index switch
        {
            DimConst dim when dim.Value >= int.MinValue && dim.Value <= int.MaxValue => (int)dim.Value,
            TensorConst tensor when tensor.Value.Shape.IsScalar => tensor.Value.ToScalar<int>(),
            _ => null,
        };
    }

    private BaseExpr? GetReplace(IR.Tuple input, BaseExpr index)
    {
        var fixedIndex = TryGetScalarIndex(index);
        return fixedIndex is int value && value >= 0 && value < input.Fields.Length ? input.Fields[value] : null;
    }
}
